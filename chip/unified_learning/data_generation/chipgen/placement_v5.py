"""V5B1 Placement Algorithm: Macro-first packing + jittered-grid std-cell fill + legalization.

Replaces O(N²) random rejection placement with fast, legal-by-construction algorithm.
"""

from typing import Tuple, List, Optional
import torch
import random
import math
import numpy as np

from .spatial_hash import SpatialHash2D
from .v5_config import V5Config


class V5Placer:
    """
    V5 Placement Algorithm.
    
    Pipeline:
    1. Macro placement (periphery/islands)
    2. Std-cell grid-based fill (density-targeted)
    3. Local legalization
    """
    
    def __init__(self, cfg: V5Config, seed: Optional[int] = None, device: str = "cpu"):
        """
        Initialize V5 placer.
        
        Args:
            cfg: V5Config with placement parameters
            seed: Random seed for determinism
            device: Device for tensors (CPU recommended)
        """
        self.cfg = cfg
        self.device = device
        
        # Initialize RNG
        self.rng = random.Random(seed if seed is not None else 42)
        
        # Get placement config (with defaults)
        self.macro_keepout = getattr(cfg, 'macro_keepout', 2.0)
        self.stdcell_macro_keepout = getattr(cfg, 'stdcell_macro_keepout', 1.2)  # Default: 1.2 microns keepout for std-cells from macros
        self.macro_border_bias = getattr(cfg, 'macro_border_bias', 0.3)
        self.macro_max_tries = getattr(cfg, 'macro_max_tries', 200)
        self.macro_packer = getattr(cfg, 'macro_packer', 'shelf')

        # Determinism / realism controls
        self.macro_deterministic = getattr(cfg, 'macro_deterministic', False)
        self.macro_island_layout = getattr(cfg, 'macro_island_layout', 'random')
        self.macro_area_target_mode = getattr(cfg, 'macro_area_target_mode', 'random')
        self.stdcell_grid_shuffle = getattr(cfg, 'stdcell_grid_shuffle', True)
        
        # Store base hash cell size (adjusted adaptively based on graph size)
        self.base_hash_cell_size = getattr(cfg, 'hash_cell_size', 3.0)
        self.hash_cell_size = self.base_hash_cell_size  # Adjusted in place() method
        self.poisson_r = getattr(cfg, 'poisson_r', 2.5)
        self.poisson_max_attempts = getattr(cfg, 'poisson_max_attempts', 30)
        self.local_legalize_iters = getattr(cfg, 'local_legalize_iters', 2)
        self.local_shift_step = getattr(cfg, 'local_shift_step', 1.25)
        
        # TIER-1: Bounded overlap tolerance parameters
        self.max_overlap_pairs_per_instance = getattr(cfg, 'max_overlap_pairs_per_instance', 2)
        self.max_overlap_area_ratio = getattr(cfg, 'max_overlap_area_ratio', 0.01)  # 1% of instance area
        self.skip_legalization_if_good = getattr(cfg, 'skip_legalization_if_good', True)
        self.legalization_local_density_threshold = getattr(cfg, 'legalization_local_density_threshold', 0.7)
    
    def place(self, sizes: torch.Tensor, types: torch.Tensor, 
              canvas_W: float, canvas_H: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Place instances on canvas.
        
        Args:
            sizes: [N, 2] tensor of (w, h) in microns
            types: [N] tensor with type codes: 0=stdcell, 1=macro, 2=port
            canvas_W: Canvas width in microns
            canvas_H: Canvas height in microns
            
        Returns:
            (positions, sizes_out, mask, placed_mask) where:
            - positions: [N, 2] tensor of centers (x, y) in microns (only for placed instances)
            - sizes_out: [N, 2] tensor (only for placed instances)
            - mask: [N] bool tensor (is_port, only for placed instances)
            - placed_mask: [N] bool tensor indicating which instances were successfully placed
        """
        # Convert to CPU numpy for easier manipulation
        if sizes.device.type != 'cpu':
            sizes = sizes.cpu()
        if types.device.type != 'cpu':
            types = types.cpu()
        
        sizes_np = sizes.numpy()
        types_np = types.numpy()
        
        N = len(sizes_np)
        
        # Store types_np for use in _local_legalize (to skip stdcells)
        self.types_np = types_np
        
        # Separate instances by type
        stdcell_mask = (types_np == 0)
        macro_mask = (types_np == 1)
        port_mask = (types_np == 2)
        
        stdcell_indices = [i for i in range(N) if stdcell_mask[i]]
        macro_indices = [i for i in range(N) if macro_mask[i]]
        port_indices = [i for i in range(N) if port_mask[i]]
        
        # Adaptive hash cell size: scale with instance density for consistent behavior
        # For large graphs (10k+), use finer cells; for small graphs (500), use coarser cells
        if N > 5000:
            density_factor = math.sqrt(5000.0 / N)
            self.hash_cell_size = max(1.0, self.base_hash_cell_size * density_factor)
        elif N < 1000:
            density_factor = math.sqrt(N / 1000.0)
            self.hash_cell_size = min(6.0, self.base_hash_cell_size * density_factor)
        else:
            self.hash_cell_size = self.base_hash_cell_size
        
        # Initialize spatial hash (primary overlap checker)
        spatial_hash = SpatialHash2D(canvas_W, canvas_H, self.hash_cell_size)
        
        # Initialize output arrays (centers)
        positions = [[0.0, 0.0] for _ in range(N)]
        placed = [False] * N
        
        # Step B: Place macros first
        if macro_indices:
            self._place_macros(macro_indices, sizes_np, positions, placed, 
                             spatial_hash, canvas_W, canvas_H)
        
        # Step C: Place ports BEFORE stdcells (so we can filter occupied grids)
        if port_indices:
            self._place_ports(port_indices, sizes_np, positions, placed,
                            spatial_hash, canvas_W, canvas_H)
        
        # Step D: Place std-cells (grid-based, density-targeted)
        # Grids will be filtered to exclude those occupied by macros and ports
        if stdcell_indices:
            self._place_stdcells(stdcell_indices, sizes_np, positions, placed,
                               spatial_hash, canvas_W, canvas_H,
                               macro_indices=macro_indices,
                               port_indices=port_indices)
        
        # Step F: Local legalization (ONLY for macros and ports, NOT stdcells - grids are non-overlapping by design!)
        self._local_legalize(positions, sizes_np, placed, spatial_hash, 
                           canvas_W, canvas_H, skip_stdcells=True)
        
        # Step G: Validate and fix any unplaced instances
        unplaced_indices = [i for i in range(N) if not placed[i]]
        if unplaced_indices:
            enable_emergency = getattr(self.cfg, 'enable_emergency_placement', True)
            if enable_emergency:
                self._emergency_place(unplaced_indices, sizes_np, positions, placed,
                                    spatial_hash, canvas_W, canvas_H, types_np=types_np)
        
        # Final validation: ensure all instances are within bounds (overlap checking disabled for performance)
        self._validate_placement(positions, sizes_np, placed, canvas_W, canvas_H, check_overlaps=False)
        
        # Filter out unplaced instances - only return placed ones
        placed_indices = [i for i in range(N) if placed[i]]
        
        if not placed_indices:
            # Edge case: no instances placed
            positions_tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            sizes_tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            mask_tensor = torch.empty((0,), dtype=torch.bool, device=self.device)
            placed_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
            return positions_tensor, sizes_tensor, mask_tensor, placed_mask
        
        # Extract only placed instances
        placed_positions = [positions[i] for i in placed_indices]
        placed_sizes_np = sizes_np[placed_indices]
        placed_port_mask = [port_mask[i] for i in placed_indices]
        
        # Convert to tensors
        positions_tensor = torch.tensor(placed_positions, dtype=torch.float32, device=self.device)
        sizes_tensor = torch.tensor(placed_sizes_np, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(placed_port_mask, dtype=torch.bool, device=self.device)
        
        # Create placed_mask for original indices
        placed_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        for i in placed_indices:
            placed_mask[i] = True
        
        # Calculate and report placement density
        import numpy as np
        total_placed_area = np.sum(placed_sizes_np[:, 0] * placed_sizes_np[:, 1])
        canvas_area = canvas_W * canvas_H
        final_density = total_placed_area / canvas_area if canvas_area > 0 else 0.0
        target_density = getattr(self.cfg, 'target_density', 0.6)
        
        print(f"  Placement density: {final_density:.3f} (target: {target_density:.3f}, "
              f"placed: {len(placed_indices)}/{N} instances)")
        
        return positions_tensor, sizes_tensor, mask_tensor, placed_mask
    
    def _place_macros(self, macro_indices: List[int], sizes_np, positions: List[List[float]],
                     placed: List[bool], spatial_hash: SpatialHash2D, 
                     canvas_W: float, canvas_H: float):
        """Place macros using realistic industry patterns:
        
        1. Periphery-aligned: Macros along edges/corners (most common)
        2. Clustered islands: Groups of macros forming SRAM banks/cache slices
        3. Hybrid: Combination of both (most realistic)
        """
        if not macro_indices:
            return
        
        # Separate super-macros from regular macros FIRST (before scaling)
        # Super-macros are typically much larger than regular macros
        # Use 3x the median macro size as threshold to identify super-macros
        macro_areas = [sizes_np[i][0] * sizes_np[i][1] for i in macro_indices]
        if len(macro_areas) > 0:
            median_macro_area = sorted(macro_areas)[len(macro_areas) // 2]
            super_macro_threshold = median_macro_area * 3.0
        else:
            super_macro_threshold = float('inf')
        
        super_macro_indices = [i for i in macro_indices if sizes_np[i][0] * sizes_np[i][1] >= super_macro_threshold]
        regular_macro_indices = [i for i in macro_indices if sizes_np[i][0] * sizes_np[i][1] < super_macro_threshold]
        
        # Calculate target macro area (per-graph diversity: 15-55% including "community" designs)
        canvas_area = canvas_W * canvas_H
        macro_area_target_min = getattr(self.cfg, 'macro_area_target_min', 0.20)
        macro_area_target_max = getattr(self.cfg, 'macro_area_target_max', 0.35)
        macro_area_target_direct = getattr(self.cfg, 'macro_area_target', None)
        if macro_area_target_direct is not None:
            target_macro_area = canvas_area * macro_area_target_direct
        elif self.macro_deterministic or self.macro_area_target_mode == "mid":
            target_macro_area = canvas_area * (0.5 * (macro_area_target_min + macro_area_target_max))
        else:
            target_macro_area = canvas_area * self.rng.uniform(macro_area_target_min, macro_area_target_max)
        
        # Calculate current macro areas (including super-macros for target calculation)
        super_macro_area = sum(sizes_np[i][0] * sizes_np[i][1] for i in super_macro_indices)
        regular_macro_area = sum(sizes_np[i][0] * sizes_np[i][1] for i in regular_macro_indices)
        current_total_macro_area = super_macro_area + regular_macro_area
        
        # Adjust REGULAR macro sizes to hit target area (excluding super-macros)
        # Target area for regular macros = total target - super-macro area
        if regular_macro_area > 0 and len(regular_macro_indices) > 0:
            target_regular_macro_area = max(0, target_macro_area - super_macro_area)
            scale_factor = (target_regular_macro_area / regular_macro_area) ** 0.5
            # Allow scaling from 0.2x (small designs) to 3.0x (large designs)
            scale_factor = max(0.2, min(3.0, scale_factor))
            for i in regular_macro_indices:
                sizes_np[i][0] *= scale_factor
                sizes_np[i][1] *= scale_factor
        
        # Sort super-macros by area (largest first) - place largest super-macros first
        sorted_super_macros = sorted(super_macro_indices, key=lambda i: sizes_np[i][0] * sizes_np[i][1], reverse=True)
        
        # Shuffle regular macros (random order)
        if self.macro_deterministic:
            # Deterministic ordering: largest regular macros first
            regular_macro_indices_shuffled = sorted(
                regular_macro_indices,
                key=lambda i: sizes_np[i][0] * sizes_np[i][1],
                reverse=True,
            )
        else:
            regular_macro_indices_shuffled = regular_macro_indices.copy()
            self.rng.shuffle(regular_macro_indices_shuffled)
        
        # Combine: super-macros first (sorted by size), then regular macros (random)
        all_macros_ordered = sorted_super_macros + regular_macro_indices_shuffled
        
        # Macro community mode: cluster ALL macros into one region (older designs: macros occupy half chip)
        macro_community_mode = getattr(self.cfg, 'macro_community_mode', False)
        if macro_community_mode:
            self._place_macros_community(
                all_macros_ordered, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H
            )
            return
        
        # For super-macros: use special placement strategy (place them first, anywhere on canvas)
        if len(sorted_super_macros) > 0:
            self._place_super_macros_flexible(sorted_super_macros, sizes_np, positions, placed,
                                              spatial_hash, canvas_W, canvas_H, max_retries=10)
        
        # Determine placement strategy for regular macros
        strategy = getattr(self.cfg, 'macro_placement_strategy', 'hybrid')
        island_fraction = getattr(self.cfg, 'macro_island_fraction', 0.3)
        
        if strategy == 'periphery':
            periphery_macros = regular_macro_indices_shuffled
            island_macros = []
        elif strategy == 'islands':
            periphery_macros = []
            island_macros = regular_macro_indices_shuffled
        else:  # hybrid
            n_island = int(len(regular_macro_indices_shuffled) * island_fraction)
            island_macros = regular_macro_indices_shuffled[:n_island]
            periphery_macros = regular_macro_indices_shuffled[n_island:]
        
        # Place periphery macros (along edges, prioritizing corners) with retry
        self._place_periphery_macros_with_retry(periphery_macros, sizes_np, positions, placed, 
                                                 spatial_hash, canvas_W, canvas_H, max_retries=5)
        
        # Place island macros (clustered groups) with retry
        self._place_island_macros_with_retry(island_macros, sizes_np, positions, placed,
                                             spatial_hash, canvas_W, canvas_H, max_retries=5)
        
        # Report macro area coverage
        placed_macro_area = sum(sizes_np[i][0] * sizes_np[i][1] for i in macro_indices if placed[i])
        macro_coverage = placed_macro_area / canvas_area if canvas_area > 0 else 0.0
        
        # Report super-macro info
        placed_super_macros = [i for i in super_macro_indices if placed[i]]
        if len(placed_super_macros) > 0:
            super_macro_areas = [sizes_np[i][0] * sizes_np[i][1] for i in placed_super_macros]
            super_macro_area_total = sum(super_macro_areas)
            super_macro_coverage = super_macro_area_total / canvas_area if canvas_area > 0 else 0.0
            largest_super_macro_area = max(super_macro_areas)
            largest_super_macro_coverage = largest_super_macro_area / canvas_area if canvas_area > 0 else 0.0
            print(f"  Macro coverage: {macro_coverage:.3f} ({len([i for i in macro_indices if placed[i]])}/{len(macro_indices)} placed)")
            print(f"  Super-macros: {len(placed_super_macros)}/{len(super_macro_indices)} placed, "
                  f"total_area={super_macro_coverage*100:.1f}%, largest={largest_super_macro_coverage*100:.1f}%")
        else:
            print(f"  Macro coverage: {macro_coverage:.3f} ({len([i for i in macro_indices if placed[i]])}/{len(macro_indices)} placed)")
            if len(super_macro_indices) > 0:
                print(f"  WARNING: {len(super_macro_indices)} super-macros generated but 0 placed!")
    
    def _place_super_macros_flexible(self, super_macro_indices: List[int], sizes_np, positions: List[List[float]],
                                     placed: List[bool], spatial_hash: SpatialHash2D,
                                     canvas_W: float, canvas_H: float, max_retries: int = 10):
        """Place super-macros with flexible placement strategy - try corners, edges, and center regions."""
        if not super_macro_indices:
            return
        
        keepout = self.macro_keepout
        max_macro_tries = getattr(self.cfg, 'macro_max_tries', 200) * max_retries
        
        for macro_idx in super_macro_indices:
            if placed[macro_idx]:
                continue
            
            w, h = sizes_np[macro_idx]
            placed_success = False
            
            # Strategy 1: Try corners first (larger corner regions for super-macros)
            corner_size = min(canvas_W, canvas_H) * 0.25  # 25% for super-macros (larger than regular 15%)
            corners = [
                ('bottom-left', keepout, keepout, corner_size, corner_size),
                ('bottom-right', canvas_W - keepout - corner_size, keepout, corner_size, corner_size),
                ('top-left', keepout, canvas_H - keepout - corner_size, corner_size, corner_size),
                ('top-right', canvas_W - keepout - corner_size, canvas_H - keepout - corner_size, corner_size, corner_size),
            ]
            
            for corner_name, cx, cy, cw, ch in corners:
                if placed_success:
                    break
                for attempt in range(max_macro_tries // 4):
                    x = self.rng.uniform(cx + w/2, cx + cw - w/2)
                    y = self.rng.uniform(cy + h/2, cy + ch - h/2)
                    x = max(cx + w/2, min(cx + cw - w/2, x))
                    y = max(cy + h/2, min(cy + ch - h/2, y))
                    
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_success = True
                        break
            
            # Strategy 2: Try edges if corners failed
            if not placed_success:
                edges = ['left', 'right', 'top', 'bottom']
                for edge in edges:
                    if placed_success:
                        break
                    for attempt in range(max_macro_tries // 4):
                        if edge == 'left':
                            x = keepout + w / 2
                            y = self.rng.uniform(h/2 + keepout, canvas_H - h/2 - keepout)
                        elif edge == 'right':
                            x = canvas_W - keepout - w / 2
                            y = self.rng.uniform(h/2 + keepout, canvas_H - h/2 - keepout)
                        elif edge == 'top':
                            y = canvas_H - keepout - h / 2
                            x = self.rng.uniform(w/2 + keepout, canvas_W - w/2 - keepout)
                        else:  # bottom
                            y = keepout + h / 2
                            x = self.rng.uniform(w/2 + keepout, canvas_W - w/2 - keepout)
                        
                        if not spatial_hash.overlaps_any(x, y, w, h):
                            positions[macro_idx] = [x, y]
                            placed[macro_idx] = True
                            spatial_hash.insert(macro_idx, x, y, w, h)
                            placed_success = True
                            break
            
            # Strategy 3: Try anywhere on canvas if edges failed
            if not placed_success:
                for attempt in range(max_macro_tries):
                    x = self.rng.uniform(w/2 + keepout, canvas_W - w/2 - keepout)
                    y = self.rng.uniform(h/2 + keepout, canvas_H - h/2 - keepout)
                    
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_success = True
                        break
            
            # Debug: Warn if super-macro failed to place
            if not placed_success:
                print(f"  WARNING: Super-macro {macro_idx} (size={w:.1f}x{h:.1f} µm, "
                      f"area={w*h:.1f} µm², {w*h/(canvas_W*canvas_H)*100:.1f}% of canvas) failed to place")
    
    def _place_macros_community(self, macro_indices: List[int], sizes_np, positions: List[List[float]],
                                 placed: List[bool], spatial_hash: SpatialHash2D,
                                 canvas_W: float, canvas_H: float):
        """Place ALL macros in one contiguous region (community cluster).
        
        Older real-world designs: macros occupy one large region (e.g. half the chip),
        std-cells fill the rest. Region: left/right/top/bottom half.
        """
        if not macro_indices:
            return
        keepout = self.macro_keepout
        max_tries = getattr(self.cfg, 'macro_max_tries', 200) * 5
        
        # Pick region: left, right, top, or bottom half (each ~40-50% of canvas)
        region_choice = self.rng.choice(['left', 'right', 'top', 'bottom'])
        margin = min(canvas_W, canvas_H) * 0.05
        if region_choice == 'left':
            x_min, x_max = margin, canvas_W * 0.48
            y_min, y_max = margin, canvas_H - margin
        elif region_choice == 'right':
            x_min, x_max = canvas_W * 0.52, canvas_W - margin
            y_min, y_max = margin, canvas_H - margin
        elif region_choice == 'top':
            x_min, x_max = margin, canvas_W - margin
            y_min, y_max = canvas_H * 0.52, canvas_H - margin
        else:  # bottom
            x_min, x_max = margin, canvas_W - margin
            y_min, y_max = margin, canvas_H * 0.48
        
        # Sort by area (largest first) for packing
        ordered = sorted(macro_indices, key=lambda i: sizes_np[i][0] * sizes_np[i][1], reverse=True)
        
        # Shelf packing within region
        cur_x = x_min + keepout
        cur_y = y_min + keepout
        row_h = 0.0
        region_w = x_max - x_min - 2 * keepout
        region_h = y_max - y_min - 2 * keepout
        
        for idx in ordered:
            if placed[idx]:
                continue
            w, h = sizes_np[idx]
            if w > region_w or h > region_h:
                # Too large for region, try random placement in region
                for _ in range(max_tries // 4):
                    x = self.rng.uniform(x_min + w/2 + keepout, x_max - w/2 - keepout)
                    y = self.rng.uniform(y_min + h/2 + keepout, y_max - h/2 - keepout)
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[idx] = [x, y]
                        placed[idx] = True
                        spatial_hash.insert(idx, x, y, w, h)
                        break
                continue
            # New row if needed
            if cur_x + w > x_max - keepout:
                cur_x = x_min + keepout
                cur_y = cur_y + row_h + keepout * 0.5
                row_h = 0.0
            if cur_y + h > y_max - keepout:
                break
            x = cur_x + w / 2
            y = cur_y + h / 2
            if not spatial_hash.overlaps_any(x, y, w, h):
                positions[idx] = [x, y]
                placed[idx] = True
                spatial_hash.insert(idx, x, y, w, h)
                cur_x = cur_x + w + keepout * 0.5
                row_h = max(row_h, h)
            else:
                cur_x = cur_x + keepout
        
        # Fallback for any unplaced: try anywhere in region
        for idx in ordered:
            if placed[idx]:
                continue
            w, h = sizes_np[idx]
            for _ in range(max_tries):
                x = self.rng.uniform(x_min + w/2 + keepout, x_max - w/2 - keepout)
                y = self.rng.uniform(y_min + h/2 + keepout, y_max - h/2 - keepout)
                if not spatial_hash.overlaps_any(x, y, w, h):
                    positions[idx] = [x, y]
                    placed[idx] = True
                    spatial_hash.insert(idx, x, y, w, h)
                    break
        
        placed_count = sum(1 for i in macro_indices if placed[i])
        print(f"  Macro community ({region_choice}): {placed_count}/{len(macro_indices)} placed")
    
    def _place_periphery_macros_with_retry(self, macro_indices: List[int], sizes_np, positions: List[List[float]],
                                           placed: List[bool], spatial_hash: SpatialHash2D,
                                           canvas_W: float, canvas_H: float, max_retries: int = 5):
        """Place periphery macros with retry logic - try each macro up to max_retries times."""
        if not macro_indices:
            return
        
        # Increase max_macro_tries for retry attempts
        original_max_tries = getattr(self.cfg, 'macro_max_tries', 200)
        # Use more attempts per macro when retrying
        temp_max_tries = original_max_tries * max_retries
        
        # Temporarily increase max_tries
        old_max_tries = getattr(self.cfg, 'macro_max_tries', 200)
        self.cfg.macro_max_tries = temp_max_tries
        
        try:
            if self.macro_deterministic:
                self._place_periphery_macros_deterministic_shelves(
                    macro_indices, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H
                )
            else:
                self._place_periphery_macros(macro_indices, sizes_np, positions, placed,
                                             spatial_hash, canvas_W, canvas_H)
        finally:
            # Restore original max_tries
            self.cfg.macro_max_tries = old_max_tries
    
    def _place_island_macros_with_retry(self, macro_indices: List[int], sizes_np, positions: List[List[float]],
                                        placed: List[bool], spatial_hash: SpatialHash2D,
                                        canvas_W: float, canvas_H: float, max_retries: int = 5):
        """Place island macros with retry logic - try each macro up to max_retries times."""
        if not macro_indices:
            return
        
        # Increase max_macro_tries for retry attempts
        original_max_tries = getattr(self.cfg, 'macro_max_tries', 200)
        # Use more attempts per macro when retrying
        temp_max_tries = original_max_tries * max_retries
        
        # Temporarily increase max_tries
        old_max_tries = getattr(self.cfg, 'macro_max_tries', 200)
        self.cfg.macro_max_tries = temp_max_tries
        
        try:
            if self.macro_deterministic and self.macro_island_layout == "grid":
                self._place_island_macros_grid(
                    macro_indices, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H
                )
            else:
                self._place_island_macros(macro_indices, sizes_np, positions, placed,
                                         spatial_hash, canvas_W, canvas_H)
        finally:
            # Restore original max_tries
            self.cfg.macro_max_tries = old_max_tries

    def _place_periphery_macros_deterministic_shelves(
        self,
        macro_indices: List[int],
        sizes_np,
        positions: List[List[float]],
        placed: List[bool],
        spatial_hash: SpatialHash2D,
        canvas_W: float,
        canvas_H: float,
    ):
        """
        Deterministic periphery macro placement (floorplan-like).

        Places macros along edges using simple shelf packers (left/right vertical shelves,
        bottom/top horizontal shelves). This reduces placement entropy while keeping
        the common "macros on periphery" pattern.

        Falls back to the original stochastic placement for any macro that can't be placed.
        """
        if not macro_indices:
            return

        keepout = self.macro_keepout
        min_spacing = keepout * 0.5

        # Prefer larger macros first for better packing.
        ordered = sorted(macro_indices, key=lambda i: sizes_np[i][0] * sizes_np[i][1], reverse=True)

        # Edge cursors (where the next macro starts along that edge)
        left_y = keepout
        right_y = keepout
        bottom_x = keepout
        top_x = keepout

        def _try_place_at(x: float, y: float, w: float, h: float) -> bool:
            if x - w / 2 < 0 or x + w / 2 > canvas_W or y - h / 2 < 0 or y + h / 2 > canvas_H:
                return False
            return (not spatial_hash.overlaps_any(x, y, w, h))

        for idx in ordered:
            if placed[idx]:
                continue
            w, h = sizes_np[idx]

            placed_success = False

            # Deterministic corner tries (exact corners, then slightly inset).
            corner_candidates = [
                (keepout + w / 2, keepout + h / 2),  # bottom-left
                (canvas_W - keepout - w / 2, keepout + h / 2),  # bottom-right
                (keepout + w / 2, canvas_H - keepout - h / 2),  # top-left
                (canvas_W - keepout - w / 2, canvas_H - keepout - h / 2),  # top-right
            ]
            for (x, y) in corner_candidates:
                if _try_place_at(x, y, w, h):
                    positions[idx] = [x, y]
                    placed[idx] = True
                    spatial_hash.insert(idx, x, y, w, h)
                    placed_success = True
                    break

            if placed_success:
                continue

            # Shelf packing along edges (scan forward deterministically).
            # Left shelf
            x_left = keepout + w / 2
            y = left_y + h / 2
            if y + h / 2 <= canvas_H - keepout and _try_place_at(x_left, y, w, h):
                positions[idx] = [x_left, y]
                placed[idx] = True
                spatial_hash.insert(idx, x_left, y, w, h)
                left_y = y + h / 2 + min_spacing
                continue

            # Right shelf
            x_right = canvas_W - keepout - w / 2
            y = right_y + h / 2
            if y + h / 2 <= canvas_H - keepout and _try_place_at(x_right, y, w, h):
                positions[idx] = [x_right, y]
                placed[idx] = True
                spatial_hash.insert(idx, x_right, y, w, h)
                right_y = y + h / 2 + min_spacing
                continue

            # Bottom shelf
            y_bottom = keepout + h / 2
            x = bottom_x + w / 2
            if x + w / 2 <= canvas_W - keepout and _try_place_at(x, y_bottom, w, h):
                positions[idx] = [x, y_bottom]
                placed[idx] = True
                spatial_hash.insert(idx, x, y_bottom, w, h)
                bottom_x = x + w / 2 + min_spacing
                continue

            # Top shelf
            y_top = canvas_H - keepout - h / 2
            x = top_x + w / 2
            if x + w / 2 <= canvas_W - keepout and _try_place_at(x, y_top, w, h):
                positions[idx] = [x, y_top]
                placed[idx] = True
                spatial_hash.insert(idx, x, y_top, w, h)
                top_x = x + w / 2 + min_spacing
                continue

            # Fallback to original stochastic method for any hard-to-place macro.
            # (This preserves placement robustness.)
            self._place_periphery_macros([idx], sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)

    def _place_island_macros_grid(
        self,
        macro_indices: List[int],
        sizes_np,
        positions: List[List[float]],
        placed: List[bool],
        spatial_hash: SpatialHash2D,
        canvas_W: float,
        canvas_H: float,
    ):
        """
        Deterministic island placement: islands arranged on a coarse grid in the core area.

        This approximates SRAM-bank / cache-slice patterns and reduces stochasticity.
        """
        if not macro_indices:
            return

        keepout = self.macro_keepout
        island_size_range = getattr(self.cfg, 'macro_island_size_range', (3, 8))
        min_island_size, max_island_size = island_size_range

        # Group macros into islands deterministically (preserve input ordering).
        islands = []
        remaining = list(macro_indices)
        while remaining:
            island_size = max(min_island_size, min(max_island_size, len(remaining)))
            island = remaining[:island_size]
            remaining = remaining[island_size:]
            islands.append(island)

        n_islands = len(islands)
        if n_islands == 0:
            return

        # Grid of island centers in the core region
        margin = max(canvas_W, canvas_H) * 0.15
        x0, x1 = margin, canvas_W - margin
        y0, y1 = margin, canvas_H - margin

        if x1 <= x0 or y1 <= y0:
            # Degenerate core region; fall back to random islands
            return self._place_island_macros(macro_indices, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)

        import math
        cols = int(math.ceil(math.sqrt(n_islands)))
        rows = int(math.ceil(n_islands / cols))
        cols = max(cols, 1)
        rows = max(rows, 1)

        # Build list of grid cells (r, c). When macro_island_grid_shuffle is True, shuffle
        # so island positions vary across samples (top/bottom/center) instead of always
        # filling bottom row first (which caused fixed "middle bottom" pattern).
        cell_indices = [(r, c) for r in range(rows) for c in range(cols)]
        if getattr(self.cfg, 'macro_island_grid_shuffle', True):
            self.rng.shuffle(cell_indices)

        # Place each island's macros within a local "cell" around its center using a simple row packer.
        cell_w = (x1 - x0) / cols
        cell_h = (y1 - y0) / rows

        island_idx = 0
        for r, c in cell_indices:
            if island_idx >= n_islands:
                break
            island = islands[island_idx]
            island_idx += 1

            # Define island cell bounds (shrink slightly to leave gap between islands)
            cell_x_min = x0 + c * cell_w + keepout
            cell_x_max = x0 + (c + 1) * cell_w - keepout
            cell_y_min = y0 + r * cell_h + keepout
            cell_y_max = y0 + (r + 1) * cell_h - keepout

            if cell_x_max <= cell_x_min or cell_y_max <= cell_y_min:
                continue

            # Pack macros in this island deterministically: largest first, row-wise.
            island_ordered = sorted(island, key=lambda i: sizes_np[i][0] * sizes_np[i][1], reverse=True)
            cur_x = cell_x_min
            cur_y = cell_y_min
            row_h = 0.0

            for mid in island_ordered:
                if placed[mid]:
                    continue
                w, h = sizes_np[mid]

                # Start new row if needed
                if cur_x + w > cell_x_max:
                    cur_x = cell_x_min
                    cur_y = cur_y + row_h + keepout * 0.5
                    row_h = 0.0

                # If we can't fit vertically, stop trying in this cell
                if cur_y + h > cell_y_max:
                    break

                x = cur_x + w / 2
                y = cur_y + h / 2

                # Ensure within canvas bounds and non-overlap
                x = max(w / 2 + keepout, min(canvas_W - w / 2 - keepout, x))
                y = max(h / 2 + keepout, min(canvas_H - h / 2 - keepout, y))

                if not spatial_hash.overlaps_any(x, y, w, h):
                    positions[mid] = [x, y]
                    placed[mid] = True
                    spatial_hash.insert(mid, x, y, w, h)
                    cur_x = cur_x + w + keepout * 0.5
                    row_h = max(row_h, h)

        # Fallback: any remaining unplaced macros from islands go to stochastic island placement
        leftover = [i for i in macro_indices if not placed[i]]
        if leftover:
            self._place_island_macros(leftover, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)
    
    def _place_periphery_macros(self, macro_indices: List[int], sizes_np, positions: List[List[float]],
                                placed: List[bool], spatial_hash: SpatialHash2D,
                                canvas_W: float, canvas_H: float):
        """Place macros along chip periphery (edges and corners). Most common pattern."""
        if not macro_indices:
            return
        
        keepout = self.macro_keepout
        min_spacing = keepout * 0.5
        
        # Define corner regions (prioritize corners for largest macros)
        corner_size = min(canvas_W, canvas_H) * 0.15  # 15% of smaller dimension
        corners = [
            ('bottom-left', keepout, keepout, corner_size, corner_size),
            ('bottom-right', canvas_W - keepout - corner_size, keepout, corner_size, corner_size),
            ('top-left', keepout, canvas_H - keepout - corner_size, corner_size, corner_size),
            ('top-right', canvas_W - keepout - corner_size, canvas_H - keepout - corner_size, corner_size, corner_size),
        ]
        
        # Place largest macros in corners first
        n_corners = min(len(macro_indices), 4)
        corner_macros = macro_indices[:n_corners]
        edge_macros = macro_indices[n_corners:]
        
        # Place corner macros with retries
        max_macro_tries = getattr(self.cfg, 'macro_max_tries', 200)
        for idx, macro_idx in enumerate(corner_macros):
            w, h = sizes_np[macro_idx]
            placed_corner = False
            
            if idx < len(corners):
                corner_name, cx, cy, cw, ch = corners[idx]
                # Try multiple positions in corner region
                for attempt in range(max_macro_tries):
                    # Try random position within corner region
                    x = self.rng.uniform(cx + w/2, min(cx + cw - w/2, cx + cw))
                    y = self.rng.uniform(cy + h/2, min(cy + ch - h/2, cy + ch))
                    x = max(cx + w/2, min(cx + cw - w/2, x))
                    y = max(cy + h/2, min(cy + ch - h/2, y))
                    
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_corner = True
                        break
            
            if not placed_corner:
                # Fallback: try random placement near edge
                edge_macros.append(macro_idx)
        
        # Distribute remaining macros across edges
        edges = ['left', 'right', 'top', 'bottom']
        edge_positions = {
            'left': keepout,
            'right': canvas_W - keepout,
            'top': canvas_H - keepout,
            'bottom': keepout
        }
        edge_coords = {edge: 0.0 for edge in edges}
        
        for macro_idx in edge_macros:
            w, h = sizes_np[macro_idx]
            placed_edge = False
            
            # Try multiple edges and positions
            for attempt in range(max_macro_tries):
                # Choose edge (weighted by available space)
                edge_weights = []
                for edge in edges:
                    if edge in ['left', 'right']:
                        available = canvas_H - edge_coords[edge]
                    else:
                        available = canvas_W - edge_coords[edge]
                    edge_weights.append(max(0.1, available))
                
                edge = self.rng.choices(edges, weights=edge_weights)[0]
                
                # Try to place along chosen edge
                if edge == 'left':
                    x = edge_positions[edge] + w / 2
                    # Try random y position along edge
                    y = self.rng.uniform(h/2 + keepout, canvas_H - h/2 - keepout)
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_edge = True
                        break
                elif edge == 'right':
                    x = edge_positions[edge] - w / 2
                    y = self.rng.uniform(h/2 + keepout, canvas_H - h/2 - keepout)
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_edge = True
                        break
                elif edge == 'top':
                    y = edge_positions[edge] - h / 2
                    x = self.rng.uniform(w/2 + keepout, canvas_W - w/2 - keepout)
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_edge = True
                        break
                else:  # bottom
                    y = edge_positions[edge] + h / 2
                    x = self.rng.uniform(w/2 + keepout, canvas_W - w/2 - keepout)
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_edge = True
                        break
            
            # If still not placed, try random position anywhere on canvas
            if not placed_edge:
                for attempt in range(max_macro_tries // 4):  # Fewer attempts for random fallback
                    x = self.rng.uniform(w/2 + keepout, canvas_W - w/2 - keepout)
                    y = self.rng.uniform(h/2 + keepout, canvas_H - h/2 - keepout)
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[macro_idx] = [x, y]
                        placed[macro_idx] = True
                        spatial_hash.insert(macro_idx, x, y, w, h)
                        placed_edge = True
                        break
    
    def _place_island_macros(self, macro_indices: List[int], sizes_np, positions: List[List[float]],
                            placed: List[bool], spatial_hash: SpatialHash2D,
                            canvas_W: float, canvas_H: float):
        """Place macros in clustered islands (SRAM banks, cache slices). Second most common pattern."""
        if not macro_indices:
            return
        
        keepout = self.macro_keepout
        island_size_range = getattr(self.cfg, 'macro_island_size_range', (3, 8))
        min_island_size, max_island_size = island_size_range
        
        # Group macros into islands
        islands = []
        remaining = macro_indices.copy()
        
        while remaining:
            island_size = self.rng.randint(min_island_size, max_island_size + 1)
            island_size = min(island_size, len(remaining))
            island = remaining[:island_size]
            remaining = remaining[island_size:]
            islands.append(island)
        
        # Place each island as a cluster
        for island in islands:
            # Choose island center (avoid edges, leave room for periphery macros)
            margin = max(canvas_W, canvas_H) * 0.15  # 15% margin from edges
            center_x = self.rng.uniform(margin, canvas_W - margin)
            center_y = self.rng.uniform(margin, canvas_H - margin)
            
            # Place macros in island (compact cluster)
            island_spacing = keepout * 1.5
            placed_in_island = []
            
            for macro_idx in island:
                w, h = sizes_np[macro_idx]
                
                # Try to place near island center
                best_pos = None
                best_dist = float('inf')
                
                for attempt in range(50):
                    # Random offset from center (within reasonable radius)
                    radius = max(w, h) * (1 + len(placed_in_island) * 0.3)
                    angle = self.rng.uniform(0, 2 * 3.14159)
                    offset_x = self.rng.uniform(-radius, radius) * 0.5
                    offset_y = self.rng.uniform(-radius, radius) * 0.5
                    
                    x = center_x + offset_x
                    y = center_y + offset_y
                    
                    # Ensure within bounds
                    x = max(w / 2 + keepout, min(canvas_W - w / 2 - keepout, x))
                    y = max(h / 2 + keepout, min(canvas_H - h / 2 - keepout, y))
                    
                    # Check overlap
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        # Prefer positions closer to center
                        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                        if dist < best_dist:
                            best_pos = (x, y)
                            best_dist = dist
                
                if best_pos:
                    x, y = best_pos
                    positions[macro_idx] = [x, y]
                    placed[macro_idx] = True
                    spatial_hash.insert(macro_idx, x, y, w, h)
                    placed_in_island.append(macro_idx)
    
    def _generate_structured_grids(self, stdcell_sizes: np.ndarray,
                                    canvas_W: float, canvas_H: float
                                    ) -> List[Tuple[float, float, float, float]]:
        """Generate non-overlapping grid slots by row-by-row packing.

        Unlike random (RSA) placement, row-by-row packing reliably tiles the entire
        canvas, so we always have enough slots to reach any target density — even 75%+.
        Slots are non-overlapping by construction; no grid-to-grid overlap checking is
        needed during activation.

        Args:
            stdcell_sizes: [N, 2] array of (w, h) for all std-cells.
            canvas_W: Canvas width.
            canvas_H: Canvas height.

        Returns:
            List of (cx, cy, w, h) grid slot centres and sizes.
            Shuffled when stdcell_grid_shuffle=True (default).
        """
        if len(stdcell_sizes) == 0:
            return []

        widths  = stdcell_sizes[:, 0]
        heights = stdcell_sizes[:, 1]
        is_packing_alg = getattr(self.cfg, 'generation_pipeline', 'v5') == 'packing_alg'

        w_mean = float(np.mean(widths));   w_std = float(np.std(widths))
        w_min  = float(np.min(widths));    w_max = float(np.max(widths))
        h_mean = float(np.mean(heights));  h_std = float(np.std(heights))
        h_min  = float(np.min(heights));   h_max = float(np.max(heights))

        grid_points: List[Tuple[float, float, float, float]] = []
        y = 0.0
        fixed_row_height = getattr(self.cfg, 'stdcell_row_height', None)
        if fixed_row_height is not None:
            fixed_row_height = float(fixed_row_height)

        while y < canvas_H:
            # packing_alg can force one physical row height for all std-cells.
            if fixed_row_height is not None:
                row_h = fixed_row_height
            else:
                row_h = self.rng.gauss(h_mean, h_std * 0.5)
                row_h = max(h_min, min(h_max * 1.5, row_h))
            # At canvas boundary: skip the last sliver rather than clipping to it.
            # Clipping created cells with height 0.001–0.1 µm paired with normal widths
            # (aspect ratios up to 1000) which are visually wrong and unrealistic.
            if y + row_h > canvas_H:
                if fixed_row_height is not None:
                    break
                if canvas_H - y < h_min:  # remaining space smaller than min cell height — drop it
                    break
                row_h = canvas_H - y      # acceptable partial row
            if row_h <= 0:
                break

            row_cy = y + row_h / 2
            x = 0.0

            while x < canvas_W:
                # Sample cell width from the width distribution
                cell_w = self.rng.gauss(w_mean, w_std * 0.5)
                width_cap = w_max if is_packing_alg else (w_max * 1.5)
                cell_w = max(w_min, min(width_cap, cell_w))
                # Same boundary fix for the last cell in a row
                if x + cell_w > canvas_W:
                    remaining_w = canvas_W - x
                    if remaining_w < w_min:  # remaining sliver — drop it
                        break
                    if is_packing_alg and remaining_w > w_max:
                        break
                    cell_w = remaining_w
                if cell_w <= 0:
                    break

                cell_cx = x + cell_w / 2
                grid_points.append((cell_cx, row_cy, cell_w, row_h))
                x += cell_w

            y += row_h

        # Always shuffle: without this, unshuffled row-by-row order fills bottom-to-top
        # and the top half of the canvas is empty when density target is reached early.
        self.rng.shuffle(grid_points)

        return grid_points
    
    def _place_stdcells(self, stdcell_indices: List[int], sizes_np, positions: List[List[float]],
                       placed: List[bool], spatial_hash: SpatialHash2D,
                       canvas_W: float, canvas_H: float,
                       macro_indices: List[int] = None,
                       port_indices: List[int] = None):
        """Place std-cells using grid-based placement ONLY.
        
        Args:
            macro_indices: List of macro indices (for overlap detection)
        """
        # CRITICAL: Only use grid-based placement (jittered method removed to prevent overlaps)
        self._place_stdcells_simple_grid(stdcell_indices, sizes_np, positions, placed,
                                        spatial_hash, canvas_W, canvas_H,
                                        macro_indices, port_indices)
    
    def _place_stdcells_simple_grid(self, stdcell_indices: List[int], sizes_np, positions: List[List[float]],
                       placed: List[bool], spatial_hash: SpatialHash2D,
                       canvas_W: float, canvas_H: float,
                       macro_indices: List[int] = None,
                       port_indices: List[int] = None):
        """Place std-cells using grid-based placement.

        Strategy:
        1. Generate grids with sizes sampled from std-cell distribution (grids tile the canvas)
        2. Filter out grids overlapping with macros
        3. Activate grids one by one until target density is reached OR no grid slots remain
        4. Grid sizes become instance sizes (pre-sampled std-cell sizes are ignored)

        Stopping condition: target density reached, or the structured grid is exhausted.
        Node count does NOT limit placement.
        
        Args:
            macro_indices: List of macro indices (for overlap detection)
            stdcell_indices: List of instance indices to assign activated grids to
            sizes_np: Size array (std-cell sizes here are only used for grid generation stats, then ignored)
        """
        import numpy as np
        
        if not stdcell_indices:
            return
        
        # Extract std-cell size distribution ONLY for grid generation statistics
        # These sizes are used to compute chip size, but actual instance sizes come from grids
        stdcell_sizes = np.array([[sizes_np[i][0], sizes_np[i][1]] for i in stdcell_indices])
        
        # Generate candidate grids using row-by-row structured packing.
        # Structured grids tile the entire canvas without overlap, so we always have
        # enough slots to reach any target density (fixes RSA jamming at ~20-25%).
        grid_points = self._generate_structured_grids(stdcell_sizes, canvas_W, canvas_H)
        
        # spatial_hash already contains all placed macros and ports.
        # Structured grids are never inserted into it, so the hash only tracks macro/port obstacles.
        available_grids = grid_points
        # Calculate target density
        canvas_area = canvas_W * canvas_H
        target_density = getattr(self.cfg, 'target_density', 0.6)
        target_area = canvas_area * target_density
        
        # Calculate already placed area (macros)
        placed_area = 0.0
        if macro_indices:
            for i in macro_indices:
                if placed[i]:
                    placed_area += sizes_np[i][0] * sizes_np[i][1]
        
        # Activate grids until target density is reached OR we run out of grids.
        # Structured grids are non-overlapping by construction, so no grid-to-grid
        # overlap check is needed — we only skip grids that overlap macros or ports.
        activated_count = 0
        current_area = placed_area
        
        for grid_x, grid_y, grid_w, grid_h in available_grids:
            # Check current density BEFORE activating this grid
            current_density = current_area / canvas_area if canvas_area > 0 else 0.0
            
            # PRIMARY stop: density reached.
            if current_density >= target_density * 0.99:
                break

            # FALLBACK stop: ran out of instance slots (shouldn't happen — 10x pool generated).
            if activated_count >= len(stdcell_indices):
                break
            
            # Make instance slightly smaller than grid to create minimal spacing between instances
            spacing_factor_w = getattr(self.cfg, 'stdcell_grid_spacing_factor_w', 0.99)
            spacing_factor_h = getattr(self.cfg, 'stdcell_grid_spacing_factor_h', 0.99)
            instance_w = grid_w * spacing_factor_w
            instance_h = grid_h * spacing_factor_h
            instance_area = instance_w * instance_h
            
            # Skip if this grid slot directly overlaps a macro or port.
            # (Activated grids are NOT inserted into the spatial hash — they are
            # non-overlapping by construction, so no grid-to-grid check is needed.)
            if spatial_hash.overlaps_any(grid_x, grid_y, grid_w, grid_h):
                continue

            # Check keepout buffer around macros (macros need a larger empty margin)
            # OPTIMIZED: Use spatial_hash.query() to get only nearby candidates, then check if any are macros
            # This is much faster than looping through ALL macros (O(k) vs O(M) where k << M typically)
            keepout = self.stdcell_macro_keepout
            if macro_indices and keepout > 0:
                # Query spatial_hash with expanded bounds (grid + keepout) to get nearby candidates
                # This returns only instances in nearby cells, not all instances
                nearby_candidates = spatial_hash.query(grid_x, grid_y, grid_w + 2 * keepout, grid_h + 2 * keepout)
                
                # Check if any nearby candidate is a macro and violates keepout
                # Only need to check macros, not other instances (already checked above)
                macro_set = set(macro_indices)
                grid_x_min = grid_x - grid_w / 2 - keepout
                grid_y_min = grid_y - grid_h / 2 - keepout
                grid_x_max = grid_x + grid_w / 2 + keepout
                grid_y_max = grid_y + grid_h / 2 + keepout
                
                grid_overlaps_macro = False
                for candidate_idx in nearby_candidates:
                    # Only check macros
                    if candidate_idx not in macro_set or not placed[candidate_idx]:
                        continue
                    
                    # Check if expanded grid overlaps with macro (including keepout)
                    macro_x, macro_y = positions[candidate_idx]
                    macro_w, macro_h = sizes_np[candidate_idx]
                    macro_x_min = macro_x - macro_w / 2
                    macro_y_min = macro_y - macro_h / 2
                    macro_x_max = macro_x + macro_w / 2
                    macro_y_max = macro_y + macro_h / 2
                    
                    if (grid_x_min < macro_x_max and grid_x_max > macro_x_min and
                        grid_y_min < macro_y_max and grid_y_max > macro_y_min):
                        # Grid is too close to a macro (within keepout distance)
                        grid_overlaps_macro = True
                        break
                
                if grid_overlaps_macro:
                    # Skip this grid - it's too close to a macro (within keepout distance)
                    continue
            
            # Assign this grid to next available instance
            instance_idx = stdcell_indices[activated_count]
            
            # Activate this grid slot — grid size becomes the instance size.
            positions[instance_idx] = [grid_x, grid_y]
            sizes_np[instance_idx][0] = instance_w
            sizes_np[instance_idx][1] = instance_h
            placed[instance_idx] = True
            # NOTE: Do NOT insert into spatial_hash — structured grids are non-overlapping
            # by construction.  Inserting them would create 4× blocking area per slot and
            # cause RSA jamming, limiting density to ~20-25% instead of the target 40-76%.
            
            activated_count += 1
            current_area += instance_area
            
            # After placing, check density again (in case this grid pushed us over target)
            current_density_after = current_area / canvas_area if canvas_area > 0 else 0.0
            if current_density_after >= target_density * 0.99:
                break
        
        # Report placement statistics
        final_density = current_area / canvas_area if canvas_area > 0 else 0.0
        total_grids = len(available_grids)
        skipped_count = total_grids - activated_count if total_grids >= activated_count else 0
        # WARN: If the instance pool was exhausted before reaching target density, the graph is under-dense.
        # This should not happen with the 10x pool; flag it for investigation if it does.
        hit_slot_cap = activated_count >= len(stdcell_indices)
        below_target = final_density < target_density * 0.99
        if hit_slot_cap and below_target:
            print(f"  [WARN] Instance pool exhausted before target density: {activated_count}/{len(stdcell_indices)} slots, "
                  f"density {final_density:.3f} < target {target_density:.3f}. "
                  "Increase n_instances_to_generate multiplier to fix.")
        print(f"  Std-cell placement: {activated_count}/{len(stdcell_indices)} instances assigned, "
              f"{activated_count}/{total_grids} grids activated ({skipped_count} skipped), "
              f"density: {final_density:.3f} (target: {target_density:.3f})")
    
    def _place_stdcells_jittered(self, stdcell_indices: List[int], sizes_np, positions: List[List[float]],
                                 placed: List[bool], spatial_hash: SpatialHash2D,
                                 canvas_W: float, canvas_H: float):
        """Place std-cells using jittered grid fill (old method).
        
        Strategy:
        1. Compute grid spacing based on average area and target density
        2. Generate regular grid points
        3. For each grid point, try to place std-cells with jitter
        4. Use local jitter attempts if first attempt fails
        5. Fallback to random placement for remaining cells
        """
        if not stdcell_indices:
            return
        
        # Compute average std-cell area for grid spacing
        stdcell_areas = [sizes_np[i][0] * sizes_np[i][1] for i in stdcell_indices]
        avg_area = sum(stdcell_areas) / len(stdcell_areas) if stdcell_areas else 5.0
        
        # Grid spacing: s = sqrt(avg_area / target_density)
        target_density = getattr(self.cfg, 'target_density', 0.6)
        s = math.sqrt(avg_area / target_density)
        
        # Ensure grid spacing is reasonable (at least 1.0, use max std-cell size as upper bound)
        max_w = max(sizes_np[i][0] for i in stdcell_indices) if stdcell_indices else 5.0
        max_h = max(sizes_np[i][1] for i in stdcell_indices) if stdcell_indices else 3.0
        s = max(1.0, min(s, max_w * 1.5, max_h * 1.5))  # Ensure grid fits largest std-cell
        
        # Generate grid points covering free space
        grid_points = []
        # Use max std-cell size for grid bounds (to ensure all cells can fit)
        grid_start_x = int(max_w / 2)
        grid_start_y = int(max_h / 2)
        grid_step = int(max(1, s))  # Ensure step >= 1 to avoid infinite loop
        
        for y in range(grid_start_y, int(canvas_H - max_h/2), grid_step):
            for x in range(grid_start_x, int(canvas_W - max_w/2), grid_step):
                grid_points.append((x, y))
        
        # Shuffle for random order
        self.rng.shuffle(grid_points)
        
        # Place std-cells at grid points with jitter
        stdcell_idx = 0
        for grid_x, grid_y in grid_points:
            if stdcell_idx >= len(stdcell_indices):
                break
            
            idx = stdcell_indices[stdcell_idx]
            w, h = sizes_np[idx]
            
            # Jitter point
            jitter_x = self.rng.uniform(-0.3 * s, 0.3 * s)
            jitter_y = self.rng.uniform(-0.3 * s, 0.3 * s)
            x = grid_x + jitter_x
            y = grid_y + jitter_y
            
            # Clamp to canvas bounds
            x = max(w/2, min(canvas_W - w/2, x))
            y = max(h/2, min(canvas_H - h/2, y))
            
            # Try K local jitters if first attempt fails
            placed_cell = False
            for local_attempt in range(self.poisson_max_attempts):
                if local_attempt > 0:
                    # Additional jitter
                    local_jitter_x = self.rng.uniform(-s * 0.2, s * 0.2)
                    local_jitter_y = self.rng.uniform(-s * 0.2, s * 0.2)
                    x = grid_x + jitter_x + local_jitter_x
                    y = grid_y + jitter_y + local_jitter_y
                    x = max(w/2, min(canvas_W - w/2, x))
                    y = max(h/2, min(canvas_H - h/2, y))
                
                # Check overlap (spatial_hash is precise, no need for occupancy_grid)
                if spatial_hash.overlaps_any(x, y, w, h):
                    continue
                
                # Place std-cell
                positions[idx] = [x, y]
                placed[idx] = True
                spatial_hash.insert(idx, x, y, w, h)
                placed_cell = True
                stdcell_idx += 1
                break
        
        # Place remaining std-cells randomly in free space
        for idx in stdcell_indices[stdcell_idx:]:
            if placed[idx]:
                continue
            
            w, h = sizes_np[idx]
            
            # Check if std-cell can fit
            min_x = w / 2
            max_x = canvas_W - w / 2
            min_y = h / 2
            max_y = canvas_H - h / 2
            
            if max_x <= min_x or max_y <= min_y:
                # Too large, skip (will be handled by emergency placement)
                continue
            
            placed_cell = False
            
            for attempt in range(self.poisson_max_attempts * 2):  # More attempts for fallback
                # Try random position (spatial_hash will check legality)
                x = self.rng.uniform(min_x, max_x)
                y = self.rng.uniform(min_y, max_y)
                
                # Check legality
                if spatial_hash.overlaps_any(x, y, w, h):
                    continue
                
                # Place std-cell
                positions[idx] = [x, y]
                placed[idx] = True
                spatial_hash.insert(idx, x, y, w, h)
                placed_cell = True
                break
    
    def _place_ports(self, port_indices: List[int], sizes_np, positions: List[List[float]],
                    placed: List[bool], spatial_hash: SpatialHash2D,
                    canvas_W: float, canvas_H: float):
        """Place ports on chip borders only.

        Strategies:
        - border_balanced: Deterministic allocation across all 4 edges.
        - border_uniform:   Ports uniformly distributed along all 4 edges.
        - border_clustered: Ports concentrated on 1–3 selected border edges.
        """
        strategy = getattr(self.cfg, 'port_placement_strategy', 'border_uniform')
        if strategy == 'border_balanced':
            self._place_ports_border_balanced(port_indices, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)
        elif strategy == 'border_clustered':
            self._place_ports_border_clustered(port_indices, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)
        else:
            self._place_ports_border(port_indices, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)

    def _place_ports_border_balanced(self, port_indices: List[int], sizes_np, positions: List[List[float]],
                                     placed: List[bool], spatial_hash: SpatialHash2D,
                                     canvas_W: float, canvas_H: float):
        """Place ports on all four borders with balanced counts and even spacing."""
        if not port_indices:
            return

        sides = ['left', 'right', 'top', 'bottom']
        side_lengths = {
            'left': max(0.0, canvas_H - sizes_np[port_indices[0]][1]),
            'right': max(0.0, canvas_H - sizes_np[port_indices[0]][1]),
            'top': max(0.0, canvas_W - sizes_np[port_indices[0]][0]),
            'bottom': max(0.0, canvas_W - sizes_np[port_indices[0]][0]),
        }
        total_length = sum(side_lengths.values())
        if total_length <= 0:
            return

        counts = {side: 1 for side in sides}
        remaining = max(0, len(port_indices) - len(sides))
        if remaining > 0:
            ideals = {side: len(port_indices) * side_lengths[side] / total_length for side in sides}
            while remaining > 0:
                side = max(sides, key=lambda s: ideals[s] - counts[s])
                counts[side] += 1
                remaining -= 1

        self.rng.shuffle(port_indices)
        min_spacing = float(getattr(self.cfg, 'packing_alg_port_min_spacing', 1.25))
        cursor = 0
        for side in sides:
            side_count = counts[side]
            if side_count <= 0:
                continue

            port_subset = port_indices[cursor:cursor + side_count]
            cursor += side_count
            if not port_subset:
                continue

            w, h = sizes_np[port_subset[0]]
            if side in ('left', 'right'):
                usable = max(0.0, canvas_H - h)
                if usable <= 0:
                    continue
                step = usable / (side_count + 1)
                if step < min_spacing:
                    side_count = max(1, int(usable / max(min_spacing, 1e-6)))
                    port_subset = port_subset[:side_count]
                    step = usable / (side_count + 1)
                y_positions = [h / 2 + step * (i + 1) for i in range(side_count)]
                x_pos = w / 2 if side == 'left' else canvas_W - w / 2
                candidates = [(idx, x_pos, y) for idx, y in zip(port_subset, y_positions)]
            else:
                usable = max(0.0, canvas_W - w)
                if usable <= 0:
                    continue
                step = usable / (side_count + 1)
                if step < min_spacing:
                    side_count = max(1, int(usable / max(min_spacing, 1e-6)))
                    port_subset = port_subset[:side_count]
                    step = usable / (side_count + 1)
                x_positions = [w / 2 + step * (i + 1) for i in range(side_count)]
                y_pos = canvas_H - h / 2 if side == 'top' else h / 2
                candidates = [(idx, x, y_pos) for idx, x in zip(port_subset, x_positions)]

            for idx, x, y in candidates:
                w_i, h_i = sizes_np[idx]
                if not spatial_hash.overlaps_any(x, y, w_i, h_i):
                    positions[idx] = [x, y]
                    placed[idx] = True
                    spatial_hash.insert(idx, x, y, w_i, h_i)

        # Rare fallback: if any balanced slot could not be used, fall back to the
        # all-border search while keeping every port on the boundary.
        unplaced = [idx for idx in port_indices if not placed[idx]]
        if unplaced:
            self._place_ports_border(unplaced, sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)
    
    def _place_ports_border(self, port_indices: List[int], sizes_np, positions: List[List[float]],
                    placed: List[bool], spatial_hash: SpatialHash2D,
                    canvas_W: float, canvas_H: float):
        """Place ports on canvas borders (uniform distribution along edges)."""
        for idx in port_indices:
            w, h = sizes_np[idx]
            
            # Check overlap with ALL placed instances (including other ports)
            # NO EXCLUSIONS: Ports should not overlap with anything
            
            # CRITICAL: Ports MUST be on borders only - try all borders systematically
            placed_port = False
            
            # Define all border positions we can try
            # Try multiple positions along each border for better coverage
            border_positions = []
            num_positions_per_border = 20  # Try 20 positions per border
            
            # Left border: x = w/2, y varies
            if canvas_H - h > 0:  # Check if port can fit vertically
                y_min = h/2
                y_max = canvas_H - h/2
                for i in range(num_positions_per_border):
                    y_pos = y_min + (y_max - y_min) * i / max(1, num_positions_per_border - 1)
                    y_pos = max(y_min, min(y_max, y_pos))  # Clamp to valid range
                    border_positions.append(('left', w/2, y_pos))
            
            # Right border: x = canvas_W - w/2, y varies
            if canvas_H - h > 0:
                y_min = h/2
                y_max = canvas_H - h/2
                for i in range(num_positions_per_border):
                    y_pos = y_min + (y_max - y_min) * i / max(1, num_positions_per_border - 1)
                    y_pos = max(y_min, min(y_max, y_pos))
                    border_positions.append(('right', canvas_W - w/2, y_pos))
            
            # Top border: y = canvas_H - h/2, x varies
            if canvas_W - w > 0:  # Check if port can fit horizontally
                x_min = w/2
                x_max = canvas_W - w/2
                for i in range(num_positions_per_border):
                    x_pos = x_min + (x_max - x_min) * i / max(1, num_positions_per_border - 1)
                    x_pos = max(x_min, min(x_max, x_pos))  # Clamp to valid range
                    border_positions.append(('top', x_pos, canvas_H - h/2))
            
            # Bottom border: y = h/2, x varies
            if canvas_W - w > 0:
                x_min = w/2
                x_max = canvas_W - w/2
                for i in range(num_positions_per_border):
                    x_pos = x_min + (x_max - x_min) * i / max(1, num_positions_per_border - 1)
                    x_pos = max(x_min, min(x_max, x_pos))
                    border_positions.append(('bottom', x_pos, h/2))
            
            # Shuffle border positions for randomness
            self.rng.shuffle(border_positions)
            
            # Try each border position
            for border_name, x, y in border_positions:
                # CRITICAL: Check if legal - ports must NOT overlap with ANYTHING (instances or other ports)
                if not spatial_hash.overlaps_any(x, y, w, h):
                    # Found valid border position - place port
                    positions[idx] = [x, y]
                    placed[idx] = True
                    spatial_hash.insert(idx, x, y, w, h)
                    placed_port = True
                    break
            
            # NO FALLBACK: If port cannot be placed on any border, leave it unplaced
            if not placed_port:
                # Port remains unplaced (placed[idx] = False)
                pass
    
    def _place_ports_border_clustered(self, port_indices: List[int], sizes_np, positions: List[List[float]],
                                      placed: List[bool], spatial_hash: SpatialHash2D,
                                      canvas_W: float, canvas_H: float):
        """Place ports on 1–3 border edges only (clustered variant).

        Ports remain on the die boundary at all times; only the active edge(s) vary.
        If a port can't fit on the chosen edges, falls back to all-border uniform.
        """
        all_edges = ['left', 'right', 'top', 'bottom']
        n_active = self.rng.choice([1, 2, 2, 3])  # weights: 25% one, 50% two, 25% three
        active_edges = self.rng.sample(all_edges, n_active)
        num_per_edge = 40  # candidate positions sampled per active edge

        for idx in port_indices:
            w, h = sizes_np[idx]
            border_positions = []

            if 'left' in active_edges and canvas_H - h > 0:
                y_min, y_max = h / 2, canvas_H - h / 2
                for _ in range(num_per_edge):
                    border_positions.append(('left', w / 2, self.rng.uniform(y_min, y_max)))

            if 'right' in active_edges and canvas_H - h > 0:
                y_min, y_max = h / 2, canvas_H - h / 2
                for _ in range(num_per_edge):
                    border_positions.append(('right', canvas_W - w / 2, self.rng.uniform(y_min, y_max)))

            if 'top' in active_edges and canvas_W - w > 0:
                x_min, x_max = w / 2, canvas_W - w / 2
                for _ in range(num_per_edge):
                    border_positions.append(('top', self.rng.uniform(x_min, x_max), canvas_H - h / 2))

            if 'bottom' in active_edges and canvas_W - w > 0:
                x_min, x_max = w / 2, canvas_W - w / 2
                for _ in range(num_per_edge):
                    border_positions.append(('bottom', self.rng.uniform(x_min, x_max), h / 2))

            if not border_positions:
                # Active edges don't fit this port — fall back to uniform border placement
                self._place_ports_border([idx], sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)
                continue

            self.rng.shuffle(border_positions)
            placed_port = False
            for _edge, x, y in border_positions:
                if not spatial_hash.overlaps_any(x, y, w, h):
                    positions[idx] = [x, y]
                    placed[idx] = True
                    spatial_hash.insert(idx, x, y, w, h)
                    placed_port = True
                    break

            if not placed_port:
                # Clustered edges are too full — fall back to all-border uniform
                self._place_ports_border([idx], sizes_np, positions, placed, spatial_hash, canvas_W, canvas_H)
    
    def _is_instance_good(self, idx: int, positions: List[List[float]], sizes_np,
                          spatial_hash: SpatialHash2D, canvas_W: float, canvas_H: float) -> bool:
        """TIER-1: Check if instance is 'good' (no overlaps, low local density).
        
        Returns True if instance:
        - Has no overlaps
        - Local density is below threshold
        
        This allows skipping legalization for instances that don't need it.
        """
        x, y = positions[idx]
        w, h = sizes_np[idx]
        
        # Check for overlaps using spatial_hash (fast)
        if spatial_hash.overlaps_any(x, y, w, h, exclude_rect_ids={idx}):
            return False
        
        # Check local density (simple heuristic: count neighbors in nearby region)
        # Query a slightly larger region to estimate local density
        query_radius = max(w, h) * 2.0
        neighbors = spatial_hash.query(x, y, query_radius, query_radius)
        neighbors.discard(idx)
        
        # Estimate local density: area of neighbors / query area
        neighbor_area = 0.0
        for other_idx in neighbors:
            if other_idx < len(sizes_np):
                ow, oh = sizes_np[other_idx]
                neighbor_area += ow * oh
        
        query_area = query_radius * query_radius * 4  # (2*radius)^2
        local_density = neighbor_area / query_area if query_area > 0 else 0.0
        
        return local_density < self.legalization_local_density_threshold
    
    def _local_legalize(self, positions: List[List[float]], sizes_np,
                       placed: List[bool], spatial_hash: SpatialHash2D,
                       canvas_W: float, canvas_H: float, skip_stdcells: bool = False):
        """Local legalization: push overlapping instances apart.
        
        TIER-1: Skip legalization for 'good' instances (no overlaps, low local density).
        
        Args:
            skip_stdcells: If True, skip legalization for stdcells (they're in non-overlapping grids)
        """
        import numpy as np
        
        # Rebuild spatial hash with all placed instances
        spatial_hash.clear()
        for i, (x, y) in enumerate(positions):
            if placed[i]:
                w, h = sizes_np[i]
                spatial_hash.insert(i, x, y, w, h)
        
        # Get types if available (to skip stdcells)
        types_np = None
        if skip_stdcells and hasattr(self, 'types_np'):
            types_np = self.types_np
        
        # Legalization iterations
        for iteration in range(self.local_legalize_iters):
            # Random order - exclude stdcells if skip_stdcells=True
            if skip_stdcells and types_np is not None:
                indices = [i for i in range(len(positions)) if placed[i] and types_np[i] != 0]
            else:
                indices = [i for i in range(len(positions)) if placed[i]]
            
            if not indices:
                break
                
            self.rng.shuffle(indices)
            
            for idx in indices:
                x, y = positions[idx]
                w, h = sizes_np[idx]
                
                # TIER-1: Skip legalization for 'good' instances (fast early-exit heuristic)
                if self.skip_legalization_if_good:
                    if self._is_instance_good(idx, positions, sizes_np, spatial_hash, canvas_W, canvas_H):
                        continue
                
                # Query neighbors
                candidates = spatial_hash.query(x, y, w, h)
                candidates.discard(idx)  # Exclude self
                
                if not candidates:
                    continue
                
                # Check for overlaps
                overlaps = []
                x_min = x - w/2
                y_min = y - h/2
                x_max = x + w/2
                y_max = y + h/2
                
                for other_idx in candidates:
                    if other_idx >= len(positions) or not placed[other_idx]:
                        continue
                    ox, oy = positions[other_idx]
                    ow, oh = sizes_np[other_idx]
                    
                    ox_min = ox - ow/2
                    oy_min = oy - oh/2
                    ox_max = ox + ow/2
                    oy_max = oy + oh/2
                    
                    # Check intersection
                    if (x_min < ox_max and ox_min < x_max and
                        y_min < oy_max and oy_min < y_max):
                        overlaps.append(other_idx)
                
                # If no overlaps, skip legalization
                if not overlaps:
                    continue
                
                # TIER-1: Allow bounded overlap (hard local cap)
                # If overlaps are within tolerance, skip legalization for this instance
                if len(overlaps) <= self.max_overlap_pairs_per_instance:
                    # Check overlap area ratio
                    instance_area = w * h
                    max_allowed_overlap_area = instance_area * self.max_overlap_area_ratio
                    
                    total_overlap_area = 0.0
                    for other_idx in overlaps:
                        ox, oy = positions[other_idx]
                        ow, oh = sizes_np[other_idx]
                        
                        # Compute intersection area
                        overlap_w = max(0, min(x_max, ox + ow/2) - max(x_min, ox - ow/2))
                        overlap_h = max(0, min(y_max, oy + oh/2) - max(y_min, oy - oh/2))
                        total_overlap_area += overlap_w * overlap_h
                    
                    if total_overlap_area <= max_allowed_overlap_area:
                        # Overlaps are within tolerance - skip legalization
                        continue
                
                # Compute push vector (away from centroid of overlapping neighbors)
                overlap_centroid_x = sum(positions[o][0] for o in overlaps) / len(overlaps)
                overlap_centroid_y = sum(positions[o][1] for o in overlaps) / len(overlaps)
                
                dx = x - overlap_centroid_x
                dy = y - overlap_centroid_y
                
                # Normalize
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 1e-6:
                    dx /= dist
                    dy /= dist
                else:
                    # Random direction if too close
                    angle = self.rng.uniform(0, 2 * math.pi)
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                
                # Propose move
                step = self.local_shift_step
                new_x = x + step * dx
                new_y = y + step * dy
                
                # Clamp to bounds
                new_x = max(w/2, min(canvas_W - w/2, new_x))
                new_y = max(h/2, min(canvas_H - h/2, new_y))
                
                # TIER-0 FIX: Re-query neighbors at NEW location using overlaps_any
                # This is critical - checking at old location causes "moved into new cell → invisible collision" bug
                # Use spatial_hash.overlaps_any which efficiently queries only overlapping grid cells
                if not spatial_hash.overlaps_any(new_x, new_y, w, h, exclude_rect_ids={idx}):
                    # Move is legal - accept it
                    spatial_hash.remove(idx)
                    positions[idx] = [new_x, new_y]
                    # Update spatial hash
                    spatial_hash.insert(idx, new_x, new_y, w, h)
    
    def _emergency_place(self, unplaced_indices: List[int], sizes_np, positions: List[List[float]],
                        placed: List[bool], spatial_hash: SpatialHash2D,
                        canvas_W: float, canvas_H: float, types_np=None):
        """Emergency placement for instances that failed normal placement.
        
        TIER-0 FIX: Sequential placement (not batch) to prevent new-new overlaps.
        TIER-0 FIX: Reject instances that are too large for canvas (don't force-place).
        
        CRITICAL: Should NOT place stdcells - they come from grids only!
        """
        if not unplaced_indices:
            return
        
        # CRITICAL: Filter out stdcells - they should NOT be emergency-placed!
        if types_np is not None:
            unplaced_indices = [idx for idx in unplaced_indices if types_np[idx] != 0]
        
        if not unplaced_indices:
            return
        
        # TIER-0 FIX: Process instances sequentially (not batch) to prevent new-new overlaps
        # Emergency set is small (<5-10% of instances), so sequential is still fast
        candidates_per_instance = 5  # Try 5 candidates per instance
        
        for idx in unplaced_indices:
            w, h = sizes_np[idx]
            min_x = w / 2
            max_x = canvas_W - w / 2
            min_y = h / 2
            max_y = canvas_H - h / 2
            
            # TIER-0 FIX: Reject instances that are too large for canvas (don't force-place)
            # These cases are unlearnable anyway, and polluting training data is worse than dropping samples
            if max_x <= min_x or max_y <= min_y:
                # Instance exceeds canvas - reject (leave unplaced)
                continue
            
            # Generate candidate positions
            grid_step = max(w, h) * 1.5
            candidates = []
            
            # Grid-based candidates
            for y_grid in range(int(min_y), int(max_y), max(1, int(grid_step))):
                for x_grid in range(int(min_x), int(max_x), max(1, int(grid_step))):
                    if len(candidates) >= candidates_per_instance:
                        break
                    x = x_grid + self.rng.uniform(0, min(w, grid_step * 0.3))
                    y = y_grid + self.rng.uniform(0, min(h, grid_step * 0.3))
                    x = max(min_x, min(max_x, x))
                    y = max(min_y, min(max_y, y))
                    candidates.append((x, y))
            
            # Add random candidates to fill up to candidates_per_instance
            while len(candidates) < candidates_per_instance:
                x = self.rng.uniform(min_x, max_x)
                y = self.rng.uniform(min_y, max_y)
                candidates.append((x, y))
            
            # Try candidates sequentially - place first one that doesn't overlap
            placed_success = False
            for x, y in candidates:
                # TIER-0 FIX: Check overlaps using spatial_hash (includes all previously placed instances)
                if not spatial_hash.overlaps_any(x, y, w, h):
                    positions[idx] = [x, y]
                    placed[idx] = True
                    # TIER-0 FIX: Update spatial_hash immediately after placement
                    spatial_hash.insert(idx, x, y, w, h)
                    placed_success = True
                    break
            
            # Last resort: try corners if grid candidates failed
            if not placed_success:
                corner_candidates = [
                    (min_x, min_y),  # Bottom-left
                    (max_x, min_y),  # Bottom-right
                    (min_x, max_y),  # Top-left
                    (max_x, max_y),  # Top-right
                ]
                self.rng.shuffle(corner_candidates)  # Random order
                
                for x, y in corner_candidates:
                    if not spatial_hash.overlaps_any(x, y, w, h):
                        positions[idx] = [x, y]
                        placed[idx] = True
                        spatial_hash.insert(idx, x, y, w, h)
                        placed_success = True
                        break
            
            # If all candidates overlap, leave instance unplaced rather than creating overlap
            # (Overlapping placement is worse than missing instances)
    
    def _validate_placement(self, positions: List[List[float]], sizes_np,
                           placed: List[bool], canvas_W: float, canvas_H: float,
                           types_np=None, check_overlaps: bool = False):
        """Validate that all placed instances are within bounds.
        
        Repair out-of-bounds instances without introducing new overlaps.

        Historically this function silently clamped instances back into the canvas
        without checking whether the repaired position collided with other placed
        nodes. That breaks the macro-first + masked-grid invariant by allowing a
        macro to be pushed on top of already-legal std-cells during final repair.
        For any out-of-bounds node, search for a nearby legal in-bounds position;
        if none is found, mark it unplaced instead of creating invalid geometry.
        """

        def _clamp_center(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
            return (
                max(w / 2, min(canvas_W - w / 2, x)),
                max(h / 2, min(canvas_H - h / 2, y)),
            )

        def _build_hash(exclude_idx: Optional[int] = None) -> SpatialHash2D:
            repair_hash = SpatialHash2D(canvas_W, canvas_H, self.hash_cell_size)
            for j, (ox, oy) in enumerate(positions):
                if not placed[j] or j == exclude_idx:
                    continue
                ow, oh = sizes_np[j]
                repair_hash.insert(j, ox, oy, ow, oh)
            return repair_hash

        def _candidate_positions(base_x: float, base_y: float, w: float, h: float):
            yield (base_x, base_y)

            step = max(0.5, float(getattr(self, "local_shift_step", 1.25)), min(w, h) * 0.5)
            for radius_mul in (1.0, 2.0, 4.0, 8.0):
                delta = step * radius_mul
                for dx, dy in (
                    (-delta, 0.0), (delta, 0.0), (0.0, -delta), (0.0, delta),
                    (-delta, -delta), (-delta, delta), (delta, -delta), (delta, delta),
                ):
                    yield _clamp_center(base_x + dx, base_y + dy, w, h)

            # Fall back to a sparse in-bounds scan before dropping the instance.
            min_x = w / 2
            max_x = canvas_W - w / 2
            min_y = h / 2
            max_y = canvas_H - h / 2
            if max_x <= min_x or max_y <= min_y:
                return

            grid_step = max(1.0, max(w, h) * 1.5)
            y_scan = min_y
            attempts = 0
            while y_scan <= max_y and attempts < 64:
                x_scan = min_x
                while x_scan <= max_x and attempts < 64:
                    yield (x_scan, y_scan)
                    x_scan += grid_step
                    attempts += 1
                y_scan += grid_step

        for i, (x, y) in enumerate(positions):
            if not placed[i]:
                continue
            
            w, h = sizes_np[i]
            
            # Check bounds (considering width/height)
            x_min = x - w / 2
            x_max = x + w / 2
            y_min = y - h / 2
            y_max = y + h / 2
            
            if x_min < 0 or x_max > canvas_W or y_min < 0 or y_max > canvas_H:
                base_x, base_y = _clamp_center(x, y, w, h)
                repair_hash = _build_hash(exclude_idx=i)

                repaired = False
                seen = set()
                for cand_x, cand_y in _candidate_positions(base_x, base_y, w, h):
                    key = (round(cand_x, 6), round(cand_y, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    if not repair_hash.overlaps_any(cand_x, cand_y, w, h):
                        positions[i] = [cand_x, cand_y]
                        repaired = True
                        break

                if not repaired:
                    placed[i] = False

        return 0