"""V5 Fast Sparse Synthetic Algorithm.

Fast sparse synthetic netlist generation with:
- Bimodal instance generation (physical units in microns)
- Dynamic canvas with varying aspect ratios
- Degree-constrained sparse edge generation
- Rent's Rule terminal assignment
- CUDA acceleration
"""

import math
import random
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from torch_geometric.data import Data

from .v5_config import V5Config
from .v5_distributions import get_distribution
from .placement_v5 import V5Placer


class V5:
    """
    V5: Fast Sparse Synthetic Algorithm.
    
    Generates synthetic netlists with sparse connectivity (~6.2 average degree),
    bimodal instance sizes, and realistic placement.
    """
    
    def __init__(self, config: V5Config):
        """
        Initialize V5 algorithm.
        
        Args:
            config: V5Config object with all parameters
        """
        self.cfg = config
        # Deterministic per-call counter (used to derive per-graph seeds when cfg.seed is set).
        self._sample_count = 0
        
        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)  # Also seed Python random for consistency
    
    def _estimate_chip_size(self, N_reference: int, device: str = "cuda",
                            target_density_override: Optional[float] = None,
                            aspect_ratio_range_override: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Estimate chip size from N_reference before instance generation.
        Used to scale super-macro sizes proportionally.
        For packing_alg (stdcell-only): simpler formula, no macro shadow.

        Returns:
            (estimated_chip_width, estimated_chip_height) in microns
        """
        cfg = self.cfg
        is_packing_alg = getattr(cfg, "generation_pipeline", "v5") == "packing_alg"
        
        if is_packing_alg:
            # packing_alg: stdcell-dominant, no macros; ports are tiny and ignored for area estimate.
            n_stdcells_est = N_reference
            stdcell_area_est = n_stdcells_est * np.mean(cfg.stdcell_w_range) * np.mean(cfg.stdcell_h_range)
            A_total_est = stdcell_area_est
            effective_density_factor = 1.0  # No macro shadow
        else:
            # v5: ports + macros + stdcells
            n_ports_est = max(1, int(N_reference * cfg.port_fraction))
            n_macros_est = max(1, int(N_reference * cfg.macro_fraction))
            n_stdcells_est = N_reference - n_ports_est - n_macros_est
            port_area_est = n_ports_est * 1.0 * 1.0
            macro_area_est = n_macros_est * np.mean(cfg.macro_w_range) * np.mean(cfg.macro_h_range)
            stdcell_area_est = n_stdcells_est * np.mean(cfg.stdcell_w_range) * np.mean(cfg.stdcell_h_range)
            A_total_est = port_area_est + macro_area_est + stdcell_area_est
            macro_area_fraction_est = macro_area_est / A_total_est if A_total_est > 0 else 0.2
            macro_shadow_factor = getattr(cfg, 'macro_shadow_factor', 0.35)
            shadow_waste_fraction = macro_area_fraction_est * macro_shadow_factor
            effective_density_factor = 1.0 / (1.0 + shadow_waste_fraction)
        
        placement_efficiency = getattr(cfg, 'placement_efficiency', 0.92)
        grid_size_bias = getattr(cfg, 'grid_size_bias_factor', 0.70)
        spacing_efficiency = getattr(cfg, 'placement_spacing_efficiency', 0.99)
        if is_packing_alg:
            # packing_alg: higher efficiency (stdcells place well, no macro failures)
            placement_efficiency = getattr(cfg, 'stdcell_placement_efficiency', 0.995)
            grid_size_bias = getattr(cfg, 'grid_size_bias_factor', 0.97)
        A_expected_placed = A_total_est * placement_efficiency * grid_size_bias * spacing_efficiency
        _eff_density = target_density_override if target_density_override is not None else cfg.target_density
        A_chip_est = A_expected_placed / (_eff_density * effective_density_factor)

        # Use mean aspect ratio for estimation
        _ar_range = aspect_ratio_range_override if aspect_ratio_range_override is not None else cfg.canvas_aspect_ratio_range
        aspect_ratio_mean = (_ar_range[0] + _ar_range[1]) / 2.0
        chip_width_est = (A_chip_est * aspect_ratio_mean) ** 0.5
        chip_height_est = (A_chip_est / aspect_ratio_mean) ** 0.5
        
        # Floor so small-N graphs keep a minimum physical scale while preserving aspect ratio.
        min_dim = getattr(cfg, 'min_chip_dim', None)
        if min_dim is not None and min_dim > 0:
            scale = max(1.0, min_dim / chip_width_est, min_dim / chip_height_est)
            chip_width_est *= scale
            chip_height_est *= scale
        
        return chip_width_est, chip_height_est

    def _calculate_packing_alg_port_count(
        self,
        chip_width: float,
        chip_height: float,
        target_n: int,
        available_instances: int,
    ) -> int:
        """Compute packing_alg port count from the realized canvas size."""
        cfg = self.cfg
        perimeter = max(float(chip_width + chip_height), 0.0)
        k = float(getattr(cfg, "packing_alg_port_perimeter_scale", 0.35))
        min_ports = max(4, int(getattr(cfg, "packing_alg_min_ports", 4)))
        max_ports_cfg = getattr(cfg, "packing_alg_max_ports", None)

        n_ports = int(round(k * perimeter))
        n_ports = max(min_ports, n_ports)
        if max_ports_cfg is not None:
            n_ports = min(int(max_ports_cfg), n_ports)

        # Guard against pathological small pools while keeping the relationship tied to target N.
        n_ports = min(n_ports, max(0, int(target_n)))
        n_ports = min(n_ports, max(0, int(available_instances)))
        return n_ports

    def _assign_packing_alg_ports(
        self,
        w: torch.Tensor,
        h: torch.Tensor,
        area: torch.Tensor,
        chip_width: float,
        chip_height: float,
        target_n: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert the smallest packing_alg instances into fixed-size border ports."""
        is_port = torch.zeros_like(w, dtype=torch.float32, device=device)
        n_ports = self._calculate_packing_alg_port_count(chip_width, chip_height, target_n, len(w))
        if n_ports <= 0:
            return is_port, w, h, area

        port_indices = torch.argsort(area)[:n_ports]
        w = w.clone()
        h = h.clone()
        area = area.clone()
        is_port[port_indices] = 1.0
        w[port_indices] = 1.0
        h[port_indices] = 1.0
        area[port_indices] = 1.0
        return is_port, w, h, area
    
    def _generate_bimodal_instances(self, N_inst: int, N_reference: Optional[int] = None, device: str = "cuda",
                                    diversity_overrides: Optional[Dict] = None) -> dict:
        """
        Generate instances with bimodal distribution (microns).
        
        Args:
            N_inst: Number of instances to generate (may be buffered for soft max_instance)
            N_reference: Reference count for calculating fractions (port_fraction, macro_fraction)
                        If None, uses N_inst. Used for soft max_instance to maintain correct ratios.
            device: Device for tensors
            diversity_overrides: Optional per-graph overrides for stdcell size ranges/variance
            
        Returns:
            Dictionary with 'w', 'h', 'area', 'is_macro', 'is_port', 'cell_type'
        """
        cfg = self.cfg
        overrides = diversity_overrides or {}
        
        # Use reference count for fraction calculations (maintain correct ratios)
        if N_reference is None:
            N_reference = N_inst
        
        # Estimate chip size to scale super-macros proportionally.
        # packing_alg now assigns ports after the realized canvas is known, so this estimate
        # is only used for size scaling, not for port count.
        chip_width_est, chip_height_est = self._estimate_chip_size(N_reference, device=device)
        chip_size_est = max(chip_width_est, chip_height_est)  # Use larger dimension for scaling
        
        # Debug: Print chip size estimate
        if N_reference >= 1000:
            print(f"  Chip size estimate: [{chip_width_est:.1f}, {chip_height_est:.1f}] µm (max: {chip_size_est:.1f} µm)")
        
        is_port = torch.zeros(N_inst, dtype=torch.float32, device=device)
        n_ports = 0
        # packing_alg: std-cell + border-port generation (no macros)
        use_packing_alg = getattr(cfg, "generation_pipeline", "v5") == "packing_alg"
        if not use_packing_alg:
            # Sample ports based on reference count
            n_ports = max(1, int(N_reference * cfg.port_fraction))
            n_ports = min(n_ports, N_inst)  # Can't exceed actual instances
            port_indices = torch.randperm(N_inst, device=device)[:n_ports]
            is_port[port_indices] = 1.0
        
        # Sample macros (among non-ports) based on reference count
        non_port_mask = (is_port == 0)
        n_non_ports_actual = non_port_mask.sum().item()
        n_non_ports_reference = max(0, N_reference - n_ports)
        
        if use_packing_alg:
            n_super_macros = 0
            n_macros = 0
        else:
            # Generate super-macros: scale count with N. For small graphs, use fewer macros.
            super_macro_fraction = getattr(cfg, 'super_macro_fraction', 0.001)
            super_macro_min_count = getattr(cfg, 'super_macro_min_count', 10)  # Minimum super-macros
            super_macro_max_fraction = getattr(cfg, 'super_macro_max_fraction', 0.01)  # Max 1% of instances
            
            # Calculate super-macro count: fraction-based + minimum (only for larger graphs), capped at max fraction
            n_super_macros_from_fraction = max(0, int(n_non_ports_reference * super_macro_fraction))
            n_super_macros_from_min = (
                super_macro_min_count if N_reference >= 1000
                else (max(1, N_reference // 100) if N_reference >= 500 else 0)
            )
            n_super_macros_from_max = int(n_non_ports_reference * super_macro_max_fraction)

            # Use the maximum of fraction-based and minimum, but cap at max fraction
            n_super_macros = max(n_super_macros_from_fraction, n_super_macros_from_min)
            n_super_macros = min(n_super_macros, n_super_macros_from_max, n_non_ports_actual)

            # Regular macros: always generate at least 1 for N>=50 — small graphs should still have
            # macros, just smaller ones (sizes scale with chip_size_est via macro_size_scale).
            # Previously small_graph_macro_scale = N/500 caused int(100×0.03×0.2)=0 for N=100.
            super_macro_only_prob = getattr(cfg, 'super_macro_only_prob', 0.15)
            # Only use super-macro-only mode when there ARE actually super-macros to place.
            use_super_macro_only = (random.random() < super_macro_only_prob) and (n_super_macros > 0)

            if use_super_macro_only:
                n_macros = 0
            else:
                min_macros = 1 if N_reference >= 50 else 0
                n_macros_base = max(min_macros, int(n_non_ports_reference * cfg.macro_fraction))
                n_macros = min(n_macros_base, n_non_ports_actual - n_super_macros)  # Don't exceed available
        
        non_port_indices = torch.where(non_port_mask)[0]
        is_macro = torch.zeros(N_inst, dtype=torch.float32, device=device)
        is_super_macro = torch.zeros(N_inst, dtype=torch.float32, device=device)
        
        if n_super_macros > 0 and len(non_port_indices) > 0:
            super_macro_among_non_ports = torch.randperm(len(non_port_indices), device=device)[:n_super_macros]
            super_macro_indices = non_port_indices[super_macro_among_non_ports]
            is_macro[super_macro_indices] = 1.0
            is_super_macro[super_macro_indices] = 1.0
        
        # Regular macros (excluding super-macros)
        remaining_non_port_mask = non_port_mask & (is_super_macro == 0)
        remaining_non_port_indices = torch.where(remaining_non_port_mask)[0]
        if n_macros > 0 and len(remaining_non_port_indices) > 0:
            macro_among_remaining = torch.randperm(len(remaining_non_port_indices), device=device)[:n_macros]
            macro_indices = remaining_non_port_indices[macro_among_remaining]
            is_macro[macro_indices] = 1.0
        
        # Generate sizes
        w = torch.empty(N_inst, dtype=torch.float32, device=device)
        h = torch.empty(N_inst, dtype=torch.float32, device=device)
        
        # Standard cells - realistic, consistent sizes
        # Height: almost fixed (all stdcells in a library share the same height)
        # Width: quantized (integer multiples of site width), varies but realistic
        stdcell_mask = (is_port == 0) & (is_macro == 0)
        n_stdcells = stdcell_mask.sum().item()
        if n_stdcells > 0:
            # Use per-graph overrides for diversity, else config defaults
            stdcell_h_range = overrides.get('stdcell_h_range', cfg.stdcell_h_range)
            stdcell_w_range = overrides.get('stdcell_w_range', cfg.stdcell_w_range)
            stdcell_height_std = overrides.get('stdcell_height_std', getattr(cfg, 'stdcell_height_std', 0.02))
            fixed_row_height = overrides.get('stdcell_row_height', getattr(cfg, 'stdcell_row_height', None))
            stdcell_height_mean = (stdcell_h_range[0] + stdcell_h_range[1]) / 2.0  # Center of range
            stdcell_site_width = getattr(cfg, 'stdcell_site_width', 0.05)  # Site width for quantization
            
            # Generate heights: packing_alg can force a single physical row height.
            if fixed_row_height is not None:
                h_stdcells = torch.full((n_stdcells,), float(fixed_row_height), dtype=torch.float32, device=device)
            else:
                h_stdcells = torch.normal(stdcell_height_mean, stdcell_height_std, (n_stdcells,), device=device)
                h_stdcells = torch.clamp(h_stdcells, stdcell_h_range[0], stdcell_h_range[1])
            
            # Generate widths: quantized to site width multiples, allow wider range
            min_sites = max(1, int(math.ceil(stdcell_w_range[0] / stdcell_site_width)))
            max_sites = max(min_sites, int(stdcell_w_range[1] / stdcell_site_width))
            num_sites = torch.randint(min_sites, max_sites + 1, (n_stdcells,), device=device)
            w_stdcells = num_sites.float() * stdcell_site_width
            
            w[stdcell_mask] = w_stdcells
            h[stdcell_mask] = h_stdcells
        
        # Macros (super-macros and regular macros)
        n_macros_actual = is_macro.sum().item()
        if n_macros_actual > 0:
            macro_mask_bool = is_macro.bool()
            super_macro_mask_bool = is_super_macro.bool()
            regular_macro_mask_bool = macro_mask_bool & (~super_macro_mask_bool)
            
            # Super-macros: scale proportionally with chip size
            n_super = super_macro_mask_bool.sum().item()
            if n_super > 0:
                # Super-macro sizes scale proportionally with chip size
                # Reduced sizes: 5-12% of chip dimension (reduced from 20-30%)
                super_macro_size_min_frac = getattr(cfg, 'super_macro_size_min_frac', 0.05)  # 5% of chip size
                super_macro_size_max_frac = getattr(cfg, 'super_macro_size_max_frac', 0.12)  # 12% of chip size (reduced)
                
                # Calculate size ranges based on chip size
                super_macro_size_min = chip_size_est * super_macro_size_min_frac
                super_macro_size_max = chip_size_est * super_macro_size_max_frac
                
                # Ensure minimum absolute size, but cap by chip size for small graphs.
                # For small chips, super-macro must fit (never exceed ~30% of chip dimension).
                # Use very small absolute minimum to allow super-macros to scale purely with chip size
                super_macro_abs_min = min(
                    getattr(cfg, 'super_macro_abs_min_size', 5.0),  # Much smaller minimum (was 30.0)
                    chip_size_est * 0.30  # Cap: never exceed 30% of chip dimension
                )
                # Absolute max scales with chip size: small chips get fraction-only; large chips capped at config max
                super_macro_abs_max_frac = getattr(cfg, 'super_macro_abs_max_frac', 0.15)
                abs_max_cap = getattr(cfg, 'super_macro_abs_max_size', 500.0)
                super_macro_abs_max = min(abs_max_cap, chip_size_est * super_macro_abs_max_frac)
                super_macro_abs_max = max(super_macro_abs_max, super_macro_abs_min)  # ensure valid range on small chips
                
                super_macro_w_min = max(super_macro_size_min, super_macro_abs_min)
                super_macro_w_max = min(super_macro_size_max, super_macro_abs_max)
                if super_macro_w_min > super_macro_w_max:
                    super_macro_w_max = super_macro_w_min
                super_macro_h_min = max(super_macro_size_min, super_macro_abs_min)
                super_macro_h_max = min(super_macro_size_max, super_macro_abs_max)
                
                # Generate sizes
                w[super_macro_mask_bool] = torch.rand(n_super, device=device) * (
                    super_macro_w_max - super_macro_w_min
                ) + super_macro_w_min
                h[super_macro_mask_bool] = torch.rand(n_super, device=device) * (
                    super_macro_h_max - super_macro_h_min
                ) + super_macro_h_min
                
                # Debug: Print super-macro sizing info
                w_actual = w[super_macro_mask_bool].cpu().numpy()
                h_actual = h[super_macro_mask_bool].cpu().numpy()
                chip_dim_min = min(chip_width_est, chip_height_est)
                chip_dim_max = max(chip_width_est, chip_height_est)
                areas_actual = w_actual * h_actual
                canvas_area_est = chip_width_est * chip_height_est
                if len(w_actual) > 0:
                    print(f"  Generated {len(w_actual)} super-macros: "
                          f"size=[{w_actual.min():.1f}-{w_actual.max():.1f}, {h_actual.min():.1f}-{h_actual.max():.1f}] µm, "
                          f"fraction=[{w_actual.min()/chip_dim_max*100:.1f}%-{w_actual.max()/chip_dim_max*100:.1f}%], "
                          f"area_fraction={areas_actual.sum()/canvas_area_est*100:.1f}%")
            
            # Regular macros: scale size with chip (and thus with target node count)
            # For small graphs, macros should be proportionally smaller
            n_regular = regular_macro_mask_bool.sum().item()
            if n_regular > 0:
                ref_dim = getattr(cfg, 'macro_size_reference_chip_dim', 120.0)  # Increased from 80.0 for more aggressive scaling
                # Use square root scaling for more aggressive reduction on small graphs
                # This makes macros scale more proportionally with graph size
                macro_size_scale = min(1.0, (chip_size_est / ref_dim) ** 0.7)  # 0.7 power for more aggressive scaling
                macro_w_min = cfg.macro_w_range[0] * macro_size_scale
                macro_w_max = cfg.macro_w_range[1] * macro_size_scale
                macro_h_min = cfg.macro_h_range[0] * macro_size_scale
                macro_h_max = cfg.macro_h_range[1] * macro_size_scale
                w[regular_macro_mask_bool] = torch.rand(n_regular, device=device) * (
                    macro_w_max - macro_w_min
                ) + macro_w_min
                h[regular_macro_mask_bool] = torch.rand(n_regular, device=device) * (
                    macro_h_max - macro_h_min
                ) + macro_h_min
        
        # Ports (fixed size)
        n_ports_actual = is_port.sum().item()
        if n_ports_actual > 0:
            port_mask_bool = is_port.bool()
            w[port_mask_bool] = torch.ones(int(n_ports_actual), device=device) * 1.0
            h[port_mask_bool] = torch.ones(int(n_ports_actual), device=device) * 1.0
        
        # Calculate areas
        area = w * h
        
        # Cell types (simplified)
        cell_type = torch.zeros(N_inst, dtype=torch.int64, device=device)
        cell_type[is_port.bool()] = 0  # Ports
        cell_type[is_macro.bool()] = 1  # Macros
        stdcell_mask = (is_port == 0) & (is_macro == 0)
        if stdcell_mask.sum() > 0:
            # Assign standard cell types 2-30
            stdcell_types = torch.randint(2, 31, (stdcell_mask.sum().item(),), device=device)
            cell_type[stdcell_mask] = stdcell_types
        
        return {
            'w': w,
            'h': h,
            'area': area,
            'is_macro': is_macro,
            'is_port': is_port,
            'is_super_macro': is_super_macro,
            'cell_type': cell_type,
        }
    
    
    def _sample_neighbors_batch(
        self,
        batch_dists: torch.Tensor,       # (B, N_targets)
        batch_k: torch.Tensor,           # (B,)
        batch_src_indices: torch.Tensor, # (B,) indices into the original batch array
        target_indices: torch.Tensor,    # (N_targets,) global terminal indices
        num_terminals: torch.Tensor,
        is_source: torch.Tensor,
        is_macro: torch.Tensor,
        is_port: torch.Tensor,
        cluster_assignments: Optional[torch.Tensor],
        src_vs: torch.Tensor,            # (B,) global instance IDs
        distance_scale: float,
        T: int,
        chip_diagonal: Optional[float] = None,
        distance_decay_type: str = "hybrid",
        distance_power_alpha: float = 1.5,
        distance_crossover: float = 30.0,
        extra_valid_mask: Optional[torch.Tensor] = None,  # (B, N_targets) additional validity mask
        intra_cluster_boost: float = 2.5,
        edge_neighbor_selection: str = "deterministic_topk",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns TENSORS of connected edges (src_idx_in_batch, target_global_flat_idx).
        No python lists.
        
        Implements Rentian locality: heavy head near 0 (very local connections) + power-law tail.
        Uses mixture model: aggressive local exponential + global power-law component.
        """
        B, N_targets = batch_dists.shape
        device = batch_dists.device
        
        # --- 1. Probability Calculation with Rentian Locality ---
        # Normalize distances by chip diagonal if available (for proper Rentian distribution)
        # This ensures scale-invariant distribution matching real netlists
        if chip_diagonal is not None and chip_diagonal > 0:
            normalized_dists = batch_dists / chip_diagonal
            # Convert scale to normalized units (as fraction of diagonal)
            scale_normalized = distance_scale / chip_diagonal
            crossover_normalized = distance_crossover / chip_diagonal if distance_crossover > 0 else None
            dists_for_prob = normalized_dists
        else:
            dists_for_prob = batch_dists
            scale_normalized = distance_scale
            crossover_normalized = distance_crossover
        
        scale = torch.tensor(scale_normalized, device=device, dtype=batch_dists.dtype)
        
        # Implement Rentian locality: mixture of aggressive local + power-law tail
        if distance_decay_type == "exponential":
            # Pure exponential (original behavior)
            probs = torch.exp(-dists_for_prob / scale)
        elif distance_decay_type == "power_law":
            # Pure power-law: p(d) ∝ d^(-alpha)
            # Add small epsilon to avoid division by zero
            eps = torch.tensor(1e-6, device=device, dtype=batch_dists.dtype)
            probs = (dists_for_prob + eps) ** (-distance_power_alpha)
            # Normalize so max prob is reasonable
            probs = probs / (probs.max() + eps)
        elif distance_decay_type == "hybrid":
            # Hybrid: aggressive exponential for short distances, power-law for long
            # This creates heavy head near 0 + long tail (Rentian distribution)
            if crossover_normalized is not None:
                crossover_dist = torch.tensor(crossover_normalized, device=device, dtype=batch_dists.dtype)
            else:
                crossover_dist = torch.tensor(0.05, device=device, dtype=batch_dists.dtype)  # Default: 5% of diagonal
            
            # Local component: very aggressive exponential for short distances
            # Use smaller scale for local connections to create heavy head near 0
            # Scale is already normalized, so use a small fraction (0.02-0.05 of diagonal)
            local_scale = torch.tensor(0.03, device=device, dtype=batch_dists.dtype)  # 3% of diagonal
            local_probs = torch.exp(-dists_for_prob / local_scale)
            
            # Global component: power-law tail for long distances
            eps = torch.tensor(1e-6, device=device, dtype=batch_dists.dtype)
            # Power-law component: p(d) ∝ (d + eps)^(-alpha)
            power_probs = (dists_for_prob + eps) ** (-distance_power_alpha)
            # Normalize power-law component to match exponential at crossover
            power_at_crossover = (crossover_dist + eps) ** (-distance_power_alpha)
            exp_at_crossover = torch.exp(-crossover_dist / local_scale)
            power_norm = exp_at_crossover / (power_at_crossover + eps)
            power_probs = power_probs * power_norm
            
            # Mix: use exponential for short distances, power-law for long
            # Smooth transition at crossover
            is_local = dists_for_prob < crossover_dist
            probs = torch.where(
                is_local,
                local_probs,  # Aggressive exponential for local (heavy head)
                power_probs   # Power-law for global (long tail)
            )
            
            # Add small global component weight to ensure long-range connections
            # This ensures we have a non-trivial tail (Rentian property)
            global_weight = 0.12  # 12% weight for global component
            probs = (1.0 - global_weight) * probs + global_weight * power_probs
        else:
            # Fallback to exponential
            probs = torch.exp(-dists_for_prob / scale)
        
        # --- 2. Masking (Vectorized) ---
        # Map targets to properties
        dst_vs = target_indices.div(T, rounding_mode='floor')
        dst_ts = target_indices % T
        
        # Sink Validity: must exist and not be a source
        dst_exists = dst_ts < num_terminals[dst_vs]
        dst_not_src = 1.0 - is_source[dst_vs, dst_ts].float()
        mask = dst_exists.float() * dst_not_src
        
        # Self-loop mask
        src_vs_expanded = src_vs.unsqueeze(1)
        dst_vs_expanded = dst_vs.unsqueeze(0)
        mask = mask * (src_vs_expanded != dst_vs_expanded).float()

        # Optional additional mask (e.g. stdcell hard radius constraint)
        if extra_valid_mask is not None:
            mask = mask * extra_valid_mask.float()
        
        probs *= mask
        
        # --- 3. Biasing (Simplified Matrix Ops) ---
        # Pre-fetch source properties
        src_macro = is_macro[src_vs].bool().unsqueeze(1)
        src_port = is_port[src_vs].bool().unsqueeze(1)
        src_std = (~src_macro) & (~src_port)
        
        # Pre-fetch dest properties
        dst_macro = is_macro[dst_vs].bool().unsqueeze(0)
        dst_std = (~dst_macro) & (~is_port[dst_vs].bool().unsqueeze(0))
        
        # Apply bias using where (faster than branching)
        bias = torch.ones_like(probs)
        bias = torch.where(src_macro & dst_std, torch.tensor(5.0, device=device), bias)
        bias = torch.where(src_macro & dst_macro, torch.tensor(0.1, device=device), bias)
        bias = torch.where(src_std & dst_std, torch.tensor(1.5, device=device), bias)
        bias = torch.where(src_std & dst_macro, torch.tensor(0.3, device=device), bias)
        
        probs *= bias

        # Cluster boost (per-graph strength)
        if cluster_assignments is not None:
            c_src = cluster_assignments[src_vs].unsqueeze(1)
            c_dst = cluster_assignments[dst_vs].unsqueeze(0)
            cluster_match = (c_src == c_dst) & (c_src >= 0)
            probs = torch.where(cluster_match, probs * intra_cluster_boost, probs)

        # --- 4. Neighbor selection ---
        # "deterministic_topk": stable, HPWL-friendly (same placement → same graph)
        # "gumbel_topk": stochastic (same placement → different graphs across calls)
        selection = edge_neighbor_selection
        if selection == "deterministic_topk":
            # Use log-probability as score. This is already HPWL-friendly due to distance decay in probs.
            scores = torch.log(probs + 1e-12)
            # Stable tie-break: prefer smaller target flat index when scores tie.
            # Keep magnitude tiny so it only affects exact ties.
            tie = (-target_indices.to(dtype=scores.dtype)) * 1e-12  # (N_targets,)
            scores = scores + tie.unsqueeze(0)
        else:
            # Stochastic: adds Gumbel noise and then takes top-k (sampling without replacement).
            gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
            scores = torch.log(probs + 1e-10) + gumbel
        
        max_k = int(batch_k.max().item())
        if max_k > N_targets:
            max_k = N_targets
        
        if max_k <= 0:
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)
        
        _, top_indices = torch.topk(scores, max_k, dim=1)
        
        # Filter based on actual k needed per source
        k_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_k.unsqueeze(1)
        
        # Also ensure prob > 0 (valid edge)
        valid_prob_mask = torch.gather(probs, 1, top_indices) > 1e-10
        final_mask = k_mask & valid_prob_mask
        
        # Extract indices
        valid_rows, valid_cols = torch.nonzero(final_mask, as_tuple=True)
        
        # Return indices:
        # valid_rows: index in the batch (0 to B-1)
        # top_indices[...]: index in the target_indices array
        selected_target_indices = top_indices[valid_rows, valid_cols]
        
        return valid_rows, selected_target_indices
    
    def _sample_edges_with_batched_distances(
        self, terminal_positions: torch.Tensor,
        source_terminal_list: list,
        num_terminals: torch.Tensor, is_source: torch.Tensor,
        distance_scale: float, device: torch.device,
        is_macro: torch.Tensor, is_port: torch.Tensor,
        cluster_assignments: torch.Tensor, all_edges: list,
        instance_positions: torch.Tensor,
        is_stdcell: torch.Tensor,
        stdcell_hard_radius: Optional[float] = None,
        stdcell_hard_radius_norm: int = 1,
        spatial_cell_size: Optional[float] = None,
        chip_width: float = 0,
        chip_height: float = 0,
        distance_decay_type: Optional[str] = None,
        distance_power_alpha: Optional[float] = None,
        distance_crossover: Optional[float] = None,
        edge_neighbor_selection: Optional[str] = None,
        intra_cluster_boost: Optional[float] = None,
    ):
        """
        Optimized Batch Sampler with On-The-Fly Grid Gathering.
        Fail-fast: no device checks, no fallbacks, tensor-only operations.
        """
        if not source_terminal_list:
            return

        T = terminal_positions.shape[1]
        
        # 1. Convert Source List to Tensors (Much faster slicing later)
        src_list_tensor = torch.tensor(source_terminal_list, dtype=torch.long, device=device)
        all_src_vs = src_list_tensor[:, 0]
        all_src_ts = src_list_tensor[:, 1]
        all_ks = src_list_tensor[:, 2]
        
        total_sources = len(source_terminal_list)
        batch_size = getattr(self.cfg, 'batch_size', 1000)  # Use config value, default 1000
        
        # 2. Build Spatial Grid Once
        # Use smaller cell size for more local connections (Rentian locality)
        # Smaller cells = more local candidates, better for heavy head distribution
        if spatial_cell_size is None:
            # Default to ~10-15µm for more local connections (was 25µm)
            # This ensures we capture very local std-cell connections
            spatial_cell_size = getattr(self.cfg, 'spatial_candidate_cell_size', 12.0)
            
        perm, cell_starts, cell_counts, grid_W, grid_H, inst_cell_ids = \
            self._build_spatial_grid_vectorized(instance_positions, spatial_cell_size, chip_width, chip_height)

        # Pre-compute neighbor offsets (5x5 for Rentian distribution: local + tail)
        # Expanded from 3x3 to capture more candidates for power-law tail
        # Still heavily weighted toward center (local) via probability decay
        neighbor_radius = getattr(self.cfg, 'spatial_neighbor_radius', 2)  # 2 = 5x5 grid
        offsets = torch.arange(-neighbor_radius, neighbor_radius + 1, device=device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        neighbor_offsets = torch.stack((dx.flatten(), dy.flatten()), dim=1)  # (25, 2) for radius=2

        # Output buffers
        out_src_vs = []
        out_src_ts = []
        out_dst_vs = []
        out_dst_ts = []

        # 3. Batch Loop
        for start_idx in range(0, total_sources, batch_size):
            end_idx = min(start_idx + batch_size, total_sources)
            
            # Slice tensors
            batch_src_vs = all_src_vs[start_idx:end_idx]
            batch_src_ts = all_src_ts[start_idx:end_idx]
            batch_k = all_ks[start_idx:end_idx]
            B = len(batch_src_vs)
            
            # A. Identify relevant grid cells for this batch
            batch_cells = inst_cell_ids[batch_src_vs]  # (B,)
            b_cx = batch_cells % grid_W
            b_cy = batch_cells // grid_W
            
            # Broadcast to find neighbors: (B, 1, 2) + (1, 9, 2) -> (B, 9, 2)
            b_coords = torch.stack((b_cx, b_cy), dim=1).unsqueeze(1)
            n_coords = b_coords + neighbor_offsets.unsqueeze(0)
            
            # Filter valid
            nx = n_coords[:, :, 0]
            ny = n_coords[:, :, 1]
            valid = (nx >= 0) & (nx < grid_W) & (ny >= 0) & (ny < grid_H)
            
            # Get unique neighbor cells involved in this batch
            valid_nx = nx[valid]
            valid_ny = ny[valid]
            unique_n_ids = torch.unique(valid_ny * grid_W + valid_nx)
            
            # B. Gather Instances from these cells
            relevant_starts = cell_starts[unique_n_ids]
            relevant_counts = cell_counts[unique_n_ids]
            
            # Filter empty cells
            active_mask = (relevant_starts >= 0) & (relevant_counts > 0)
            relevant_starts = relevant_starts[active_mask]
            relevant_counts = relevant_counts[active_mask]
            
            # Construct target list (instance indices)
            target_instance_pool = []
            if len(relevant_starts) > 0:
                # Loop over cells (N < 500) to collect indices - negligible overhead
                for s, c in zip(relevant_starts.tolist(), relevant_counts.tolist()):
                    target_instance_pool.append(perm[s : s+c])
                target_instances = torch.cat(target_instance_pool)
            else:
                continue

            # C. Expand Targets to Terminals
            t_range = torch.arange(T, device=device)
            ti_grid = target_instances.unsqueeze(1).repeat(1, T).flatten()
            tt_grid = t_range.unsqueeze(0).repeat(len(target_instances), 1).flatten()
            
            target_indices_flat = ti_grid * T + tt_grid
            
            # D. Compute Distances & Sample
            batch_pos = terminal_positions[batch_src_vs, batch_src_ts]  # (B, 2)
            
            # Filter targets by num_terminals immediately to reduce matrix size
            t_counts = num_terminals[ti_grid]
            valid_t = tt_grid < t_counts
            
            final_targets = target_indices_flat[valid_t]
            final_target_pos = terminal_positions[ti_grid[valid_t], tt_grid[valid_t]]
            
            if len(final_targets) == 0:
                continue

            # Optional: stdcell hard-radius cutoff (HPWL-friendly).
            # This enforces: stdcell -> stdcell connections only within a given radius,
            # while leaving macro/port connectivity unconstrained.
            extra_valid_mask = None
            if stdcell_hard_radius is not None and stdcell_hard_radius > 0:
                # Source instance centers (B, 2)
                src_centers = instance_positions[batch_src_vs]  # (B, 2)
                # Target instance centers aligned with final_targets pool (N_pool, 2)
                tgt_vs = ti_grid[valid_t]
                tgt_centers = instance_positions[tgt_vs]  # (N_pool, 2)

                # Compute center distances (B, N_pool) in chosen norm
                if stdcell_hard_radius_norm == 2:
                    # L2
                    diffs = src_centers[:, None, :] - tgt_centers[None, :, :]
                    center_d = torch.sqrt(torch.clamp((diffs * diffs).sum(dim=-1), min=0.0))
                else:
                    # L1 (default)
                    center_d = (src_centers[:, None, :] - tgt_centers[None, :, :]).abs().sum(dim=-1)

                src_is_std = is_stdcell[batch_src_vs].bool().unsqueeze(1)  # (B,1)
                tgt_is_std = is_stdcell[tgt_vs].bool().unsqueeze(0)        # (1,N_pool)
                std_pair = src_is_std & tgt_is_std

                within = center_d <= float(stdcell_hard_radius)
                # Allow all non-stdcell pairs; restrict stdcell-stdcell pairs to within radius
                extra_valid_mask = (~std_pair) | within

            # CDIST (B, Pool_Size)
            dists = torch.cdist(batch_pos, final_target_pos, p=1)
            
            # Calculate chip diagonal for normalization
            chip_diagonal = math.sqrt(chip_width * chip_width + chip_height * chip_height) if chip_width > 0 and chip_height > 0 else None

            # Resolve effective decay params (per-graph overrides take priority over cfg)
            _decay_type = distance_decay_type if distance_decay_type is not None else getattr(self.cfg, 'distance_decay_type', 'hybrid')
            _power_alpha = distance_power_alpha if distance_power_alpha is not None else getattr(self.cfg, 'distance_power_alpha', 1.5)
            _crossover = distance_crossover if distance_crossover is not None else getattr(self.cfg, 'distance_crossover', 30.0)
            _edge_sel = edge_neighbor_selection if edge_neighbor_selection is not None else getattr(self.cfg, 'edge_neighbor_selection', 'deterministic_topk')
            _cluster_boost = intra_cluster_boost if intra_cluster_boost is not None else getattr(self.cfg, 'intra_cluster_boost', 2.5)

            # Sample
            row_ids, selected_target_idx = self._sample_neighbors_batch(
                dists, batch_k,
                torch.arange(B, device=device),  # batch indices (0 to B-1)
                final_targets,  # The pool of global indices
                num_terminals, is_source, is_macro, is_port, cluster_assignments,
                batch_src_vs,  # Actual global src IDs
                distance_scale, T,
                chip_diagonal=chip_diagonal,
                distance_decay_type=_decay_type,
                distance_power_alpha=_power_alpha,
                distance_crossover=_crossover,
                extra_valid_mask=extra_valid_mask,
                intra_cluster_boost=_cluster_boost,
                edge_neighbor_selection=_edge_sel,
            )
            
            # E. Store Result (Tensors)
            # row_ids are indices into the batch (0 to B-1)
            out_src_vs.append(batch_src_vs[row_ids])
            out_src_ts.append(batch_src_ts[row_ids])
            
            # selected_target_idx are indices into final_targets array
            # Decode target flat indices to (dst_v, dst_t)
            selected_target_flat = final_targets[selected_target_idx]
            out_dst_vs.append(selected_target_flat // T)
            out_dst_ts.append(selected_target_flat % T)

        # 4. Final Cat
        if len(out_src_vs) > 0:
            final_src_v = torch.cat(out_src_vs)
            final_src_t = torch.cat(out_src_ts)
            final_dst_v = torch.cat(out_dst_vs)
            final_dst_t = torch.cat(out_dst_ts)
            
            # Stack into (E, 4) tensor directly
            edges_tensor = torch.stack([final_src_v, final_src_t, final_dst_v, final_dst_t], dim=1)
            
            # Convert to list at the very end ONLY for compatibility
            all_edges.extend(edges_tensor.tolist())
    
    def _calculate_canvas_size(self, instance_areas: torch.Tensor, is_port: torch.Tensor,
                              is_macro: Optional[torch.Tensor] = None, device: str = "cuda",
                              target_density_override: Optional[float] = None,
                              aspect_ratio_range_override: Optional[Tuple[float, float]] = None,
                              diversity_overrides: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Calculate dynamic canvas size based on instance areas and target density.
        Only counts non-port instances (ports don't contribute to density).
        Accounts for expected placement failures to achieve target density.

        Args:
            instance_areas: (N,) tensor of instance areas
            is_port: (N,) tensor indicating ports
            is_macro: (N,) tensor indicating macros (optional, for better failure estimation)
            device: Device
            target_density_override: Per-graph density target (overrides cfg.target_density)
            aspect_ratio_range_override: Per-graph (lo, hi) for aspect ratio sampling
            
        Returns:
            (chip_width, chip_height) in microns
        """
        cfg = self.cfg
        is_packing_alg = getattr(cfg, "generation_pipeline", "v5") == "packing_alg"
        
        # Only count non-port instances for canvas sizing (std-cells + macros)
        non_port_mask = (is_port == 0)
        A_total = instance_areas[non_port_mask].sum().item()  # Total instance area (std-cells + macros) in µm²
        
        if is_packing_alg:
            # packing_alg: stdcell-only, no macros. Simpler strategy.
            placement_efficiency = (diversity_overrides.get('placement_efficiency') 
                                   if diversity_overrides and 'placement_efficiency' in diversity_overrides
                                   else getattr(cfg, 'stdcell_placement_efficiency', 0.995))
            effective_density_factor = 1.0  # No macro shadow
        else:
            # v5: Account for expected placement failures (macros fail more often)
            placement_efficiency = (diversity_overrides.get('placement_efficiency') 
                                   if diversity_overrides and 'placement_efficiency' in diversity_overrides
                                   else getattr(cfg, 'placement_efficiency', 0.92))
        
        # If we have macro info (v5 only), estimate more accurately
        if not is_packing_alg and is_macro is not None:
            macro_mask = is_macro.bool() & non_port_mask
            stdcell_mask = (~is_macro.bool()) & non_port_mask
            
            macro_area = instance_areas[macro_mask].sum().item() if macro_mask.any() else 0.0
            stdcell_area = instance_areas[stdcell_mask].sum().item() if stdcell_mask.any() else 0.0
            
            # Macros have lower placement success rate (especially large ones)
            # Failed macros represent significant area even if count is small
            # Based on observed behavior: ~20% of macros fail to place, and they're often the largest ones
            macro_efficiency = getattr(cfg, 'macro_placement_efficiency', 0.70)  # ~70% of macro AREA places (large ones often fail)
            stdcell_efficiency = getattr(cfg, 'stdcell_placement_efficiency', 0.995)  # ~99.5% of stdcells place
            
            # Weighted average based on area
            if A_total > 0:
                macro_area_frac = macro_area / A_total
                stdcell_area_frac = stdcell_area / A_total
                placement_efficiency = macro_area_frac * macro_efficiency + stdcell_area_frac * stdcell_efficiency
        
        # Adjust total area by expected placement efficiency
        # We want: chip_area * target_density = A_placed
        # But we estimate: A_placed ≈ A_total * placement_efficiency
        # So: chip_area = (A_total * placement_efficiency) / target_density
        
        # CRITICAL FIX: Account for grid sampling bias
        # The placement algorithm samples grids from gaussian(mean, std*0.5), not from the original
        # uniform distribution. This systematically under-represents extreme sizes.
        # Grids are more concentrated around the mean, reducing average area.
        # 
        # Additionally, multiple compounding factors reduce actual placed area:
        # 1. Grid sampling: gaussian(mean, std/2) vs uniform distribution
        # 2. Grid margin: grids sized at width*(1+0.01)
        # 3. Spacing factor: instances sized at grid*0.99*0.99
        # 4. Macro shadow: areas around macros can't be used for stdcell grids
        # 
        # Combined effect: placed area is systematically smaller than calculated
        
        # Grid size bias: grids sampled from gaussian, reducing effective area
        # packing_alg: 0.97 (structured grids pack well); v5: 0.70 (accounts for macro keepout)
        default_grid_bias = 0.97 if is_packing_alg else 0.70
        grid_size_bias = (diversity_overrides.get('grid_size_bias_factor')
                         if diversity_overrides and 'grid_size_bias_factor' in diversity_overrides
                         else getattr(cfg, 'grid_size_bias_factor', default_grid_bias))
        
        spacing_efficiency = getattr(cfg, 'placement_spacing_efficiency', 0.99)
        
        if not is_packing_alg:
            # Macro shadow: space around macros where grids can't be placed (v5 only)
            macro_shadow_factor = getattr(cfg, 'macro_shadow_factor', 0.35)
            if is_macro is not None and macro_mask.any() and A_total > 0:
                macro_area_fraction = macro_area / A_total
                shadow_waste_fraction = macro_area_fraction * macro_shadow_factor
                effective_density_factor = 1.0 / (1.0 + shadow_waste_fraction)
            else:
                shadow_waste_fraction = 0.2 * macro_shadow_factor
                effective_density_factor = 1.0 / (1.0 + shadow_waste_fraction)
        
        # Apply all correction factors
        # A_placed_actual = A_total * placement_efficiency * grid_bias * spacing_eff
        A_expected_placed = A_total * placement_efficiency * grid_size_bias * spacing_efficiency

        # Per-graph effective density and aspect ratio range
        _eff_density = target_density_override if target_density_override is not None else cfg.target_density
        _ar_range = aspect_ratio_range_override if aspect_ratio_range_override is not None else cfg.canvas_aspect_ratio_range

        # Calculate canvas: account for shadow waste reducing effective density
        # Formula: A_chip = A_placed / (target_density * shadow_eff)
        A_chip = A_expected_placed / (_eff_density * effective_density_factor)

        # Sample aspect ratio (width/height) from the effective range
        aspect_ratio = torch.rand(1, device=device).item() * (
            _ar_range[1] - _ar_range[0]
        ) + _ar_range[0]
        
        # Calculate canvas dimensions from area and aspect ratio
        # If aspect_ratio = width/height, then:
        #   width * height = A_chip
        #   width = height * aspect_ratio
        #   => height^2 * aspect_ratio = A_chip
        #   => height = sqrt(A_chip / aspect_ratio)
        #   => width = sqrt(A_chip * aspect_ratio)
        chip_width = (A_chip * aspect_ratio) ** 0.5  # in µm
        chip_height = (A_chip / aspect_ratio) ** 0.5  # in µm
        
        # Floor so small-N graphs keep minimum physical scale while preserving aspect ratio.
        min_dim = getattr(cfg, 'min_chip_dim', None)
        if min_dim is not None and min_dim > 0:
            scale = max(1.0, min_dim / chip_width, min_dim / chip_height)
            chip_width *= scale
            chip_height *= scale
        
        # VALIDATION: chip_size must be positive and reasonable
        # This catches degenerate cases: all ports, zero area, etc.
        if chip_width <= 0 or chip_height <= 0 or not (torch.isfinite(torch.tensor([chip_width, chip_height])).all()):
            raise ValueError(
                f"Invalid chip_size computed in v5 algorithm: W={chip_width:.6f}, H={chip_height:.6f}. "
                f"This indicates degenerate data. "
                f"Details: A_total={A_total:.6f}, A_expected_placed={A_expected_placed:.6f}, "
                f"A_chip={A_chip:.6f}, aspect_ratio={aspect_ratio:.6f}, "
                f"num_instances={len(instance_areas)}, num_ports={is_port.sum().item() if isinstance(is_port, torch.Tensor) else sum(is_port)}. "
                f"Check: Are all instances ports? Is instance area zero?"
            )
        
        # Additional sanity check: chip_size should be at least 1 micron
        if chip_width < 1.0 or chip_height < 1.0:
            raise ValueError(
                f"Chip size too small: W={chip_width:.6f}µm, H={chip_height:.6f}µm. "
                f"Minimum chip size is 1.0µm. This indicates invalid data generation."
            )
        
        return chip_width, chip_height
    
    def _build_spatial_grid_vectorized(self, positions: torch.Tensor, cell_size: float, 
                                       canvas_W: float, canvas_H: float):
        """
        Builds spatial grid metadata for on-the-fly querying.
        
        Returns:
            perm: Permutation indices sorting instances by cell ID
            cell_starts: (grid_W * grid_H,) tensor with start index in perm for each cell (-1 if empty)
            cell_counts: (grid_W * grid_H,) tensor with count of instances per cell
            grid_W: Number of grid cells in width
            grid_H: Number of grid cells in height
            cell_ids: (N,) tensor of cell ID for each instance
        """
        device = positions.device
        
        # 1. Compute Cell IDs
        grid_W = int(math.ceil(canvas_W / cell_size))
        grid_H = int(math.ceil(canvas_H / cell_size))
        
        cx = (positions[:, 0] / cell_size).long().clamp(0, grid_W - 1)
        cy = (positions[:, 1] / cell_size).long().clamp(0, grid_H - 1)
        cell_ids = cy * grid_W + cx
        
        # 2. Sort instances by Cell ID
        sorted_cell_ids, perm = torch.sort(cell_ids)
        
        # 3. Build dense lookup table
        num_cells = grid_W * grid_H
        unique_cells, counts = torch.unique_consecutive(sorted_cell_ids, return_counts=True)
        
        cell_starts = torch.full((num_cells,), -1, dtype=torch.long, device=device)
        cell_counts = torch.zeros((num_cells,), dtype=torch.long, device=device)
        
        # Calculate starts based on cumulative sum
        first_indices = torch.searchsorted(sorted_cell_ids, unique_cells)
        cell_starts[unique_cells] = first_indices
        cell_counts[unique_cells] = counts
        
        return perm, cell_starts, cell_counts, grid_W, grid_H, cell_ids
    
    
    def _compute_stdcell_clusters(self, positions: torch.Tensor, stdcell_mask: torch.Tensor,
                                  cluster_radius: float, device: str) -> torch.Tensor:
        """Compute cluster assignments for std-cells using spatial clustering.
        
        Returns:
            cluster_assignments: (V,) tensor with cluster ID for each instance (-1 for non-stdcells)
        """
        V = positions.shape[0]
        cluster_assignments = torch.full((V,), -1, dtype=torch.long, device=device)
        
        stdcell_indices = torch.where(stdcell_mask)[0]
        if len(stdcell_indices) == 0:
            return cluster_assignments
        
        stdcell_positions = positions[stdcell_indices]
        n_stdcells = len(stdcell_indices)
        
        # Simple spatial clustering: assign each std-cell to nearest cluster center
        # Generate cluster centers based on density
        # Estimate number of clusters: roughly one per cluster_radius^2 area
        pos_min = positions[stdcell_mask].min(dim=0)[0]
        pos_max = positions[stdcell_mask].max(dim=0)[0]
        canvas_area = (pos_max[0] - pos_min[0]) * (pos_max[1] - pos_min[1])
        n_clusters = max(1, int(canvas_area / (cluster_radius ** 2)))
        n_clusters = min(n_clusters, n_stdcells // 10)  # At least 10 std-cells per cluster
        
        # Initialize cluster centers randomly from std-cell positions
        cluster_centers_idx = torch.randperm(n_stdcells, device=device)[:n_clusters]
        cluster_centers = stdcell_positions[cluster_centers_idx]
        
        # Assign each std-cell to nearest cluster (vectorized)
        # Compute all distances at once: (n_stdcells, n_clusters)
        stdcell_pos_expanded = stdcell_positions.unsqueeze(1)  # (n_stdcells, 1, 2)
        cluster_centers_expanded = cluster_centers.unsqueeze(0)  # (1, n_clusters, 2)
        distances_all = torch.norm(stdcell_pos_expanded - cluster_centers_expanded, p=2, dim=2)  # (n_stdcells, n_clusters)
        nearest_clusters = torch.argmin(distances_all, dim=1)  # (n_stdcells,)
        
        # Assign cluster IDs
        for i, idx in enumerate(stdcell_indices):
            cluster_assignments[idx] = nearest_clusters[i].item()
        
        return cluster_assignments
    
    def get_terminal_offsets(self, x_sizes: torch.Tensor, y_sizes: torch.Tensor, 
                            max_num_terminals: int, reference: str = "center",
                            is_macro: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate terminal offsets.
        
        For macros: pins are placed on perimeter (edges)
        For std-cells: pins are placed randomly on boundary
        """
        device = x_sizes.device
        V = x_sizes.shape[0]
        
        # Initialize offsets
        boundary_offset = torch.zeros((V, max_num_terminals, 2), device=device)
        
        if is_macro is not None:
            macro_mask = is_macro.bool()
            stdcell_mask = ~macro_mask
        else:
            macro_mask = torch.zeros(V, dtype=torch.bool, device=device)
            stdcell_mask = torch.ones(V, dtype=torch.bool, device=device)
        
        # For macros: place pins on perimeter (edges)
        if macro_mask.any():
            n_macros = macro_mask.sum().item()
            macro_x = x_sizes[macro_mask]
            macro_y = y_sizes[macro_mask]
            
            for t in range(max_num_terminals):
                # Choose which edge for each macro (left, right, top, bottom)
                edge_choice = torch.randint(0, 4, (n_macros,), device=device)
                
                # Left edge: x = -w/2, y = random
                left_mask = edge_choice == 0
                if left_mask.any():
                    boundary_offset[macro_mask, t, 0][left_mask] = -macro_x[left_mask] / 2
                    boundary_offset[macro_mask, t, 1][left_mask] = (torch.rand(left_mask.sum().item(), device=device) - 0.5) * macro_y[left_mask]
                
                # Right edge: x = w/2, y = random
                right_mask = edge_choice == 1
                if right_mask.any():
                    boundary_offset[macro_mask, t, 0][right_mask] = macro_x[right_mask] / 2
                    boundary_offset[macro_mask, t, 1][right_mask] = (torch.rand(right_mask.sum().item(), device=device) - 0.5) * macro_y[right_mask]
                
                # Bottom edge: x = random, y = -h/2
                bottom_mask = edge_choice == 2
                if bottom_mask.any():
                    boundary_offset[macro_mask, t, 0][bottom_mask] = (torch.rand(bottom_mask.sum().item(), device=device) - 0.5) * macro_x[bottom_mask]
                    boundary_offset[macro_mask, t, 1][bottom_mask] = -macro_y[bottom_mask] / 2
                
                # Top edge: x = random, y = h/2
                top_mask = edge_choice == 3
                if top_mask.any():
                    boundary_offset[macro_mask, t, 0][top_mask] = (torch.rand(top_mask.sum().item(), device=device) - 0.5) * macro_x[top_mask]
                    boundary_offset[macro_mask, t, 1][top_mask] = macro_y[top_mask] / 2
        
        # For std-cells: use original random boundary placement
        if stdcell_mask.any():
            n_stdcells = stdcell_mask.sum().item()
            stdcell_x = x_sizes[stdcell_mask]
            stdcell_y = y_sizes[stdcell_mask]
            half_perim = (stdcell_x + stdcell_y)
            
            for t in range(max_num_terminals):
                terminal_locations = torch.rand(n_stdcells, device=device) * half_perim
                terminal_flip = (torch.rand(n_stdcells, device=device) < 0.5).float() * 2 - 1
                
                boundary_offset_x = torch.clamp(terminal_locations, torch.zeros_like(stdcell_x), stdcell_x) - (stdcell_x / 2)
                boundary_offset_y = torch.clamp(terminal_locations - stdcell_x, torch.zeros_like(stdcell_y), stdcell_y) - (stdcell_y / 2)
                
                boundary_offset_x = terminal_flip * boundary_offset_x
                boundary_offset_y = terminal_flip * boundary_offset_y
                
                boundary_offset[stdcell_mask, t, 0] = boundary_offset_x
                boundary_offset[stdcell_mask, t, 1] = boundary_offset_y
        
        if reference == "bottom_left":
            boundary_offset[:, :, 0] += x_sizes.unsqueeze(1) / 2
            boundary_offset[:, :, 1] += y_sizes.unsqueeze(1) / 2
        
        return boundary_offset
    
    def connect_isolated_instances_sparse(
        self, edge_tensor: torch.Tensor, terminal_positions: torch.Tensor,
        out_degree: torch.Tensor, num_instances: int, max_num_terminals: int,
        instance_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Connect isolated instances using kNN optimization."""
        if edge_tensor.shape[0] == 0:
            return edge_tensor
        
        device = terminal_positions.device
        V, T, _ = terminal_positions.shape
        
        # Use long dtype to match edge_tensor dtype
        in_degree = torch.zeros(num_instances, dtype=torch.long, device='cpu')
        dst_instances = edge_tensor[:, 2].cpu().long()
        in_degree.index_add_(0, dst_instances, torch.ones_like(dst_instances, dtype=torch.long))
        
        degree = out_degree.sum(dim=-1).cpu().long() + in_degree
        isolated_mask = (degree == 0)
        isolated_instances = torch.nonzero(isolated_mask, as_tuple=False).squeeze(-1).tolist()
        
        if not isolated_instances:
            return edge_tensor
        
        if instance_positions is None:
            instance_positions = terminal_positions[:, 0, :]
        
        if instance_positions.device != device:
            instance_positions = instance_positions.to(device)
        
        # OPTIMIZATION: Compute kNN once for all isolated instances instead of O(V²) per isolated
        # Use torch.cdist to compute all pairwise distances at once: (V, V)
        all_instance_distances = torch.cdist(instance_positions, instance_positions, p=1)  # (V, V)
        
        # Set diagonal to large value to exclude self
        all_instance_distances.fill_diagonal_(float('inf'))
        
        new_edges = []
        for i in isolated_instances:
            # Get nearest neighbor from precomputed distances
            target_instance = torch.argmin(all_instance_distances[i, :]).item()
            
            # Find best terminal on target instance
            pos_i = instance_positions[i, :]
            target_positions = terminal_positions[target_instance, :, :]
            terminal_distances = torch.norm(target_positions - pos_i, p=1, dim=-1)
            outdegree_target = out_degree[target_instance, :].to(device)
            terminal_has_outdegree = (outdegree_target > 0).float()
            terminal_distances = terminal_distances - (terminal_has_outdegree * terminal_distances * 0.3)
            terminal_idx = torch.argmin(terminal_distances).item()
            
            new_edges.append([target_instance, terminal_idx, i, 0])
        
        if new_edges:
            # Create new edges on the same device as edge_tensor
            new_edge_tensor = torch.tensor(new_edges, dtype=edge_tensor.dtype, device=edge_tensor.device)
            edge_tensor = torch.cat([edge_tensor, new_edge_tensor], dim=0)
        
        return edge_tensor

    def _first_source_terminal(self, v: int, num_terminals: torch.Tensor, is_source: torch.Tensor) -> Optional[int]:
        n_t = int(num_terminals[v].item())
        for t in range(n_t):
            if float(is_source[v, t].item()) > 0.5:
                return t
        return None

    def _first_sink_terminal(self, v: int, num_terminals: torch.Tensor, is_source: torch.Tensor) -> Optional[int]:
        n_t = int(num_terminals[v].item())
        for t in range(n_t):
            if float(is_source[v, t].item()) <= 0.5:
                return t
        return None

    def _first_any_terminal(self, v: int, num_terminals: torch.Tensor) -> Optional[int]:
        n_t = int(num_terminals[v].item())
        if n_t <= 0:
            return None
        return 0

    def _enforce_port_to_nonport_connectivity_sparse(
        self,
        edge_tensor: torch.Tensor,
        *,
        positions: torch.Tensor,
        is_port: torch.Tensor,
        num_terminals: torch.Tensor,
        is_source: torch.Tensor,
        num_instances: int,
    ) -> torch.Tensor:
        """
        Ensure every port is connected to at least one non-port instance.

        Connectivity is checked in an undirected sense over instance graph edges.
        For any violating port, add one nearest non-port edge while preserving
        source->sink orientation when possible.
        """
        if num_instances <= 1:
            return edge_tensor

        is_port_cpu = is_port.detach().cpu().bool()
        port_instances = torch.where(is_port_cpu)[0].tolist()
        if not port_instances:
            return edge_tensor

        non_port_instances = torch.where(~is_port_cpu)[0].tolist()
        if not non_port_instances:
            return edge_tensor

        neighbors = [set() for _ in range(num_instances)]
        if edge_tensor.shape[0] > 0:
            e_cpu = edge_tensor[:, [0, 2]].detach().cpu().long()
            for src, dst in e_cpu.tolist():
                if src == dst:
                    continue
                if 0 <= src < num_instances and 0 <= dst < num_instances:
                    neighbors[src].add(dst)
                    neighbors[dst].add(src)

        pos_cpu = positions.detach().cpu()
        new_edges = []

        for p in port_instances:
            has_nonport_neighbor = any((not bool(is_port_cpu[n])) for n in neighbors[p])
            if has_nonport_neighbor:
                continue

            port_src = self._first_source_terminal(p, num_terminals, is_source)
            port_sink = self._first_sink_terminal(p, num_terminals, is_source)
            port_any = self._first_any_terminal(p, num_terminals)
            if port_any is None:
                continue

            cand = torch.tensor(non_port_instances, dtype=torch.long)
            d = (pos_cpu[cand] - pos_cpu[p]).abs().sum(dim=-1)
            sorted_idx = torch.argsort(d)

            added = False
            for idx in sorted_idx.tolist():
                target = int(cand[idx].item())
                target_src = self._first_source_terminal(target, num_terminals, is_source)
                target_sink = self._first_sink_terminal(target, num_terminals, is_source)
                target_any = self._first_any_terminal(target, num_terminals)
                if target_any is None:
                    continue

                # Prefer strict source->sink semantics.
                if target_src is not None and port_sink is not None:
                    new_edges.append([target, target_src, p, port_sink])
                    neighbors[target].add(p)
                    neighbors[p].add(target)
                    added = True
                    break

                if port_src is not None and target_sink is not None:
                    new_edges.append([p, port_src, target, target_sink])
                    neighbors[target].add(p)
                    neighbors[p].add(target)
                    added = True
                    break

                # Hard fallback: still connect port<->non-port even when source/sink
                # terminals are unavailable due degenerate terminal direction labels.
                if port_src is not None:
                    new_edges.append([p, port_src, target, target_any])
                    neighbors[target].add(p)
                    neighbors[p].add(target)
                    added = True
                    break
                if target_src is not None:
                    new_edges.append([target, target_src, p, port_any])
                    neighbors[target].add(p)
                    neighbors[p].add(target)
                    added = True
                    break

            if added:
                continue

        if not new_edges:
            return edge_tensor

        new_edge_tensor = torch.tensor(new_edges, dtype=edge_tensor.dtype, device=edge_tensor.device)
        return torch.cat([edge_tensor, new_edge_tensor], dim=0)

    def _bridge_disconnected_components_sparse(
        self,
        edge_tensor: torch.Tensor,
        *,
        num_instances: int,
        positions: torch.Tensor,
        num_terminals: torch.Tensor,
        is_source: torch.Tensor,
        chip_width: float,
        chip_height: float,
    ) -> torch.Tensor:
        """
        Deterministically connect disconnected components with nearest bridges (HPWL-minimizing).

        This makes the sparse inst-inst graph more netlist-like (dominant giant component),
        improves learnability, and stabilizes Laplacian PE computations.
        """
        cfg = self.cfg
        if not getattr(cfg, "bridge_components", True):
            return edge_tensor
        max_bridges = int(getattr(cfg, "max_component_bridges", 16))
        if max_bridges <= 0:
            return edge_tensor
        if num_instances <= 1 or edge_tensor.shape[0] == 0:
            return edge_tensor

        # Use CPU for deterministic connectivity + spatial lookup.
        pos_cpu = positions.detach().cpu()
        src_v_cpu = edge_tensor[:, 0].detach().cpu().long()
        dst_v_cpu = edge_tensor[:, 2].detach().cpu().long()

        # ---------------------------
        # Union-Find (Disjoint Set)
        # ---------------------------
        parent = list(range(num_instances))
        rank = [0] * num_instances

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for a, b in zip(src_v_cpu.tolist(), dst_v_cpu.tolist()):
            a = int(a)
            b = int(b)
            if a != b:
                union(a, b)

        roots = [find(i) for i in range(num_instances)]
        comp_map = {}
        comp_id = [0] * num_instances
        comp_sizes = []
        for i, r in enumerate(roots):
            if r not in comp_map:
                comp_map[r] = len(comp_map)
                comp_sizes.append(0)
            cid = comp_map[r]
            comp_id[i] = cid
            comp_sizes[cid] += 1

        n_comp = len(comp_sizes)
        if n_comp <= 1:
            return edge_tensor

        # Largest component as main
        main_cid = int(max(range(n_comp), key=lambda c: (comp_sizes[c], -c)))
        other_cids = [c for c in range(n_comp) if c != main_cid]
        other_cids.sort(key=lambda c: (-comp_sizes[c], c))  # deterministic

        # Spatial grid for nearest neighbor search
        cell_size = float(getattr(cfg, "spatial_candidate_cell_size", 12.0))
        perm, cell_starts, cell_counts, grid_W, grid_H, cell_ids = self._build_spatial_grid_vectorized(
            pos_cpu, cell_size, float(chip_width), float(chip_height)
        )
        perm = perm.cpu()
        cell_starts = cell_starts.cpu()
        cell_counts = cell_counts.cpu()
        cell_ids = cell_ids.cpu()

        main_mask = torch.tensor([c == main_cid for c in comp_id], dtype=torch.bool)

        num_term_cpu = num_terminals.detach().cpu().long()
        is_source_cpu = is_source.detach().cpu()

        def pick_src_terminal(v: int) -> int:
            n_t = int(num_term_cpu[v].item())
            for t in range(n_t):
                if float(is_source_cpu[v, t].item()) > 0.5:
                    return t
            return 0

        def pick_sink_terminal(v: int) -> int:
            n_t = int(num_term_cpu[v].item())
            for t in range(n_t):
                if float(is_source_cpu[v, t].item()) <= 0.5:
                    return t
            return 0

        R = int(getattr(cfg, "bridge_search_radius", 4))
        R = max(0, R)

        new_edges = []
        bridges_added = 0
        for cid in other_cids:
            if bridges_added >= max_bridges:
                break

            comp_nodes = [i for i, c in enumerate(comp_id) if c == cid]
            if not comp_nodes:
                continue

            best = None  # (dist, u, v) lexicographic tie-break
            for u in comp_nodes:
                u_xy = pos_cpu[u]
                cell = int(cell_ids[u].item())
                cx = cell % grid_W
                cy = cell // grid_W

                # Expand neighborhood until candidates found
                for r in range(R + 1):
                    cand_vs = []
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            nx = cx + dx
                            ny = cy + dy
                            if nx < 0 or nx >= grid_W or ny < 0 or ny >= grid_H:
                                continue
                            nid = ny * grid_W + nx
                            s = int(cell_starts[nid].item())
                            if s < 0:
                                continue
                            ccount = int(cell_counts[nid].item())
                            if ccount <= 0:
                                continue
                            nodes = perm[s : s + ccount]
                            nodes_main = nodes[main_mask[nodes]]
                            if nodes_main.numel() > 0:
                                cand_vs.append(nodes_main)
                    if cand_vs:
                        cand = torch.cat(cand_vs).unique()
                        d = (pos_cpu[cand] - u_xy.unsqueeze(0)).abs().sum(dim=-1)
                        min_d, arg = torch.min(d, dim=0)
                        v = int(cand[int(arg.item())].item())
                        cand_tuple = (float(min_d.item()), int(u), int(v))
                        if best is None or cand_tuple < best:
                            best = cand_tuple
                        break

            if best is None:
                continue

            _, u_best, v_best = best
            su = pick_src_terminal(u_best)
            tv = pick_sink_terminal(v_best)
            # If u has no sources but v does, reverse direction to keep source->sink semantics.
            if float(is_source_cpu[u_best, su].item()) <= 0.5 and float(is_source_cpu[v_best, pick_src_terminal(v_best)].item()) > 0.5:
                su = pick_src_terminal(v_best)
                tv = pick_sink_terminal(u_best)
                u_best, v_best = v_best, u_best

            new_edges.append([u_best, su, v_best, tv])
            # Grow main set (so subsequent components can connect to an expanding giant component)
            for i, c in enumerate(comp_id):
                if c == cid:
                    main_mask[i] = True
            bridges_added += 1

        if not new_edges:
            return edge_tensor

        new_edge_tensor = torch.tensor(new_edges, dtype=edge_tensor.dtype, device=edge_tensor.device)
        return torch.cat([edge_tensor, new_edge_tensor], dim=0)
    
    def generate_edge_list_sparse(self, edge_tensor: torch.Tensor, terminal_offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sparse edge tensor to edge_index and edge_attr."""
        if edge_tensor.shape[0] == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 4))
        
        edge_index_forward = edge_tensor[:, [0, 2]]
        edge_index_reverse = edge_tensor[:, [2, 0]]
        
        edge_attr_source = terminal_offsets[edge_tensor[:, 0], edge_tensor[:, 1], :]
        edge_attr_sink = terminal_offsets[edge_tensor[:, 2], edge_tensor[:, 3], :]
        edge_attr_forward = torch.concat((edge_attr_source, edge_attr_sink), dim=-1)
        edge_attr_reverse = torch.concat((edge_attr_sink, edge_attr_source), dim=-1)
        
        edge_index = torch.concat((edge_index_forward, edge_index_reverse), dim=0).T
        edge_attr = torch.concat((edge_attr_forward, edge_attr_reverse), dim=0)
        
        return edge_index.clone(), edge_attr.clone()
    
    def _sample_diversity_params(self) -> Dict:
        """Sample per-graph diversity parameters.

        Every call returns a fresh dict of overrides that vary placement geometry
        and graph structure independently across graphs.  When
        enable_diversity_randomization is False the dict is empty (config defaults
        are used for every graph).
        """
        cfg = self.cfg
        if not getattr(cfg, 'enable_diversity_randomization', True):
            return {}

        overrides: Dict[str, Any] = {}

        # ── Std-cell size diversity ───────────────────────────────────────────
        # packing_alg keeps one physical row height to avoid N-dependent geometry drift.
        is_packing_alg = getattr(cfg, "generation_pipeline", "v5") == "packing_alg"
        if not is_packing_alg:
            h_min_lo, h_min_hi = getattr(cfg, 'stdcell_h_range_min', (0.15, 0.25))
            h_max_lo, h_max_hi = getattr(cfg, 'stdcell_h_range_max', (0.6, 1.2))
            w_min_lo, w_min_hi = getattr(cfg, 'stdcell_w_range_min', (0.5, 1.0))
            w_max_lo, w_max_hi = getattr(cfg, 'stdcell_w_range_max', (3.0, 8.0))
            h_low = random.uniform(h_min_lo, h_min_hi)
            h_high = random.uniform(h_max_lo, h_max_hi)
            overrides['stdcell_h_range'] = (min(h_low, h_high), max(h_low, h_high))
        else:
            fixed_row_height = getattr(cfg, 'stdcell_row_height', None)
            if fixed_row_height is not None:
                fixed_row_height = float(fixed_row_height)
                overrides['stdcell_row_height'] = fixed_row_height
                overrides['stdcell_h_range'] = (fixed_row_height, fixed_row_height)
                overrides['stdcell_height_std'] = 0.0
            w_min_lo, w_min_hi = getattr(cfg, 'stdcell_w_range_min', (0.3, 0.8))
            w_max_lo, w_max_hi = getattr(cfg, 'stdcell_w_range_max', (4.0, 12.0))

        w_low = random.uniform(w_min_lo, w_min_hi)
        w_high = random.uniform(w_max_lo, w_max_hi)

        # Avoid size-generalization leakage: packing_alg uses a fixed width support,
        # while v5 keeps the chip-relative cap for macro-rich designs.
        if is_packing_alg:
            width_cap = getattr(cfg, 'packing_alg_stdcell_width_cap', None)
            if width_cap is not None and width_cap > 0:
                w_high = min(w_high, float(width_cap))
            w_low = min(w_low, w_high)
        else:
            n_for_chip_est = cfg.max_instance
            chip_w_est, chip_h_est = self._estimate_chip_size(n_for_chip_est)
            chip_dim_est = (chip_w_est * chip_h_est) ** 0.5  # geometric-mean chip size
            max_w_abs = max(0.5, chip_dim_est * 0.05)        # at least 0.5 µm
            w_high = min(w_high, max_w_abs)
            w_low  = min(w_low,  w_high * 0.5)               # keep min ≤ max/2

        overrides['stdcell_w_range'] = (min(w_low, w_high), max(w_low, w_high))

        if not (is_packing_alg and 'stdcell_height_std' in overrides):
            std_lo, std_hi = getattr(cfg, 'stdcell_height_std_range', (0.01, 0.12))
            if is_packing_alg:
                std_lo, std_hi = (0.02, 0.20)
            overrides['stdcell_height_std'] = random.uniform(std_lo, std_hi)

        # ── Placement density / shape ─────────────────────────────────────────
        if is_packing_alg:
            # Keep density high but realistic. Near-1.0 targets are often unattainable
            # in small graphs and create large density mismatch.
            base_density = float(getattr(cfg, "target_density", 0.82))
            low = max(0.72, base_density - 0.06)
            high = min(0.90, base_density + 0.04)
            overrides['target_density'] = random.uniform(low, high)

            # Respect an explicitly fixed canvas aspect ratio (used by the packing_alg
            # dataset script for easier training). Otherwise keep some shape variety.
            cfg_ar_range = getattr(cfg, 'canvas_aspect_ratio_range', (0.4, 2.5))
            if abs(float(cfg_ar_range[0]) - float(cfg_ar_range[1])) < 1e-9:
                ar = float(cfg_ar_range[0])
            else:
                r = random.random()
                if r < 0.15:
                    ar = random.uniform(1.3, 2.0)  # wider range for landscape
                elif r < 0.30:
                    ar = random.uniform(0.50, 0.77)  # wider range for portrait
                else:
                    ar = random.uniform(0.80, 1.25)  # wider range for near-square
            overrides['canvas_aspect_ratio_range'] = (ar, ar)
            
            # Small variation in estimation factors for modest diversity without destabilizing density.
            overrides['placement_efficiency'] = random.uniform(0.92, 0.97)
            overrides['grid_size_bias_factor'] = random.uniform(0.94, 0.99)
        else:
            # Real designs span roughly 60 – 85 %; density is the sole placement stop condition.
            overrides['target_density'] = random.uniform(0.60, 0.85)

            # Broader range than old 4:6–6:4 default.
            # 12% wide, 12% tall, 76% near-square.
            r = random.random()
            if r < 0.12:
                ar = random.uniform(1.5, 3.0)    # wide  (landscape)
            elif r < 0.24:
                ar = random.uniform(0.33, 0.67)  # tall  (portrait)
            else:
                ar = random.uniform(0.70, 1.43)  # near-square
            overrides['canvas_aspect_ratio_range'] = (ar, ar)  # fix to exact value

        # ── Macro placement ───────────────────────────────────────────────────
        # Always stochastic per-graph (macro_deterministic=True was causing fixed patterns)
        overrides['macro_deterministic'] = False
        overrides['macro_area_target_mode'] = 'random'

        area_lo, area_hi = getattr(cfg, 'macro_area_target_range', (0.12, 0.55))
        overrides['macro_area_target_min'] = area_lo
        overrides['macro_area_target_max'] = area_hi
        # Clamp: macros must leave at least 25% of density budget for stdcells.
        # Without this, macro_area_target=0.55 with target_density=0.40 → 0 stdcells placed.
        _density_for_macro = overrides.get('target_density', getattr(cfg, 'target_density', 0.6))
        macro_area_target = random.uniform(area_lo, min(area_hi, _density_for_macro * 0.75))
        overrides['macro_area_target'] = macro_area_target

        # Community mode: all macros packed into one half of the chip (older designs)
        community_prob = getattr(cfg, 'macro_community_mode_prob', 0.20)
        overrides['macro_community_mode'] = random.random() < community_prob

        # Macro placement strategy per-graph
        overrides['macro_placement_strategy'] = random.choices(
            ['periphery', 'islands', 'hybrid'],
            weights=[0.30, 0.25, 0.45],
        )[0]

        # ── Std-cell fill order: vary density gradient direction ──────────────
        overrides['stdcell_grid_shuffle'] = random.random() < 0.5

        # ── Port placement: packing_alg always uses all four borders to avoid new
        # size-dependent edge-activation patterns. Full v5 keeps its mixed strategy.
        if is_packing_alg:
            overrides['port_placement_strategy'] = 'border_balanced'
        elif random.random() < 0.60:
            overrides['port_placement_strategy'] = 'border_uniform'
        else:
            overrides['port_placement_strategy'] = 'border_clustered'

        # ── Degree distribution + edge locality ────────────────────────────────
        if is_packing_alg:
            # Keep connectivity stable for packing_alg learnability.
            mean_deg = float(getattr(cfg, 'packing_alg_target_degree_mean', 6.0))
            std_deg = float(getattr(cfg, 'packing_alg_target_degree_std', 1.0))
            overrides['target_degree_mean'] = mean_deg
            overrides['target_degree_std'] = std_deg
            # Tight locality range to prevent large per-graph density variance.
            overrides['edge_distance_scale'] = random.uniform(12.0, 18.0)
            overrides['distance_power_alpha'] = random.uniform(1.2, 1.8)
            overrides['distance_crossover'] = random.uniform(16.0, 28.0)
        else:
            mean_deg = random.uniform(3.0, 12.0)
            overrides['target_degree_mean'] = mean_deg
            overrides['target_degree_std'] = mean_deg * random.uniform(0.5, 1.5)

            macro_mean = mean_deg * random.uniform(3.0, 8.0)
            overrides['macro_degree_mean'] = macro_mean
            overrides['macro_degree_std'] = macro_mean * random.uniform(0.4, 1.2)
            overrides['macro_degree_min'] = max(5, int(mean_deg * 1.5))
            overrides['macro_degree_max'] = max(60, int(macro_mean * 6))

            # Low edge_distance_scale  → very local connections (HPWL-minimal)
            # High edge_distance_scale → connections span the chip (routing-heavy)
            overrides['edge_distance_scale'] = random.uniform(6.0, 50.0)
            overrides['distance_power_alpha'] = random.uniform(0.8, 2.5)
            overrides['distance_crossover'] = random.uniform(8.0, 60.0)

        # ── Edge neighbour selection: mix stochastic and deterministic ────────
        # gumbel_topk: same placement → different graphs across generation runs
        # deterministic_topk: same placement → identical graph (stable training)
        # ~40 % stochastic increases layout-graph diversity in the training set.
        overrides['edge_neighbor_selection'] = random.choices(
            ['deterministic_topk', 'gumbel_topk'],
            weights=[0.60, 0.40],
        )[0]

        # ── Clustering: locality strength varies by design ────────────────────
        if is_packing_alg:
            # Disable absolute-length clustering and hard cutoffs to avoid new
            # geometry-dependent graph patterns at larger chip sizes.
            overrides['enable_clustering'] = False
            overrides['intra_cluster_boost'] = 1.0
            overrides['stdcell_hard_radius'] = None
        else:
            overrides['enable_clustering'] = random.random() < 0.75
            overrides['intra_cluster_boost'] = random.uniform(1.0, 6.0)
            overrides['cluster_radius'] = random.uniform(20.0, 100.0)

        # ── Port connectivity pattern ─────────────────────────────────────────
        p_macro = random.uniform(0.10, 0.65)
        p_cluster = min(random.uniform(0.10, 0.55), max(0.05, 0.90 - p_macro))
        p_port = max(0.0, 1.0 - p_macro - p_cluster)
        overrides['port_to_macro_prob'] = p_macro
        overrides['port_to_cluster_prob'] = p_cluster
        overrides['port_to_port_prob'] = p_port

        return overrides
    
    def _make_placer_config(self, cfg: V5Config, overrides: Dict[str, Any]) -> Any:
        """Create config for placer with per-graph diversity overrides."""
        if not overrides:
            return cfg

        # All keys that placement_v5.py reads via getattr(self.cfg, ...)
        placer_keys = {
            # macro area
            'macro_area_target_min', 'macro_area_target_max', 'macro_area_target',
            'macro_area_target_mode',
            # macro layout behaviour
            'macro_deterministic', 'macro_island_layout', 'macro_island_grid_shuffle',
            'macro_placement_strategy', 'macro_community_mode',
            # stdcell fill
            'stdcell_grid_shuffle',
            'stdcell_row_height',
            # placement density (placer reads target_density for its stop condition)
            'target_density',
            # port placement
            'port_placement_strategy',
            # max_instance (for flexible node counts - stop when approximately max_instance instances placed)
            'max_instance',
        }

        # Build overlay: object that delegates to cfg but overrides specific attrs
        class _ConfigOverlay:
            def __init__(self, base, ov):
                self._base = base
                self._ov = {k: v for k, v in ov.items() if k in placer_keys}
            def __getattr__(self, name):
                if name in self._ov:
                    return self._ov[name]
                return getattr(self._base, name)

        return _ConfigOverlay(cfg, overrides)
    
    def sample(self) -> Tuple[torch.Tensor, Data]:
        """
        Generate a synthetic netlist using V5 algorithm.
        
        Returns:
            (positions, data) tuple where:
            - positions: (V, 2) tensor of instance positions (centers) in microns
            - data: PyG Data object with:
                - x: (V, 2) instance sizes [width, height] in microns
                - edge_index: (2, E) edge connectivity
                - edge_attr: (E, 4) pin offsets in microns [src_pin_x, src_pin_y, dst_pin_x, dst_pin_y]
                  (offset from instance center; from get_terminal_offsets)
                - is_ports: (V,) boolean mask for ports
        """
        cfg = self.cfg
        device = cfg.placement_device if cfg.placement_device else 'cuda'
        max_generation_attempts = max(1, int(getattr(cfg, 'max_generation_attempts', 1)))
        idx = self._sample_count  # Increment only when we accept

        def _mix_seed(base: int, s: int, salt: int) -> int:
            x = (int(base) & ((1 << 64) - 1)) ^ (int(s) * 0x9E3779B97F4A7C15) ^ (int(salt) * 0xBF58476D1CE4E5B9)
            x = (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
            x = (x ^ (x >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
            x = x ^ (x >> 31)
            return int(x % (2**31 - 1))

        for retry in range(max_generation_attempts):
            # Per-graph diversity: sample placement parameters (different each retry)
            diversity_overrides = self._sample_diversity_params()
            dov = diversity_overrides
            _density_pg = dov.get('target_density', cfg.target_density)
            _ar_range_pg = dov.get('canvas_aspect_ratio_range', None)

            # Seeds: include retry so each attempt gets a different graph
            if cfg.seed is not None:
                graph_seed = _mix_seed(cfg.seed, idx * 1000 + retry, salt=0)
                placer_seed = _mix_seed(cfg.seed, idx * 1000 + retry, salt=1)
                torch.manual_seed(graph_seed)
                random.seed(graph_seed)
            else:
                graph_seed = None
                placer_seed = torch.randint(0, 2**31, (1,), device='cpu').item()

            # Step 1: Generate instance sizes (bimodal, physical units)
            use_packing_alg = getattr(cfg, "generation_pipeline", "v5") == "packing_alg"
            if use_packing_alg:
                # packing_alg: avoid hard cap at target_nodes by using a soft buffered pool.
                # Keep chip-size reference tied to target_nodes (max_instance), not pool size.
                target_n = max(1, int(getattr(cfg, "max_instance", 1)))
                base_pool = max(1, int(getattr(cfg, "packing_alg_instance_pool_size", target_n)))
                pool_mult = float(getattr(cfg, "packing_alg_instance_buffer_multiplier", 2.0))
                buffered_pool = max(base_pool, int(target_n * pool_mult))
                n_instances_to_generate = max(1, buffered_pool)
                N_reference_for_gen = target_n
            else:
                # v5: generate 10x max_instance so placement stops by density, not slot cap.
                n_instances_to_generate = int(cfg.max_instance * 10.0)
                N_reference_for_gen = cfg.max_instance  # For macro/port fractions
            
            instance_features = self._generate_bimodal_instances(
                N_inst=n_instances_to_generate,
                N_reference=N_reference_for_gen,
                device=device,
                diversity_overrides=diversity_overrides,
            )
            w = instance_features['w']
            h = instance_features['h']
            areas = instance_features['area']
            is_macro = instance_features['is_macro']
            is_port = instance_features['is_port']
            is_super_macro = instance_features.get('is_super_macro', torch.zeros_like(is_macro))
            
            # Step 2: Calculate canvas size using reference instance count.
            # Default v5: size chip for cfg.max_instance and let placement stop by density.
            # packing_alg: size chip from target_nodes (max_instance), while using buffered
            # candidate pool to avoid slot-cap-induced under-density.
            if getattr(cfg, "generation_pipeline", "v5") == "packing_alg":
                n_reference = max(1, int(getattr(cfg, "max_instance", len(areas))))
            else:
                n_reference = min(cfg.max_instance, len(areas))
            areas_reference = areas[:n_reference]
            is_port_reference = is_port[:n_reference]
            is_macro_reference = is_macro[:n_reference]
    
            chip_width, chip_height = self._calculate_canvas_size(
                areas_reference, is_port_reference, is_macro=is_macro_reference, device=device,
                target_density_override=_density_pg,
                aspect_ratio_range_override=_ar_range_pg,
                diversity_overrides=diversity_overrides,
            )

            if use_packing_alg:
                is_port, w, h, areas = self._assign_packing_alg_ports(
                    w, h, areas, chip_width, chip_height, N_reference_for_gen, device
                )
            
            # Step 3: Sort by area (descending) for placement
            _, indices = torch.sort(areas, descending=True)
            w_sorted = w[indices]
            h_sorted = h[indices]
            is_port_sorted = is_port[indices]
            is_macro_sorted = is_macro[indices]
            
            # Step 4: V5 Placement (macro-first + grid-based + legalization)
            # Prepare inputs for V5B1Placer
            w_sorted_cpu = w_sorted.cpu()
            h_sorted_cpu = h_sorted.cpu()
            is_port_sorted_cpu = is_port_sorted.cpu()
            is_macro_sorted_cpu = is_macro_sorted.cpu()
            
            # Create sizes tensor [N, 2]
            sizes_tensor = torch.stack([w_sorted_cpu, h_sorted_cpu], dim=-1)
            
            # Create types array: 0=stdcell, 1=macro, 2=port (vectorized)
            types_tensor = torch.zeros(len(w_sorted_cpu), dtype=torch.int32, device='cpu')
            types_tensor[is_port_sorted_cpu > 0.5] = 2  # ports
            types_tensor[is_macro_sorted_cpu > 0.5] = 1  # macros (overwrites ports if both, but ports are separate)
            
            # Place using V5Placer (with per-graph diversity overrides)
            placer_cfg = self._make_placer_config(cfg, diversity_overrides)
            placer = V5Placer(placer_cfg, seed=placer_seed, device=device)
            positions, sizes_placed, mask, placed_mask = placer.place(
                sizes=sizes_tensor,
                types=types_tensor,
                canvas_W=chip_width,
                canvas_H=chip_height
            )
            
            # Get number of placed instances (already filtered by placer)
            num_instances = positions.shape[0]
    
            if num_instances == 0:
                continue  # Retry: no instances placed
    
            # Report achieved density for visibility, but do not reject under-dense graphs.
            canvas_area = chip_width * chip_height
            total_placed_area = (sizes_placed[:, 0] * sizes_placed[:, 1]).sum().item()
            final_density = total_placed_area / canvas_area if canvas_area > 0 else 0.0
    
            # Accept this graph
            self._sample_count += 1
    
            # Filter sorted arrays to only include placed instances
            # placed_mask[i] indicates if sorted index i was placed
            placed_mask_cpu = placed_mask.cpu() if placed_mask.device.type != 'cpu' else placed_mask
            placed_indices_sorted = torch.where(placed_mask_cpu)[0].tolist()
            
            # Filter is_macro_sorted and is_port_sorted to only placed instances
            is_macro_placed = is_macro_sorted[placed_indices_sorted]
            is_port_placed = is_port_sorted[placed_indices_sorted]
            is_super_macro_sorted = is_super_macro[indices]
            is_super_macro_placed = is_super_macro_sorted[placed_indices_sorted]
            
            # Use sizes_placed for terminal assignment (already filtered by placer)
            sizes = sizes_placed
            
            
            # Step 5: Rent's Rule Terminal Assignment
            instance_area = sizes[:, 0] * sizes[:, 1]
            num_terminals_dist = get_distribution("cond_binomial", {
                "binom_p": cfg.num_terminals_binom_p,
                "binom_min_n": cfg.num_terminals_min_n,
                "t": cfg.num_terminals_t,
                "p": cfg.num_terminals_p,
            })
            num_terminals = num_terminals_dist.sample(instance_area).int()
            
            # Apply tier-specific limits (using filtered arrays)
            stdcell_mask = (~is_macro_placed.bool()) & (~is_port_placed.bool())
            macro_mask = is_macro_placed.bool()
            port_mask = is_port_placed.bool()
            
            # Standard cells: max 5 terminals
            if stdcell_mask.any():
                num_terminals[stdcell_mask] = torch.clamp(num_terminals[stdcell_mask], min=1, max=5)
            # Macros: max 500 terminals
            if macro_mask.any():
                num_terminals[macro_mask] = torch.clamp(num_terminals[macro_mask], min=1, max=500)
            
            num_terminals = torch.clamp(num_terminals, min=1)
            max_num_terminals = torch.max(num_terminals)
            
            # Generate terminal offsets
            # Generate terminal offsets (with macro pin placement on perimeter)
            terminal_offsets = self.get_terminal_offsets(
                sizes[:, 0], sizes[:, 1], max_num_terminals, reference="center",
                is_macro=is_macro_placed
            )
            
            # Step 6: Degree-Constrained Sparse Edge Generation
            terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets  # (V, T, 2)
            
            # Move to edge device
            edge_device = cfg.edge_device if cfg.edge_device else device
            if cfg.edge_device:
                terminal_positions = terminal_positions.to(edge_device)
                num_terminals_edge = num_terminals.to(edge_device)
            else:
                num_terminals_edge = num_terminals
            
            # Sample source terminals with macro-specific fanout pattern
            # Macros: more outputs (fanout to std-cells), fewer inputs
            macro_output_fraction = getattr(cfg, 'macro_output_fraction', 0.7)
            is_source = torch.zeros((num_instances, max_num_terminals), device=edge_device)
            
            # For macros: assign terminals as outputs/inputs based on fraction
            if macro_mask.any():
                macro_indices = torch.where(macro_mask)[0]
                for macro_idx in macro_indices:
                    num_t = int(num_terminals[macro_idx].item())
                    n_outputs = max(1, int(num_t * macro_output_fraction))
                    # First n_outputs are sources (outputs), rest are sinks (inputs)
                    is_source[macro_idx, :n_outputs] = 1.0
            
            # For std-cells and ports: use default probability
            non_macro_mask = ~macro_mask
            if non_macro_mask.any():
                is_source[non_macro_mask] = get_distribution("bernoulli", {"probs": cfg.source_terminal_prob}).sample(
                    (non_macro_mask.sum().item(), max_num_terminals)
                ).to(edge_device)
            
            # Per-graph clustering params
            _enable_clustering_pg = dov.get('enable_clustering', getattr(cfg, 'enable_clustering', True))
            _cluster_radius_pg = dov.get('cluster_radius', getattr(cfg, 'cluster_radius', 50.0))
            _intra_cluster_boost_pg = dov.get('intra_cluster_boost', getattr(cfg, 'intra_cluster_boost', 2.5))
    
            # Compute std-cell clusters for hierarchical connectivity (if enabled)
            cluster_assignments = None
            if _enable_clustering_pg and stdcell_mask.any():
                cluster_assignments = self._compute_stdcell_clusters(
                    positions, stdcell_mask, _cluster_radius_pg, device
                )
            
            # Per-graph degree distribution (from diversity_overrides, else cfg defaults)
            _degree_mean_pg = dov.get('target_degree_mean', cfg.target_degree_mean)
            _degree_std_pg = dov.get('target_degree_std', cfg.target_degree_std)
            macro_degree_mean = dov.get('macro_degree_mean', getattr(cfg, 'macro_degree_mean', _degree_mean_pg * 5.0))
            macro_degree_std = dov.get('macro_degree_std', getattr(cfg, 'macro_degree_std', _degree_mean_pg * 4.0))
            macro_degree_min = dov.get('macro_degree_min', getattr(cfg, 'macro_degree_min', 15))
            macro_degree_max = dov.get('macro_degree_max', getattr(cfg, 'macro_degree_max', 300))
            stdcell_degree_mean = getattr(cfg, 'stdcell_degree_mean', _degree_mean_pg * 0.9)
            
            target_degrees = torch.zeros(num_instances, dtype=torch.float32, device=device)
            
            def sample_realistic_degrees(n: int, mean: float, std: float, min_degree: int = 2, use_mixture: bool = False, max_degree: int = 50) -> torch.Tensor:
                """
                Sample degrees with realistic industrial netlist distribution.
                Avoids degree=1 dominance by using shifted normal distribution.
                Ensures mass at degrees 2-5, with gradual tail up to max_degree.
                
                Args:
                    n: Number of samples
                    mean: Mean degree
                    std: Standard deviation (larger std = longer tail)
                    min_degree: Minimum degree (default 2 to avoid leaf-heavy graphs)
                    use_mixture: If True, use mixture of normal + explicit 2-5 distribution (for std-cells)
                    max_degree: Maximum degree (default 50, matching industrial netlists)
                """
                if use_mixture:
                    # Mixture model: 40% explicit 2-5 distribution, 60% shifted normal with tail
                    # This ensures strong mass at degrees 2-5 while allowing gradual tail to ~50
                    explicit_degrees = torch.tensor([2, 3, 4, 5], dtype=torch.float32, device=device)
                    explicit_probs = torch.tensor([0.25, 0.30, 0.25, 0.20], dtype=torch.float32, device=device)
                    
                    n_explicit = int(n * 0.4)  # Increased from 0.3 to ensure more mass at 2-5
                    n_normal = n - n_explicit
                    
                    # Sample from explicit distribution (ensures mass at 2-5)
                    explicit_samples = torch.multinomial(explicit_probs, n_explicit, replacement=True)
                    explicit_values = explicit_degrees[explicit_samples]
                    
                    # Sample from normal distribution (allows tail up to ~50)
                    if n_normal > 0:
                        normal_samples_raw = get_distribution("normal", {"mean": mean, "std": std}).sample((n_normal,))
                        if isinstance(normal_samples_raw, torch.Tensor):
                            normal_samples = normal_samples_raw.to(device)
                        else:
                            normal_samples = torch.tensor(normal_samples_raw, dtype=torch.float32, device=device)
                        # Clamp to [min_degree, max_degree] to allow tail but avoid extreme outliers
                        normal_samples = torch.clamp(normal_samples, min=min_degree, max=max_degree)
                        combined = torch.cat([normal_samples, explicit_values])
                    else:
                        combined = explicit_values
                    
                    # Shuffle to avoid ordering bias
                    perm = torch.randperm(len(combined), device=device)
                    return combined[perm]
                else:
                    # Simple shifted normal: sample from normal, clamp to [min_degree, max_degree]
                    sampled_raw = get_distribution("normal", {"mean": mean, "std": std}).sample((n,))
                    if isinstance(sampled_raw, torch.Tensor):
                        sampled = sampled_raw.to(device)
                    else:
                        sampled = torch.tensor(sampled_raw, dtype=torch.float32, device=device)
                    # Clamp to [min_degree, max_degree] to allow tail but avoid extreme outliers
                    return torch.clamp(sampled, min=min_degree, max=max_degree)
            
            if macro_mask.any():
                # Macros: VERY high degrees (fan out to many std-cells)
                # Use specialized macro degree sampling: high mean, high variance, high minimum
                n_macros = macro_mask.sum().item()
                
                # Sample from log-normal distribution (better for high-degree tail)
                # Log-normal ensures most macros have high degree, with some having very high degree
                log_mean = math.log(macro_degree_mean) - 0.5 * math.log(1 + (macro_degree_std / macro_degree_mean) ** 2)
                log_std = math.log(1 + (macro_degree_std / macro_degree_mean) ** 2) ** 0.5
                
                # Sample from log-normal
                log_normal_samples = torch.exp(torch.normal(log_mean, log_std, (n_macros,), device=device))
                
                # Clamp to [min, max] range
                sampled = torch.clamp(log_normal_samples, min=macro_degree_min, max=macro_degree_max)
                
                target_degrees[macro_mask] = sampled
            
            if stdcell_mask.any():
                # Std-cells: mass at 2-5, gradual tail up to ~50
                # Use mixture model to ensure mass at 2-5 while allowing tail
                sampled = sample_realistic_degrees(
                    stdcell_mask.sum().item(),
                    stdcell_degree_mean,
                    _degree_std_pg * 1.0,  # per-graph std; full value to allow tail up to ~50
                    min_degree=2,
                    use_mixture=True,  # Use mixture for std-cells: mass at 2-5 + tail
                    max_degree=50
                )
                target_degrees[stdcell_mask] = sampled
            
            if port_mask.any():
                # Ports: low-medium degree (2-10), but not degree=1
                sampled = sample_realistic_degrees(
                    port_mask.sum().item(),
                    3.0,  # Slightly higher mean
                    cfg.target_degree_std * 0.5,  # Smaller std for ports
                    min_degree=2,
                    use_mixture=False,
                    max_degree=20  # Ports can have moderate fanout
                )
                target_degrees[port_mask] = sampled
            
            target_degrees = target_degrees.int()
            
            # Collect all source terminals with their instance info and target degrees
            source_terminal_list = []  # List of (instance_idx, terminal_idx, k_needed)
            for i in range(num_instances):
                num_terms_i = int(num_terminals[i].item())
                target_deg_i = int(target_degrees[i].item())
                
                if num_terms_i == 0:
                    continue
                
                # Get source terminals for this instance
                source_terminal_mask = is_source[i, :num_terms_i] > 0.5
                source_terminals = torch.where(source_terminal_mask)[0].tolist()
                
                if len(source_terminals) == 0:
                    continue
                
                # Distribute target degree across source terminals
                remaining_degree = target_deg_i
                for term_idx, t in enumerate(source_terminals):
                    if remaining_degree <= 0:
                        break
                    
                    # Last terminal gets remaining connections
                    if term_idx == len(source_terminals) - 1:
                        k_needed = remaining_degree
                    else:
                        # Distribute evenly
                        k_needed = max(1, remaining_degree // (len(source_terminals) - term_idx))
                    
                    source_terminal_list.append((i, t, k_needed))
                    remaining_degree -= k_needed
            
            # Pre-compute instance type tensors once (not in loop!)
            edge_device = cfg.edge_device if cfg.edge_device else device
            is_macro_edge = is_macro_placed.to(edge_device) if cfg.edge_device else is_macro_placed
            is_port_edge = is_port_placed.to(edge_device) if cfg.edge_device else is_port_placed
            cluster_assignments_edge = cluster_assignments.to(edge_device) if cluster_assignments is not None else None
    
            # GPU hard radius cutoff (fast): stdcell -> stdcell only within radius (improves HPWL, avoids long edges)
            edge_dev_obj = torch.device(edge_device) if isinstance(edge_device, str) else edge_device
            instance_positions_edge = positions.to(edge_dev_obj) if positions.device != edge_dev_obj else positions
            is_stdcell_edge = (~is_macro_edge.bool()) & (~is_port_edge.bool())
            
            # Per-graph edge generation parameters
            _edge_dist_scale_pg = dov.get('edge_distance_scale', cfg.edge_distance_scale)
            _decay_type_pg = dov.get('distance_decay_type', getattr(cfg, 'distance_decay_type', 'hybrid'))
            _power_alpha_pg = dov.get('distance_power_alpha', getattr(cfg, 'distance_power_alpha', 1.5))
            _crossover_pg = dov.get('distance_crossover', getattr(cfg, 'distance_crossover', 20.0))
            _edge_sel_pg = dov.get('edge_neighbor_selection', getattr(cfg, 'edge_neighbor_selection', 'deterministic_topk'))
    
            # Optimized: On-the-fly grid gathering (no pre-computed candidate lists)
            all_edges = []
            _hard_radius_pg = dov.get(
                'stdcell_hard_radius',
                (float(getattr(cfg, "stdcell_hard_radius", 0.0)) if getattr(cfg, "enable_stdcell_hard_radius_cuda", False) else None),
            )

            self._sample_edges_with_batched_distances(
                terminal_positions, source_terminal_list, num_terminals, is_source,
                _edge_dist_scale_pg, edge_dev_obj, is_macro_edge, is_port_edge,
                cluster_assignments_edge, all_edges,
                instance_positions=instance_positions_edge,
                is_stdcell=is_stdcell_edge,
                stdcell_hard_radius=_hard_radius_pg,
                stdcell_hard_radius_norm=int(getattr(cfg, "stdcell_hard_radius_norm", 1)),
                spatial_cell_size=getattr(cfg, 'spatial_candidate_cell_size', None),
                chip_width=chip_width,
                chip_height=chip_height,
                distance_decay_type=_decay_type_pg,
                distance_power_alpha=_power_alpha_pg,
                distance_crossover=_crossover_pg,
                edge_neighbor_selection=_edge_sel_pg,
                intra_cluster_boost=_intra_cluster_boost_pg,
            )
            
            # Step 7: Port connectivity pattern (realistic: ports connect to macros/clusters, not random)
            # Ports should connect to: macros (40%), std-cell clusters (40%), or other ports (20%)
            port_instances = torch.where(is_port_placed.bool())[0].tolist()
            macro_instances = torch.where(is_macro_placed.bool())[0].tolist()
            
            if port_instances:
                port_to_macro_prob = dov.get('port_to_macro_prob', getattr(cfg, 'port_to_macro_prob', 0.4))
                port_to_cluster_prob = dov.get('port_to_cluster_prob', getattr(cfg, 'port_to_cluster_prob', 0.4))
                port_to_port_prob = dov.get('port_to_port_prob', getattr(cfg, 'port_to_port_prob', 0.2))
                
                # Get cluster representatives (one std-cell per cluster for port connections)
                cluster_representatives = {}
                if cluster_assignments is not None:
                    for v in range(num_instances):
                        if stdcell_mask[v] and cluster_assignments[v].item() >= 0:
                            cluster_id = cluster_assignments[v].item()
                            if cluster_id not in cluster_representatives:
                                cluster_representatives[cluster_id] = v
                
                for port_i in port_instances:
                    num_terms_port = int(num_terminals[port_i].item())
                    source_terminals_port = torch.where(is_source[port_i, :num_terms_port] > 0.5)[0].tolist()
                    
                    for t in source_terminals_port:
                        # Match port wiring style to the per-graph edge selection strategy.
                        selection = _edge_sel_pg
    
                        def _u01_from_int(x: int) -> float:
                            # Deterministic pseudo-uniform in [0,1). (SplitMix64-like)
                            x = (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
                            x = (x ^ (x >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
                            x = x ^ (x >> 31)
                            return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    
                        if selection == "deterministic_topk":
                            # Choose target category deterministically using a hash of (port_i, t).
                            # This preserves the macro/cluster/port mixture ratios without RNG.
                            rand_val = _u01_from_int(int(port_i) * 1315423911 + int(t) * 2654435761)
    
                            # Choose the closest target within the chosen category to encourage low-HPWL wiring.
                            port_xy = positions[port_i].detach().cpu()
    
                            if rand_val < port_to_macro_prob and macro_instances:
                                macro_xy = positions[macro_instances].detach().cpu()
                                d = (macro_xy - port_xy.unsqueeze(0)).abs().sum(dim=-1)
                                target_v = macro_instances[int(torch.argmin(d).item())]
                            elif rand_val < port_to_macro_prob + port_to_cluster_prob and cluster_representatives:
                                reps = list(cluster_representatives.values())
                                reps_xy = positions[reps].detach().cpu()
                                d = (reps_xy - port_xy.unsqueeze(0)).abs().sum(dim=-1)
                                target_v = reps[int(torch.argmin(d).item())]
                            elif rand_val < port_to_macro_prob + port_to_cluster_prob + port_to_port_prob and len(port_instances) > 1:
                                other_ports = [p for p in port_instances if p != port_i]
                                other_xy = positions[other_ports].detach().cpu()
                                d = (other_xy - port_xy.unsqueeze(0)).abs().sum(dim=-1)
                                target_v = other_ports[int(torch.argmin(d).item())]
                            else:
                                continue
    
                            # Deterministic sink terminal choice: pick the first sink terminal (not a source).
                            n_tgt = int(num_terminals[target_v].item())
                            sink_idx = None
                            for tt in range(min(int(max_num_terminals.item()), n_tgt)):
                                if is_source[target_v, tt].item() < 0.5:
                                    sink_idx = tt
                                    break
                            if sink_idx is None:
                                continue
                            target_t = sink_idx
                        else:
                            # Legacy stochastic behavior
                            rand_val = torch.rand(1).item()
                            
                            if rand_val < port_to_macro_prob and macro_instances:
                                # Connect to a macro
                                target_v = macro_instances[torch.randint(0, len(macro_instances), (1,), device=device).item()]
                            elif rand_val < port_to_macro_prob + port_to_cluster_prob and cluster_representatives:
                                # Connect to a cluster representative (std-cell in cluster)
                                cluster_ids = list(cluster_representatives.keys())
                                chosen_cluster = cluster_ids[torch.randint(0, len(cluster_ids), (1,), device=device).item()]
                                target_v = cluster_representatives[chosen_cluster]
                            elif rand_val < port_to_macro_prob + port_to_cluster_prob + port_to_port_prob and len(port_instances) > 1:
                                # Connect to another port
                                other_ports = [p for p in port_instances if p != port_i]
                                target_v = other_ports[torch.randint(0, len(other_ports), (1,), device=device).item()]
                            else:
                                # Fallback: skip this connection (ports don't connect randomly to std-cells)
                                continue
                            
                            target_t = torch.randint(
                                0, min(max_num_terminals, int(num_terminals[target_v].item())), 
                                (1,), device=device
                            ).item()
                        
                        # Must be source->sink
                        if is_source[target_v, target_t].item() < 0.5:
                            all_edges.append([port_i, t, target_v, target_t])
    
            # Quick locality metric (proxy for HPWL): std-cell <-> std-cell edge lengths
            try:
                if all_edges:
                    edges_tmp = torch.tensor(all_edges, dtype=torch.long, device='cpu')
                    src_v = edges_tmp[:, 0]
                    dst_v = edges_tmp[:, 2]
                    is_stdcell = (~is_macro_placed.bool()) & (~is_port_placed.bool())
                    std_mask = is_stdcell[src_v] & is_stdcell[dst_v]
                    if std_mask.any():
                        p_cpu = positions.detach().cpu()
                        d = (p_cpu[src_v[std_mask]] - p_cpu[dst_v[std_mask]]).abs().sum(dim=-1)  # L1
                        d_sorted, _ = torch.sort(d)
                        p95 = d_sorted[int(0.95 * (len(d_sorted) - 1))].item() if len(d_sorted) > 1 else d_sorted[0].item()
                        print(
                            f"  Stdcell-Stdcell edge L1 distance: mean={d.mean().item():.2f}µm, p95={p95:.2f}µm "
                            f"(hard_radius={_hard_radius_pg}, norm={getattr(cfg,'stdcell_hard_radius_norm',None)})"
                        )
            except Exception:
                pass
            
            # Connect isolated instances
            if all_edges:
                edge_tensor = torch.tensor(all_edges, dtype=torch.long, device=device)
                out_degree = torch.zeros((num_instances, max_num_terminals), dtype=torch.int32, device=device)
                for edge_idx in range(edge_tensor.shape[0]):
                    edge = edge_tensor[edge_idx]
                    out_degree[edge[0].item(), edge[1].item()] += 1
                
                terminal_pos_for_conn = terminal_positions.cpu() if cfg.edge_device else terminal_positions
                out_degree_for_conn = out_degree.cpu() if cfg.edge_device else out_degree
                positions_for_conn = positions.cpu() if cfg.edge_device else positions
                
                edge_tensor = self.connect_isolated_instances_sparse(
                    edge_tensor, terminal_pos_for_conn, out_degree_for_conn,
                    num_instances, max_num_terminals,
                    instance_positions=positions_for_conn
                )
            else:
                edge_tensor = torch.empty((0, 4), dtype=torch.long, device=device)
    
            # Deterministically bridge disconnected components to encourage a dominant giant component (netlist-like)
            if edge_tensor.shape[0] > 0:
                edge_tensor = self._bridge_disconnected_components_sparse(
                    edge_tensor,
                    num_instances=num_instances,
                    positions=positions_for_conn if all_edges else positions,
                    num_terminals=num_terminals,
                    is_source=is_source,
                    chip_width=chip_width,
                    chip_height=chip_height,
                )

            # Enforce invariant: every port must connect to at least one non-port instance.
            edge_tensor = self._enforce_port_to_nonport_connectivity_sparse(
                edge_tensor,
                positions=positions_for_conn if all_edges else positions,
                is_port=is_port_placed,
                num_terminals=num_terminals,
                is_source=is_source,
                num_instances=num_instances,
            )
            
            # Convert to edge_index and edge_attr
            if edge_tensor.shape[0] > 0:
                if edge_tensor.device != terminal_offsets.device:
                    terminal_offsets = terminal_offsets.to(edge_tensor.device)
                edge_index, edge_attr = self.generate_edge_list_sparse(edge_tensor, terminal_offsets)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_attr = torch.zeros((0, 4), dtype=torch.float32, device=device)
            
            # mask is already returned from V5B1Placer.place() (only for placed instances)
            # All unplaced instances have been filtered out by the placer
            # Use it directly (it's already in the correct order)
            
            # Store chip_size (canvas size) in data for later use in normalization
            # This is the actual canvas dimensions used for placement
            chip_size_tensor = torch.tensor([chip_width, chip_height], dtype=torch.float32, device=device)
    
            # ----------------------------------------------------------------------
            # NEW: Save authoritative node-type masks to eliminate downstream ambiguity.
            # ----------------------------------------------------------------------
            is_port_bool = is_port_placed.bool()
            is_macro_bool = is_macro_placed.bool()
            is_stdcell_bool = (~is_port_bool) & (~is_macro_bool)
    
            # Diffusion mask: std-cells are diffused; macros + ports are fixed.
            pos_mask = is_stdcell_bool.clone()
            
            data = Data(
                x=sizes, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                is_ports=mask,
                chip_size=chip_size_tensor,  # Store canvas size for data loading pipeline
                # NEW: pin count per instance (transfer-safe feature; available in real netlists too)
                num_terminals=num_terminals.to(device=device),
                # NEW: explicit type masks (authoritative from generator)
                is_macro=is_macro_bool,
                is_super_macro=is_super_macro_placed.bool(),
                is_stdcell=is_stdcell_bool,
                # NEW: authoritative diffusion mask (stdcells only)
                pos_mask=pos_mask,
                # NEW: seeds for reproducibility/debugging
                graph_seed=(torch.tensor(graph_seed, device=device, dtype=torch.long) if graph_seed is not None else None),
                placer_seed=torch.tensor(placer_seed, device=device, dtype=torch.long),
            )

            # ── Optional: DreamPlace-style stdcell gradient optimization ──────
            if getattr(cfg, 'dreamplace_enabled', False):
                try:
                    from .dreamplace_stdcell import StdcellOptimizer, DreamPlaceConfig
                    dp_cfg = DreamPlaceConfig(
                        num_iterations      = getattr(cfg, 'dreamplace_num_iterations',    300),
                        learning_rate       = getattr(cfg, 'dreamplace_learning_rate',     0.01),
                        optimizer_type      = getattr(cfg, 'dreamplace_optimizer_type',    'adam'),
                        wa_gamma            = getattr(cfg, 'dreamplace_wa_gamma',          1.0),
                        num_bins_x          = getattr(cfg, 'dreamplace_num_bins_x',        32),
                        num_bins_y          = getattr(cfg, 'dreamplace_num_bins_y',        32),
                        target_density      = getattr(cfg, 'dreamplace_density_target',    0.7),
                        lambda_density_max  = getattr(cfg, 'dreamplace_lambda_density_max', 10.0),
                        lambda_growth_rate  = getattr(cfg, 'dreamplace_lambda_growth_rate', 0.02),
                        warmup_iters        = getattr(cfg, 'dreamplace_warmup_iters',      50),
                        lambda_boundary     = getattr(cfg, 'dreamplace_lambda_boundary',   1.0),
                        grad_clip_norm      = getattr(cfg, 'dreamplace_grad_clip_norm',    5.0),
                        legalize            = getattr(cfg, 'dreamplace_legalize',          True),
                        stdcell_grid_spacing_factor_w = getattr(cfg, 'stdcell_grid_spacing_factor_w', 0.97),
                        stdcell_grid_spacing_factor_h = getattr(cfg, 'stdcell_grid_spacing_factor_h', 0.98),
                        stdcell_macro_keepout         = getattr(cfg, 'stdcell_macro_keepout',         0.2),
                        fallback_on_nan     = getattr(cfg, 'dreamplace_fallback_on_nan',   True),
                        fallback_hpwl_ratio = getattr(cfg, 'dreamplace_fallback_hpwl_ratio', 2.0),
                        device              = getattr(cfg, 'dreamplace_device',            'auto'),
                    )
                    positions, data = StdcellOptimizer(dp_cfg).optimize(positions, data)
                except Exception as _dp_exc:
                    import warnings as _warnings
                    _warnings.warn(
                        f"[DreamPlace] Optimizer failed, using original positions: {_dp_exc}"
                    )

            return positions, data

        raise RuntimeError(
            f"Failed to generate graph after {max_generation_attempts} attempt(s)"
        )
