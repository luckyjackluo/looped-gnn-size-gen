"""GPU-optimized placement engine for placement-first generation."""

import torch
from typing import Tuple, Optional


class GPUPlacement:
    """
    GPU-accelerated placement engine using PyTorch tensors.
    
    Checks overlaps in parallel on GPU for fast placement.
    """
    
    def __init__(self, device: str = "cuda", chip_min: float = -1.0, chip_max: float = 1.0, 
                 chip_width: Optional[float] = None, chip_height: Optional[float] = None):
        """
        Initialize placement engine.
        
        Args:
            device: Device for computation ("cuda" or "cpu")
            chip_min: Minimum chip coordinate (for square canvas)
            chip_max: Maximum chip coordinate (for square canvas)
            chip_width: Chip width (for rectangular canvas, V5)
            chip_height: Chip height (for rectangular canvas, V5)
        """
        self.device = device
        
        # Support both square and rectangular canvas
        if chip_width is not None and chip_height is not None:
            # Rectangular canvas (V5)
            self.chip_x_min = 0.0
            self.chip_x_max = chip_width
            self.chip_y_min = 0.0
            self.chip_y_max = chip_height
        else:
            # Square canvas (backward compatible)
            self.chip_x_min = chip_min
            self.chip_x_max = chip_max
            self.chip_y_min = chip_min
            self.chip_y_max = chip_max
        
        # Store boxes as (N, 4): [x_min, y_min, x_max, y_max]
        self.boxes = torch.empty((0, 4), device=device, dtype=torch.float32)
        
        # Store instance data (for output)
        self.all_x = []
        self.all_y = []
        self.all_x_size = []
        self.all_y_size = []
        self.is_port_list = []
    
    def check_legality(self, x_pos: float, y_pos: float, x_size: float, y_size: float) -> bool:
        """
        Check if new box overlaps any existing box - fully vectorized.
        
        Args:
            x_pos: Center x coordinate
            y_pos: Center y coordinate
            x_size: Width
            y_size: Height
            
        Returns:
            True if legal (no overlaps, within bounds), False otherwise
        """
        # Convert to box bounds
        x_min = x_pos - x_size / 2
        y_min = y_pos - y_size / 2
        x_max = x_pos + x_size / 2
        y_max = y_pos + y_size / 2
        
        # Check chip bounds (supports rectangular canvas)
        if x_min < self.chip_x_min or x_max > self.chip_x_max:
            return False
        if y_min < self.chip_y_min or y_max > self.chip_y_max:
            return False
        
        # If no existing boxes, it's legal
        if self.boxes.shape[0] == 0:
            return True
        
        # Vectorized overlap check against ALL existing boxes at once
        # Two boxes overlap if they overlap in BOTH x AND y
        existing = self.boxes  # (N, 4)
        
        overlap_x = (x_min < existing[:, 2]) & (existing[:, 0] < x_max)
        overlap_y = (y_min < existing[:, 3]) & (existing[:, 1] < y_max)
        overlaps = overlap_x & overlap_y  # (N,) bool tensor
        
        # Legal if no overlaps
        return not overlaps.any().item()
    
    def commit_instance(self, x_pos: float, y_pos: float, x_size: float, y_size: float, is_port: bool = False):
        """
        Add a new instance to the placement.
        
        Args:
            x_pos: Center x coordinate
            y_pos: Center y coordinate
            x_size: Width
            y_size: Height
            is_port: Whether this is a port (ports don't block placement)
        """
        x_min = x_pos - x_size / 2
        y_min = y_pos - y_size / 2
        x_max = x_pos + x_size / 2
        y_max = y_pos + y_size / 2
        
        # Add to boxes if not a port
        if not is_port:
            new_box = torch.tensor(
                [[x_min, y_min, x_max, y_max]],
                device=self.device,
                dtype=torch.float32
            )
            self.boxes = torch.cat([self.boxes, new_box], dim=0)
        
        # Store for output
        self.all_x.append(x_pos)
        self.all_y.append(y_pos)
        self.all_x_size.append(x_size)
        self.all_y_size.append(y_size)
        self.is_port_list.append(is_port)
    
    def get_density(self) -> float:
        """Return current density (fraction of chip area occupied)."""
        if self.boxes.shape[0] == 0:
            return 0.0
        
        widths = self.boxes[:, 2] - self.boxes[:, 0]
        heights = self.boxes[:, 3] - self.boxes[:, 1]
        areas = widths * heights
        chip_area = (self.chip_x_max - self.chip_x_min) * (self.chip_y_max - self.chip_y_min)
        return areas.sum().item() / chip_area
    
    def get_positions(self) -> torch.Tensor:
        """Returns tensor(V, 2) of x,y placements (centers)."""
        if len(self.all_x) == 0:
            return torch.empty((0, 2), device=self.device, dtype=torch.float32)
        
        positions = torch.stack(
            (torch.tensor(self.all_x, device=self.device),
             torch.tensor(self.all_y, device=self.device)),
            dim=-1
        )
        return positions
    
    def get_sizes(self) -> torch.Tensor:
        """Returns tensor(V, 2) of x,y sizes."""
        if len(self.all_x_size) == 0:
            return torch.empty((0, 2), device=self.device, dtype=torch.float32)
        
        sizes = torch.stack(
            (torch.tensor(self.all_x_size, device=self.device),
             torch.tensor(self.all_y_size, device=self.device)),
            dim=-1
        )
        return sizes
    
    def get_mask(self) -> torch.Tensor:
        """Returns bool tensor of is_port."""
        if len(self.is_port_list) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)
        
        return torch.tensor(self.is_port_list, device=self.device, dtype=torch.bool)

