"""Occupancy grid for fast free-space queries and sampling.

Uses a bin mask for fast rejection (not precise, but very fast).
Precise legality is handled by SpatialHash2D.
"""

from typing import List, Tuple, Optional
import math
import random


class OccupancyGrid:
    """
    Occupancy grid for fast free-space queries.
    
    Uses a 2D boolean mask where each bin represents a region of space.
    Fast rejection test: if any bin overlapped by a rectangle is occupied,
    the rectangle is likely not free (may have false positives).
    
    Note: This is for fast rejection only. Always use SpatialHash2D for
    precise legality checks.
    """
    
    def __init__(self, W: float, H: float, bin_size: float):
        """
        Initialize occupancy grid.
        
        Args:
            W: Canvas width
            H: Canvas height
            bin_size: Size of each bin
        """
        self.W = W
        self.H = H
        self.bin_size = bin_size
        
        # Grid dimensions
        self.grid_w = int(math.ceil(W / bin_size))
        self.grid_h = int(math.ceil(H / bin_size))
        
        # Occupancy mask: True = occupied, False = free
        self.mask = [[False for _ in range(self.grid_w)] for _ in range(self.grid_h)]
    
    def _get_bin_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to bin coordinates."""
        ix = int(math.floor(x / self.bin_size))
        iy = int(math.floor(y / self.bin_size))
        ix = max(0, min(self.grid_w - 1, ix))
        iy = max(0, min(self.grid_h - 1, iy))
        return ix, iy
    
    def _get_bin_bounds(self, x_min: float, y_min: float, x_max: float, y_max: float) -> Tuple[int, int, int, int]:
        """
        Get bin bounds for rectangle.
        
        Returns:
            (ix_min, iy_min, ix_max, iy_max)
        """
        # Clamp to canvas
        x_min = max(0.0, min(self.W, x_min))
        y_min = max(0.0, min(self.H, y_min))
        x_max = max(0.0, min(self.W, x_max))
        y_max = max(0.0, min(self.H, y_max))
        
        ix_min = int(math.floor(x_min / self.bin_size))
        iy_min = int(math.floor(y_min / self.bin_size))
        ix_max = int(math.floor(x_max / self.bin_size))
        iy_max = int(math.floor(y_max / self.bin_size))
        
        # Clamp to grid bounds
        ix_min = max(0, min(self.grid_w - 1, ix_min))
        iy_min = max(0, min(self.grid_h - 1, iy_min))
        ix_max = max(0, min(self.grid_w - 1, ix_max))
        iy_max = max(0, min(self.grid_h - 1, iy_max))
        
        return ix_min, iy_min, ix_max, iy_max
    
    def rasterize_rects(self, rects: List[Tuple[float, float, float, float]], 
                       value: bool = True, margin: float = 0.0):
        """
        Rasterize rectangles into occupancy grid.
        
        Args:
            rects: List of (x_center, y_center, w, h) tuples
            value: Value to set (True = occupied, False = free)
            margin: Additional margin around rectangles
        """
        for x, y, w, h in rects:
            # Convert center to bounds
            x_min = x - w / 2 - margin
            y_min = y - h / 2 - margin
            x_max = x + w / 2 + margin
            y_max = y + h / 2 + margin
            
            # Get bin bounds
            ix_min, iy_min, ix_max, iy_max = self._get_bin_bounds(x_min, y_min, x_max, y_max)
            
            # Mark bins
            for iy in range(iy_min, iy_max + 1):
                for ix in range(ix_min, ix_max + 1):
                    self.mask[iy][ix] = value
    
    def is_free_rect(self, x: float, y: float, w: float, h: float, margin: float = 0.0) -> bool:
        """
        Fast bin-level check if rectangle is free.
        
        This is a fast rejection test. May have false positives (says free
        when actually occupied), but should have no false negatives (won't
        say occupied when actually free).
        
        Args:
            x, y: Center coordinates
            w, h: Width and height
            margin: Additional margin
            
        Returns:
            True if appears free (all bins are free), False if likely occupied
        """
        # Convert center to bounds
        x_min = x - w / 2 - margin
        y_min = y - h / 2 - margin
        x_max = x + w / 2 + margin
        y_max = y + h / 2 + margin
        
        # Get bin bounds
        ix_min, iy_min, ix_max, iy_max = self._get_bin_bounds(x_min, y_min, x_max, y_max)
        
        # Check if any bin is occupied
        for iy in range(iy_min, iy_max + 1):
            for ix in range(ix_min, ix_max + 1):
                if self.mask[iy][ix]:
                    return False
        
        return True
    
    def free_bins_list(self) -> List[Tuple[int, int]]:
        """
        Get list of all free bin coordinates.
        
        Returns:
            List of (ix, iy) tuples for free bins
        """
        free_bins = []
        for iy in range(self.grid_h):
            for ix in range(self.grid_w):
                if not self.mask[iy][ix]:
                    free_bins.append((ix, iy))
        return free_bins
    
    def sample_free_point(self, rng: random.Random) -> Optional[Tuple[float, float]]:
        """
        Sample a random point from a free bin.
        
        Args:
            rng: Random number generator
            
        Returns:
            (x, y) world coordinates, or None if no free bins
        """
        free_bins = self.free_bins_list()
        if not free_bins:
            return None
        
        # Sample a free bin
        ix, iy = rng.choice(free_bins)
        
        # Sample point within bin
        x = rng.uniform(ix * self.bin_size, (ix + 1) * self.bin_size)
        y = rng.uniform(iy * self.bin_size, (iy + 1) * self.bin_size)
        
        # Clamp to canvas
        x = max(0.0, min(self.W, x))
        y = max(0.0, min(self.H, y))
        
        return (x, y)
    
    def clear(self):
        """Clear occupancy grid (mark all bins as free)."""
        self.mask = [[False for _ in range(self.grid_w)] for _ in range(self.grid_h)]
