"""Uniform grid spatial index for fast rectangle overlap queries.

Provides O(1) amortized lookup for rectangle intersection checks.
"""

from typing import List, Set, Tuple
import math


class SpatialHash2D:
    """
    Uniform grid spatial hash for 2D rectangles.
    
    Uses a uniform grid to partition space into cells. Each rectangle
    is inserted into all grid cells it overlaps. Queries check only
    the cells overlapped by the query rectangle, then perform precise
    AABB intersection checks.
    
    Complexity:
    - Insert: O(1) amortized (assuming rectangles are small relative to grid)
    - Query: O(1) amortized (only checks overlapping cells)
    - Overlaps_any: O(k) where k is number of candidates in overlapping cells
    """
    
    def __init__(self, W: float, H: float, cell_size: float):
        """
        Initialize spatial hash.
        
        Args:
            W: Canvas width
            H: Canvas height
            cell_size: Size of each grid cell (should be ~median rectangle size)
        """
        self.W = W
        self.H = H
        self.cell_size = cell_size
        
        # Grid dimensions
        self.grid_w = int(math.ceil(W / cell_size))
        self.grid_h = int(math.ceil(H / cell_size))
        
        # Grid: list of lists, each cell contains set of rect_ids
        self.grid: List[List[Set[int]]] = [
            [set() for _ in range(self.grid_w)]
            for _ in range(self.grid_h)
        ]
        
        # Store rectangle data: rects_db[rect_id] = (x_min, y_min, x_max, y_max)
        self.rects_db: List[Tuple[float, float, float, float]] = []
    
    def _get_grid_cells(self, x_min: float, y_min: float, x_max: float, y_max: float) -> List[Tuple[int, int]]:
        """
        Get all grid cells overlapped by rectangle.
        
        Args:
            x_min, y_min, x_max, y_max: Rectangle bounds
            
        Returns:
            List of (ix, iy) grid cell coordinates
        """
        # Clamp to canvas bounds
        x_min = max(0.0, min(self.W, x_min))
        y_min = max(0.0, min(self.H, y_min))
        x_max = max(0.0, min(self.W, x_max))
        y_max = max(0.0, min(self.H, y_max))
        
        # Grid coordinates
        ix_min = int(math.floor(x_min / self.cell_size))
        iy_min = int(math.floor(y_min / self.cell_size))
        ix_max = int(math.floor(x_max / self.cell_size))
        iy_max = int(math.floor(y_max / self.cell_size))
        
        # Clamp to grid bounds
        ix_min = max(0, min(self.grid_w - 1, ix_min))
        iy_min = max(0, min(self.grid_h - 1, iy_min))
        ix_max = max(0, min(self.grid_w - 1, ix_max))
        iy_max = max(0, min(self.grid_h - 1, iy_max))
        
        cells = []
        for iy in range(iy_min, iy_max + 1):
            for ix in range(ix_min, ix_max + 1):
                cells.append((ix, iy))
        
        return cells
    
    def insert(self, rect_id: int, x: float, y: float, w: float, h: float):
        """
        Insert rectangle into spatial hash.
        
        Args:
            rect_id: Unique identifier for rectangle
            x, y: Center coordinates
            w, h: Width and height
        """
        # Convert center to AABB bounds
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        
        # Store rectangle data
        if rect_id >= len(self.rects_db):
            self.rects_db.extend([None] * (rect_id - len(self.rects_db) + 1))
        self.rects_db[rect_id] = (x_min, y_min, x_max, y_max)
        
        # Insert into all overlapping grid cells
        cells = self._get_grid_cells(x_min, y_min, x_max, y_max)
        for ix, iy in cells:
            self.grid[iy][ix].add(rect_id)

    def remove(self, rect_id: int):
        """Remove rectangle from spatial hash if it exists."""
        if rect_id >= len(self.rects_db):
            return
        rect = self.rects_db[rect_id]
        if rect is None:
            return

        x_min, y_min, x_max, y_max = rect
        cells = self._get_grid_cells(x_min, y_min, x_max, y_max)
        for ix, iy in cells:
            self.grid[iy][ix].discard(rect_id)
        self.rects_db[rect_id] = None
    
    def query(self, x: float, y: float, w: float, h: float) -> Set[int]:
        """
        Query rectangles that might overlap with given rectangle.
        
        Returns candidate rect_ids (may include false positives).
        Use overlaps_any() for precise checks.
        
        Args:
            x, y: Center coordinates
            w, h: Width and height
            
        Returns:
            Set of candidate rect_ids
        """
        # Convert center to AABB bounds
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        
        # Get overlapping grid cells
        cells = self._get_grid_cells(x_min, y_min, x_max, y_max)
        
        # Collect all candidate rect_ids
        candidates = set()
        for ix, iy in cells:
            candidates.update(self.grid[iy][ix])
        
        return candidates
    
    def _rects_intersect(self, 
                        x1_min: float, y1_min: float, x1_max: float, y1_max: float,
                        x2_min: float, y2_min: float, x2_max: float, y2_max: float) -> bool:
        """
        Check if two axis-aligned rectangles intersect.
        
        Two rectangles intersect if they overlap in BOTH x AND y.
        """
        return (x1_min < x2_max and x2_min < x1_max and
                y1_min < y2_max and y2_min < y1_max)
    
    def overlaps_any(self, x: float, y: float, w: float, h: float, 
                    exclude_rect_ids: Set[int] = None) -> bool:
        """
        Check if rectangle overlaps any existing rectangle.
        
        Args:
            x, y: Center coordinates
            w, h: Width and height
            exclude_rect_ids: Set of rect_ids to exclude from check (e.g., self)
            
        Returns:
            True if overlaps any, False otherwise
        """
        if exclude_rect_ids is None:
            exclude_rect_ids = set()
        
        # Convert center to AABB bounds
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        
        # Query candidates
        candidates = self.query(x, y, w, h)
        
        # Precise intersection check
        for rect_id in candidates:
            if rect_id in exclude_rect_ids:
                continue
            
            if rect_id >= len(self.rects_db) or self.rects_db[rect_id] is None:
                continue
            
            r_x_min, r_y_min, r_x_max, r_y_max = self.rects_db[rect_id]
            
            if self._rects_intersect(x_min, y_min, x_max, y_max,
                                   r_x_min, r_y_min, r_x_max, r_y_max):
                return True
        
        return False
    
    def clear(self):
        """Clear all rectangles from spatial hash."""
        self.grid = [
            [set() for _ in range(self.grid_w)]
            for _ in range(self.grid_h)
        ]
        self.rects_db = []
