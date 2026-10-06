"""Configuration for placement-first generation."""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class PlacementConfig:
    """Configuration for placement-first generation."""
    
    # Stop density (fraction of chip area to fill)
    stop_density: float = 0.6  # 60% utilization
    
    # Or use distribution for stop density
    stop_density_dist: Optional[Dict[str, Any]] = None  # e.g., {"type": "uniform", "low": 0.5, "high": 0.7}
    
    # Placement attempts
    max_attempts_per_instance: int = 100
    
    # Aspect ratio distribution
    aspect_ratio_dist: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.5,
        "high": 2.0
    })
    
    # Instance size distribution (long dimension)
    instance_size_dist: Dict[str, Any] = field(default_factory=lambda: {
        "type": "lognormal",
        "mean": 0.0,
        "std": 0.5
    })
    
    # Terminal generation
    num_terminals_dist: Dict[str, Any] = field(default_factory=lambda: {
        "type": "poisson",
        "rate": 3.0
    })
    
    # Edge generation (distance-based probability)
    edge_dist: Dict[str, Any] = field(default_factory=lambda: {
        "type": "exponential",
        "scale": 0.1,
        "prob_multiplier_factor": 1.0,
        "prob_multiplier_exp": 0.0,
        "prob_clip": 1.0,
        "global_scale": True
    })
    
    # Source terminal distribution
    source_terminal_dist: Dict[str, Any] = field(default_factory=lambda: {
        "type": "bernoulli",
        "probs": 0.3  # 30% of terminals are sources
    })
    
    # Terminal placement
    interior_terminals_dist: Optional[Dict[str, Any]] = None  # None = all on boundary
    interior_terminals_loc: str = "uniform"  # "uniform" or "center"
    
    # Distance metric
    distance_norm_order: int = 1  # 1 = L1, 2 = L2, "inf" = L∞
    
    # Performance
    placement_device: Optional[str] = None  # None = CPU, "cuda" = GPU
    edge_device: Optional[str] = None  # None = CPU, "cuda" = GPU
    edge_chunk_size: Optional[int] = None  # None = no chunking, int = chunk size
    
    # Chip bounds (normalized coordinates)
    chip_min: float = -1.0
    chip_max: float = 1.0
    
    # Zero edge attributes (for testing)
    zero_edge_attr: bool = False

