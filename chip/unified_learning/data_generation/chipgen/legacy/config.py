"""Configuration for synthetic netlist generation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SizeConfig:
    """Size bin configuration with exact target proportions."""

    # Size bins: [min, max) for N_inst
    bins: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1000, 5000),      # 60%
        (5000, 20000),     # 20%
        (20000, 50000),    # 10%
        (50000, 70000),    # 5%
        (70000, 100000),   # 5%
    ])

    # Weights for each bin (must sum to 1.0)
    weights: List[float] = field(default_factory=lambda: [0.60, 0.20, 0.10, 0.05, 0.05])

    bin_names: List[str] = field(default_factory=lambda: [
        "1k-5k", "5k-20k", "20k-50k", "50k-70k", "70k-100k"
    ])


@dataclass
class InstanceConfig:
    """Instance (cell/port) feature configuration.

    Target area ratios (typical scenarios):
    1. Classic stdcell-dominant: 0-10% macro, 90-100% stdcell (ISPD benchmarks)
    2. Modern SoC: 20-40% macro, 60-80% stdcell (CPU/accelerator blocks) ← DEFAULT
    3. Memory-heavy: 50-80% macro, 20-50% stdcell (AI/GPU/NPU designs)

    Current settings target scenario 2 (modern SoC).
    """

    # Port settings
    port_fraction: float = 0.02  # 2% of instances are ports

    # Macro settings (among non-ports)
    # Adjusted to achieve ~25-35% macro area ratio for modern SoC designs
    macro_fraction: float = 0.015  # 1.5% of non-ports are macros

    # Standard cell size ranges (width, height in microns)
    stdcell_w_range: Tuple[float, float] = (1.0, 5.0)
    stdcell_h_range: Tuple[float, float] = (1.0, 3.0)

    # Macro size ranges (realistic for modern designs: ~10-100x larger than stdcells)
    # Old values (50-300)² were too large, giving 99%+ macro area
    macro_w_range: Tuple[float, float] = (8.0, 30.0)
    macro_h_range: Tuple[float, float] = (8.0, 30.0)

    # Pin capacity (avg pins per instance)
    pin_cap_range: Tuple[float, float] = (2.0, 12.0)

    # Whether to generate positions for ports
    generate_port_positions: bool = False
    port_boundary_margin: float = 10.0


@dataclass
class HierarchyConfig:
    """Hierarchical clustering configuration."""

    # Branching factor for hierarchy (cluster size growth)
    k: int = 4  # Balance between locality and global reach

    # Max hierarchy levels (level 0 = local, higher = more global)
    max_level: int = 5  # 4^6 = 4096 instances at max level

    # Probability distribution over levels (geometric decay favors local)
    # p(level=l) ∝ decay^l
    # Higher decay = more uniform across levels (more global connectivity)
    level_decay: float = 0.70  # Balanced for Rent's exponent ~0.6

    # Fraction of edges to sample globally (ignoring hierarchy)
    # Higher values improve Rent's exponent but reduce realism
    # Real circuits are mostly local (should be ~0.1-0.2)
    global_edge_fraction: float = 0.20  # 20% global, 80% hierarchical (realistic)

    # Random seed for permutation (set to None for random)
    perm_seed: int = 42


@dataclass
class DegreeConfig:
    """Net degree distribution configuration."""

    # Target average instance degree (pins per instance)
    avg_inst_degree: float = 5.0

    # Degree mixture proportions (small, medium, large)
    p_small: float = 0.50  # Reduced from 0.70
    p_medium: float = 0.45  # Increased from 0.299
    p_huge: float = 0.05  # Increased from 0.001 for better Rent's rule

    # Degree ranges for each category
    small_deg_range: Tuple[int, int] = (2, 4)
    medium_deg_range: Tuple[int, int] = (5, 15)
    huge_deg_range: Tuple[int, int] = (16, 40)

    # Cap huge degree based on N_inst to avoid memory issues
    huge_deg_cap_factor: float = 0.3  # max_huge_deg = factor * N_inst


@dataclass
class PinConfig:
    """Pin offset and driver configuration."""

    # Probability of pin on left/right edge (vs top/bottom) for stdcells
    stdcell_lr_prob: float = 0.8

    # Edge margin for pin placement (as fraction of w/h)
    edge_margin: float = 0.1

    # For huge nets, bias driver selection toward ports
    huge_net_driver_port_bias: float = 0.7  # prob of choosing port if available


@dataclass
class NetlistConfig:
    """Main configuration container."""

    size: SizeConfig = field(default_factory=SizeConfig)
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    degree: DegreeConfig = field(default_factory=DegreeConfig)
    pin: PinConfig = field(default_factory=PinConfig)

    # Device
    device: str = "cuda"
    
    # Generation mode: "netlist_first" (default) or "placement_first"
    generation_mode: str = "netlist_first"
    
    # Placement config (only used if generation_mode == "placement_first")
    placement: Optional["PlacementConfig"] = None

    def __post_init__(self):
        """Validate configuration."""
        assert abs(sum(self.size.weights) - 1.0) < 1e-6, "Size weights must sum to 1.0"
        assert len(self.size.bins) == len(self.size.weights), "Bins and weights must match"
        assert abs(self.degree.p_small + self.degree.p_medium + self.degree.p_huge - 1.0) < 1e-6, \
            "Degree mixture must sum to 1.0"
        assert self.generation_mode in ["netlist_first", "placement_first"], \
            "generation_mode must be 'netlist_first' or 'placement_first'"
        
        # Import here to avoid circular import
        if self.generation_mode == "placement_first" and self.placement is None:
            from .placement_config import PlacementConfig
            self.placement = PlacementConfig()
