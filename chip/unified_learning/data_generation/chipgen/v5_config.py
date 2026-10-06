"""Configuration for V5 fast sparse synthetic algorithm."""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class V5Config:
    """Configuration for V5 algorithm."""
    # Pipeline mode
    # - "v5": full synthetic flow (std-cells + macros + ports)
    # - "packing_alg": simplified constraint-packing flow (std-cells + border ports, no macros)
    generation_pipeline: str = "v5"
    
    # Instance generation parameters (physical units in microns)
    max_instance: int = 50000
    # packing_alg only: fixed instance pool size (no max cap based on target nodes).
    # Chip size is computed from actual instance areas (stdcell-only strategy, no macros).
    packing_alg_instance_pool_size: int = 100000
    # packing_alg only: soft buffer factor for candidate stdcells.
    # Actual generation uses max(packing_alg_instance_pool_size, max_instance * factor),
    # so target_nodes does not become a hard cap on placed-node count.
    packing_alg_instance_buffer_multiplier: float = 2.0
    # packing_alg only: ports scale with estimated canvas perimeter (W + H), not instance count.
    # n_ports = clamp(round(packing_alg_port_perimeter_scale * (W + H)),
    #                 packing_alg_min_ports, packing_alg_max_ports)
    packing_alg_port_perimeter_scale: float = 0.35
    packing_alg_min_ports: int = 4
    packing_alg_max_ports: Optional[int] = None
    # packing_alg only: keep target average degree stable for learnability.
    packing_alg_target_degree_mean: float = 6.0
    packing_alg_target_degree_std: float = 1.0
    port_fraction: float = 0.02
    macro_fraction: float = 0.03  # More regular macros (non-super) for realistic designs
    stdcell_w_range: Tuple[float, float] = (0.5, 10.0)  # Wider range for more variation
    stdcell_h_range: Tuple[float, float] = (0.15, 1.5)  # Wider range with variable row-height
    stdcell_height_mean: float = 0.4  # Mean height for modern nodes (µm)
    stdcell_height_std: float = 0.15  # Larger variation for changing row-height (was 0.02)
    stdcell_site_width: float = 0.05  # Site width for quantization (µm)
    stdcell_row_height: Optional[float] = None  # When set, all std-cells share one physical row height
    packing_alg_stdcell_width_cap: Optional[float] = 4.0  # Fixed width cap to avoid N-dependent geometry support
    packing_alg_port_min_spacing: float = 1.25  # Minimum center-to-center spacing for balanced border ports
    macro_w_range: Tuple[float, float] = (8.0, 30.0)
    macro_h_range: Tuple[float, float] = (8.0, 30.0)
    
    # Chip/macro scaling vs target size (max_instance)
    # When set, chip dimensions are floored so small-N graphs keep a reasonable physical scale;
    # macro and super-macro sizes then scale with this effective chip size (and thus with N).
    min_chip_dim: Optional[float] = 15.0  # Minimum width/height (µm); reduced from 50 (50 forced n=100 to 50×50 µm)
    # Reference chip dimension (µm) at which regular macros use full macro_w_range/macro_h_range.
    # macro_size_scale = min(1.0, chip_size_est / macro_size_reference_chip_dim). Increase for larger macros on small chips.
    # Reduced from 80.0 to make macros scale more aggressively with graph size (smaller for small graphs)
    macro_size_reference_chip_dim: float = 120.0

    # Canvas sizing efficiency factor for structured grid placement.
    # Old value was 0.70 (calibrated for RSA-jammed random grids that jammed at ~20-25% density).
    # Structured row-by-row grids pack the canvas near-perfectly, so the bias is close to 1.0.
    # 0.97 accounts for minor boundary effects and macro keepout zone losses.
    grid_size_bias_factor: float = 0.97
    
    # Placement parameters
    stop_density: float = 0.6
    max_attempts_per_instance: int = 100
    target_density: float = 0.6
    # Generation attempts per requested graph.
    # We no longer reject graphs for being under-dense; keep this at 1 to accept
    # the first successfully generated graph instead of silently retrying.
    max_generation_attempts: int = 1
    # Deprecated / unused: density-based rejection was removed.
    min_density_ratio: float = 0.0
    max_density_rejection_retries: int = 1
    canvas_aspect_ratio_range: Tuple[float, float] = (0.4, 2.5)  # default range; diversity overrides per-graph
    
    # V5B1 Placement parameters
    # Macro placement
    macro_keepout: float = 2.0  # Keepout margin around macros for macro placement (microns) - maintained
    stdcell_macro_keepout: float = 0.2  # Keepout distance for std-cells from macros (microns) - reduced from 1.2
    # Reduced spacing between stdcells and macros, but macro-to-macro spacing remains at 2.0
    macro_border_bias: float = 0.3  # Probability to try border regions first
    macro_max_tries: int = 200  # Max attempts per macro
    macro_packer: str = "shelf"  # Packing strategy: "shelf" or "skyline"
    macro_placement_strategy: str = "hybrid"  # "periphery", "islands", or "hybrid" (most realistic)
    macro_island_fraction: float = 0.3  # Fraction of macros in islands (for hybrid/islands strategies)
    macro_island_size_range: Tuple[int, int] = (3, 8)  # Number of macros per island

    # Placement stochasticity controls.
    # All default to diverse/stochastic behavior. _sample_diversity_params() further randomizes these
    # per-graph when enable_diversity_randomization=True.
    macro_deterministic: bool = False  # False = stochastic macro placement (diverse across graphs)
    macro_island_layout: str = "grid"  # "random" or "grid" (grid = SRAM-bank-like islands)
    macro_island_grid_shuffle: bool = True  # Shuffle grid cell assignment so islands vary across samples
    macro_area_target_mode: str = "random"  # "random" = sample area target per graph from [min, max]
    stdcell_grid_shuffle: bool = True  # Shuffle grid fill order so density gradient varies per graph
    
    # Std-cell placement
    stdcell_placement_strategy: str = "simple_grid"  # "simple_grid" or "jittered"
    enable_emergency_placement: bool = True  # If False, skip emergency placement (failed instances excluded)
    bin_size: float = 8.0  # Occupancy grid bin size (microns)
    hash_cell_size: float = 3.0  # Spatial hash cell size (microns, ~median stdcell size)
    poisson_r: float = 2.5  # Base radius for Poisson/jitter (microns)
    poisson_max_attempts: int = 30  # Max attempts for std-cell placement
    stdcell_grid_spacing_factor_w: float = 0.97  # Instance width relative to grid width
    stdcell_grid_spacing_factor_h: float = 0.98  # Instance height relative to grid height
    local_legalize_iters: int = 2  # Number of legalization iterations
    local_shift_step: float = 1.25  # Step size for legalization moves (microns)
    
    # Terminal generation (Rent's Rule)
    num_terminals_binom_p: float = 0.5
    num_terminals_min_n: int = 2
    num_terminals_t: float = 1.0  # Scaling factor (lower = more terminals)
    num_terminals_p: float = 0.65  # Rent exponent
    
    # Edge generation
    target_degree_mean: float = 5.66  # Mean degree matching industrial netlists
    target_degree_std: float = 5.66  # Standard deviation allowing tail up to ~50 degrees
    edge_distance_scale: float = 15.0  # Scale for distance-based probability (microns) - reduced for more local connections (Rentian locality)
    source_terminal_prob: float = 0.3

    # Deterministic edge neighbor selection (Option A)
    # - "gumbel_topk": stochastic (current behavior; multiple graphs for same placement possible)
    # - "deterministic_topk": no RNG in neighbor selection; favors low-HPWL locality and stable graphs
    edge_neighbor_selection: str = "deterministic_topk"

    # Connectivity realism / robustness: deterministically bridge disconnected components
    # This encourages a dominant giant component (more realistic netlist-like connectivity)
    # while keeping HPWL small by choosing nearest bridges in physical space.
    bridge_components: bool = True
    max_component_bridges: int = 16  # cap (C-1) bridges to avoid pathological overhead
    bridge_search_radius: int = 4    # in spatial grid cells (radius=4 => 9x9 neighborhood)
    
    # Macro degree parameters (macros fan out to many std-cells)
    macro_degree_mean: float = 28.3  # Mean degree for macros (5x overall mean, ~30 connections typical)
    macro_degree_std: float = 22.64  # High variance for macros (4x overall std)
    macro_degree_min: int = 15  # Macros should have at least 15 connections
    macro_degree_max: int = 300  # Some macros can have very high fanout (200-300+)
    long_range_macro_prob: float = 0.0  # Disabled - long-range connections are rare in real netlists
    
    # Macro area target (as fraction of total canvas area)
    macro_area_target_min: float = 0.20  # Minimum macro area fraction (20%)
    macro_area_target_max: float = 0.35  # Maximum macro area fraction (35%)
    
    # Diversity: per-graph randomization of placement parameters (for model to learn diverse layouts)
    enable_diversity_randomization: bool = True  # Randomize params from graph to graph
    # Std-cell size diversity: sample from these ranges per graph
    stdcell_h_range_min: Tuple[float, float] = (0.1, 0.2)   # Min of (low, high) for h_range (wider range)
    stdcell_h_range_max: Tuple[float, float] = (1.0, 2.0)  # Max of (low, high) for h_range (allow much larger cells)
    stdcell_w_range_min: Tuple[float, float] = (0.3, 0.8)  # Min of (low, high) for w_range (wider range)
    stdcell_w_range_max: Tuple[float, float] = (5.0, 15.0) # Max of (low, high) for w_range (allow much wider cells)
    stdcell_height_std_range: Tuple[float, float] = (0.05, 0.25)  # Per-graph height variance (0.05=moderate, 0.25=high variance for row-height changes)
    # Macro area diversity: sample target from wider range (20% to 55% - "community" designs)
    macro_area_target_range: Tuple[float, float] = (0.15, 0.55)  # Min and max for macro_area_target
    macro_community_mode_prob: float = 0.25  # Probability of "community" layout (macros clustered, up to half chip)
    # Port placement diversity: not always uniform on borders
    port_placement_uniform_prob: float = 0.5   # Prob of uniform border placement; else scattered/clustered
    
    # Macro size distribution (super-macros vs regular macros)
    super_macro_fraction: float = 0.001  # Base fraction of instances that are super-macros (0.1%)
    super_macro_min_count: int = 10  # Minimum number of super-macros (ensures dozens at large N)
    super_macro_max_fraction: float = 0.01  # Maximum fraction (1% of instances)
    super_macro_size_min_frac: float = 0.03  # Super-macro size: 3% of chip dimension (proportional scaling, can be very small)
    super_macro_size_max_frac: float = 0.15  # Super-macro size: 15% of chip dimension
    super_macro_abs_min_size: float = 5.0   # Absolute minimum size (µm) - much smaller, allows very small super-macros on small chips
    super_macro_abs_max_size: float = 500.0  # Absolute maximum size (µm) for small chips (scales with chip size for large chips)
    super_macro_abs_max_frac: float = 0.20  # Absolute maximum as fraction of chip size (20% for very large chips)
    super_macro_only_prob: float = 0.15  # Probability of super-macro-only designs (no regular macros, leaving small space for stdcells)
    # Legacy (deprecated): super_macro_w_range and super_macro_h_range are now calculated from chip size
    
    # Connectivity clustering
    enable_clustering: bool = True  # Enable hierarchical clustering for std-cells
    cluster_radius: float = 50.0  # Radius for std-cell clusters (microns)
    intra_cluster_boost: float = 2.5  # Boost factor for connections within same cluster

    # Local connectivity (HPWL-friendly): std-cell -> std-cell only within radius
    # NOTE: disabled by default because the current CPU implementation of the radius
    # candidate precomputation can be slow for large graphs. Distance decay + clustering
    # already encourage strong locality.
    enable_stdcell_local_radius: bool = False
    stdcell_local_radius: float = 50.0  # microns
    stdcell_local_bin_size: float = 50.0  # spatial bin size for neighbor lookup (microns)

    # Fast GPU hard radius cutoff (recommended): apply stdcell->stdcell radius mask on-GPU during edge sampling
    enable_stdcell_hard_radius_cuda: bool = True
    stdcell_hard_radius: float = 50.0  # microns
    stdcell_hard_radius_norm: int = 1  # 1 = L1 (HPWL-like), 2 = L2
    
    # Distance decay model (Rentian locality: heavy head + power-law tail)
    distance_decay_type: str = "hybrid"  # "exponential", "power_law", or "hybrid"
    distance_power_alpha: float = 1.5  # Power-law exponent (for hybrid/power_law)
    distance_crossover: float = 20.0  # Distance threshold for exponential->power-law (microns) - reduced for more local connections
    
    # Spatial grid for edge candidate gathering (smaller = more local)
    spatial_candidate_cell_size: float = 12.0  # Grid cell size for spatial candidate gathering (microns) - smaller for more local connections
    spatial_neighbor_radius: int = 2  # Neighbor search radius for spatial grid (2 = 5x5 cells) - expanded for Rentian tail
    
    # Macro pin placement
    macro_pin_on_perimeter: bool = True  # Place macro pins on perimeter (not random)
    
    # Macro fanout pattern
    macro_output_fraction: float = 0.7  # Fraction of macro terminals that are outputs (fanout to std-cells)
    
    # Port connectivity
    port_to_macro_prob: float = 0.4  # Port connects to macro (40%)
    port_to_cluster_prob: float = 0.4  # Port connects to std-cell cluster (40%)
    port_to_port_prob: float = 0.2  # Port connects to other port (20%)
    
    # Performance
    placement_device: str = "cpu"  # CPU is faster due to less sync overhead
    edge_device: str = "cpu"
    batch_size: int = 10000  # Batch size for edge sampling (increase for larger RAM usage, default: 10000)

    # ── DreamPlace stdcell optimizer ──────────────────────────────────────
    # Post-processing gradient optimizer applied after V5.sample() returns.
    # Macros and ports remain fixed; only stdcell positions are optimized.
    # All fields default to off/conservative so existing configs are unaffected.
    dreamplace_enabled: bool = False
    # Gradient descent
    dreamplace_num_iterations: int = 300
    dreamplace_learning_rate: float = 0.01
    dreamplace_optimizer_type: str = "adam"       # "adam" | "sgd"
    # WA-HPWL smoothing
    dreamplace_wa_gamma: float = 1.0              # smaller = sharper / closer to true HPWL
    # Density penalty
    dreamplace_num_bins_x: int = 32
    dreamplace_num_bins_y: int = 32
    dreamplace_density_target: float = 0.7        # per-bin density target
    dreamplace_lambda_density_max: float = 10.0   # hard cap on density penalty weight
    dreamplace_lambda_growth_rate: float = 0.02   # exponential lambda growth per iter
    dreamplace_warmup_iters: int = 50             # HPWL-only iters before density penalty
    # Boundary penalty
    dreamplace_lambda_boundary: float = 1.0
    # Gradient clipping
    dreamplace_grad_clip_norm: float = 5.0        # 0 = disabled
    # Legalization
    dreamplace_legalize: bool = True
    # Fallback
    dreamplace_fallback_on_nan: bool = True
    dreamplace_fallback_hpwl_ratio: float = 2.0   # revert if HPWL degrades > 2x
    # Device for optimizer (independent of placement_device)
    dreamplace_device: str = "auto"               # "auto" = GPU if available, else CPU

    # Output
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "generation_pipeline": self.generation_pipeline,
            "max_instance": self.max_instance,
            "packing_alg_instance_pool_size": self.packing_alg_instance_pool_size,
            "packing_alg_instance_buffer_multiplier": self.packing_alg_instance_buffer_multiplier,
            "packing_alg_port_perimeter_scale": self.packing_alg_port_perimeter_scale,
            "packing_alg_min_ports": self.packing_alg_min_ports,
            "packing_alg_max_ports": self.packing_alg_max_ports,
            "packing_alg_target_degree_mean": self.packing_alg_target_degree_mean,
            "packing_alg_target_degree_std": self.packing_alg_target_degree_std,
            "port_fraction": self.port_fraction,
            "macro_fraction": self.macro_fraction,
            "stdcell_w_range": list(self.stdcell_w_range),
            "stdcell_h_range": list(self.stdcell_h_range),
            "stdcell_row_height": self.stdcell_row_height,
            "packing_alg_stdcell_width_cap": self.packing_alg_stdcell_width_cap,
            "packing_alg_port_min_spacing": self.packing_alg_port_min_spacing,
            "macro_w_range": list(self.macro_w_range),
            "macro_h_range": list(self.macro_h_range),
            "min_chip_dim": self.min_chip_dim,
            "macro_size_reference_chip_dim": self.macro_size_reference_chip_dim,
            "stop_density": self.stop_density,
            "max_attempts_per_instance": self.max_attempts_per_instance,
            "target_density": self.target_density,
            "max_generation_attempts": self.max_generation_attempts,
            "min_density_ratio": self.min_density_ratio,
            "max_density_rejection_retries": self.max_density_rejection_retries,
            "canvas_aspect_ratio_range": list(self.canvas_aspect_ratio_range),
            "num_terminals_binom_p": self.num_terminals_binom_p,
            "num_terminals_min_n": self.num_terminals_min_n,
            "num_terminals_t": self.num_terminals_t,
            "num_terminals_p": self.num_terminals_p,
            "target_degree_mean": self.target_degree_mean,
            "target_degree_std": self.target_degree_std,
            "edge_distance_scale": self.edge_distance_scale,
            "source_terminal_prob": self.source_terminal_prob,
            "edge_neighbor_selection": self.edge_neighbor_selection,
            "bridge_components": self.bridge_components,
            "max_component_bridges": self.max_component_bridges,
            "bridge_search_radius": self.bridge_search_radius,
            "long_range_macro_prob": self.long_range_macro_prob,
            "macro_area_target_min": self.macro_area_target_min,
            "macro_area_target_max": self.macro_area_target_max,
            "macro_deterministic": self.macro_deterministic,
            "macro_island_layout": self.macro_island_layout,
            "macro_island_grid_shuffle": self.macro_island_grid_shuffle,
            "macro_area_target_mode": self.macro_area_target_mode,
            "stdcell_grid_shuffle": self.stdcell_grid_shuffle,
            "super_macro_fraction": self.super_macro_fraction,
            "super_macro_min_count": self.super_macro_min_count,
            "super_macro_max_fraction": self.super_macro_max_fraction,
            "super_macro_size_min_frac": self.super_macro_size_min_frac,
            "super_macro_size_max_frac": self.super_macro_size_max_frac,
            "super_macro_abs_min_size": self.super_macro_abs_min_size,
            "super_macro_abs_max_size": self.super_macro_abs_max_size,
            "macro_degree_mean": self.macro_degree_mean,
            "macro_degree_std": self.macro_degree_std,
            "macro_degree_min": self.macro_degree_min,
            "macro_degree_max": self.macro_degree_max,
            "enable_clustering": self.enable_clustering,
            "cluster_radius": self.cluster_radius,
            "intra_cluster_boost": self.intra_cluster_boost,
            "enable_stdcell_local_radius": self.enable_stdcell_local_radius,
            "stdcell_local_radius": self.stdcell_local_radius,
            "stdcell_local_bin_size": self.stdcell_local_bin_size,
            "enable_stdcell_hard_radius_cuda": self.enable_stdcell_hard_radius_cuda,
            "stdcell_hard_radius": self.stdcell_hard_radius,
            "stdcell_hard_radius_norm": self.stdcell_hard_radius_norm,
            "distance_decay_type": self.distance_decay_type,
            "distance_power_alpha": self.distance_power_alpha,
            "distance_crossover": self.distance_crossover,
            "spatial_candidate_cell_size": self.spatial_candidate_cell_size,
            "spatial_neighbor_radius": self.spatial_neighbor_radius,
            "macro_pin_on_perimeter": self.macro_pin_on_perimeter,
            "macro_output_fraction": self.macro_output_fraction,
            "port_to_macro_prob": self.port_to_macro_prob,
            "port_to_cluster_prob": self.port_to_cluster_prob,
            "port_to_port_prob": self.port_to_port_prob,
            "placement_device": self.placement_device,
            "edge_device": self.edge_device,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "enable_diversity_randomization": getattr(self, 'enable_diversity_randomization', True),
            "macro_keepout": self.macro_keepout,
            "macro_border_bias": self.macro_border_bias,
            "macro_max_tries": self.macro_max_tries,
            "macro_packer": self.macro_packer,
            "bin_size": self.bin_size,
            "hash_cell_size": self.hash_cell_size,
            "poisson_r": self.poisson_r,
            "poisson_max_attempts": self.poisson_max_attempts,
            "local_legalize_iters": self.local_legalize_iters,
            "local_shift_step": self.local_shift_step,
            "stdcell_placement_strategy": self.stdcell_placement_strategy,
            "enable_emergency_placement": self.enable_emergency_placement,
            # DreamPlace optimizer
            "dreamplace_enabled": self.dreamplace_enabled,
            "dreamplace_num_iterations": self.dreamplace_num_iterations,
            "dreamplace_learning_rate": self.dreamplace_learning_rate,
            "dreamplace_optimizer_type": self.dreamplace_optimizer_type,
            "dreamplace_wa_gamma": self.dreamplace_wa_gamma,
            "dreamplace_num_bins_x": self.dreamplace_num_bins_x,
            "dreamplace_num_bins_y": self.dreamplace_num_bins_y,
            "dreamplace_density_target": self.dreamplace_density_target,
            "dreamplace_lambda_density_max": self.dreamplace_lambda_density_max,
            "dreamplace_lambda_growth_rate": self.dreamplace_lambda_growth_rate,
            "dreamplace_warmup_iters": self.dreamplace_warmup_iters,
            "dreamplace_lambda_boundary": self.dreamplace_lambda_boundary,
            "dreamplace_grad_clip_norm": self.dreamplace_grad_clip_norm,
            "dreamplace_legalize": self.dreamplace_legalize,
            "dreamplace_fallback_on_nan": self.dreamplace_fallback_on_nan,
            "dreamplace_fallback_hpwl_ratio": self.dreamplace_fallback_hpwl_ratio,
            "dreamplace_device": self.dreamplace_device,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V5Config":
        """Create config from dictionary."""
        # Convert lists back to tuples
        if "stdcell_w_range" in d and isinstance(d["stdcell_w_range"], list):
            d["stdcell_w_range"] = tuple(d["stdcell_w_range"])
        if "stdcell_h_range" in d and isinstance(d["stdcell_h_range"], list):
            d["stdcell_h_range"] = tuple(d["stdcell_h_range"])
        if "macro_w_range" in d and isinstance(d["macro_w_range"], list):
            d["macro_w_range"] = tuple(d["macro_w_range"])
        if "macro_h_range" in d and isinstance(d["macro_h_range"], list):
            d["macro_h_range"] = tuple(d["macro_h_range"])
        if "canvas_aspect_ratio_range" in d and isinstance(d["canvas_aspect_ratio_range"], list):
            d["canvas_aspect_ratio_range"] = tuple(d["canvas_aspect_ratio_range"])
        # Legacy support: ignore removed/renamed fields
        d.pop("instance_buffer_multiplier", None)
        if "super_macro_w_range" in d:
            d.pop("super_macro_w_range")
        if "super_macro_h_range" in d:
            d.pop("super_macro_h_range")

        # Legacy defaults: newer diverse/stochastic behavior
        d.setdefault("generation_pipeline", "v5")
        d.setdefault("macro_deterministic", False)
        d.setdefault("macro_island_layout", "grid")
        d.setdefault("macro_island_grid_shuffle", True)
        d.setdefault("macro_area_target_mode", "random")
        d.setdefault("stdcell_grid_shuffle", True)
        # Default to deterministic graphs (transfer-aligned). Older dicts that don't specify this
        # should inherit deterministic behavior, not the legacy stochastic sampler.
        d.setdefault("edge_neighbor_selection", "deterministic_topk")
        d.setdefault("bridge_components", True)
        d.setdefault("max_component_bridges", 16)
        d.setdefault("bridge_search_radius", 4)
        # DreamPlace optimizer (all default to off for backwards compatibility)
        d.setdefault("dreamplace_enabled", False)
        d.setdefault("dreamplace_num_iterations", 300)
        d.setdefault("dreamplace_learning_rate", 0.01)
        d.setdefault("dreamplace_optimizer_type", "adam")
        d.setdefault("dreamplace_wa_gamma", 1.0)
        d.setdefault("dreamplace_num_bins_x", 32)
        d.setdefault("dreamplace_num_bins_y", 32)
        d.setdefault("dreamplace_density_target", 0.7)
        d.setdefault("dreamplace_lambda_density_max", 10.0)
        d.setdefault("dreamplace_lambda_growth_rate", 0.02)
        d.setdefault("dreamplace_warmup_iters", 50)
        d.setdefault("dreamplace_lambda_boundary", 1.0)
        d.setdefault("dreamplace_grad_clip_norm", 5.0)
        d.setdefault("dreamplace_legalize", True)
        d.setdefault("dreamplace_fallback_on_nan", True)
        d.setdefault("dreamplace_fallback_hpwl_ratio", 2.0)
        d.setdefault("dreamplace_device", "auto")

        return cls(**d)
