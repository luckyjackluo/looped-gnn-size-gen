"""Configuration system for model ablation components."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class RandomFourierEncodingConfig:
    """Configuration for Random Fourier Features (RFF) encoding."""
    
    # Whether to use RFF
    enabled: bool = False
    
    # Number of random features (output dimension = 2 * num_features)
    num_features: int = 64
    
    # Scale parameter (bandwidth) for RFF
    scale: float = 1.0
    
    # Whether to learn the scale parameter
    learnable_scale: bool = False


@dataclass
class RelativePositionalEncodingConfig:
    """Configuration for relative positional encoding based on coordinates."""
    
    # Whether to use relative positional encoding
    enabled: bool = False
    
    # Dimension of positional encoding
    encoding_dim: int = 64
    
    # Maximum distance for relative encoding (normalization)
    max_distance: Optional[float] = None
    
    # Whether to use learnable embeddings
    learnable: bool = True


@dataclass
class GlobalModuleConfig:
    """
    Configuration for pluggable global module system.
    
    The global module processes features after GNN blocks to add global context.
    It can be attention-based, MLP-based, pooling-based, etc.
    """
    
    # Whether to use a global module
    enabled: bool = False
    
    # Type of global module: "attention", "mlp", "identity", "graph_pooling", "hash_grid", "perceiver", or custom registered name
    module_type: str = "attention"
    
    # Placement: "after_gnn" (after all GNN blocks) or "interleaved" (one per GNN block)
    placement: Literal["after_gnn", "interleaved"] = "interleaved"
    
    # Module-specific parameters (passed to the module constructor)
    # For "attention" module:
    attention_type: Literal["transformer", "self_attention", "local_windows", "knn_local"] = "transformer"
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True
    ff_dim: Optional[int] = None  # If None, defaults to 4 * hidden_dim
    activation: str = "gelu"
    apply_to_nets: bool = False
    window_size: int = 128
    global_layers: int = 1
    att_gnn_style: bool = True  # If True, use AttGNNBlock style (Attention → LayerNorm → Linear)
                                # If False, use standard transformer style (pre-norm)
    
    # For "mlp" module:
    mlp_num_layers: int = 2
    mlp_activation: str = "relu"
    
    # For "graph_pooling" module:
    pool_type: str = "mean"  # "mean", "max", "sum", or "mean_max"
    pool_num_layers: int = 1
    
    # For "hash_grid" module (multi-scale hierarchical attention):
    hash_grid_levels: int = 2
    hash_grid_base_cell_size: Optional[float] = None  # If None, use density-controlled (recommended)
    hash_grid_cell_size_multiplier: Optional[float] = None  # If None, computed automatically
    hash_grid_dim: int = 64
    hash_grid_window_sizes: list = field(default_factory=lambda: [9, 9])
    hash_grid_shifted: bool = True
    hash_grid_ff_num_layers: int = 1
    hash_grid_ff_size_factor: int = 2
    hash_grid_rehash_interval: int = 1
    hash_grid_att_implementation: str = "default"  # "default", "flash", "performer"
    hash_grid_target_occupancy: float = 4.0  # Target nodes per cell for finest level (density control)
    hash_grid_target_graph_size: int = 1000  # Target graph size for coarser level computation
    hash_grid_num_attention_layers: int = 1  # Number of attention layers per scale (for curriculum learning)
    
    # For "perceiver" module (efficient cross-attention pattern):
    perceiver_num_global_tokens: int = 64  # Number of global tokens K (K << N for efficiency)
    perceiver_num_self_attn_layers: int = 1  # Number of self-attention layers on global tokens
    
    # For "gated_residual" and "residual_add" wrapper modules:
    # These wrap a base module with an extension module for curriculum learning
    base_module_type: Optional[str] = None  # Type of base module (e.g., "hash_grid", "attention")
    base_module_params: dict = field(default_factory=dict)  # Parameters for base module
    extension_module_type: Optional[str] = None  # Type of extension module
    extension_module_params: dict = field(default_factory=dict)  # Parameters for extension module
    gate_init: float = 0.01  # Initial gate value (for gated_residual)
    gate_learnable: bool = True  # Whether gate is learnable
    gate_schedule: Optional[str] = None  # Gate schedule (None, "linear", "cosine")
    gate_max: float = 1.0  # Maximum gate value
    alpha: float = 0.1  # Fixed weight for residual_add
    
    # Additional custom parameters (for future modules)
    custom_params: dict = field(default_factory=dict)


@dataclass
class GlobalAttentionConfig:
    """
    Legacy configuration for backward compatibility.
    This is now a wrapper around GlobalModuleConfig.
    """
    
    # Whether to use global attention
    enabled: bool = False
    
    # Placement: "after_gnn" (after all GNN blocks) or "interleaved" (one per GNN block)
    placement: Literal["after_gnn", "interleaved"] = "after_gnn"
    
    # Type of attention: "transformer", "self_attention", "local_windows", or "multi_scale_cls"
    attention_type: Literal["transformer", "self_attention", "local_windows", "multi_scale_cls"] = "transformer"
    
    # Number of attention heads
    num_heads: int = 8
    
    # Number of transformer layers
    num_layers: int = 2
    
    # Dropout rate
    dropout: float = 0.1
    
    # Whether to use layer normalization
    use_layer_norm: bool = True
    
    # Feed-forward dimension (for transformer)
    ff_dim: Optional[int] = None  # If None, defaults to 4 * hidden_dim
    
    # Activation function
    activation: str = "gelu"
    
    # Whether to apply attention to nets (disabled by default for memory efficiency)
    # Nets are usually much larger than instances, so applying attention to them
    # can cause OOM. Set to False to only apply attention to instances.
    apply_to_nets: bool = False
    
    # Window size for local_windows attention (maximum nodes per window)
    window_size: int = 128

    # Number of global layers for multi_scale_cls (global CLS over window CLS)
    global_layers: int = 1


@dataclass
class RobustEncodeConfig:
    """Configuration for robust coordinate encoding."""

    # Whether to use robust encoding (canonicalization + deterministic Fourier)
    enabled: bool = False

    # Number of frequency bands (K). Output dim = 4 * num_freq_bands
    num_freq_bands: int = 8

    # Whether to clamp normalized coordinates to [-1, 1] (default False to preserve coordinate info)
    clamp_norm: bool = False


@dataclass
class AugmentedEmbeddingConfig:
    """Configuration for pre-trained model augmented embeddings."""

    # Whether to use augmented embeddings from pre-trained model
    enabled: bool = False

    # Path to pre-trained model checkpoint
    pretrained_model_path: Optional[str] = None

    # Embedding dimension from pre-trained model (e.g., 64 for DiGCN)
    pretrained_emb_dim: int = 64

    # How to combine embeddings: "concat" (concatenate), "add" (addition), or "project_add" (project then add)
    combine_method: Literal["concat", "add", "project_add"] = "concat"

    # Whether to freeze the pre-trained model (recommended: True)
    freeze_pretrained: bool = True

    # Whether to augment instance embeddings
    augment_inst: bool = True

    # Whether to augment net embeddings
    augment_net: bool = True

    # Pre-trained model architecture config (for loading the model)
    pretrained_num_layers: int = 4
    pretrained_gnn_type: str = "digcn"
    pretrained_use_vn: bool = False
    pretrained_aggr: str = "max"


@dataclass
class AttGNNConfig:
    """Configuration for AttGNN (Attention-based GNN) model."""
    
    # Hidden size (typically matches model hidden_dim)
    hidden_size: int = 256
    
    # Hidden node features per block (list of dimensions)
    hidden_node_features: list = field(default_factory=lambda: [256, 256, 256, 256])
    
    # Attention node features per block (list of dimensions, 0 to disable attention in that block)
    attention_node_features: list = field(default_factory=lambda: [256, 256, 256, 256])
    
    # Number of GNN layers per block
    layers_per_block: int = 2
    
    # Convolution layer parameters
    conv_params: dict = field(default_factory=lambda: {"layer_type": "gcn"})
    
    # MLP parameters (between blocks)
    mlp_num_layers: int = 0
    mlp_size_factor: int = 0
    
    # Input encoding dimension (for positional encoding)
    input_encoding_dim: int = 0
    
    # Whether to use direct attention input (use x_skip instead of x)
    dir_att_input: bool = False
    
    # Mask key for masking nodes (optional)
    mask_key: Optional[str] = None
    
    # Dropout rate
    dropout: float = 0.0
    
    # Attention parameters
    num_heads: int = 4
    ff_num_layers: int = 2
    ff_size_factor: int = 2
    att_implementation: Literal["default", "flash", "performer"] = "default"
    
    # Conditioning node features dimension (from data.x: cell_type, w, h, area, is_macro, is_port, pin_cap)
    cond_node_features: int = 7
    
    # Hash-grid hierarchical attention
    enabled_hash_grid: bool = False
    hash_grid_levels: int = 2
    hash_grid_base_cell: float = 8.0
    hash_grid_cell_mult: float = 2.0
    hash_grid_dim: int = 64
    hash_grid_window_sizes: list = field(default_factory=lambda: [9, 9])
    hash_grid_shifted: bool = True
    hash_grid_ff_layers: int = 1
    hash_grid_ff_factor: int = 2
    hash_grid_rehash_interval: int = 1


@dataclass
class ModelAblationConfig:
    """Main configuration for model ablation components."""
    
    # Hidden dimension (must match model hidden_dim)
    hidden_dim: int = 256
    
    # Optional input coordinates x_t (for multi-step refinement)
    # If None, assumes one-step regression
    use_x_init: bool = False
    
    # Dimension of x_t coordinates (typically 2 for x, y)
    x_init_dim: int = 2
    
    # Random Fourier encoding configuration
    rff: RandomFourierEncodingConfig = field(default_factory=RandomFourierEncodingConfig)
    
    # Relative positional encoding configuration
    rel_pos: RelativePositionalEncodingConfig = field(
        default_factory=RelativePositionalEncodingConfig
    )
    
    # Robust coordinate encoding configuration (canonicalization + deterministic Fourier)
    robust_encode: RobustEncodeConfig = field(default_factory=RobustEncodeConfig)
    
    # Global module configuration (pluggable system)
    global_module: GlobalModuleConfig = field(default_factory=GlobalModuleConfig)
    
    # Legacy: Global attention configuration (for backward compatibility)
    # This is automatically converted to global_module if global_module is not explicitly set
    global_attention: GlobalAttentionConfig = field(
        default_factory=GlobalAttentionConfig
    )
    
    # AttGNN configuration (used when gnn_type="att_gnn")
    att_gnn: AttGNNConfig = field(default_factory=AttGNNConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_x_init:
            # If using x_init, at least one encoding should be enabled
            if not (self.rff.enabled or self.rel_pos.enabled or self.robust_encode.enabled):
                raise ValueError(
                    "If use_x_init=True, at least one of rff.enabled, "
                    "rel_pos.enabled, or robust_encode.enabled must be True"
                )
        
        # Validate global module configuration
        if self.global_module.enabled:
            if self.global_module.placement not in ["after_gnn", "interleaved"]:
                raise ValueError(
                    f"Unknown placement: {self.global_module.placement}. "
                    f"Must be 'after_gnn' or 'interleaved'"
                )
        
        # Backward compatibility: convert global_attention to global_module if needed
        if self.global_attention.enabled and not self.global_module.enabled:
            # Auto-convert legacy config
            self.global_module.enabled = True
            self.global_module.module_type = "attention"
            self.global_module.placement = self.global_attention.placement
            self.global_module.attention_type = self.global_attention.attention_type
            self.global_module.num_heads = self.global_attention.num_heads
            self.global_module.num_layers = self.global_attention.num_layers
            self.global_module.dropout = self.global_attention.dropout
            self.global_module.use_layer_norm = self.global_attention.use_layer_norm
            self.global_module.ff_dim = self.global_attention.ff_dim
            self.global_module.activation = self.global_attention.activation
            self.global_module.apply_to_nets = self.global_attention.apply_to_nets
            self.global_module.window_size = self.global_attention.window_size
            self.global_module.global_layers = self.global_attention.global_layers
        
        # Validate legacy global_attention if still used
        if self.global_attention.enabled:
            if self.global_attention.attention_type not in ["transformer", "self_attention", "local_windows", "multi_scale_cls"]:
                raise ValueError(
                    f"Unknown attention_type: {self.global_attention.attention_type}"
                )
            if self.global_attention.placement not in ["after_gnn", "interleaved"]:
                raise ValueError(
                    f"Unknown placement: {self.global_attention.placement}. "
                    f"Must be 'after_gnn' or 'interleaved'"
                )

