"""Unified Model Builder - config-driven architecture assembly."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple, List
from torch_geometric.data import HeteroData, Data

from .encoders.mlp import MLPEncoder, MLPDecoder


def _pad_flat_to_batch(x: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """Convert flat (N_total, F) to padded (B, max_nodes, F) for AttGNN-style (B, V, F) input."""
    device = x.device
    batch_size = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=batch_size).tolist()
    max_nodes = max(counts)
    cumsum = torch.cat([torch.tensor([0], device=device), torch.tensor(counts[:-1], device=device).cumsum(0)])
    arange_all = torch.arange(x.shape[0], device=device)
    batch_positions = arange_all - cumsum[batch]
    batched = torch.zeros(batch_size, max_nodes, x.shape[1], dtype=x.dtype, device=device)
    batched[batch, batch_positions] = x
    return batched, counts


def _unbatch_to_flat(batched_x: torch.Tensor, batch: torch.Tensor, counts: List[int]) -> torch.Tensor:
    """Convert (B, max_nodes, F) back to flat (N_total, F)."""
    device = batched_x.device
    counts_tensor = torch.tensor(counts, device=device)
    cumsum = torch.cat([torch.tensor([0], device=device), counts_tensor[:-1].cumsum(0)])
    arange_all = torch.arange(batch.shape[0], device=device)
    batch_positions = arange_all - cumsum[batch]
    return batched_x[batch, batch_positions]


_coord_logger = logging.getLogger(__name__)
_coord_warn_count = 0


def _validate_normalized_coords(coords: Optional[torch.Tensor], name: str, tol: float = 1.1) -> None:
    """
    Soft-check coordinate tensor in normalized space (approximately [-1, 1]).

    We intentionally do not raise on mild range overshoot during ODE sampling.
    Flow-matching trajectories are expected to stay in-range in theory, but
    numerical integration / velocity error can create small excursions.
    """
    if coords is None:
        return
    if not isinstance(coords, torch.Tensor) or coords.numel() == 0:
        return
    if not torch.isfinite(coords).all():
        raise ValueError(f"{name} contains NaN/Inf; expected finite normalized coordinates in [-1,1].")
    cmin = coords.min().item()
    cmax = coords.max().item()
    if cmin < -tol or cmax > tol:
        global _coord_warn_count
        if _coord_warn_count < 20:
            _coord_logger.warning(
                "%s outside expected normalized range: [%.4f, %.4f] (tol=%.2f). "
                "This may indicate integration drift or a normalization mismatch.",
                name, cmin, cmax, tol
            )
            _coord_warn_count += 1
from .encoders.fourier import RandomFourierEncoding, FourierFeatureEncoder
from .encoders.positional import RobustCoordinateEncoding
from .processors.mpnn.bipartite import InstNetBlock
from .processors.mpnn.unified_gnn import UnifiedGNNLayer
from .processors.global_modules.global_modules import create_global_module
from .processors.global_modules.att_gnn import AttGNN
from .processors.global_modules.residual_gated_adapter import ResidualGatedAdapter


class FeatureFiLM(nn.Module):
    """Per-feature affine modulation from a conditioning vector."""

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond is None:
            return x
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class GraphSizeConditioner(nn.Module):
    """Build node-wise conditioning from per-graph size statistics."""

    def __init__(self, cond_hidden_dim: int, output_dim: int):
        super().__init__()
        self.stats_encoder = nn.Sequential(
            nn.Linear(2, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, output_dim),
        )

    def forward(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        use_num_edges: bool = True,
    ) -> torch.Tensor:
        device = edge_index.device
        dtype = torch.float32

        if batch is None:
            num_nodes_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            num_edges_per_graph = torch.tensor([float(edge_index.shape[1])], device=device, dtype=dtype)
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            num_nodes_per_graph = torch.bincount(batch, minlength=num_graphs).to(device=device, dtype=dtype)
            if use_num_edges and edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                num_edges_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(device=device, dtype=dtype)
            else:
                num_edges_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        # Keep the same style as dataset graph stats: smooth log normalization.
        n_feat = torch.log1p(num_nodes_per_graph) / 9.21
        e_feat = torch.log1p(num_edges_per_graph) / 9.21
        graph_stats = torch.stack([n_feat, e_feat], dim=-1)
        graph_cond = self.stats_encoder(graph_stats)
        return graph_cond[node_batch]


def _gnn_layers_for_processor_block(
    gnn_blocks: nn.ModuleList,
    block_idx: int,
    *,
    layers_per_block: int,
    gnn_layer_ranges: Optional[List[Optional[Tuple[int, int]]]] = None,
    gnn_block_mask: Optional[List[bool]] = None,
) -> List[Tuple[int, nn.Module]]:
    """Return ``[(layer_idx, layer), ...]`` for one block-based processor block."""
    if gnn_layer_ranges is not None:
        entry = gnn_layer_ranges[block_idx]
        if entry is None:
            return []
        start, end = entry
        return [(i, gnn_blocks[i]) for i in range(start, end)]
    if gnn_block_mask is not None and not gnn_block_mask[block_idx]:
        return []
    start = block_idx * layers_per_block
    end = start + layers_per_block
    return [(i, gnn_blocks[i]) for i in range(start, end)]


def _build_gnn_layers(
    processor_cfg: Dict[str, Any],
    hidden_dim: int,
    time_embed_dim: Optional[int] = None
) -> Tuple[
    nn.ModuleList,
    Optional[nn.ModuleList],
    Optional[int],
    Optional[int],
    Optional[nn.ModuleList],
    Optional[List[Optional[Tuple[int, int]]]],
]:
    """
    Helper function to build GNN layers and optional global modules.
    Returns (gnn_blocks, global_modules, num_blocks, layers_per_block,
    gated_adapters, gnn_layer_ranges).

    For block-based processors with ``processor.gnn_blocks``, GNN layers are
    built only for the selected blocks (sparse layout). ``gnn_layer_ranges[b]``
    is ``(start, end)`` into ``gnn_blocks`` for block *b*, or ``None`` when
    that block has no GNN layers.
    
    This can be used for both encoder and processor to share the same architecture.
    """
    processor_type = processor_cfg['type']
    
    if processor_type == 'bipartite_gnn':
        # Bipartite GNN: inst-net-inst message passing (for HeteroData)
        layer_configs = processor_cfg.get('layer_configs', None)
        num_layers = processor_cfg.get('num_layers', 4)
        
        if layer_configs is not None:
            # Per-layer configuration
            if len(layer_configs) != num_layers:
                raise ValueError(f"layer_configs length ({len(layer_configs)}) must match num_layers ({num_layers})")
            gnn_blocks = nn.ModuleList([
                InstNetBlock(
                    inst_dim=hidden_dim,
                    net_dim=hidden_dim,
                    edge_dim=processor_cfg.get('edge_dim', hidden_dim),
                    heads=processor_cfg.get('heads', 4),
                    dropout=processor_cfg.get('dropout', 0.1),
                    use_gcn=layer_cfg.get('use_gcn', True)
                )
                for layer_cfg in layer_configs
            ])
        else:
            # Default: all layers use same config
            default_use_gcn = processor_cfg.get('use_gcn', True)
            gnn_blocks = nn.ModuleList([
                InstNetBlock(
                    inst_dim=hidden_dim,
                    net_dim=hidden_dim,
                    edge_dim=processor_cfg.get('edge_dim', hidden_dim),
                    heads=processor_cfg.get('heads', 4),
                    dropout=processor_cfg.get('dropout', 0.1),
                    use_gcn=default_use_gcn
                )
                for _ in range(num_layers)
            ])

        # Optional global modules
        if processor_cfg.get('use_global_module', False):
            global_cfg = processor_cfg.get('global_module', {})
            global_params = global_cfg.get('params', {})
            if time_embed_dim is not None:
                global_params['time_embed_dim'] = time_embed_dim
            global_modules = nn.ModuleList([
                create_global_module(
                    module_type=global_cfg.get('type', 'attention'),
                    hidden_dim=hidden_dim,
                    **global_params
                )
                for _ in range(num_layers)
            ])
            
            # Create gated adapters if enabled
            use_gated_adapter = processor_cfg.get('use_gated_adapter', False)
            if use_gated_adapter:
                adapter_cfg = processor_cfg.get('gated_adapter', {})
                gated_adapters = nn.ModuleList([
                    ResidualGatedAdapter(
                        hidden_dim=hidden_dim,
                        gate_type=adapter_cfg.get('gate_type', 'mlp'),
                        init_gate_scale=adapter_cfg.get('init_gate_scale', 0.1),
                    )
                    for _ in range(num_layers)
                ])
            else:
                gated_adapters = None
        else:
            global_modules = None
            gated_adapters = None
            
        return gnn_blocks, global_modules, None, None, gated_adapters, None
        
    elif processor_type == 'gnn':
        # Homogeneous GNN: inst-inst message passing (for homogeneous Data)
        num_blocks = processor_cfg.get('num_blocks', None)
        layer_configs = processor_cfg.get('layer_configs', None)
        num_layers = processor_cfg.get('num_layers', None)
        
        if num_blocks is not None and layer_configs is not None:
            # Block-based architecture
            layers_per_block = len(layer_configs)

            gnn_blocks_cfg = processor_cfg.get('gnn_blocks', None)
            if gnn_blocks_cfg is None:
                selected_gnn_blocks = set(range(num_blocks))
            else:
                selected_gnn_blocks = set(int(i) for i in gnn_blocks_cfg)
                invalid = sorted(
                    i for i in selected_gnn_blocks if i < 0 or i >= num_blocks
                )
                if invalid:
                    raise ValueError(
                        f"processor.gnn_blocks has invalid indices {invalid}; "
                        f"valid range is [0, {num_blocks - 1}]"
                    )

            # Build GNN layers only for selected blocks (sparse layout).
            gnn_blocks = nn.ModuleList()
            gnn_layer_ranges: List[Optional[Tuple[int, int]]] = []
            for block_idx in range(num_blocks):
                if block_idx not in selected_gnn_blocks:
                    gnn_layer_ranges.append(None)
                    continue
                start = len(gnn_blocks)
                for layer_cfg in layer_configs:
                    gat_concat = layer_cfg.get('concat', None)
                    layer_kwargs = {
                        k: v for k, v in layer_cfg.items()
                        if k not in ('layer_type', 'heads', 'concat')
                    }
                    layer_type_lower = layer_cfg.get('layer_type', '').lower()
                    if gat_concat is not None and layer_type_lower in ('gat', 'gatv2'):
                        layer_kwargs['concat'] = gat_concat

                    gnn_blocks.append(
                        UnifiedGNNLayer(
                            layer_type=layer_cfg.get('layer_type', 'gcn'),
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=layer_cfg.get('heads', 1),
                            dropout=processor_cfg.get('dropout', 0.1),
                            edge_dim=processor_cfg.get('edge_dim', None),
                            use_pre_norm=processor_cfg.get('use_pre_norm', False),
                            **layer_kwargs
                        )
                    )
                gnn_layer_ranges.append((start, len(gnn_blocks)))
            
            # Create global modules (optionally only on selected blocks)
            if processor_cfg.get('use_global_module', False):
                global_cfg = processor_cfg.get('global_module', {})
                global_params = global_cfg.get('params', {})
                if time_embed_dim is not None:
                    global_params['time_embed_dim'] = time_embed_dim
                global_module_blocks = processor_cfg.get('global_module_blocks', None)
                if global_module_blocks is None:
                    selected_blocks = set(range(num_blocks))
                else:
                    selected_blocks = set(int(i) for i in global_module_blocks)
                    invalid = sorted(i for i in selected_blocks if i < 0 or i >= num_blocks)
                    if invalid:
                        raise ValueError(
                            f"processor.global_module_blocks has invalid indices {invalid}; "
                            f"valid range is [0, {num_blocks - 1}]"
                        )

                # Keep one module per block so forward logic stays unchanged.
                # Non-selected blocks use identity (zero delta) as "no global module".
                global_modules = nn.ModuleList([
                    create_global_module(
                        module_type=(
                            global_cfg.get('type', 'attention')
                            if block_idx in selected_blocks
                            else 'identity'
                        ),
                        hidden_dim=hidden_dim,
                        **(global_params if block_idx in selected_blocks else {})
                    )
                    for block_idx in range(num_blocks)
                ])
                
                # Create gated adapters if enabled
                use_gated_adapter = processor_cfg.get('use_gated_adapter', False)
                if use_gated_adapter:
                    adapter_cfg = processor_cfg.get('gated_adapter', {})
                    gated_adapters = nn.ModuleList([
                        ResidualGatedAdapter(
                            hidden_dim=hidden_dim,
                            gate_type=adapter_cfg.get('gate_type', 'mlp'),
                            init_gate_scale=adapter_cfg.get('init_gate_scale', 0.1),
                            use_time_gate=adapter_cfg.get('use_time_gate', True),
                        )
                        for _ in range(num_blocks)
                    ])
                else:
                    gated_adapters = None
            else:
                global_modules = None
                gated_adapters = None
                
            return (
                gnn_blocks,
                global_modules,
                num_blocks,
                layers_per_block,
                gated_adapters,
                gnn_layer_ranges,
            )

        elif layer_configs is not None and num_layers is not None:
            # Flat layer configuration
            if len(layer_configs) != num_layers:
                raise ValueError(f"layer_configs length ({len(layer_configs)}) must match num_layers ({num_layers})")
            gnn_blocks = nn.ModuleList()
            for layer_cfg in layer_configs:
                layer_kwargs = {
                    k: v for k, v in layer_cfg.items()
                    if k not in ('layer_type', 'heads')
                }
                gnn_blocks.append(
                    UnifiedGNNLayer(
                        layer_type=layer_cfg.get('layer_type', 'gcn'),
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=layer_cfg.get('heads', 1),
                        dropout=processor_cfg.get('dropout', 0.1),
                        edge_dim=processor_cfg.get('edge_dim', None),
                        use_pre_norm=processor_cfg.get('use_pre_norm', False),
                        **layer_kwargs,
                    )
                )
            
            # Optional global modules
            if processor_cfg.get('use_global_module', False):
                global_cfg = processor_cfg.get('global_module', {})
                global_params = global_cfg.get('params', {})
                if time_embed_dim is not None:
                    global_params['time_embed_dim'] = time_embed_dim
                global_module_layers = processor_cfg.get('global_module_layers', None)
                if global_module_layers is None:
                    selected_layers = set(range(num_layers))
                else:
                    selected_layers = set(int(i) for i in global_module_layers)
                    invalid = sorted(i for i in selected_layers if i < 0 or i >= num_layers)
                    if invalid:
                        raise ValueError(
                            f"processor.global_module_layers has invalid indices {invalid}; "
                            f"valid range is [0, {num_layers - 1}]"
                        )

                global_modules = nn.ModuleList([
                    create_global_module(
                        module_type=(
                            global_cfg.get('type', 'attention')
                            if layer_idx in selected_layers
                            else 'identity'
                        ),
                        hidden_dim=hidden_dim,
                        **(global_params if layer_idx in selected_layers else {})
                    )
                    for layer_idx in range(num_layers)
                ])
                
                # Create gated adapters if enabled
                use_gated_adapter = processor_cfg.get('use_gated_adapter', False)
                if use_gated_adapter:
                    adapter_cfg = processor_cfg.get('gated_adapter', {})
                    gated_adapters = nn.ModuleList([
                        ResidualGatedAdapter(
                            hidden_dim=hidden_dim,
                            gate_type=adapter_cfg.get('gate_type', 'mlp'),
                            init_gate_scale=adapter_cfg.get('init_gate_scale', 0.1),
                            use_time_gate=adapter_cfg.get('use_time_gate', True),
                        )
                        for _ in range(num_layers)
                    ])
                else:
                    gated_adapters = None
            else:
                global_modules = None
                gated_adapters = None
                
            return gnn_blocks, global_modules, None, None, gated_adapters, None
        else:
            # Default: simple flat config
            num_layers = num_layers or 4
            layer_type = processor_cfg.get('layer_type', 'gcn')
            gnn_blocks = nn.ModuleList([
                UnifiedGNNLayer(
                    layer_type=layer_type,
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=processor_cfg.get('heads', 1),
                    dropout=processor_cfg.get('dropout', 0.1),
                    edge_dim=processor_cfg.get('edge_dim', None),
                    use_pre_norm=processor_cfg.get('use_pre_norm', False),
                )
                for _ in range(num_layers)
            ])
            
            return gnn_blocks, None, None, None, None, None

    elif processor_type == 'att_gnn':
        # Full AttGNN from att_gnn.py: stacked (ResGNN + AttGNN) blocks with configurable layers
        num_blocks = processor_cfg.get('num_blocks', 3)
        hidden_node_features = processor_cfg.get(
            'hidden_node_features',
            [processor_cfg.get('hidden_dim', hidden_dim)] * num_blocks
        )
        attention_node_features = processor_cfg.get(
            'attention_node_features',
            [processor_cfg.get('hidden_dim', hidden_dim)] * num_blocks
        )
        if len(hidden_node_features) != num_blocks or len(attention_node_features) != num_blocks:
            raise ValueError(
                f"hidden_node_features and attention_node_features must have length num_blocks ({num_blocks})"
            )
        cond_node_features = processor_cfg.get('cond_node_features', 0)
        edge_features = processor_cfg.get('edge_dim', 4)
        t_encoding_dim = time_embed_dim if time_embed_dim is not None else 0
        conv_params = processor_cfg.get('conv_params', {'layer_type': 'gcn'})
        att_gnn_module = AttGNN(
            in_node_features=hidden_dim,
            out_node_features=hidden_dim,
            hidden_size=hidden_dim,
            hidden_node_features=hidden_node_features,
            attention_node_features=attention_node_features,
            cond_node_features=cond_node_features,
            edge_features=edge_features,
            layers_per_block=processor_cfg.get('layers_per_block', 2),
            t_encoding_dim=t_encoding_dim,
            conv_params=conv_params,
            mlp_num_layers=processor_cfg.get('mlp_num_layers', 0),
            mlp_size_factor=processor_cfg.get('mlp_size_factor', 0),
            input_encoding_dim=processor_cfg.get('input_encoding_dim', 0),
            dropout=processor_cfg.get('dropout', 0.1),
            device='cpu',
            res_gnn_layers_per_block=processor_cfg.get('res_gnn_layers_per_block'),
            att_gnn_layers_per_block=processor_cfg.get('att_gnn_layers_per_block'),
            num_heads=processor_cfg.get('num_heads', 8),
            ff_num_layers=processor_cfg.get('ff_num_layers', 2),
            ff_size_factor=processor_cfg.get('ff_size_factor', 2),
            att_implementation=processor_cfg.get('att_implementation', 'default'),
        )
        return nn.ModuleList([att_gnn_module]), None, None, None, None, None

    elif processor_type == 'incidence_transformer':
        from .processors.incidence_transformer import (
            IncidenceTransformerConfig,
            IncidenceTransformerProcessor,
        )
        it_config = IncidenceTransformerConfig(
            d_node_in=hidden_dim,
            d_edge_in=processor_cfg.get('edge_dim', 4),
            d_model=processor_cfg.get('d_model', hidden_dim),
            num_layers=processor_cfg.get('num_layers', 12),
            num_heads_local=processor_cfg.get('num_heads_local', 8),
            num_heads_global=processor_cfg.get('num_heads_global', 4),
            dropout=processor_cfg.get('dropout', 0.0),
            ffn_multiplier=processor_cfg.get('ffn_multiplier', 4.0),
            ffn_activation=processor_cfg.get('ffn_activation', 'gelu'),
            use_global_branch=processor_cfg.get('use_global_branch', True),
            global_branch_type=processor_cfg.get('global_branch_type', 'memory'),
            num_memory_tokens=processor_cfg.get('num_memory_tokens', 16),
            global_branch_every_n=processor_cfg.get('global_branch_every_n', 1),
            use_node_ids=processor_cfg.get('use_node_ids', True),
            d_node_id=processor_cfg.get('d_node_id', 32),
            max_nodes=processor_cfg.get('max_nodes', 10000),
            id_type=processor_cfg.get('id_type', 'learned'),
            id_noise_scale=processor_cfg.get('id_noise_scale', 0.0),
            use_degree_encoding=processor_cfg.get('use_degree_encoding', True),
            max_in_degree=processor_cfg.get('max_in_degree', 128),
            max_out_degree=processor_cfg.get('max_out_degree', 128),
            use_edge_bias=processor_cfg.get('use_edge_bias', False),
            residual_scale=processor_cfg.get('residual_scale', 1.0),
            use_activation_checkpointing=processor_cfg.get('use_activation_checkpointing', False),
        )
        processor_module = IncidenceTransformerProcessor(it_config)
        return nn.ModuleList([processor_module]), None, None, None, None, None

    else:
        raise ValueError(f"Unsupported processor type: {processor_type}. Supported: 'bipartite_gnn', 'gnn', 'att_gnn', 'incidence_transformer'")


class UnifiedModel(nn.Module):
    """
    Unified model that assembles encoder → processor → decoder based on config.
    Supports both ChipGen (HeteroData) and TopoGeoNet (homogeneous) workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.task = config.get('task', 'regression')  # regression, diffusion, classification

        # Build encoder
        encoder_cfg = config['encoder']
        encoder_type = encoder_cfg['type']  # 'mlp', 'bipartite_gnn', 'gnn'

        self.input_dim = encoder_cfg['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']

        # Coordinate injection mode for the encoder input.
        # The public input_dim still counts the prepended z_t channels so training/data plumbing
        # remains backward compatible. This flag only controls what the encoder actually consumes.
        coord_mode_default = 'raw+fourier' if encoder_cfg.get('use_coord_fourier', False) else 'raw'
        self.coord_encoding_mode = encoder_cfg.get('coord_encoding_mode', coord_mode_default)
        valid_coord_modes = {'raw', 'raw+fourier', 'fourier_only', 'none'}
        if self.coord_encoding_mode not in valid_coord_modes:
            raise ValueError(
                f"Unknown encoder.coord_encoding_mode='{self.coord_encoding_mode}'. "
                f"Valid options: {sorted(valid_coord_modes)}"
            )

        # Random Fourier encoding for coordinates (optional)
        # Extracts first 2 dims (z_tx, z_ty) and adds Fourier features
        # Optimized for coordinates in [-0.5, 0.5] range (domain width = 1.0)
        self.use_coord_fourier = encoder_cfg.get('use_coord_fourier', False)
        self.coord_fourier_dim = 0
        self.coord_fourier_input_scale = 1.0
        if self.use_coord_fourier:
            fourier_cfg = encoder_cfg.get('coord_fourier', {})
            # Our normalized coordinates are typically z in [-1, 1] (chip-centered, scaled by W/2,H/2).
            # The RFF implementation/documentation assumes a unit-width domain (roughly [-0.5, 0.5]).
            # Scaling coords by 0.5 aligns these conventions and improves "cycles across chip" interpretability.
            self.coord_fourier_input_scale = float(fourier_cfg.get('input_scale', 1.0))
            self.coord_fourier = RandomFourierEncoding(
                input_dim=2,  # 2D coordinates
                num_features=fourier_cfg.get('num_features', 32),
                scale=fourier_cfg.get('scale', 1.0),  # Legacy parameter
                learnable_scale=fourier_cfg.get('learnable_scale', False),  # Legacy parameter
                use_log_uniform=fourier_cfg.get('use_log_uniform', True),  # Recommended for [-0.5, 0.5]
                min_freq=fourier_cfg.get('min_freq', 0.1),  # Global scale frequencies
                max_freq=fourier_cfg.get('max_freq', 1000.0),  # Local scale frequencies
                seed=fourier_cfg.get('seed', None)  # For reproducibility
            )
            self.coord_fourier_dim = self.coord_fourier.output_dim  # 2 * num_features
        else:
            self.coord_fourier = None

        if self.coord_encoding_mode in ('raw+fourier', 'fourier_only') and not self.use_coord_fourier:
            raise ValueError(
                f"encoder.coord_encoding_mode='{self.coord_encoding_mode}' requires encoder.use_coord_fourier=true"
            )

        # Positional encoder (optional) - NOT used in encoder
        # Note: Positional encoding IS used in transformer (global_module) via inst_pos=x_t parameter
        # The encoder receives z_t directly in node features [z_tx, z_ty, ...]
        # The transformer receives x_t separately for positional encoding (see forward() method)
        self.pos_encoder = None
        pos_dim = 0

        # Main encoder
        # The data pipeline still prepends z_t to node features, but the encoder can choose
        # to consume raw coords, Fourier-only coords, or no direct coordinate channels.
        if self.coord_encoding_mode == 'raw':
            total_input_dim = self.input_dim
        elif self.coord_encoding_mode == 'raw+fourier':
            total_input_dim = self.input_dim + self.coord_fourier_dim
        elif self.coord_encoding_mode == 'fourier_only':
            total_input_dim = (self.input_dim - 2) + self.coord_fourier_dim
        else:  # 'none'
            total_input_dim = self.input_dim - 2

        # Sin-cos positional encoding: use_pos_encoding or legacy use_positional_encoding
        pos_enc_cfg = encoder_cfg.get('pos_encoding', {})
        use_pos_encoding = encoder_cfg.get('use_pos_encoding', encoder_cfg.get('use_positional_encoding', False))
        pos_encoding_dim = pos_enc_cfg.get('positional_encoding_dim', encoder_cfg.get('positional_encoding_dim', 32))
        pos_encoding_max_freq = pos_enc_cfg.get('positional_encoding_max_freq', encoder_cfg.get('positional_encoding_max_freq', 100.0))
        
        # Store encoder type for forward pass
        self.encoder_type = encoder_type
        
        if encoder_type == 'mlp':
            # Traditional MLP encoder (lightweight feature extraction)
            # Supports optional sin-cos positional encoding (use_pos_encoding) or RFF (use_coord_fourier)
            self.encoder = MLPEncoder(
                input_dim=total_input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=encoder_cfg.get('num_layers', 2),
                dropout=encoder_cfg.get('dropout', 0.1),
                use_layer_norm=encoder_cfg.get('use_layer_norm', encoder_cfg.get('use_batch_norm', True)),
                use_positional_encoding=use_pos_encoding,
                positional_encoding_dim=pos_encoding_dim,
                positional_encoding_max_freq=pos_encoding_max_freq,
                physical_feature_dims=encoder_cfg.get('physical_feature_dims', None)  # NEW: dual-branch encoding
            )
            self.encoder_gnn_blocks = None
            self.encoder_global_modules = None
            self.encoder_num_blocks = None
            self.encoder_layers_per_block = None
            self.encoder_use_residual = None
            
        elif encoder_type in ('bipartite_gnn', 'gnn'):
            # GNN-based encoder (HUGE encoder + simple solver pattern)
            # First project to hidden_dim with a simple MLP
            self.encoder = MLPEncoder(
                input_dim=total_input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=1,  # Just a projection layer
                dropout=0.0
            )
            
            # Then apply GNN layers (reuses processor architecture)
            # Do not pass time_embed_dim to encoder global modules (single time injection only, after encoder)
            self.encoder_gnn_blocks, self.encoder_global_modules, self.encoder_num_blocks, self.encoder_layers_per_block, self.encoder_gated_adapters, self.encoder_gnn_layer_ranges = _build_gnn_layers(
                encoder_cfg, self.hidden_dim, None
            )
            self.encoder_use_residual = encoder_cfg.get('use_residual', True)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}. Supported: 'mlp', 'bipartite_gnn', 'gnn'")

        # Build processor (optional - can be None for encoder-only models like MGM pretraining)
        processor_cfg = config.get('processor', None)
        
        if processor_cfg is not None:
            processor_type = processor_cfg['type']  # 'bipartite_gnn', 'gnn'
            self.processor_type = processor_type
            
            # Use helper function to build processor layers
            # Do not pass time_embed_dim to processor global modules (single time injection only, after encoder)
            self.gnn_blocks, self.global_modules, self.num_blocks, self.layers_per_block, self.gated_adapters, self.processor_gnn_layer_ranges = _build_gnn_layers(
                processor_cfg, self.hidden_dim, None
            )

            # Optional per-block GNN execution mask for block-based homogeneous processors.
            # Enables mixed schedules like:
            # - early blocks: pure GNN (local-only)
            # - late blocks: global-only (no GNN compute)
            self.processor_gnn_block_mask = None
            if self.processor_type == 'gnn' and self.num_blocks is not None:
                gnn_blocks_cfg = processor_cfg.get('gnn_blocks', None)
                if gnn_blocks_cfg is None:
                    self.processor_gnn_block_mask = [True] * self.num_blocks
                else:
                    selected = set(int(i) for i in gnn_blocks_cfg)
                    invalid = sorted(i for i in selected if i < 0 or i >= self.num_blocks)
                    if invalid:
                        raise ValueError(
                            f"processor.gnn_blocks has invalid indices {invalid}; "
                            f"valid range is [0, {self.num_blocks - 1}]"
                        )
                    self.processor_gnn_block_mask = [
                        (block_idx in selected) for block_idx in range(self.num_blocks)
                    ]
            
            # Store residual connection flag (for homogeneous GNN only)
            self.use_residual = processor_cfg.get('use_residual', True)
        else:
            # No processor (encoder → decoder directly)
            self.processor_type = None
            self.gnn_blocks = None
            self.processor_gnn_layer_ranges = None
            self.global_modules = None
            self.num_blocks = None
            self.layers_per_block = None
            self.use_residual = None
            self.processor_gnn_block_mask = None

        # Build decoder
        decoder_cfg = config.get('decoder', {})
        # For EDM (x0) and flow_matching (velocity), don't use zero_init.
        # Zero init is for epsilon prediction (DDPM) only.
        # Flow matching predicts velocity v = z_1 - z_0, not noise epsilon
        param = config.get('diffusion', {}).get('parameterization')
        use_zero_init = False if param in ('edm_x0_precond', 'flow_matching', 'flow_matching_raw') else (self.task == 'diffusion')
        self.two_scale_velocity = (self.task == 'diffusion' and config.get('diffusion', {}).get('two_scale_velocity', False))
        if self.two_scale_velocity:
            # Two-scale velocity: v_phys = L * v_die + s_c * v_cell. Two heads, each 2D.
            self.decoder = None
            self.decoder_die = MLPDecoder(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=2,
                num_layers=decoder_cfg.get('num_layers', 2),
                dropout=decoder_cfg.get('dropout', 0.1),
                zero_init=use_zero_init
            )
            self.decoder_cell = MLPDecoder(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=2,
                num_layers=decoder_cfg.get('num_layers', 2),
                dropout=decoder_cfg.get('dropout', 0.1),
                zero_init=use_zero_init
            )
            # Effective output is 4 (v_die_x, v_die_y, v_cell_x, v_cell_y)
            self._output_dim_effective = 4
        else:
            self.decoder_die = None
            self.decoder_cell = None
            self._output_dim_effective = self.output_dim
            self.decoder = MLPDecoder(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=decoder_cfg.get('num_layers', 2),
                dropout=decoder_cfg.get('dropout', 0.1),
                zero_init=use_zero_init
            )

        # Diffusion: single time MLP, output added to initial node embedding only (no FiLM, no time_proj)
        # Set use_time_embed: false in config to turn off time conditioning entirely.
        if self.task == 'diffusion':
            diff_cfg = config.get('diffusion', {})
            use_time_embed = diff_cfg.get('use_time_embed', True)
            if use_time_embed:
                from ..training.diffusion_modules import TimeEmbedding
                time_mlp_dim = diff_cfg.get('time_mlp_hidden_dim', config.get('time_mlp_hidden_dim', 128))
                self.time_embedding = TimeEmbedding(
                    mlp_hidden_dim=time_mlp_dim,
                    output_dim=self.hidden_dim,
                )
            else:
                self.time_embedding = None

        # Optional regression graph-size conditioning (FiLM-style).
        self.size_conditioning_enabled = False
        self.size_conditioning_apply_to_encoder = False
        self.size_conditioning_apply_to_processor = False
        self.size_conditioning_apply_to_decoder = False
        self.size_conditioning_use_num_edges = True
        self.size_conditioner = None
        self.encoder_condition_film = None
        self.encoder_layer_films = None
        self.processor_layer_films = None
        self.decoder_condition_film = None
        if self.task == "regression":
            size_cfg = config.get("size_conditioning", {})
            if bool(size_cfg.get("enabled", False)):
                cond_hidden_dim = int(size_cfg.get("conditioning_hidden_dim", 64))
                self.size_conditioning_enabled = True
                self.size_conditioning_apply_to_encoder = bool(size_cfg.get("apply_to_encoder", True))
                self.size_conditioning_apply_to_processor = bool(size_cfg.get("apply_to_processor", True))
                self.size_conditioning_apply_to_decoder = bool(size_cfg.get("apply_to_decoder", True))
                self.size_conditioning_use_num_edges = bool(size_cfg.get("use_num_edges", True))

                self.size_conditioner = GraphSizeConditioner(
                    cond_hidden_dim=cond_hidden_dim,
                    output_dim=self.hidden_dim,
                )

                if self.size_conditioning_apply_to_encoder:
                    self.encoder_condition_film = FeatureFiLM(self.hidden_dim, self.hidden_dim)
                    if self.encoder_gnn_blocks is not None:
                        self.encoder_layer_films = nn.ModuleList([
                            FeatureFiLM(self.hidden_dim, self.hidden_dim)
                            for _ in range(len(self.encoder_gnn_blocks))
                        ])

                if self.size_conditioning_apply_to_processor and self.gnn_blocks is not None:
                    self.processor_layer_films = nn.ModuleList([
                        FeatureFiLM(self.hidden_dim, self.hidden_dim)
                        for _ in range(len(self.gnn_blocks))
                    ])

                if self.size_conditioning_apply_to_decoder:
                    self.decoder_condition_film = FeatureFiLM(self.hidden_dim, self.hidden_dim)

        # Optional looped-processor PEFT: unroll the frozen processor for K
        # iterations, with a tiny per-(iter, graph-size[, block]) FiLM (+gate)
        # adapter steering the hidden state between iterations.
        self.loop_conditioning_enabled = False
        self.loop_mode = None
        self.loop_num_iterations = 1
        self.loop_use_outer_residual = False
        self.loop_gradient_checkpoint = False
        self.loop_conditioner = None
        # Triple-controller (FiLM) only: pre-computed
        # ``{block_idx -> position}`` map describing where each of the three
        # FiLM heads fires inside the per-block processor loop. ``None`` for
        # every other ``loop_mode``; populated below when
        # ``loop_conditioning.mode == 'triple_film_loop'``.
        self._loop_triple_block_positions: Optional[Dict[int, int]] = None
        # Staged-controller (FiLM) only: pre-computed
        # ``{'early'|'late'|'xfm' -> block_idx}`` map describing which
        # processor block each of the three sequential phase loops drives.
        # ``None`` for every other ``loop_mode``; populated below when
        # ``loop_conditioning.mode == 'staged_film_loop'``.
        self._loop_staged_block_indices: Optional[Dict[str, int]] = None
        # gnn_gated_halt_loop only: GNN block indices that get looped per
        # iteration (each iter applies these blocks in order + gate), and
        # global-module block indices that run ONCE after the loop ends
        # (Variant A) or ONCE per iteration's decode path (Variant B).
        self._loop_gated_gnn_block_indices: Optional[List[int]] = None
        self._loop_gated_global_block_indices: Optional[List[int]] = None
        # gnn_gated_halt_loop / Variant B only.
        self.halt_module: Optional[nn.Module] = None
        # PonderNet halting state. Only populated when
        # loop_conditioning.halting.enabled is True.
        self.loop_pondernet_enabled = False
        self.loop_halt_inference_threshold = 0.5
        self.loop_halt_prior_lambda = 0.4
        self.loop_halt_reg_weight = 0.0
        self.loop_halt_reg_warmup_steps = 0
        # 'pondernet_geometric' (default) | 'softmax_over_k'.
        # Both consume the per-step halt logits emitted by the HaltModule and
        # produce a halt distribution p_k that sums to 1 over k.
        self.loop_halt_distribution = "pondernet_geometric"
        # 'soft' (default, return expected_pred = sum_k p_k * pred_k) | 'hard'
        # (PonderNet-only early-exit at the cumulative threshold).
        self.loop_halt_inference_mode = "soft"
        # iterGNN Variant B only: recompute the per-iter (post-loop transformer +
        # decoder) pass during backward instead of caching its activations.
        # Keeps peak training memory ~O(1) in K at the cost of ~1.5x compute.
        self.loop_decode_gradient_checkpoint = False
        # Side-channel populated each forward when ponder is enabled; consumed
        # by the regression loss to compute the per-iter weighted MSE + KL.
        self._last_ponder_aux: Optional[Dict[str, torch.Tensor]] = None
        # Side-channel for the gated residual loop conditioner (both variants):
        # holds {gate_per_iter [K, N], gate_logit_per_iter [K, N],
        #        gate_activation (str), node_batch [N]}. The loss path reads
        # this to add the optional gate L0/entropy regulariser, and the
        # analysis path reads it for the gate-zero-fraction plots.
        self._last_gate_aux: Optional[Dict[str, torch.Tensor]] = None
        # Gate regularisation weights (read by the loss path).
        self.loop_gate_l0_weight = 0.0
        self.loop_gate_l0_warmup_steps = 0
        self.loop_gate_entropy_weight = 0.0
        self.loop_gate_entropy_warmup_steps = 0
        # Index of the final transformer (global-module) block inside the processor
        # stack. Used by dual-controller loop modes to know where to insert the
        # "after GNNs" / "after final transformer" controller calls. Computed once
        # at init from the processor config so the forward path can stay generic.
        self.final_transformer_block_idx: Optional[int] = None
        processor_cfg_for_idx = config.get("processor", {}) or {}
        if (
            self.processor_type == "gnn"
            and self.num_blocks is not None
            and processor_cfg_for_idx.get("use_global_module", False)
        ):
            global_module_blocks_cfg = processor_cfg_for_idx.get(
                "global_module_blocks", None
            )
            if global_module_blocks_cfg:
                self.final_transformer_block_idx = max(
                    int(i) for i in global_module_blocks_cfg
                )
            else:
                self.final_transformer_block_idx = self.num_blocks - 1

        if self.task == "regression":
            loop_cfg = config.get("loop_conditioning", {})
            if bool(loop_cfg.get("enabled", False)):
                mode = str(loop_cfg.get("mode", "film_loop"))
                _valid_modes = (
                    "film_loop",
                    "per_block_steer",
                    "transformer_loop",
                    "identity",
                    "dual_film_loop",
                    "dual_transformer_loop",
                    "triple_film_loop",
                    "staged_film_loop",
                    "gnn_gated_halt_loop",
                )
                if mode not in _valid_modes:
                    raise ValueError(
                        f"loop_conditioning.mode='{mode}' is not supported. "
                        f"Valid options: {list(_valid_modes)}"
                    )
                num_iters = int(loop_cfg.get("num_iterations", 2))
                if num_iters < 1:
                    raise ValueError(
                        f"loop_conditioning.num_iterations must be >= 1, got {num_iters}"
                    )
                conditioning_dim = int(loop_cfg.get("conditioning_dim", 64))
                use_size_input = bool(loop_cfg.get("use_size_input", True))
                zero_init_film = bool(loop_cfg.get("zero_init_film", True))

                self.loop_conditioning_enabled = True
                self.loop_mode = mode
                self.loop_num_iterations = num_iters
                self.loop_use_outer_residual = bool(loop_cfg.get("use_outer_residual", False))
                # Per-iteration gradient checkpointing: trade compute for memory
                # by recomputing each (frozen) processor pass during backward
                # instead of caching its activations. With shared FiLM, this
                # makes peak training memory ~O(1) in K instead of O(K), which
                # matters once K gets large (e.g. K=8 / K=16). Default off so
                # existing K=2/K=4 runs stay bit-identical.
                self.loop_gradient_checkpoint = bool(
                    loop_cfg.get("gradient_checkpoint", False)
                )

                # PonderNet halting block (optional). When enabled, the loop
                # body decodes after every iteration, the conditioner emits a
                # per-graph halt logit, and the loss path computes the
                # PonderNet expected MSE + KL-to-truncated-geometric prior.
                halt_cfg = loop_cfg.get("halting", {}) or {}
                halting_enabled = bool(halt_cfg.get("enabled", False))
                init_halt_bias = float(halt_cfg.get("init_halt_bias", -2.0))
                if halting_enabled and mode not in ("film_loop", "gnn_gated_halt_loop"):
                    raise ValueError(
                        "loop_conditioning.halting.enabled=True is currently "
                        "supported only with mode='film_loop' or "
                        f"'gnn_gated_halt_loop' (got mode='{mode}')."
                    )
                # Dual-, triple-, and staged-controller modes all need a known
                # transformer-block boundary in the processor so we know where
                # to apply the "after-GNNs" / "after-final-transformer" /
                # (for triple) the extra "after-first-GNN-block" / (for
                # staged) the per-phase block restrictions. Catch
                # misconfiguration eagerly (rather than failing inside forward).
                if mode in (
                    "dual_film_loop",
                    "dual_transformer_loop",
                    "triple_film_loop",
                    "staged_film_loop",
                ):
                    if self.final_transformer_block_idx is None:
                        raise ValueError(
                            f"loop_conditioning.mode='{mode}' requires the processor "
                            f"to expose a final-transformer block via "
                            f"processor.use_global_module=True (with optional "
                            f"processor.global_module_blocks). Got processor_type="
                            f"{self.processor_type!r}, num_blocks={self.num_blocks}."
                        )
                    if self.loop_gradient_checkpoint:
                        raise ValueError(
                            f"loop_conditioning.gradient_checkpoint=True is not "
                            f"supported with mode='{mode}' yet (the multi-controller "
                            f"variants thread state across iterations inside the "
                            f"processor pass)."
                        )

                # Triple-controller mode: pre-compute the (block_idx -> position)
                # map so the forward path can fire post-block at three distinct
                # insertion points without re-deriving them every iteration.
                # Schedule:
                #     block_idx == first_gnn_block        -> position 0
                #     block_idx == last_gnn_block         -> position 1  (= dual pos 0)
                #     block_idx == final_transformer_blk  -> position 2  (= dual pos 1)
                if mode == "triple_film_loop":
                    if self.processor_gnn_block_mask is None:
                        raise ValueError(
                            "loop_conditioning.mode='triple_film_loop' requires the "
                            "processor to be a block-based GNN with at least two "
                            "GNN blocks (got processor_gnn_block_mask=None)."
                        )
                    gnn_block_indices = [
                        i for i, on in enumerate(self.processor_gnn_block_mask) if on
                    ]
                    if len(gnn_block_indices) < 2:
                        raise ValueError(
                            "loop_conditioning.mode='triple_film_loop' needs at "
                            "least two GNN blocks (one for the 'after first GNN "
                            "block' controller, one for the 'after last GNN "
                            f"block' controller). Got gnn_blocks={gnn_block_indices}."
                        )
                    first_gnn_idx = gnn_block_indices[0]
                    last_gnn_idx = gnn_block_indices[-1]
                    if last_gnn_idx >= self.final_transformer_block_idx:
                        raise ValueError(
                            "loop_conditioning.mode='triple_film_loop' assumes the "
                            "final transformer block runs strictly after the last "
                            "GNN block, but got last_gnn_block_idx="
                            f"{last_gnn_idx} >= final_transformer_block_idx="
                            f"{self.final_transformer_block_idx}."
                        )
                    # All three insertion points must be distinct (otherwise two
                    # FiLM heads would fire on the same hidden state at the same
                    # boundary, which is almost never what the user wants).
                    pos_map = {
                        first_gnn_idx: 0,
                        last_gnn_idx: 1,
                        self.final_transformer_block_idx: 2,
                    }
                    if len(pos_map) != 3:
                        raise ValueError(
                            "loop_conditioning.mode='triple_film_loop' requires "
                            "three distinct insertion blocks but got first_gnn_idx="
                            f"{first_gnn_idx}, last_gnn_idx={last_gnn_idx}, "
                            f"final_transformer_idx={self.final_transformer_block_idx}."
                        )
                    self._loop_triple_block_positions = pos_map

                # Staged-controller mode: pre-compute the
                # ``{phase_name -> block_idx}`` map describing which processor
                # block each of the three phase loops drives. Schedule:
                #     phase 'early' -> first GNN block       (LOCAL head #1)
                #     phase 'late'  -> last  GNN block       (LOCAL head #3)
                #     phase 'xfm'   -> final transformer blk (LOCAL head #5)
                # The two TRANSITION heads (#2, #4) fire between phases.
                # Strict ordering (early < late < xfm) is required so the
                # phases run in the same order the frozen processor expects.
                # gnn_gated_halt_loop mode: needs at least one GNN block (looped
                # K times with the per-node gated residual) and one global-module
                # block (the post-loop fixed transformer) running strictly after
                # the GNN blocks. Pre-compute the index lists once.
                if mode == "gnn_gated_halt_loop":
                    if self.processor_gnn_block_mask is None:
                        raise ValueError(
                            "loop_conditioning.mode='gnn_gated_halt_loop' requires the "
                            "processor to be a block-based GNN (got "
                            "processor_gnn_block_mask=None)."
                        )
                    gnn_block_indices = [
                        i for i, on in enumerate(self.processor_gnn_block_mask) if on
                    ]
                    if not gnn_block_indices:
                        raise ValueError(
                            "loop_conditioning.mode='gnn_gated_halt_loop' needs at "
                            "least one GNN block in processor.gnn_blocks; got "
                            f"gnn_blocks={gnn_block_indices}."
                        )
                    if self.final_transformer_block_idx is None:
                        raise ValueError(
                            "loop_conditioning.mode='gnn_gated_halt_loop' requires the "
                            "processor to expose at least one global_module block "
                            "(processor.use_global_module=True) so the fixed "
                            "post-loop transformer has somewhere to live."
                        )
                    global_blocks_cfg = (
                        processor_cfg_for_idx.get("global_module_blocks", None)
                    )
                    if global_blocks_cfg is None:
                        # Default to "everything outside the gnn blocks".
                        gnn_set = set(gnn_block_indices)
                        global_block_indices = [
                            i for i in range(self.num_blocks) if i not in gnn_set
                        ]
                    else:
                        global_block_indices = sorted(int(i) for i in global_blocks_cfg)
                    if not global_block_indices:
                        raise ValueError(
                            "loop_conditioning.mode='gnn_gated_halt_loop' could not "
                            "find any global-module blocks in the processor schedule."
                        )
                    if max(gnn_block_indices) >= min(global_block_indices):
                        raise ValueError(
                            "loop_conditioning.mode='gnn_gated_halt_loop' requires all "
                            "GNN blocks to precede the global-module blocks. Got "
                            f"gnn_blocks={gnn_block_indices}, "
                            f"global_blocks={global_block_indices}."
                        )
                    self._loop_gated_gnn_block_indices = gnn_block_indices
                    self._loop_gated_global_block_indices = global_block_indices

                if mode == "staged_film_loop":
                    if self.processor_gnn_block_mask is None:
                        raise ValueError(
                            "loop_conditioning.mode='staged_film_loop' requires the "
                            "processor to be a block-based GNN with at least two "
                            "GNN blocks (got processor_gnn_block_mask=None)."
                        )
                    gnn_block_indices = [
                        i for i, on in enumerate(self.processor_gnn_block_mask) if on
                    ]
                    if len(gnn_block_indices) < 2:
                        raise ValueError(
                            "loop_conditioning.mode='staged_film_loop' needs at "
                            "least two GNN blocks (one for the 'early GNN' phase "
                            "and one for the 'late GNN' phase). Got "
                            f"gnn_blocks={gnn_block_indices}."
                        )
                    early_gnn_idx = gnn_block_indices[0]
                    late_gnn_idx = gnn_block_indices[-1]
                    if early_gnn_idx == late_gnn_idx:
                        raise ValueError(
                            "loop_conditioning.mode='staged_film_loop' needs two "
                            "DISTINCT GNN blocks (early != late). Got "
                            f"first_gnn_idx={early_gnn_idx} == last_gnn_idx="
                            f"{late_gnn_idx}."
                        )
                    if late_gnn_idx >= self.final_transformer_block_idx:
                        raise ValueError(
                            "loop_conditioning.mode='staged_film_loop' assumes the "
                            "final transformer block runs strictly after the last "
                            "GNN block, but got last_gnn_block_idx="
                            f"{late_gnn_idx} >= final_transformer_block_idx="
                            f"{self.final_transformer_block_idx}."
                        )
                    self._loop_staged_block_indices = {
                        "early": early_gnn_idx,
                        "late": late_gnn_idx,
                        "xfm": self.final_transformer_block_idx,
                    }
                if halting_enabled:
                    self.loop_pondernet_enabled = True
                    self.loop_halt_inference_threshold = float(
                        halt_cfg.get("inference_threshold", 0.5)
                    )
                    self.loop_halt_prior_lambda = float(
                        halt_cfg.get("prior_lambda", 0.4)
                    )
                    if not 0.0 < self.loop_halt_prior_lambda < 1.0:
                        raise ValueError(
                            "loop_conditioning.halting.prior_lambda must be in "
                            f"(0, 1); got {self.loop_halt_prior_lambda}"
                        )
                    self.loop_halt_reg_weight = float(
                        halt_cfg.get("reg_weight", 0.01)
                    )
                    self.loop_halt_reg_warmup_steps = int(
                        halt_cfg.get("reg_warmup_steps", 0)
                    )
                    # Counter incremented once per training-mode forward;
                    # used for linear warmup of the KL regulariser weight.
                    self.register_buffer(
                        "_ponder_step",
                        torch.zeros((), dtype=torch.long),
                        persistent=False,
                    )

                if mode == "film_loop":
                    from .loop_conditioner import LoopFiLMConditioner
                    self.loop_conditioner = LoopFiLMConditioner(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_film=zero_init_film,
                        halting_enabled=halting_enabled,
                        init_halt_bias=init_halt_bias,
                    )
                elif mode == "per_block_steer":
                    if self.num_blocks is None:
                        raise ValueError(
                            "loop_conditioning.mode='per_block_steer' requires a "
                            "block-based gnn processor (processor.num_blocks set)."
                        )
                    init_gate_bias = float(loop_cfg.get("init_gate_bias", 4.0))
                    from .loop_conditioner import LoopSteerConditioner
                    self.loop_conditioner = LoopSteerConditioner(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        num_blocks=self.num_blocks,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_film=zero_init_film,
                        init_gate_bias=init_gate_bias,
                    )
                elif mode == "transformer_loop":
                    if halting_enabled:
                        raise ValueError(
                            "loop_conditioning.halting.enabled=True is not supported "
                            "with mode='transformer_loop'."
                        )
                    transformer_dim = int(loop_cfg.get("transformer_dim", 128))
                    num_heads = int(loop_cfg.get("num_heads", 4))
                    ffn_dim = int(loop_cfg.get("ffn_dim", 256))
                    zero_init_output = bool(loop_cfg.get("zero_init_output", True))
                    from .loop_conditioner import LoopTransformerConnector
                    self.loop_conditioner = LoopTransformerConnector(
                        hidden_dim=self.hidden_dim,
                        raw_input_dim=self.input_dim,
                        num_iterations=num_iters,
                        transformer_dim=transformer_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_output=zero_init_output,
                    )
                elif mode == "identity":
                    # "No controller" K-iteration unroll: the (frozen) processor
                    # is run K times in a row, with the output of iteration k
                    # feeding directly into iteration k+1. No learnable adapter
                    # exists -> trainable_components should normally be just
                    # [decoder]. Used as the strict ablation baseline against
                    # which the FiLM / Transformer / dual controllers are
                    # measured.
                    self.loop_conditioner = None
                elif mode == "dual_film_loop":
                    from .loop_conditioner import DualLoopFiLMConditioner
                    self.loop_conditioner = DualLoopFiLMConditioner(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_film=zero_init_film,
                    )
                elif mode == "triple_film_loop":
                    from .loop_conditioner import TripleLoopFiLMConditioner
                    self.loop_conditioner = TripleLoopFiLMConditioner(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_film=zero_init_film,
                    )
                elif mode == "staged_film_loop":
                    from .loop_conditioner import StagedLoopFiLMConditioner
                    self.loop_conditioner = StagedLoopFiLMConditioner(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_film=zero_init_film,
                    )
                elif mode == "gnn_gated_halt_loop":
                    # Per-node gated residual loop with optional separate
                    # HaltModule (Variant B / iterGNN). See
                    # ``loop_conditioner.GatedResidualLoopConditioner`` for the
                    # gate-activation switch and
                    # ``loop_conditioner.HaltModule`` for the per-step logit
                    # head; the distribution shape (PonderNet cumulative vs.
                    # softmax over k) is selected in ``forward`` via
                    # ``self.loop_halt_distribution``.
                    gate_cfg = (loop_cfg.get("gate", {}) or {})
                    gate_activation = str(
                        gate_cfg.get("activation", "hard_concrete")
                    )
                    gate_init_bias = float(gate_cfg.get("init_gate_bias", 2.2))
                    from .loop_conditioner import (
                        GatedResidualLoopConditioner,
                        HaltModule,
                    )
                    self.loop_conditioner = GatedResidualLoopConditioner(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        conditioning_dim=conditioning_dim,
                        activation=gate_activation,
                        use_size_input=use_size_input,
                        init_gate_bias=gate_init_bias,
                        hc_beta=float(gate_cfg.get("hc_beta", 2.0 / 3.0)),
                        hc_gamma=float(gate_cfg.get("hc_gamma", -0.1)),
                        hc_zeta=float(gate_cfg.get("hc_zeta", 1.1)),
                        gumbel_temperature_start=float(
                            gate_cfg.get("gumbel_temperature_start", 1.0)
                        ),
                        gumbel_temperature_end=float(
                            gate_cfg.get("gumbel_temperature_end", 0.1)
                        ),
                        gumbel_temperature_anneal_steps=int(
                            gate_cfg.get("gumbel_temperature_anneal_steps", 0)
                        ),
                        gate_sharpness=float(
                            gate_cfg.get("gate_sharpness", 5.0)
                        ),
                    )
                    # Gate regularisation weights (read by the loss path).
                    self.loop_gate_l0_weight = float(
                        gate_cfg.get("l0_weight", 0.0)
                    )
                    self.loop_gate_l0_warmup_steps = int(
                        gate_cfg.get("l0_warmup_steps", 0)
                    )
                    self.loop_gate_entropy_weight = float(
                        gate_cfg.get("entropy_weight", 0.0)
                    )
                    self.loop_gate_entropy_warmup_steps = int(
                        gate_cfg.get("entropy_warmup_steps", 0)
                    )
                    # Variant B-specific knobs (only relevant when halting).
                    self.loop_halt_distribution = str(
                        halt_cfg.get("distribution", "pondernet_geometric")
                    )
                    if self.loop_halt_distribution not in (
                        "pondernet_geometric",
                        "softmax_over_k",
                    ):
                        raise ValueError(
                            "loop_conditioning.halting.distribution must be "
                            "'pondernet_geometric' or 'softmax_over_k'; got "
                            f"'{self.loop_halt_distribution}'"
                        )
                    self.loop_halt_inference_mode = str(
                        halt_cfg.get("inference_mode", "soft")
                    )
                    if self.loop_halt_inference_mode not in ("soft", "hard"):
                        raise ValueError(
                            "loop_conditioning.halting.inference_mode must be "
                            "'soft' or 'hard'; got "
                            f"'{self.loop_halt_inference_mode}'"
                        )
                    self.loop_decode_gradient_checkpoint = bool(
                        loop_cfg.get("decode_gradient_checkpoint", False)
                    )
                    if halting_enabled:
                        self.halt_module = HaltModule(
                            hidden_dim=self.hidden_dim,
                            num_iterations=num_iters,
                            conditioning_dim=conditioning_dim,
                            use_size_input=use_size_input,
                            use_gate_statistic=bool(
                                halt_cfg.get("use_gate_statistic", True)
                            ),
                            init_halt_bias=init_halt_bias,
                        )
                    else:
                        self.halt_module = None
                elif mode == "dual_transformer_loop":
                    transformer_dim = int(loop_cfg.get("transformer_dim", 128))
                    num_heads = int(loop_cfg.get("num_heads", 4))
                    ffn_dim = int(loop_cfg.get("ffn_dim", 256))
                    zero_init_output = bool(loop_cfg.get("zero_init_output", True))
                    from .loop_conditioner import DualLoopTransformerConnector
                    self.loop_conditioner = DualLoopTransformerConnector(
                        hidden_dim=self.hidden_dim,
                        num_iterations=num_iters,
                        transformer_dim=transformer_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        conditioning_dim=conditioning_dim,
                        use_size_input=use_size_input,
                        zero_init_output=zero_init_output,
                    )

    def _apply_processor_gnn_block(
        self,
        block_idx: int,
        inst_h: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        data,
        pos_for_global: Optional[torch.Tensor],
        processor_batch: Optional[torch.Tensor],
        size_cond_node: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run GNN layers for one processor block (sparse or dense layout)."""
        for layer_idx, gnn_layer in _gnn_layers_for_processor_block(
            self.gnn_blocks,
            block_idx,
            layers_per_block=self.layers_per_block,
            gnn_layer_ranges=self.processor_gnn_layer_ranges,
            gnn_block_mask=self.processor_gnn_block_mask,
        ):
            if (
                self.size_conditioning_enabled
                and self.processor_layer_films is not None
                and size_cond_node is not None
            ):
                inst_h = self.processor_layer_films[layer_idx](
                    inst_h, size_cond_node
                )
            gnn_data = {"edge_index": edge_index}
            if edge_attr is not None:
                gnn_data["edge_attr"] = edge_attr
            if pos_for_global is not None:
                gnn_data["pos"] = pos_for_global
            elif hasattr(data, "pos") and data.pos is not None:
                gnn_data["pos"] = data.pos
            if processor_batch is not None:
                gnn_data["batch"] = processor_batch
            inst_x_new = gnn_layer(gnn_data, x=inst_h)
            if self.use_residual:
                inst_h = inst_h + inst_x_new
            else:
                inst_h = torch.relu(inst_x_new)
        return inst_h

    def run_homogeneous_processor_blocks(
        self,
        inst_h: torch.Tensor,
        data,
        edge_index: torch.Tensor,
        *,
        edge_attr: Optional[torch.Tensor] = None,
        pos_for_global: Optional[torch.Tensor] = None,
        time_embed_for_global: Optional[torch.Tensor] = None,
        step_idx_for_global: Optional[int] = None,
        t_continuous: Optional[torch.Tensor] = None,
        size_cond_node: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the block-structured homogeneous GNN processor (no loop hooks).

        Matches the processor branch in ``forward`` / ``_run_processor_once`` when
        loop conditioning is disabled, including ``processor_gnn_block_mask``.
        """
        if self.processor_type != "gnn" or self.num_blocks is None:
            raise RuntimeError(
                "run_homogeneous_processor_blocks requires a block-based "
                f"homogeneous gnn processor (got processor_type="
                f"{self.processor_type!r}, num_blocks={self.num_blocks})."
            )

        processor_batch = data.batch if hasattr(data, "batch") else None

        for block_idx in range(self.num_blocks):
            inst_h = self._apply_processor_gnn_block(
                block_idx,
                inst_h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                data=data,
                pos_for_global=pos_for_global,
                processor_batch=processor_batch,
                size_cond_node=size_cond_node,
            )

            if self.global_modules is not None:
                global_module = self.global_modules[block_idx]
                inst_batch = data.batch if hasattr(data, "batch") else None
                inst_delta, _ = global_module(
                    inst_x=inst_h,
                    net_x=inst_h,
                    inst_batch=inst_batch,
                    inst_pos=pos_for_global,
                    data=data,
                    time_embed=time_embed_for_global,
                    step_idx=step_idx_for_global,
                    t_continuous=t_continuous,
                )
                if (
                    hasattr(self, "gated_adapters")
                    and self.gated_adapters is not None
                ):
                    gated_adapter = self.gated_adapters[block_idx]
                    inst_h = gated_adapter(
                        gnn_output=inst_h,
                        transformer_output=inst_delta,
                        time_embed=time_embed_for_global,
                    )
                else:
                    inst_h = inst_h + 0.1 * inst_delta

        return inst_h

    def forward(
        self,
        data: HeteroData,
        x_t: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
        return_encoder_output: bool = False,
        return_intermediate: bool = False,
    ):
        """Forward pass through encoder → processor → decoder."""
        # Safety check: diffusion coordinates passed to model must be normalized.
        if self.task == 'diffusion' and x_t is not None:
            _validate_normalized_coords(x_t, name="x_t", tol=1.1)

        # Handle heterogeneous vs homogeneous data
        is_hetero = hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data)

        if is_hetero:
            inst_feats = data['inst'].x
            net_feats = data['net'].x
            edge_index = data['inst', 'to', 'net'].edge_index
            edge_attr = data['inst', 'to', 'net'].edge_attr if hasattr(data['inst', 'to', 'net'], 'edge_attr') else None
        else:
            inst_feats = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        size_cond_node = None
        if self.size_conditioning_enabled and not is_hetero:
            size_cond_node = self.size_conditioner(
                batch=(data.batch if hasattr(data, "batch") else None),
                edge_index=edge_index,
                num_nodes=inst_feats.shape[0],
                use_num_edges=self.size_conditioning_use_num_edges,
            )

        def _prepare_encoder_features(feats: torch.Tensor) -> torch.Tensor:
            coords = feats[:, :2]
            rest_feats = feats[:, 2:]

            if self.coord_encoding_mode == 'raw':
                return feats

            if self.coord_encoding_mode == 'none':
                return rest_feats

            coords_for_rff = coords * self.coord_fourier_input_scale
            fourier_coords = self.coord_fourier(coords_for_rff)

            if self.coord_encoding_mode == 'fourier_only':
                return torch.cat([fourier_coords, rest_feats], dim=1)

            return torch.cat([coords, fourier_coords, rest_feats], dim=1)

        inst_feats_with_fourier = _prepare_encoder_features(inst_feats)

        inst_x = self.encoder(inst_feats_with_fourier)
        if self.size_conditioning_enabled and self.size_conditioning_apply_to_encoder and size_cond_node is not None:
            inst_x = self.encoder_condition_film(inst_x, size_cond_node)

        # Single time conditioning for diffusion: additive after encoder.
        # Also pass time embedding to global modules/adapters for time-aware fusion.
        time_embed_for_global = None
        step_idx_for_global = None
        if self.task == 'diffusion' and t_continuous is not None and getattr(self, 'time_embedding', None) is not None:
            time_embed_for_global = self.time_embedding(t_continuous)
            inst_x = inst_x + time_embed_for_global  # (N, hidden_dim)
            # Derive a scalar step index (used by global modules that operate at block/global level).
            diff_cfg = self.config.get('diffusion', {})
            num_steps = int(diff_cfg.get('num_steps', 1000))
            t_mean = float(t_continuous.mean().detach().item())
            t_mean = max(0.0, min(1.0, t_mean))
            step_idx_for_global = int(round(t_mean * max(num_steps - 1, 0)))

        if is_hetero:
            net_feats_with_fourier = _prepare_encoder_features(net_feats)
            
            net_x = self.encoder(net_feats_with_fourier)
            if self.task == 'diffusion' and t_continuous is not None and getattr(self, 'time_embedding', None) is not None:
                net_x = net_x + self.time_embedding(t_continuous)

        # Apply GNN-based encoder if configured (HUGE encoder pattern)
        if self.encoder_type in ('bipartite_gnn', 'gnn'):
            if self.encoder_type == 'bipartite_gnn' and is_hetero:
                # Bipartite GNN encoder: inst-net-inst message passing
                for i, block in enumerate(self.encoder_gnn_blocks):
                    inst_x, net_x = block(inst_x, net_x, edge_index, edge_attr)

                    if self.encoder_global_modules is not None:
                        global_module = self.encoder_global_modules[i]
                        inst_batch = data['inst'].batch if hasattr(data['inst'], 'batch') else None
                        net_batch = data['net'].batch if hasattr(data['net'], 'batch') else None
                        inst_delta, net_delta = global_module(
                            inst_x=inst_x,
                            net_x=net_x,
                            inst_batch=inst_batch,
                            net_batch=net_batch,
                            inst_pos=x_t,
                            data=data,
                            time_embed=time_embed_for_global,
                            step_idx=step_idx_for_global,
                            t_continuous=t_continuous,
                        )
                        inst_x = inst_x + inst_delta
                        net_x = net_x + net_delta
                        
            elif self.encoder_type == 'gnn' and not is_hetero:
                # Homogeneous GNN encoder: inst-inst message passing
                encoder_pos = x_t if x_t is not None else (data.pos if hasattr(data, 'pos') and data.pos is not None else None)
                encoder_batch = data.batch if hasattr(data, 'batch') else None
                if self.encoder_num_blocks is not None:
                    # Block-based encoder
                    for block_idx in range(self.encoder_num_blocks):
                        block_start_idx = block_idx * self.encoder_layers_per_block
                        block_end_idx = (block_idx + 1) * self.encoder_layers_per_block
                        
                        block_input = inst_x
                        
                        for layer_idx in range(block_start_idx, block_end_idx):
                            gnn_layer = self.encoder_gnn_blocks[layer_idx]
                            if self.size_conditioning_enabled and self.encoder_layer_films is not None and size_cond_node is not None:
                                inst_x = self.encoder_layer_films[layer_idx](inst_x, size_cond_node)
                            gnn_data = {'edge_index': edge_index}
                            if edge_attr is not None:
                                gnn_data['edge_attr'] = edge_attr
                            if encoder_pos is not None:
                                gnn_data['pos'] = encoder_pos
                            if encoder_batch is not None:
                                gnn_data['batch'] = encoder_batch
                            
                            inst_x_new = gnn_layer(gnn_data, x=inst_x)
                            
                            if self.encoder_use_residual:
                                inst_x = inst_x + inst_x_new
                                # No ReLU after residual (avoids killing negatives / positive bias)
                            else:
                                inst_x = inst_x_new
                                inst_x = torch.relu(inst_x)
                        
                        if self.encoder_global_modules is not None:
                            global_module = self.encoder_global_modules[block_idx]
                            inst_batch = data.batch if hasattr(data, 'batch') else None
                            inst_delta, _ = global_module(
                                inst_x=inst_x,
                                net_x=inst_x,
                                inst_batch=inst_batch,
                                inst_pos=x_t,
                                data=data,
                                time_embed=time_embed_for_global,
                                step_idx=step_idx_for_global,
                                t_continuous=t_continuous,
                            )
                            # CRITICAL FIX: Scale transformer residual to prevent explosion
                            inst_x = inst_x + 0.1 * inst_delta

                        if self.encoder_use_residual:
                            inst_x = inst_x + block_input
                else:
                    # Flat layer encoder
                    for i, block in enumerate(self.encoder_gnn_blocks):
                        if self.size_conditioning_enabled and self.encoder_layer_films is not None and size_cond_node is not None:
                            inst_x = self.encoder_layer_films[i](inst_x, size_cond_node)
                        gnn_data = {'edge_index': edge_index}
                        if edge_attr is not None:
                            gnn_data['edge_attr'] = edge_attr
                        if encoder_pos is not None:
                            gnn_data['pos'] = encoder_pos
                        if encoder_batch is not None:
                            gnn_data['batch'] = encoder_batch
                        
                        inst_x_new = block(gnn_data, x=inst_x)
                        
                        if self.encoder_use_residual:
                            inst_x = inst_x + inst_x_new
                            # No ReLU after residual (avoids killing negatives / positive bias)
                        else:
                            inst_x = inst_x_new
                            inst_x = torch.relu(inst_x)
                        
                        if self.encoder_global_modules is not None:
                            global_module = self.encoder_global_modules[i]
                            inst_batch = data.batch if hasattr(data, 'batch') else None
                            inst_delta, _ = global_module(
                                inst_x=inst_x,
                                net_x=inst_x,
                                inst_batch=inst_batch,
                                inst_pos=x_t,
                                data=data,
                                time_embed=time_embed_for_global,
                                step_idx=step_idx_for_global,
                                t_continuous=t_continuous,
                            )
                            # CRITICAL FIX: Scale transformer residual to prevent explosion
                            inst_x = inst_x + 0.1 * inst_delta
        
        # Return encoder output if requested (for MGM pretraining, etc.)
        if return_encoder_output:
            if is_hetero:
                return inst_x, net_x
            else:
                return inst_x, None
        
        # Cross-iteration state for dual-controller modes. The dual modes apply
        # one steering call BEFORE the final transformer block ("after GNNs")
        # and one AFTER it ("after final transformer"); the Transformer flavour
        # also threads a per-position memory token across iterations. These two
        # tensors are owned by `forward` and updated via `nonlocal` from inside
        # `_run_processor_once` so the K-loop callsite stays simple.
        dual_memory_pos0: Optional[torch.Tensor] = None
        dual_memory_pos1: Optional[torch.Tensor] = None

        def _run_processor_once(
            inst_h: torch.Tensor,
            net_h: Optional[torch.Tensor],
            pos_for_global: Optional[torch.Tensor],
            loop_iter: Optional[int] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Process (skip if no processor - encoder -> decoder directly)
            if self.processor_type is None:
                return inst_h, net_h

            # Per-block FiLM + gated residual hook is active only for the
            # `per_block_steer` loop mode and only when called from inside
            # the outer K-iteration loop (i.e. loop_iter is not None).
            use_per_block_steer = (
                self.loop_conditioning_enabled
                and self.loop_mode == "per_block_steer"
                and loop_iter is not None
                and self.loop_conditioner is not None
            )

            # Dual-controller modes: apply two steering calls strictly at the
            # final transformer block boundary (block_idx == final_transformer_block_idx):
            # one BEFORE the block runs (= "after GNNs") and one AFTER it runs
            # (= "after final transformer"). Active only inside the K-loop.
            use_dual_steer = (
                self.loop_conditioning_enabled
                and self.loop_mode in ("dual_film_loop", "dual_transformer_loop")
                and loop_iter is not None
                and self.loop_conditioner is not None
                and self.final_transformer_block_idx is not None
            )

            # Triple-controller mode: fire POST-block at three distinct
            # insertion points (first GNN block / last GNN block / final
            # transformer block) using the pre-computed
            # `_loop_triple_block_positions` map. All three fires happen at
            # the SAME structural place in the per-block loop (post-block,
            # after any per-block-steer hook), which keeps the wiring simple:
            # "post-block on block X with position
            # _loop_triple_block_positions[X]" if X is in the map.
            use_triple_steer = (
                self.loop_conditioning_enabled
                and self.loop_mode == "triple_film_loop"
                and loop_iter is not None
                and self.loop_conditioner is not None
                and self._loop_triple_block_positions is not None
            )

            if self.processor_type == 'bipartite_gnn' and is_hetero:
                for i, block in enumerate(self.gnn_blocks):
                    inst_h, net_h = block(inst_h, net_h, edge_index, edge_attr)

                    if self.global_modules is not None:
                        global_module = self.global_modules[i]
                        inst_batch = data['inst'].batch if hasattr(data['inst'], 'batch') else None
                        net_batch = data['net'].batch if hasattr(data['net'], 'batch') else None
                        inst_delta, net_delta = global_module(
                            inst_x=inst_h,
                            net_x=net_h,
                            inst_batch=inst_batch,
                            net_batch=net_batch,
                            inst_pos=pos_for_global,
                            data=data,
                            time_embed=time_embed_for_global,
                            step_idx=step_idx_for_global,
                            t_continuous=t_continuous,
                        )
                        inst_h = inst_h + inst_delta
                        net_h = net_h + net_delta
                return inst_h, net_h

            if self.processor_type == 'att_gnn' and not is_hetero:
                att_gnn_net = self.gnn_blocks[0]
                inst_batch = getattr(data, 'batch', None)
                if time_embed_for_global is not None and inst_batch is not None and getattr(data, 'num_graphs', 1) > 1:
                    ptr = data.ptr
                    t_per_graph = t_continuous[ptr[:-1]]
                    time_embed_for_attgnn = self.time_embedding(t_per_graph)
                elif time_embed_for_global is not None:
                    t_per_graph = t_continuous[0:1]
                    time_embed_for_attgnn = self.time_embedding(t_per_graph)
                else:
                    time_embed_for_attgnn = None

                if inst_batch is not None and getattr(data, 'num_graphs', 1) > 1:
                    inst_x_padded, counts = _pad_flat_to_batch(inst_h, inst_batch)
                    if pos_for_global is not None:
                        x_t_padded, _ = _pad_flat_to_batch(pos_for_global, inst_batch)
                    else:
                        x_t_padded = None
                    inst_x_padded = att_gnn_net(
                        inst_x_padded,
                        data,
                        t_embed=time_embed_for_attgnn,
                        positions=x_t_padded,
                    )
                    inst_h = _unbatch_to_flat(inst_x_padded, inst_batch, counts)
                else:
                    if inst_h.dim() == 2:
                        inst_h = inst_h.unsqueeze(0)
                    x_t_for_attgnn = pos_for_global.unsqueeze(0) if pos_for_global is not None and pos_for_global.dim() == 2 else pos_for_global
                    inst_h = att_gnn_net(
                        inst_h,
                        data,
                        t_embed=time_embed_for_attgnn,
                        positions=x_t_for_attgnn,
                    )
                    if inst_h.shape[0] == 1:
                        inst_h = inst_h.squeeze(0)
                return inst_h, net_h

            if self.processor_type == 'incidence_transformer' and not is_hetero:
                it_processor = self.gnn_blocks[0]
                result = it_processor(
                    x=inst_h,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=data.batch if hasattr(data, 'batch') else None,
                )
                inst_h = result['z_V']
                return inst_h, net_h

            if self.processor_type == 'gnn' and not is_hetero:
                processor_batch = data.batch if hasattr(data, 'batch') else None
                if hasattr(self, 'num_blocks') and self.num_blocks is not None:
                    for block_idx in range(self.num_blocks):
                        # Dual-controller: apply position-0 ("after GNNs")
                        # steering right before the final transformer block runs.
                        is_dual_steer_block = (
                            use_dual_steer
                            and block_idx == self.final_transformer_block_idx
                        )
                        if is_dual_steer_block:
                            nonlocal dual_memory_pos0
                            if self.loop_mode == "dual_film_loop":
                                inst_h = self.loop_conditioner(
                                    inst_h,
                                    position=0,
                                    iteration=loop_iter,
                                    batch=processor_batch,
                                    edge_index=edge_index,
                                    num_nodes=inst_h.shape[0],
                                )
                            else:  # dual_transformer_loop
                                inst_h, dual_memory_pos0 = self.loop_conditioner(
                                    inst_h,
                                    dual_memory_pos0,
                                    position=0,
                                    iteration=loop_iter,
                                    batch=processor_batch,
                                    edge_index=edge_index,
                                    num_nodes=inst_h.shape[0],
                                )

                        # LoopSteer: pre-block FiLM modulation. Save the block
                        # input so the post-block gate can blend (skip vs exec).
                        if use_per_block_steer:
                            block_input_for_steer = inst_h
                            inst_h = self.loop_conditioner.pre_block(
                                inst_h,
                                iteration=loop_iter,
                                block_idx=block_idx,
                                batch=processor_batch,
                                edge_index=edge_index,
                                num_nodes=inst_h.shape[0],
                            )

                        inst_h = self._apply_processor_gnn_block(
                            block_idx,
                            inst_h,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            data=data,
                            pos_for_global=pos_for_global,
                            processor_batch=processor_batch,
                            size_cond_node=size_cond_node,
                        )

                        if self.global_modules is not None:
                            global_module = self.global_modules[block_idx]
                            inst_batch = data.batch if hasattr(data, 'batch') else None
                            inst_delta, _ = global_module(
                                inst_x=inst_h,
                                net_x=inst_h,
                                inst_batch=inst_batch,
                                inst_pos=pos_for_global,
                                data=data,
                                time_embed=time_embed_for_global,
                                step_idx=step_idx_for_global,
                                t_continuous=t_continuous,
                            )
                            if hasattr(self, 'gated_adapters') and self.gated_adapters is not None:
                                gated_adapter = self.gated_adapters[block_idx]
                                inst_h = gated_adapter(
                                    gnn_output=inst_h,
                                    transformer_output=inst_delta,
                                    time_embed=time_embed_for_global,
                                )
                            else:
                                inst_h = inst_h + 0.1 * inst_delta

                        # LoopSteer: post-block gated residual blending.
                        if use_per_block_steer:
                            inst_h, _gate_values = self.loop_conditioner.post_block(
                                block_input_for_steer,
                                inst_h,
                                iteration=loop_iter,
                                block_idx=block_idx,
                                batch=processor_batch,
                                edge_index=edge_index,
                                num_nodes=inst_h.shape[0],
                            )

                        # Dual-controller: apply position-1 ("after final transformer")
                        # steering right after the final transformer block ran.
                        if is_dual_steer_block:
                            nonlocal dual_memory_pos1
                            if self.loop_mode == "dual_film_loop":
                                inst_h = self.loop_conditioner(
                                    inst_h,
                                    position=1,
                                    iteration=loop_iter,
                                    batch=processor_batch,
                                    edge_index=edge_index,
                                    num_nodes=inst_h.shape[0],
                                )
                            else:  # dual_transformer_loop
                                inst_h, dual_memory_pos1 = self.loop_conditioner(
                                    inst_h,
                                    dual_memory_pos1,
                                    position=1,
                                    iteration=loop_iter,
                                    batch=processor_batch,
                                    edge_index=edge_index,
                                    num_nodes=inst_h.shape[0],
                                )

                        # Triple-controller (FiLM-only): fire post-block at the
                        # block_idx -> position entry, if any. This handles all
                        # three insertion points (after first GNN block / after
                        # last GNN block / after final transformer block) in one
                        # uniform place. Applied AFTER per-block-steer so the
                        # composition order matches dual_film_loop's post-block
                        # branch above.
                        if use_triple_steer:
                            triple_position = (
                                self._loop_triple_block_positions.get(block_idx)
                            )
                            if triple_position is not None:
                                inst_h = self.loop_conditioner(
                                    inst_h,
                                    position=triple_position,
                                    iteration=loop_iter,
                                    batch=processor_batch,
                                    edge_index=edge_index,
                                    num_nodes=inst_h.shape[0],
                                )
                else:
                    for i, block in enumerate(self.gnn_blocks):
                        if self.size_conditioning_enabled and self.processor_layer_films is not None and size_cond_node is not None:
                            inst_h = self.processor_layer_films[i](inst_h, size_cond_node)
                        gnn_data = {'edge_index': edge_index}
                        if edge_attr is not None:
                            gnn_data['edge_attr'] = edge_attr
                        if pos_for_global is not None:
                            gnn_data['pos'] = pos_for_global
                        elif hasattr(data, 'pos') and data.pos is not None:
                            gnn_data['pos'] = data.pos
                        if processor_batch is not None:
                            gnn_data['batch'] = processor_batch
                        inst_x_new = block(gnn_data, x=inst_h)
                        if self.use_residual:
                            inst_h = inst_h + inst_x_new
                        else:
                            inst_h = torch.relu(inst_x_new)

                        if self.global_modules is not None:
                            global_module = self.global_modules[i]
                            inst_batch = data.batch if hasattr(data, 'batch') else None
                            inst_delta, _ = global_module(
                                inst_x=inst_h,
                                net_x=inst_h,
                                inst_batch=inst_batch,
                                inst_pos=pos_for_global,
                                data=data,
                                time_embed=time_embed_for_global,
                                step_idx=step_idx_for_global,
                                t_continuous=t_continuous,
                            )
                            if hasattr(self, 'gated_adapters') and self.gated_adapters is not None:
                                gated_adapter = self.gated_adapters[i]
                                inst_h = gated_adapter(
                                    gnn_output=inst_h,
                                    transformer_output=inst_delta,
                                    time_embed=time_embed_for_global,
                                )
                            else:
                                inst_h = inst_h + 0.1 * inst_delta
                return inst_h, net_h

            return inst_h, net_h

        def _run_single_processor_block(
            inst_h: torch.Tensor,
            block_idx: int,
            pos_for_global: Optional[torch.Tensor],
        ) -> torch.Tensor:
            """Run exactly ONE processor block (its GNN layers and/or its
            global module) on ``inst_h``.

            Mirrors the per-block branch inside ``_run_processor_once`` but
            with no per-block-steer / dual / triple controller hooks. Used
            by ``staged_film_loop`` where the staging structure replaces the
            interleaved in-loop hooks.
            """
            if self.processor_type != "gnn" or self.num_blocks is None:
                raise RuntimeError(
                    "_run_single_processor_block requires a block-based "
                    "homogeneous gnn processor (got processor_type="
                    f"{self.processor_type!r}, num_blocks={self.num_blocks})."
                )
            if not 0 <= block_idx < self.num_blocks:
                raise ValueError(
                    f"block_idx {block_idx} out of range "
                    f"[0, {self.num_blocks})."
                )

            processor_batch = data.batch if hasattr(data, "batch") else None
            inst_h = self._apply_processor_gnn_block(
                block_idx,
                inst_h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                data=data,
                pos_for_global=pos_for_global,
                processor_batch=processor_batch,
                size_cond_node=size_cond_node,
            )

            if self.global_modules is not None:
                global_module = self.global_modules[block_idx]
                inst_batch = data.batch if hasattr(data, "batch") else None
                inst_delta, _ = global_module(
                    inst_x=inst_h,
                    net_x=inst_h,
                    inst_batch=inst_batch,
                    inst_pos=pos_for_global,
                    data=data,
                    time_embed=time_embed_for_global,
                    step_idx=step_idx_for_global,
                    t_continuous=t_continuous,
                )
                if (
                    hasattr(self, "gated_adapters")
                    and self.gated_adapters is not None
                ):
                    gated_adapter = self.gated_adapters[block_idx]
                    inst_h = gated_adapter(
                        gnn_output=inst_h,
                        transformer_output=inst_delta,
                        time_embed=time_embed_for_global,
                    )
                else:
                    inst_h = inst_h + 0.1 * inst_delta

            return inst_h

        def _decode_outputs(inst_h: torch.Tensor, net_h: Optional[torch.Tensor]):
            if self.two_scale_velocity:
                v_die = self.decoder_die(inst_h)
                v_cell = self.decoder_cell(inst_h)
                inst_output_local = torch.cat([v_die, v_cell], dim=-1)
            else:
                inst_output_local = self.decoder(inst_h)

            if is_hetero:
                if self.two_scale_velocity:
                    net_v_die = self.decoder_die(net_h)
                    net_v_cell = self.decoder_cell(net_h)
                    net_output_local = torch.cat([net_v_die, net_v_cell], dim=-1)
                else:
                    net_output_local = self.decoder(net_h)
            else:
                out_dim = self._output_dim_effective if self.two_scale_velocity else self.output_dim
                net_output_local = torch.zeros(0, out_dim, device=inst_output_local.device)
            return inst_output_local, net_output_local

        if self.loop_conditioning_enabled and not is_hetero:
            # Unroll the (frozen) processor for K iterations. Between
            # iterations (film_loop) or inside each block (per_block_steer),
            # a tiny learned adapter steers the hidden state conditioned on
            # (iteration, graph size, [block]).
            loop_batch = data.batch if hasattr(data, 'batch') else None
            num_nodes = inst_x.shape[0]
            use_grad_ckpt = (
                self.loop_gradient_checkpoint
                and self.training
                and torch.is_grad_enabled()
            )

            def _ckpt_film(inp: torch.Tensor) -> torch.Tensor:
                # Closure captures x_t; processor is shared across iterations.
                out, _ = _run_processor_once(inp, None, x_t)
                return out

            def _ckpt_steer(inp: torch.Tensor, loop_iter_t: torch.Tensor) -> torch.Tensor:
                # loop_iter is passed as a 0-d long tensor so checkpoint sees
                # only tensor positional args; we unpack to int inside.
                out, _ = _run_processor_once(
                    inp, None, x_t, loop_iter=int(loop_iter_t.item())
                )
                return out

            def _decode_inst_only(h: torch.Tensor) -> torch.Tensor:
                """Apply optional decoder-side size FiLM, then run the decoder.

                Used for PonderNet to decode the hidden state after every
                iteration of the K-loop with the same head we use at the end.
                """
                h_dec = h
                if (
                    self.size_conditioning_enabled
                    and self.size_conditioning_apply_to_decoder
                    and size_cond_node is not None
                ):
                    h_dec = self.decoder_condition_film(h_dec, size_cond_node)
                if self.two_scale_velocity:
                    return torch.cat(
                        [self.decoder_die(h_dec), self.decoder_cell(h_dec)], dim=-1
                    )
                return self.decoder(h_dec)

            # ------------------------------------------------------------
            # gnn_gated_halt_loop (Variant A + Variant B / iterGNN).
            # Variant A (halting disabled): K iterations of GNN-only blocks
            # with a per-node gated residual; the fixed post-loop transformer
            # runs ONCE; single decode at the end.
            # Variant B (halting enabled): same K-loop, but the post-loop
            # transformer + decoder are run after EACH iteration to produce
            # ``pred_k``, and a per-step halt logit from ``self.halt_module``
            # builds the halt distribution ``p_k`` (PonderNet OR softmax over
            # k). The model returns ``expected_pred = sum_k p_k * pred_k``
            # and stores ``self._last_ponder_aux`` for the loss to consume.
            # ------------------------------------------------------------
            if self.loop_mode == "gnn_gated_halt_loop":
                from .loop_conditioner import _graph_size_features_from_batch

                gnn_indices = self._loop_gated_gnn_block_indices or []
                global_indices = self._loop_gated_global_block_indices or []
                K = self.loop_num_iterations

                # GNN-only one-iteration helper.
                def _run_gnn_blocks_only(h_in: torch.Tensor) -> torch.Tensor:
                    h_out = h_in
                    for blk_idx in gnn_indices:
                        h_out = _run_single_processor_block(h_out, blk_idx, x_t)
                    return h_out

                # Post-loop fixed transformer (runs the global-module blocks
                # of the processor in order).
                def _run_global_blocks_only(h_in: torch.Tensor) -> torch.Tensor:
                    h_out = h_in
                    for blk_idx in global_indices:
                        h_out = _run_single_processor_block(h_out, blk_idx, x_t)
                    return h_out

                def _decode_after_globals(h_in: torch.Tensor) -> torch.Tensor:
                    return _decode_inst_only(_run_global_blocks_only(h_in))

                # Per-iter ``[N]`` gate values and logits, stacked at the end.
                gate_per_iter_list: List[torch.Tensor] = []
                gate_logit_per_iter_list: List[torch.Tensor] = []

                # Shared graph-size stats used by gate aux + halt module + p_k.
                _stats_, node_batch_g, num_graphs_g = (
                    _graph_size_features_from_batch(
                        loop_batch, edge_index, num_nodes
                    )
                )

                use_decode_ckpt = (
                    self.loop_decode_gradient_checkpoint
                    and self.training
                    and torch.is_grad_enabled()
                )

                if self.loop_pondernet_enabled:
                    # ----- Variant B / iterGNN -----------------------------
                    pred_per_iter: List[torch.Tensor] = []
                    halt_logits_list: List[torch.Tensor] = []
                    for k in range(K):
                        if use_grad_ckpt:
                            h_raw = torch.utils.checkpoint.checkpoint(
                                _run_gnn_blocks_only,
                                inst_x,
                                use_reentrant=False,
                            )
                        else:
                            h_raw = _run_gnn_blocks_only(inst_x)
                        inst_x, g_k, gate_logit_k = self.loop_conditioner(
                            h_prev=inst_x,
                            h_new=h_raw,
                            iteration=k,
                            batch=loop_batch,
                            edge_index=edge_index,
                            num_nodes=num_nodes,
                        )
                        gate_per_iter_list.append(g_k)
                        gate_logit_per_iter_list.append(gate_logit_k)

                        if use_decode_ckpt:
                            pred_k = torch.utils.checkpoint.checkpoint(
                                _decode_after_globals,
                                inst_x,
                                use_reentrant=False,
                            )
                        else:
                            pred_k = _decode_after_globals(inst_x)
                        pred_per_iter.append(pred_k)

                        halt_logit_k = self.halt_module(
                            inst_x,
                            iteration=k,
                            batch=loop_batch,
                            edge_index=edge_index,
                            num_nodes=num_nodes,
                            gate_per_node=g_k.detach(),
                        )
                        halt_logits_list.append(halt_logit_k)

                    # Build halt distribution p_k over k in fp32.
                    halt_logits = torch.stack(halt_logits_list, dim=0).float()  # [K, G]
                    if self.loop_halt_distribution == "pondernet_geometric":
                        halt_probs_list: List[torch.Tensor] = []
                        remainders = torch.ones(
                            num_graphs_g, device=inst_x.device, dtype=torch.float32
                        )
                        for k in range(K):
                            if k == K - 1:
                                lam_k = torch.ones_like(halt_logits[k])
                            else:
                                lam_k = torch.sigmoid(halt_logits[k])
                            p_k = remainders * lam_k
                            halt_probs_list.append(p_k)
                            remainders = remainders * (1.0 - lam_k)
                        halt_probs = torch.stack(halt_probs_list, dim=0)
                    else:  # 'softmax_over_k'
                        halt_probs = torch.softmax(halt_logits, dim=0)

                    pred_per_iter_t_fp32 = torch.stack(pred_per_iter, dim=0).float()
                    p_k_node = halt_probs[:, node_batch_g]  # [K, N]
                    expected_pred = (
                        p_k_node.unsqueeze(-1) * pred_per_iter_t_fp32
                    ).sum(dim=0)

                    gate_per_iter = torch.stack(gate_per_iter_list, dim=0)
                    gate_logit_per_iter = torch.stack(gate_logit_per_iter_list, dim=0)

                    self._last_ponder_aux = {
                        "halt_probs": halt_probs,
                        "halt_logits": halt_logits,
                        "pred_per_iter": pred_per_iter_t_fp32,
                        "node_batch": node_batch_g,
                        "distribution": self.loop_halt_distribution,
                    }
                    self._last_gate_aux = {
                        "gate_per_iter": gate_per_iter,
                        "gate_logit_per_iter": gate_logit_per_iter,
                        "gate_activation": self.loop_conditioner.activation,
                        "node_batch": node_batch_g,
                    }
                    if self.training:
                        self._ponder_step += 1

                    out_dim_eff = (
                        self._output_dim_effective if self.two_scale_velocity
                        else self.output_dim
                    )
                    net_output_local = torch.zeros(
                        0, out_dim_eff, device=expected_pred.device
                    )
                    return expected_pred, net_output_local

                # ----- Variant A / gated_loop (no halting) -----------------
                for k in range(K):
                    if use_grad_ckpt:
                        h_raw = torch.utils.checkpoint.checkpoint(
                            _run_gnn_blocks_only,
                            inst_x,
                            use_reentrant=False,
                        )
                    else:
                        h_raw = _run_gnn_blocks_only(inst_x)
                    inst_x, g_k, gate_logit_k = self.loop_conditioner(
                        h_prev=inst_x,
                        h_new=h_raw,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )
                    gate_per_iter_list.append(g_k)
                    gate_logit_per_iter_list.append(gate_logit_k)

                gate_per_iter = torch.stack(gate_per_iter_list, dim=0)
                gate_logit_per_iter = torch.stack(gate_logit_per_iter_list, dim=0)
                self._last_gate_aux = {
                    "gate_per_iter": gate_per_iter,
                    "gate_logit_per_iter": gate_logit_per_iter,
                    "gate_activation": self.loop_conditioner.activation,
                    "node_batch": node_batch_g,
                }
                # Post-loop fixed transformer runs ONCE. Fall through to the
                # standard decoder_condition_film + _decode_outputs tail at the
                # bottom of this branch.
                inst_x = _run_global_blocks_only(inst_x)
                net_x = None
                if (
                    self.size_conditioning_enabled
                    and self.size_conditioning_apply_to_decoder
                    and size_cond_node is not None
                ):
                    inst_x = self.decoder_condition_film(inst_x, size_cond_node)
                return _decode_outputs(inst_x, net_x if is_hetero else None)

            # ------------------------------------------------------------
            # PonderNet halting path (mode='film_loop' only).
            # Decode after every iteration k, accumulate per-graph halt
            # probabilities p_k via PonderNet recurrence, and store
            # (halt_probs, pred_per_iter, node_batch) in self._last_ponder_aux
            # for the loss to consume. The model returns the *expected*
            # output sum_k p_k_node * pred_k_node so eval/metrics paths see
            # a single [N, D] tensor as before.
            # ------------------------------------------------------------
            if self.loop_pondernet_enabled:
                if self.loop_mode == "transformer_loop":
                    raise RuntimeError(
                        "PonderNet halting (loop_conditioning.halting.enabled=True) "
                        "is not supported with mode='transformer_loop'."
                    )
                K = self.loop_num_iterations
                pred_per_iter: List[torch.Tensor] = []
                halt_probs_list: List[torch.Tensor] = []
                # Resolve num_graphs / node_batch once via the conditioner
                # helper (also used inside the FiLM call below).
                _stats, node_batch_pn, num_graphs_pn = (
                    self.loop_conditioner._graph_size_features(
                        loop_batch, edge_index, num_nodes
                    )
                )
                # Halt probabilities are the product of K sigmoids; under AMP
                # this chain underflows in fp16 (sigmoid saturates -> 1 - lam
                # rounds to 0 -> remainders = 0 -> log(p) = -inf -> NaN in
                # the KL). Pin the entire halt-prob recurrence to fp32 so the
                # loss is numerically stable. Cost: K per-graph scalar ops.
                remainders = torch.ones(
                    num_graphs_pn, device=inst_x.device, dtype=torch.float32
                )

                for k in range(K):
                    inst_x_pre, halt_logit_k = self.loop_conditioner(
                        inst_x,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )
                    if use_grad_ckpt:
                        inst_x_new = torch.utils.checkpoint.checkpoint(
                            _ckpt_film, inst_x_pre, use_reentrant=False
                        )
                    else:
                        inst_x_new, _ = _run_processor_once(inst_x_pre, None, x_t)
                    if self.loop_use_outer_residual:
                        inst_x = inst_x + inst_x_new
                    else:
                        inst_x = inst_x_new

                    pred_k = _decode_inst_only(inst_x)
                    pred_per_iter.append(pred_k)

                    # PonderNet recurrence (per-graph), in fp32:
                    #   lambda_k = sigmoid(halt_logit_k)            for k < K-1
                    #   lambda_{K-1} = 1                            (must halt)
                    #   p_k        = remainders * lambda_k
                    #   remainders = remainders * (1 - lambda_k)
                    # The last-iter halt-logit is replaced by 1 directly so
                    # the gradient wrt halt_logit_{K-1} is exactly 0 (we never
                    # use it).
                    halt_logit_k_fp32 = halt_logit_k.float()
                    if k == K - 1:
                        lam_k = torch.ones_like(halt_logit_k_fp32)
                    else:
                        lam_k = torch.sigmoid(halt_logit_k_fp32)
                    p_k = remainders * lam_k
                    halt_probs_list.append(p_k)
                    remainders = remainders * (1.0 - lam_k)

                # halt_probs: [K, G] (fp32),  pred_per_iter_t: [K, N, D] (model dtype).
                halt_probs = torch.stack(halt_probs_list, dim=0)
                pred_per_iter_t = torch.stack(pred_per_iter, dim=0)
                # Promote per-iter predictions to fp32 for the weighted sum
                # so the returned expected output is numerically clean
                # regardless of AMP state. This also keeps the loss path
                # fp32 (target_norm is fp32, halt_probs is fp32).
                pred_per_iter_t_fp32 = pred_per_iter_t.float()
                p_k_node = halt_probs[:, node_batch_pn]  # [K, N], fp32
                expected_pred = (
                    p_k_node.unsqueeze(-1) * pred_per_iter_t_fp32
                ).sum(dim=0)

                # Bookkeep for the loss path. Halt probs and per-iter
                # predictions are pinned to fp32 so the KL and expected MSE
                # are stable under AMP (see the rationale comment above).
                self._last_ponder_aux = {
                    "halt_probs": halt_probs,
                    "pred_per_iter": pred_per_iter_t_fp32,
                    "node_batch": node_batch_pn,
                }

                # Bump the warmup counter (used by the loss to anneal the KL
                # weight). Only counts in training mode so eval doesn't move
                # the schedule.
                if self.training:
                    self._ponder_step += 1

                # Skip the trailing decoder_condition_film + _decode_outputs
                # path (we already decoded per iteration). Build the matching
                # zero net_output for homogeneous outputs.
                out_dim_eff = (
                    self._output_dim_effective if self.two_scale_velocity
                    else self.output_dim
                )
                net_output_local = torch.zeros(
                    0, out_dim_eff, device=expected_pred.device
                )
                return expected_pred, net_output_local

            # ------------------------------------------------------------
            # Standard (non-halting) loop: K iterations, single decode.
            # ------------------------------------------------------------
            connector_memory = None  # used by transformer_loop mode only
            # Reset dual-mode cross-iteration memories at the start of the loop.
            # `_run_processor_once` reads/writes these via `nonlocal` for the
            # dual_transformer_loop mode (None on first iteration -> learned init).
            dual_memory_pos0 = None
            dual_memory_pos1 = None

            if self.loop_mode == "staged_film_loop":
                # Staged 3-phase loop: each phase loops its own block K times
                # with a POST-block LOCAL FiLM (head #1, #3, #5), and a
                # one-shot TRANSITION FiLM fires between adjacent phases
                # (head #2, #4). Bypasses the standard outer K-loop entirely:
                # the staging itself IS the iteration structure.
                #
                # use_outer_residual is intentionally ignored here (the
                # concept doesn't apply: there is no "outer" iteration
                # producing inst_x_new for residual addition).
                K = self.loop_num_iterations
                early_idx = self._loop_staged_block_indices["early"]
                late_idx = self._loop_staged_block_indices["late"]
                xfm_idx = self._loop_staged_block_indices["xfm"]

                # --- Phase 1: early GNN block, looped K times ------------
                for k in range(K):
                    inst_x = _run_single_processor_block(
                        inst_x, early_idx, x_t
                    )
                    inst_x = self.loop_conditioner.forward_local(
                        inst_x,
                        phase=0,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )

                # --- Transition 1: early -> late (one-shot) --------------
                inst_x = self.loop_conditioner.forward_transition(
                    inst_x,
                    transition=0,
                    batch=loop_batch,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                )

                # --- Phase 2: late GNN block, looped K times -------------
                for k in range(K):
                    inst_x = _run_single_processor_block(
                        inst_x, late_idx, x_t
                    )
                    inst_x = self.loop_conditioner.forward_local(
                        inst_x,
                        phase=1,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )

                # --- Transition 2: late -> transformer (one-shot) --------
                inst_x = self.loop_conditioner.forward_transition(
                    inst_x,
                    transition=1,
                    batch=loop_batch,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                )

                # --- Phase 3: Transformer block, looped K times ----------
                for k in range(K):
                    inst_x = _run_single_processor_block(
                        inst_x, xfm_idx, x_t
                    )
                    inst_x = self.loop_conditioner.forward_local(
                        inst_x,
                        phase=2,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )

                net_x = None
                if (
                    self.size_conditioning_enabled
                    and self.size_conditioning_apply_to_decoder
                    and size_cond_node is not None
                ):
                    inst_x = self.decoder_condition_film(inst_x, size_cond_node)
                return _decode_outputs(inst_x, net_x if is_hetero else None)

            for k in range(self.loop_num_iterations):
                if self.loop_mode == "film_loop":
                    inst_x_pre, _halt_logit_unused = self.loop_conditioner(
                        inst_x,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )
                    if use_grad_ckpt:
                        inst_x_new = torch.utils.checkpoint.checkpoint(
                            _ckpt_film, inst_x_pre, use_reentrant=False
                        )
                    else:
                        inst_x_new, _ = _run_processor_once(inst_x_pre, None, x_t)
                elif self.loop_mode == "transformer_loop":
                    inst_x_pre, connector_memory = self.loop_conditioner(
                        inst_x,
                        inst_feats,
                        connector_memory,
                        iteration=k,
                        batch=loop_batch,
                        edge_index=edge_index,
                        num_nodes=num_nodes,
                    )
                    if use_grad_ckpt:
                        inst_x_new = torch.utils.checkpoint.checkpoint(
                            _ckpt_film, inst_x_pre, use_reentrant=False
                        )
                    else:
                        inst_x_new, _ = _run_processor_once(inst_x_pre, None, x_t)
                elif self.loop_mode == "identity":
                    # Pure-stack ablation: run the (frozen) processor with no
                    # learnable controller in between. The output of iteration
                    # k feeds directly into iteration k+1's processor.
                    if use_grad_ckpt:
                        inst_x_new = torch.utils.checkpoint.checkpoint(
                            _ckpt_film, inst_x, use_reentrant=False
                        )
                    else:
                        inst_x_new, _ = _run_processor_once(inst_x, None, x_t)
                elif self.loop_mode in (
                    "dual_film_loop",
                    "dual_transformer_loop",
                    "triple_film_loop",
                ):
                    # Multi-controller modes fire INSIDE the processor at
                    # block boundaries; just hand `loop_iter=k` to
                    # `_run_processor_once`. Memory threading for the
                    # dual_transformer_loop flavour happens via the
                    # `dual_memory_pos*` nonlocals captured by
                    # `_run_processor_once`. The triple_film_loop variant has
                    # no cross-iteration memory: each iteration sees a fresh
                    # FiLM-only state derived from (k, position, graph size).
                    inst_x_new, _ = _run_processor_once(
                        inst_x, None, x_t, loop_iter=k
                    )
                else:  # "per_block_steer"
                    if use_grad_ckpt:
                        loop_iter_t = torch.tensor(
                            k, dtype=torch.long, device=inst_x.device
                        )
                        inst_x_new = torch.utils.checkpoint.checkpoint(
                            _ckpt_steer, inst_x, loop_iter_t, use_reentrant=False
                        )
                    else:
                        inst_x_new, _ = _run_processor_once(
                            inst_x, None, x_t, loop_iter=k
                        )
                if self.loop_use_outer_residual:
                    inst_x = inst_x + inst_x_new
                else:
                    inst_x = inst_x_new
            net_x = None
        else:
            inst_x, net_x = _run_processor_once(inst_x, net_x if is_hetero else None, x_t)
        if self.size_conditioning_enabled and self.size_conditioning_apply_to_decoder and size_cond_node is not None:
            inst_x = self.decoder_condition_film(inst_x, size_cond_node)
        return _decode_outputs(inst_x, net_x if is_hetero else None)
