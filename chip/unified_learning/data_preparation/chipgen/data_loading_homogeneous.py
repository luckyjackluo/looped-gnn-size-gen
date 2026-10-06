"""Direct data loading for chipdiffusion-style homogeneous graphs (no HeteroData conversion)."""

import math
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurable node features: name -> output dim (for auto input_dim)
# ---------------------------------------------------------------------------
# Log scale for graph stats so 1..~10000 maps to roughly [0, 1]
GRAPH_STATS_LOG_SCALE = 9.21  # log(1 + 10000) ≈ 9.21

NODE_FEATURE_DIMS: Dict[str, int] = {
    "w_norm_d": 1,
    "h_norm_d": 1,
    "w_log_rel": 1,
    "h_log_rel": 1,
    "aspect_ratio_norm": 1,
    "area_norm": 1,
    "area_physical_unit": 1,  # N-invariant: log(area/L^2) then tanh/2 -> [0,1]
    "chip_size": 2,           # chip_w_rel, chip_h_rel
    "log_deg": 1,
    "pin_count": 1,
    "type_indicators": 3,     # is_stdcell, is_macro, is_port
    "num_nodes": 1,           # graph-level: log(1+N) / scale, broadcast to each node
    "num_edges": 1,           # graph-level: log(1+E) / scale, broadcast to each node
    "eigenvectors": 0,        # variable dim: num_eigenvectors from dataset config
    "canvas_aspect_ratio": 1, # tanh(log(W/H) / 2) in (-1,1): encodes canvas shape for all nodes
    "port_loc": 2,            # [-1,1]-normalised (x,y) for port nodes, (0,0) for all others
    "metis_partition_id_norm": 1,  # partition_id / K in [0,1): which Metis partition this node belongs to
    "metis_partition_pe": 0,       # variable dim: max_k cols of quotient-graph Laplacian eigenvectors
}

VALID_NODE_FEATURES = frozenset(NODE_FEATURE_DIMS.keys())


def _lap_pe_stem_for_pickle(
    pickle_path: Union[str, Path],
    lap_eigenvector_stem: Optional[str],
) -> str:
    """Stem used for ``{stem}_lap_pe.pickle`` next to ``eigenvector_dir``."""
    if lap_eigenvector_stem:
        return lap_eigenvector_stem
    return Path(pickle_path).stem


def get_default_node_features() -> List[str]:
    """Default feature list when dataset.node_features is omitted."""
    base = [
        "w_norm_d", "h_norm_d", "w_log_rel", "h_log_rel",
        "aspect_ratio_norm", "area_norm",
    ]
    base = base + ["chip_size", "log_deg", "pin_count", "type_indicators"]
    return base


def get_encoder_input_dim(dataset_cfg: Dict[str, Any]) -> int:
    """
    Compute encoder input_dim from dataset config.
    input_dim = 2 (coords) + sum(dims of selected node_features).
    Special cases:
      'eigenvectors'       contributes dataset_cfg.num_eigenvectors dims.
      'metis_partition_pe' contributes dataset_cfg.metis_max_k dims.
    """
    node_features = dataset_cfg.get("node_features")
    num_eigen = dataset_cfg.get("num_eigenvectors", 0) or 0
    metis_max_k = dataset_cfg.get("metis_max_k", 64) or 64

    if node_features is None:
        node_features = get_default_node_features()

    for f in node_features:
        if f not in VALID_NODE_FEATURES:
            raise ValueError(
                f"Unknown node feature '{f}'. Valid: {sorted(VALID_NODE_FEATURES)}"
            )
    variable_features = {"eigenvectors", "metis_partition_pe"}
    static_dim = sum(NODE_FEATURE_DIMS[f] for f in node_features if f not in variable_features)
    eigen_dim = num_eigen if "eigenvectors" in node_features else 0
    metis_pe_dim = metis_max_k if "metis_partition_pe" in node_features else 0
    return 2 + static_dim + eigen_dim + metis_pe_dim


def _compute_global_norm_node_features(
    x: torch.Tensor,
    sizes: torch.Tensor,
    *,
    chip_size: Optional[torch.Tensor] = None,
    size_stats: Optional[dict] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute normalized node features using global, fixed, timestep-invariant normalization.
    
    Implements the spec for from-scratch diffusion:
      - Coordinate normalization: z = (x - c) / S where c = (W/2, H/2), S = diag(W/2, H/2)
      - Size normalization: log-scale with median and MAD (median absolute deviation)
      - Node features: [z_x, z_y, w_norm, h_norm, r, a] where r = log(w/h), a = log(wh)
    
    Args:
        x: (V, 2) raw center coordinates
        sizes: (V, 2) instance sizes [width, height]
        chip_size: (2,) chip size [W, H]. If None, uses bounding box from x.
        size_stats: Optional dict with precomputed size statistics:
            - m_w, m_h: medians
            - s_w, s_h: MAD values
            If None, computes from current sizes.
        eps: Small epsilon for numerical stability
    
    Returns:
        node_feat: (V, 6) normalized features [z_x, z_y, w_norm, h_norm, r, a]
        s: (V, 2) anisotropic scale [W/2, H/2] per node (same for all nodes)
        mu: (V, 2) center [W/2, H/2] per node (same for all nodes)
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must be (V,2), got {tuple(x.shape)}")
    if sizes.ndim != 2 or sizes.shape[1] < 2:
        raise ValueError(f"sizes must be (V,>=2) with [w,h,...], got {tuple(sizes.shape)}")
    V = x.shape[0]
    device = x.device
    dtype = x.dtype

    widths = sizes[:, 0].to(dtype=dtype)
    heights = sizes[:, 1].to(dtype=dtype)

    # 1) Coordinate normalization: global, fixed, from chip_size
    if chip_size is not None:
        chip_size = chip_size.to(device=device, dtype=dtype)
        if chip_size.numel() != 2:
            raise ValueError(f"chip_size must have 2 elements, got {tuple(chip_size.shape)}")
        W, H = chip_size[0], chip_size[1]
    else:
        # Fallback: use bounding box from coordinates
        span = (x.max(dim=0).values - x.min(dim=0).values).clamp(min=eps)
        W, H = span[0], span[1]
    
    # Global center and scale (same for all nodes)
    c = torch.tensor([W / 2.0, H / 2.0], device=device, dtype=dtype)  # (2,)
    S = torch.tensor([W / 2.0, H / 2.0], device=device, dtype=dtype)  # (2,)
    
    # Normalize coordinates: z = (x - c) / S
    z = (x - c.unsqueeze(0)) / S.unsqueeze(0)  # (V, 2)
    z_x = z[:, 0:1]  # (V, 1)
    z_y = z[:, 1:2]  # (V, 1)
    
    # mu and s are global (same for all nodes)
    mu = c.unsqueeze(0).expand(V, 2)  # (V, 2)
    s = S.unsqueeze(0).expand(V, 2)  # (V, 2)
    
    # 2) Size normalization: log-scale with median and MAD
    if size_stats is not None:
        m_w = size_stats['m_w']
        m_h = size_stats['m_h']
        s_w = size_stats['s_w']
        s_h = size_stats['s_h']
    else:
        # Compute statistics from current sizes
        log_w = torch.log(torch.clamp(widths, min=eps))
        log_h = torch.log(torch.clamp(heights, min=eps))
        
        m_w = torch.median(log_w)
        m_h = torch.median(log_h)
        
        # MAD (median absolute deviation)
        mad_w = torch.median(torch.abs(log_w - m_w))
        mad_h = torch.median(torch.abs(log_h - m_h))
        
        # If MAD is too small (near-constant sizes), use std as fallback
        # Use raw MAD/std; only floor at eps to avoid division by zero (no min=0.1 to preserve scale)
        s_w = mad_w if mad_w > 1e-3 else log_w.std()
        s_h = mad_h if mad_h > 1e-3 else log_h.std()
        s_w = s_w.clamp(min=eps)
        s_h = s_h.clamp(min=eps)
    
    # Normalize sizes: (log(w) - log(m_w)) / s_w
    w_norm = (torch.log(torch.clamp(widths.unsqueeze(-1), min=eps)) - m_w) / s_w  # (V, 1)
    h_norm = (torch.log(torch.clamp(heights.unsqueeze(-1), min=eps)) - m_h) / s_h  # (V, 1)
    
    # 3) Additional features: aspect ratio and area
    r = torch.log(torch.clamp(widths / torch.clamp(heights, min=eps), min=eps)).unsqueeze(-1)  # (V, 1)
    a = torch.log(torch.clamp(widths * heights, min=eps)).unsqueeze(-1)  # (V, 1)
    
    # Pack node features: [z_x, z_y, w_norm, h_norm, r, a]
    node_feat = torch.cat([z_x, z_y, w_norm, h_norm, r, a], dim=1)  # (V, 6)
    
    return node_feat, s, mu


def load_pickle_file_as_data(
    pickle_path: Union[str, Path],
    normalize_positions: bool = True,
    normalize_by_graph_stats: bool = False,
    add_graph_stats: bool = False,
    init_noise_std: float = 0.0,
    device: str = "cpu",
    normalization_stats: Optional[dict] = None,
    diffuse_macros: bool = False,
    diffuse_macros_only: bool = False,
    eigenvector_dir: Optional[Union[str, Path]] = None,
    num_eigenvectors: Optional[int] = None,
    min_graph_size: Optional[int] = None,
    max_graph_size: Optional[int] = None,
    node_features: Optional[List[str]] = None,
    placement_from_v5_placer: bool = False,
    placer_seed: Optional[int] = None,
    zero_edge_attr: bool = False,
    normalize_edge_attr_with_coordinates: bool = True,
    metis_max_k: int = 64,
    require_aux_features: bool = True,
    lap_eigenvector_stem: Optional[str] = None,
    load_graph_lap_eigenvectors: Optional[bool] = None,
) -> List[Tuple[Data, torch.Tensor]]:
    """
    Load pickle file containing (x, cond) tuples directly as Data objects.
    
    This function loads chipdiffusion-style pickle files without converting to HeteroData.
    The cond Data object already has the correct homogeneous graph structure with
    instance-to-instance edges.
    
    Args:
        pickle_path: Path to pickle file
        normalize_positions: Deprecated (kept for API compatibility). Recommended pipeline does not min-max / [-1,1] normalize.
        normalize_by_graph_stats: Deprecated (kept for API compatibility).
        add_graph_stats: Deprecated (kept for API compatibility).
        device: Device for tensors
        eigenvector_dir: Optional directory containing eigenvector pickle files (named
            ``{raw_file_name}_lap_pe.pickle``). Typical layout: ``data/chipgen/processed/pe``.
        num_eigenvectors: Optional number of eigenvectors (for validation). If None, inferred from loaded data.
        min_graph_size: If set, only include graphs with num_nodes >= this value (filter out smaller graphs).
        max_graph_size: If set, only include graphs with num_nodes <= this value (filter out larger graphs).
        zero_edge_attr: If True, edge_attr is set to zeros (shape preserved for GNN compatibility).
        normalize_edge_attr_with_coordinates: If True, divide pin-offset edge_attr by the
            same per-chip coordinate scales used for positions. If False, keep edge_attr
            in raw physical units from the pickle.
        node_features: List of feature names to include (order = concat order). If None, uses get_default_node_features().
            Valid: w_norm_d, h_norm_d, w_log_rel, h_log_rel, aspect_ratio_norm, area_norm,
            area_physical_unit, chip_size, log_deg, pin_count, type_indicators, num_nodes, num_edges, eigenvectors.
        placement_from_v5_placer: If True, ignore pickle placement and run V5Placer to place macros+ports first
            (following the same pattern as v5 algorithm). Stdcells get chip-center placeholder (model will place them).
            Use for real-world netlists without ground-truth positions.
        placer_seed: Random seed for V5Placer when placement_from_v5_placer=True.
        require_aux_features: If True (default), graphs that list ``eigenvectors`` or METIS-related
            features in ``node_features`` must have corresponding data (LapPE files / METIS fields on
            ``cond``); otherwise raise. If False, missing aux data logs warnings and uses zeros.
        lap_eigenvector_stem: If set, use this string instead of the graph pickle's stem when
            resolving ``{stem}_lap_pe.pickle`` under ``eigenvector_dir`` (e.g. renamed shard files).
        load_graph_lap_eigenvectors: If True, load ``*_lap_pe.pickle`` so ``data.lap_eigenvectors``
            is set for :class:`MetisPerceiverGlobalModule` even when ``eigenvectors`` is not listed in
            ``node_features``. If None (default), auto-enable when ``eigenvector_dir`` and
            ``num_eigenvectors`` are set, ``eigenvectors`` is not in ``node_features``, and either
            ``metis_partition_pe`` or ``metis_partition_id_norm`` is requested.

    Returns:
        List of (Data, positions) tuples where:
        - Data: Homogeneous graph with instance nodes and instance-instance edges
        - positions: (V, 2) tensor of instance positions
    """
    if node_features is None:
        node_features = get_default_node_features()
    for f in node_features:
        if f not in VALID_NODE_FEATURES:
            raise ValueError(
                f"Unknown node feature '{f}'. Valid: {sorted(VALID_NODE_FEATURES)}"
            )
    use_eigenvectors_feature = "eigenvectors" in node_features
    use_metis_id = "metis_partition_id_norm" in node_features
    use_metis_pe = "metis_partition_pe" in node_features

    if load_graph_lap_eigenvectors is None:
        load_graph_lap_eigenvectors = (
            (not use_eigenvectors_feature)
            and eigenvector_dir is not None
            and (num_eigenvectors or 0) > 0
            and (use_metis_id or use_metis_pe)
        )
    need_eigenvector_sidecar = use_eigenvectors_feature or bool(load_graph_lap_eigenvectors)

    with open(pickle_path, "rb") as f:
        samples = pickle.load(f)

    if isinstance(samples, tuple) and len(samples) == 2:
        samples = [samples]
    elif not isinstance(samples, list):
        raise ValueError(f"Unexpected pickle format: {type(samples)}")

    lap_pe_stem = _lap_pe_stem_for_pickle(pickle_path, lap_eigenvector_stem)
    eigenvector_path = (
        Path(eigenvector_dir) / f"{lap_pe_stem}_lap_pe.pickle"
        if eigenvector_dir is not None
        else None
    )

    eigenvectors_list: Optional[list] = None
    if need_eigenvector_sidecar:
        if require_aux_features:
            if eigenvector_dir is None:
                if use_eigenvectors_feature:
                    raise ValueError(
                        f"{pickle_path}: node_features includes 'eigenvectors' but dataset eigenvector_dir is not set."
                    )
                raise ValueError(
                    f"{pickle_path}: Metis partition features require graph Laplacian eigenvectors "
                    f"(data.lap_eigenvectors) for MetisPerceiverGlobalModule. Set dataset.eigenvector_dir "
                    f"to the folder with *_lap_pe.pickle sidecars (e.g. data/chipgen/processed/pe)."
                )
            ev_dir = Path(eigenvector_dir)
            if not ev_dir.is_dir():
                raise FileNotFoundError(
                    f"{pickle_path}: eigenvector_dir does not exist or is not a directory: {ev_dir}"
                )
            if eigenvector_path is None or not eigenvector_path.is_file():
                raise FileNotFoundError(
                    f"{pickle_path}: required eigenvector file missing: {eigenvector_path}"
                )
            try:
                with open(eigenvector_path, "rb") as f:
                    eigenvectors_list = pickle.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load eigenvectors from {eigenvector_path}: {e}"
                ) from e
            if len(eigenvectors_list) != len(samples):
                raise ValueError(
                    f"{pickle_path}: eigenvector count {len(eigenvectors_list)} != "
                    f"sample count {len(samples)} in pickle."
                )
            if num_eigenvectors is not None and num_eigenvectors > 0 and len(eigenvectors_list) > 0:
                ev0 = eigenvectors_list[0]
                expected_dim = (
                    ev0.shape[1]
                    if isinstance(ev0, torch.Tensor)
                    else len(ev0[0])
                )
                if expected_dim != num_eigenvectors:
                    raise ValueError(
                        f"{eigenvector_path}: expected {num_eigenvectors} eigenvector columns, "
                        f"found {expected_dim}."
                    )
        else:
            if eigenvector_dir is not None:
                ev_dir = Path(eigenvector_dir)
                if ev_dir.exists() and eigenvector_path is not None and eigenvector_path.exists():
                    try:
                        with open(eigenvector_path, "rb") as f:
                            eigenvectors_list = pickle.load(f)
                        if num_eigenvectors is not None and len(eigenvectors_list) > 0:
                            ev0 = eigenvectors_list[0]
                            expected_dim = (
                                ev0.shape[1]
                                if isinstance(ev0, torch.Tensor)
                                else len(ev0[0])
                            )
                            if expected_dim != num_eigenvectors:
                                print(
                                    f"Warning: Expected {num_eigenvectors} eigenvectors, "
                                    f"but found {expected_dim} in {eigenvector_path}"
                                )
                    except Exception as e:
                        print(f"Warning: Failed to load eigenvectors from {eigenvector_path}: {e}")
                        eigenvectors_list = None
                elif eigenvector_path is not None:
                    print(f"Warning: Eigenvector file not found: {eigenvector_path}")
            else:
                if use_eigenvectors_feature:
                    print("Warning: node_features includes 'eigenvectors' but eigenvector_dir is None")
                elif load_graph_lap_eigenvectors:
                    print(
                        f"Warning: Metis lap eigenvectors requested but eigenvector_dir is None ({pickle_path})"
                    )
            if eigenvectors_list is not None and len(eigenvectors_list) != len(samples):
                print(
                    f"Warning: Number of eigenvectors ({len(eigenvectors_list)}) does not match "
                    f"number of samples ({len(samples)}) in {pickle_path}"
                )
                print("  Disabling eigenvectors / lap_eigenvectors for this file")
                eigenvectors_list = None

    result = []
    for idx, (x, cond) in enumerate(samples):
        # x: (V, 2) positions
        # cond: PyG Data object with edge_index, x (sizes), etc.
        
        x = x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
        edge_index = cond.edge_index.to(device)
        sizes = cond.x.to(device) if hasattr(cond, 'x') else None  # (V, 2) [width, height]
        
        if placement_from_v5_placer and sizes is not None:
            # Run V5Placer to place macros+ports (same pattern as v5 algorithm).
            # Stdcells get chip-center placeholder; model will place them from noise.
            chip_sz = cond.chip_size if hasattr(cond, 'chip_size') else None
            if chip_sz is not None:
                if isinstance(chip_sz, (list, tuple)):
                    chip_sz = torch.tensor(chip_sz, dtype=torch.float32, device='cpu')
                W, H = float(chip_sz[0]), float(chip_sz[1])
            else:
                W = float(sizes[:, 0].max() * 10)
                H = float(sizes[:, 1].max() * 10)
            is_port = cond.is_ports.bool() if hasattr(cond, 'is_ports') and cond.is_ports is not None else torch.zeros(sizes.shape[0], dtype=torch.bool)
            is_macro = cond.is_macro.bool() if hasattr(cond, 'is_macro') and cond.is_macro is not None else (sizes[:, 1] > 2.0)  # heuristic fallback
            is_macro = is_macro & (~is_port)
            types_arr = torch.zeros(sizes.shape[0], dtype=torch.int32)
            types_arr[is_port] = 2
            types_arr[is_macro] = 1
            from unified_learning.data_generation.chipgen.placement_v5 import V5Placer
            from unified_learning.data_generation.chipgen.v5_config import V5Config
            cfg = V5Config()
            placer = V5Placer(cfg, seed=placer_seed if placer_seed is not None else idx, device='cpu')
            pos_placed, _, _, placed_mask = placer.place(
                sizes.cpu().float(), types_arr.cpu(), canvas_W=W, canvas_H=H
            )
            chip_center = torch.tensor([W / 2.0, H / 2.0], dtype=torch.float32, device=device)
            x = chip_center.unsqueeze(0).expand(sizes.shape[0], 2).clone()
            placed_indices = torch.where(placed_mask.cpu())[0]
            for j, orig_idx in enumerate(placed_indices.tolist()):
                x[orig_idx] = pos_placed[j].to(device)
            # Macros and ports are fixed during sampling. Stdcells use placer output as pos_target
            # placeholder (model generates their positions from noise; pos_target unused for them).
        
        # Extract graph structure from cond
        
        # Build instance features
        if sizes is None:
            raise ValueError("cond must have 'x' attribute with instance sizes")
        
        V = x.shape[0]
        widths = sizes[:, 0]
        heights = sizes[:, 1]
        
        # New recommended normalization (no min-max / [-1,1] normalization).
        # `normalize_positions`, `normalize_by_graph_stats`, `add_graph_stats` are kept for API compatibility
        # but are not used for the recommended normalization.
        chip_size = cond.chip_size if hasattr(cond, "chip_size") else None
        use_inferred_center = False
        c_x, c_y = None, None  # used only when chip_size inferred from instance box
        if chip_size is not None and isinstance(chip_size, (list, tuple)):
            chip_size = torch.tensor(chip_size, dtype=torch.float32, device=device)
        elif isinstance(chip_size, torch.Tensor):
            chip_size = chip_size.to(device)
        else:
            # Compute chip boundary from instance bounding box (all cells including macros)
            # Each instance has center (x_i, y_i) and size (w_i, h_i); extent is [x_i - w_i/2, x_i + w_i/2] x [y_i - h_i/2, y_i + h_i/2]
            left_edge = (x[:, 0] - widths / 2.0).min().item()
            right_edge = (x[:, 0] + widths / 2.0).max().item()
            bottom_edge = (x[:, 1] - heights / 2.0).min().item()
            top_edge = (x[:, 1] + heights / 2.0).max().item()
            W_val = right_edge - left_edge
            H_val = top_edge - bottom_edge
            c_x = left_edge + W_val / 2.0
            c_y = bottom_edge + H_val / 2.0
            chip_size = torch.tensor([W_val, H_val], dtype=torch.float32, device=device)
            use_inferred_center = True
            
            # VALIDATION: Check if inferred chip_size is valid
            if W_val <= 0 or H_val <= 0:
                raise ValueError(
                    f"Invalid chip_size computed from bounding box in data loading: W={W_val:.6f}, H={H_val:.6f}. "
                    f"This indicates degenerate data (all instances at same position or invalid layout). "
                    f"Bounding box: left={left_edge:.6f}, right={right_edge:.6f}, "
                    f"bottom={bottom_edge:.6f}, top={top_edge:.6f}, "
                    f"num_instances={len(x)}. "
                    f"Check: Are all instances overlapping? Is the layout valid?"
                )
        
        # VALIDATION: Check chip_size from cond is valid (after conversion to tensor)
        if chip_size is not None:
            if chip_size[0] <= 0 or chip_size[1] <= 0:
                raise ValueError(
                    f"Invalid chip_size from cond in data loading: W={chip_size[0]:.6f}, H={chip_size[1]:.6f}. "
                    f"Chip size must be positive. This indicates invalid data generation."
                )
            if not torch.isfinite(chip_size).all():
                raise ValueError(
                    f"Non-finite chip_size from cond: {chip_size.tolist()}. "
                    f"This indicates NaN/Inf in data generation."
                )

        # Optional refinement diffusion: create a fixed per-sample coordinate gauge from X_init.
        # - pos_target: ground-truth clean positions (X0)
        # - pos_init: constructed initial positions (X_init) used to define gauge (mu, s)
        pos_target = x
        if init_noise_std and init_noise_std > 0:
            pos_init = pos_target + torch.randn_like(pos_target) * float(init_noise_std)
        else:
            pos_init = pos_target

        # ============================================================================
        # COORDINATE NORMALIZATION: Min/Max to [-1, 1]
        # ============================================================================
        # Center coordinates, then scale by half-range so positions in [0,W]x[0,H] map to [-1,1]x[-1,1]
        # Step 1: Center: x_centered = x - c (c = chip center)
        # Step 2: Min/max scale: z = x_centered / (W/2, H/2) -> z in [-1, 1]
        W, H = chip_size[0], chip_size[1]

        # Center: when chip_size from cond, assume positions in [0,W]x[0,H] so c = (W/2, H/2)
        # When inferred from instance box, use center of box (c_x, c_y)
        if use_inferred_center:
            c = torch.tensor([c_x, c_y], device=device, dtype=chip_size.dtype)  # (2,)
        else:
            c = torch.tensor([W / 2.0, H / 2.0], device=device, dtype=chip_size.dtype)  # (2,)

        # Step 1: Center coordinates
        x_centered = pos_target - c.unsqueeze(0)  # (V, 2)

        # Step 2: Min/max scale to [-1, 1] (anisotropic: s_x = W/2, s_y = H/2)
        scale_x = W / 2.0
        scale_y = H / 2.0
        
        # VALIDATION: Check scale factors are valid (prevent division by zero)
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError(
                f"Invalid scale factors in data loading: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}. "
                f"This indicates chip_size is zero or invalid: W={W:.6f}, H={H:.6f}. "
                f"This should have been caught earlier."
            )
        
        s_per_dim = torch.tensor([scale_x, scale_y], device=device, dtype=chip_size.dtype)  # (2,)
        z = x_centered / s_per_dim.unsqueeze(0)  # (V, 2) - coordinates in [-1, 1]

        # VALIDATION: Check normalized coordinates are in expected range [-1, 1]
        # Allow small tolerance (0.1) for numerical errors
        z_min, z_max = z.min().item(), z.max().item()
        if z_min < -1.1 or z_max > 1.1:
            raise ValueError(
                f"Normalized coordinates outside expected range [-1, 1] in data loading: "
                f"min={z_min:.6f}, max={z_max:.6f}. "
                f"This indicates invalid data or normalization error. "
                f"chip_size={chip_size.tolist()}, "
                f"s_per_dim={s_per_dim.tolist()}, "
                f"x_centered range: [{x_centered.min().item():.6f}, {x_centered.max().item():.6f}], "
                f"num_instances={V}. "
                f"Check: Are positions within chip bounds? Is chip_size correct?"
            )
        
        # Store normalization parameters (mu = center, s = half-range per dimension)
        mu = c.unsqueeze(0).expand(V, 2)  # (V, 2) - center [W/2, H/2] per node
        s = s_per_dim.unsqueeze(0).expand(V, 2)  # (V, 2) - [W/2, H/2] per node
        
        # ============================================================================
        # SIZE FEATURES: Coordinate-scale normalized only (same scale as coords)
        # ============================================================================
        # Clamp sizes to positive only to avoid division by zero; normalize by half-range.
        # w_norm_d = width / (W/2) can be >1 for large macros; keep unbounded to preserve real info.
        sizes_eps = 1e-6
        widths = sizes[:, 0].clamp(min=sizes_eps)
        heights = sizes[:, 1].clamp(min=sizes_eps)
        w_norm_d = (widths / scale_x).unsqueeze(-1)  # (V, 1) - no max clamp to preserve large macros
        h_norm_d = (heights / scale_y).unsqueeze(-1)  # (V, 1)
        # Replace any non-finite with safe value (e.g. from invalid input)
        w_norm_d = torch.where(torch.isfinite(w_norm_d), w_norm_d, torch.full_like(w_norm_d, 1.0))
        h_norm_d = torch.where(torch.isfinite(h_norm_d), h_norm_d, torch.full_like(h_norm_d, 1.0))
        
        # ============================================================================
        # GEOMETRIC FEATURES: Aspect ratio and area (normalized to [0, 1] scale)
        # ============================================================================
        eps = 1e-6
        # Aspect ratio: log(w/h), normalized (no clamp to [0,1] to preserve extreme ratios)
        # Typical range: aspect ratios 0.1 to 10 → log(0.1) = -2.3, log(10) = 2.3
        # Normalize by dividing by 5 and shifting: (log(w/h) + 2.5) / 5
        aspect_ratio_raw = torch.log(torch.clamp(widths / torch.clamp(heights, min=eps), min=eps))
        aspect_ratio_norm = ((aspect_ratio_raw + 2.5) / 5.0).unsqueeze(-1)  # (V, 1)
        
        # Area: log(wh), normalized to [0, 1] (per-graph median/MAD)
        area_raw = torch.log(torch.clamp(widths * heights, min=eps))
        area_median = torch.median(area_raw)
        area_mad = torch.median(torch.abs(area_raw - area_median))
        area_mad = (area_mad if area_mad > 1e-3 else area_raw.std()).clamp(min=eps)
        area_norm = ((area_raw - area_median) / area_mad / 5.0)
        area_norm = ((area_norm + 1.0) / 2.0).unsqueeze(-1)
        # N-invariant area (optional extra feature): log(area/L^2) then tanh/2 -> [0,1]
        L_die = max(float(W), float(H))
        area_ref = max(L_die * L_die, eps)
        area_log_rel = area_raw - math.log(area_ref)
        area_physical_unit = ((torch.tanh(area_log_rel / 5.0) + 1.0) / 2.0).unsqueeze(-1)
        
        # ============================================================================
        # GLOBAL/CHIP FEATURES - NORMALIZED to [-1, 1] scale
        # ============================================================================
        ROW_HEIGHT = 0.9898  # μm - standard-cell row height constant
        # Normalize chip size: log(chip_size / row_h) / 10.0
        # Typical range: 300-10000 → log(303-10101) = 5.7-9.2 → normalized to 0.57-0.92
        chip_w_rel_raw = torch.log(torch.clamp(chip_size[0] / ROW_HEIGHT, min=1e-6))
        chip_h_rel_raw = torch.log(torch.clamp(chip_size[1] / ROW_HEIGHT, min=1e-6))
        chip_w_rel = (chip_w_rel_raw / 10.0).unsqueeze(0).expand(V, 1)  # (V, 1) normalized
        chip_h_rel = (chip_h_rel_raw / 10.0).unsqueeze(0).expand(V, 1)  # (V, 1) normalized

        # ============================================================================
        # SIZE FEATURES (TRANSFER-SAFE): log-size relative to row height
        # ============================================================================
        # These are stable across chip sizes (unlike w_norm_d/h_norm_d) and exist in real netlists.
        # Use a smooth bounded map to [0,1] to keep feature scales consistent.
        # Typical:
        # - stdcells: width ~ [1,5]µm, height ~ [0.2,0.6]µm
        # - macros: width/height ~ [8,30]µm (or larger)
        w_log_rel = torch.log(torch.clamp(widths / ROW_HEIGHT, min=eps))  # (V,)
        h_log_rel = torch.log(torch.clamp(heights / ROW_HEIGHT, min=eps))  # (V,)
        w_log_rel_feat = ((torch.tanh(w_log_rel / 5.0) + 1.0) / 2.0).unsqueeze(-1)  # (V,1) in [0,1]
        h_log_rel_feat = ((torch.tanh(h_log_rel / 5.0) + 1.0) / 2.0).unsqueeze(-1)  # (V,1) in [0,1]
        
        # Type indicators (for pos_mask and features) — REQUIRED from generator (V5 / IBM convert)
        # Macro and super-macro classification is critical; no heuristic fallback.
        if not (hasattr(cond, 'is_ports') and cond.is_ports is not None):
            raise ValueError(
                "Graph must have 'is_ports' mask. Use V5 or convert_ibm_to_v5 format."
            )
        if not (hasattr(cond, 'is_macro') and cond.is_macro is not None):
            raise ValueError(
                "Graph must have 'is_macro' mask. Use V5 or convert_ibm_to_v5 format."
            )
        if not (hasattr(cond, 'is_stdcell') and cond.is_stdcell is not None):
            raise ValueError(
                "Graph must have 'is_stdcell' mask. Use V5 or convert_ibm_to_v5 format."
            )
        if not (hasattr(cond, 'is_super_macro') and cond.is_super_macro is not None):
            raise ValueError(
                "Graph must have 'is_super_macro' mask. Use V5 or convert_ibm_to_v5 format."
            )

        is_port_bool = cond.is_ports.bool().to(device)
        is_macro_bool = cond.is_macro.bool().to(device) & (~is_port_bool)
        is_stdcell_bool = cond.is_stdcell.bool().to(device)
        is_super_macro_bool = cond.is_super_macro.bool().to(device)

        # VALIDATION: Ports must always have valid locations (from pickle or placer).
        if is_port_bool.any():
            port_pos = pos_target[is_port_bool]
            if (~torch.isfinite(port_pos)).any():
                raise ValueError(
                    f"Ports must have valid locations in any case. "
                    f"Found NaN/Inf in port positions (n_ports={is_port_bool.sum().item()}). "
                    f"Ensure the pickle or placer provides finite positions for all port nodes."
                )

        # VALIDATION: When diffuse_macros is False, macros (and super_macros) must have valid locations.
        if not diffuse_macros and is_macro_bool.any():
            macro_pos = pos_target[is_macro_bool]
            if (~torch.isfinite(macro_pos)).any():
                raise ValueError(
                    f"With diffuse_macros=False, macros (and super_macros) must have valid locations. "
                    f"Found NaN/Inf in macro positions (n_macros={is_macro_bool.sum().item()}). "
                    f"Either set diffuse_macros=True or provide finite positions for macros from pickle/placer."
                )
        
        # ============================================================================
        # CONNECTIVITY FEATURES - NORMALIZED to [0, 1] scale
        # ============================================================================
        deg = degree(edge_index[0], num_nodes=V, dtype=torch.float32)  # Out-degree
        # Use ONLY log_degree (bounded, reliable normalization)
        # Normalize log degree by 5 (log(100) ≈ 4.6, log(1000) ≈ 6.9)
        # Range: log(1) to log(1000) → 0 to ~1.4 after /5 normalization
        log_deg = (torch.log(1.0 + deg) / 5.0).unsqueeze(-1)  # (V, 1) normalized

        # ============================================================================
        # PIN COUNT (TRANSFER-SAFE): prefer authoritative pin count if available
        # ============================================================================
        # V5 now saves `num_terminals` in cond. Real netlists also have pin counts per instance.
        if hasattr(cond, "num_terminals") and cond.num_terminals is not None:
            pin_count = cond.num_terminals.to(device=device).to(dtype=torch.float32).clamp(min=0.0)  # (V,)
        else:
            # Backward-compatible proxy: use degree as a weak surrogate (not ideal, but avoids missing feature).
            pin_count = deg.to(dtype=torch.float32)
        pin_log = torch.log1p(pin_count).clamp(min=0.0)
        pin_count_feat = ((torch.tanh(pin_log / 5.0) + 1.0) / 2.0).unsqueeze(-1)  # (V,1) in [0,1]

        # REMOVED: deg_norm (raw degree, unbounded, can be >> 100)
        # REMOVED: neighbor_deg_mean (unbounded, unreliable normalization)
        
        # ============================================================================
        # TYPE INDICATORS
        # ============================================================================
        is_macro = is_macro_bool.float().unsqueeze(-1)  # (V, 1)
        is_port = is_port_bool.float().unsqueeze(-1)  # (V, 1)
        is_stdcell = is_stdcell_bool.float().unsqueeze(-1)  # (V, 1)

        # ============================================================================
        # CANVAS ASPECT RATIO: tanh(log(W/H) / 2) in (-1, 1), broadcast to all nodes
        # Encodes the shape of the placement canvas so the model knows its geometry.
        # ============================================================================
        canvas_ar_log = math.log(max(float(W) / max(float(H), 1e-6), 1e-6))
        canvas_ar_val = math.tanh(canvas_ar_log / 2.0)
        canvas_ar_feat = torch.full((V, 1), canvas_ar_val, device=device, dtype=torch.float32)

        # ============================================================================
        # PORT LOCATION: [-1,1]-normalised (x,y) for port nodes, (0,0) otherwise.
        # At test-time ports are fixed; giving the model their positions helps it
        # learn to place std-cells relative to known I/O pads.
        # z is already computed above (coordinates in [-1,1]).
        # ============================================================================
        port_loc_feat = z.clone().to(dtype=torch.float32)   # (V, 2) - copy of normalised positions
        port_loc_feat[~is_port_bool] = 0.0                  # zero-out non-port nodes

        # ============================================================================
        # GRAPH STATISTICS (optional): number of nodes, number of edges (log-normalized)
        # ============================================================================
        E = edge_index.shape[1] if edge_index is not None else 0
        num_nodes_val = torch.log(1.0 + torch.tensor(float(V), device=device, dtype=torch.float32)) / GRAPH_STATS_LOG_SCALE
        num_edges_val = torch.log(1.0 + torch.tensor(float(E), device=device, dtype=torch.float32)) / GRAPH_STATS_LOG_SCALE
        num_nodes_feat = num_nodes_val.unsqueeze(0).expand(V, 1).clamp(0.0, 1.0)   # (V, 1)
        num_edges_feat = num_edges_val.unsqueeze(0).expand(V, 1).clamp(0.0, 1.0)   # (V, 1)
        
        # ============================================================================
        # FINAL NODE FEATURE SET (configurable via node_features list)
        # ============================================================================
        # ============================================================================
        # METIS PARTITION FEATURES (optional): partition ID and Laplacian PE
        # Require the pickle to have been pre-annotated by
        # scripts/dataset_generation/add_metis_partition_to_dataset.py
        # ============================================================================
        has_metis = (
            hasattr(cond, "metis_partition_id") and cond.metis_partition_id is not None
            and hasattr(cond, "num_partitions") and cond.num_partitions is not None
        )
        if require_aux_features:
            if (use_metis_id or use_metis_pe) and not has_metis:
                raise ValueError(
                    f"{pickle_path}: sample index {idx} is missing METIS annotations on cond "
                    f"(metis_partition_id / num_partitions). "
                    f"Run scripts/dataset_generation/add_metis_partition_to_dataset.py with the "
                    f"same partition_size as training."
                )
            if use_metis_pe and (
                not hasattr(cond, "node_partition_pe") or cond.node_partition_pe is None
            ):
                raise ValueError(
                    f"{pickle_path}: sample index {idx} is missing cond.node_partition_pe."
                )

        if (use_metis_id or use_metis_pe) and has_metis:
            _pid = cond.metis_partition_id.to(device=device)      # LongTensor[V]
            _K = int(cond.num_partitions[0].item())
            metis_id_norm_feat = (_pid.float() / max(_K, 1)).unsqueeze(-1)  # [V, 1] in [0,1)
        else:
            metis_id_norm_feat = torch.zeros(V, 1, device=device, dtype=torch.float32)

        if use_metis_pe and has_metis and hasattr(cond, "node_partition_pe"):
            _npe = cond.node_partition_pe.to(device=device, dtype=torch.float32)  # [V, max_k]
            # Validate / pad column count
            if _npe.shape[1] < metis_max_k:
                _pad = torch.zeros(V, metis_max_k - _npe.shape[1], device=device, dtype=torch.float32)
                _npe = torch.cat([_npe, _pad], dim=1)
            elif _npe.shape[1] > metis_max_k:
                _npe = _npe[:, :metis_max_k]
            metis_pe_feat = _npe  # [V, metis_max_k]
        else:
            metis_pe_feat = torch.zeros(V, metis_max_k, device=device, dtype=torch.float32)

        feature_tensors = {
            "w_norm_d": w_norm_d,
            "h_norm_d": h_norm_d,
            "w_log_rel": w_log_rel_feat,
            "h_log_rel": h_log_rel_feat,
            "aspect_ratio_norm": aspect_ratio_norm,
            "area_norm": area_norm,
            "area_physical_unit": area_physical_unit,
            "chip_size": torch.cat([chip_w_rel, chip_h_rel], dim=1),
            "log_deg": log_deg,
            "pin_count": pin_count_feat,
            "type_indicators": torch.cat([is_stdcell, is_macro, is_port], dim=1),
            "num_nodes": num_nodes_feat,
            "num_edges": num_edges_feat,
            "canvas_aspect_ratio": canvas_ar_feat,
            "port_loc": port_loc_feat,
            "metis_partition_id_norm": metis_id_norm_feat,
            "metis_partition_pe": metis_pe_feat,
        }
        eigenvectors = None
        need_lap_tensor = (
            (use_eigenvectors_feature or load_graph_lap_eigenvectors)
            and num_eigenvectors is not None
            and num_eigenvectors > 0
        )
        if need_lap_tensor:
            if eigenvector_dir is not None and eigenvectors_list is not None and idx < len(eigenvectors_list):
                eigenvectors = eigenvectors_list[idx]
                if isinstance(eigenvectors, torch.Tensor):
                    eigenvectors = eigenvectors.to(device)
                else:
                    eigenvectors = torch.tensor(eigenvectors, device=device, dtype=torch.float32)
                if eigenvectors.shape[0] != V:
                    if eigenvectors.shape[0] < V:
                        padding = torch.zeros((V - eigenvectors.shape[0], eigenvectors.shape[1]), device=device, dtype=eigenvectors.dtype)
                        eigenvectors = torch.cat([eigenvectors, padding], dim=0)
                    else:
                        eigenvectors = eigenvectors[:V]
                if eigenvectors.shape[1] != num_eigenvectors:
                    if eigenvectors.shape[1] < num_eigenvectors:
                        padding = torch.zeros((V, num_eigenvectors - eigenvectors.shape[1]), device=device, dtype=eigenvectors.dtype)
                        eigenvectors = torch.cat([eigenvectors, padding], dim=1)
                    else:
                        eigenvectors = eigenvectors[:, :num_eigenvectors]
            else:
                if require_aux_features:
                    raise ValueError(
                        f"{pickle_path}: sample index {idx} has no eigenvector row "
                        f"(eigenvectors_list missing or idx out of range)."
                    )
                eigenvectors = torch.zeros((V, num_eigenvectors), device=device, dtype=torch.float32)

        if use_eigenvectors_feature:
            if num_eigenvectors is None or num_eigenvectors <= 0:
                raise ValueError(
                    "node_features includes 'eigenvectors' but dataset.num_eigenvectors is missing or <= 0."
                )
            feature_tensors["eigenvectors"] = eigenvectors

        # Graph Laplacian PE for MetisPerceiverGlobalModule (scatter_mean on tokens).
        # Populated from the same *_lap_pe.pickle sidecar as node eigenvectors when requested.
        _lap_eigenvectors_for_data = (
            eigenvectors if (use_eigenvectors_feature or load_graph_lap_eigenvectors) else None
        )

        node_features_tensor = torch.cat(
            [feature_tensors[name] for name in node_features], dim=1
        )  # (V, feature_dim)

        # Column offset of metis_partition_pe in node_features_tensor.
        # Used by sign-flip augmentation in collate_homogeneous_graphs so it knows
        # which slice of data.x to flip consistently with data.node_partition_pe.
        _metis_pe_col_start: Optional[int] = None
        if use_metis_pe and has_metis:
            _col = 0
            for _fn in node_features:
                if _fn == "metis_partition_pe":
                    break
                _col += (num_eigenvectors or 0) if _fn == "eigenvectors" else NODE_FEATURE_DIMS[_fn]
            _metis_pe_col_start = _col

        # ============================================================================
        # EDGE FEATURES: optionally normalized by the same half-range as coordinates
        # Pickle cond.edge_attr is [src_pin_x, src_pin_y, dst_pin_x, dst_pin_y] in microns
        # (pin offset from instance center; from V5 get_terminal_offsets).
        # When normalize_edge_attr_with_coordinates=True:
        #   anisotropic normalize as x / scale_x and y / scale_y.
        # When False:
        #   keep raw pin offsets in physical units to preserve absolute size signal.
        # When zero_edge_attr=True, use zeros (shape preserved for GNN edge_dim).
        # ============================================================================
        edge_attr_raw = cond.edge_attr.to(device) if hasattr(cond, "edge_attr") and cond.edge_attr is not None else None
        if zero_edge_attr:
            E = edge_index.shape[1]
            edge_attr = torch.zeros((E, 4), device=device, dtype=torch.float32)
        elif edge_attr_raw is not None:
            edge_attr = edge_attr_raw.clone()
            if normalize_edge_attr_with_coordinates:
                edge_attr[:, [0, 2]] = edge_attr[:, [0, 2]] / scale_x  # src_pin_x, dst_pin_x
                edge_attr[:, [1, 3]] = edge_attr[:, [1, 3]] / scale_y  # src_pin_y, dst_pin_y
        else:
            edge_attr = None
        
        # Create Data object
        data = Data(
            x=node_features_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # Keep authoritative physical instance sizes so downstream code (train/test/analysis)
        # does not need to infer width/height from feature column layouts.
        data.instance_sizes = torch.stack([widths, heights], dim=1)

        # Store raw positions and normalization parameters:
        # - pos_target: ground-truth (X0) used for diffusion loss
        # - pos_init: constructed initial placement (X_init) used for refinement diffusion
        # - mu: (V, 2) global center per node (same for all)
        # - s: (V, 2) half-range [W/2, H/2] per node (same for all)
        # - chip_size: (2,) true [W, H] for error-percentage and visualization
        data.pos_target = pos_target
        data.pos_init = pos_init
        data.mu = mu  # (V, 2)
        data.pos_original = pos_target
        data.s = s  # (V, 2) - diagonal scale [D, D]
        data.chip_size = chip_size.clone()  # (2,) true chip size [W, H]
        
        # Preserve node type masks (exact V5 / IBM convert values)
        data.is_ports = is_port_bool
        data.is_macro = is_macro_bool
        data.is_stdcell = is_stdcell_bool
        data.is_super_macro = is_super_macro_bool

        # Preserve scale_to_original (IBM convert: HPWL_v5 * scale_to_original = HPWL_original µm)
        if hasattr(cond, 'scale_to_original') and cond.scale_to_original is not None:
            scale_val = cond.scale_to_original
            data.scale_to_original = scale_val if isinstance(scale_val, (int, float)) else float(scale_val)

        # Add pos_mask: which nodes to diffuse (True) vs keep fixed (False)
        # Prefer authoritative pos_mask from generator if present.
        if hasattr(cond, 'pos_mask') and cond.pos_mask is not None and not diffuse_macros and not diffuse_macros_only:
            data.pos_mask = cond.pos_mask.bool().to(device)
        else:
            if diffuse_macros_only:
                # Only macros diffused; std-cells and ports fixed (pure macro placement)
                data.pos_mask = is_macro_bool.clone()
                if not is_macro_bool.any():
                    continue  # Skip graphs with no macros
            elif diffuse_macros:
                # Only ports fixed; std-cells and macros are diffused
                data.pos_mask = torch.logical_not(is_port_bool)
            else:
                # Std-cells diffused; macros and ports fixed
                data.pos_mask = torch.logical_not(torch.logical_or(is_macro_bool, is_port_bool))
        
        # Graph Laplacian eigenvectors as a standalone field for MetisPerceiverGlobalModule.
        if _lap_eigenvectors_for_data is not None:
            data.lap_eigenvectors = _lap_eigenvectors_for_data  # [V, num_eigenvectors]
        else:
            data.lap_eigenvectors = None

        # Column offset of 'eigenvectors' in data.x for sign-flip augmentation.
        _eigenvector_col_start: Optional[int] = None
        if use_eigenvectors_feature and eigenvectors is not None:
            _col = 0
            for _fn in node_features:
                if _fn == "eigenvectors":
                    break
                _col += (num_eigenvectors or 0) if _fn == "eigenvectors" else NODE_FEATURE_DIMS[_fn]
            _eigenvector_col_start = _col
        data.eigenvector_col_start = _eigenvector_col_start
        data.num_eigenvector_dims = num_eigenvectors if use_eigenvectors_feature else None

        # Pass-through Metis partition fields so the global module can access them
        # at training time (MetisPerceiverGlobalModule reads these from batched data).
        # node_partition_pe must be width metis_max_k (same as pe_proj / encoder), not the
        # raw variable-width tensor from add_metis --max_k auto; x already uses metis_pe_feat.
        if has_metis:
            data.metis_partition_id = cond.metis_partition_id.to(device)
            if use_metis_pe:
                data.node_partition_pe = metis_pe_feat
            elif hasattr(cond, "node_partition_pe") and cond.node_partition_pe is not None:
                _r = cond.node_partition_pe.to(device=device, dtype=torch.float32)
                if _r.shape[1] < metis_max_k:
                    _pad = torch.zeros(
                        V, metis_max_k - _r.shape[1], device=device, dtype=torch.float32
                    )
                    data.node_partition_pe = torch.cat([_r, _pad], dim=1)
                elif _r.shape[1] > metis_max_k:
                    data.node_partition_pe = _r[:, :metis_max_k]
                else:
                    data.node_partition_pe = _r
            else:
                data.node_partition_pe = torch.zeros(
                    V, metis_max_k, device=device, dtype=torch.float32
                )
            data.num_partitions = cond.num_partitions.to(device)  # LongTensor[V]
            # Column offset of metis_partition_pe in data.x — used by sign-flip augmentation.
            # None when metis_partition_pe is not in node_features (no slice to flip in x).
            data.metis_pe_col_start = _metis_pe_col_start
        else:
            # Sentinel: no partition data — global module falls back gracefully
            data.metis_partition_id = None
            data.node_partition_pe = None
            data.num_partitions = None
            data.metis_pe_col_start = None

        # Strict graph size filter: keep only graphs within [min_graph_size, max_graph_size]
        if min_graph_size is not None and V < min_graph_size:
            continue
        if max_graph_size is not None and V > max_graph_size:
            continue
        
        # Return positions (raw, unnormalized)
        result.append((data, pos_target))
    
    return result


def load_pickle_dir_as_data(
    data_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    debug: bool = False,
    normalize_positions: bool = True,
    normalize_by_graph_stats: bool = False,
    add_graph_stats: bool = False,
    init_noise_std: float = 0.0,
    device: str = "cpu",
    normalization_stats: Optional[dict] = None,
    diffuse_macros: bool = False,
    diffuse_macros_only: bool = False,
    eigenvector_dir: Optional[Union[str, Path]] = None,
    num_eigenvectors: Optional[int] = None,
    min_graph_size: Optional[int] = None,
    max_graph_size: Optional[int] = None,
    node_features: Optional[List[str]] = None,
    zero_edge_attr: bool = False,
    normalize_edge_attr_with_coordinates: bool = True,
    metis_max_k: int = 64,
    require_aux_features: bool = True,
    lap_eigenvector_stem: Optional[str] = None,
    load_graph_lap_eigenvectors: Optional[bool] = None,
) -> List[Tuple[Data, torch.Tensor]]:
    """
    Load all pickle files from a directory directly as Data objects.

    This loads chipdiffusion-style data without converting to HeteroData.

    Args:
        data_dir: Directory containing .pickle files
        max_samples: Maximum number of samples to load
        debug: If True, only load first file
        normalize_positions: Deprecated (kept for API compatibility).
        normalize_by_graph_stats: Deprecated (kept for API compatibility).
        add_graph_stats: Deprecated (kept for API compatibility).
        init_noise_std: Standard deviation of noise for refinement diffusion
        device: Device for tensors
        normalization_stats: Dataset-level normalization statistics (use precomputed stats for stable training)
        node_features: List of feature names to include. If None, uses get_default_node_features().

    Returns:
        List of (Data, positions) tuples
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find all .pickle files
    pickle_files = sorted(data_dir.glob("*.pickle"))
    
    if len(pickle_files) == 0:
        raise ValueError(f"No .pickle files found in {data_dir}")
    
    if debug:
        print(f"DEBUG MODE: Loading only the first .pickle file")
        pickle_files = pickle_files[:1]
    else:
        print(f"Found {len(pickle_files)} .pickle files in {data_dir}")
    
    all_data = []
    for pickle_file in tqdm(pickle_files, desc="Loading pickle files"):
        try:
            samples = load_pickle_file_as_data(
                pickle_file,
                normalize_positions=normalize_positions,
                normalize_by_graph_stats=normalize_by_graph_stats,
                add_graph_stats=add_graph_stats,
                init_noise_std=init_noise_std,
                device=device,
                normalization_stats=normalization_stats,
                diffuse_macros=diffuse_macros,
                diffuse_macros_only=diffuse_macros_only,
                eigenvector_dir=eigenvector_dir,
                num_eigenvectors=num_eigenvectors,
                min_graph_size=min_graph_size,
                max_graph_size=max_graph_size,
                node_features=node_features,
                zero_edge_attr=zero_edge_attr,
                normalize_edge_attr_with_coordinates=normalize_edge_attr_with_coordinates,
                metis_max_k=metis_max_k,
                require_aux_features=require_aux_features,
                lap_eigenvector_stem=lap_eigenvector_stem,
                load_graph_lap_eigenvectors=load_graph_lap_eigenvectors,
            )
            all_data.extend(samples)
            
            # Check if we've reached max_samples
            if max_samples is not None and len(all_data) >= max_samples:
                all_data = all_data[:max_samples]
                break
            
            if debug:
                break
                
        except Exception as e:
            if require_aux_features:
                raise
            print(f"Warning: Failed to load {pickle_file}: {e}, skipping")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Loaded {len(all_data)} samples from pickle files")
    return all_data


class HomogeneousGraphDataset(Dataset):
    """
    Dataset for chipdiffusion-style homogeneous graphs.
    
    Stores (Data, positions) tuples where Data is a homogeneous graph
    with instance-to-instance edges (not bipartite).
    """
    
    def __init__(self, data_list: List[Tuple[Data, torch.Tensor]]):
        """
        Args:
            data_list: List of (Data, positions) tuples
        """
        self.data_list = data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """
        Returns:
            Tuple of (Data, positions) where:
            - Data: Homogeneous graph with node features and edges
            - positions: (V, 2) tensor of instance positions
        """
        return self.data_list[idx]

    def graph_num_nodes(self, idx: int) -> int:
        """Return node count for sample *idx* without materialising tensors."""
        data, _pos = self.data_list[idx]
        return data.x.shape[0]


class SizeBucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups graphs of similar node-count together and
    keeps total nodes per batch roughly constant.

    Instead of a fixed graph count, each batch is filled until adding
    another graph would exceed ``max_total_nodes``.  This means batches
    of small graphs contain many graphs while batches of large graphs
    contain fewer, keeping GPU memory and compute roughly constant.

    Algorithm (each epoch):
      1. Sort dataset indices by node count.
      2. Partition into contiguous *mega-buckets* (for intra-bucket
         randomness without mixing distant sizes).
      3. Shuffle indices **within** each mega-bucket.
      4. Greedily pack batches respecting ``max_total_nodes`` and
         ``max_batch_size``.
      5. Optionally shuffle the order of the resulting batches.
    """

    def __init__(
        self,
        dataset: HomogeneousGraphDataset,
        batch_size: int,
        max_total_nodes: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_multiplier: int = 10,
    ):
        self.dataset = dataset
        self.max_batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_multiplier = max(bucket_multiplier, 1)

        self._sizes = [dataset.graph_num_nodes(i) for i in range(len(dataset))]

        if max_total_nodes is not None:
            self.max_total_nodes = max_total_nodes
        else:
            # Heuristic: reference_graph_size * batch_size.
            # Use the median graph size so the budget is reasonable for
            # the majority of the dataset.
            sorted_sizes = sorted(self._sizes)
            median_size = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 1
            self.max_total_nodes = median_size * batch_size

    def _pack_batches(self, indices: List[int]) -> List[List[int]]:
        """Greedily pack indices into batches respecting the node budget."""
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_nodes = 0
        for idx in indices:
            n = self._sizes[idx]
            # Always allow at least one graph per batch
            if current_batch and (current_nodes + n > self.max_total_nodes
                                  or len(current_batch) >= self.max_batch_size):
                batches.append(current_batch)
                current_batch = []
                current_nodes = 0
            current_batch.append(idx)
            current_nodes += n
        if current_batch:
            if self.drop_last and len(batches) > 0:
                pass  # drop the incomplete final batch
            else:
                batches.append(current_batch)
        return batches

    def __iter__(self):
        indices = list(range(len(self._sizes)))
        indices.sort(key=lambda i: self._sizes[i])

        mega = self.bucket_multiplier * self.max_batch_size
        batches: List[List[int]] = []
        for start in range(0, len(indices), mega):
            bucket = indices[start : start + mega]
            if self.shuffle:
                random.shuffle(bucket)
            batches.extend(self._pack_batches(bucket))

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        # Exact count requires running the packing; cache after first iter.
        # For tqdm progress bars an estimate is fine.
        if not hasattr(self, "_cached_len"):
            indices = list(range(len(self._sizes)))
            indices.sort(key=lambda i: self._sizes[i])
            mega = self.bucket_multiplier * self.max_batch_size
            count = 0
            for start in range(0, len(indices), mega):
                bucket = indices[start : start + mega]
                count += len(self._pack_batches(bucket))
            self._cached_len = count
        return self._cached_len


class LoopScheduleBatchSampler(SizeBucketBatchSampler):
    """Batch sampler for looped training that groups by nominal loop tier.

    This keeps graphs that map to the same schedule-derived loop count ``K``
    in the same sampling pool, then applies the usual node-budget packing
    within each tier. The result is that a batch-level ``K`` chosen from
    ``max(graph_num_nodes)`` is much less likely to over-train smaller graphs.
    """

    def __init__(
        self,
        dataset: HomogeneousGraphDataset,
        batch_size: int,
        loop_count_fn: Callable[[int], int],
        max_total_nodes: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_multiplier: int = 4,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            max_total_nodes=max_total_nodes,
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_multiplier=bucket_multiplier,
        )
        self.loop_count_fn = loop_count_fn

        tier_to_indices: Dict[int, List[int]] = {}
        for idx, size in enumerate(self._sizes):
            tier = int(self.loop_count_fn(int(size)))
            tier_to_indices.setdefault(tier, []).append(idx)
        self._tier_to_indices = tier_to_indices

    def _pack_tier(self, tier_indices: List[int]) -> List[List[int]]:
        ordered = list(tier_indices)
        ordered.sort(key=lambda i: self._sizes[i])

        mega = self.bucket_multiplier * self.max_batch_size
        tier_batches: List[List[int]] = []
        for start in range(0, len(ordered), mega):
            bucket = ordered[start : start + mega]
            if self.shuffle:
                random.shuffle(bucket)
            tier_batches.extend(self._pack_batches(bucket))
        return tier_batches

    def __iter__(self):
        tier_ids = sorted(self._tier_to_indices.keys())
        if self.shuffle:
            random.shuffle(tier_ids)

        batches: List[List[int]] = []
        for tier in tier_ids:
            batches.extend(self._pack_tier(self._tier_to_indices[tier]))

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        if not hasattr(self, "_cached_len"):
            self._cached_len = sum(
                len(self._pack_tier(indices))
                for indices in self._tier_to_indices.values()
            )
        return self._cached_len


def collate_homogeneous_graphs(
    batch: List[Tuple[Data, torch.Tensor]],
    augment_metis_signs: bool = False,
) -> Data:
    """
    Custom collate function for batching (Data, positions) tuples.

    Args:
        batch: List of (Data, positions) tuples
        augment_metis_signs: If True, randomly flip the sign of each eigenvector
            column independently per graph.  Applied to:
              1. data.node_partition_pe + its slice of data.x  (quotient Laplacian PE)
              2. data.lap_eigenvectors  + its slice of data.x  (graph Laplacian PE)
            Enabled for training only; disabled for validation.  This is the standard
            sign-invariance augmentation for spectral positional encodings.

    Returns:
        batched_data: Batch object containing all graphs with positions stored in .pos
    """
    # Separate data and positions
    data_list = [item[0] for item in batch]
    positions_list = [item[1] for item in batch]

    # ---- Eigenvector sign-flip augmentation (training only) ----
    if augment_metis_signs:
        for data in data_list:
            x_cloned = False

            # (a) Quotient-graph Laplacian PE (metis_partition_pe in data.x + node_partition_pe)
            if (getattr(data, "node_partition_pe", None) is not None
                    and getattr(data, "metis_pe_col_start", None) is not None):
                max_k = data.node_partition_pe.shape[1]
                signs = torch.randint(0, 2, (max_k,), dtype=data.node_partition_pe.dtype,
                                      device=data.node_partition_pe.device) * 2 - 1
                data.node_partition_pe = data.node_partition_pe * signs.unsqueeze(0)
                col = data.metis_pe_col_start
                if not x_cloned:
                    data.x = data.x.clone()
                    x_cloned = True
                data.x[:, col:col + max_k] = data.x[:, col:col + max_k] * signs.unsqueeze(0)

            # (b) Graph Laplacian eigenvectors (lap_eigenvectors + eigenvectors slice of data.x)
            if getattr(data, "lap_eigenvectors", None) is not None:
                ndim = data.lap_eigenvectors.shape[1]
                signs_lap = torch.randint(0, 2, (ndim,), dtype=data.lap_eigenvectors.dtype,
                                          device=data.lap_eigenvectors.device) * 2 - 1
                data.lap_eigenvectors = data.lap_eigenvectors * signs_lap.unsqueeze(0)
                ev_col = getattr(data, "eigenvector_col_start", None)
                if ev_col is not None:
                    if not x_cloned:
                        data.x = data.x.clone()
                        x_cloned = True
                    data.x[:, ev_col:ev_col + ndim] = data.x[:, ev_col:ev_col + ndim] * signs_lap.unsqueeze(0)

    # Extract graph-level attributes that shouldn't be batched automatically
    # chip_size is (2,) per graph, not node-level, so PyG's Batch can't handle it
    chip_sizes = []
    scale_to_original_list = []
    for data in data_list:
        if hasattr(data, 'chip_size') and data.chip_size is not None:
            chip_sizes.append(data.chip_size.clone())
            # Temporarily remove chip_size to avoid batching issues
            delattr(data, 'chip_size')
        else:
            chip_sizes.append(None)
        if hasattr(data, 'scale_to_original') and data.scale_to_original is not None:
            scale_to_original_list.append(data.scale_to_original)
            delattr(data, 'scale_to_original')
        else:
            scale_to_original_list.append(None)

    # Batch Data objects using PyG's Batch
    # PyG's Batch.from_data_list preserves custom attributes like pos_original and norm_stats
    batched_data = Batch.from_data_list(data_list)

    # Restore chip_size and scale_to_original to individual Data objects (for potential future unbatching)
    for i, data in enumerate(data_list):
        if chip_sizes[i] is not None:
            data.chip_size = chip_sizes[i]
        if scale_to_original_list[i] is not None:
            data.scale_to_original = scale_to_original_list[i]

    # Concatenate positions
    batched_positions = torch.cat(positions_list, dim=0)

    # Store positions in the batched data for easy access
    batched_data.pos = batched_positions
    
    # Ensure pos_original is preserved if available (for unnormalized RMSE computation)
    if not hasattr(batched_data, 'pos_original') and all(hasattr(d, 'pos_original') for d in data_list):
        pos_original_list = [d.pos_original for d in data_list]
        batched_data.pos_original = torch.cat(pos_original_list, dim=0)

    # Preserve chip_size as (num_graphs, 2) so diffusion can use data.chip_size[data.batch] for error-percentage
    if all(cs is not None for cs in chip_sizes):
        chip_sizes_tensor = [cs.view(2) for cs in chip_sizes]
        batched_data.chip_size = torch.stack(chip_sizes_tensor, dim=0)  # (num_graphs, 2)

    # Preserve scale_to_original (IBM convert: HPWL_v5 * scale_to_original = HPWL_original µm)
    if any(s is not None for s in scale_to_original_list):
        batched_data.scale_to_original = scale_to_original_list

    # Per-graph node counts so callers can derive max individual graph size
    batched_data.graph_num_nodes = torch.tensor(
        [d.x.shape[0] for d in data_list], dtype=torch.long,
    )

    return batched_data
