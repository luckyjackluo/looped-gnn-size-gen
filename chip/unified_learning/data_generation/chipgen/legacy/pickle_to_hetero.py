"""Convert pickle files from chipdiffusion format to PyG HeteroData format."""

import pickle
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data, HeteroData

from tqdm import tqdm


def normalize_positions_to_minus_one_one(
    pos: torch.Tensor,
    chip_size: Optional[torch.Tensor] = None,
    chip_offset: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Normalize positions to [-1, 1] range.
    
    Based on chipdiffusion's preprocess_graph function:
    - If chip_size is provided: x = 2 * ((x - chip_offset) / scale / chip_size) - 1
    - Otherwise: normalize using min/max: x = 2 * (x - min) / (max - min) - 1
    
    Args:
        pos: (V, 2) tensor of positions
        chip_size: Optional (1, 2) or (2,) tensor of chip size [W, H]
        chip_offset: Optional (1, 2) or (2,) tensor of chip offset [x0, y0]
        scale: Scale factor (default: 1.0)
        
    Returns:
        Normalized positions in [-1, 1] range
    """
    pos = pos.clone()
    
    if chip_size is not None:
        # Use chip_size-based normalization (like chipdiffusion)
        chip_size = chip_size.view(1, 2) if chip_size.dim() == 1 else chip_size
        if chip_offset is not None:
            chip_offset = chip_offset.view(1, 2) if chip_offset.dim() == 1 else chip_offset
            pos = (pos - chip_offset) / scale
        pos = 2 * (pos / chip_size) - 1
    else:
        # Fallback: normalize using min/max
        pos_min = pos.min(dim=0, keepdim=True)[0]
        pos_max = pos.max(dim=0, keepdim=True)[0]
        pos_range = pos_max - pos_min
        pos_range = torch.clamp(pos_range, min=1e-8)  # Avoid division by zero
        pos = 2 * (pos - pos_min) / pos_range - 1
    
    return pos


def denormalize_positions_from_minus_one_one(
    pos_normalized: torch.Tensor,
    chip_size: Optional[torch.Tensor] = None,
    chip_offset: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    pos_min: Optional[torch.Tensor] = None,
    pos_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Denormalize positions from [-1, 1] range back to original coordinates.
    
    This is the inverse of normalize_positions_to_minus_one_one.
    
    Args:
        pos_normalized: (V, 2) tensor of normalized positions in [-1, 1]
        chip_size: Optional (1, 2) or (2,) tensor of chip size [W, H] (for chip_size-based denormalization)
        chip_offset: Optional (1, 2) or (2,) tensor of chip offset [x0, y0]
        scale: Scale factor (default: 1.0)
        pos_min: Optional (1, 2) or (2,) tensor of minimum positions (for min/max denormalization)
        pos_max: Optional (1, 2) or (2,) tensor of maximum positions (for min/max denormalization)
        
    Returns:
        Denormalized positions in original coordinate space
    """
    pos = pos_normalized.clone()
    
    if chip_size is not None:
        # Use chip_size-based denormalization (inverse of chip_size normalization)
        chip_size = chip_size.view(1, 2) if chip_size.dim() == 1 else chip_size
        # Inverse: x = chip_size * (x_norm + 1) / 2
        pos = chip_size * (pos + 1) / 2
        if chip_offset is not None:
            chip_offset = chip_offset.view(1, 2) if chip_offset.dim() == 1 else chip_offset
            pos = pos * scale + chip_offset
    elif pos_min is not None and pos_max is not None:
        # Use min/max denormalization (inverse of min/max normalization)
        pos_min = pos_min.view(1, 2) if pos_min.dim() == 1 else pos_min
        pos_max = pos_max.view(1, 2) if pos_max.dim() == 1 else pos_max
        pos_range = pos_max - pos_min
        pos_range = torch.clamp(pos_range, min=1e-8)  # Avoid division by zero
        # Inverse: x = pos_min + (x_norm + 1) * pos_range / 2
        pos = pos_min + (pos + 1) * pos_range / 2
    else:
        # If no denormalization info provided, return as-is (assume already denormalized)
        pass
    
    return pos


def normalize_by_graph_statistics(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    return_stats: bool = True
) -> Union[Tuple[torch.Tensor, torch.Tensor, dict], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Normalize positions and sizes by graph-level statistics instead of chip_size.
    
    This approach ensures instance sizes remain consistent across different graph sizes.
    
    Args:
        pos: (V, 2) tensor of positions (can be in any scale)
        sizes: (V, 2) tensor of instance sizes [width, height]
        return_stats: If True, return normalization statistics for later denormalization
        
    Returns:
        pos_normalized: (V, 2) tensor in [-1, 1] range
        sizes_normalized: (V, 2) tensor normalized by graph median size
        stats (optional): Dict with {pos_min, pos_max, pos_range, size_median, size_mean}
    """
    pos = pos.clone()
    sizes = sizes.clone()
    
    # Normalize positions by actual extent (not chip_size)
    pos_min = pos.min(dim=0, keepdim=True)[0]  # (1, 2)
    pos_max = pos.max(dim=0, keepdim=True)[0]  # (1, 2)
    pos_range = pos_max - pos_min
    pos_range = torch.clamp(pos_range, min=1e-8)  # Avoid division by zero
    
    pos_normalized = 2 * (pos - pos_min) / pos_range - 1  # [-1, 1]
    
    # Normalize sizes by MEDIAN size in this graph (robust to outliers)
    areas = sizes[:, 0] * sizes[:, 1]
    size_median = areas.median()
    size_mean = areas.mean()
    
    # Scale sizes relative to typical instance size in this graph
    # This makes instance sizes comparable across different graphs
    sizes_normalized = sizes / torch.sqrt(size_median).clamp(min=1e-8)
    
    if return_stats:
        stats = {
            'pos_min': pos_min,
            'pos_max': pos_max,
            'pos_range': pos_range,
            'size_median': size_median,
            'size_mean': size_mean,
            'num_instances': pos.shape[0],
        }
        return pos_normalized, sizes_normalized, stats
    else:
        return pos_normalized, sizes_normalized


def compute_graph_statistics(pos: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    Compute graph-level statistics for conditioning.
    
    These statistics help the model understand the scale and characteristics of the graph.
    
    Args:
        pos: (V, 2) tensor of positions (normalized or unnormalized)
        sizes: (V, 2) tensor of instance sizes
        
    Returns:
        graph_stats: (5,) tensor with [num_instances_norm, mean_area_norm, median_area_norm, 
                                        density_estimate, aspect_ratio_mean]
    """
    V = pos.shape[0]
    
    # Number of instances (normalized to [0, 1] range, log scale)
    # Assuming typical range is [10, 1000] instances
    num_instances_norm = torch.log10(torch.tensor(V, dtype=torch.float32).clamp(min=10, max=10000)) / 4.0  # [0, 1]
    
    # Instance areas
    areas = sizes[:, 0] * sizes[:, 1]
    mean_area = areas.mean()
    median_area = areas.median()
    
    # Normalize areas (assuming typical range [0.001, 0.1])
    mean_area_norm = torch.log10(mean_area.clamp(min=1e-4, max=1.0) + 1e-6) / 4.0 + 1.0  # Roughly [0, 1]
    median_area_norm = torch.log10(median_area.clamp(min=1e-4, max=1.0) + 1e-6) / 4.0 + 1.0
    
    # Estimate density (total area / bounding box area)
    pos_min = pos.min(dim=0)[0]
    pos_max = pos.max(dim=0)[0]
    bbox_area = ((pos_max[0] - pos_min[0]) * (pos_max[1] - pos_min[1])).clamp(min=1e-8)
    total_area = areas.sum()
    density_estimate = (total_area / bbox_area).clamp(0.0, 1.0)
    
    # Average aspect ratio
    aspect_ratios = (sizes[:, 0] / sizes[:, 1].clamp(min=1e-8)).clamp(0.1, 10.0)
    aspect_ratio_mean = aspect_ratios.mean() / 10.0  # Normalize to roughly [0, 1]
    
    graph_stats = torch.stack([
        num_instances_norm,
        mean_area_norm,
        median_area_norm,
        density_estimate,
        aspect_ratio_mean,
    ])
    
    return graph_stats


def load_pickle_file(pickle_path: Union[str, Path]) -> List[Tuple[torch.Tensor, Data]]:
    """
    Load pickle file containing list of (x, cond) tuples.
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        List of (x, cond) tuples where:
        - x: (V, 2) tensor of positions
        - cond: PyG Data object with graph structure
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        # Check if it's list of (x, cond) tuples or list of Data objects
        if len(data) > 0:
            first = data[0]
            if isinstance(first, tuple) and len(first) == 2:
                # List of (x, cond) tuples
                return data
            elif isinstance(first, Data):
                # List of Data objects - need to extract positions
                # This format doesn't have separate x, so we'll need to handle it differently
                raise ValueError("Pickle file contains list of Data objects without positions. "
                               "Expected format: list of (x, cond) tuples.")
        return data
    elif isinstance(data, tuple) and len(data) == 2:
        # Single tuple
        return [data]
    else:
        raise ValueError(f"Unexpected pickle file format: {type(data)}")


def data_to_heterodata(
    x: torch.Tensor, 
    cond: Data, 
    device: str = "cpu",
    normalize_positions: bool = True,
    normalize_by_graph_stats: bool = False,
    add_graph_stats: bool = False,
    chip_size: Optional[torch.Tensor] = None,
    chip_offset: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> HeteroData:
    """
    Convert PyG Data object to HeteroData format.
    
    The Data object has:
    - x: (V, 2) sizes [width, height]
    - edge_index: (2, E) connectivity (instance to instance)
    - edge_attr: (E, 4) pin attributes [src_pin_x, src_pin_y, dst_pin_x, dst_pin_y]
    - is_ports: (V,) boolean mask
    
    We need to convert to HeteroData with:
    - inst.x: (V, 7) instance features
    - net.x: (N_net, 3) net features
    - inst->net edges: (2, E) [inst_id, net_id]
    - inst.pos: (V, 2) positions (normalized to [-1, 1])
    
    Args:
        x: (V, 2) tensor of instance positions (centers)
        cond: PyG Data object with graph structure
        device: Device for tensors
        normalize_positions: If True, normalize positions to [-1, 1] range
        normalize_by_graph_stats: If True, use graph-level statistics for normalization 
                                   instead of chip_size (more consistent across graph sizes)
        add_graph_stats: If True, append graph-level statistics to instance features
        chip_size: Optional chip size for normalization (if available)
        chip_offset: Optional chip offset for normalization
        scale: Scale factor for normalization
        
    Returns:
        HeteroData object
    """
    # Move to device
    x = x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
    
    # Extract data from cond
    sizes = cond.x.to(device) if hasattr(cond, 'x') else None  # (V, 2) or (V, F)
    
    # Keep track of unnormalized sizes for feature computation
    sizes_unnormalized = sizes.clone() if sizes is not None else None
    
    # Store normalization statistics for later denormalization
    norm_stats_dict = {}
    
    # Normalize positions and sizes
    if normalize_positions:
        if normalize_by_graph_stats:
            # Use graph-level statistics for normalization (more consistent across graph sizes)
            x_original = x.clone()  # Keep original for computing stats
            x, sizes_normalized, norm_stats = normalize_by_graph_statistics(x, sizes, return_stats=True)
            # Keep unnormalized sizes for computing absolute area features
            # Update sizes to normalized version for other computations
            sizes = sizes_normalized
            # Store normalization stats for denormalization
            norm_stats_dict = {
                'normalize_by_graph_stats': True,
                'pos_min': norm_stats['pos_min'],
                'pos_max': norm_stats['pos_max'],
                'pos_range': norm_stats['pos_range'],
            }
        else:
            # Use chip_size-based normalization (original approach)
            # Check if chip_size is available in cond
            if chip_size is None and hasattr(cond, 'chip_size') and cond.chip_size is not None:
                chip_size_val = cond.chip_size
                if isinstance(chip_size_val, (list, tuple)):
                    chip_size = torch.tensor(chip_size_val, dtype=torch.float32, device=device)
                elif isinstance(chip_size_val, torch.Tensor):
                    chip_size = chip_size_val.to(device)
            
            x_original = x.clone()  # Keep original for computing stats
            x = normalize_positions_to_minus_one_one(x, chip_size, chip_offset, scale)
            # Store normalization stats for denormalization
            if chip_size is not None:
                norm_stats_dict = {
                    'normalize_by_graph_stats': False,
                    'chip_size': chip_size.clone(),
                    'chip_offset': chip_offset.clone() if chip_offset is not None else None,
                    'scale': scale,
                }
            else:
                # Fallback: compute min/max from original
                pos_min = x_original.min(dim=0, keepdim=True)[0]
                pos_max = x_original.max(dim=0, keepdim=True)[0]
                norm_stats_dict = {
                    'normalize_by_graph_stats': False,
                    'pos_min': pos_min,
                    'pos_max': pos_max,
                    'chip_size': None,
                }
    edge_index = cond.edge_index.to(device)  # (2, E)
    edge_attr = cond.edge_attr.to(device) if hasattr(cond, 'edge_attr') else None  # (E, 4)
    is_ports = cond.is_ports.to(device) if hasattr(cond, 'is_ports') else None  # (V,)
    
    V = x.shape[0]
    
    # Handle sizes - if cond.x has more than 2 features, take first 2
    if sizes is None:
        raise ValueError("Data object must have 'x' attribute with sizes")
    
    if sizes.shape[1] >= 2:
        widths = sizes[:, 0]
        heights = sizes[:, 1]
    else:
        raise ValueError(f"Expected sizes to have at least 2 features, got {sizes.shape[1]}")
    
    # Compute normalized areas
    areas = widths * heights
    
    # Also compute unnormalized areas if using graph-level normalization
    if normalize_positions and normalize_by_graph_stats and sizes_unnormalized is not None:
        widths_unnorm = sizes_unnormalized[:, 0]
        heights_unnorm = sizes_unnormalized[:, 1]
        areas_unnormalized = widths_unnorm * heights_unnorm
    else:
        # If not using graph-level normalization, normalized and unnormalized are the same
        areas_unnormalized = areas
    
    # Handle is_ports and is_macro
    if is_ports is not None:
        is_port = is_ports.float() if is_ports.dtype != torch.float32 else is_ports
    else:
        is_port = torch.zeros(V, dtype=torch.float32, device=device)
    
    # Check for is_macros (alternative naming)
    if hasattr(cond, 'is_macros'):
        is_macro = cond.is_macros.to(device).float()
    elif hasattr(cond, 'is_macro'):
        is_macro = cond.is_macro.to(device).float()
    else:
        # Infer macros from size (larger than average)
        area_threshold = areas.mean() + 2 * areas.std()
        is_macro = (areas > area_threshold).float()
    
    # Cell types: 0=port, 1=macro, 2-30=stdcell
    cell_type = torch.zeros(V, dtype=torch.long, device=device)
    cell_type[is_port.bool()] = 0
    cell_type[is_macro.bool()] = 1
    stdcell_mask = (is_port == 0) & (is_macro == 0)
    if stdcell_mask.any():
        # Assign stdcell types 2-30
        n_stdcells = stdcell_mask.sum().item()
        stdcell_types = torch.randint(2, 31, (n_stdcells,), device=device)
        cell_type[stdcell_mask] = stdcell_types
    
    # Pin capacity (estimate from degree)
    if edge_index.shape[1] > 0:
        degrees = torch.zeros(V, dtype=torch.float32, device=device)
        unique, counts = torch.unique(edge_index[0], return_counts=True)
        degrees[unique] = counts.float()
        pin_cap = degrees.clamp(min=2.0, max=12.0)
    else:
        pin_cap = torch.ones(V, dtype=torch.float32, device=device) * 2.0
    
    # Convert edge_index from instance-to-instance to instance-to-net
    # Each edge in the original graph becomes a net connecting two instances
    E = edge_index.shape[1]
    
    if E == 0:
        # No edges - create empty HeteroData
        hetero_data = HeteroData()
        
        # Instance features with both normalized and unnormalized areas
        if normalize_positions and normalize_by_graph_stats:
            inst_x = torch.stack([
                cell_type.float(),
                widths,
                heights,
                areas,
                areas_unnormalized,
                is_macro,
                is_port,
                pin_cap
            ], dim=1)  # (V, 8)
        else:
            inst_x = torch.stack([
                cell_type.float(),
                widths,
                heights,
                areas,
                is_macro,
                is_port,
                pin_cap
            ], dim=1)  # (V, 7)
        
        # Optionally append graph-level statistics
        if add_graph_stats:
            graph_stats = compute_graph_statistics(x, sizes)
            graph_stats_broadcast = graph_stats.unsqueeze(0).expand(V, -1)
            inst_x = torch.cat([inst_x, graph_stats_broadcast], dim=1)
        
        hetero_data['inst'].x = inst_x
        hetero_data['inst'].pos = x
        hetero_data['inst'].pos_mask = torch.ones(V, dtype=torch.float32, device=device)
        
        # Empty nets
        hetero_data['net'].x = torch.empty((0, 3), dtype=torch.float32, device=device)
        hetero_data['inst', 'to', 'net'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        hetero_data['inst', 'to', 'net'].edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device)
        
        return hetero_data
    
    # Handle undirected edges - edges are duplicated (forward and reverse)
    # We need to group edges into nets
    # Strategy: Each unique edge (ignoring direction) becomes a net
    
    # Get unique edges (undirected)
    src = edge_index[0]
    dst = edge_index[1]
    
    # Create undirected edge representation (smaller index first)
    undirected_edges = torch.stack([
        torch.minimum(src, dst),
        torch.maximum(src, dst)
    ], dim=0)
    
    # Find unique nets
    # Use a hash-like approach: create unique IDs for each undirected edge
    edge_ids = undirected_edges[0] * V + undirected_edges[1]
    unique_edge_ids, net_ids = torch.unique(edge_ids, return_inverse=True)
    N_net = len(unique_edge_ids)
    
    # Create inst_id_per_edge and net_id_per_edge
    inst_id_per_edge = edge_index[0]  # Source instances
    net_id_per_edge = net_ids  # Net IDs
    
    # Also add reverse edges (destination instances to same nets)
    inst_id_per_edge = torch.cat([inst_id_per_edge, edge_index[1]])
    net_id_per_edge = torch.cat([net_id_per_edge, net_ids])
    
    # Compute net degrees
    deg = torch.zeros(N_net, dtype=torch.long, device=device)
    for net_id in range(N_net):
        deg[net_id] = (net_id_per_edge == net_id).sum().item()
    
    # Create edge attributes
    if edge_attr is not None:
        if edge_attr.shape[1] >= 4:
            # edge_attr is [src_pin_x, src_pin_y, dst_pin_x, dst_pin_y]
            # For HeteroData, we need [pin_dx, pin_dy, is_driver, pin_role]
            # pin_dx, pin_dy are relative to instance center
            src_inst = edge_index[0]
            dst_inst = edge_index[1]
            
            # Get pin positions (absolute)
            src_pin_x = edge_attr[:, 0]
            src_pin_y = edge_attr[:, 1]
            dst_pin_x = edge_attr[:, 2]
            dst_pin_y = edge_attr[:, 3]
            
            # Convert to relative offsets (from instance center)
            # Instance centers are at x[inst_id]
            src_centers = x[src_inst]  # (E, 2)
            dst_centers = x[dst_inst]  # (E, 2)
            
            src_pin_pos = torch.stack([src_pin_x, src_pin_y], dim=1)  # (E, 2)
            dst_pin_pos = torch.stack([dst_pin_x, dst_pin_y], dim=1)  # (E, 2)
            
            src_offset = src_pin_pos - src_centers  # (E, 2)
            dst_offset = dst_pin_pos - dst_centers  # (E, 2)
            
            # Create edge attributes: [pin_dx, pin_dy, is_driver, pin_role]
            # For src->net: source is driver
            edge_attr_src = torch.cat([
                src_offset,
                torch.ones(E, 1, device=device),  # is_driver
                torch.zeros(E, 1, device=device)  # pin_role
            ], dim=1)
            
            # For dst->net: destination is not driver
            edge_attr_dst = torch.cat([
                dst_offset,
                torch.zeros(E, 1, device=device),  # is_driver
                torch.ones(E, 1, device=device)  # pin_role
            ], dim=1)
            
            edge_attr_hetero = torch.cat([edge_attr_src, edge_attr_dst], dim=0)
        else:
            # Fallback: create default edge attributes
            edge_attr_hetero = torch.zeros(
                inst_id_per_edge.shape[0], 4, dtype=torch.float32, device=device
            )
    else:
        # No edge attributes - create defaults
        edge_attr_hetero = torch.zeros(
            inst_id_per_edge.shape[0], 4, dtype=torch.float32, device=device
        )
    
    # Create hierarchy levels (dummy - not used in placement-first)
    level = torch.zeros(N_net, dtype=torch.long, device=device)
    
    # Create HeteroData
    hetero_data = HeteroData()
    
    # Instance features: [cell_type, w_norm, h_norm, area_norm, area_unnorm, is_macro, is_port, pin_cap]
    # Include both normalized and unnormalized areas when using graph-level normalization
    if normalize_positions and normalize_by_graph_stats:
        inst_x = torch.stack([
            cell_type.float(),
            widths,              # normalized width
            heights,             # normalized height
            areas,               # normalized area
            areas_unnormalized,  # unnormalized (absolute) area
            is_macro,
            is_port,
            pin_cap
        ], dim=1)  # (V, 8)
    else:
        # Standard features (backward compatible)
        inst_x = torch.stack([
            cell_type.float(),
            widths,
            heights,
            areas,
            is_macro,
            is_port,
            pin_cap
        ], dim=1)  # (V, 7)
    
    # Optionally append graph-level statistics to each instance
    if add_graph_stats:
        # Compute graph statistics (before normalization, using original sizes)
        graph_stats = compute_graph_statistics(x, sizes)  # (5,)
        # Broadcast to all instances
        graph_stats_broadcast = graph_stats.unsqueeze(0).expand(V, -1)  # (V, 5)
        # Concatenate with instance features
        inst_x = torch.cat([inst_x, graph_stats_broadcast], dim=1)  # (V, 7+5=12)
    
    hetero_data['inst'].x = inst_x
    hetero_data['inst'].pos = x
    hetero_data['inst'].pos_mask = torch.ones(V, dtype=torch.float32, device=device)
    
    # Net features: [degree, is_huge, level]
    huge_threshold = 50
    is_huge = (deg > huge_threshold).float()
    net_x = torch.stack([
        deg.float(),
        is_huge,
        level.float()
    ], dim=1)
    
    hetero_data['net'].x = net_x
    
    # Edges
    hetero_data['inst', 'to', 'net'].edge_index = torch.stack([
        inst_id_per_edge, net_id_per_edge
    ], dim=0)
    hetero_data['inst', 'to', 'net'].edge_attr = edge_attr_hetero
    
    # Store normalization statistics for later denormalization
    if normalize_positions and norm_stats_dict:
        hetero_data.norm_stats = norm_stats_dict
    
    return hetero_data


def convert_pickle_to_hetero(
    pickle_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    normalize_positions: bool = True,
    chip_size: Optional[torch.Tensor] = None,
    chip_offset: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> List[HeteroData]:
    """
    Convert pickle file to list of HeteroData objects.
    
    Args:
        pickle_path: Path to pickle file
        output_path: Optional path to save converted data (as .pt file)
        device: Device for tensors
        
    Returns:
        List of HeteroData objects
    """
    # Load pickle file
    samples = load_pickle_file(pickle_path)
    
    # Convert each sample
    hetero_graphs = []
    for x, cond in samples:
        hetero_data = data_to_heterodata(
            x, cond, 
            device=device,
            normalize_positions=normalize_positions,
            chip_size=chip_size,
            chip_offset=chip_offset,
            scale=scale
        )
        hetero_graphs.append(hetero_data)
    
    # Save if output path provided
    if output_path is not None:
        torch.save(hetero_graphs, output_path)
        print(f"Saved {len(hetero_graphs)} graphs to {output_path}")
    
    return hetero_graphs


def convert_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.pickle",
    device: str = "cpu"
):
    """
    Convert all pickle files in a directory to HeteroData format.
    
    Args:
        input_dir: Input directory containing pickle files
        output_dir: Output directory for converted .pt files
        pattern: Glob pattern for pickle files
        device: Device for tensors
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pickle_files = list(input_dir.glob(pattern))
    print(f"Found {len(pickle_files)} pickle files")
    
    for pickle_file in tqdm(pickle_files):
        output_file = output_dir / f"{pickle_file.stem}.pt"
        
        try:
            hetero_graphs = convert_pickle_to_hetero(pickle_file, output_file, device=device)
        except Exception as e:
            print(f"  Error converting {pickle_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert pickle files to HeteroData format")
    parser.add_argument("input", type=str, help="Input pickle file or directory")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--pattern", type=str, default="*.pickle", help="Glob pattern for files")
    parser.add_argument("--device", type=str, default="cpu", help="Device for tensors")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        output_path = args.output if args.output else input_path.with_suffix('.pt')
        convert_pickle_to_hetero(input_path, output_path, device=args.device)
    elif input_path.is_dir():
        # Directory
        output_dir = Path(args.output) if args.output else input_path / "hetero_converted"
        convert_directory(input_path, output_dir, pattern=args.pattern, device=args.device)
    else:
        print(f"Error: {input_path} is not a valid file or directory")

