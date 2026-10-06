"""Data loading utilities for converting HeteroData to homogeneous Data format for AttGNN."""

from typing import Optional, List, Tuple
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected


def heterodata_to_data(
    hetero_data: HeteroData,
    device: str = "cpu",
    aggregate_edge_attrs: str = "mean"
) -> Tuple[Data, torch.Tensor]:
    """
    Convert HeteroData (bipartite graph) to homogeneous Data format.
    
    Converts bipartite graph (inst-net) to homogeneous graph (inst-inst) by:
    - Creating instance-to-instance edges from inst-net-inst paths
    - Aggregating edge attributes from inst-net edges
    - Preserving node features: inst.x → Data.x
    - Extracting positions: inst.pos → separate tensor
    
    Args:
        hetero_data: HeteroData object with inst and net nodes
        device: Device for tensors
        aggregate_edge_attrs: How to aggregate edge attributes ("mean", "sum", "max", "min")
        
    Returns:
        Tuple of (Data object, positions tensor):
        - Data: Homogeneous graph with inst-inst edges
        - positions: (N_inst, 2) tensor of instance positions
    """
    # Extract instance features and positions
    inst_x = hetero_data['inst'].x.to(device)  # (N_inst, F_inst)
    inst_pos = hetero_data['inst'].pos.to(device) if hasattr(hetero_data['inst'], 'pos') and hetero_data['inst'].pos is not None else None  # (N_inst, 2)
    
    # Extract net features (for reference, but not used in homogeneous graph)
    net_x = hetero_data['net'].x.to(device) if 'net' in hetero_data.node_types else None  # (N_net, F_net)
    
    # Extract inst-net edges
    if ('inst', 'to', 'net') in hetero_data.edge_types:
        inst_net_edge_index = hetero_data[('inst', 'to', 'net')].edge_index.to(device)  # (2, E_inst_net)
        inst_net_edge_attr = hetero_data[('inst', 'to', 'net')].edge_attr.to(device) if hasattr(hetero_data[('inst', 'to', 'net')], 'edge_attr') and hetero_data[('inst', 'to', 'net')].edge_attr is not None else None  # (E_inst_net, F_edge)
    else:
        # No edges - create empty graph
        data = Data(
            x=inst_x,
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            edge_attr=None
        )
        return data, inst_pos if inst_pos is not None else torch.zeros((inst_x.shape[0], 2), device=device)
    
    N_inst = inst_x.shape[0]
    N_net = net_x.shape[0] if net_x is not None else (inst_net_edge_index[1].max().item() + 1 if inst_net_edge_index.shape[1] > 0 else 0)
    
    # Convert inst-net edges to inst-inst edges
    # Strategy: For each net, create edges between all pairs of instances connected to that net
    # This creates a clique for each net
    
    # Group instances by net
    inst_ids = inst_net_edge_index[0]  # (E_inst_net,)
    net_ids = inst_net_edge_index[1]    # (E_inst_net,)
    
    # Build inst-inst edges
    inst_inst_edges = []
    inst_inst_edge_attrs = []
    
    # For each net, create edges between all pairs of instances
    for net_id in range(N_net):
        # Find all instances connected to this net
        net_mask = (net_ids == net_id)
        connected_insts = inst_ids[net_mask].unique()
        
        if len(connected_insts) < 2:
            # Need at least 2 instances to create an edge
            continue
        
        # Create edges between all pairs (undirected)
        for i in range(len(connected_insts)):
            for j in range(i + 1, len(connected_insts)):
                inst_i = connected_insts[i].item()
                inst_j = connected_insts[j].item()
                
                # Add both directions for undirected graph
                inst_inst_edges.append([inst_i, inst_j])
                inst_inst_edges.append([inst_j, inst_i])
                
                # Aggregate edge attributes from inst-net edges
                if inst_net_edge_attr is not None:
                    # Get edge attributes for this net
                    net_edge_mask = net_mask
                    net_edge_attrs = inst_net_edge_attr[net_edge_mask]  # (num_insts_in_net, F_edge)
                    
                    # Aggregate attributes
                    if aggregate_edge_attrs == "mean":
                        agg_attr = net_edge_attrs.mean(dim=0)
                    elif aggregate_edge_attrs == "sum":
                        agg_attr = net_edge_attrs.sum(dim=0)
                    elif aggregate_edge_attrs == "max":
                        agg_attr = net_edge_attrs.max(dim=0)[0]
                    elif aggregate_edge_attrs == "min":
                        agg_attr = net_edge_attrs.min(dim=0)[0]
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregate_edge_attrs}")
                    
                    # Use same aggregated attribute for both directions
                    inst_inst_edge_attrs.append(agg_attr)
                    inst_inst_edge_attrs.append(agg_attr)
    
    if len(inst_inst_edges) == 0:
        # No edges created - create empty graph
        edge_attr_dim = inst_net_edge_attr.shape[1] if inst_net_edge_attr is not None and len(inst_net_edge_attr.shape) > 1 else 0
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32, device=device) if edge_attr_dim > 0 else None
    else:
        # Convert to tensor
        edge_index = torch.tensor(inst_inst_edges, dtype=torch.long, device=device).t()  # (2, E_inst_inst)
        
        # Remove duplicate edges (keep only unique edges)
        # Use to_undirected to ensure we have a clean undirected graph
        edge_attr_tensor = torch.stack(inst_inst_edge_attrs) if inst_inst_edge_attrs else None
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr_tensor)
    
    # Create Data object
    data = Data(
        x=inst_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    
    # Add any additional attributes from hetero_data['inst']
    if hasattr(hetero_data['inst'], 'batch'):
        data.batch = hetero_data['inst'].batch.to(device)
    if hasattr(hetero_data['inst'], 'pos_mask'):
        data.pos_mask = hetero_data['inst'].pos_mask.to(device)
    
    # Return positions (or create zeros if not available)
    positions = inst_pos if inst_pos is not None else torch.zeros((N_inst, 2), device=device)
    
    return data, positions


def load_graph_data_from_dir(
    data_dir: str,
    max_samples: Optional[int] = None,
    debug: bool = False,
    normalize_positions: bool = True,
    convert_to_data: bool = False,
    aggregate_edge_attrs: str = "mean"
) -> List[Tuple[Data, torch.Tensor]]:
    """
    Load HeteroData files and optionally convert to Data format.
    
    Args:
        data_dir: Directory containing .pt files with HeteroData
        max_samples: Maximum number of samples to load
        debug: If True, only load first file
        normalize_positions: If True, normalize positions to [-1, 1]
        convert_to_data: If True, convert HeteroData to Data format
        aggregate_edge_attrs: How to aggregate edge attributes when converting
        
    Returns:
        If convert_to_data=True: List of (Data, positions) tuples
        If convert_to_data=False: List of HeteroData objects (same as original function)
    """
    from pathlib import Path
    from tqdm import tqdm
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find all .pt files recursively
    pt_files = sorted(data_dir.rglob("*.pt"))
    
    if len(pt_files) == 0:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    if debug:
        print(f"DEBUG MODE: Loading only the first .pt file")
        pt_files = pt_files[:1]
    else:
        print(f"Found {len(pt_files)} .pt files in {data_dir}")
    
    all_data = []
    for pt_file in tqdm(pt_files, desc="Loading data"):
        try:
            loaded = torch.load(pt_file, map_location='cpu')
            
            # Handle different formats
            if isinstance(loaded, HeteroData):
                data_list = [loaded]
            elif isinstance(loaded, list):
                data_list = [d for d in loaded if isinstance(d, HeteroData)]
            elif isinstance(loaded, tuple):
                data_list = [d for d in loaded if isinstance(d, HeteroData)]
            else:
                print(f"Warning: Unexpected format in {pt_file}: {type(loaded)}, skipping")
                continue
            
            # Convert if requested
            if convert_to_data:
                for hetero_data in data_list:
                    data, positions = heterodata_to_data(
                        hetero_data,
                        device='cpu',
                        aggregate_edge_attrs=aggregate_edge_attrs
                    )
                    all_data.append((data, positions))
            else:
                all_data.extend(data_list)
            
            # Check if we've reached max_samples
            if max_samples is not None and len(all_data) >= max_samples:
                all_data = all_data[:max_samples]
                break
            
            # In debug mode, stop after first file
            if debug:
                break
                
        except Exception as e:
            print(f"Warning: Failed to load {pt_file}: {e}, skipping")
            continue
    
    print(f"Loaded {len(all_data)} samples")
    return all_data
