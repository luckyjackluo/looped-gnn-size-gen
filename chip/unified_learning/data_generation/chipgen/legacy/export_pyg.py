"""Export to PyTorch Geometric HeteroData format."""

import torch
from typing import Dict

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


def to_heterodata(
    inst_feats: Dict[str, torch.Tensor],
    deg: torch.Tensor,
    level: torch.Tensor,
    inst_id_per_edge: torch.Tensor,
    net_id_per_edge: torch.Tensor,
    edge_attr: torch.Tensor,
    device: str = "cuda"
) -> "HeteroData":
    """
    Convert generated tensors to PyG HeteroData format.

    Schema:
    - data['inst'].x: [N_inst, F] instance features
    - data['net'].x: [N_net, G] net features
    - data['inst', 'to', 'net'].edge_index: [2, E]
    - data['inst', 'to', 'net'].edge_attr: [E, 4]

    Args:
        inst_feats: Dictionary with instance features
        deg: [N_net] net degrees
        level: [N_net] hierarchy levels
        inst_id_per_edge: [E] instance IDs
        net_id_per_edge: [E] net IDs
        edge_attr: [E, 4] edge attributes
        device: device

    Returns:
        HeteroData object
    """
    if HeteroData is None:
        raise ImportError("torch_geometric not installed. Install with: pip install torch-geometric")

    data = HeteroData()

    # --- Instance node features ---
    # [cell_type, w, h, area, is_macro, is_port, pin_cap]
    inst_x = torch.stack([
        inst_feats["cell_type"].float(),
        inst_feats["w"],
        inst_feats["h"],
        inst_feats["area"],
        inst_feats["is_macro"],
        inst_feats["is_port"],
        inst_feats["pin_cap"],
    ], dim=1)

    data['inst'].x = inst_x

    # Optional: positions and pos_mask
    if "pos" in inst_feats:
        data['inst'].pos = inst_feats["pos"]
        data['inst'].pos_mask = inst_feats["pos_mask"]

    # --- Net node features ---
    # [degree, is_huge, level]
    N_net = len(deg)
    huge_threshold = 50  # can be passed as parameter
    is_huge = (deg > huge_threshold).float()

    net_x = torch.stack([
        deg.float(),
        is_huge,
        level.float(),
    ], dim=1)

    data['net'].x = net_x

    # --- Edges ---
    edge_index = torch.stack([inst_id_per_edge, net_id_per_edge], dim=0)

    data['inst', 'to', 'net'].edge_index = edge_index
    data['inst', 'to', 'net'].edge_attr = edge_attr

    # Optionally add reverse edges (many GNNs need this)
    # data['net', 'to', 'inst'].edge_index = edge_index.flip(0)

    return data


def save_heterodata(data: "HeteroData", path: str):
    """Save HeteroData to disk."""
    torch.save(data, path)


def load_heterodata(path: str) -> "HeteroData":
    """Load HeteroData from disk."""
    return torch.load(path)
