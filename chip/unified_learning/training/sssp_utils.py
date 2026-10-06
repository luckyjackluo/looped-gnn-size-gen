"""
Utilities for Single-Source Shortest Path (SSSP) regression on graphs.

- Compute edge attributes (distance, dx, dy, dz) from node positions.
- Compute SSSP distances from a source node using edge lengths (Dijkstra).
- SSSPHead: predict per-node distance from source given node embeddings and source index.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Optional

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_edge_attr_from_pos(data: Data, edge_dim: int = 4) -> Data:
    """
    Add edge_attr to a graph from node positions.
    
    For each edge (u, v): distance = ||pos[v] - pos[u]||,
    and optionally [dist, dx, dy, dz] for edge_dim=4.
    
    Args:
        data: PyG Data with pos [N, 3], edge_index [2, E]
        edge_dim: 1 for distance only, 4 for [dist, dx, dy, dz]
    
    Returns:
        data (cloned) with data.edge_attr [E, edge_dim] set.
    """
    data = data.clone()
    pos = data.pos
    edge_index = data.edge_index
    u, v = edge_index[0], edge_index[1]
    diff = pos[v] - pos[u]  # [E, 3]
    dist = (diff ** 2).sum(dim=1, keepdim=True).sqrt()  # [E, 1]
    dist = dist.clamp(min=1e-6)  # avoid zeros for numerical stability
    if edge_dim == 1:
        data.edge_attr = dist
    else:
        # [dist, dx, dy, dz]
        data.edge_attr = torch.cat([dist, diff], dim=1)
    return data


def sssp_distances_numpy(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_nodes: int,
    source: int,
) -> np.ndarray:
    """
    Compute shortest path distances from source to all nodes (Dijkstra).
    
    Args:
        edge_index: [2, E] int array (src, dst)
        edge_weight: [E] float array, must be non-negative
        num_nodes: number of nodes
        source: source node index (0 to num_nodes-1)
    
    Returns:
        dist: [num_nodes] float array; unreachable nodes get np.inf
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for SSSP. Install with: pip install scipy")
    row, col = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    w = np.asarray(edge_weight, dtype=np.float64).ravel()
    # Undirected: add both (u,v) and (v,u) with same weight
    row2 = np.concatenate([row, col])
    col2 = np.concatenate([col, row])
    w2 = np.concatenate([w, w])
    m = csr_matrix((w2, (row2, col2)), shape=(num_nodes, num_nodes))
    dists = dijkstra(m, indices=source)
    return np.asarray(dists, dtype=np.float32)


def compute_sssp_labels(
    data: Data,
    source_idx: int,
    edge_weight_idx: int = 0,
) -> torch.Tensor:
    """
    Compute SSSP distances from source_idx to all nodes.
    
    Uses data.edge_index and data.edge_attr[:, edge_weight_idx] as edge lengths.
    
    Args:
        data: PyG Data with edge_index, edge_attr
        source_idx: source node index (0 to N-1)
        edge_weight_idx: column index in edge_attr for weight (default 0 = distance)
    
    Returns:
        y: [num_nodes] tensor of shortest path distances from source
    """
    ei = data.edge_index
    if data.edge_attr.dim() == 1:
        w = data.edge_attr
    else:
        w = data.edge_attr[:, edge_weight_idx]
    n = data.num_nodes
    ei_np = ei.cpu().numpy()
    w_np = w.cpu().numpy()
    dist = sssp_distances_numpy(ei_np, w_np, n, source_idx)
    return torch.from_numpy(dist).to(dtype=data.pos.dtype, device=data.pos.device)


def prepare_graph_for_sssp(data: Data, edge_dim: int = 4) -> Data:
    """
    Prepare a Perlin-style graph for SSSP: set data.x from pos and add edge_attr.
    """
    data = data.clone()
    if not hasattr(data, 'x') or data.x is None:
        if hasattr(data, 'pos') and data.pos is not None:
            data.x = data.pos.clone()
        else:
            raise ValueError("Both data.x and data.pos are None")
    data = compute_edge_attr_from_pos(data, edge_dim=edge_dim)
    return data


class SSSPHead(nn.Module):
    """
    Predict single-source shortest path distance per node from encoder embeddings.
    
    For each node i: input = concat(node_emb[i], source_emb) -> MLP -> scalar.
    source_emb is the embedding of the source node for the graph that node i belongs to.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        in_d = hidden_dim * 2  # node_emb + source_emb
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        node_emb: torch.Tensor,
        batch: Batch,
        source_global_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_emb: [N_total, hidden_dim] from encoder
            batch: PyG Batch with batch.batch [N_total], batch.ptr [num_graphs+1]
            source_global_idx: [num_graphs] global node index of source per graph
        
        Returns:
            pred: [N_total, 1] predicted SSSP distance per node
        """
        num_graphs = source_global_idx.size(0)
        # [num_graphs, hidden_dim]
        source_emb = node_emb[source_global_idx]
        # [N_total, hidden_dim]: for each node, the source embedding of its graph
        graph_idx = batch.batch
        source_emb_per_node = source_emb[graph_idx]
        inp = torch.cat([node_emb, source_emb_per_node], dim=1)
        return self.mlp(inp)
