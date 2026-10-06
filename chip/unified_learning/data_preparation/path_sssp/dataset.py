"""Path-graph single-source shortest-path dataset for FM diffusion experiments.

Graph structure
---------------
Linear chain:  0 ─ 1 ─ 2 ─ … ─ L  (bidirectional; L edges, L+1 nodes).

Source node:   always node 0.

Edge weights:  i.i.d. Uniform(weight_low, weight_high), all positive.
               w[i] is the weight of the undirected edge between node i and i+1.

Target labels: prefix-sum distances from the source.
               y_0 = 0
               y_i = w[0] + w[1] + … + w[i-1]

This is the simplest graph where the "causal / sequential" nature of SSSP is
exact: node i needs i hops of information to know its own distance.  A K-layer
GNN can exactly predict only nodes with i ≤ K; a looped model with R iterations
of K layers has effective receptive field R*K.

Node features
-------------
Only a binary source flag is used: x[i] = [1.0] if i == 0 else [0.0].
No positional or size features are included.  This is deliberate: any feature
encoding i/L or log(L) would make the feature distribution shift with path
length, preventing OOD size generalisation.  With only the source flag and edge
weights, every local neighbourhood looks identical regardless of L — the model
must learn a purely local Bellman-Ford update rule to succeed.

PyG Data fields
---------------
x              [L+1, 1]  node features: source flag only
edge_index     [2, 2L]   bidirectional COO edges
edge_attr      [2L, 1]   edge weights (same value for both directions)
y_node         [L+1, 1]  shortest-path distances from source (float32)
target_mask    [L+1]     all True — every node is supervised
hop_dist       [L+1]     hop distance from source = node index i (long)
path_length    scalar    L as a long tensor
"""

from __future__ import annotations

from typing import List

import torch
from torch_geometric.data import Data


NODE_FEAT_DIM: int = 1  # source flag only; no positional/size features


def build_path_sssp_graph(
    path_length: int,
    *,
    seed: int,
    weight_low: float = 0.1,
    weight_high: float = 1.0,
) -> Data:
    """Build one path-graph SSSP instance.

    Args:
        path_length: Number of edges L.  Produces L+1 nodes.
        seed:        RNG seed for reproducibility.
        weight_low:  Minimum edge weight (positive).
        weight_high: Maximum edge weight.
    """
    assert weight_low > 0, "All edge weights must be strictly positive."
    L = path_length
    N = L + 1

    rng = torch.Generator()
    rng.manual_seed(seed)
    w = weight_low + (weight_high - weight_low) * torch.rand(L, generator=rng)

    # Bidirectional edges: forward (i→i+1) then backward (i+1→i)
    fwd_src = torch.arange(L, dtype=torch.long)
    fwd_dst = torch.arange(1, N, dtype=torch.long)
    edge_index = torch.stack(
        [torch.cat([fwd_src, fwd_dst]), torch.cat([fwd_dst, fwd_src])],
        dim=0,
    )  # [2, 2L]
    edge_attr = w.repeat(2).unsqueeze(-1)  # [2L, 1]

    # Distances: prefix sums
    y_node = torch.zeros(N, 1, dtype=torch.float32)
    y_node[1:, 0] = w.cumsum(0)

    # Node features: source flag only.  No positional or size information.
    x = torch.zeros(N, 1)
    x[0, 0] = 1.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_node=y_node,
        target_mask=torch.ones(N, dtype=torch.bool),
        hop_dist=torch.arange(N, dtype=torch.long),
        path_length=torch.tensor(L, dtype=torch.long),
        num_nodes=N,
    )


def make_dataset(
    path_lengths: List[int],
    n_per_length: int,
    *,
    seed_offset: int = 0,
    weight_low: float = 0.1,
    weight_high: float = 1.0,
) -> List[Data]:
    """Generate a flat list of path-graph SSSP instances.

    Seeds are deterministic: seed = seed_offset + L * 10_000 + k, so train
    (seed_offset=0) and val (seed_offset=100_000) never share graphs.
    """
    graphs: List[Data] = []
    for L in path_lengths:
        for k in range(n_per_length):
            seed = seed_offset + L * 10_000 + k
            graphs.append(
                build_path_sssp_graph(
                    L, seed=seed, weight_low=weight_low, weight_high=weight_high
                )
            )
    return graphs
