"""Prediction heads for looped placement regression.

All heads share weights across the K loop iterations — same nn.Module
instance is called once per iteration, not separate instances per step.
"""

import torch
import torch.nn as nn


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = torch.zeros(dim_size, src.shape[1] if src.dim() > 1 else 1, device=src.device, dtype=src.dtype)
    if src.dim() == 1:
        src = src.unsqueeze(1)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    return (out / count.clamp_min(1.0)).squeeze(-1) if out.shape[1] == 1 else out / count.clamp_min(1.0)


class _MLP(nn.Module):
    """Simple feedforward MLP with SiLU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None, num_layers: int = 2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        layers = []
        cur = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(cur, hidden_dim), nn.SiLU()]
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodePredictionHead(nn.Module):
    """
    Predicts absolute (x, y) position in row-height units for each node.

    Input:
        h:          [N, d] node embeddings at iteration k
        he:         [E, d] edge embeddings at iteration k
        edge_index: [2, E]

    Output: [N, 2] predicted positions in row-height units.

    Design: mean-aggregates edge embeddings into destination nodes, then
    concatenates with node embedding before the final MLP.  Aggregating
    edge info gives the head access to local topological context that
    isn't fully absorbed by the node state.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = _MLP(2 * hidden_dim, 2, hidden_dim=hidden_dim, num_layers=3)

    def forward(
        self,
        h: torch.Tensor,
        he: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        # Mean-aggregate incident edge embeddings into each node
        m = _scatter_mean(
            self.edge_agg_proj(he), dst, dim=0, dim_size=h.shape[0]
        )  # [N, d]
        combined = torch.cat([h, m], dim=-1)  # [N, 2d]
        return self.mlp(combined)  # [N, 2]


class EdgePredictionHead(nn.Module):
    """
    Predicts pairwise displacement r_ij = (x_i - x_j) / h_row for each edge.

    This is used ONLY as an auxiliary loss — it is never fed back into the
    model state.  The O(1) magnitude target (vs O(sqrt(N)) for absolute
    positions) provides a clean gradient signal at every iteration and
    regularises the edge embedding quality.

    Output: [E, 2] pairwise displacements in row-height units.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = _MLP(hidden_dim, 2, hidden_dim=hidden_dim // 2, num_layers=2)

    def forward(self, he: torch.Tensor) -> torch.Tensor:
        return self.mlp(he)  # [E, 2]


class ConvergenceHead(nn.Module):
    """
    Predicts a convergence logit from global node pooling.

    sigmoid(logit) = c where:
      c = 1  →  model believes placement has converged (early exit OK).
      c = 0  →  model believes more iterations are needed.

    Supervision (during training):
        c_gt = sigmoid(-gamma * MSE(x^(k), x_gt) / sqrt(N))

    Used for adaptive early-exit at inference (exit when sigmoid(logit) > threshold).

    Batch support: if batch tensor is provided, returns one score per
    graph in the batch. Otherwise returns a single scalar.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = _MLP(hidden_dim, 1, hidden_dim=hidden_dim // 2, num_layers=2)

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)  # [1, d]
        else:
            pooled = _scatter_mean(h, batch, dim=0)  # [B, d]
        return self.mlp(pooled).squeeze(-1)  # scalar or [B] logits
