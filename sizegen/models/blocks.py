"""Shared building blocks: MLP, one PreNorm+residual message-passing step.

Design lessons ported from UnifiedLearning (looped_placement_impl.md §4,
loop_conditioner.py): PreNorm before every sub-block (PostNorm explodes at
large K), residual updates throughout (gradient highway through the unroll),
sum-free per-node aggregation (mean over neighbors keeps message magnitude
size-invariant).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class ProcessorBlock(nn.Module):
    """One local update step R_theta: GATv2 message passing + FFN.

    PreNorm + residual; purely local (graph-neighbor) — the receptive field
    grows exactly one hop per application, mirroring the task operator U_N.
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gnn = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = mlp(hidden_dim, 2 * hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = h + self.gnn(self.norm1(h), edge_index)
        h = h + self.ffn(self.norm2(h))
        return h
