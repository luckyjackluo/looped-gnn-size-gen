"""
Utilities for encoder-only velocity regression on Perlin terrain graphs.

- Prepare graphs by setting data.x from pos and computing edge_attr from pos.
- Validate / expose node-level velocity targets stored in graph attributes.
- Provide a simple per-node regression head for 2D velocity prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from unified_learning.training.sssp_utils import compute_edge_attr_from_pos


def prepare_graph_for_velocity_regression(
    data: Data,
    *,
    edge_dim: int = 4,
    target_attr: str = "slowness",
    expected_output_dim: Optional[int] = None,
) -> Data:
    """
    Prepare a Perlin graph for node-wise velocity regression.

    Expected target storage:
    - `data.y`: [N, 2] velocity vector per node, or
    - another attribute named by `target_attr`.
    """
    data = data.clone()
    if not hasattr(data, "x") or data.x is None:
        if hasattr(data, "pos") and data.pos is not None:
            data.x = data.pos.clone()
        else:
            raise ValueError("Both data.x and data.pos are None")

    data = compute_edge_attr_from_pos(data, edge_dim=edge_dim)

    target = getattr(data, target_attr, None)
    if target is None and target_attr == "slowness":
        if hasattr(data, "speed") and data.speed is not None:
            target = 1.0 / data.speed.clamp_min(1e-6)
        elif hasattr(data, "y") and data.y is not None and data.y.dim() == 2 and data.y.size(-1) == 2:
            speed = torch.linalg.norm(data.y, dim=-1, keepdim=True)
            target = 1.0 / speed.clamp_min(1e-6)
    if target is None:
        raise ValueError(
            f"Velocity target attribute '{target_attr}' not found on graph. "
            "Regenerate the Perlin graph dataset with --attach_velocity_targets."
        )
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if target.dim() != 2:
        raise ValueError(
            f"Expected target shape [num_nodes, channels], got {tuple(target.shape)}"
        )
    if expected_output_dim is not None and target.size(-1) != expected_output_dim:
        raise ValueError(
            f"Expected target dim {expected_output_dim}, got shape {tuple(target.shape)} "
            f"for attribute '{target_attr}'"
        )

    out_dtype = data.pos.dtype if hasattr(data, "pos") and data.pos is not None else data.x.dtype
    data.y = target.to(dtype=out_dtype)
    return data


class VelocityRegressionHead(nn.Module):
    """Predict a solver target per node from encoder embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        layers = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_emb)
