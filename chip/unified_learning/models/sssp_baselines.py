"""Baseline models for SSSP diagnostics."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch


def build_source_target_features(
    batch: Batch,
    *,
    include_node_features: bool = False,
) -> torch.Tensor:
    """Build per-node ``[source, target]`` features for coordinate regression."""
    if not hasattr(batch, "pos") or batch.pos is None:
        raise ValueError("CoordinateOnlySSSPMLP requires batch.pos")
    if not hasattr(batch, "source_idx"):
        raise ValueError("CoordinateOnlySSSPMLP requires batch.source_idx")
    if not hasattr(batch, "ptr") or not hasattr(batch, "batch"):
        raise ValueError("CoordinateOnlySSSPMLP expects a PyG Batch")

    source_local = batch.source_idx
    if source_local.dim() > 1:
        source_local = source_local.reshape(-1)
    source_global = batch.ptr[:-1].to(source_local.device) + source_local.to(batch.ptr.device)
    source_pos = batch.pos[source_global.to(batch.pos.device)]
    source_pos_per_node = source_pos[batch.batch]
    parts = [source_pos_per_node, batch.pos]

    if include_node_features:
        if not hasattr(batch, "x") or batch.x is None:
            raise ValueError("include_node_features=True requires batch.x")
        source_x = batch.x[source_global.to(batch.x.device)]
        source_x_per_node = source_x[batch.batch]
        parts.extend([source_x_per_node, batch.x])
    return torch.cat(parts, dim=-1)


class CoordinateOnlySSSPMLP(nn.Module):
    """Predict SSSP distance from source/target coordinates only."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "silu",
        include_node_features: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.include_node_features = bool(include_node_features)

        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.SiLU()

        layers = []
        dim = int(input_dim)
        for layer_idx in range(num_layers):
            out_dim = int(output_dim) if layer_idx == num_layers - 1 else int(hidden_dim)
            layers.append(nn.Linear(dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
            dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Batch, x_t=None):  # noqa: ANN001 - matches UnifiedModel API
        del x_t
        features = build_source_target_features(
            batch,
            include_node_features=self.include_node_features,
        )
        pred = self.mlp(features)
        return pred, torch.zeros(0, pred.shape[-1], device=pred.device, dtype=pred.dtype)

