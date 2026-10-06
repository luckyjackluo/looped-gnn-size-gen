"""Prediction heads for algorithmic graph tasks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class _MLPStack(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeRegressionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_source_conditioning: bool = False,
    ):
        super().__init__()
        input_dim = hidden_dim * 2 if use_source_conditioning else hidden_dim
        self.use_source_conditioning = use_source_conditioning
        self.mlp = _MLPStack(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)

    def forward(
        self,
        node_emb: torch.Tensor,
        *,
        batch: Optional[torch.Tensor] = None,
        source_global_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_source_conditioning:
            if batch is None or source_global_idx is None:
                raise ValueError("batch and source_global_idx are required for source-conditioned heads")
            source_emb = node_emb[source_global_idx]
            node_emb = torch.cat([node_emb, source_emb[batch]], dim=-1)
        return self.mlp(node_emb)


class NodeClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_source_conditioning: bool = False,
    ):
        super().__init__()
        input_dim = hidden_dim * 2 if use_source_conditioning else hidden_dim
        self.use_source_conditioning = use_source_conditioning
        self.mlp = _MLPStack(input_dim, hidden_dim, num_classes, num_layers=num_layers, dropout=dropout)

    def forward(
        self,
        node_emb: torch.Tensor,
        *,
        batch: Optional[torch.Tensor] = None,
        source_global_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_source_conditioning:
            if batch is None or source_global_idx is None:
                raise ValueError("batch and source_global_idx are required for source-conditioned heads")
            source_emb = node_emb[source_global_idx]
            node_emb = torch.cat([node_emb, source_emb[batch]], dim=-1)
        return self.mlp(node_emb)


class GraphRegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, *, output_dim: int = 1, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mlp = _MLPStack(hidden_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, node_emb: torch.Tensor, *, batch: torch.Tensor) -> torch.Tensor:
        graph_emb = global_mean_pool(node_emb, batch)
        return self.mlp(graph_emb)


class GraphClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, *, num_classes: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mlp = _MLPStack(hidden_dim, hidden_dim, num_classes, num_layers=num_layers, dropout=dropout)

    def forward(self, node_emb: torch.Tensor, *, batch: torch.Tensor) -> torch.Tensor:
        graph_emb = global_mean_pool(node_emb, batch)
        return self.mlp(graph_emb)


def build_prediction_head(config: dict, hidden_dim: int) -> nn.Module:
    """Build a prediction head from config."""
    supervision_cfg = config.get("supervision", {})
    head_cfg = config.get("head", {})
    level = supervision_cfg.get("level", "node")
    task_type = supervision_cfg.get("type", "regression")
    output_dim = int(head_cfg.get("output_dim", supervision_cfg.get("output_dim", 1)))
    num_layers = int(head_cfg.get("num_layers", 2))
    dropout = float(head_cfg.get("dropout", 0.1))
    use_source_conditioning = bool(head_cfg.get("use_source_conditioning", False))

    if level == "node" and task_type == "regression":
        return NodeRegressionHead(
            hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_source_conditioning=use_source_conditioning,
        )
    if level == "node" and task_type == "classification":
        return NodeClassificationHead(
            hidden_dim,
            num_classes=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_source_conditioning=use_source_conditioning,
        )
    if level == "graph" and task_type == "regression":
        return GraphRegressionHead(
            hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if level == "graph" and task_type == "classification":
        return GraphClassificationHead(
            hidden_dim,
            num_classes=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported supervision config: level={level!r}, type={task_type!r}")
