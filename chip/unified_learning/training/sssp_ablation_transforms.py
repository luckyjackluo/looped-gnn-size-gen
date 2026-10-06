"""Lightweight graph ablations for SSSP diagnostics."""

from __future__ import annotations

import random
from typing import Iterable, List

import torch
from torch_geometric.data import Data


SSSP_ABLATIONS = frozenset(
    {
        "none",
        "original",
        "no_edges",
        "no_coordinates",
        "shuffle_coordinates",
        "zero_edge_attr",
    }
)


def _clone_graphs(graphs: Iterable[Data]) -> List[Data]:
    return [g.clone() for g in graphs]


def apply_sssp_ablation(
    graphs: Iterable[Data],
    *,
    ablation: str | None,
    seed: int = 0,
) -> List[Data]:
    """Return cloned graphs with a diagnostic ablation applied.

    These transforms intentionally keep ``y_node`` fixed: they test how the
    trained model reacts when one input channel is removed or corrupted.
    """
    name = (ablation or "none").lower()
    if name not in SSSP_ABLATIONS:
        valid = ", ".join(sorted(SSSP_ABLATIONS))
        raise ValueError(f"Unknown SSSP ablation '{ablation}'. Valid: {valid}")
    if name in {"none", "original"}:
        return _clone_graphs(graphs)

    out = _clone_graphs(graphs)
    rng = random.Random(seed)

    for idx, data in enumerate(out):
        if name == "no_edges":
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_dim = int(data.edge_attr.shape[-1]) if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.dim() > 1 else 1
            data.edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
        elif name == "zero_edge_attr":
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                data.edge_attr = torch.zeros_like(data.edge_attr)
        elif name == "no_coordinates":
            if hasattr(data, "pos") and data.pos is not None:
                data.pos = torch.zeros_like(data.pos)
            if hasattr(data, "x") and data.x is not None and data.x.shape[-1] >= 2:
                data.x = data.x.clone()
                data.x[:, :2] = 0.0
        elif name == "shuffle_coordinates":
            if not hasattr(data, "pos") or data.pos is None:
                continue
            gen = torch.Generator()
            gen.manual_seed(seed + idx + rng.randrange(10_000_000))
            perm = torch.randperm(data.num_nodes, generator=gen)
            data.pos = data.pos[perm]
            if hasattr(data, "x") and data.x is not None and data.x.shape[-1] >= 2:
                data.x = data.x.clone()
                data.x[:, :2] = data.x[perm, :2]
    return out

