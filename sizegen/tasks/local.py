"""Bounded-local (F1) control targets: L(N) = O(1).

Theorem 20's control prediction: on F1 all tier gaps vanish, and the
resolution profile's far field C(N) stays bounded (in fact the truncation
error hits exactly zero at k = L).
"""

from typing import Optional

import numpy as np

from ..graphs.generators import Graph
from .base import TaskResult


def degree_target(graph: Graph, rng: Optional[np.random.Generator] = None) -> TaskResult:
    """f*(v) = deg(v): determined by the 1-hop neighborhood (L = 1)."""
    rng = rng or np.random.default_rng()
    deg = graph.degrees().astype(np.float64)
    x = rng.uniform(0.0, 1.0, size=graph.num_nodes)  # decoy feature
    # m_0 = best radius-0 guess (zero); m_k for k >= 1 = exact.
    iterates = [np.zeros_like(deg)] + [deg.copy(), deg.copy()]
    return TaskResult(
        target=deg,
        inputs=np.stack([x, np.ones_like(x)], axis=1),
        iterates=iterates,
        rho=None,
        meta={"task": "degree", "L": 1},
    )


def one_hop_mean(graph: Graph, rng: Optional[np.random.Generator] = None) -> TaskResult:
    """f*(v) = mean of a random node field over N(v): also L = 1."""
    rng = rng or np.random.default_rng()
    n = graph.num_nodes
    x = rng.uniform(0.0, 1.0, size=n)
    adj = graph.adjacency()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    tgt = np.zeros(n)
    nz = deg > 0
    tgt[nz] = (adj @ x)[nz] / deg[nz]
    iterates = [x.copy(), tgt.copy(), tgt.copy()]
    return TaskResult(
        target=tgt,
        inputs=np.stack([x, deg / max(deg.max(), 1.0)], axis=1),
        iterates=iterates,
        rho=None,
        meta={"task": "one_hop_mean", "L": 1},
    )
