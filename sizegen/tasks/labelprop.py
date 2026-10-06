"""Damped iterative label propagation (Zhou et al., 2003) — second F2 task.

Update (paper §7.2):   h^{k+1} = alpha * S h^k + (1 - alpha) * y0
with S = D^{-1/2} A D^{-1/2} the symmetric normalized adjacency
(||S||_2 <= 1) and y0 random O(1) node labels.  Contractive with rate
rho = alpha; fixed point f* = (1 - alpha)(I - alpha S)^{-1} y0.

Same theory as PageRank but a *different operator family* (symmetric vs
row-stochastic propagation) — the replication task.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..graphs.generators import Graph
from .base import TaskResult


def _sym_normalized_adjacency(graph: Graph) -> sp.csr_matrix:
    adj = graph.adjacency()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    d_half = sp.diags(inv_sqrt)
    return d_half @ adj @ d_half


def labelprop(
    graph: Graph,
    alpha: float = 0.85,
    k_max: int = 64,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-13,
    seed_mode: str = "iid",
) -> TaskResult:
    """Solve damped label propagation; record m_0..m_{k_max} and exact f*.

    seed_mode='smooth' draws y0 from a long-wavelength field over positions
    (geometric graphs only) — same (A3) rationale as the pagerank variant."""
    rng = rng or np.random.default_rng()
    n = graph.num_nodes
    s_mat = _sym_normalized_adjacency(graph)
    if seed_mode == "smooth":
        from .pagerank import _smooth_seed

        if graph.pos is None:
            raise ValueError("seed_mode='smooth' needs a geometric graph (pos)")
        y0 = 2.0 * _smooth_seed(graph.pos, rng) - 1.0  # range ~ [-1, 1]
    else:
        y0 = rng.choice([-1.0, 1.0], size=n) * rng.uniform(0.5, 1.5, size=n)

    h = y0.copy()
    iterates = [h.copy()]
    for _ in range(k_max):
        h = alpha * (s_mat @ h) + (1.0 - alpha) * y0
        iterates.append(h.copy())

    f_star = h.copy()
    for _ in range(20_000):
        f_next = alpha * (s_mat @ f_star) + (1.0 - alpha) * y0
        if np.max(np.abs(f_next - f_star)) < tol:
            f_star = f_next
            break
        f_star = f_next

    deg = graph.degrees()
    inputs = np.stack([y0, deg / max(deg.max(), 1.0)], axis=1)
    return TaskResult(
        target=f_star,
        inputs=inputs,
        iterates=iterates,
        rho=alpha,
        meta={"task": "labelprop", "alpha": alpha, "seed_mode": seed_mode},
    )
