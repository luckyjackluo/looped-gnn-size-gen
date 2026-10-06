"""Personalized PageRank with damping — the primary F2 / F^op task.

Update (paper §7.2):   h^{k+1} = (1 - alpha) * P h^k + alpha * s
with P = D^{-1} A the row-normalized adjacency (dangling nodes get a self
loop) and s a node-wise seed field drawn O(1) per node, so target magnitudes
are size-invariant.  The operator is contractive in the sup norm with rate
exactly rho = 1 - alpha:  alpha is the experiment's contraction dial.

L(N) is the least k with eps(k) <= tau; on an RGG in R^d the theory predicts
K* ~ log N / (d |log rho|).
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..graphs.generators import Graph
from .base import TaskResult


def _row_normalized_adjacency(graph: Graph) -> sp.csr_matrix:
    adj = graph.adjacency().tolil()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = np.where(deg == 0)[0]
    for v in dangling:  # self-loop so P stays row-stochastic
        adj[v, v] = 1.0
    adj = adj.tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_deg = sp.diags(1.0 / deg)
    return inv_deg @ adj


def _smooth_seed(pos: np.ndarray, rng: np.random.Generator, n_modes: int = 4) -> np.ndarray:
    """Long-wavelength random field over node positions (wavelength ~ box side).

    Instantiates the paper's far-field non-degeneracy (A3): with iid seeds the
    aggregate far-field contribution CONCENTRATES (LLN), so the radius-k Bayes
    error collapses far below the truncation bound and fixed-radius predictors
    face no real ceiling (observed empirically in the first E1 smoke).  A field
    with correlation length ~ diameter carries per-graph far-field information
    that no k-ball can infer — making the depth requirement genuine.
    """
    side = float(np.max(pos.max(axis=0) - pos.min(axis=0)))
    vals = np.zeros(pos.shape[0])
    for _ in range(n_modes):
        wavelength = side * rng.uniform(0.5, 1.5)
        direction = rng.normal(size=pos.shape[1])
        direction /= np.linalg.norm(direction)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        vals += amp * np.cos(2.0 * np.pi / wavelength * (pos @ direction) + phase)
    ptp = float(vals.max() - vals.min())
    return (vals - vals.min()) / max(ptp, 1e-9)


def pagerank(
    graph: Graph,
    alpha: float = 0.15,
    k_max: int = 64,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-13,
    seed_mode: str = "iid",
) -> TaskResult:
    """Solve damped PageRank; record iterates m_0..m_{k_max} and exact f*.

    seed_mode: 'iid' (concentrating far field) or 'smooth' (long-wavelength
    field over positions; requires a geometric graph — enforces (A3))."""
    rng = rng or np.random.default_rng()
    n = graph.num_nodes
    p_mat = _row_normalized_adjacency(graph)
    if seed_mode == "smooth":
        if graph.pos is None:
            raise ValueError("seed_mode='smooth' needs a geometric graph (pos)")
        s = _smooth_seed(graph.pos, rng)
    else:
        s = rng.uniform(0.0, 1.0, size=n)

    # Radius-k truncations from h^0 = s.
    h = s.copy()
    iterates = [h.copy()]
    for _ in range(k_max):
        h = (1.0 - alpha) * (p_mat @ h) + alpha * s
        iterates.append(h.copy())

    # Exact fixed point: continue to machine precision.
    f_star = h.copy()
    for _ in range(10_000):
        f_next = (1.0 - alpha) * (p_mat @ f_star) + alpha * s
        if np.max(np.abs(f_next - f_star)) < tol:
            f_star = f_next
            break
        f_star = f_next

    deg = graph.degrees()
    inputs = np.stack([s, deg / max(deg.max(), 1.0)], axis=1)
    return TaskResult(
        target=f_star,
        inputs=inputs,
        iterates=iterates,
        rho=1.0 - alpha,
        meta={"task": "pagerank", "alpha": alpha, "seed_mode": seed_mode},
    )
