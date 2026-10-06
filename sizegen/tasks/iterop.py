"""Constructed-horizon operator targets (paper (A1) operator form, §sec:synthetic).

    f*_N = D* ∘ U_N^{T*(N)} ∘ E*(x_N),      U_N(h) = (1 - alpha) Ã_N h + alpha b(x_N)

with Ã_N the symmetric normalized adjacency, fixed linear E*, D*, b, and the
iteration count T*(N) SET BY THE EXPERIMENTER.  The dependency radius is then
exactly T*(N) by construction, so a task can be placed at any rung of the
class hierarchy (Def. 1):

    horizon   T*(N)                       class
    const     K0                          F1  (bounded local)
    log       K0 * log N / log N_ref      F2^log
    poly      K0 * (N / N_ref)^{1/(2d)}   F2 \\ F2^log
    diam      K0 * (N / N_ref)^{1/d}      F3  (~ diam of an RGG in R^d)

Every family is normalized to T*(N_ref) = K0, so pretraining at N ~ N_ref
sees the SAME horizon for all four and the families only diverge OOD.

The operator is contractive with rate rho = 1 - alpha (||Ã|| <= 1), so the
radius-k truncation error obeys eps^2(k) ~ rho^{2k} C(N) for k < T*(N)
(Lemma P1) and is exactly zero for k >= T*(N).

The seed field b(x) is long-wavelength over node positions (A3: far-field
non-degeneracy), as in ``pagerank(seed_mode='smooth')``.
"""

import math
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..graphs.generators import Graph
from .base import TaskResult
from .pagerank import _smooth_seed

HORIZONS = ("const", "log", "poly", "diam")

# Fixed (size-independent) linear templates E*, b, D*.  Inputs are
# x = [s (smooth field), u (iid noise), deg/deg_max].
_W_ENC = np.array([1.0, 0.5, 0.0])   # E*(x) = s + 0.5 u
_W_BIAS = np.array([1.0, 0.0, 0.0])  # b(x)  = s
_D_SCALE, _D_SHIFT = 1.5, -0.25       # D*(h) = 1.5 h - 0.25


def horizon_T(horizon: str, n: int, d: int, k0: int = 5, n_ref: int = 200) -> int:
    """T*(N) for the requested family; T*(n_ref) = k0 for every family."""
    if horizon == "const":
        g = 1.0
    elif horizon == "log":
        g = math.log(n) / math.log(n_ref)
    elif horizon == "poly":
        g = (n / n_ref) ** (1.0 / (2.0 * d))
    elif horizon == "diam":
        g = (n / n_ref) ** (1.0 / d)
    else:
        raise ValueError(f"unknown horizon {horizon!r}; choose from {HORIZONS}")
    return max(1, int(round(k0 * g)))


def _sym_normalized_adjacency(graph: Graph) -> sp.csr_matrix:
    adj = graph.adjacency().tolil()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    for v in np.where(deg == 0)[0]:  # isolated nodes keep their own state
        adj[v, v] = 1.0
    adj = adj.tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(deg))
    return dinv @ adj @ dinv


def iterop(
    graph: Graph,
    horizon: str = "log",
    alpha: float = 0.3,
    k0: int = 5,
    n_ref: int = 200,
    k_max: int = 0,
    rng: Optional[np.random.Generator] = None,
    d: Optional[int] = None,
    operator_traj_len: int = 0,
) -> TaskResult:
    """Constructed-horizon target on a geometric graph.

    k_max: record iterates up to max(T*, k_max); iterates beyond T* equal
    the target (the radius-k truncation is exact once k >= T*).
    operator_traj_len: if > 0, also store meta['operator_traj'] = the
    CONTINUED operator trajectory D*(U_N^k E*) for k = 0..operator_traj_len-1
    (not truncated at T*) — the supervision signal for operator-aligned
    pretraining (A6): the learned block is asked to track U_N itself."""
    rng = rng or np.random.default_rng()
    if d is None:
        d = graph.meta.get("d")
        if d is None:
            if graph.pos is None:
                raise ValueError("iterop needs a geometric graph (pos) or explicit d")
            d = graph.pos.shape[1]
    if graph.pos is None:
        raise ValueError("iterop needs node positions for the smooth seed field (A3)")
    n = graph.num_nodes
    t_star = horizon_T(horizon, n, d, k0=k0, n_ref=n_ref)

    s = _smooth_seed(graph.pos, rng)
    u = rng.uniform(0.0, 1.0, size=n)
    deg = graph.degrees()
    x = np.stack([s, u, deg / max(deg.max(), 1.0)], axis=1)

    a_sym = _sym_normalized_adjacency(graph)
    h = x @ _W_ENC
    b = x @ _W_BIAS
    dec = lambda z: _D_SCALE * z + _D_SHIFT  # noqa: E731
    iterates = [dec(h)]
    for _ in range(t_star):
        h = (1.0 - alpha) * (a_sym @ h) + alpha * b
        iterates.append(dec(h))
    target = iterates[-1].copy()
    for _ in range(t_star, k_max):
        iterates.append(target.copy())
    op_traj = None
    if operator_traj_len > 0:
        op_traj = list(iterates[:min(t_star + 1, operator_traj_len)])
        for _ in range(len(op_traj), operator_traj_len):
            h = (1.0 - alpha) * (a_sym @ h) + alpha * b
            op_traj.append(dec(h))

    return TaskResult(
        target=target,
        inputs=x,
        iterates=iterates,
        rho=1.0 - alpha,
        meta={"task": "iterop", "horizon": horizon, "T_star": t_star,
              "alpha": alpha, "k0": k0, "n_ref": n_ref, "d": int(d),
              **({"operator_traj": op_traj} if op_traj is not None else {})},
    )
