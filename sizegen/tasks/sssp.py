"""SSSP hop distance — the *outside-F^op* boundary case (paper §7.2).

Bellman-Ford in the min-plus semiring is non-expansive but NOT contractive:
d_k(v) is exact once k >= hops(v) and carries no information before that.
The truncation error therefore does not decay geometrically — it decays only
as the population of not-yet-reached nodes shrinks — and the required depth
is the diameter, not log N.  This is the honest failure-mode probe (E7).
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..graphs.generators import Graph
from .base import TaskResult


def sssp_hops(
    graph: Graph,
    source: Optional[int] = None,
    k_max: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> TaskResult:
    """Hop distances from one source, with k-step Bellman-Ford truncations.

    Unreachable-at-k entries are clipped to ``clip = diameter-bound`` so the
    truncation MSE is finite; only finally-reachable nodes enter the metric
    (unreachable nodes are excluded from target and iterates alike).
    """
    rng = rng or np.random.default_rng()
    n = graph.num_nodes
    if source is None:
        source = int(rng.integers(0, n))
    adj = graph.adjacency()

    dist = sp.csgraph.shortest_path(
        adj, method="D", unweighted=True, directed=False, indices=source
    )
    reachable = np.isfinite(dist)
    clip = float(dist[reachable].max()) + 1.0

    # k-step Bellman-Ford truncation: d_k(v) = hops(v) if hops(v) <= k else clip.
    d_star = np.where(reachable, dist, clip)
    iterates = []
    for k in range(k_max + 1):
        d_k = np.where(reachable & (dist <= k), dist, clip)
        iterates.append(d_k[reachable])

    onehot = np.zeros(n)
    onehot[source] = 1.0
    deg = graph.degrees()
    return TaskResult(
        target=d_star[reachable],
        inputs=np.stack([onehot, deg / max(deg.max(), 1.0)], axis=1)[reachable],
        iterates=iterates,
        rho=None,  # non-contractive: no geometric rate
        meta={
            "task": "sssp_hops",
            "source": source,
            "num_reachable": int(reachable.sum()),
            "clip": clip,
        },
    )
