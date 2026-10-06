"""Seeded graph generators for the size-generalization experiments.

Each generator instantiates one row of the paper's Table 1 (graph distribution
P_N), with the average degree held Theta(1) as N grows so the F2 boundary is
set purely by the diameter scaling:

- rgg        : locality-preserving in R^d, bounded degree  -> diam ~ N^{1/d}
- sparse_er  : Erdos-Renyi G(N, c/N)                       -> diam ~ log N
- pref_attach: preferential attachment (scale-free)        -> diam ~ log log N

Graphs are undirected; ``edge_index`` stores both directions.
"""

from dataclasses import dataclass, field
from math import gamma, pi
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


@dataclass
class Graph:
    """Lightweight graph container (numpy-only, torch-free for Phase 0)."""

    num_nodes: int
    edge_index: np.ndarray  # (2, E) int64, both directions of each edge
    pos: Optional[np.ndarray] = None  # (N, d) for geometric graphs
    meta: Dict = field(default_factory=dict)

    def adjacency(self) -> sp.csr_matrix:
        """Symmetric 0/1 adjacency as CSR."""
        src, dst = self.edge_index
        data = np.ones(src.shape[0], dtype=np.float64)
        adj = sp.coo_matrix(
            (data, (src, dst)), shape=(self.num_nodes, self.num_nodes)
        ).tocsr()
        # Collapse any duplicate entries back to 0/1.
        adj.data[:] = 1.0
        return adj

    def degrees(self) -> np.ndarray:
        return np.asarray(self.adjacency().sum(axis=1)).ravel()


def _to_both_directions(pairs: np.ndarray) -> np.ndarray:
    """(M, 2) unique undirected pairs -> (2, 2M) directed edge_index."""
    if pairs.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return np.stack([src, dst]).astype(np.int64)


def _largest_component_relabel(graph: Graph) -> Graph:
    """Restrict to the largest connected component, relabelling nodes 0..n-1."""
    n_comp, labels = sp.csgraph.connected_components(graph.adjacency(), directed=False)
    if n_comp <= 1:
        return graph
    largest = np.bincount(labels).argmax()
    keep = labels == largest
    new_id = -np.ones(graph.num_nodes, dtype=np.int64)
    new_id[keep] = np.arange(keep.sum())
    src, dst = graph.edge_index
    mask = keep[src] & keep[dst]
    edge_index = np.stack([new_id[src[mask]], new_id[dst[mask]]])
    pos = graph.pos[keep] if graph.pos is not None else None
    meta = dict(graph.meta)
    meta["restricted_to_largest_component"] = True
    meta["original_num_nodes"] = graph.num_nodes
    return Graph(int(keep.sum()), edge_index, pos=pos, meta=meta)


def rgg(
    n: int,
    d: int = 2,
    avg_degree: float = 8.0,
    rng: Optional[np.random.Generator] = None,
    largest_component: bool = False,
) -> Graph:
    """Random geometric graph at unit density in a [0, n^{1/d}]^d box.

    Connection radius r solves ``avg_degree = V_d * r^d`` (V_d the unit-ball
    volume), so the expected degree is constant in n and the diameter scales
    as n^{1/d} — the paper's default locality-preserving carrier (A1/A5).
    """
    rng = rng or np.random.default_rng()
    side = n ** (1.0 / d)
    pos = rng.uniform(0.0, side, size=(n, d))
    v_d = pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)
    radius = (avg_degree / v_d) ** (1.0 / d)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")  # (M, 2), i < j
    g = Graph(
        n,
        _to_both_directions(pairs),
        pos=pos,
        meta={"family": "rgg", "d": d, "avg_degree": avg_degree, "radius": radius},
    )
    return _largest_component_relabel(g) if largest_component else g


def sparse_er(
    n: int,
    avg_degree: float = 8.0,
    rng: Optional[np.random.Generator] = None,
    largest_component: bool = False,
) -> Graph:
    """Sparse Erdos-Renyi G(n, p) with p = avg_degree / n.

    Sampled as G(n, m) with m ~ Binomial(C(n,2), p) and m unique random pairs,
    which avoids materializing all O(n^2) coin flips.
    """
    rng = rng or np.random.default_rng()
    p = min(avg_degree / n, 1.0)
    n_pairs = n * (n - 1) // 2
    m = rng.binomial(n_pairs, p)
    # Rejection-sample unique unordered pairs (m << n_pairs in the sparse regime).
    seen = set()
    pairs = []
    while len(pairs) < m:
        batch = rng.integers(0, n, size=(2 * (m - len(pairs)) + 16, 2))
        for a, b in batch:
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) == m:
                break
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    g = Graph(
        n,
        _to_both_directions(pairs),
        meta={"family": "sparse_er", "avg_degree": avg_degree},
    )
    return _largest_component_relabel(g) if largest_component else g


def pref_attach(
    n: int,
    m_attach: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Graph:
    """Barabasi-Albert preferential attachment (repeated-nodes construction).

    Each new node attaches to ``m_attach`` distinct existing nodes with
    probability proportional to degree. Always connected by construction.
    """
    rng = rng or np.random.default_rng()
    if n <= m_attach:
        raise ValueError("n must exceed m_attach")
    # repeated-nodes list: node i appears once per unit of degree
    repeated = []
    pairs = []
    # seed: star on m_attach + 1 nodes
    for i in range(m_attach):
        pairs.append((i, m_attach))
        repeated.extend([i, m_attach])
    for v in range(m_attach + 1, n):
        targets = set()
        while len(targets) < m_attach:
            targets.add(repeated[rng.integers(0, len(repeated))])
        for t in targets:
            pairs.append((t, v))
            repeated.extend([t, v])
    pairs = np.asarray(pairs, dtype=np.int64)
    return Graph(
        n,
        _to_both_directions(pairs),
        meta={"family": "pref_attach", "m_attach": m_attach},
    )
