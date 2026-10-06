import networkx as nx
import numpy as np
import pytest

from sizegen.graphs import rgg, sparse_er
from sizegen.tasks import degree_target, labelprop, one_hop_mean, pagerank, sssp_hops


def _to_nx(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes))
    G.add_edges_from(zip(*g.edge_index))
    return G


def test_pagerank_fixed_point_equation():
    rng = np.random.default_rng(0)
    g = sparse_er(400, rng=rng, largest_component=True)
    res = pagerank(g, alpha=0.15, rng=np.random.default_rng(1))
    # f* satisfies h = (1-a) P h + a s to machine precision
    from sizegen.tasks.pagerank import _row_normalized_adjacency

    p = _row_normalized_adjacency(g)
    s = res.inputs[:, 0]
    lhs = res.target
    rhs = 0.85 * (p @ res.target) + 0.15 * s
    assert np.max(np.abs(lhs - rhs)) < 1e-10


def test_pagerank_geometric_decay_rate():
    """Truncation error decays geometrically, within the analytic envelope.

    The contraction bound eps(k) <= rho^k eps(0) (rho = 1 - alpha) upper-bounds
    the error, so the fitted log-slope must be at least as steep as 2 log rho;
    subdominant spectral modes make finite-k decay somewhat steeper, converging
    to the envelope from below (verified empirically: slope -0.47 at k~5-30 ->
    -0.37 at k~40-60 vs analytic -0.325).
    """
    rng = np.random.default_rng(2)
    g = rgg(3000, d=2, avg_degree=8.0, rng=rng, largest_component=True)
    res = pagerank(g, alpha=0.15, k_max=60, rng=np.random.default_rng(3))
    eps2 = res.truncation_mse()
    ks = np.arange(30, 55)
    logs = np.log(eps2[30:55])
    slope, intercept = np.polyfit(ks, logs, 1)
    envelope = 2.0 * np.log(res.rho)  # -0.325 for alpha = 0.15
    # (a) geometric: log-linear fit is tight
    resid = logs - (slope * ks + intercept)
    assert np.max(np.abs(resid)) < 0.5
    # (b) respects the contraction envelope (may be steeper, never shallower)
    assert slope <= envelope + 0.02
    # (c) within 2x of the analytic rate (not a different mechanism)
    assert slope >= 2.0 * envelope


def test_labelprop_fixed_point_via_direct_solve():
    import scipy.sparse.linalg as spla
    import scipy.sparse as sp
    from sizegen.tasks.labelprop import _sym_normalized_adjacency

    rng = np.random.default_rng(4)
    g = sparse_er(300, rng=rng, largest_component=True)
    res = labelprop(g, alpha=0.85, rng=np.random.default_rng(5))
    s_mat = _sym_normalized_adjacency(g)
    y0 = res.inputs[:, 0]
    direct = spla.spsolve(
        (sp.eye(g.num_nodes) - 0.85 * s_mat).tocsc(), 0.15 * y0
    )
    assert np.max(np.abs(direct - res.target)) < 1e-8


def test_degree_and_one_hop_mean_against_nx():
    rng = np.random.default_rng(6)
    g = sparse_er(200, rng=rng)
    G = _to_nx(g)
    res = degree_target(g, rng=np.random.default_rng(7))
    nx_deg = np.array([G.degree(v) for v in range(g.num_nodes)], dtype=float)
    assert np.array_equal(res.target, nx_deg)

    res2 = one_hop_mean(g, rng=np.random.default_rng(8))
    x = res2.inputs[:, 0]
    for v in list(G.nodes)[:20]:
        nbrs = list(G.neighbors(v))
        if nbrs:
            assert abs(res2.target[v] - x[nbrs].mean()) < 1e-12


def test_sssp_against_nx():
    rng = np.random.default_rng(9)
    g = sparse_er(300, rng=rng, largest_component=True)
    res = sssp_hops(g, source=0, k_max=32, rng=np.random.default_rng(10))
    G = _to_nx(g)
    nx_dist = nx.single_source_shortest_path_length(G, 0)
    expected = np.array([nx_dist[v] for v in sorted(nx_dist)], dtype=float)
    assert np.array_equal(res.target, expected)
    # k-truncation: exact within k hops
    k = 3
    within = expected <= k
    assert np.array_equal(res.iterates[k][within], expected[within])


def test_iterates_are_radius_k_measurable():
    """m_k must not change when the graph is edited outside the k-hop ball."""
    rng = np.random.default_rng(11)
    g = rgg(1500, d=2, avg_degree=8.0, rng=rng, largest_component=True)
    res = pagerank(g, alpha=0.15, k_max=5, rng=np.random.default_rng(12))

    # Pick a probe node and find everything within 5 hops.
    import scipy.sparse as sp

    probe = 0
    d = sp.csgraph.shortest_path(
        g.adjacency(), unweighted=True, directed=False, indices=probe
    )
    far = np.where(d > 10)[0]
    if len(far) < 10:
        pytest.skip("graph too small for a far region")
    # Delete all edges strictly among far nodes.
    far_set = set(far.tolist())
    src, dst = g.edge_index
    keep = ~(np.isin(src, far) & np.isin(dst, far))
    g2 = type(g)(g.num_nodes, np.stack([src[keep], dst[keep]]), pos=g.pos)
    res2 = pagerank(g2, alpha=0.15, k_max=5, rng=np.random.default_rng(12))
    # Iterate at probe unchanged for all k <= 5 (locality of the truncation).
    for k in range(6):
        assert abs(res.iterates[k][probe] - res2.iterates[k][probe]) < 1e-12
