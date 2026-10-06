import numpy as np
import pytest

from sizegen.graphs import rgg, sparse_er, pref_attach


def test_rgg_degree_and_locality():
    rng = np.random.default_rng(0)
    g = rgg(2000, d=2, avg_degree=8.0, rng=rng)
    deg = g.degrees()
    # Boundary effects shave the mean slightly below the target.
    assert 5.0 < deg.mean() < 9.0
    # Every edge respects the connection radius.
    src, dst = g.edge_index
    dists = np.linalg.norm(g.pos[src] - g.pos[dst], axis=1)
    assert dists.max() <= g.meta["radius"] + 1e-9


def test_rgg_diameter_scales_polynomially():
    # diam(RGG in R^2) ~ sqrt(N): quadrupling N should ~double the diameter.
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    diams = []
    for n in (500, 2000):
        g = rgg(n, d=2, avg_degree=10.0, rng=rng, largest_component=True)
        d = sp.csgraph.shortest_path(
            g.adjacency(), unweighted=True, directed=False, indices=0
        )
        diams.append(np.max(d[np.isfinite(d)]))
    ratio = diams[1] / diams[0]
    assert 1.4 < ratio < 3.5  # ~2 expected


def test_sparse_er_edge_count():
    rng = np.random.default_rng(2)
    g = sparse_er(5000, avg_degree=8.0, rng=rng)
    deg = g.degrees()
    assert abs(deg.mean() - 8.0) < 0.5
    src, dst = g.edge_index
    assert not np.any(src == dst)


def test_symmetry():
    rng = np.random.default_rng(3)
    for g in (
        rgg(500, rng=rng),
        sparse_er(500, rng=rng),
        pref_attach(500, rng=rng),
    ):
        adj = g.adjacency()
        assert (adj != adj.T).nnz == 0


def test_pref_attach_connected():
    import scipy.sparse as sp

    rng = np.random.default_rng(4)
    g = pref_attach(1000, m_attach=4, rng=rng)
    n_comp, _ = sp.csgraph.connected_components(g.adjacency(), directed=False)
    assert n_comp == 1
    assert g.degrees().min() >= 4


def test_seeded_reproducibility():
    g1 = rgg(300, rng=np.random.default_rng(42))
    g2 = rgg(300, rng=np.random.default_rng(42))
    assert np.array_equal(g1.edge_index, g2.edge_index)
