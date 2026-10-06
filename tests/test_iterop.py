import numpy as np
import pytest

from sizegen.graphs import rgg
from sizegen.tasks import HORIZONS, horizon_T, iterop
from sizegen.training.data import TASKS


def test_horizon_normalized_at_nref():
    for h in HORIZONS:
        assert horizon_T(h, 200, d=2, k0=5, n_ref=200) == 5
    # ordering OOD: const < log < poly < diam
    ts = [horizon_T(h, 20000, d=2) for h in HORIZONS]
    assert ts == sorted(ts) and ts[0] == 5 and ts[-1] == 50
    assert horizon_T("poly", 20000, d=3) < horizon_T("poly", 20000, d=2)


def test_iterates_are_exact_truncations():
    g = rgg(1500, d=2, rng=np.random.default_rng(0), largest_component=True)
    res = iterop(g, horizon="log", alpha=0.3, k_max=16, rng=np.random.default_rng(1))
    T = res.meta["T_star"]
    assert 5 < T <= 16 and len(res.iterates) == 17
    eps = res.truncation_mse(reduction="sum")
    assert eps[T] == 0.0 and np.all(eps[T:] == 0.0)
    assert np.all(np.diff(eps[:T]) < 0)  # strictly decreasing before the horizon
    assert res.depth_to_tolerance(0.0) == T


def test_geometric_decay_within_envelope():
    g = rgg(4000, d=2, rng=np.random.default_rng(2), largest_component=True)
    res = iterop(g, horizon="diam", alpha=0.2, rng=np.random.default_rng(3))
    eps = res.truncation_mse(reduction="sum")
    T = res.meta["T_star"]
    k = np.arange(3, T - 1)
    slope = np.polyfit(k, np.log(eps[k]), 1)[0]
    assert slope <= 2 * np.log(res.rho) + 0.05  # at least as steep as rho^{2k}


def test_registry_parses_parametric_names():
    fn = TASKS["iterop_poly_a0.2"]
    g = rgg(300, d=2, rng=np.random.default_rng(4), largest_component=True)
    res = fn(g, np.random.default_rng(5))
    assert res.meta["horizon"] == "poly" and res.meta["alpha"] == 0.2
    assert "iterop_diam_a0.5_k4_n100" in TASKS
    assert TASKS["iterop_diam_a0.5_k4_n100"](g, np.random.default_rng(6)).meta["k0"] == 4
    with pytest.raises(KeyError):
        TASKS["iterop_cubic_a0.2"]
