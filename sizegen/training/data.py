"""Bridge from the numpy task layer to PyG Data objects."""

import re
import zlib
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from ..graphs import rgg, sparse_er
from ..tasks import (degree_target, iterop, labelprop, one_hop_mean, pagerank,
                     sssp_hops)


def _pagerank_fn(alpha: float, seed_mode: str = "iid") -> Callable:
    def fn(graph, rng):
        return pagerank(graph, alpha=alpha, k_max=0, rng=rng, seed_mode=seed_mode)

    return fn


def _labelprop_fn(alpha: float, seed_mode: str = "iid") -> Callable:
    def fn(graph, rng):
        return labelprop(graph, alpha=alpha, k_max=0, rng=rng,
                         seed_mode=seed_mode)

    return fn


def _iterop_fn(horizon: str, alpha: float, k0: int = 5, n_ref: int = 200) -> Callable:
    def fn(graph, rng):
        return iterop(graph, horizon=horizon, alpha=alpha, k0=k0, n_ref=n_ref,
                      k_max=0, rng=rng, operator_traj_len=TRAJ_PAD)

    return fn


_ITEROP_RE = re.compile(
    r"^iterop_(?P<horizon>const|log|poly|diam)_a(?P<alpha>[0-9.]+)"
    r"(?:_k(?P<k0>\d+))?(?:_n(?P<nref>\d+))?$"
)


class _TaskRegistry(dict):
    """Static task table plus parametric names of the form
    ``iterop_{const|log|poly|diam}_a{alpha}[_k{K0}][_n{N_ref}]``
    (constructed-horizon operator targets, paper §sec:synthetic)."""

    def __missing__(self, key: str) -> Callable:
        m = _ITEROP_RE.match(key)
        if m is None:
            raise KeyError(key)
        fn = _iterop_fn(
            m["horizon"], float(m["alpha"]),
            k0=int(m["k0"]) if m["k0"] else 5,
            n_ref=int(m["nref"]) if m["nref"] else 200,
        )
        self[key] = fn
        return fn

    def __contains__(self, key) -> bool:
        return dict.__contains__(self, key) or (
            isinstance(key, str) and _ITEROP_RE.match(key) is not None)


TASKS: Dict[str, Callable] = _TaskRegistry({
    "pagerank": _pagerank_fn(0.15),
    "pagerank_a0.05": _pagerank_fn(0.05),
    "pagerank_a0.3": _pagerank_fn(0.3),
    "pagerank_a0.5": _pagerank_fn(0.5),
    "pagerank_smooth_a0.05": _pagerank_fn(0.05, seed_mode="smooth"),
    "pagerank_smooth_a0.15": _pagerank_fn(0.15, seed_mode="smooth"),
    "pagerank_smooth_a0.3": _pagerank_fn(0.3, seed_mode="smooth"),
    "pagerank_smooth_a0.5": _pagerank_fn(0.5, seed_mode="smooth"),
    "labelprop": _labelprop_fn(0.85),
    "labelprop_smooth_a0.9": _labelprop_fn(0.9, seed_mode="smooth"),
    "degree": lambda g, rng: degree_target(g, rng=rng),
    "one_hop_mean": lambda g, rng: one_hop_mean(g, rng=rng),
    "sssp": lambda g, rng: _sssp_connected(g, rng),
})


def _sssp_connected(graph, rng):
    """SSSP on a connected graph (all GRAPHS entries use largest_component,
    so the reachability filter in sssp_hops is a no-op and node counts match)."""
    res = sssp_hops(graph, k_max=0, rng=rng)
    assert res.target.shape[0] == graph.num_nodes, (
        "sssp task requires a connected graph (use largest_component=True)")
    return res

GRAPHS: Dict[str, Callable] = {
    "rgg_d1": lambda n, rng: rgg(n, d=1, avg_degree=8.0, rng=rng, largest_component=True),
    "rgg_d2": lambda n, rng: rgg(n, d=2, avg_degree=8.0, rng=rng, largest_component=True),
    "rgg_d3": lambda n, rng: rgg(n, d=3, avg_degree=8.0, rng=rng, largest_component=True),
    "sparse_er": lambda n, rng: sparse_er(n, avg_degree=8.0, rng=rng, largest_component=True),
    # structurally biased carriers (A14 experiments): same geometry, sparser
    "rgg_d2_deg4": lambda n, rng: rgg(n, d=2, avg_degree=4.0, rng=rng, largest_component=True),
    "rgg_d2_deg5": lambda n, rng: rgg(n, d=2, avg_degree=5.0, rng=rng, largest_component=True),
}


TRAJ_PAD = 96  # width of the stored truncation trajectory m_0..m_{TRAJ_PAD-1}


def make_data(graph, task_result) -> Data:
    y = task_result.target
    if y.ndim == 1:
        y = y[:, None]
    data = Data(
        x=torch.from_numpy(task_result.inputs).float(),
        edge_index=torch.from_numpy(graph.edge_index).long(),
        y=torch.from_numpy(y).float(),
        num_nodes=graph.num_nodes,
    )
    # Trajectory supervision target (paper (A1)/(A6): the learned operator
    # should track U_N step by step, so iterate k is supervised toward the
    # radius-k truncation m_k, not toward f*).  Beyond the recorded horizon
    # the truncation equals the target, so we pad by repeating the last
    # iterate.  Only scalar-target iterative tasks provide this.
    its = task_result.meta.get("operator_traj") or task_result.iterates
    if its and its[0].ndim == 1 and y.shape[1] == 1:
        traj = np.stack(its[:TRAJ_PAD], axis=1)  # (N, T+1)
        if traj.shape[1] < TRAJ_PAD:
            last = np.repeat(traj[:, -1:], TRAJ_PAD - traj.shape[1], axis=1)
            traj = np.concatenate([traj, last], axis=1)
        data.y_traj = torch.from_numpy(traj).float()
    return data


def make_dataset(
    family: str,
    task: str,
    sizes: Sequence[int],
    graphs_per_size: int,
    seed: int = 0,
) -> List[Data]:
    """Seeded dataset: `graphs_per_size` graphs at each size in `sizes`."""
    graph_fn = GRAPHS[family]
    task_fn = TASKS[task]
    out = []
    for n in sizes:
        for g_idx in range(graphs_per_size):
            # stable across processes (builtin hash() is salted for strings)
            key = f"{family}|{task}|{n}|{g_idx}|{seed}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            g = graph_fn(n, rng)
            res = task_fn(g, rng)
            out.append(make_data(g, res))
    return out
