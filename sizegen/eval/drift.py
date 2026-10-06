"""E5 probes — operator drift and size-transfer preservation (paper A14).

The draft flags (A14) with "Maybe not true for all cases": single-size
fine-tuning is *assumed* to leave the deployment-size operator alignment
eps_B(N_OOD) unconstrained (>= the calibrated pretrained one).  These probes
measure that operationally, for a frozen vs fine-tuned operator:

1. trajectory_contraction: the learned operator's per-step relative residual
   ||h^{k+1} - h^k|| / ||h^k|| across depth, per graph size N.  A size-stable
   contractive operator shows geometrically decaying residuals with an
   N-independent rate; drift shows up as slower/non-decaying residuals at
   sizes away from N_adapt.

2. risk_vs_depth: deployment risk as a function of unroll depth K at a given
   N.  A drifted operator degrades (risk turns back up) beyond the depth it
   was fine-tuned at, while the frozen operator keeps improving toward its
   fixed point.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ..training.train import _nodes_per_graph, evaluate_risk


@torch.no_grad()
def trajectory_contraction(
    model: torch.nn.Module,
    dataset: List,
    K: int,
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict:
    """Per-step relative residuals of the hidden trajectory, averaged over
    the dataset. Returns {'residuals': [K floats], 'rate': fitted decay}."""
    model = model.to(device).eval()
    # Tier 0 (FixedDepthGNN): no shared operator — probe its k_fix distinct
    # blocks in sequence instead (the trajectory the fixed net actually runs).
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        K = len(blocks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sums = np.zeros(K)
    count = 0
    for batch in loader:
        batch = batch.to(device)
        npg = _nodes_per_graph(batch.batch, batch.num_graphs)
        h = model.encoder(batch.x)
        prev = h
        for k in range(K):
            h_in = prev
            if getattr(model, "controller", None) is not None:
                h_in = model.controller(h_in, batch.batch, npg, k)
            h = (blocks[k](h_in, batch.edge_index) if blocks is not None
                 else model.processor(h_in, batch.edge_index))
            num = (h - prev).norm()
            den = prev.norm().clamp(min=1e-12)
            sums[k] += float(num / den)
            prev = h
        count += 1
    residuals = (sums / max(count, 1)).tolist()
    # fitted geometric rate over the late half (residual_k ~ rate^k);
    # short trajectories (Tier 0's K_fix blocks) fit over all steps
    lo = K // 2 if K - K // 2 >= 3 else 0
    late = np.array(residuals[lo:])
    ks = np.arange(lo, K)
    rate = (
        float(np.exp(np.polyfit(ks, np.log(np.maximum(late, 1e-12)), 1)[0]))
        if len(late) >= 3
        else None
    )
    return {"residuals": residuals, "rate": rate}


def risk_vs_depth(
    model: torch.nn.Module,
    dataset: List,
    k_grid: List[int],
    device: str = "cuda",
) -> List[float]:
    return [evaluate_risk(model, dataset, K=k, device=device)["mse"] for k in k_grid]


@torch.no_grad()
def operator_alignment(
    model: torch.nn.Module,
    dataset: List,
    K: int,
    alpha: float,
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict:
    """True eps_B probe (paper A6): functional operator alignment.

    At each trajectory step compare the model's one-step update, read out
    through the decoder, against the TASK operator applied to the current
    readout:  a_k = mean_v | D(R(h_k))_v  -  U_N(D(h_k))_v |^2  with
    U_N(v) = (1-alpha) P v + alpha s  (damped PageRank; s = input seed).

    A size-stable operator keeps a_k flat across N; drift shows as a_k
    inflation at sizes away from the adaptation size.  This is the direct
    (A14) measurement the contraction-rate proxy approximates.
    """
    import torch_geometric.utils as pyg_utils

    model = model.to(device).eval()
    blocks = getattr(model, "blocks", None)  # Tier 0: per-layer alignment
    if blocks is not None:
        K = len(blocks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sums = np.zeros(K)
    count = 0
    for batch in loader:
        batch = batch.to(device)
        npg = _nodes_per_graph(batch.batch, batch.num_graphs)
        # row-normalized adjacency as sparse tensor (self-loop for dangling
        # handled implicitly: degrees > 0 on largest-component graphs)
        ei = batch.edge_index
        deg = torch.bincount(ei[0], minlength=batch.num_nodes).clamp(min=1).float()
        w = 1.0 / deg[ei[0]]
        s_seed = batch.x[:, 0:1]
        h = model.encoder(batch.x)
        for k in range(K):
            v_now = model.decoder(h)
            h_in = h
            if getattr(model, "controller", None) is not None:
                h_in = model.controller(h_in, batch.batch, npg, k)
            h_next = (blocks[k](h_in, batch.edge_index) if blocks is not None
                      else model.processor(h_in, batch.edge_index))
            if getattr(model, "anchored", False):
                a = model.anchor_mix()
                h_next = a * model.encoder(batch.x) + (1.0 - a) * h_next
            v_model = model.decoder(h_next)
            # task step applied to the model's current readout
            pv = torch.zeros_like(v_now)
            pv.index_add_(0, ei[0], w.unsqueeze(-1) * v_now[ei[1]])
            v_task = (1.0 - alpha) * pv + alpha * s_seed
            sums[k] += float(F.mse_loss(v_model, v_task))
            h = h_next
        count += 1
    return {"alignment_mse": (sums / max(count, 1)).tolist()}


@torch.no_grad()
def operator_alignment_iterop(
    model: torch.nn.Module,
    dataset: List,
    K: int,
    alpha: float,
    device: str = "cuda",
    batch_size: int = 4,
    d_scale: float = 1.5,
    d_shift: float = -0.25,
) -> Dict:
    """eps_B probe for constructed-horizon (iterop) targets.

    Task operator U(h) = (1-alpha) A_sym h + alpha s with SYMMETRIC
    normalization and fixed affine decoder D(h) = d_scale*h + d_shift.
    At each step compare the model's decoded one-step update against the
    task step applied to the model's current readout (mapped through D):
      h_now = (v_now - d_shift)/d_scale;  v_task = D((1-a) A_sym h_now + a s).
    """
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sums = np.zeros(K)
    count = 0
    for batch in loader:
        batch = batch.to(device)
        npg = _nodes_per_graph(batch.batch, batch.num_graphs)
        ei = batch.edge_index
        deg = torch.bincount(ei[0], minlength=batch.num_nodes).clamp(min=1).float()
        w = (deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt())
        s_seed = batch.x[:, 0:1]
        h = model.encoder(batch.x)
        for k in range(K):
            v_now = model.decoder(h)
            h_in = h
            if getattr(model, "controller", None) is not None:
                h_in = model.controller(h_in, batch.batch, npg, k)
            h_next = model.processor(h_in, batch.edge_index)
            if getattr(model, "anchored", False):
                a = model.anchor_mix()
                h_next = a * model.encoder(batch.x) + (1.0 - a) * h_next
            v_model = model.decoder(h_next)
            h_task = (v_now - d_shift) / d_scale
            av = torch.zeros_like(h_task)
            av.index_add_(0, ei[0], w.unsqueeze(-1) * h_task[ei[1]])
            v_task = d_scale * ((1.0 - alpha) * av + alpha * s_seed) + d_shift
            sums[k] += float(F.mse_loss(v_model, v_task))
            h = h_next
        count += 1
    return {"alignment_mse": (sums / max(count, 1)).tolist()}
