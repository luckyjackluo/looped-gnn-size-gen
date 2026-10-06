"""Training and risk evaluation for all four schemes.

One function trains any model; the scheme is expressed through
(a) which parameters are trainable ('all' | 'controller' | 'controller+decoder' — the paper's
    FT-vs-FS freeze sub-choice, cf. UnifiedLearning finetune_utils),
(b) the depth sampler K used during that stage,
(c) which dataset stage it runs on (D_train vs D_adapt).
"""

import copy
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


def _nodes_per_graph(batch_vec: torch.Tensor, num_graphs: int) -> torch.Tensor:
    return torch.bincount(batch_vec, minlength=num_graphs)


def _set_trainable(model: torch.nn.Module, mode: str) -> int:
    """'all' | 'controller' | 'controller+decoder'; returns trainable count."""
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif mode in ("controller", "controller+decoder"):
        for p in model.parameters():
            p.requires_grad = False
        if getattr(model, "controller", None) is None:
            raise ValueError(f"mode={mode!r} requires a controller")
        for p in model.controller.parameters():
            p.requires_grad = True
        if mode == "controller+decoder":
            for p in model.decoder.parameters():
                p.requires_grad = True
    else:
        raise ValueError(mode)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _looped_loss(
    outs: List[torch.Tensor],
    y: torch.Tensor,
    beta_conv: float,
    eta_mono: float,
    deltas: Optional[List[torch.Tensor]] = None,
    lambda_stab: float = 0.0,
    gamma_stab: float = 0.97,
    y_traj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deep supervision + the paper's (A11) stabilization regularizers.

    - position term: increasing weights w_k = (k+1)/K (UnifiedLearning lesson:
      final iterate most supervised, early iterates get gradient without
      dominating) — this is L_comp, driving every iteration's worth of the
      trajectory toward the target;
    - convergence regularizer beta_conv * ||out_k - out_{k-1}||^2 and
      monotonicity penalty eta_mono * relu(mse_k - mse_{k-1}) — together the
      operational L_stab, pushing the learned operator to be non-expansive
      along the trajectory (L_R L_C -> 1), which is what licenses zero-shot
      depth extension at deployment.
    """
    K = len(outs)
    if y_traj is not None:
        # Trajectory supervision: iterate k (after k+1 operator applications)
        # is matched to the task's radius-(k+1) truncation m_{k+1}, so the
        # learned block tracks U_N itself (A6) and deeper unrolling at
        # deployment is genuinely "more steps of U" rather than a collapse
        # onto the pretraining-scale target.  Uniform weights; beyond the
        # recorded horizon m_k = f*, so the final iterate still fits y.
        ws = [1.0] * K
        mses = [F.mse_loss(o, y_traj[:, min(k + 1, y_traj.shape[1] - 1)].unsqueeze(-1))
                for k, o in enumerate(outs)]
    else:
        ws = [(k + 1) / K for k in range(K)]
        mses = [F.mse_loss(o, y) for o in outs]
    wsum = sum(ws)
    loss = sum(w * m for w, m in zip(ws, mses)) / wsum
    if K > 1:
        if beta_conv > 0:
            loss = loss + beta_conv * sum(
                F.mse_loss(outs[k], outs[k - 1].detach()) for k in range(1, K)
            ) / (K - 1)
        if eta_mono > 0:
            loss = loss + eta_mono * sum(
                torch.clamp(mses[k] - mses[k - 1].detach(), min=0.0)
                for k in range(1, K)
            )
        if lambda_stab > 0 and deltas is not None and len(deltas) > 1:
            # Operator-level CONTRACTION (A11, strengthened): penalize any
            # step whose hidden update fails to shrink by factor gamma_stab
            # relative to the previous step's.  Non-expansiveness (gamma = 1)
            # is insufficient — ratio ~ 1 permits slow drift that compounds
            # over deployment unrolls far beyond the supervised horizon
            # (observed: tier1 exploded at K ~ 50 from trained K <= 12).
            # Constraining the local Lipschitz behavior toward a fixed point
            # is what extrapolates in depth.
            ratios = [
                deltas[k] / (deltas[k - 1].detach() + 1e-12)
                for k in range(1, K)
            ]
            loss = loss + lambda_stab * sum(
                torch.clamp(r - gamma_stab, min=0.0) ** 2 for r in ratios
            ) / (K - 1)
    return loss


def train_model(
    model: torch.nn.Module,
    dataset: List,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    k_sampler: Optional[Callable[[np.random.Generator], int]] = None,
    trainable: str = "all",
    device: str = "cuda",
    val_dataset: Optional[List] = None,
    val_K: int = 8,
    seed: int = 0,
    log_every: int = 20,
    verbose: bool = True,
    beta_conv: float = 0.05,
    eta_mono: float = 0.1,
    lambda_stab: float = 0.5,
    traj_sup: bool = False,
) -> Dict:
    """Generic trainer. k_sampler draws the unroll depth per step
    (None for fixed-depth models, which ignore K and skip the
    trajectory regularizers)."""
    model = model.to(device)
    n_trainable = _set_trainable(model, trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_val = math.inf
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            K = k_sampler(rng) if k_sampler is not None else None
            npg = _nodes_per_graph(batch.batch, batch.num_graphs)
            if K is not None:
                outs, deltas = model(batch.x, batch.edge_index, batch.batch, npg,
                                     K=K, return_all=True, return_deltas=True)
                y_traj = getattr(batch, "y_traj", None) if traj_sup else None
                if traj_sup and y_traj is None:
                    raise ValueError("traj_sup=True but dataset has no y_traj")
                loss = _looped_loss(outs, batch.y, beta_conv, eta_mono,
                                    deltas=deltas, lambda_stab=lambda_stab,
                                    y_traj=y_traj)
            else:
                out = model(batch.x, batch.edge_index, batch.batch, npg, K=None)
                loss = F.mse_loss(out, batch.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss) * batch.num_graphs
            cnt += batch.num_graphs
        sched.step()
        train_loss = tot / max(cnt, 1)

        val_loss = None
        if val_dataset is not None:
            val_loss = evaluate_risk(model, val_dataset, K=val_K, device=device)["mse"]
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = f"  epoch {epoch:3d} train {train_loss:.5f}"
            if val_loss is not None:
                msg += f" val {val_loss:.5f}"
            print(msg, flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val": best_val if best_val < math.inf else None,
        "n_trainable": n_trainable,
    }


@torch.no_grad()
def evaluate_risk(
    model: torch.nn.Module,
    dataset: List,
    K: Optional[int],
    device: str = "cuda",
    batch_size: int = 8,
) -> Dict:
    """Deployment risk at depth K: per-node MSE plus per-graph sum norm."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    se_sum, node_cnt, graph_sums = 0.0, 0, []
    for batch in loader:
        batch = batch.to(device)
        npg = _nodes_per_graph(batch.batch, batch.num_graphs)
        out = model(batch.x, batch.edge_index, batch.batch, npg, K=K)
        se = (out - batch.y).pow(2).sum(dim=-1)  # (N_total,)
        se_sum += float(se.sum())
        node_cnt += se.numel()
        for g in range(batch.num_graphs):
            graph_sums.append(float(se[batch.batch == g].sum()))
    return {
        "mse": se_sum / max(node_cnt, 1),
        "graph_sum_norm_mean": float(np.mean(graph_sums)),
        "K": K,
        "num_graphs": len(graph_sums),
    }


def make_k_sampler(k_min: int, k_max: int) -> Callable:
    def sampler(rng: np.random.Generator) -> int:
        return int(rng.integers(k_min, k_max + 1))

    return sampler
