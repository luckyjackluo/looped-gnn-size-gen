"""Loss for the gated-halt looped placement model (Variants A + B).

Companion to :class:`unified_learning.models.placement.GatedHaltLoopedPlacementModel`.

Variant A (``model.loop_pondernet_enabled == False``):
    Same shape as the existing ``LoopedPlacementLoss`` minus the BCE
    convergence term (the per-node gate replaces it as the halt signal):

        L = L_pos(lambda_k = k/K) + alpha * L_edge + eta * L_mono
            + beta_reg * L_smooth + L_gate

    where ``L_gate = l0_weight * E[P(gate != 0)] + entropy_weight * E[H(gate)]``,
    both with linear warmup, and the L0 expectation uses the Hard Concrete
    closed form (fallback to sigmoid(logit) for sigmoid / gumbel_st gates).

Variant B (``model.loop_pondernet_enabled == True``):
    Per-iter losses are weighted by the LEARNED halt distribution ``p_k``
    (read from ``model._last_ponder_aux``):

        L_pos     = sum_k p_k_node * MSE(x_k, x_gt)
        L_edge    = alpha * sum_k p_k_edge * MSE(r_k, r_gt)
        L_mono    = eta * sum_k max(0, MSE_k - MSE_{k-1})    (unweighted)
        L_smooth  = beta_reg * sum_k MSE(x_k, detach(x_{k-1}))
        L_dist    = KL(p_k || truncated geometric)          (pondernet)
                    OR -H(p_k)                              (softmax_over_k)
        L         = L_pos + L_edge + L_mono + L_smooth
                    + reg_weight(t) * L_dist + L_gate(t)

    ``p_k_node`` is ``p_k[graph(n)]`` per node; ``p_k_edge`` is
    ``p_k[graph(src(e))]`` per edge (src and dst share a graph for our
    graphs, so either endpoint works).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Generic helpers                                                                #
# --------------------------------------------------------------------------- #

def _masked_node_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    node_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE averaged over valid nodes with optional per-node loss weights.

    node_weight [N]: if provided, computes weighted average over nodes so that
    graphs whose weight > 1 receive proportionally more loss gradient.
    Weights should already be broadcast from per-graph to per-node by the
    caller.  Pass ``None`` for unweighted behaviour (backward compatible).
    """
    sq = (pred - target) ** 2  # [N, D]
    per_node = sq.mean(dim=-1)  # [N] - mean over output dims per node
    if mask is not None:
        per_node = per_node[mask]
        if node_weight is not None:
            node_weight = node_weight[mask]
    if node_weight is not None:
        w_sum = node_weight.sum().clamp_min(1e-12)
        return (per_node * node_weight).sum() / w_sum
    return per_node.mean() if per_node.numel() > 0 else pred.new_zeros(())


def _truncated_geometric_prior(
    K: int, prior_lambda: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Truncated-and-renormalised geometric prior over k = 0..K-1."""
    ks = torch.arange(K, device=device, dtype=dtype)
    log_one_minus = torch.log(
        torch.tensor(1.0 - prior_lambda, device=device, dtype=dtype)
    )
    log_lambda = torch.log(
        torch.tensor(prior_lambda, device=device, dtype=dtype)
    )
    p = torch.exp(ks * log_one_minus + log_lambda)
    return p / p.sum().clamp_min(1e-12)


def _gate_reg_terms(model) -> torch.Tensor:
    """L0 (Hard Concrete) and entropy regulariser, both with linear warmup."""
    aux = getattr(model, "_last_gate_aux", None)
    if aux is None:
        return torch.zeros(())
    l0_w = float(getattr(model, "loop_gate_l0_weight", 0.0))
    ent_w = float(getattr(model, "loop_gate_entropy_weight", 0.0))
    device = aux["gate_logit_per_iter"].device
    if l0_w <= 0.0 and ent_w <= 0.0:
        return torch.zeros((), device=device, dtype=torch.float32)
    step_counter = getattr(model, "_ponder_step", None)
    step = int(step_counter.item()) if step_counter is not None else 0
    out = torch.zeros((), device=device, dtype=torch.float32)
    if l0_w > 0.0:
        warm = int(getattr(model, "loop_gate_l0_warmup_steps", 0))
        frac = min(1.0, step / max(warm, 1)) if warm > 0 else 1.0
        logits = aux["gate_logit_per_iter"].float()
        gate_mod = getattr(model, "gate", None)
        if gate_mod is not None and hasattr(gate_mod, "hard_concrete_l0_probability"):
            prob_nonzero = gate_mod.hard_concrete_l0_probability(logits)
        else:
            prob_nonzero = torch.sigmoid(logits)
        out = out + (l0_w * frac) * prob_nonzero.mean()
    if ent_w > 0.0:
        warm = int(getattr(model, "loop_gate_entropy_warmup_steps", 0))
        frac = min(1.0, step / max(warm, 1)) if warm > 0 else 1.0
        gate = aux["gate_per_iter"].float().clamp(1e-6, 1.0 - 1e-6)
        entropy = -(gate * gate.log() + (1.0 - gate) * (1.0 - gate).log())
        out = out + (ent_w * frac) * entropy.mean()
    return out


# --------------------------------------------------------------------------- #
# Graph-size loss weighting                                                      #
# --------------------------------------------------------------------------- #

def compute_graph_size_node_weight(
    node_batch: torch.Tensor,
    mode: str,
) -> Optional[torch.Tensor]:
    """Return a per-node weight tensor based on the size of each node's graph.

    Args:
        node_batch: [N] long tensor mapping each node to its graph index (the
            ``batch`` attribute of a PyG ``Data`` / ``Batch`` object).
        mode: 'none' | 'sqrt' | 'linear' | 'log'.  'none' returns ``None``
            (no weighting, backward-compatible fast path).

    Returns:
        [N] float32 tensor with weights normalised so their mean = 1.0, or
        ``None`` if mode == 'none'.

    The normalisation ensures the overall loss scale stays the same as the
    unweighted case on average; only the *relative* weight between large and
    small graphs changes.
    """
    if mode == "none":
        return None
    # Count nodes per graph.
    G = int(node_batch.max().item()) + 1
    graph_sizes = torch.zeros(G, dtype=torch.float32, device=node_batch.device)
    graph_sizes.scatter_add_(
        0,
        node_batch,
        torch.ones(node_batch.shape[0], dtype=torch.float32, device=node_batch.device),
    )  # [G]

    if mode == "sqrt":
        graph_w = graph_sizes.sqrt()
    elif mode == "linear":
        graph_w = graph_sizes
    elif mode == "log":
        graph_w = (graph_sizes + 1.0).log()
    else:
        raise ValueError(
            f"graph_size_weight_mode must be 'none'/'sqrt'/'linear'/'log'; got {mode!r}"
        )

    # Normalise so mean graph weight = 1.
    graph_w = graph_w / graph_w.mean().clamp_min(1e-12)

    # Broadcast to per-node.
    return graph_w[node_batch]  # [N]


# --------------------------------------------------------------------------- #
# Variant A (no halt distribution): per-iter loss with linear ramp.              #
# --------------------------------------------------------------------------- #

def _variant_a_loss(
    node_preds: List[torch.Tensor],
    edge_preds: List[torch.Tensor],
    x_gt_rh: torch.Tensor,
    r_gt_rh: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    alpha_edge: float,
    eta_mono: float,
    beta_reg: float,
    K_total: int,
    step_offset: int,
    node_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    K = len(node_preds)
    lambda_weights = torch.arange(
        step_offset + 1, step_offset + K + 1,
        dtype=x_gt_rh.dtype, device=x_gt_rh.device,
    ) / K_total

    # Position loss: sum_k lambda_k * masked_MSE(x_k, x_gt).
    L_pos = node_preds[0].new_zeros(())
    for x_k, lam in zip(node_preds, lambda_weights):
        L_pos = L_pos + lam * _masked_node_mse(x_k, x_gt_rh, mask, node_weight)

    # Edge auxiliary loss: alpha * mean_k MSE(r_k, r_gt).
    if edge_preds:
        L_edge = alpha_edge * sum(
            F.mse_loss(r_k, r_gt_rh) for r_k in edge_preds
        ) / len(edge_preds)
    else:
        L_edge = node_preds[0].new_zeros(())

    # Monotonicity penalty (unweighted — compares iterations, not ground truth).
    if len(node_preds) >= 2:
        L_mono = node_preds[0].new_zeros(())
        for k in range(1, len(node_preds)):
            mse_k = _masked_node_mse(node_preds[k], x_gt_rh, mask)
            mse_prev = _masked_node_mse(node_preds[k - 1], x_gt_rh, mask).detach()
            L_mono = L_mono + torch.clamp(mse_k - mse_prev, min=0.0)
        L_mono = eta_mono * L_mono
    else:
        L_mono = node_preds[0].new_zeros(())

    # Step smoothness regulariser: MSE(x_k, detach(x_{k-1})).
    if len(node_preds) >= 2:
        L_smooth = beta_reg * sum(
            F.mse_loss(node_preds[k], node_preds[k - 1].detach())
            for k in range(1, len(node_preds))
        ) / max(len(node_preds) - 1, 1)
    else:
        L_smooth = node_preds[0].new_zeros(())

    total = L_pos + L_edge + L_mono + L_smooth
    parts = {
        "pos": float(L_pos.detach().item()),
        "edge": float(L_edge.detach().item()),
        "mono": float(L_mono.detach().item()),
        "reg": float(L_smooth.detach().item()),
    }
    return total, parts


# --------------------------------------------------------------------------- #
# Variant B (halt distribution): per-iter loss weighted by learned p_k.          #
# --------------------------------------------------------------------------- #

def _variant_b_loss(
    node_preds: List[torch.Tensor],
    edge_preds: List[torch.Tensor],
    halt_probs: torch.Tensor,                # [K, G] fp32
    node_batch: torch.Tensor,                # [N]
    edge_index: torch.Tensor,                # [2, E]
    x_gt_rh: torch.Tensor,
    r_gt_rh: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    alpha_edge: float,
    eta_mono: float,
    beta_reg: float,
    node_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    K = halt_probs.shape[0]
    device = halt_probs.device
    src = edge_index[0]
    edge_batch = node_batch[src]

    p_k_node = halt_probs[:, node_batch]              # [K, N]
    p_k_edge = halt_probs[:, edge_batch]              # [K, E]

    # Position loss: sum_k mean_n (mask * p_k_node * w_n * (x_k - x_gt)^2).
    sq_node_total = torch.zeros((), device=device, dtype=torch.float32)
    if mask is not None:
        mask_f = mask.to(torch.float32)
        if node_weight is not None:
            # Weighted denominator = sum of weights over valid nodes.
            w_masked = node_weight * mask_f
            denom_nodes = w_masked.sum().clamp_min(1e-12)
            for k in range(K):
                sq = ((node_preds[k] - x_gt_rh) ** 2).mean(dim=-1)  # [N]
                sq_node_total = sq_node_total + (p_k_node[k] * sq * w_masked).sum()
        else:
            denom_nodes = mask_f.sum().clamp_min(1.0)
            for k in range(K):
                sq = ((node_preds[k] - x_gt_rh) ** 2).sum(dim=-1)  # [N]
                sq_node_total = sq_node_total + (p_k_node[k] * sq * mask_f).sum()
        L_pos = sq_node_total / denom_nodes
    else:
        if node_weight is not None:
            denom_nodes = node_weight.sum().clamp_min(1e-12)
            for k in range(K):
                sq = ((node_preds[k] - x_gt_rh) ** 2).mean(dim=-1)  # [N]
                sq_node_total = sq_node_total + (p_k_node[k] * sq * node_weight).sum()
            L_pos = sq_node_total / denom_nodes
        else:
            for k in range(K):
                sq = ((node_preds[k] - x_gt_rh) ** 2).sum(dim=-1)  # [N]
                sq_node_total = sq_node_total + (p_k_node[k] * sq).sum()
            L_pos = sq_node_total / float(x_gt_rh.shape[0] * x_gt_rh.shape[1])
            L_pos = L_pos * x_gt_rh.shape[1]  # restore per-coord mean shape

    # Edge auxiliary loss.
    if edge_preds:
        sq_edge_total = torch.zeros((), device=device, dtype=torch.float32)
        for k in range(K):
            sq = ((edge_preds[k] - r_gt_rh) ** 2).sum(dim=-1)  # [E]
            sq_edge_total = sq_edge_total + (p_k_edge[k] * sq).sum()
        L_edge = alpha_edge * sq_edge_total / float(r_gt_rh.shape[0] * r_gt_rh.shape[1])
        L_edge = L_edge * r_gt_rh.shape[1]
    else:
        L_edge = halt_probs.new_zeros(())

    # Monotonicity penalty (unweighted - "predictions should improve").
    if len(node_preds) >= 2:
        L_mono = halt_probs.new_zeros(())
        for k in range(1, len(node_preds)):
            mse_k = _masked_node_mse(node_preds[k], x_gt_rh, mask)
            mse_prev = _masked_node_mse(node_preds[k - 1], x_gt_rh, mask).detach()
            L_mono = L_mono + torch.clamp(mse_k - mse_prev, min=0.0)
        L_mono = eta_mono * L_mono
    else:
        L_mono = halt_probs.new_zeros(())

    # Step smoothness regulariser.
    if len(node_preds) >= 2:
        L_smooth = beta_reg * sum(
            F.mse_loss(node_preds[k], node_preds[k - 1].detach())
            for k in range(1, len(node_preds))
        ) / max(len(node_preds) - 1, 1)
    else:
        L_smooth = halt_probs.new_zeros(())

    total = L_pos + L_edge + L_mono + L_smooth
    parts = {
        "pos": float(L_pos.detach().item()),
        "edge": float(L_edge.detach().item()),
        "mono": float(L_mono.detach().item()),
        "reg": float(L_smooth.detach().item()),
    }
    return total, parts


# --------------------------------------------------------------------------- #
# Public loss class                                                              #
# --------------------------------------------------------------------------- #

class GatedHaltPlacementLoss:
    """Aggregates all gated/halt placement loss terms.

    Config keys (all optional, with reasonable defaults):
        alpha_edge: float = 0.1
        eta_mono:   float = 0.05
        beta_reg:   float = 0.01
        graph_size_weight_mode: str = 'none'
            Per-graph loss weighting based on graph size N.  The caller must
            pass ``node_weight`` to ``__call__``; this field is stored so
            the training loop can read it via ``loss_fn.graph_size_weight_mode``.
            Supported values: 'none' | 'sqrt' | 'linear' | 'log'.

    Halting reg weight and prior_lambda come from the model itself
    (``model.loop_halt_reg_weight`` etc.) so the loss stays consistent
    with the model's config.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.alpha_edge = float(cfg.get("alpha_edge", 0.1))
        self.eta_mono = float(cfg.get("eta_mono", 0.05))
        self.beta_reg = float(cfg.get("beta_reg", 0.01))
        self.graph_size_weight_mode: str = str(
            cfg.get("graph_size_weight_mode", "none")
        )
        if self.graph_size_weight_mode not in ("none", "sqrt", "linear", "log"):
            raise ValueError(
                f"loss.graph_size_weight_mode must be one of "
                f"'none'/'sqrt'/'linear'/'log'; got {self.graph_size_weight_mode!r}"
            )

    def __call__(
        self,
        model,
        node_preds: List[torch.Tensor],
        edge_preds: List[torch.Tensor],
        x_gt_rh: torch.Tensor,
        r_gt_rh: torch.Tensor,
        mask: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        *,
        K_total: Optional[int] = None,
        step_offset: int = 0,
        node_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        K = len(node_preds)
        K_denom = K_total if K_total is not None else K

        if not getattr(model, "loop_pondernet_enabled", False):
            base, parts = _variant_a_loss(
                node_preds,
                edge_preds,
                x_gt_rh,
                r_gt_rh,
                mask,
                alpha_edge=self.alpha_edge,
                eta_mono=self.eta_mono,
                beta_reg=self.beta_reg,
                K_total=K_denom,
                step_offset=step_offset,
                node_weight=node_weight,
            )
            gate_reg = _gate_reg_terms(model).to(base.device).to(base.dtype)
            total = base + gate_reg
            parts["gate"] = float(gate_reg.detach().item())
            parts["dist"] = 0.0
            parts["loss"] = float(total.detach().item())
            return total, parts

        aux = getattr(model, "_last_ponder_aux", None)
        if aux is None:
            raise RuntimeError(
                "model.loop_pondernet_enabled=True but model._last_ponder_aux "
                "is None; forward pass did not populate it."
            )
        halt_probs = aux["halt_probs"].float()
        node_batch = aux["node_batch"]
        distribution = aux.get(
            "distribution",
            getattr(model, "loop_halt_distribution", "pondernet_geometric"),
        )

        base, parts = _variant_b_loss(
            node_preds,
            edge_preds,
            halt_probs,
            node_batch,
            edge_index,
            x_gt_rh,
            r_gt_rh,
            mask,
            alpha_edge=self.alpha_edge,
            eta_mono=self.eta_mono,
            beta_reg=self.beta_reg,
            node_weight=node_weight,
        )

        # Distribution regulariser.
        eps = 1e-7
        p_clamped = halt_probs.clamp_min(eps)
        if distribution == "pondernet_geometric":
            prior = _truncated_geometric_prior(
                K=halt_probs.shape[0],
                prior_lambda=float(getattr(model, "loop_halt_prior_lambda", 0.4)),
                device=halt_probs.device,
                dtype=halt_probs.dtype,
            )
            log_ratio = p_clamped.log() - prior.clamp_min(eps).log().unsqueeze(1)
            L_dist = (halt_probs * log_ratio).sum(dim=0).mean()
        elif distribution == "softmax_over_k":
            entropy = -(p_clamped * p_clamped.log()).sum(dim=0).mean()
            L_dist = -entropy  # negative entropy -> encourages peakedness
        else:
            raise ValueError(
                f"Unknown halt distribution {distribution!r}; valid: "
                "pondernet_geometric, softmax_over_k"
            )
        reg_weight_target = float(getattr(model, "loop_halt_reg_weight", 0.0))
        warmup_steps = int(getattr(model, "loop_halt_reg_warmup_steps", 0))
        step_counter = getattr(model, "_ponder_step", None)
        step = int(step_counter.item()) if step_counter is not None else 0
        warm_frac = min(1.0, step / max(warmup_steps, 1)) if warmup_steps > 0 else 1.0
        reg_weight = reg_weight_target * warm_frac

        gate_reg = _gate_reg_terms(model).to(base.device).to(base.dtype)
        total = base + reg_weight * L_dist.to(base.dtype) + gate_reg

        parts["dist"] = float(L_dist.detach().item())
        parts["gate"] = float(gate_reg.detach().item())
        parts["loss"] = float(total.detach().item())
        return total, parts
