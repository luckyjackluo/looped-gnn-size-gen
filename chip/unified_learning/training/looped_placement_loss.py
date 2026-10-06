"""
Loss functions for looped placement regression (from looped_placement_impl.md §3).

All losses operate in row-height units (dimensionless after dividing by h_row).

Usage:
    from unified_learning.training.looped_placement_loss import LoopedPlacementLoss

    loss_fn = LoopedPlacementLoss(config)
    loss, loss_dict = loss_fn(node_preds, edge_preds, conv_scores, x_gt_rh, r_gt_rh)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Individual loss terms                                                         #
# --------------------------------------------------------------------------- #

def position_loss(
    predictions: List[torch.Tensor],
    x_gt_rh: torch.Tensor,
    lambda_weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted MSE on absolute positions across all K iterations.

    lambda_k = k/K (linear ramp) — early iterations supervised weakly,
    final iteration most heavily supervised.

    Args:
        predictions:    List[Tensor[N,2]] — positions in row-height units
        x_gt_rh:        [N,2] ground-truth positions in row-height units
        lambda_weights: [K] increasing weights (e.g. arange(1,K+1)/K)
        mask:           [N] bool — True for nodes to include (ports/macro
                        exclusion). If None all nodes are used.
    """
    loss = predictions[0].new_zeros(())
    for x_k, lam in zip(predictions, lambda_weights):
        if mask is not None:
            sq = ((x_k - x_gt_rh) ** 2)[mask]
            mse = sq.mean() if sq.numel() > 0 else x_k.new_zeros(())
        else:
            mse = F.mse_loss(x_k, x_gt_rh)
        loss = loss + lam * mse
    return loss


def edge_aux_loss(
    edge_preds: List[torch.Tensor],
    r_gt_rh: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Auxiliary MSE on pairwise displacements r_ij = (x_i - x_j) / h_row.

    The O(1) magnitude of r_gt_rh gives a clean gradient signal at all
    iterations regardless of circuit size N.  Loss is averaged over
    iterations (equal weight — all iterations should predict displacements).

    alpha: weighting relative to position loss (recommend 0.05–0.2).
    """
    if not edge_preds:
        return torch.zeros(())
    total = sum(F.mse_loss(r_k, r_gt_rh) for r_k in edge_preds)
    return alpha * total / len(edge_preds)


def convergence_loss(
    conv_scores: List[torch.Tensor],
    predictions: List[torch.Tensor],
    x_gt_rh: torch.Tensor,
    gamma: float = 2.0,
    beta: float = 0.1,
    mask: Optional[torch.Tensor] = None,
    num_graphs: int = 1,
) -> torch.Tensor:
    """
    Binary cross-entropy on the convergence head logits.

    Target c_gt^(k) = sigmoid(-gamma * MSE_norm(x^(k), x_gt))
    where MSE_norm = MSE / sqrt(N_avg) (normalised by average per-graph
    node count for consistent gamma across different batch compositions).

    When placement quality is good, c_gt → 1 (converged, exit OK).
    When quality is poor,            c_gt → 0 (keep iterating).
    """
    N_avg = max(x_gt_rh.shape[0] / max(num_graphs, 1), 1.0)
    total = conv_scores[0].new_zeros(())
    for c_k_logit, x_k in zip(conv_scores, predictions):
        with torch.no_grad():
            if mask is not None:
                sq = ((x_k - x_gt_rh) ** 2)[mask]
                mse = sq.mean() if sq.numel() > 0 else x_k.new_zeros(())
            else:
                mse = F.mse_loss(x_k, x_gt_rh)
            mse_norm = mse / max(N_avg ** 0.5, 1.0)
            c_gt = torch.sigmoid(-gamma * mse_norm)

        # c_k_logit may be scalar (single graph) or [B] (batch)
        c_gt_expanded = c_gt.expand_as(c_k_logit)
        total = total + F.binary_cross_entropy_with_logits(
            c_k_logit, c_gt_expanded
        )
    return beta * total / max(len(conv_scores), 1)


def monotonicity_loss(
    predictions: List[torch.Tensor],
    x_gt_rh: torch.Tensor,
    eta: float = 0.05,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalise any iteration where MSE is worse than the previous iteration.

    Forces monotonic quality improvement — the model must get better at
    each step, not oscillate.

    Loss = sum_k max(0, MSE^(k) - MSE^(k-1))
    Note: MSE^(k-1) is detached so gradients only flow through x^(k).
    """
    if len(predictions) < 2:
        return predictions[0].new_zeros(())

    def _mse(x_k):
        if mask is not None:
            sq = ((x_k - x_gt_rh) ** 2)[mask]
            return sq.mean() if sq.numel() > 0 else x_k.new_zeros(())
        return F.mse_loss(x_k, x_gt_rh)

    total = predictions[0].new_zeros(())
    for k in range(1, len(predictions)):
        mse_k = _mse(predictions[k])
        mse_prev = _mse(predictions[k - 1]).detach()
        total = total + torch.clamp(mse_k - mse_prev, min=0.0)
    return eta * total


def convergence_regularizer(
    predictions: List[torch.Tensor],
    beta_conv: float = 0.01,
) -> torch.Tensor:
    """
    Soft regulariser penalising large position changes between consecutive steps.

    Encourages the model to settle rather than oscillate.
    Weight should be small (0.005–0.02).
    """
    if len(predictions) < 2:
        return predictions[0].new_zeros(())
    total = sum(
        F.mse_loss(predictions[k], predictions[k - 1].detach())
        for k in range(1, len(predictions))
    )
    return beta_conv * total / max(len(predictions) - 1, 1)


# --------------------------------------------------------------------------- #
# Combined loss                                                                 #
# --------------------------------------------------------------------------- #

class LoopedPlacementLoss:
    """
    Aggregates all placement loss terms.

    Config keys (all optional, with spec-recommended defaults):
        alpha_edge:   float = 0.1    — edge auxiliary loss weight
        beta_conv:    float = 0.1    — convergence BCE loss weight
        eta_mono:     float = 0.05   — monotonicity loss weight
        beta_reg:     float = 0.01   — convergence regulariser weight
        gamma_conv:   float = 2.0    — convergence head temperature
    """

    def __init__(self, config: Dict = None):
        cfg = config or {}
        self.alpha_edge = float(cfg.get("alpha_edge", 0.1))
        self.beta_conv = float(cfg.get("beta_conv", 0.1))
        self.eta_mono = float(cfg.get("eta_mono", 0.05))
        self.beta_reg = float(cfg.get("beta_reg", 0.01))
        self.gamma_conv = float(cfg.get("gamma_conv", 2.0))

    def __call__(
        self,
        node_preds: List[torch.Tensor],
        edge_preds: List[torch.Tensor],
        conv_scores: List[torch.Tensor],
        x_gt_rh: torch.Tensor,
        r_gt_rh: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        K_total: Optional[int] = None,
        step_offset: int = 0,
        num_graphs: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            node_preds:  List[Tensor[N,2]] — absolute positions in row-height units
            edge_preds:  List[Tensor[E,2]] — pairwise displacements
            conv_scores: List[Tensor]      — convergence scores
            x_gt_rh:     [N,2]            — ground-truth positions in row-height units
            r_gt_rh:     [E,2]            — ground-truth pairwise displacements
            mask:        [N] bool          — nodes to include in position loss (optional)
            K_total:     Total loop iterations (for correct lambda weighting
                         when using loss_window). Defaults to len(node_preds).
            step_offset: How many warmup steps preceded the first prediction
                         (i.e. first prediction corresponds to step step_offset+1).
            num_graphs:  Number of graphs in the batch (for per-graph normalisation
                         in convergence loss).

        Returns:
            total_loss: scalar tensor (backpropagatable)
            loss_dict:  dict of individual loss values for logging
        """
        K = len(node_preds)
        K_denom = K_total if K_total is not None else K
        lambda_weights = torch.arange(
            step_offset + 1, step_offset + K + 1,
            dtype=x_gt_rh.dtype, device=x_gt_rh.device,
        ) / K_denom

        L_pos = position_loss(node_preds, x_gt_rh, lambda_weights, mask)
        L_edge = edge_aux_loss(edge_preds, r_gt_rh, self.alpha_edge)
        L_conv = convergence_loss(conv_scores, node_preds, x_gt_rh, self.gamma_conv, self.beta_conv, mask, num_graphs)
        L_mono = monotonicity_loss(node_preds, x_gt_rh, self.eta_mono, mask)
        L_reg = convergence_regularizer(node_preds, self.beta_reg)

        total = L_pos + L_edge + L_conv + L_mono + L_reg

        return total, {
            "loss": total.item(),
            "pos": L_pos.item(),
            "edge": L_edge.item(),
            "conv": L_conv.item(),
            "mono": L_mono.item(),
            "reg": L_reg.item(),
        }


# --------------------------------------------------------------------------- #
# Ground truth helpers                                                          #
# --------------------------------------------------------------------------- #

def compute_edge_targets(
    x_gt: torch.Tensor,
    edge_index: torch.Tensor,
    h_row: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise displacement targets r_gt = (x_i - x_j) / h_row.

    Args:
        x_gt:       [N,2] ground-truth positions (physical µm or normalised)
        edge_index: [2,E]
        h_row:      technology row height (set to 1.0 if x_gt is already normalised)

    Returns: [E,2] displacements in row-height units, O(1) magnitude.
    """
    src, dst = edge_index
    return (x_gt[src] - x_gt[dst]) / h_row


def masked_position_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """RMSE over (optionally masked) nodes — for logging."""
    with torch.no_grad():
        if mask is not None:
            sq = ((pred - target) ** 2)[mask]
        else:
            sq = (pred - target) ** 2
        mse = sq.mean() if sq.numel() > 0 else pred.new_zeros(())
        return float(torch.sqrt(mse).item())
