"""
Incidence bipartite attention blocks.

NodeToEdgeAttention:
    Each edge token e_{j→i} attends to {v_j, v_i, e_{j→i}} (fixed window of 3).
    Implements the "message function" ϕ(h_i, h_j, e_{ji}).

EdgeToNodeAttention:
    Each node token v_i attends to incoming edge tokens {e_{j→i} : (j,i)∈E} ∪ {v_i}.
    Implements the "aggregation" ψ(h_i, Σ m_{j→i}).
    Uses scatter-based softmax for variable-size attention windows.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import IncidenceTransformerConfig


def _get_activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}[name]


def _scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Numerically-stable softmax over variable-size groups.

    Args:
        src: (E, H) unnormalized scores
        index: (E,) group assignment (values in [0, num_nodes))
        num_nodes: total number of groups

    Returns:
        (E, H) attention weights summing to 1 within each group.
    """
    H = src.size(1)
    idx = index.unsqueeze(1).expand(-1, H)  # (E, H)

    # per-group max for numerical stability — detached (gradient not needed)
    with torch.no_grad():
        max_vals = src.new_full((num_nodes, H), float("-inf"))
        max_vals.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)

    src_stable = src - max_vals.gather(0, idx)
    src_stable = src_stable.clamp(max=15.0)  # prevent exp() overflow in fp16

    exp_src = src_stable.exp()

    sum_exp = src.new_zeros(num_nodes, H)
    sum_exp.scatter_add_(0, idx, exp_src)

    return exp_src / (sum_exp.gather(0, idx) + 1e-8)


class PreNormFFN(nn.Module):
    """
    Pre-norm feed-forward block with recurrence-aware scaling.

    x ← x + scale * FFN(LN(x))

    The output linear is zero-initialized when zero_init=True, making this
    block a no-op at initialization — critical for stable recurrence.
    """

    def __init__(
        self,
        d_model: int,
        multiplier: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_scale: float = 1.0,
    ):
        super().__init__()
        d_ff = int(d_model * multiplier)
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.act = _get_activation(activation)
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop2 = nn.Dropout(dropout)
        self.output_scale = output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop1(self.act(self.linear1(h)))
        h = self.drop2(self.linear2(h))
        return x + self.output_scale * h


class NodeToEdgeAttention(nn.Module):
    """
    Incidence V→E attention (message construction).

    Each edge token e_{j→i} attends to exactly {v_j, v_i, e_{j→i}}.
    Fixed window of 3 — no scatter needed, fully parallelizable.
    Pre-norm residual: caller adds the residual.
    """

    def __init__(self, config: IncidenceTransformerConfig):
        super().__init__()
        d = config.d_model
        H = config.num_heads_local
        self.d_model = d
        self.num_heads = H
        self.d_head = d // H
        self.scale = math.sqrt(self.d_head)

        self.norm_e = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        z_V: torch.Tensor,
        z_E: torch.Tensor,
        edge_index: torch.Tensor,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            (M, d) attention output (add to z_E for residual)
        """
        M = z_E.size(0)
        H = self.num_heads
        d_h = self.d_head

        src, tgt = edge_index[0], edge_index[1]

        z_E_n = self.norm_e(z_E)
        z_V_n = self.norm_v(z_V)

        Q = self.W_q(z_E_n).view(M, H, 1, d_h)

        kv_stack = torch.stack(
            [z_V_n[src], z_V_n[tgt], z_E_n], dim=1
        )  # (M, 3, d)

        K = self.W_k(kv_stack).view(M, 3, H, d_h).permute(0, 2, 1, 3)
        V = self.W_v(kv_stack).view(M, 3, H, d_h).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        if edge_bias is not None:
            attn_logits = attn_logits + edge_bias.unsqueeze(2)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V).squeeze(2)  # (M, H, d_h)
        out = out.reshape(M, self.d_model)

        return self.W_o(out)


class EdgeToNodeAttention(nn.Module):
    """
    Incidence E→V attention (aggregation).

    Each node v_i attends to {e_{j→i} : (j,i)∈E} ∪ {v_i} (variable size).
    Uses scatter-based softmax for efficient variable-window attention.
    Pre-norm residual: caller adds the residual.
    """

    def __init__(self, config: IncidenceTransformerConfig):
        super().__init__()
        d = config.d_model
        H = config.num_heads_local
        self.d_model = d
        self.num_heads = H
        self.d_head = d // H
        self.scale = math.sqrt(self.d_head)

        self.norm_e = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        z_E: torch.Tensor,
        z_V: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            (N, d) attention output (add to z_V for residual)
        """
        N = z_V.size(0)
        M = z_E.size(0)
        H = self.num_heads
        d_h = self.d_head
        device = z_V.device

        tgt = edge_index[1]

        z_E_n = self.norm_e(z_E)
        z_V_n = self.norm_v(z_V)

        Q_v = self.W_q(z_V_n).view(N, H, d_h)

        K_e = self.W_k(z_E_n).view(M, H, d_h)
        V_e = self.W_v(z_E_n).view(M, H, d_h)

        K_self = self.W_k(z_V_n).view(N, H, d_h)
        V_self = self.W_v(z_V_n).view(N, H, d_h)

        Q_for_edges = Q_v[tgt]
        edge_scores = (Q_for_edges * K_e).sum(-1) / self.scale
        self_scores = (Q_v * K_self).sum(-1) / self.scale

        self_loop_idx = torch.arange(N, device=device)
        aug_tgt = torch.cat([tgt, self_loop_idx], dim=0)
        aug_scores = torch.cat([edge_scores, self_scores], dim=0)
        aug_V = torch.cat([V_e, V_self], dim=0)

        attn_weights = _scatter_softmax(aug_scores, aug_tgt, N)
        attn_weights = self.attn_drop(attn_weights)

        weighted = attn_weights.unsqueeze(-1) * aug_V

        out = z_V.new_zeros(N, H, d_h)
        idx_expand = aug_tgt.unsqueeze(1).unsqueeze(2).expand(-1, H, d_h)
        out.scatter_add_(0, idx_expand, weighted)

        out = out.reshape(N, self.d_model)
        return self.W_o(out)
