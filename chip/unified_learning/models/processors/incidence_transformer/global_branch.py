"""
Global branch and routing gate for nonlocal interactions.

GlobalMemoryBranch:
    O(nk) cross-attention via k << n learnable memory/hub tokens.
    Memory reads from nodes → self-attends → nodes read from memory.

DenseGlobalBranch:
    O(n^2) full self-attention on node tokens (use carefully on large graphs).

RouterGate:
    Learned per-element gating between incidence-local and global branches.
    g = σ(W·z),  output = g ⊙ z_inc + (1-g) ⊙ z_global
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import IncidenceTransformerConfig


class GlobalMemoryBranch(nn.Module):
    """
    Global context via learnable memory tokens.

    Steps:
        1. Memory tokens attend to all node tokens (read)
        2. Memory tokens self-attend
        3. Node tokens attend to memory tokens (write-back)

    Complexity: O(n·k + k²) where k = num_memory_tokens << n.
    """

    def __init__(self, config: IncidenceTransformerConfig):
        super().__init__()
        d = config.d_model
        H = config.num_heads_global
        k = config.num_memory_tokens
        dropout = config.dropout

        self.num_memory = k
        self.d_model = d

        self.memory = nn.Parameter(torch.randn(k, d) * 0.02)

        # cross-attn: memory ← nodes
        self.read_attn = nn.MultiheadAttention(d, H, dropout=dropout, batch_first=True)
        self.norm_read_q = nn.LayerNorm(d)
        self.norm_read_kv = nn.LayerNorm(d)

        # self-attn among memory tokens
        self.self_attn = nn.MultiheadAttention(d, H, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d)

        # cross-attn: nodes ← memory
        self.write_attn = nn.MultiheadAttention(d, H, dropout=dropout, batch_first=True)
        self.norm_write_q = nn.LayerNorm(d)
        self.norm_write_kv = nn.LayerNorm(d)

        # FFN on memory (lightweight)
        d_ff = int(d * 2)
        self.mem_ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_V: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_V: (N, d) flat node tokens
            batch: (N,) batch indices (None for single graph)

        Returns:
            z_V_global: (N, d) globally-updated node tokens
        """
        device = z_V.device

        # --- pack flat nodes into (B, max_n, d) ---
        if batch is None:
            z_dense = z_V.unsqueeze(0)  # (1, N, d)
            pad_mask = None
            B = 1
            counts = None
        else:
            try:
                from torch_geometric.utils import to_dense_batch
                z_dense, valid_mask = to_dense_batch(z_V, batch)  # (B, max_n, d), (B, max_n)
            except ImportError:
                raise ImportError("torch_geometric is required for batched GlobalMemoryBranch")
            pad_mask = ~valid_mask  # True where padded
            B = z_dense.size(0)
            counts = valid_mask.sum(dim=1)  # (B,)

        # memory tokens replicated per graph
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)  # (B, k, d)

        # Step 1: memory reads from nodes (V normalized for both K and V)
        kv_normed = self.norm_read_kv(z_dense)
        mem = mem + self.read_attn(
            self.norm_read_q(mem),
            kv_normed,
            kv_normed,
            key_padding_mask=pad_mask,
        )[0]

        # Step 2: memory self-attention
        mem_n = self.norm_self(mem)
        mem = mem + self.self_attn(mem_n, mem_n, mem_n)[0]

        # Step 3: memory FFN
        mem = mem + self.mem_ffn(mem)

        # Step 4: nodes read from memory (mem normalized for both K and V)
        mem_normed = self.norm_write_kv(mem)
        z_out = z_dense + self.write_attn(
            self.norm_write_q(z_dense),
            mem_normed,
            mem_normed,
        )[0]

        # --- unpack (B, max_n, d) back to flat (N, d) ---
        if batch is None:
            return z_out.squeeze(0)
        else:
            return z_out[valid_mask]  # only non-padded positions


class DenseGlobalBranch(nn.Module):
    """
    Full self-attention over all node tokens within each graph.
    O(n²) per graph — use on small/medium graphs only.
    """

    def __init__(self, config: IncidenceTransformerConfig):
        super().__init__()
        d = config.d_model
        H = config.num_heads_global
        dropout = config.dropout

        self.d_model = d
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, H, dropout=dropout, batch_first=True)

        d_ff = int(d * config.ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_V: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = z_V.device

        if batch is None:
            z_dense = z_V.unsqueeze(0)
            pad_mask = None
        else:
            from torch_geometric.utils import to_dense_batch
            z_dense, valid_mask = to_dense_batch(z_V, batch)
            pad_mask = ~valid_mask

        z_n = self.norm(z_dense)
        z_out = z_dense + self.attn(z_n, z_n, z_n, key_padding_mask=pad_mask)[0]
        z_out = z_out + self.ffn(z_out)

        if batch is None:
            return z_out.squeeze(0)
        else:
            return z_out[valid_mask]


class RouterGate(nn.Module):
    """
    Per-element learned gate between local (incidence) and global branches.

    g = σ(W·z_inc + b),  out = g ⊙ z_inc + (1 - g) ⊙ z_global

    Initialized to favor the local branch (bias > 0 → g ≈ 1).
    """

    def __init__(self, d_model: int, init_local_bias: float = 2.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_local_bias)

    def forward(self, z_inc: torch.Tensor, z_global: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_proj(z_inc))
        return g * z_inc + (1.0 - g) * z_global
