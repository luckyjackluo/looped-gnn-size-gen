"""Sinusoidal timestep embedding for loop iteration index k."""

import math
import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding of the loop iteration index k, followed by a small MLP.

    Critical purposes (from spec §4.5):
    1. Breaks fixed-point collapse — different output at each k.
    2. Enables variable-K inference — model knows its current depth.
    3. Recovers expressivity of an equivalent-depth non-looped model.

    Output: [hidden_dim] float tensor.
    """

    def __init__(self, hidden_dim: int, max_k: int = 256):
        super().__init__()
        self.max_k = max_k
        half = hidden_dim // 2
        # Log-uniform frequencies over [1, max_k]
        freqs = torch.exp(-math.log(max_k) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)  # [half]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, k: int) -> torch.Tensor:
        """k: int iteration index (1-based). Returns [hidden_dim]."""
        t = torch.tensor(float(k), device=self.freqs.device, dtype=self.freqs.dtype)
        args = t * self.freqs  # [half]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # [hidden_dim]
        return self.mlp(emb)  # [hidden_dim]
