"""FiLM controller C_phi for freeze-and-steer (paper Tier 3b, Eq. 5).

Minimal instantiation of the paper's controller: a feature-wise affine
modulation of the hidden state between recurrent applications of the frozen
operator, conditioned on (graph size, iteration index).

Ported design lessons from UnifiedLearning's LoopFiLMConditioner:
- zero-init of the FiLM head  ->  C_phi = identity at phi = 0, exactly the
  paper's (A7) requirement, and adaptation starts from the Tier-1 model.
- log-size input features (use_size_input) so the controller can steer as a
  function of deployment scale.
- iteration-index conditioning (per-(iter,size) modulation).
"""

import math

import torch
import torch.nn as nn


class FiLMController(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int = 32, max_k: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_k = max_k
        # cond inputs: [log N (normalized), k / max_k, sin/cos of k]
        self.cond_mlp = nn.Sequential(
            nn.Linear(4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.film = nn.Linear(cond_dim, 2 * hidden_dim)
        # zero-init: gamma = beta = 0  =>  identity controller at phi = 0
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _cond_features(
        self, num_nodes_per_graph: torch.Tensor, k: int, device
    ) -> torch.Tensor:
        logn = torch.log(num_nodes_per_graph.float().clamp(min=1.0)) / math.log(10_000.0)
        kk = torch.full_like(logn, float(k) / self.max_k)
        ang = 2.0 * math.pi * float(k) / self.max_k
        sk = torch.full_like(logn, math.sin(ang))
        ck = torch.full_like(logn, math.cos(ang))
        return torch.stack([logn, kk, sk, ck], dim=-1).to(device)

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_nodes_per_graph: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """h: (N_total, D); batch: (N_total,) graph id; returns steered h."""
        cond = self.cond_mlp(
            self._cond_features(num_nodes_per_graph, k, h.device)
        )  # (B, cond_dim)
        gamma, beta = self.film(cond).chunk(2, dim=-1)  # (B, D) each
        return h * (1.0 + gamma[batch]) + beta[batch]
