"""Tier 0: fixed-depth GNN — K_fix independent blocks, no weight sharing.

The receptive field is capped at K_fix hops at deployment (paper §4.1,
Prop. 13's ceiling): the network cannot be repeated deeper than trained.
Parameter count is matched to the looped model's *per-block* size times
K_fix, the honest architectural counterpart.
"""

import torch
import torch.nn as nn

from .blocks import ProcessorBlock, mlp
from .controller import FiLMController


class FixedDepthGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        k_fix: int = 4,
        heads: int = 4,
        controller: FiLMController = None,
    ):
        super().__init__()
        self.k_fix = k_fix
        # Optional adapter for the Tier-0 variants: the same zero-init FiLM
        # module as the looped FS scheme, applied between the fixed layers
        # (conditioned on graph size and layer index; identity at phi = 0).
        self.controller = controller
        self.encoder = mlp(in_dim, hidden_dim, hidden_dim, layers=3)
        self.blocks = nn.ModuleList(
            [ProcessorBlock(hidden_dim, heads=heads) for _ in range(k_fix)]
        )
        self.decoder = mlp(hidden_dim, hidden_dim, out_dim, layers=3)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        num_nodes_per_graph: torch.Tensor = None,
        K: int = None,  # accepted-and-ignored: depth is architectural here
        return_all: bool = False,
        return_deltas: bool = False,
    ):
        h = self.encoder(x)
        outs, deltas = [], []
        for k, blk in enumerate(self.blocks):
            h_prev = h
            if self.controller is not None:
                h = self.controller(h, batch, num_nodes_per_graph, k)
            h = blk(h, edge_index)
            if return_deltas:
                deltas.append((h - h_prev).norm())
            if return_all:
                outs.append(self.decoder(h))
        if return_all and return_deltas:
            return outs, deltas
        if return_all:
            return outs
        return self.decoder(h)
