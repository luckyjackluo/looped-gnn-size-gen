"""Looped GNN: encode -> K x (shared process block [+ steer]) -> decode.

The paper's iterative network (Eq. 3/5).  The process block is k-independent
(the task-side operator U_N is "the same at every iteration step", Def. 2),
so deployment depth K is a free knob: unrolling deeper extends the receptive
field with zero new parameters (Tier 1), and a FiLM controller between
applications gives Tier 3b.

Note: unlike UnifiedLearning's placement models we deliberately do NOT inject
a timestep embedding into the operator — fixed-point convergence is the
*target* behavior for contractive F^op tasks, not a failure mode, and a
k-dependent operator would leave the paper's hypothesis class.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .blocks import ProcessorBlock, mlp
from .controller import FiLMController


class LoopedGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        heads: int = 4,
        controller: Optional[FiLMController] = None,
        anchored: bool = False,
    ):
        super().__init__()
        self.encoder = mlp(in_dim, hidden_dim, hidden_dim, layers=3)
        self.processor = ProcessorBlock(hidden_dim, heads=heads)
        self.decoder = mlp(hidden_dim, hidden_dim, out_dim, layers=3)
        self.controller = controller  # None => Tier 0/1/FT; set => FS
        # Anchored (contractive-by-construction) update, mirroring the F^op
        # damped operator form  h <- (1-a) * R(h) + a * h_enc  (cf. PageRank
        # h <- (1-alpha) P h + alpha s).  The anchor pull toward the encoder
        # state bounds the trajectory at ANY unroll depth, removing the
        # trained-depth stability horizon observed in E2 (K* pinned at ~24-29
        # while the task demanded ~59).  mix in (0.02, 0.5) via sigmoid.
        self.anchored = anchored
        if anchored:
            self.anchor_logit = nn.Parameter(torch.tensor(-2.0))  # mix ~ 0.11

    def anchor_mix(self) -> torch.Tensor:
        return 0.02 + 0.48 * torch.sigmoid(self.anchor_logit)

    # -- parameter groups (paper's d_R vs d_phi split) ---------------------
    def backbone_parameters(self):
        for m in (self.encoder, self.processor, self.decoder):
            for p in m.parameters():
                yield p

    def controller_parameters(self):
        return self.controller.parameters() if self.controller is not None else iter(())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_nodes_per_graph: torch.Tensor,
        K: int,
        return_all: bool = False,
        return_deltas: bool = False,
    ):
        h_enc = self.encoder(x)
        h = h_enc
        outs: List[torch.Tensor] = []
        deltas: List[torch.Tensor] = []  # ||h^k - h^{k-1}|| per step (A11 probe)
        for k in range(K):
            h_prev = h
            if self.controller is not None:
                h = self.controller(h, batch, num_nodes_per_graph, k)
            h = self.processor(h, edge_index)
            if self.anchored:
                a = self.anchor_mix()
                h = a * h_enc + (1.0 - a) * h
            if return_deltas:
                deltas.append((h - h_prev).norm())
            if return_all:
                outs.append(self.decoder(h))
        if return_all and return_deltas:
            return outs, deltas
        if return_all:
            return outs
        return self.decoder(h)
