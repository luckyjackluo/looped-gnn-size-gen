"""IterGNN-style placement model following Algorithm 1 from:

    "Towards Scale-Invariant Graph-related Problem Solving by Iterative
     Homogeneous Graph Neural Networks"
     (NeurIPS 2020, arXiv:2010.13547)

Architecture
------------
1. Encode once: h^0 = node_encoder(x),  he^0 = edge_encoder(edge_attr)
2. Loop K iterations (shared weights — same nn.Module called K times):
     h^k  = f(h^{k-1})                  # body: all processor blocks, no input re-injection
     he^k = he^{k-1} + LN(edge_update(he, h^k_src, h^k_dst))
     x^k  = node_head(h^k)              # per-step prediction
     r^k  = edge_head(he^k)
     c^k  = sigmoid(confidence_mlp(mean_pool(h^k, batch)))  # [G] ∈ (0,1)
3. Geometric halt distribution (Algorithm 1):
     p^k = c^k * Π_{i<k}(1 - c^i),   with p^K forced so that Σ_k p^k = 1
4. Expected output (the model's single prediction):
     x_out = Σ_k  p^k[batch]  * x^k   # [N, 2]
     r_out = Σ_k  p^k[e_batch]* r^k   # [E, 2]
5. Return ([x_out], [r_out], [], [], None) — one-prediction 5-tuple that is
   directly consumable by GatedHaltPlacementLoss (Variant A with K_total=1).

Key differences from GatedHaltLoopedPlacementModel
---------------------------------------------------
* No input re-injection into h at each iteration.
* No per-node gate (GatedResidualLoopConditioner removed).
* No timestep embedding injected at each step.
* Confidence score is a simple per-graph scalar, not a per-node mechanism.
* The output returned to the trainer is already the confidence-weighted
  expectation; the trainer loss sees a single prediction (K_total=1).

Trainer compatibility
---------------------
The trainer (train_chipgen_gated_halt_looped.py) reads three attributes for
logging:  model.loop_pondernet_enabled,  model.gate.activation,  and
model.loop_halt_distribution.  Stub values are set in __init__ so no changes
to the training script are needed beyond the K_total fix described below.

Loss K_total fix
----------------
In run_epoch, pass  K_total=len(node_preds)  instead of  K_total=K  so that
the single expected prediction gets lambda_weight = 1.0 (not 1/K).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..encoders.mlp import MLPEncoder
from ..model import _build_gnn_layers, _gnn_layers_for_processor_block
from .heads import EdgePredictionHead, NodePredictionHead


def _graph_mean_pool(
    h: torch.Tensor,
    batch: Optional[torch.Tensor],
    num_graphs: int,
) -> torch.Tensor:
    """Mean-pool node features per graph. Returns [G, d]."""
    d = h.shape[1]
    if batch is None:
        return h.mean(dim=0, keepdim=True)  # [1, d]
    out = h.new_zeros(num_graphs, d)
    cnt = h.new_zeros(num_graphs, 1)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
    cnt.scatter_add_(0, batch.unsqueeze(1), h.new_ones(h.shape[0], 1))
    return out / cnt.clamp_min(1.0)


class StandardIterGNNPlacementModel(nn.Module):
    """Iterative GNN placement model following the IterGNN paper (arXiv:2010.13547).

    Config keys
    -----------
    Shares the same flat config format as GatedHaltLoopedPlacementModel.
    The ``loop_conditioning`` and ``halting`` sub-dicts are ignored.
    Additional key:

        confidence_hidden_ratio: int = 4   # confidence MLP width = hidden_dim // ratio
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        hidden_dim: int = int(config["hidden_dim"])
        node_in_dim: int = int(config["node_in_dim"])
        edge_in_dim: int = int(config.get("edge_in_dim", 4))

        self.hidden_dim = hidden_dim
        self.h_row: float = float(config.get("h_row", 1.0))
        self.min_k_inference: int = int(config.get("min_k_inference", 1))
        self.use_activation_checkpointing: bool = bool(
            config.get("use_activation_checkpointing", False)
        )

        # ------------------------------------------------------------------ #
        # Encoder — runs ONCE before the K-loop                               #
        # ------------------------------------------------------------------ #
        self.node_encoder = MLPEncoder(
            input_dim=node_in_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=int(config.get("encoder_layers", 3)),
            dropout=float(config.get("encoder_dropout", 0.0)),
            use_layer_norm=True,
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ------------------------------------------------------------------ #
        # Processor — shared weights, called K times in the loop              #
        # All blocks (GNN + optional global/transformer) run inside the loop. #
        # ------------------------------------------------------------------ #
        processor_cfg: Dict[str, Any] = config["processor"]
        if processor_cfg.get("type") != "gnn":
            raise ValueError(
                "StandardIterGNNPlacementModel requires processor.type='gnn'. "
                f"Got: {processor_cfg.get('type')}"
            )
        if int(processor_cfg.get("edge_dim", hidden_dim)) != hidden_dim:
            raise ValueError(
                f"processor.edge_dim must equal hidden_dim ({hidden_dim}). "
                f"Got edge_dim={processor_cfg.get('edge_dim')}."
            )

        _result = _build_gnn_layers(processor_cfg, hidden_dim, time_embed_dim=None)
        if len(_result) == 4:
            (
                self.gnn_blocks,
                self.global_modules,
                self.num_blocks,
                self.layers_per_block,
            ) = _result
            self.gated_adapters = None
            self.gnn_layer_ranges = None
        elif len(_result) == 5:
            (
                self.gnn_blocks,
                self.global_modules,
                self.num_blocks,
                self.layers_per_block,
                self.gated_adapters,
            ) = _result
            self.gnn_layer_ranges = None
        else:
            (
                self.gnn_blocks,
                self.global_modules,
                self.num_blocks,
                self.layers_per_block,
                self.gated_adapters,
                self.gnn_layer_ranges,
            ) = _result
        self._use_residual: bool = bool(processor_cfg.get("use_residual", True))
        self._has_recurrent_global_memory = bool(
            self.global_modules is not None
            and any(hasattr(m, "forward_with_memory") for m in self.global_modules)
        )

        # Edge update (additive residual).
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------ #
        # Confidence MLP — g() in Algorithm 1 of the IterGNN paper           #
        # Maps per-graph mean-pooled h^k to a scalar confidence c^k ∈ (0,1). #
        # ------------------------------------------------------------------ #
        conf_ratio: int = int(config.get("confidence_hidden_ratio", 4))
        conf_hidden = max(hidden_dim // conf_ratio, 1)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(hidden_dim, conf_hidden),
            nn.SiLU(),
            nn.Linear(conf_hidden, 1),
        )
        # Initialise the final layer's bias to 0 → sigmoid(0) = 0.5 initially.
        nn.init.zeros_(self.confidence_mlp[-1].bias)
        nn.init.xavier_uniform_(self.confidence_mlp[-1].weight, gain=0.1)

        # ------------------------------------------------------------------ #
        # Prediction heads — shared across K iterations                       #
        # ------------------------------------------------------------------ #
        self.node_head = NodePredictionHead(hidden_dim)
        self.edge_head = EdgePredictionHead(hidden_dim)
        self._small_init(self.node_head)
        self._small_init(self.edge_head)

        # ------------------------------------------------------------------ #
        # Trainer / evaluator compatibility stubs                              #
        # ------------------------------------------------------------------ #
        self.loop_pondernet_enabled: bool = False
        self.loop_halt_distribution: str = "itergnn_geometric"
        # gate.activation is read by the training / test scripts for logging.
        self.gate = SimpleNamespace(activation="none")
        # _last_gate_aux and _last_ponder_aux are checked for None in the eval
        # script before use, so setting them here avoids AttributeError.
        self._last_gate_aux = None
        self._last_ponder_aux = None

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _small_init(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _run_processor_once(
        self,
        h: torch.Tensor,
        he: torch.Tensor,
        edge_index: torch.Tensor,
        data,
        memory_states: Optional[List[Optional[torch.Tensor]]] = None,
        step_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]]]:
        """One pass of the shared processor (all blocks).

        Unlike GatedHaltLoopedPlacementModel, ALL blocks (including any global /
        transformer block) are executed here in every iteration.
        """
        updated_memory = (
            list(memory_states) if memory_states is not None else None
        )
        if self.num_blocks is not None:
            for block_idx in range(self.num_blocks):
                for layer_idx, gnn_layer in _gnn_layers_for_processor_block(
                    self.gnn_blocks,
                    block_idx,
                    layers_per_block=self.layers_per_block,
                    gnn_layer_ranges=self.gnn_layer_ranges,
                ):
                    gnn_data = {"edge_index": edge_index, "edge_attr": he}
                    h_new = gnn_layer(gnn_data, x=h)
                    if self._use_residual:
                        h = h + h_new
                    else:
                        h = torch.relu(h_new)

                if self.global_modules is not None:
                    global_mod = self.global_modules[block_idx]
                    inst_batch = getattr(data, "batch", None)
                    if hasattr(global_mod, "forward_with_memory"):
                        cur_state = (
                            None
                            if updated_memory is None
                            else updated_memory[block_idx]
                        )
                        inst_delta, _, next_state = global_mod.forward_with_memory(
                            inst_x=h,
                            net_x=h,
                            memory_state=cur_state,
                            inst_batch=inst_batch,
                            data=data,
                            step_idx=step_idx,
                        )
                        if updated_memory is not None:
                            updated_memory[block_idx] = next_state
                    else:
                        inst_delta, _ = global_mod(
                            inst_x=h,
                            net_x=h,
                            inst_batch=inst_batch,
                            data=data,
                            step_idx=step_idx,
                        )
                    if self.gated_adapters is not None:
                        h = self.gated_adapters[block_idx](
                            gnn_output=h, transformer_output=inst_delta
                        )
                    else:
                        h = h + 0.1 * inst_delta
        else:
            for i, block in enumerate(self.gnn_blocks):
                gnn_data = {"edge_index": edge_index, "edge_attr": he}
                h_new = block(gnn_data, x=h)
                if self._use_residual:
                    h = h + h_new
                else:
                    h = torch.relu(h_new)

                if self.global_modules is not None:
                    global_mod = self.global_modules[i]
                    inst_batch = getattr(data, "batch", None)
                    inst_delta, _ = global_mod(
                        inst_x=h,
                        net_x=h,
                        inst_batch=inst_batch,
                        data=data,
                        step_idx=step_idx,
                    )
                    if self.gated_adapters is not None:
                        h = self.gated_adapters[i](
                            gnn_output=h, transformer_output=inst_delta
                        )
                    else:
                        h = h + 0.1 * inst_delta

        return h, updated_memory

    def _init_memory_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if not self._has_recurrent_global_memory or self.global_modules is None:
            return None
        out: List[Optional[torch.Tensor]] = []
        for module in self.global_modules:
            if hasattr(module, "init_loop_state"):
                out.append(
                    module.init_loop_state(
                        batch_size=batch_size, device=device, dtype=dtype
                    )
                )
            else:
                out.append(None)
        return out

    # ---------------------------------------------------------------------- #
    # Forward — Algorithm 1 from arXiv:2010.13547                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        data,
        K: int,
        return_all: bool = True,
    ) -> Tuple[
        List[torch.Tensor],            # node_preds  — length 1: [x_expected]
        List[torch.Tensor],            # edge_preds  — length 1: [r_expected]
        List[torch.Tensor],            # gate_per_iter — always []
        List[torch.Tensor],            # gate_logits   — always []
        None,                          # halt_logits   — always None
    ]:
        """Run K iterations, build geometric halt distribution, return expected output.

        The returned ``node_preds`` contains **one** tensor: the confidence-
        weighted expected position prediction  x_out = Σ_k p^k * x^k.
        The trainer loss should therefore use  K_total = len(node_preds) = 1
        so the single prediction receives lambda_weight = 1.0.
        """
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")

        edge_index = data.edge_index
        batch: Optional[torch.Tensor] = getattr(data, "batch", None)

        # Encode once.
        h = self.node_encoder(data.x)            # [N, d]
        he = self.edge_encoder(data.edge_attr)   # [E, d]

        num_nodes: int = int(h.shape[0])
        num_graphs: int = (int(batch.max().item()) + 1) if batch is not None else 1
        memory_states = self._init_memory_states(
            batch_size=num_graphs, device=h.device, dtype=h.dtype
        )

        src, dst = edge_index

        # Per-iteration accumulators.
        node_preds_all: List[torch.Tensor] = []
        edge_preds_all: List[torch.Tensor] = []
        conf_scores: List[torch.Tensor] = []   # each: [G]

        for k in range(1, K + 1):
            # --- body: h^k = f(h^{k-1}) — NO input re-injection ------------
            if (
                self.training
                and self.use_activation_checkpointing
                and not self._has_recurrent_global_memory
            ):
                h = checkpoint(
                    lambda h_in, he_in, _k=k: self._run_processor_once(
                        h_in, he_in, edge_index, data, None, _k
                    )[0],
                    h, he, use_reentrant=False,
                )
            else:
                h, memory_states = self._run_processor_once(
                    h, he, edge_index, data, memory_states, k
                )

            # --- edge update -----------------------------------------------
            he_delta = self.edge_update(
                torch.cat([he, h[src], h[dst]], dim=-1)
            )
            he = he + self.edge_update_norm(he_delta)

            # --- per-step predictions --------------------------------------
            x_k = self.node_head(h, he, edge_index)  # [N, 2]
            r_k = self.edge_head(he)                   # [E, 2]
            node_preds_all.append(x_k)
            edge_preds_all.append(r_k)

            # --- confidence score: c^k = σ(g(mean_pool(h^k))) --------------
            h_mean = _graph_mean_pool(h, batch, num_graphs)   # [G, d]
            c_k = torch.sigmoid(
                self.confidence_mlp(h_mean).squeeze(-1)
            ).float()                                          # [G]
            conf_scores.append(c_k)

        # --- Build geometric halt distribution (Algorithm 1) ---------------
        # p^k = c^k * Π_{i=1}^{k-1}(1 - c^i),  p^K forced to exhaust remainder.
        halt_probs: List[torch.Tensor] = []
        remainders = torch.ones(num_graphs, device=h.device, dtype=torch.float32)
        for k in range(K):
            if k == K - 1:
                p_k = remainders                          # force sum-to-one at last step
            else:
                p_k = remainders * conf_scores[k]
                remainders = remainders * (1.0 - conf_scores[k])
            halt_probs.append(p_k)  # [G]

        # --- Expected output: x_out = Σ_k p^k[batch] * x^k ----------------
        x_expected = torch.zeros(num_nodes, 2, device=h.device, dtype=h.dtype)
        r_expected = torch.zeros(int(edge_index.shape[1]), 2, device=h.device, dtype=h.dtype)

        edge_batch = batch[src] if batch is not None else torch.zeros(
            src.shape[0], dtype=torch.long, device=h.device
        )

        for k in range(K):
            p_node = halt_probs[k][batch if batch is not None else torch.zeros(
                num_nodes, dtype=torch.long, device=h.device
            )].unsqueeze(-1).to(h.dtype)                 # [N, 1]
            x_expected = x_expected + p_node * node_preds_all[k]

            p_edge = halt_probs[k][edge_batch].unsqueeze(-1).to(h.dtype)  # [E, 1]
            r_expected = r_expected + p_edge * edge_preds_all[k]

        # Return a single-element list so the trainer sees one prediction.
        # Set K_total=len(node_preds)=1 in the trainer to get lambda_weight=1.0.
        return (
            [x_expected],
            [r_expected],
            [],          # no gate
            [],          # no gate logits
            None,        # no halt logits side-channel
        )

    @torch.no_grad()
    def predict_physical(self, data, K: int) -> Tuple[torch.Tensor, int]:
        """Inference: return positions in physical µm + K_used."""
        self.eval()
        node_preds, _, _, _, _ = self.forward(data, K=K, return_all=True)
        return node_preds[0] * self.h_row, K
