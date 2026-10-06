"""Iterative-from-scratch looped placement model with per-node gated residual.

Sibling of :class:`LoopedPlacementModel`. Both models train end-to-end from
random init on the same data; this file adds two new variants that are
directly comparable to the existing one:

    Variant A (``halting.enabled == False``): per-node *hard* gated residual
        replaces the implicit "always absorb the new iter's update" of
        ``LoopedPlacementModel``. Gate activation can be Hard Concrete (the
        default - capable of exact 0 / 1 outputs and trainable with a
        closed-form L0 probability) or Gumbel-Sigmoid with a
        straight-through estimator. There is no separate halt module: the
        gate IS the halt mechanism. A node whose gate hits 0 freezes for
        the rest of the loop. Final prediction is the last iteration's
        node head (``x_K``), same shape as the existing model.

    Variant B (``halting.enabled == True``, iterGNN):
        - per-node *soft* sigmoid gate (cannot hit exact 0 - the discrete
          stop decision is decoupled from the residual update).
        - separate :class:`HaltModule` emits a per-graph halt logit at every
          iteration. The model builds a halt distribution ``p_k`` over k
          via either a PonderNet cumulative recurrence or a softmax over
          k. Inference returns ``expected_pred = sum_k p_k * x_k`` (soft)
          or hard early-exit at the cumulative threshold.

What is shared with :class:`LoopedPlacementModel`:
    Encoder runs once, processor weights are SHARED across the K iterations
    (same nn.Module called K times), prediction heads are shared, edge
    embeddings ``he`` thread through and are updated each iter, timestep
    sinusoidal embedding of k is injected each step, PreNorm throughout.

Architectural differences from :class:`LoopedPlacementModel`:
    1. The BCE convergence head is REMOVED. The gate (Variant A) or
       HaltModule (Variant B) takes over the "should we keep going?" role
       and gets a learnable signal end-to-end instead of via a frozen-
       target BCE.
    2. The per-iter residual update is no longer the unconditional clipped
       sum ``h_post = clip(h_prev + delta)``. Instead, after running one
       iteration of (processor + edge update + timestep injection) to
       produce ``h_post``, we mix back with ``h_prev`` via the gate:
       ``h = (1 - g) * h_prev + g * h_post``.
    3. Variant B exposes per-iter predictions to the loss (already true of
       the existing model) but reweights them by the learned ``p_k``
       instead of a fixed ``lambda_k = k/K`` ramp.

The model populates two side-channels every forward (both can be None for
non-supported variants):

    self._last_gate_aux   = {
        "gate_per_iter"      : [K, N]  in [0, 1] (exact 0/1 possible)
        "gate_logit_per_iter": [K, N]
        "gate_activation"    : str
        "node_batch"         : [N]
    }
    self._last_ponder_aux = {                           # Variant B only
        "halt_probs"   : [K, G]  fp32, sums to 1 along k
        "halt_logits"  : [K, G]  fp32
        "pred_per_iter": [K, N, 2]  fp32  (positions in row-height units)
        "node_batch"   : [N]
        "distribution" : 'pondernet_geometric' | 'softmax_over_k'
    }

The downstream :func:`gated_halt_placement_loss` reads these side-channels
to compute the weighted-by-``p_k`` task loss and the regularisation terms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..encoders.mlp import MLPEncoder
from ..loop_conditioner import (
    GatedResidualLoopConditioner,
    HaltModule,
    _graph_size_features_from_batch,
)
from ..model import _build_gnn_layers, _gnn_layers_for_processor_block
from .heads import EdgePredictionHead, NodePredictionHead
from .timestep import TimestepEmbedding


class GatedHaltLoopedPlacementModel(nn.Module):
    """Looped placement regression with per-node gated residual + optional
    learnable halt distribution.

    Config keys
    -----------
    Most of the existing :class:`LoopedPlacementModel` keys are accepted as
    is. The new sub-keys live under ``loop_conditioning`` and ``halting``:

    loop_conditioning:
        num_iterations: int        # K used by the gate's iter embedding (>=
                                   # K_max if K varies per batch).
        conditioning_dim: int = 64
        use_size_input: bool = True
        gate:
            activation: 'hard_concrete' (default) | 'gumbel_st' | 'sigmoid'
            init_gate_bias: float = 2.2
            hc_beta / hc_gamma / hc_zeta: Hard Concrete stretch params
            gumbel_temperature_*: Gumbel-ST anneal params
            l0_weight / l0_warmup_steps: read by the loss path
            entropy_weight / entropy_warmup_steps: read by the loss path

    halting:
        enabled: bool = False               # False -> Variant A, True -> B
        distribution: 'pondernet_geometric' (default) | 'softmax_over_k'
        inference_mode: 'soft' (default) | 'hard'
        inference_threshold: float = 0.5
        prior_lambda: float = 0.4           # for the geometric prior
        reg_weight: float = 0.01            # weight on KL / negative entropy
        reg_warmup_steps: int = 0
        init_halt_bias: float = -2.0        # start in "always run K" mode
        use_gate_statistic: bool = True
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        hidden_dim: int = int(config["hidden_dim"])
        node_in_dim: int = int(config["node_in_dim"])
        edge_in_dim: int = int(config.get("edge_in_dim", 4))

        self.hidden_dim = hidden_dim
        self.h_row: float = float(config.get("h_row", 1.0))
        # The "convergence_threshold" knob is intentionally absent here; the
        # halt module (Variant B) consumes ``halting.inference_threshold``
        # and Variant A halts implicitly via gate values reaching 0.
        self.min_k_inference: int = int(config.get("min_k_inference", 1))
        self.use_activation_checkpointing: bool = bool(
            config.get("use_activation_checkpointing", False)
        )

        # ------------------------------------------------------------------ #
        # Encoder - runs ONCE before the loop                                 #
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
        # Processor - shared weights, called K times                          #
        # ------------------------------------------------------------------ #
        processor_cfg: Dict[str, Any] = config["processor"]
        if processor_cfg.get("type") != "gnn":
            raise ValueError(
                "GatedHaltLoopedPlacementModel requires processor.type='gnn'. "
                f"Got: {processor_cfg.get('type')}"
            )
        if int(processor_cfg.get("edge_dim", hidden_dim)) != hidden_dim:
            raise ValueError(
                f"processor.edge_dim must equal hidden_dim ({hidden_dim}) "
                f"so that he can be passed as edge_attr. "
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

        # Determine which block indices to loop (GNN only) and which to use
        # as a stateless per-iteration readout (global/Perceiver blocks).
        # When gnn_blocks and global_module_blocks are both explicit in the
        # config, only the GNN blocks are iterated; global blocks run once
        # per step as a readout and their output is NOT fed back into h so
        # the loop hidden state only passes through GNN layers.
        _gnn_blks_cfg = processor_cfg.get("gnn_blocks")
        _global_blks_cfg = processor_cfg.get("global_module_blocks")
        if (
            self.num_blocks is not None
            and _gnn_blks_cfg is not None
            and _global_blks_cfg is not None
        ):
            self._gnn_loop_block_indices: Optional[List[int]] = [int(i) for i in _gnn_blks_cfg]
            self._global_readout_block_indices: List[int] = [int(i) for i in _global_blks_cfg]
        else:
            # Legacy fallback: run all blocks in the loop (no explicit split).
            self._gnn_loop_block_indices = None  # None → run all blocks
            self._global_readout_block_indices = []

        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update_norm = nn.LayerNorm(hidden_dim)

        self.timestep_emb = TimestepEmbedding(
            hidden_dim=hidden_dim,
            max_k=int(config.get("max_k", 256)),
        )
        self.timestep_proj = nn.Linear(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # Prediction heads - shared across K iterations                       #
        # ------------------------------------------------------------------ #
        self.node_head = NodePredictionHead(hidden_dim)
        self.edge_head = EdgePredictionHead(hidden_dim)
        self._small_init(self.node_head)
        self._small_init(self.edge_head)

        # ------------------------------------------------------------------ #
        # Gated residual + optional HaltModule                                #
        # ------------------------------------------------------------------ #
        loop_cfg = dict(config.get("loop_conditioning", {}) or {})
        num_iters_gate = int(
            loop_cfg.get("num_iterations", int(config.get("K_max", 10)))
        )
        conditioning_dim = int(loop_cfg.get("conditioning_dim", 64))
        use_size_input = bool(loop_cfg.get("use_size_input", True))
        gate_cfg = dict(loop_cfg.get("gate", {}) or {})
        gate_activation = str(gate_cfg.get("activation", "hard_concrete"))
        self.gate = GatedResidualLoopConditioner(
            hidden_dim=hidden_dim,
            num_iterations=num_iters_gate,
            conditioning_dim=conditioning_dim,
            activation=gate_activation,
            use_size_input=use_size_input,
            init_gate_bias=float(gate_cfg.get("init_gate_bias", 2.2)),
            hc_beta=float(gate_cfg.get("hc_beta", 2.0 / 3.0)),
            hc_gamma=float(gate_cfg.get("hc_gamma", -0.1)),
            hc_zeta=float(gate_cfg.get("hc_zeta", 1.1)),
            gumbel_temperature_start=float(
                gate_cfg.get("gumbel_temperature_start", 1.0)
            ),
            gumbel_temperature_end=float(
                gate_cfg.get("gumbel_temperature_end", 0.1)
            ),
            gumbel_temperature_anneal_steps=int(
                gate_cfg.get("gumbel_temperature_anneal_steps", 0)
            ),
            gate_sharpness=float(gate_cfg.get("gate_sharpness", 5.0)),
        )
        # Gate reg weights are read by the loss path.
        self.loop_gate_l0_weight = float(gate_cfg.get("l0_weight", 0.0))
        self.loop_gate_l0_warmup_steps = int(gate_cfg.get("l0_warmup_steps", 0))
        self.loop_gate_entropy_weight = float(gate_cfg.get("entropy_weight", 0.0))
        self.loop_gate_entropy_warmup_steps = int(
            gate_cfg.get("entropy_warmup_steps", 0)
        )

        halt_cfg = dict(loop_cfg.get("halting", {}) or {})
        # Maintain naming parity with UnifiedModel.
        self.loop_pondernet_enabled = bool(halt_cfg.get("enabled", False))
        self.loop_halt_distribution = str(
            halt_cfg.get("distribution", "pondernet_geometric")
        )
        if self.loop_halt_distribution not in (
            "pondernet_geometric",
            "softmax_over_k",
        ):
            raise ValueError(
                "halting.distribution must be 'pondernet_geometric' or "
                f"'softmax_over_k'; got {self.loop_halt_distribution!r}"
            )
        self.loop_halt_inference_mode = str(halt_cfg.get("inference_mode", "soft"))
        if self.loop_halt_inference_mode not in ("soft", "hard"):
            raise ValueError(
                "halting.inference_mode must be 'soft' or 'hard'; got "
                f"{self.loop_halt_inference_mode!r}"
            )
        self.loop_halt_inference_threshold = float(
            halt_cfg.get("inference_threshold", 0.5)
        )
        self.loop_halt_prior_lambda = float(halt_cfg.get("prior_lambda", 0.4))
        self.loop_halt_reg_weight = float(halt_cfg.get("reg_weight", 0.01))
        self.loop_halt_reg_warmup_steps = int(halt_cfg.get("reg_warmup_steps", 0))

        if self.loop_pondernet_enabled:
            self.halt_module = HaltModule(
                hidden_dim=hidden_dim,
                num_iterations=num_iters_gate,
                conditioning_dim=conditioning_dim,
                use_size_input=use_size_input,
                use_gate_statistic=bool(halt_cfg.get("use_gate_statistic", True)),
                init_halt_bias=float(halt_cfg.get("init_halt_bias", -2.0)),
            )
        else:
            self.halt_module = None

        # Side-channels (populated each forward).
        self._last_gate_aux: Optional[Dict[str, torch.Tensor]] = None
        self._last_ponder_aux: Optional[Dict[str, torch.Tensor]] = None
        self.register_buffer(
            "_ponder_step",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _small_init(module: nn.Module):
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
        block_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]]]:
        """One pass of the shared-weight processor blocks.

        Args:
            block_indices: If given, only run these block indices (0-based).
                ``None`` runs all blocks.  Pass ``self._gnn_loop_block_indices``
                for the K-loop body and ``self._global_readout_block_indices``
                for the per-iteration stateless readout.
        """
        updated_memory_states = (
            list(memory_states) if memory_states is not None else None
        )
        if self.num_blocks is not None:
            _run_block_iter = (
                block_indices if block_indices is not None else range(self.num_blocks)
            )
            for block_idx in _run_block_iter:
                for layer_idx, gnn_layer in _gnn_layers_for_processor_block(
                    self.gnn_blocks,
                    block_idx,
                    layers_per_block=self.layers_per_block,
                    gnn_layer_ranges=self.gnn_layer_ranges,
                ):
                    gnn_data = {"edge_index": edge_index, "edge_attr": he}
                    h_new = gnn_layer(gnn_data, x=h)
                    if self._use_residual:
                        # Per-layer residual without the clipped wrapper:
                        # the OUTER per-iter gate is the policy lever, not
                        # the per-layer norm clip from LoopedPlacementModel.
                        h = h + h_new
                    else:
                        h = torch.relu(h_new)

                if self.global_modules is not None:
                    global_mod = self.global_modules[block_idx]
                    inst_batch = getattr(data, "batch", None)
                    if hasattr(global_mod, "forward_with_memory"):
                        current_state = (
                            None
                            if updated_memory_states is None
                            else updated_memory_states[block_idx]
                        )
                        inst_delta, _, next_state = global_mod.forward_with_memory(
                            inst_x=h,
                            net_x=h,
                            memory_state=current_state,
                            inst_batch=inst_batch,
                            data=data,
                            step_idx=step_idx,
                        )
                        if updated_memory_states is not None:
                            updated_memory_states[block_idx] = next_state
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
                    if hasattr(global_mod, "forward_with_memory"):
                        current_state = (
                            None
                            if updated_memory_states is None
                            else updated_memory_states[i]
                        )
                        inst_delta, _, next_state = global_mod.forward_with_memory(
                            inst_x=h,
                            net_x=h,
                            memory_state=current_state,
                            inst_batch=inst_batch,
                            data=data,
                            step_idx=step_idx,
                        )
                        if updated_memory_states is not None:
                            updated_memory_states[i] = next_state
                    else:
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
        return h, updated_memory_states

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
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        data,
        K: int,
        return_all: bool = True,
    ) -> Tuple[
        List[torch.Tensor],          # node_preds
        List[torch.Tensor],          # edge_preds
        List[torch.Tensor],          # gate per iter [N]
        List[torch.Tensor],          # gate_logit per iter [N]
        Optional[List[torch.Tensor]],# halt_logits per iter [G] (Variant B only)
    ]:
        """Run the gated K-iteration loop and return per-iter outputs.

        Returns (always 5-tuple to keep the loss API uniform):
            node_preds:       list of [N, 2] positions in row-height units, len=K
            edge_preds:       list of [E, 2] displacements, len=K
            gate_per_iter:    list of [N] gate values in [0, 1], len=K
            gate_logits:      list of [N] gate logits (pre-activation), len=K
            halt_logits:      Variant B: list of [G] halt logits per iter; else None
        """
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")

        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        # ---- Encode once -----------------------------------------------
        h = self.node_encoder(data.x)            # [N, d]
        he = self.edge_encoder(data.edge_attr)   # [E, d]
        num_nodes = int(h.shape[0])
        batch_size = int(batch.max().item()) + 1 if batch is not None else 1
        memory_states = self._init_memory_states(
            batch_size=batch_size, device=h.device, dtype=h.dtype
        )

        # Resolve graph-size stats once for the halt module / aux.
        _stats, node_batch_g, num_graphs_g = _graph_size_features_from_batch(
            batch, edge_index, num_nodes
        )

        node_preds: List[torch.Tensor] = []
        edge_preds: List[torch.Tensor] = []
        gate_per_iter: List[torch.Tensor] = []
        gate_logit_per_iter: List[torch.Tensor] = []
        halt_logits_per_iter: List[torch.Tensor] = (
            [] if self.loop_pondernet_enabled else None
        )

        for k in range(1, K + 1):
            iter_idx = k - 1  # 0-indexed iteration for the gate's embedding.
            h_prev = h
            he_prev = he

            # --- 1) Run GNN-only blocks (shared weights, looped). --------
            # Global/Perceiver blocks are excluded here; they run as a
            # stateless readout in step 5 below.
            _loop_bi = self._gnn_loop_block_indices
            if (
                self.training
                and self.use_activation_checkpointing
                and not self._has_recurrent_global_memory
            ):
                h_after_proc = checkpoint(
                    lambda h_in, he_in, _k=k, _bi=_loop_bi: self._run_processor_once(
                        h_in, he_in, edge_index, data, None, _k, _bi,
                    )[0],
                    h, he, use_reentrant=False,
                )
            else:
                h_after_proc, memory_states = self._run_processor_once(
                    h, he, edge_index, data, memory_states, k, _loop_bi,
                )

            # --- 2) Edge update. ----------------------------------------
            src, dst = edge_index
            he_delta = self.edge_update(
                torch.cat([he, h_after_proc[src], h_after_proc[dst]], dim=-1)
            )
            he_after = he_prev + self.edge_update_norm(he_delta)

            # --- 3) Timestep injection. ---------------------------------
            t_emb = self.timestep_proj(self.timestep_emb(k))  # [d]
            h_post = h_after_proc + t_emb.unsqueeze(0).expand_as(h_after_proc)

            # --- 4) Per-node gated residual. ----------------------------
            # The gate's iter_embedding has size num_iterations from config;
            # clamp the index in the (rare) eval-time case K > that bound.
            iter_idx_clamped = min(iter_idx, self.gate.num_iterations - 1)
            h, g_k, gate_logit_k = self.gate(
                h_prev=h_prev,
                h_new=h_post,
                iteration=iter_idx_clamped,
                batch=batch,
                edge_index=edge_index,
                num_nodes=num_nodes,
            )
            # Edge embeddings get the same gate per-edge (mean of endpoint
            # gates) so they don't desync from h.
            edge_gate = 0.5 * (g_k[src] + g_k[dst]).unsqueeze(-1)
            he = (1.0 - edge_gate) * he_prev + edge_gate * he_after

            gate_per_iter.append(g_k)
            gate_logit_per_iter.append(gate_logit_k)

            # --- 5) Per-iteration prediction heads (GNN state only). ----
            # Global/Perceiver blocks run ONCE after the full loop (step 8)
            # on the final hidden state. Intermediate iterations use h
            # directly so the transformer is not repeated K times.
            x_k = self.node_head(h, he, edge_index)  # [N, 2]
            r_k = self.edge_head(he)                   # [E, 2]
            if return_all:
                node_preds.append(x_k)
                edge_preds.append(r_k)

            # --- 7) Halt module (Variant B only). -----------------------
            if self.loop_pondernet_enabled:
                halt_logit_k = self.halt_module(
                    h,
                    iteration=iter_idx_clamped,
                    batch=batch,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    gate_per_node=g_k.detach(),
                )
                halt_logits_per_iter.append(halt_logit_k)

        # --- 8) Post-loop transformer pass (runs ONCE). ----------------
        # Run global/Perceiver blocks on the final GNN hidden state to
        # produce the transformer-refined final prediction, replacing the
        # last per-iteration prediction (k=K) which was based on raw h.
        if self._global_readout_block_indices and return_all and node_preds:
            if (
                self.training
                and self.use_activation_checkpointing
                and not self._has_recurrent_global_memory
            ):
                _readout_bi = self._global_readout_block_indices
                h_final = checkpoint(
                    lambda h_in, he_in, _bi=_readout_bi: (
                        self._run_processor_once(
                            h_in, he_in, edge_index, data, None, K, _bi,
                        )[0]
                    ),
                    h, he, use_reentrant=False,
                )
            else:
                h_final, _ = self._run_processor_once(
                    h, he, edge_index, data, None, K,
                    self._global_readout_block_indices,
                )
            node_preds[-1] = self.node_head(h_final, he, edge_index)
            edge_preds[-1] = self.edge_head(he)

        # Stash gate aux for the loss / analysis paths.
        self._last_gate_aux = {
            "gate_per_iter": torch.stack(gate_per_iter, dim=0),
            "gate_logit_per_iter": torch.stack(gate_logit_per_iter, dim=0),
            "gate_activation": self.gate.activation,
            "node_batch": node_batch_g,
        }

        if self.loop_pondernet_enabled:
            # Build halt distribution p_k over k in fp32 for stability.
            halt_logits = torch.stack(halt_logits_per_iter, dim=0).float()  # [K, G]
            if self.loop_halt_distribution == "pondernet_geometric":
                halt_probs_list: List[torch.Tensor] = []
                remainders = torch.ones(
                    num_graphs_g, device=h.device, dtype=torch.float32
                )
                for k in range(K):
                    if k == K - 1:
                        lam_k = torch.ones_like(halt_logits[k])
                    else:
                        lam_k = torch.sigmoid(halt_logits[k])
                    p_k = remainders * lam_k
                    halt_probs_list.append(p_k)
                    remainders = remainders * (1.0 - lam_k)
                halt_probs = torch.stack(halt_probs_list, dim=0)
            else:  # 'softmax_over_k'
                halt_probs = torch.softmax(halt_logits, dim=0)

            pred_per_iter_t = torch.stack(node_preds, dim=0).float()  # [K, N, 2]
            self._last_ponder_aux = {
                "halt_probs": halt_probs,
                "halt_logits": halt_logits,
                "pred_per_iter": pred_per_iter_t,
                "node_batch": node_batch_g,
                "distribution": self.loop_halt_distribution,
            }
            if self.training:
                self._ponder_step += 1

        return (
            node_preds,
            edge_preds,
            gate_per_iter,
            gate_logit_per_iter,
            halt_logits_per_iter,
        )

    @torch.no_grad()
    def predict_physical(self, data, K: int) -> Tuple[torch.Tensor, int]:
        """Run inference and return positions in physical µm + K_used."""
        self.eval()
        node_preds, _, _, _, _ = self.forward(data, K=K, return_all=True)
        if self.loop_pondernet_enabled and self._last_ponder_aux is not None:
            if self.loop_halt_inference_mode == "soft":
                aux = self._last_ponder_aux
                p_k_node = aux["halt_probs"][:, aux["node_batch"]]  # [K, N]
                stacked = torch.stack(node_preds, dim=0).float()
                x_K = (p_k_node.unsqueeze(-1) * stacked).sum(dim=0).to(stacked.dtype)
                K_used = K
            else:  # 'hard'
                aux = self._last_ponder_aux
                cum = aux["halt_probs"].cumsum(dim=0)  # [K, G]
                thresh = self.loop_halt_inference_threshold
                # First k where cumulative halt prob crosses threshold,
                # mean-pooled across graphs in the batch.
                first_hit = (cum >= thresh).float().argmax(dim=0)  # [G]
                K_used = int(first_hit.float().mean().clamp_min(self.min_k_inference).item()) + 1
                K_used = min(K_used, K)
                x_K = node_preds[K_used - 1]
        else:
            x_K = node_preds[-1]
            K_used = K
        return x_K * self.h_row, K_used
