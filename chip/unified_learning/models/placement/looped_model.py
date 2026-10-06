"""
Looped Placement Regression Model.

Architecture (from looped_placement_impl.md §2):
    [Encoder]   — runs ONCE, produces h^(0) and he^(0)
    [Processor] — shared weights, called K times
                  each step: node GNN update + edge update + timestep injection
    [Heads]     — shared weights, called at each k to produce x^(k), r^(k), c^(k)

Design constraints respected:
    - Encoder runs once — not per iteration.
    - Processor weights SHARED across all K iterations (same nn.Module called K times).
    - All prediction heads SHARED across iterations.
    - x^(k) is a direct position prediction, NOT a displacement from x^(k-1).
    - Previous positions are NOT fed back into the processor.
    - Edge embeddings he flow through and are updated each iteration.
    - Timestep sinusoidal embedding of k injected at each processor step.
    - PreNorm throughout (norm before each sub-block, not after residual).
    - Residual updates with per-node norm clipping.

Coordinate convention (§1 of spec):
    All internal values are in row-height units: value / h_row.
    h_row is set in config (technology constant, default 1.0).
    With h_row=1.0 the model works with whatever normalisation the
    data loader already applies (backward-compatible with existing pipeline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, List, Optional, Tuple

from ..encoders.mlp import MLPEncoder
from ..model import _build_gnn_layers, _gnn_layers_for_processor_block
from .timestep import TimestepEmbedding
from .heads import NodePredictionHead, EdgePredictionHead, ConvergenceHead


class LoopedPlacementModel(nn.Module):
    """
    Looped GNN model for standard-cell placement regression.

    Config keys (all under a flat dict):

        node_in_dim:            int   — input node feature dimension (no coord channels)
        edge_in_dim:            int   — input edge feature dimension (default 4, pin offsets)
        hidden_dim:             int   — model width d (default 256)
        h_row:                  float — row height in µm; divide by this to get row-height units
                                        set to 1.0 to work with existing canvas normalisation
        encoder_layers:         int   — MLP encoder depth (default 3)
        encoder_dropout:        float — MLP encoder dropout (default 0.0)
        max_k:                  int   — max loop iterations for TimestepEmbedding (default 256)
        residual_clip_tau:      float — per-node residual norm clip threshold (default 1.0)
        convergence_threshold:  float — c score above which early exit fires (default 0.95)
        min_k_inference:        int   — always run at least this many iterations (default 4)
        processor:              dict  — processor config (same format as UnifiedModel processor,
                                        type must be 'gnn'; edge_dim must equal hidden_dim)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        hidden_dim: int = int(config["hidden_dim"])
        node_in_dim: int = int(config["node_in_dim"])
        edge_in_dim: int = int(config.get("edge_in_dim", 4))

        self.hidden_dim = hidden_dim
        self.h_row: float = float(config.get("h_row", 1.0))
        self.convergence_threshold: float = float(config.get("convergence_threshold", 0.95))
        self.min_k_inference: int = int(config.get("min_k_inference", 4))
        self.residual_clip_tau: float = float(config.get("residual_clip_tau", 1.0))
        self.use_activation_checkpointing: bool = bool(
            config.get("use_activation_checkpointing", False)
        )

        # ------------------------------------------------------------------ #
        # Encoder — runs ONCE before the loop                                 #
        # ------------------------------------------------------------------ #
        self.node_encoder = MLPEncoder(
            input_dim=node_in_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=int(config.get("encoder_layers", 3)),
            dropout=float(config.get("encoder_dropout", 0.0)),
            use_layer_norm=True,
        )
        # Edge encoder: project raw pin-offset edge features to hidden_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ------------------------------------------------------------------ #
        # Processor — shared weights, called K times                          #
        # ------------------------------------------------------------------ #
        processor_cfg: Dict[str, Any] = config["processor"]
        if processor_cfg.get("type") != "gnn":
            raise ValueError(
                "LoopedPlacementModel requires processor.type='gnn'. "
                f"Got: {processor_cfg.get('type')}"
            )
        # edge_dim in processor must equal hidden_dim so that edge embeddings
        # (he, [E, hidden_dim]) can be passed as edge_attr each iteration.
        if int(processor_cfg.get("edge_dim", hidden_dim)) != hidden_dim:
            raise ValueError(
                f"processor.edge_dim must equal hidden_dim ({hidden_dim}) "
                f"so that he can be passed as edge_attr. "
                f"Got edge_dim={processor_cfg.get('edge_dim')}."
            )

        _result = _build_gnn_layers(processor_cfg, hidden_dim, time_embed_dim=None)
        if len(_result) == 4:
            self.gnn_blocks, self.global_modules, self.num_blocks, self.layers_per_block = _result
            self.gated_adapters = None
            self.gnn_layer_ranges = None
        elif len(_result) == 5:
            self.gnn_blocks, self.global_modules, self.num_blocks, self.layers_per_block, self.gated_adapters = _result
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
            and any(hasattr(module, "forward_with_memory") for module in self.global_modules)
        )

        # Edge update MLP — shared across K iterations (spec §2.3 step 6)
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update_norm = nn.LayerNorm(hidden_dim)

        # Timestep embedding — injects iteration index k into h (spec §4.5)
        self.timestep_emb = TimestepEmbedding(
            hidden_dim=hidden_dim,
            max_k=int(config.get("max_k", 256)),
        )
        self.timestep_proj = nn.Linear(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # Prediction heads — shared across K iterations                       #
        # ------------------------------------------------------------------ #
        self.node_head = NodePredictionHead(hidden_dim)
        self.edge_head = EdgePredictionHead(hidden_dim)
        self.conv_head = ConvergenceHead(hidden_dim)

        # Small init for prediction heads (spec §7)
        self._small_init(self.node_head)
        self._small_init(self.edge_head)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _small_init(module: nn.Module):
        """Xavier-uniform with gain=0.1 for linear layers — keeps early predictions small."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _clipped_residual(self, h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """h + delta with per-node norm clipping (spec §4.4)."""
        tau = self.residual_clip_tau
        dnorm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        delta = delta * (tau / dnorm).clamp(max=1.0)
        return h + delta

    def _run_processor_once(
        self,
        h: torch.Tensor,
        he: torch.Tensor,
        edge_index: torch.Tensor,
        data,
        memory_states: Optional[List[Optional[torch.Tensor]]] = None,
        step_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]]]:
        """
        One pass of the shared-weight GNN processor blocks.

        Passes `he` (current edge embeddings) as `edge_attr` so that GATv2
        inside UnifiedGNNLayer uses them for message passing.
        Returns updated h and optional per-block recurrent memory states
        (he is NOT updated here — caller does that).
        """
        updated_memory_states = list(memory_states) if memory_states is not None else None
        if self.num_blocks is not None:
            # Block-based: num_blocks × layers_per_block
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
                        h = self._clipped_residual(h, h_new)
                    else:
                        h = torch.relu(h_new)

                if self.global_modules is not None:
                    global_mod = self.global_modules[block_idx]
                    inst_batch = getattr(data, "batch", None)
                    if hasattr(global_mod, "forward_with_memory"):
                        current_state = None if updated_memory_states is None else updated_memory_states[block_idx]
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
            # Flat: iterate over all GNN layers
            for i, block in enumerate(self.gnn_blocks):
                gnn_data = {"edge_index": edge_index, "edge_attr": he}
                h_new = block(gnn_data, x=h)
                if self._use_residual:
                    h = self._clipped_residual(h, h_new)
                else:
                    h = torch.relu(h_new)

                if self.global_modules is not None:
                    global_mod = self.global_modules[i]
                    inst_batch = getattr(data, "batch", None)
                    if hasattr(global_mod, "forward_with_memory"):
                        current_state = None if updated_memory_states is None else updated_memory_states[i]
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

        memory_states: List[Optional[torch.Tensor]] = []
        for module in self.global_modules:
            if hasattr(module, "init_loop_state"):
                memory_states.append(
                    module.init_loop_state(
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                memory_states.append(None)
        return memory_states

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        data,
        K: int,
        return_all: bool = True,
        loss_window: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]] | torch.Tensor:
        """
        Args:
            data:        PyG Data with .x [N, node_in_dim], .edge_index [2, E],
                         .edge_attr [E, edge_in_dim], optionally .batch [N].
            K:           Number of processor iterations.
            return_all:  If True return (node_preds, edge_preds, conv_scores)
                         lists of length ≤ K.  If False return only the final x^(K).
            loss_window: If set during training, only the last `loss_window`
                         iterations produce predictions and participate in the
                         backward pass.  Earlier iterations run under
                         torch.no_grad() and are detached (truncated BPTT).
                         Ignored during eval or when K ≤ loss_window.

        Returns (if return_all=True):
            node_preds:  List[Tensor[N,2]]  — absolute positions in row-height units
            edge_preds:  List[Tensor[E,2]]  — pairwise displacements (aux loss)
            conv_scores: List[Tensor]       — convergence scores, scalar or [B]

        Returns (if return_all=False):
            x_K: Tensor[N,2] — final positions in row-height units
        """
        edge_index = data.edge_index  # [2, E]
        batch = getattr(data, "batch", None)

        # ---- Encode once ---- #
        h = self.node_encoder(data.x)            # [N, d]
        he = self.edge_encoder(data.edge_attr)   # [E, d]
        batch_size = int(batch.max().item()) + 1 if batch is not None else 1
        memory_states = self._init_memory_states(
            batch_size=batch_size,
            device=h.device,
            dtype=h.dtype,
        )

        node_preds: List[torch.Tensor] = []
        edge_preds: List[torch.Tensor] = []
        conv_scores: List[torch.Tensor] = []

        # Gradient window: only the last loss_window steps track gradients.
        pred_start_k = 1
        if self.training and loss_window is not None and 0 < loss_window < K:
            pred_start_k = K - loss_window + 1

        for k in range(1, K + 1):
            # --- Warmup phase: no gradient, no heads ---
            if k < pred_start_k:
                with torch.no_grad():
                    h_new, memory_states = self._run_processor_once(
                        h, he, edge_index, data, memory_states, k,
                    )
                    h = h_new
                    src, dst = edge_index
                    he_delta = self.edge_update(torch.cat([he, h[src], h[dst]], dim=-1))
                    he = he + self.edge_update_norm(he_delta)
                    t_emb = self.timestep_proj(self.timestep_emb(k))
                    h = self._clipped_residual(h, t_emb.unsqueeze(0).expand_as(h))
                continue

            # --- Detach at boundary to start a fresh computation graph ---
            if k == pred_start_k and pred_start_k > 1:
                h = h.detach().requires_grad_()
                he = he.detach().requires_grad_()
                if memory_states is not None:
                    memory_states = [
                        s.detach().requires_grad_() if s is not None else None
                        for s in memory_states
                    ]

            # 1. Node update via shared-weight GNN processor
            if self.training and self.use_activation_checkpointing and not self._has_recurrent_global_memory:
                h = checkpoint(
                    lambda h_in, he_in, _k=k: self._run_processor_once(
                        h_in,
                        he_in,
                        edge_index,
                        data,
                        None,
                        _k,
                    )[0],
                    h,
                    he,
                    use_reentrant=False,
                )
            else:
                h, memory_states = self._run_processor_once(
                    h,
                    he,
                    edge_index,
                    data,
                    memory_states,
                    k,
                )

            # 2. Edge update: cat(he, h[src], h[dst]) → delta he
            src, dst = edge_index
            he_delta = self.edge_update(torch.cat([he, h[src], h[dst]], dim=-1))
            he = he + self.edge_update_norm(he_delta)

            # 3. Timestep injection
            t_emb = self.timestep_proj(self.timestep_emb(k))  # [d]
            h = self._clipped_residual(h, t_emb.unsqueeze(0).expand_as(h))

            # 4. Shared prediction heads
            x_k = self.node_head(h, he, edge_index)   # [N, 2] row-height units
            r_k = self.edge_head(he)                   # [E, 2]
            c_k = self.conv_head(h, batch)             # scalar or [B] logits

            if return_all:
                node_preds.append(x_k)
                edge_preds.append(r_k)
                conv_scores.append(c_k)

            # Early exit at inference (not during training)
            if not self.training and k >= self.min_k_inference:
                c_prob = torch.sigmoid(c_k)
                c_val = c_prob.mean().item() if c_prob.dim() > 0 else c_prob.item()
                if c_val > self.convergence_threshold:
                    break

        if return_all:
            return node_preds, edge_preds, conv_scores
        return x_k  # noqa: F821  (always assigned since K >= 1)

    def predict_physical(self, data, K: int) -> Tuple[torch.Tensor, int]:
        """
        Run inference and return positions in physical µm.

        Returns:
            positions: [N, 2] in physical µm
            K_used:    actual iterations used (may be < K if converged early)
        """
        self.eval()
        with torch.no_grad():
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)

            h = self.node_encoder(data.x)
            he = self.edge_encoder(data.edge_attr)
            batch_size = int(batch.max().item()) + 1 if batch is not None else 1
            memory_states = self._init_memory_states(
                batch_size=batch_size,
                device=h.device,
                dtype=h.dtype,
            )

            x_k = None
            k_used = 0
            for k in range(1, K + 1):
                h, memory_states = self._run_processor_once(
                    h,
                    he,
                    edge_index,
                    data,
                    memory_states,
                    k,
                )
                src, dst = edge_index
                he_delta = self.edge_update(torch.cat([he, h[src], h[dst]], dim=-1))
                he = he + self.edge_update_norm(he_delta)
                t_emb = self.timestep_proj(self.timestep_emb(k))
                h = self._clipped_residual(h, t_emb.unsqueeze(0).expand_as(h))

                x_k = self.node_head(h, he, edge_index)
                c_k = self.conv_head(h, batch)
                k_used = k

                if k >= self.min_k_inference:
                    c_prob = torch.sigmoid(c_k)
                    c_val = c_prob.mean().item() if c_prob.dim() > 0 else c_prob.item()
                    if c_val > self.convergence_threshold:
                        break

            return x_k * self.h_row, k_used
