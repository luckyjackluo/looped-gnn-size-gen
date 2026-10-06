"""Iteration- and size-conditioned FiLM adapter for looped-processor finetuning.

When a pretrained frozen processor is unrolled for K iterations on larger graphs,
this module produces a tiny per-(iteration, graph-size) FiLM modulation that is
applied to the hidden state before each iteration. All backbone parameters stay
frozen; only this adapter (plus optionally the decoder) trains.

At init, the FiLM output is zero so each iteration starts as identity: the model
exactly reproduces the frozen pretrained behavior at step 0, and training learns
a small per-step steering correction conditioned on (iteration, graph size).

PonderNet-style halting (optional, ``halting_enabled=True``):
    Adds a tiny per-graph halt head that, given the same conditioning vector
    ``cond_k(graph)`` plus a (zero-init) projection of the *mean-pooled current
    hidden state*, produces a halt logit ``halt_logit_k`` per graph. The model
    consumes these logits to build a PonderNet halting distribution
    ``p(halt = k | graph)`` over iterations and weights the per-iteration loss
    by it (see ``unified_learning/models/model.py`` and the regression
    pipeline). Initialisation puts ``init_halt_bias = -2.0`` on the halt-head
    bias and zeros on every halt-related weight, so:
      * at step 0 every graph runs all K iterations (sigmoid(-2) ~= 0.12 per
        step, with ``p_K = 1`` enforced at the last step),
      * the size MLP and zero-initialised state projection make halting
        explicitly conditioned on (iteration, graph size, content), so the
        adapter can learn "small graphs halt early, large/hard graphs keep
        iterating".

Parameter cost (hidden_dim=256, d_cond=64, K=4, with size input):
    iter_embedding     : K * d_cond                      =     256
    size_mlp           : 2*d_cond + d_cond*d_cond + bias =   4,352
    film (Linear)      : d_cond * (2*hidden_dim) + bias  =  33,280
    + halting (optional):
    halt_state_proj    : hidden_dim*d_cond + bias        =  16,448
    halt_head          : d_cond + bias                   =      65
    ----------------------------------------------------------------
    total (no halt)                                      ~  37,888
    total (with halt)                                    ~  54,401
"""
from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LoopFiLMConditioner(nn.Module):
    """Per-iteration FiLM adapter conditioned on (iteration index k, graph size).

    For each iteration ``k`` in ``{0, ..., K-1}``:

        cond_k(graph) = iter_embed[k] + size_mlp([log1p(N)/9.21, log1p(E)/9.21])
                        [+ time_proj(time_embed) when use_time_input]
        gamma, beta   = film(cond_k).chunk(2)
        h_modulated   = (1 + gamma) * h + beta

    The FiLM output linear is zero-initialised so the adapter starts as identity.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        use_time_input: bool = False,
        time_embedding_dim: int = 128,
        zero_init_film: bool = True,
        halting_enabled: bool = False,
        init_halt_bias: float = -2.0,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.conditioning_dim = conditioning_dim
        self.use_size_input = use_size_input
        self.use_time_input = bool(use_time_input)
        self.time_embedding_dim = int(time_embedding_dim)
        self.halting_enabled = bool(halting_enabled)

        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        if self.use_time_input:
            self.time_proj = nn.Linear(self.time_embedding_dim, conditioning_dim)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)
        else:
            self.time_proj = None

        self.film = nn.Linear(conditioning_dim, 2 * hidden_dim)
        if zero_init_film:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        # PonderNet halt head (optional). Zero-init the state projection so
        # the halt logit at step 0 is exactly init_halt_bias for every graph
        # (size-dependence is also zero-init via iter_embedding=0 and the
        # size_mlp output passing through a Linear that we do NOT zero-init,
        # but the halt_head Linear is zero-init so the only non-zero quantity
        # at step 0 is the bias). This makes the model start in
        # "always-run-K-iterations" mode and gradually learn to halt earlier.
        if self.halting_enabled:
            self.halt_state_proj = nn.Linear(hidden_dim, conditioning_dim)
            nn.init.zeros_(self.halt_state_proj.weight)
            nn.init.zeros_(self.halt_state_proj.bias)
            self.halt_head = nn.Linear(conditioning_dim, 1)
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, float(init_halt_bias))
        else:
            self.halt_state_proj = None
            self.halt_head = None

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (size_stats [G,2], node_batch [N], num_graphs)."""
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor([float(edge_index.shape[1])], device=device, dtype=dtype)
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            num_graphs = 1
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch, num_graphs

    def _mean_pool(
        self,
        h: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Mean-pool node features into per-graph features. Shape [G, hidden_dim]."""
        if num_graphs == 1:
            return h.mean(dim=0, keepdim=True)
        sum_h = torch.zeros(num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        sum_h.index_add_(0, node_batch, h)
        n_per_graph = torch.bincount(node_batch, minlength=num_graphs).to(
            device=h.device, dtype=h.dtype
        ).clamp_min(1.0)
        return sum_h / n_per_graph.unsqueeze(-1)

    def _build_conditioning(
        self,
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (per_node_cond, per_graph_cond, node_batch, num_graphs)."""
        device = edge_index.device
        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )
        if self.use_size_input:
            stats, node_batch, num_graphs = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_cond = self.size_mlp(stats)
            per_graph_cond = size_cond + iter_cond.unsqueeze(0)
            per_node_cond = per_graph_cond[node_batch]
        else:
            if batch is None:
                node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
                num_graphs = 1
            else:
                node_batch = batch
                num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            per_graph_cond = iter_cond.unsqueeze(0).expand(num_graphs, -1)
            per_node_cond = iter_cond.unsqueeze(0).expand(num_nodes, -1)

        if self.use_time_input:
            if time_embed is None:
                raise ValueError(
                    "LoopFiLMConditioner.use_time_input=True but time_embed is None"
                )
            if time_embed.shape[-1] != self.time_embedding_dim:
                raise ValueError(
                    f"time_embed dim {time_embed.shape[-1]} != "
                    f"time_embedding_dim {self.time_embedding_dim}"
                )
            time_node_cond = self.time_proj(time_embed)
            per_node_cond = per_node_cond + time_node_cond
            pooled_time = self._mean_pool(time_embed, node_batch, num_graphs)
            per_graph_cond = per_graph_cond + self.time_proj(pooled_time)

        return per_node_cond, per_graph_cond, node_batch, num_graphs

    def apply_film(
        self,
        h: torch.Tensor,
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """FiLM-modulate ``h``; return (modulated_h, per_graph_cond, per_node_cond, node_batch)."""
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        per_node_cond, per_graph_cond, node_batch, _num_graphs = self._build_conditioning(
            iteration,
            batch=batch,
            edge_index=edge_index,
            num_nodes=num_nodes,
            time_embed=time_embed,
        )
        scale_shift = self.film(per_node_cond)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        modulated_h = h * (1.0 + gamma) + beta
        return modulated_h, per_graph_cond, per_node_cond, node_batch

    def forward(
        self,
        h: torch.Tensor,
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FiLM-modulate ``h`` and (optionally) emit a per-graph halt logit.

        Returns:
            modulated_h: ``h`` after iteration- and size-conditioned FiLM.
            halt_logit:  ``[num_graphs]`` per-graph halt logit, or ``None`` if
                halting is disabled. Caller turns this into a stopping
                probability (PonderNet style).
        """
        modulated_h, per_graph_cond, _per_node_cond, node_batch = self.apply_film(
            h,
            iteration,
            batch=batch,
            edge_index=edge_index,
            num_nodes=num_nodes,
            time_embed=time_embed,
        )

        halt_logit: Optional[torch.Tensor] = None
        if self.halting_enabled:
            num_graphs = int(per_graph_cond.shape[0])
            pooled_h = self._mean_pool(h, node_batch, num_graphs)
            halt_input = per_graph_cond + self.halt_state_proj(pooled_h)
            halt_logit = self.halt_head(halt_input).squeeze(-1)

        return modulated_h, halt_logit


class LoopControllerMemoryBlock(nn.Module):
    """Per-node controller memory threaded across loop iterations.

    At iteration ``k``:
        cond_k' = cond_k + read_proj(memory_k)
        memory_{k+1} = memory_k + write_delta([memory_k, cond_k', ...])

    With ``zero_init=True`` (legacy), read/write start at identity-on-zero so step
    0 matches a memory-less controller — but read_proj.weight=0 blocks gradients
    to the write path (memory never learns). Use ``zero_init=False`` with small
    weight init and ``write_gain < 1`` for trainable but stable memory.

    Optional ``use_hidden_write`` appends a zero-init projection of the post-mix
    hidden state so each iteration can persist processor outcomes in memory.
    """

    def __init__(
        self,
        *,
        conditioning_dim: int,
        memory_dim: int,
        hidden_dim: Optional[int] = None,
        use_hidden_write: bool = False,
        zero_init: bool = True,
        write_gain: float = 1.0,
        read_gain: float = 1.0,
        memory_clip: Optional[float] = None,
    ):
        super().__init__()
        self.conditioning_dim = int(conditioning_dim)
        self.memory_dim = int(memory_dim)
        self.zero_init = bool(zero_init)
        self.write_gain = float(write_gain)
        self.read_gain = float(read_gain)
        self.memory_clip = float(memory_clip) if memory_clip is not None else None
        self.use_hidden_write = bool(use_hidden_write) and hidden_dim is not None
        if self.use_hidden_write and hidden_dim is None:
            raise ValueError("use_hidden_write=True requires hidden_dim")

        self.memory_init = nn.Parameter(torch.zeros(1, self.memory_dim))
        self.read_proj = nn.Linear(self.memory_dim, self.conditioning_dim)

        if self.use_hidden_write:
            self.write_h_proj = nn.Linear(int(hidden_dim), self.conditioning_dim)
            nn.init.zeros_(self.write_h_proj.weight)
            nn.init.zeros_(self.write_h_proj.bias)
        else:
            self.write_h_proj = None

        write_in = self.memory_dim + self.conditioning_dim
        if self.use_hidden_write:
            write_in += self.conditioning_dim
        self.write_mlp = nn.Sequential(
            nn.Linear(write_in, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
        )

        if self.zero_init:
            nn.init.zeros_(self.read_proj.weight)
            nn.init.zeros_(self.read_proj.bias)
            nn.init.zeros_(self.write_mlp[-1].weight)
            nn.init.zeros_(self.write_mlp[-1].bias)
        else:
            nn.init.normal_(self.read_proj.weight, std=0.01)
            nn.init.zeros_(self.read_proj.bias)
            nn.init.normal_(self.write_mlp[0].weight, std=0.01)
            nn.init.zeros_(self.write_mlp[0].bias)
            nn.init.normal_(self.write_mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.write_mlp[-1].bias)

    def _clip_memory(self, memory: torch.Tensor) -> torch.Tensor:
        if self.memory_clip is None:
            return memory
        return memory.clamp(-self.memory_clip, self.memory_clip)

    def init_memory(
        self,
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.memory_init.to(device=device, dtype=dtype).expand(num_nodes, -1)

    def read(self, memory: torch.Tensor) -> torch.Tensor:
        """Map prior memory to a conditioning offset ``[N, d_cond]``."""
        memory = self._clip_memory(memory)
        return self.read_gain * self.read_proj(memory)

    def write(
        self,
        memory: torch.Tensor,
        controller_state: torch.Tensor,
        *,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Persist the current controller state for the next iteration."""
        memory = self._clip_memory(memory)
        parts = [memory, controller_state]
        if self.write_h_proj is not None and hidden_state is not None:
            parts.append(self.write_h_proj(hidden_state))
        delta = self.write_mlp(torch.cat(parts, dim=-1))
        return memory + self.write_gain * delta


class LoopFiLMGatedOutputConditioner(LoopFiLMConditioner):
    """FiLM loop adapter with optional gated hidden mixing and velocity gates.

    Default FM path (``gated.decode_once=True``, ``gated.use_gated_hidden_mix=False``):
    K iterations of FiLM → processor with ``h <- h_proc`` (transformer-aligned), then
    a single ``decoder(h)`` after the loop (regression-style).

    Legacy path (``gated.decode_once=False``): per-iteration decode with gated
    ``v_acc`` mixing and optional gated hidden carry:

        h_{k+1}  = (1 - g_h) * h_k + g_h * h_tilde   when use_gated_hidden_mix
        v_acc    = (1 - g_v) * v_acc + g_v * v_tilde

    ``g_h`` / ``g_v`` are per-node sigmoids when ``node_wise_gates=True`` (default
    for new configs): each node uses the same conditioning vector as FiLM
    (``per_node_cond``), optionally plus a zero-init projection of the current
    hidden state ``h`` for node-specific variation. When ``node_wise_gates=False``,
    gates are one scalar per graph (legacy checkpoint behaviour).

    Optional ``LoopControllerMemoryBlock`` threads a per-node memory vector across
    iterations: iteration ``k`` reads ``memory_k`` into the conditioning vector,
    then writes ``memory_{k+1}`` from the current controller state.
    Mix FiLM layers are zero-init so they start as identity on the new branch;
    ``g -> 0`` then freezes ``h`` / ``v_acc`` at the previous level exactly.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        velocity_dim: int = 2,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        use_time_input: bool = False,
        time_embedding_dim: int = 128,
        zero_init_film: bool = True,
        use_mix_film: bool = True,
        init_gate_bias: float = 2.0,
        node_wise_gates: bool = False,
        gate_use_hidden_state: bool = True,
        use_controller_memory: bool = False,
        controller_memory_dim: Optional[int] = None,
        memory_zero_init: bool = True,
        memory_use_hidden_write: bool = False,
        memory_write_gain: float = 1.0,
        memory_read_gain: float = 1.0,
        memory_clip: Optional[float] = None,
        use_raw_input: bool = False,
        raw_input_dim: Optional[int] = None,
        use_transformer_prefilm: bool = False,
        transformer_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_iterations=num_iterations,
            conditioning_dim=conditioning_dim,
            use_size_input=use_size_input,
            use_time_input=use_time_input,
            time_embedding_dim=time_embedding_dim,
            zero_init_film=zero_init_film,
            halting_enabled=False,
        )
        self.velocity_dim = int(velocity_dim)
        self.use_mix_film = bool(use_mix_film)
        self.node_wise_gates = bool(node_wise_gates)
        self.gate_use_hidden_state = bool(gate_use_hidden_state)
        self.use_controller_memory = bool(use_controller_memory)
        self.use_raw_input = bool(use_raw_input)
        self.use_transformer_prefilm = bool(use_transformer_prefilm)

        if self.use_transformer_prefilm:
            if not self.use_raw_input or raw_input_dim is None:
                raise ValueError(
                    "use_transformer_prefilm=True requires use_raw_input=True and raw_input_dim"
                )
            if self.use_controller_memory:
                raise ValueError(
                    "use_transformer_prefilm=True is incompatible with use_controller_memory=True"
                )
            self.feat_mlp = None
            self.token_fusion = LoopTransformerConnector(
                hidden_dim=hidden_dim,
                raw_input_dim=int(raw_input_dim),
                num_iterations=num_iterations,
                transformer_dim=int(transformer_dim),
                num_heads=int(num_heads),
                ffn_dim=int(ffn_dim),
                conditioning_dim=int(conditioning_dim),
                use_size_input=use_size_input,
                use_time_input=use_time_input,
                time_embedding_dim=time_embedding_dim,
                zero_init_output=True,
                memory_clip=float(memory_clip) if memory_clip is not None else 8.0,
            )
            self.prefilm_film = nn.Linear(int(transformer_dim), 2 * hidden_dim)
            nn.init.zeros_(self.prefilm_film.weight)
            nn.init.zeros_(self.prefilm_film.bias)
            self.fusion_to_cond = nn.Linear(int(transformer_dim), conditioning_dim)
            nn.init.normal_(self.fusion_to_cond.weight, std=0.01)
            nn.init.zeros_(self.fusion_to_cond.bias)
        else:
            self.token_fusion = None
            self.prefilm_film = None
            self.fusion_to_cond = None

        if self.use_raw_input and not self.use_transformer_prefilm:
            if raw_input_dim is None:
                raise ValueError("use_raw_input=True requires raw_input_dim")
            self.feat_mlp = nn.Sequential(
                nn.Linear(int(raw_input_dim), conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.feat_mlp = None

        if self.use_transformer_prefilm:
            self.memory_block = None
        elif self.use_controller_memory:
            mem_dim = (
                int(controller_memory_dim)
                if controller_memory_dim is not None
                else int(conditioning_dim)
            )
            self.memory_block = LoopControllerMemoryBlock(
                conditioning_dim=int(conditioning_dim),
                memory_dim=mem_dim,
                hidden_dim=hidden_dim,
                use_hidden_write=bool(memory_use_hidden_write),
                zero_init=bool(memory_zero_init),
                write_gain=float(memory_write_gain),
                read_gain=float(memory_read_gain),
                memory_clip=memory_clip,
            )
        else:
            self.memory_block = None

        self.gate_h_head = nn.Linear(conditioning_dim, 1)
        self.gate_v_head = nn.Linear(conditioning_dim, 1)
        if self.node_wise_gates:
            nn.init.normal_(self.gate_h_head.weight, std=0.01)
            nn.init.normal_(self.gate_v_head.weight, std=0.01)
        else:
            nn.init.zeros_(self.gate_h_head.weight)
            nn.init.zeros_(self.gate_v_head.weight)
        nn.init.constant_(self.gate_h_head.bias, float(init_gate_bias))
        nn.init.constant_(self.gate_v_head.bias, float(init_gate_bias))

        if self.node_wise_gates and self.gate_use_hidden_state:
            self.gate_h_state_proj = nn.Linear(hidden_dim, conditioning_dim)
            self.gate_v_state_proj = nn.Linear(hidden_dim, conditioning_dim)
            nn.init.zeros_(self.gate_h_state_proj.weight)
            nn.init.zeros_(self.gate_h_state_proj.bias)
            nn.init.zeros_(self.gate_v_state_proj.weight)
            nn.init.zeros_(self.gate_v_state_proj.bias)
        else:
            self.gate_h_state_proj = None
            self.gate_v_state_proj = None

        if self.use_mix_film:
            self.h_mix_film = nn.Linear(conditioning_dim, 2 * hidden_dim)
            self.v_mix_film = nn.Linear(conditioning_dim, 2 * self.velocity_dim)
            nn.init.zeros_(self.h_mix_film.weight)
            nn.init.zeros_(self.h_mix_film.bias)
            nn.init.zeros_(self.v_mix_film.weight)
            nn.init.zeros_(self.v_mix_film.bias)
        else:
            self.h_mix_film = None
            self.v_mix_film = None

    def _apply_raw_feature_cond(
        self,
        per_node_cond: torch.Tensor,
        per_graph_cond: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
        raw_x: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.feat_mlp is None or raw_x is None:
            return per_node_cond, per_graph_cond
        feat_cond = self.feat_mlp(raw_x)
        per_node_cond = per_node_cond + feat_cond
        per_graph_cond = per_graph_cond + self._mean_pool(
            feat_cond, node_batch, num_graphs
        )
        return per_node_cond, per_graph_cond

    def _mean_gate_per_graph(
        self,
        gate_node: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Reduce per-node gates ``[N]`` to per-graph means ``[G]`` for logging."""
        if gate_node.dim() == 2:
            gate_node = gate_node.squeeze(-1)
        if num_graphs == 1:
            return gate_node.mean().reshape(1)
        sum_g = torch.zeros(num_graphs, device=gate_node.device, dtype=gate_node.dtype)
        sum_g.index_add_(0, node_batch, gate_node)
        counts = torch.bincount(node_batch, minlength=num_graphs).to(
            device=gate_node.device, dtype=gate_node.dtype
        ).clamp_min(1.0)
        return sum_g / counts

    def _compute_gate(
        self,
        per_node_cond: torch.Tensor,
        per_graph_cond: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
        head: nn.Linear,
        state_proj: Optional[nn.Linear],
        h_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (gate_node [N,1], gate_graph_mean [G]) from controller conditioning."""
        if self.node_wise_gates:
            gate_input = per_node_cond
            if (
                self.gate_use_hidden_state
                and state_proj is not None
                and h_state is not None
            ):
                gate_input = gate_input + state_proj(h_state)
            gate_node = torch.sigmoid(head(gate_input))
            gate_graph = self._mean_gate_per_graph(gate_node, node_batch, num_graphs)
            return gate_node, gate_graph

        gate_graph = torch.sigmoid(head(per_graph_cond).squeeze(-1))
        gate_node = gate_graph[node_batch].unsqueeze(-1)
        return gate_node, gate_graph

    def _apply_mix_film(
        self,
        x: torch.Tensor,
        per_node_cond: torch.Tensor,
        film_linear: nn.Linear,
    ) -> torch.Tensor:
        scale_shift = film_linear(per_node_cond)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def transform_hidden_for_mix(
        self,
        h_new: torch.Tensor,
        per_node_cond: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_mix_film or self.h_mix_film is None:
            return h_new
        return self._apply_mix_film(h_new, per_node_cond, self.h_mix_film)

    def transform_velocity_for_mix(
        self,
        v_new: torch.Tensor,
        per_node_cond: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_mix_film or self.v_mix_film is None:
            return v_new
        return self._apply_mix_film(v_new, per_node_cond, self.v_mix_film)

    def mix_hidden(
        self,
        h_prev: torch.Tensor,
        h_new: torch.Tensor,
        per_node_cond: torch.Tensor,
        per_graph_cond: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_tilde = self.transform_hidden_for_mix(h_new, per_node_cond, node_batch)
        _gate_node, gate_graph = self._compute_gate(
            per_node_cond,
            per_graph_cond,
            node_batch,
            num_graphs,
            self.gate_h_head,
            self.gate_h_state_proj,
            h_prev,
        )
        gate_node = _gate_node
        blended = (1.0 - gate_node) * h_prev + gate_node * h_tilde
        return blended, gate_graph

    def mix_velocity(
        self,
        v_prev: Optional[torch.Tensor],
        v_new: torch.Tensor,
        per_node_cond: torch.Tensor,
        per_graph_cond: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
        h_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        v_tilde = self.transform_velocity_for_mix(v_new, per_node_cond, node_batch)
        if v_prev is None:
            return v_tilde, None
        _gate_node, gate_graph = self._compute_gate(
            per_node_cond,
            per_graph_cond,
            node_batch,
            num_graphs,
            self.gate_v_head,
            self.gate_v_state_proj,
            h_state,
        )
        gate_node = _gate_node
        blended = (1.0 - gate_node) * v_prev + gate_node * v_tilde
        return blended, gate_graph

    def apply_film(
        self,
        h: torch.Tensor,
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_embed: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        raw_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """FiLM-modulate ``h``; optionally read threaded controller memory."""
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        per_node_cond, per_graph_cond, node_batch, num_graphs = self._build_conditioning(
            iteration,
            batch=batch,
            edge_index=edge_index,
            num_nodes=num_nodes,
            time_embed=time_embed,
        )

        if self.use_transformer_prefilm:
            if raw_x is None:
                raise ValueError("use_transformer_prefilm=True requires raw_x")
            assert self.token_fusion is not None and self.prefilm_film is not None
            h_in = h
            if not torch.isfinite(h_in).all():
                h_in = torch.nan_to_num(h_in, nan=0.0, posinf=1e4, neginf=-1e4)
            emb_out, memory, _ = self.token_fusion.fuse_tokens(
                h_in,
                raw_x,
                memory,
                iteration,
                batch=batch,
                edge_index=edge_index,
                num_nodes=num_nodes,
                time_embed=time_embed,
            )
            assert self.fusion_to_cond is not None
            per_node_cond = self.fusion_to_cond(emb_out)
            per_graph_cond = self._mean_pool(per_node_cond, node_batch, num_graphs)
            scale_shift = self.prefilm_film(emb_out)
        else:
            per_node_cond, per_graph_cond = self._apply_raw_feature_cond(
                per_node_cond,
                per_graph_cond,
                node_batch,
                num_graphs,
                raw_x,
            )
            if self.memory_block is not None:
                if memory is None:
                    memory = self.memory_block.init_memory(
                        num_nodes, device=h.device, dtype=h.dtype
                    )
                per_node_cond = per_node_cond + self.memory_block.read(memory)
            scale_shift = self.film(per_node_cond)

        gamma, beta = scale_shift.chunk(2, dim=-1)
        modulated_h = h * (1.0 + gamma) + beta
        return modulated_h, per_graph_cond, per_node_cond, node_batch, memory

    def step_memory(
        self,
        memory: Optional[torch.Tensor],
        controller_state: torch.Tensor,
        *,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Write the current controller state; return memory for iteration k+1."""
        memory_block = getattr(self, "memory_block", None)
        if memory_block is None or memory is None:
            return memory
        return memory_block.write(
            memory,
            controller_state,
            hidden_state=hidden_state,
        )


class LoopSteerConditioner(nn.Module):
    """Per-(iteration, block) FiLM + gated residual for looped-processor PEFT.

    Stronger variant of ``LoopFiLMConditioner``. For each iteration ``k`` AND each
    processor block ``b``:

        cond_{k,b}(g) = iter_embed[k] + block_embed[b]
                      + size_mlp([log1p(N)/9.21, log1p(E)/9.21])
        gamma, beta   = film(cond_{k,b}).chunk(2)               # pre-block FiLM
        g             = sigmoid(gate_head(cond_{k,b}))           # per-graph gate
        h_pre         = (1 + gamma) * h + beta
        h_block       = block(h_pre)                             # run the block
        h_next        = (1 - g) * h + g * h_block                # gated skip

    The gate ``g`` is a per-graph scalar so each graph in a batch can independently
    choose whether block ``b`` on iteration ``k`` is (soft-)executed (``g=1``),
    skipped (``g=0``), or partially applied.

    Initialisation:
      * ``film`` zero-init  => FiLM is identity                  (no steering)
      * ``gate_head.bias = init_gate_bias`` (default ``+4``)      => sigmoid ~ 0.98
        (near full execution) so the loop starts as *vanilla K-pass behaviour*
        and training learns when to skip/steer.

    The FiLM and gate linears are shared across all (k, b); they differentiate
    only through the sum of the three embeddings fed in as ``cond``. This keeps
    the parameter footprint essentially identical to ``LoopFiLMConditioner``:

        (hidden_dim=256, d_cond=64, K=2, num_blocks=3)
        iter_embedding         : K * d_cond              =   128
        block_embedding        : L * d_cond              =   192
        size_mlp               : 2*d_cond + d_cond^2 + b =  4,416
        film  (Linear)         : d_cond * 2*hidden_dim   = 33,280
        gate_head (Linear)     : d_cond * 1 + 1          =    65
        ---------------------------------------------------------
        total                                            ~38,081
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        num_blocks: int,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        zero_init_film: bool = True,
        init_gate_bias: float = 4.0,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.num_blocks = num_blocks
        self.conditioning_dim = conditioning_dim
        self.use_size_input = use_size_input

        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        self.block_embedding = nn.Embedding(num_blocks, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)
        nn.init.zeros_(self.block_embedding.weight)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        self.film = nn.Linear(conditioning_dim, 2 * hidden_dim)
        if zero_init_film:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        self.gate_head = nn.Linear(conditioning_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(init_gate_bias))

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor([float(edge_index.shape[1])], device=device, dtype=dtype)
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch

    def _per_graph_cond(
        self,
        iteration: int,
        block_idx: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        if not 0 <= block_idx < self.num_blocks:
            raise ValueError(
                f"block_idx {block_idx} out of range [0, {self.num_blocks})"
            )
        device = edge_index.device
        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )
        block_cond = self.block_embedding(
            torch.tensor(block_idx, device=device, dtype=torch.long)
        )
        if self.use_size_input:
            stats, node_batch = self._graph_size_features(batch, edge_index, num_nodes)
            size_cond = self.size_mlp(stats)
            per_graph = size_cond + iter_cond.unsqueeze(0) + block_cond.unsqueeze(0)
        else:
            per_graph = (iter_cond + block_cond).unsqueeze(0)
            if batch is None:
                node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            else:
                node_batch = batch
        return per_graph, node_batch

    def pre_block(
        self,
        h: torch.Tensor,
        *,
        iteration: int,
        block_idx: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """FiLM-modulate the hidden state before block ``block_idx`` at iteration ``k``."""
        per_graph, node_batch = self._per_graph_cond(
            iteration, block_idx, batch, edge_index, num_nodes
        )
        per_node = per_graph[node_batch]
        gamma, beta = self.film(per_node).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def post_block(
        self,
        h_orig: torch.Tensor,
        h_new: torch.Tensor,
        *,
        iteration: int,
        block_idx: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blend h_orig (skip) and h_new (execute) via a per-graph sigmoid gate.

        Returns (blended_h, per_graph_gate) so callers can log or regularise the
        gate values (useful for overthinking ablations and analysis figures).
        """
        per_graph, node_batch = self._per_graph_cond(
            iteration, block_idx, batch, edge_index, num_nodes
        )
        gate_logit = self.gate_head(per_graph).squeeze(-1)
        gate_graph = torch.sigmoid(gate_logit)
        gate_node = gate_graph[node_batch].unsqueeze(-1)
        blended = (1.0 - gate_node) * h_orig + gate_node * h_new
        return blended, gate_graph


class LoopTransformerConnector(nn.Module):
    """Stronger cross-iteration connector for looped-processor PEFT.

    Replaces the simple per-iteration FiLM head with a small model that:

    1. Encodes raw input node features via a dedicated ``feat_mlp``.
    2. Encodes the frozen-processor output embedding via a separate ``emb_mlp``.
    3. Forms a conditioning token from (iteration index, graph size, optional time).
    4. Maintains a **per-node memory token** threaded across iterations so that
       the connector at step k has full awareness of all prior steering decisions.

    For each iteration k, the four tokens are stacked as:

        tokens = [feat_tok, emb_tok, cond_tok, memory_tok]   # [N, 4, d_model]

    A single pre-LN Transformer block (multi-head self-attention + FFN) processes
    each node's T=4 token sequence independently (cost O(N·T²), not O(N²)).

    Outputs:
        ``modulated_h``: ``emb_tok`` output → zero-init FiLM proj →
                         ``h' = h*(1+γ) + β``  (identity at init)
        ``memory_out``:  ``memory_tok`` output → threaded to iteration k+1

    At ``k=0`` the memory is initialised from a learned ``memory_init`` parameter
    (zero-init, broadcast to ``[N, d_model]``), so the connector starts as
    identity and training gradually learns to exploit the memory channel.

    Parameter budget (hidden_dim=256, d_model=128, ffn_dim=256, K=8):
        feat_mlp    (input_dim→128→128 + LN)  ~50k
        emb_mlp     (256→128→128 + LN)        ~50k
        iter_embed  K×128                        1k
        size_mlp    2→64→128                    8.7k
        memory_init 1×128                       0.1k
        MHA         128, 4 heads               ~66k
        FFN         128→256→128                ~66k
        LayerNorms  ×2                          0.5k
        FiLM proj   128→512 (zero-init)        ~66k
        ─────────────────────────────────────  ≈309k
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        raw_input_dim: int,
        num_iterations: int,
        transformer_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        use_time_input: bool = False,
        time_embedding_dim: int = 128,
        zero_init_output: bool = True,
        memory_clip: Optional[float] = 8.0,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if transformer_dim % num_heads != 0:
            raise ValueError(
                f"transformer_dim ({transformer_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.transformer_dim = transformer_dim
        self.use_size_input = use_size_input
        self.use_time_input = bool(use_time_input)
        self.time_embedding_dim = int(time_embedding_dim)
        self.memory_clip = float(memory_clip) if memory_clip is not None else None

        # Separate MLP encoders for the two input streams
        self.feat_mlp = nn.Sequential(
            nn.Linear(raw_input_dim, transformer_dim),
            nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        self.emb_mlp = nn.Sequential(
            nn.Linear(hidden_dim, transformer_dim),
            nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )

        # Iteration embedding
        self.iter_embedding = nn.Embedding(num_iterations, transformer_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        # Graph-size conditioning MLP (optional)
        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, transformer_dim),
            )
        else:
            self.size_mlp = None

        if self.use_time_input:
            self.time_proj = nn.Linear(self.time_embedding_dim, transformer_dim)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)
        else:
            self.time_proj = None

        # Learnable memory initialisation (broadcast to [N, d_model] at k=0)
        self.memory_init = nn.Parameter(torch.zeros(1, transformer_dim))

        # 1-layer pre-LN Transformer block operating over T=4 tokens per node.
        # batch_first=True so input shape is [N, T, d_model].
        self.attn_norm = nn.LayerNorm(transformer_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(transformer_dim)
        self.ffn = nn.Sequential(
            nn.Linear(transformer_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, transformer_dim),
        )

        # FiLM output projection: zero-init so modulation starts as identity
        self.film = nn.Linear(transformer_dim, 2 * hidden_dim)
        if zero_init_output:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    # ------------------------------------------------------------------
    # Graph-size helpers (same logic as LoopFiLMConditioner)
    # ------------------------------------------------------------------

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (size_stats [G, 2], node_batch [N], num_graphs)."""
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor(
                [float(edge_index.shape[1])], device=device, dtype=dtype
            )
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            num_graphs = 1
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch, num_graphs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def fuse_tokens(
        self,
        h: torch.Tensor,
        raw_x: torch.Tensor,
        memory: Optional[torch.Tensor],
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_embed: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """4-token self-attn fusion; returns ``(emb_out, memory_out, attn_weights)``."""
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )

        N = h.shape[0]
        device = h.device

        feat_tok = self.feat_mlp(raw_x)
        emb_tok = self.emb_mlp(h)

        iter_tok = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )

        if self.use_size_input:
            stats, node_batch, _num_graphs = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_tok = self.size_mlp(stats)
            cond_tok = iter_tok.unsqueeze(0) + size_tok[node_batch]
        else:
            cond_tok = iter_tok.unsqueeze(0).expand(N, -1)

        if self.use_time_input:
            if time_embed is None:
                raise ValueError(
                    "LoopTransformerConnector.use_time_input=True but time_embed is None"
                )
            cond_tok = cond_tok + self.time_proj(time_embed)

        if memory is None:
            memory_tok = self.memory_init.expand(N, -1)
        else:
            memory_tok = memory
            if self.memory_clip is not None:
                memory_tok = memory_tok.clamp(-self.memory_clip, self.memory_clip)

        tokens = torch.stack([feat_tok, emb_tok, cond_tok, memory_tok], dim=1)

        tokens_normed = self.attn_norm(tokens)
        if not torch.isfinite(tokens_normed).all():
            clip = self.memory_clip if self.memory_clip is not None else 1e4
            tokens_normed = torch.nan_to_num(
                tokens_normed, nan=0.0, posinf=clip, neginf=-clip
            )
        attn_dtype = tokens_normed.dtype
        attn_in = tokens_normed.float()
        sdp_ctx = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
            if tokens.is_cuda
            else contextlib.nullcontext()
        )
        with sdp_ctx:
            attn_out, attn_weights = self.attn(
                attn_in,
                attn_in,
                attn_in,
                need_weights=need_attn_weights,
            )
        attn_out = attn_out.to(dtype=attn_dtype)
        tokens = tokens + attn_out

        tokens_normed2 = self.ffn_norm(tokens)
        tokens = tokens + self.ffn(tokens_normed2)

        emb_out = tokens[:, 1, :]
        memory_out = tokens[:, 3, :].contiguous()
        if self.memory_clip is not None:
            memory_out = memory_out.clamp(-self.memory_clip, self.memory_clip)
        if not need_attn_weights:
            attn_weights = None
        return emb_out, memory_out, attn_weights

    def forward(
        self,
        h: torch.Tensor,
        raw_x: torch.Tensor,
        memory: Optional[torch.Tensor],
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one connector step and return ``(modulated_h, memory_out)``."""
        emb_out, memory_out, _ = self.fuse_tokens(
            h,
            raw_x,
            memory,
            iteration,
            batch=batch,
            edge_index=edge_index,
            num_nodes=num_nodes,
            time_embed=time_embed,
        )
        gamma, beta = self.film(emb_out).chunk(2, dim=-1)
        modulated_h = h * (1.0 + gamma) + beta
        return modulated_h, memory_out


class DualLoopFiLMConditioner(nn.Module):
    """Two-position FiLM controller for looped-processor PEFT.

    Designed for the size-adaptation processor stack (GNN blocks followed by a
    final Transformer block). Each iteration applies *two* independent FiLM
    modulations on the hidden state:

        position 0 ("after GNNs"):
            applied right BEFORE the final transformer block runs (i.e. on the
            output of the last GNN block).
        position 1 ("after final transformer"):
            applied right AFTER the final transformer block runs (i.e. on the
            output of the entire processor for that iteration).

    Both heads share the iteration embedding and (optional) graph-size MLP, but
    have independent FiLM Linears so the two positions can learn distinct
    modulations. A small position embedding lets each head also read which
    insertion point it is at.

    Initialisation (``zero_init_film=True``):
        * iteration embedding zero-init
        * position embedding zero-init
        * both FiLM Linears zero-init
      -> at step 0, both modulations are identity, and the model exactly
         reproduces the frozen K-pass behaviour. Training learns small
         per-(iteration, position, graph-size) corrections.

    Parameter cost (hidden_dim=256, d_cond=64, K=6, with size input):
        iter_embedding     : K * d_cond                      =     384
        position_embedding : 2 * d_cond                      =     128
        size_mlp           : 2*d_cond + d_cond*d_cond + bias =   4,352
        film_after_gnn     : d_cond * (2*hidden_dim) + bias  =  33,280
        film_after_xformer : d_cond * (2*hidden_dim) + bias  =  33,280
        ----------------------------------------------------------------
        total                                                ~  71,424
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        zero_init_film: bool = True,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.conditioning_dim = conditioning_dim
        self.use_size_input = use_size_input

        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        # 0 = after-GNNs (pre-transformer), 1 = after-final-transformer
        self.position_embedding = nn.Embedding(2, conditioning_dim)
        nn.init.zeros_(self.position_embedding.weight)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        # One FiLM head per insertion point: independent gamma/beta linears.
        self.film_after_gnn = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.film_after_transformer = nn.Linear(conditioning_dim, 2 * hidden_dim)
        if zero_init_film:
            for film in (self.film_after_gnn, self.film_after_transformer):
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor(
                [float(edge_index.shape[1])], device=device, dtype=dtype
            )
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            num_graphs = 1
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch, num_graphs

    def forward(
        self,
        h: torch.Tensor,
        *,
        position: int,
        iteration: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """FiLM-modulate ``h`` at the requested insertion ``position``.

        Args:
            h:         Hidden state, shape ``[N, hidden_dim]``.
            position:  ``0`` for "after GNNs", ``1`` for "after final transformer".
            iteration: Current loop iteration index in ``[0, K)``.
            batch:     Node-to-graph assignment ``[N]`` or ``None``.
            edge_index: Edge index tensor for the batch.
            num_nodes: Total number of nodes ``N``.
        """
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        if position not in (0, 1):
            raise ValueError(f"position must be 0 or 1, got {position}")

        device = h.device
        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )
        pos_cond = self.position_embedding(
            torch.tensor(position, device=device, dtype=torch.long)
        )

        if self.use_size_input:
            stats, node_batch, num_graphs = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_cond = self.size_mlp(stats)  # [G, d_cond]
            per_graph_cond = size_cond + iter_cond.unsqueeze(0) + pos_cond.unsqueeze(0)
            per_node_cond = per_graph_cond[node_batch]
        else:
            per_node_cond = (iter_cond + pos_cond).unsqueeze(0).expand(h.shape[0], -1)

        film_layer = (
            self.film_after_gnn if position == 0 else self.film_after_transformer
        )
        scale_shift = film_layer(per_node_cond)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class TripleLoopFiLMConditioner(nn.Module):
    """Three-position FiLM controller for looped-processor PEFT.

    Strict superset of :class:`DualLoopFiLMConditioner`: where the dual variant
    fires once before and once after the final Transformer block, the triple
    variant adds a *third* FiLM head that fires between the first and last GNN
    block. The intended processor stack is

        GNN block 0  ─►  [position 0]  ─►  GNN block 1  ─►  [position 1]  ─►
        Transformer block  ─►  [position 2]  ─►  (next iter / decode)

    so each iteration of the K-loop applies *three* independent FiLM
    modulations on the hidden state:

        position 0 ("after first GNN block"):
            applied right AFTER the first GNN block runs (i.e. between the two
            GNN blocks). NEW relative to ``dual_film_loop``.
        position 1 ("after final GNN block / before final transformer"):
            applied right AFTER the last GNN block runs, which is the same
            insertion point as ``DualLoopFiLMConditioner`` position 0.
        position 2 ("after final transformer"):
            applied right AFTER the final transformer block runs, same
            insertion point as ``DualLoopFiLMConditioner`` position 1.

    All three heads share the iteration embedding and (optional) graph-size
    MLP, but have independent FiLM Linears so the three positions can learn
    distinct modulations. A position embedding (3 entries) lets each head also
    read which insertion point it is at.

    Initialisation (``zero_init_film=True``):
        * iteration embedding zero-init
        * position embedding zero-init
        * all three FiLM Linears zero-init
      -> at step 0, all three modulations are identity, and the model exactly
         reproduces the frozen K-pass behaviour. Training learns small
         per-(iteration, position, graph-size) corrections.

    Parameter cost (hidden_dim=256, d_cond=64, K=6, with size input):
        iter_embedding     : K * d_cond                      =     384
        position_embedding : 3 * d_cond                      =     192
        size_mlp           : 2*d_cond + d_cond*d_cond + bias =   4,352
        film_after_gnn0    : d_cond * (2*hidden_dim) + bias  =  33,280
        film_after_gnn1    : d_cond * (2*hidden_dim) + bias  =  33,280
        film_after_xformer : d_cond * (2*hidden_dim) + bias  =  33,280
        ----------------------------------------------------------------
        total                                                ~ 104,768
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        zero_init_film: bool = True,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.conditioning_dim = conditioning_dim
        self.use_size_input = use_size_input
        self.num_positions = 3

        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        # 0 = after-first-GNN-block (between GNN blocks)
        # 1 = after-last-GNN-block (= pre-final-transformer)
        # 2 = after-final-transformer
        self.position_embedding = nn.Embedding(3, conditioning_dim)
        nn.init.zeros_(self.position_embedding.weight)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        # One FiLM head per insertion point. The naming intentionally mirrors
        # DualLoopFiLMConditioner so the static analyzer auto-discovers them
        # (it scans for nn.Linear children with 'film' in the name and
        # out_features == 2 * hidden_dim).
        self.film_after_gnn0 = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.film_after_gnn1 = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.film_after_transformer = nn.Linear(conditioning_dim, 2 * hidden_dim)
        if zero_init_film:
            for film in (
                self.film_after_gnn0,
                self.film_after_gnn1,
                self.film_after_transformer,
            ):
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor(
                [float(edge_index.shape[1])], device=device, dtype=dtype
            )
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            num_graphs = 1
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch, num_graphs

    def forward(
        self,
        h: torch.Tensor,
        *,
        position: int,
        iteration: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """FiLM-modulate ``h`` at the requested insertion ``position``.

        Args:
            h:         Hidden state, shape ``[N, hidden_dim]``.
            position:  ``0`` for "after first GNN block",
                       ``1`` for "after last GNN block / before final transformer",
                       ``2`` for "after final transformer".
            iteration: Current loop iteration index in ``[0, K)``.
            batch:     Node-to-graph assignment ``[N]`` or ``None``.
            edge_index: Edge index tensor for the batch.
            num_nodes: Total number of nodes ``N``.
        """
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        if position not in (0, 1, 2):
            raise ValueError(f"position must be 0, 1, or 2, got {position}")

        device = h.device
        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )
        pos_cond = self.position_embedding(
            torch.tensor(position, device=device, dtype=torch.long)
        )

        if self.use_size_input:
            stats, node_batch, _num_graphs = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_cond = self.size_mlp(stats)  # [G, d_cond]
            per_graph_cond = size_cond + iter_cond.unsqueeze(0) + pos_cond.unsqueeze(0)
            per_node_cond = per_graph_cond[node_batch]
        else:
            per_node_cond = (iter_cond + pos_cond).unsqueeze(0).expand(h.shape[0], -1)

        if position == 0:
            film_layer = self.film_after_gnn0
        elif position == 1:
            film_layer = self.film_after_gnn1
        else:
            film_layer = self.film_after_transformer
        scale_shift = film_layer(per_node_cond)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class DualLoopTransformerConnector(nn.Module):
    """Two-position Transformer-connector controller for looped-processor PEFT.

    Stronger sibling of ``DualLoopFiLMConditioner``: at each of the two
    insertion points (after the GNN blocks, and after the final transformer
    block) the connector runs a small pre-LN Transformer block over a short
    per-node token sequence and emits a FiLM steering signal. Each insertion
    point also threads its OWN cross-iteration memory token across all K
    iterations, so the two positions accumulate independent state across the
    unroll. There is no raw-input stream: the controller only consumes the
    current hidden state, the per-iteration / per-graph-size conditioning,
    and the position-specific memory.

    For each (iteration k, position p in {0, 1}) the per-node token sequence is

        tokens = [emb_tok, cond_tok, memory_tok_p]   # [N, T=3, d_model]

    A single 1-layer MHA + FFN block is shared across positions (parameter
    efficient), with a position embedding added into ``cond_tok`` so the block
    sees which insertion point it is at. Each position has its own:
        * ``emb_mlp_p`` (h -> d_model encoder)
        * ``memory_init_p`` (learned [1, d_model] init at k=0)
        * ``film_p`` (zero-init FiLM head -> identity at step 0)

    Returns ``(modulated_h, memory_out_p)``: caller threads the new memory
    back in at iteration k+1 for the same position.

    Parameter budget (hidden_dim=256, d_model=128, ffn_dim=256, K=8, x2 pos):
        emb_mlp_pos0    (256→128→128 + LN)        ~50k
        emb_mlp_pos1    (256→128→128 + LN)        ~50k
        iter_embed      K×128                       1k
        size_mlp        2→64→128                    8.7k
        position_embed  2×128                       0.3k
        memory_init x2  2×128                       0.3k
        MHA (shared)    128, 4 heads               ~66k
        FFN (shared)    128→256→128                ~66k
        LayerNorms x2 (shared)                      0.5k
        film_pos0       128→512 (zero-init)        ~66k
        film_pos1       128→512 (zero-init)        ~66k
        ─────────────────────────────────────  ≈375k
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        transformer_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if transformer_dim % num_heads != 0:
            raise ValueError(
                f"transformer_dim ({transformer_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.transformer_dim = transformer_dim
        self.use_size_input = use_size_input

        # Per-position embedding encoders: each position has its own h -> d_model
        # encoder so the two insertion points can specialise their feature view.
        self.emb_mlp_pos0 = nn.Sequential(
            nn.Linear(hidden_dim, transformer_dim),
            nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        self.emb_mlp_pos1 = nn.Sequential(
            nn.Linear(hidden_dim, transformer_dim),
            nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )

        # Iteration embedding (zero-init -> conditioning starts at zero)
        self.iter_embedding = nn.Embedding(num_iterations, transformer_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        # Position embedding so the shared transformer block can distinguish
        # which insertion point it is processing.
        self.position_embedding = nn.Embedding(2, transformer_dim)
        nn.init.zeros_(self.position_embedding.weight)

        # Optional graph-size MLP shared across positions and iterations.
        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, transformer_dim),
            )
        else:
            self.size_mlp = None

        # Two independent learnable memory init tokens (one per position).
        self.memory_init_pos0 = nn.Parameter(torch.zeros(1, transformer_dim))
        self.memory_init_pos1 = nn.Parameter(torch.zeros(1, transformer_dim))

        # Shared 1-layer pre-LN Transformer block (over T=3 tokens / node).
        self.attn_norm = nn.LayerNorm(transformer_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(transformer_dim)
        self.ffn = nn.Sequential(
            nn.Linear(transformer_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, transformer_dim),
        )

        # Per-position FiLM heads (zero-init -> identity at init).
        self.film_pos0 = nn.Linear(transformer_dim, 2 * hidden_dim)
        self.film_pos1 = nn.Linear(transformer_dim, 2 * hidden_dim)
        if zero_init_output:
            for film in (self.film_pos0, self.film_pos1):
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor(
                [float(edge_index.shape[1])], device=device, dtype=dtype
            )
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            num_graphs = 1
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch, num_graphs

    def init_memory(
        self,
        position: int,
        num_nodes: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the initial memory tensor ``[N, d_model]`` for ``position``."""
        if position == 0:
            return self.memory_init_pos0.to(device).expand(num_nodes, -1)
        if position == 1:
            return self.memory_init_pos1.to(device).expand(num_nodes, -1)
        raise ValueError(f"position must be 0 or 1, got {position}")

    def forward(
        self,
        h: torch.Tensor,
        memory: Optional[torch.Tensor],
        *,
        position: int,
        iteration: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one connector step at ``position`` and return ``(modulated_h, memory_out)``.

        Args:
            h:         Current hidden state, shape ``[N, hidden_dim]``.
            memory:    Memory at this position from the previous iteration,
                       shape ``[N, transformer_dim]``, or ``None`` to use the
                       learned per-position initial memory.
            position:  ``0`` for "after GNNs", ``1`` for "after final transformer".
            iteration: Current loop iteration index in ``[0, K)``.
            batch:     Node-to-graph assignment ``[N]`` or ``None``.
            edge_index: Edge index tensor.
            num_nodes: Total number of nodes ``N``.

        Returns:
            modulated_h: ``h`` after FiLM steering at this position.
            memory_out:  Updated position-specific memory, threaded to k+1.
        """
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        if position not in (0, 1):
            raise ValueError(f"position must be 0 or 1, got {position}")

        N = h.shape[0]
        device = h.device

        # ---- 1. Encode current hidden state with the position-specific MLP. ----
        emb_mlp = self.emb_mlp_pos0 if position == 0 else self.emb_mlp_pos1
        emb_tok = emb_mlp(h)  # [N, d_model]

        # ---- 2. Conditioning token: iteration + position + graph size. ----
        iter_tok = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )  # [d_model]
        pos_tok = self.position_embedding(
            torch.tensor(position, device=device, dtype=torch.long)
        )  # [d_model]

        if self.use_size_input:
            stats, node_batch, _num_graphs = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_tok = self.size_mlp(stats)           # [G, d_model]
            cond_tok = (
                iter_tok.unsqueeze(0)
                + pos_tok.unsqueeze(0)
                + size_tok[node_batch]
            )                                          # [N, d_model]
        else:
            cond_tok = (iter_tok + pos_tok).unsqueeze(0).expand(N, -1)  # [N, d_model]

        # ---- 3. Memory token: per-position learned init at k=0. ----
        if memory is None:
            memory_tok = self.init_memory(position, N, device=device)
        else:
            memory_tok = memory  # [N, d_model]

        # ---- 4. Stack tokens -> [N, T=3, d_model] ----
        tokens = torch.stack([emb_tok, cond_tok, memory_tok], dim=1)

        # ---- 5. Pre-LN Transformer block (shared across positions). ----
        tokens_normed = self.attn_norm(tokens)
        attn_out, _ = self.attn(tokens_normed, tokens_normed, tokens_normed)
        tokens = tokens + attn_out

        tokens_normed2 = self.ffn_norm(tokens)
        ffn_out = self.ffn(tokens_normed2)
        tokens = tokens + ffn_out

        # ---- 6. Read emb token -> position-specific FiLM -> modulate h. ----
        emb_out = tokens[:, 0, :]                      # [N, d_model]
        film_layer = self.film_pos0 if position == 0 else self.film_pos1
        gamma, beta = film_layer(emb_out).chunk(2, dim=-1)  # [N, hidden_dim]
        modulated_h = h * (1.0 + gamma) + beta

        # ---- 7. Read memory token -> thread to next iteration at this pos. ----
        memory_out = tokens[:, 2, :].contiguous()      # [N, d_model]

        return modulated_h, memory_out


class StagedLoopFiLMConditioner(nn.Module):
    """Five-controller FiLM for *staged* (non-interleaved) looped-processor PEFT.

    Designed to pair with a 3-block processor (early GNN -> late GNN ->
    Transformer). Whereas ``TripleLoopFiLMConditioner`` interleaves the three
    blocks inside a single outer K-loop, this conditioner drives **three
    sequential phase loops** with a one-shot transition FiLM between each
    pair of phases:

        Phase 1 (early GNN block, K iterations)
            for k in 0..K-1:
                h = early_GNN_block(h)
                h = FiLM_local_early(h, k)            # head #1 (LOCAL, iter)
        Transition 1
            h = FiLM_transition_early_to_late(h)      # head #2 (one-shot)
        Phase 2 (late GNN block, K iterations)
            for k in 0..K-1:
                h = late_GNN_block(h)
                h = FiLM_local_late(h, k)             # head #3 (LOCAL, iter)
        Transition 2
            h = FiLM_transition_late_to_transformer(h) # head #4 (one-shot)
        Phase 3 (Transformer block, K iterations)
            for k in 0..K-1:
                h = transformer_block(h)
                h = FiLM_local_transformer(h, k)      # head #5 (LOCAL, iter)

    LOCAL controllers are POST-block (matching ``TripleLoopFiLMConditioner``
    convention) and are conditioned on (iteration index k, phase id, optional
    graph size). They share the iteration embedding and (optional) graph-size
    MLP across the 3 phases but have independent FiLM Linears so each phase
    can specialise.

    TRANSITION controllers are SINGLE-SHOT FiLMs that fire exactly once
    between adjacent phases. They have no iteration dim; the only conditioning
    is a learned per-transition embedding plus (optional) graph size.

    All five FiLM Linears are zero-initialised so each head starts as
    identity. At step 0 the model exactly reproduces
    ``K * (early_GNN + late_GNN + Transformer)`` forward passes of the
    frozen backbone, and training learns small per-step / per-transition
    steering corrections.

    Parameter cost (hidden_dim=256, d_cond=64, K=6, with size input):
        iter_embedding                 : K * d_cond                      =     384
        local_position_embedding       : 3 * d_cond                      =     192
        transition_position_embedding  : 2 * d_cond                      =     128
        size_mlp                       : 2*d_cond + d_cond*d_cond + bias =   4,352
        film_local_{early,late,xfm}    : 3 * (d_cond * 2*hidden_dim + b) =  99,840
        film_transition_{e2l,l2x}      : 2 * (d_cond * 2*hidden_dim + b) =  66,560
        ----------------------------------------------------------------
        total                                                            ~ 171,456
    """

    NUM_LOCAL_PHASES = 3   # early / late / transformer
    NUM_TRANSITIONS = 2    # early->late / late->transformer

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        zero_init_film: bool = True,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.conditioning_dim = conditioning_dim
        self.use_size_input = use_size_input

        # Shared iteration embedding (used by all 3 LOCAL heads).
        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        # Per-phase position embedding for LOCAL heads (0=early, 1=late, 2=xfm).
        self.local_position_embedding = nn.Embedding(
            self.NUM_LOCAL_PHASES, conditioning_dim
        )
        nn.init.zeros_(self.local_position_embedding.weight)

        # Per-transition position embedding for TRANSITION heads
        # (0 = early -> late, 1 = late -> transformer).
        self.transition_position_embedding = nn.Embedding(
            self.NUM_TRANSITIONS, conditioning_dim
        )
        nn.init.zeros_(self.transition_position_embedding.weight)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        # 5 FiLM heads. Naming uses 'film' so the static loop-controller
        # analyzer (which scans for nn.Linear children with 'film' in the
        # name and out_features == 2 * hidden_dim) auto-discovers them.
        self.film_local_early = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.film_local_late = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.film_local_transformer = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.film_transition_early_to_late = nn.Linear(
            conditioning_dim, 2 * hidden_dim
        )
        self.film_transition_late_to_transformer = nn.Linear(
            conditioning_dim, 2 * hidden_dim
        )
        if zero_init_film:
            for film in (
                self.film_local_early,
                self.film_local_late,
                self.film_local_transformer,
                self.film_transition_early_to_late,
                self.film_transition_late_to_transformer,
            ):
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)

    def _graph_size_features(
        self,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        device = edge_index.device
        dtype = torch.float32
        if batch is None:
            n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
            e_per_graph = torch.tensor(
                [float(edge_index.shape[1])], device=device, dtype=dtype
            )
            node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
            num_graphs = 1
        else:
            node_batch = batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
            if edge_index.numel() > 0:
                edge_batch = batch[edge_index[0]]
                e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                    device=device, dtype=dtype
                )
            else:
                e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

        n_feat = torch.log1p(n_per_graph) / 9.21
        e_feat = torch.log1p(e_per_graph) / 9.21
        stats = torch.stack([n_feat, e_feat], dim=-1)
        return stats, node_batch, num_graphs

    def _local_film_for_phase(self, phase: int) -> nn.Linear:
        if phase == 0:
            return self.film_local_early
        if phase == 1:
            return self.film_local_late
        if phase == 2:
            return self.film_local_transformer
        raise ValueError(f"phase must be in {{0, 1, 2}}, got {phase}")

    def _transition_film_for(self, transition: int) -> nn.Linear:
        if transition == 0:
            return self.film_transition_early_to_late
        if transition == 1:
            return self.film_transition_late_to_transformer
        raise ValueError(f"transition must be in {{0, 1}}, got {transition}")

    def forward_local(
        self,
        h: torch.Tensor,
        *,
        phase: int,
        iteration: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """POST-block FiLM call for one LOCAL controller iteration in a phase.

        Args:
            h:         Hidden state, shape ``[N, hidden_dim]``.
            phase:     ``0`` early-GNN, ``1`` late-GNN, ``2`` transformer.
            iteration: Iteration index in ``[0, K)`` of the per-phase loop.
            batch:     Node-to-graph assignment ``[N]`` or ``None``.
            edge_index: Edge index tensor for the batch.
            num_nodes: Total number of nodes ``N``.
        """
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        if phase not in (0, 1, 2):
            raise ValueError(f"phase must be 0, 1, or 2, got {phase}")

        device = h.device
        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )
        pos_cond = self.local_position_embedding(
            torch.tensor(phase, device=device, dtype=torch.long)
        )
        if self.use_size_input:
            stats, node_batch, _ = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_cond = self.size_mlp(stats)  # [G, d_cond]
            per_graph_cond = size_cond + iter_cond.unsqueeze(0) + pos_cond.unsqueeze(0)
            per_node_cond = per_graph_cond[node_batch]
        else:
            per_node_cond = (iter_cond + pos_cond).unsqueeze(0).expand(h.shape[0], -1)

        film_layer = self._local_film_for_phase(phase)
        gamma, beta = film_layer(per_node_cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def forward_transition(
        self,
        h: torch.Tensor,
        *,
        transition: int,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """One-shot FiLM call for a TRANSITION controller between two phases.

        Args:
            h:         Hidden state, shape ``[N, hidden_dim]``.
            transition: ``0`` for early-to-late, ``1`` for late-to-transformer.
            batch:     Node-to-graph assignment ``[N]`` or ``None``.
            edge_index: Edge index tensor for the batch.
            num_nodes: Total number of nodes ``N``.
        """
        if transition not in (0, 1):
            raise ValueError(f"transition must be 0 or 1, got {transition}")

        device = h.device
        pos_cond = self.transition_position_embedding(
            torch.tensor(transition, device=device, dtype=torch.long)
        )
        if self.use_size_input:
            stats, node_batch, _ = self._graph_size_features(
                batch, edge_index, num_nodes
            )
            size_cond = self.size_mlp(stats)  # [G, d_cond]
            per_graph_cond = size_cond + pos_cond.unsqueeze(0)
            per_node_cond = per_graph_cond[node_batch]
        else:
            per_node_cond = pos_cond.unsqueeze(0).expand(h.shape[0], -1)

        film_layer = self._transition_film_for(transition)
        gamma, beta = film_layer(per_node_cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Gated residual loop conditioner + halt module (iterGNN family).
#
# Used by ``loop_mode == 'gnn_gated_halt_loop'`` in ``UnifiedModel``. Together
# they implement two sibling variants of an iterative GNN-only loop trained
# from scratch (no pretrain warm-start):
#
#   Variant A ("gated_loop"):
#     - per-node gate with a STRONG activation (Hard Concrete OR Gumbel-ST)
#       so the gate can hit *exactly* 0 / 1 and freeze a node's hidden state.
#     - no halt module; the gate IS the halt mechanism.
#
#   Variant B ("iterGNN"):
#     - per-node gate with a soft sigmoid activation.
#     - separate ``HaltModule`` emits per-graph halt logits per step k, which
#       the model turns into a distribution ``p_k`` (PonderNet OR softmax
#       over k) used to weight the per-iteration losses / predictions.
#
# Both modules are deliberately small (parameter cost dominated by the FiLM-
# style hidden-to-cond_dim projection inside the gate MLP).
# ---------------------------------------------------------------------------


def _graph_size_features_from_batch(
    batch: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(size_stats [G, 2], node_batch [N], num_graphs)``.

    ``size_stats`` carries the log-normalised (num_nodes, num_edges) features
    used by the existing loop conditioners. Shared helper so the gated
    conditioner and the halt module agree on per-graph size stats.
    """
    device = edge_index.device
    dtype = torch.float32
    if batch is None:
        n_per_graph = torch.tensor([float(num_nodes)], device=device, dtype=dtype)
        e_per_graph = torch.tensor(
            [float(edge_index.shape[1])], device=device, dtype=dtype
        )
        node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
        num_graphs = 1
    else:
        node_batch = batch
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        n_per_graph = torch.bincount(batch, minlength=num_graphs).to(
            device=device, dtype=dtype
        )
        if edge_index.numel() > 0:
            edge_batch = batch[edge_index[0]]
            e_per_graph = torch.bincount(edge_batch, minlength=num_graphs).to(
                device=device, dtype=dtype
            )
        else:
            e_per_graph = torch.zeros(num_graphs, device=device, dtype=dtype)

    n_feat = torch.log1p(n_per_graph) / 9.21
    e_feat = torch.log1p(e_per_graph) / 9.21
    stats = torch.stack([n_feat, e_feat], dim=-1)
    return stats, node_batch, num_graphs


def _scatter_mean(
    h: torch.Tensor, node_batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Mean-pool node features ``[N, D]`` to per-graph features ``[G, D]``."""
    if num_graphs == 1:
        return h.mean(dim=0, keepdim=True)
    sum_h = torch.zeros(num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
    sum_h.index_add_(0, node_batch, h)
    n_per_graph = (
        torch.bincount(node_batch, minlength=num_graphs)
        .to(device=h.device, dtype=h.dtype)
        .clamp_min(1.0)
    )
    return sum_h / n_per_graph.unsqueeze(-1)


class GatedResidualLoopConditioner(nn.Module):
    """Per-node residual gate for an iterative GNN-only loop.

    Implements ``h_next = (1 - g_{k,n}) * h_prev + g_{k,n} * h_new`` where the
    per-node gate ``g_{k,n} in [0, 1]`` is produced by a small MLP from
    ``[h_prev_n, h_new_n, iter_embed_k, size_emb_g(n)]`` and passed through
    one of three configurable activations:

    * ``hard_concrete`` (Louizos, Welling, Kingma 2017): stretched-and-clipped
      logistic distribution. Reparameterised, fully differentiable, and the
      gate output can hit *exactly* 0 / 1. Comes with a closed-form L0-style
      probability for the "encourage halting" regulariser.
    * ``gumbel_st``: Gumbel-Sigmoid sample with straight-through estimator.
      Forward output is exactly ``{0, 1}``; backward gradients flow through
      the soft Gumbel-Sigmoid sample. Temperature anneals linearly across
      training steps from ``temperature_start`` to ``temperature_end``.
    * ``sigmoid``: plain sigmoid. Gates live in the open interval ``(0, 1)``;
      never hits exact 0. Used by Variant B / iterGNN where the discrete
      halting decision is delegated to ``HaltModule``.

    Init bias of the gate head is set so ``g ~= 0.9`` at step 0 (regardless
    of activation), so the loop starts as "trust the new GNN output, run
    the vanilla K-loop" and training learns to gate down.

    Parameter cost (hidden_dim=256, d_cond=64, K=8):
        iter_embedding : K * d_cond                  =     512
        size_mlp       : 2->d_cond->d_cond           =   4,352
        gate_mlp       : (2*hidden + 2*d_cond)->d_cond->1
                       ~ 41,153
        -----------------------------------------------------
        total                                         ~  46,017
    """

    VALID_ACTIVATIONS = ("hard_concrete", "gumbel_st", "sigmoid", "sigmoid_sharp")

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        conditioning_dim: int = 64,
        activation: str = "hard_concrete",
        use_size_input: bool = True,
        init_gate_bias: float = 2.2,
        # Hard Concrete params (only used when activation == 'hard_concrete')
        hc_beta: float = 2.0 / 3.0,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        # Gumbel-ST params (only used when activation == 'gumbel_st')
        gumbel_temperature_start: float = 1.0,
        gumbel_temperature_end: float = 0.1,
        gumbel_temperature_anneal_steps: int = 0,
        # sigmoid_sharp params (only used when activation == 'sigmoid_sharp')
        gate_sharpness: float = 5.0,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {self.VALID_ACTIVATIONS}, got {activation!r}"
            )
        if hc_zeta <= hc_gamma:
            raise ValueError(
                f"hard_concrete zeta must be > gamma; got zeta={hc_zeta}, gamma={hc_gamma}"
            )
        if hc_gamma >= 0.0:
            raise ValueError(
                f"hard_concrete gamma must be < 0 to allow exact 0 outputs; got {hc_gamma}"
            )
        if hc_zeta <= 1.0:
            raise ValueError(
                f"hard_concrete zeta must be > 1 to allow exact 1 outputs; got {hc_zeta}"
            )
        if gate_sharpness <= 0.0:
            raise ValueError(
                f"gate_sharpness must be > 0; got {gate_sharpness}"
            )

        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.conditioning_dim = conditioning_dim
        self.activation = activation
        self.use_size_input = use_size_input

        self.hc_beta = float(hc_beta)
        self.hc_gamma = float(hc_gamma)
        self.hc_zeta = float(hc_zeta)
        self.gumbel_temperature_start = float(gumbel_temperature_start)
        self.gumbel_temperature_end = float(gumbel_temperature_end)
        self.gumbel_temperature_anneal_steps = int(gumbel_temperature_anneal_steps)
        self.gate_sharpness = float(gate_sharpness)

        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        # Gate head: input is [h_prev, h_new, iter_embed, size_emb], output a
        # scalar logit per node.
        in_dim = 2 * hidden_dim + conditioning_dim + (
            conditioning_dim if use_size_input else 0
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, 1),
        )
        # Init bias so that at step 0 the gate is near 1 ("trust the GNN").
        # For sigmoid/hard_concrete, sigmoid(2.2) ~= 0.9. For gumbel-ST the
        # same bias biases the soft sample toward 1, so most forwards hit
        # the "round-to-1" branch.
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, float(init_gate_bias))

        # Step counter used to anneal the Gumbel temperature during training.
        # Persistent=False so checkpoints stay portable.
        self.register_buffer(
            "_step", torch.zeros((), dtype=torch.long), persistent=False
        )

    # ---------------------------------------------------------------- #
    # Public helpers                                                     #
    # ---------------------------------------------------------------- #

    def gumbel_temperature(self) -> float:
        """Current Gumbel-Sigmoid temperature given the step counter."""
        if self.activation != "gumbel_st":
            return 1.0
        if self.gumbel_temperature_anneal_steps <= 0:
            return self.gumbel_temperature_end
        frac = min(1.0, float(self._step.item()) / float(
            self.gumbel_temperature_anneal_steps
        ))
        return (
            self.gumbel_temperature_start
            + frac * (self.gumbel_temperature_end - self.gumbel_temperature_start)
        )

    def hard_concrete_l0_probability(
        self, gate_logits: torch.Tensor
    ) -> torch.Tensor:
        """Return P(gate != 0) for the given logits under Hard Concrete.

        For Hard Concrete with stretch ``(gamma, zeta)`` and temperature
        ``beta``, the probability mass NOT placed at exact zero is
        ``sigmoid(logit - beta * log(-gamma / zeta))``. The complement is the
        L0 penalty: encouraging this term to be small drives gates toward
        exact 0. (See Louizos et al. 2017, eq. (12).)
        """
        if self.activation != "hard_concrete":
            # Fallback that still produces a sensible "fraction-active" signal
            # for sigmoid / gumbel_st (sigmoid of the logit), so the same
            # config knob can be reused as an L1 sparsity term.
            return torch.sigmoid(gate_logits)
        shift = self.hc_beta * float(
            torch.log(torch.tensor(-self.hc_gamma / self.hc_zeta))
        )
        return torch.sigmoid(gate_logits - shift)

    # ---------------------------------------------------------------- #
    # Gate activation                                                    #
    # ---------------------------------------------------------------- #

    def _apply_hard_concrete(self, logit: torch.Tensor) -> torch.Tensor:
        """Hard Concrete reparameterisation. Returns gate values in [0, 1]
        with non-zero probability mass at exact 0 and 1.

        In eval mode the random variable is dropped (deterministic).
        """
        if self.training:
            eps = 1e-6
            u = torch.rand_like(logit).clamp(eps, 1.0 - eps)
            s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + logit) / self.hc_beta)
        else:
            s = torch.sigmoid(logit / self.hc_beta)
        s_bar = s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma
        return torch.clamp(s_bar, 0.0, 1.0)

    def _apply_gumbel_st(self, logit: torch.Tensor) -> torch.Tensor:
        """Gumbel-Sigmoid sample with a straight-through estimator.

        Forward output is exactly ``{0, 1}``; backward uses the gradient of
        the soft Gumbel-Sigmoid sample. Eval mode is deterministic (no
        Gumbel noise) and uses a sharp threshold at logit > 0.
        """
        if self.training:
            tau = max(self.gumbel_temperature(), 1e-3)
            eps = 1e-9
            u1 = torch.rand_like(logit).clamp(eps, 1.0 - eps)
            u2 = torch.rand_like(logit).clamp(eps, 1.0 - eps)
            g1 = -torch.log(-torch.log(u1))
            g2 = -torch.log(-torch.log(u2))
            soft = torch.sigmoid((logit + g1 - g2) / tau)
            hard = (soft > 0.5).to(soft.dtype)
            return soft + (hard - soft).detach()
        else:
            return (logit > 0.0).to(logit.dtype)

    def _apply_sigmoid(self, logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit)

    def _apply_sigmoid_sharp(self, logit: torch.Tensor) -> torch.Tensor:
        """Sharpened sigmoid: sigmoid(T * logit) with T = gate_sharpness.

        For T ≫ 1 the output saturates near 0 or 1 for logit magnitudes of
        roughly 1/T, giving bimodal gate distributions without any
        reparameterisation, random noise, or hard clipping.  The gate is still
        differentiable everywhere (gradient = T * σ * (1 − σ)).  Pair with
        entropy_weight > 0 in the loss to add an explicit push toward 0/1.

        Effective init gate ≈ 0.9 requires init_gate_bias ≈ 2.2 / T
        (e.g. T=5 → init_gate_bias=0.44).
        """
        return torch.sigmoid(self.gate_sharpness * logit)

    # ---------------------------------------------------------------- #
    # Forward                                                            #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        h_prev: torch.Tensor,
        h_new: torch.Tensor,
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the gated residual update.

        Returns:
            h_next:     ``[N, hidden_dim]`` modulated hidden state.
            gate:       ``[N]`` per-node gate value in ``[0, 1]``.
            gate_logit: ``[N]`` per-node pre-activation logit (used by
                        L0 / entropy regularisers without re-evaluating
                        the MLP).
        """
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        device = h_prev.device
        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )

        if self.use_size_input:
            stats, node_batch, _num_graphs = _graph_size_features_from_batch(
                batch, edge_index, num_nodes
            )
            size_cond = self.size_mlp(stats)
            per_node_size = size_cond[node_batch]
            per_node_iter = iter_cond.unsqueeze(0).expand(num_nodes, -1)
            mlp_input = torch.cat(
                [h_prev, h_new, per_node_iter, per_node_size], dim=-1
            )
        else:
            per_node_iter = iter_cond.unsqueeze(0).expand(num_nodes, -1)
            mlp_input = torch.cat([h_prev, h_new, per_node_iter], dim=-1)

        gate_logit = self.gate_mlp(mlp_input).squeeze(-1)  # [N]

        if self.activation == "hard_concrete":
            gate = self._apply_hard_concrete(gate_logit)
        elif self.activation == "gumbel_st":
            gate = self._apply_gumbel_st(gate_logit)
        elif self.activation == "sigmoid_sharp":
            gate = self._apply_sigmoid_sharp(gate_logit)
        else:  # "sigmoid"
            gate = self._apply_sigmoid(gate_logit)

        gate_per_node = gate.unsqueeze(-1)  # [N, 1]
        h_next = (1.0 - gate_per_node) * h_prev + gate_per_node * h_new

        if self.training:
            self._step += 1
        return h_next, gate, gate_logit


class HaltModule(nn.Module):
    """Per-step halt logit head for iterGNN soft halting (Variant B).

    For each iteration ``k`` and each graph in the batch, emits a scalar
    halt logit conditioned on:

        * mean-pooled current hidden state ``mean_n h_{k,n}``  (per-graph)
        * iteration embedding ``iter_embed[k]``
        * graph-size embedding from ``[log1p(N)/9.21, log1p(E)/9.21]``
        * mean of the per-node gate values at iteration k  (per-graph)

    The recurrence that turns these K logits into a halt distribution
    ``p_k`` lives in ``UnifiedModel.forward`` and supports two shapes:

        * ``pondernet_geometric``: cumulative ``lambda_k = sigmoid(logit_k)``
          with the last step forced to 1 (PonderNet style).
        * ``softmax_over_k``: ``softmax`` over ``[logit_0, ..., logit_{K-1}]``.

    Init bias is ``-2.0`` so the PonderNet path starts in "always run K"
    mode (sigmoid(-2) ~= 0.12 per step, last step forced to 1) and gradually
    learns to halt earlier. For the softmax path the same negative bias is
    benign (uniform-like prior over k at init since iter_embedding is
    zero-init too).

    Parameter cost (hidden_dim=256, d_cond=64):
        iter_embedding : K * d_cond            =     512  (K=8)
        size_mlp       : 2->d_cond->d_cond     =   4,352
        state_proj     : hidden_dim->d_cond    =  16,448
        halt_head      : (3*d_cond)->1         =     193
        ---------------------------------------------------
        total                                  ~  21,505
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_iterations: int,
        conditioning_dim: int = 64,
        use_size_input: bool = True,
        use_gate_statistic: bool = True,
        init_halt_bias: float = -2.0,
    ):
        super().__init__()
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.conditioning_dim = conditioning_dim
        self.use_size_input = use_size_input
        self.use_gate_statistic = use_gate_statistic

        self.iter_embedding = nn.Embedding(num_iterations, conditioning_dim)
        nn.init.zeros_(self.iter_embedding.weight)

        # State conditioning: per-graph mean-pooled hidden -> conditioning_dim.
        # Zero-init so at step 0 the halt logit is determined only by the
        # bias (we want a defined "always run K" starting point).
        self.state_proj = nn.Linear(hidden_dim, conditioning_dim)
        nn.init.zeros_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)

        if use_size_input:
            self.size_mlp = nn.Sequential(
                nn.Linear(2, conditioning_dim),
                nn.SiLU(),
                nn.Linear(conditioning_dim, conditioning_dim),
            )
        else:
            self.size_mlp = None

        # The halt head consumes (state, iter_embed, size_embed, gate_stat).
        # gate_stat is a single scalar per graph so it adds one input dim.
        in_dim = conditioning_dim + conditioning_dim
        if use_size_input:
            in_dim += conditioning_dim
        if use_gate_statistic:
            in_dim += 1
        self.halt_head = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, float(init_halt_bias))

    def forward(
        self,
        h: torch.Tensor,
        iteration: int,
        *,
        batch: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        gate_per_node: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-graph halt logit of shape ``[num_graphs]``."""
        if not 0 <= iteration < self.num_iterations:
            raise ValueError(
                f"iteration {iteration} out of range [0, {self.num_iterations})"
            )
        device = h.device
        _stats, node_batch, num_graphs = _graph_size_features_from_batch(
            batch, edge_index, num_nodes
        )
        pooled = _scatter_mean(h, node_batch, num_graphs)             # [G, D]
        state_cond = self.state_proj(pooled)                          # [G, d_cond]

        iter_cond = self.iter_embedding(
            torch.tensor(iteration, device=device, dtype=torch.long)
        )                                                              # [d_cond]
        iter_cond_g = iter_cond.unsqueeze(0).expand(num_graphs, -1)   # [G, d_cond]

        parts = [state_cond, iter_cond_g]
        if self.use_size_input:
            stats_g, _, _ = _graph_size_features_from_batch(
                batch, edge_index, num_nodes
            )
            parts.append(self.size_mlp(stats_g))                       # [G, d_cond]
        if self.use_gate_statistic:
            if gate_per_node is None:
                gate_stat = torch.zeros(num_graphs, 1, device=device, dtype=h.dtype)
            else:
                gate_per_graph = _scatter_mean(
                    gate_per_node.unsqueeze(-1), node_batch, num_graphs
                )                                                       # [G, 1]
                gate_stat = gate_per_graph.to(h.dtype)
            parts.append(gate_stat)
        halt_input = torch.cat(parts, dim=-1)                          # [G, in_dim]
        return self.halt_head(halt_input).squeeze(-1)                  # [G]
