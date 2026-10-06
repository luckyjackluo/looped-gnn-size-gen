"""
FMLoopedDenoiserWrapper — loop the GNN processor of a pretrained FM denoiser.

Controller modes (``controller_kwargs.mode``):
  - ``film`` (default): K FiLM+processor passes, single decode at end.
  - ``transformer_loop``: ``LoopTransformerConnector`` (feat+emb+memory tokens,
    1-layer mini-Transformer per node) before each processor pass.
  - ``gated_output``: per-level decode + gated mix of h and v (see below).
  - ``ponder``: legacy PonderNet soft-halting over level outputs.

Gated-output recipe (FM, default ``gated.decode_once=True``):
    h_k --FiLM--> frozen_processor --> h_{k+1} = h_proc   (transformer-aligned)
    Optional ``gated.use_gated_hidden_mix=True`` restores regression-style
    h_{k+1} = (1-g_h)*h_k + g_h*mix_FiLM(h_proc).
    v_pred = decoder(h_K) once after the loop (regression / transformer pattern).

Legacy ``gated.decode_once=False`` (per-level decode + v_acc mixing):
    v_k = decoder(h_proc);  v_acc = gated mix over levels; intermediate loss optional.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unified_learning.models.model import UnifiedModel
from unified_learning.models.loop_conditioner import (
    LoopFiLMConditioner,
    LoopFiLMGatedOutputConditioner,
    LoopTransformerConnector,
)
from unified_learning.data_preparation.chipgen.data_loading_homogeneous import (
    get_encoder_input_dim,
)


def _fm_velocity_mse(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    t_cont: Optional[torch.Tensor] = None,
    use_refinement_weighting: bool = False,
    refinement_weight_strength: float = 0.0,
) -> torch.Tensor:
    sq_err = (v_pred.float() - v_target.float()) ** 2
    if use_refinement_weighting and t_cont is not None:
        sq_err = sq_err * (
            1.0 + refinement_weight_strength * t_cont.float()
        ).unsqueeze(-1)
    if mask is not None and mask.any():
        sq_err = sq_err[mask]
    return sq_err.mean()


def _truncated_geometric_prior(
    K: int, prior_lambda: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    ks = torch.arange(K, device=device, dtype=dtype)
    log_p = ks * torch.log(torch.tensor(1.0 - prior_lambda, device=device, dtype=dtype))
    log_p = log_p + torch.log(torch.tensor(prior_lambda, device=device, dtype=dtype))
    p = torch.exp(log_p)
    return p / p.sum().clamp_min(1e-12)


def _level_intermediate_weights(
    K: int,
    decay: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-level weights for intermediate loss; deeper k get less weight when decay < 1.

    w_k = decay^k  (k=0 is the first loop level, highest weight when decay in (0,1)).
    Weights are normalized to sum to K so the scale matches uniform averaging at decay=1.
    """
    if K <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    if decay >= 1.0:
        return torch.ones(K, device=device, dtype=dtype)
    ks = torch.arange(K, device=device, dtype=dtype)
    w = decay ** ks
    return w * (float(K) / w.sum().clamp_min(1e-12))


def compute_fm_gated_intermediate_loss(
    denoiser: "FMLoopedDenoiserWrapper",
    v_target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    t_cont: Optional[torch.Tensor] = None,
    use_refinement_weighting: bool = False,
    refinement_weight_strength: float = 0.0,
) -> torch.Tensor:
    """Supervise gated v_final; optionally every level velocity v_k (decay-weighted)."""
    aux = denoiser._last_gated_aux
    if aux is None:
        raise RuntimeError(
            "denoiser.loop_gated_output_enabled=True but _last_gated_aux is missing"
        )

    v_per_iter = aux["v_per_iter"]
    v_final = aux["v_final"]
    loss_final = _fm_velocity_mse(
        v_final,
        v_target,
        mask=mask,
        t_cont=t_cont,
        use_refinement_weighting=use_refinement_weighting,
        refinement_weight_strength=refinement_weight_strength,
    )
    w_final = float(denoiser.loop_gated_final_loss_weight)
    if not denoiser.loop_gated_supervise_intermediate_levels:
        return w_final * loss_final

    K = v_per_iter.shape[0]
    level_weights = _level_intermediate_weights(
        K,
        float(denoiser.loop_gated_level_loss_decay),
        device=v_per_iter.device,
        dtype=torch.float32,
    )
    level_losses = []
    for k in range(K):
        level_losses.append(
            _fm_velocity_mse(
                v_per_iter[k],
                v_target,
                mask=mask,
                t_cont=t_cont,
                use_refinement_weighting=use_refinement_weighting,
                refinement_weight_strength=refinement_weight_strength,
            )
        )
    stacked = torch.stack(level_losses)
    loss_levels = (level_weights * stacked).sum() / level_weights.sum().clamp_min(1e-12)
    return loss_levels + w_final * loss_final


def compute_fm_ponder_loss(
    denoiser: "FMLoopedDenoiserWrapper",
    v_target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    t_cont: Optional[torch.Tensor] = None,
    use_refinement_weighting: bool = False,
    refinement_weight_strength: float = 0.0,
) -> torch.Tensor:
    """PonderNet-weighted FM velocity loss + optional KL regulariser."""
    aux = denoiser._last_ponder_aux
    if aux is None:
        raise RuntimeError(
            "denoiser.loop_pondernet_enabled=True but _last_ponder_aux is missing"
        )

    halt_probs = aux["halt_probs"].float()
    pred_per_iter = aux["pred_per_iter"].float()
    node_batch = aux["node_batch"]

    K = pred_per_iter.shape[0]
    target = v_target.float().unsqueeze(0).expand(K, -1, -1)
    sq_err = (pred_per_iter - target) ** 2
    if use_refinement_weighting and t_cont is not None:
        t_weights = (
            1.0 + refinement_weight_strength * t_cont.float()
        ).unsqueeze(-1).unsqueeze(0)
        sq_err = sq_err * t_weights

    per_node_loss = sq_err.mean(dim=-1)
    if mask is not None and mask.any():
        per_node_loss = per_node_loss[:, mask]
        node_batch_eff = node_batch[mask]
    else:
        node_batch_eff = node_batch

    p_k_node = halt_probs[:, node_batch_eff]
    task_loss = (p_k_node * per_node_loss).sum(dim=0).mean()

    eps = 1e-7
    prior = _truncated_geometric_prior(
        K=K,
        prior_lambda=float(denoiser.loop_halt_prior_lambda),
        device=halt_probs.device,
        dtype=halt_probs.dtype,
    )
    p = halt_probs.clamp_min(eps)
    log_ratio = p.log() - prior.clamp_min(eps).log().unsqueeze(1)
    kl_loss = (p * log_ratio).sum(dim=0).mean()

    reg_weight_target = float(denoiser.loop_halt_reg_weight)
    warmup_steps = int(denoiser.loop_halt_reg_warmup_steps)
    step = int(denoiser._ponder_step.item()) if hasattr(denoiser, "_ponder_step") else 0
    warm_frac = min(1.0, step / max(warmup_steps, 1)) if warmup_steps > 0 else 1.0
    reg_weight = reg_weight_target * warm_frac

    return task_loss + reg_weight * kl_loss


def _resolve_controller_time_embedding_dim(
    model: UnifiedModel, controller_kwargs: Dict[str, Any]
) -> int:
    """Return the time vector dim the controller must expect.

    ``UnifiedModel.time_embedding`` outputs ``hidden_dim`` (typically 256),
    not ``diffusion.time_embedding_dim`` in YAML (that names the MLP width).
    """
    te = getattr(model, "time_embedding", None)
    if te is not None:
        actual = int(te.output_dim)
        cfg_val = controller_kwargs.get("time_embedding_dim")
        if cfg_val is not None and int(cfg_val) != actual:
            print(
                "[FMLoopedDenoiserWrapper] controller_kwargs.time_embedding_dim="
                f"{cfg_val} ignored; using backbone TimeEmbedding.output_dim={actual}"
            )
        return actual
    return int(controller_kwargs.get("time_embedding_dim", 128))


class FMLoopedDenoiserWrapper(nn.Module):
    """
    Loops the GNN processor of a pretrained FM UnifiedModel K times.

    Interface (same as UnifiedModelDenoiserWrapper):
        forward(data, z_t, t_continuous) → (v_pred [N,2], None)
    """

    def __init__(
        self,
        model: UnifiedModel,
        K: int = 6,
        freeze_backbone: bool = True,
        freeze_decoder: bool = False,
        controller_kwargs: Optional[Dict[str, Any]] = None,
        halting_kwargs: Optional[Dict[str, Any]] = None,
        gated_kwargs: Optional[Dict[str, Any]] = None,
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self.K = K
        self.gradient_checkpoint = gradient_checkpoint
        d = model.hidden_dim

        self.input_dim = model.input_dim
        self.model = SimpleNamespace(input_dim=model.input_dim)

        ckw = dict(controller_kwargs or {})
        halt_cfg = dict(halting_kwargs or {})
        gate_cfg = dict(gated_kwargs or {})

        controller_mode = str(ckw.pop("mode", "film"))
        halting_enabled = bool(ckw.get("halting_enabled", False))
        if controller_mode == "ponder":
            halting_enabled = True
        if controller_mode == "gated_output" and halting_enabled:
            raise ValueError(
                "controller_kwargs.mode='gated_output' is incompatible with "
                "halting_enabled=True; use gated output mixing instead of PonderNet."
            )

        time_dim = _resolve_controller_time_embedding_dim(model, ckw)
        if ckw.get("use_time_input") is None:
            ckw["use_time_input"] = getattr(model, "time_embedding", None) is not None
        ckw["time_embedding_dim"] = time_dim

        common_kw = dict(
            hidden_dim=d,
            num_iterations=K,
            conditioning_dim=int(ckw.get("conditioning_dim", 64)),
            use_size_input=bool(ckw.get("use_size_input", True)),
            use_time_input=bool(ckw.get("use_time_input", False)),
            time_embedding_dim=time_dim,
            zero_init_film=bool(ckw.get("zero_init_film", True)),
        )

        if controller_mode == "gated_output":
            out_dim = int(getattr(model, "output_dim", 2))
            use_raw_input = bool(ckw.get("use_raw_input", False))
            self.controller = LoopFiLMGatedOutputConditioner(
                **common_kw,
                velocity_dim=out_dim,
                use_mix_film=bool(ckw.get("use_mix_film", True)),
                init_gate_bias=float(ckw.get("init_gate_bias", 2.0)),
                node_wise_gates=bool(ckw.get("node_wise_gates", False)),
                gate_use_hidden_state=bool(ckw.get("gate_use_hidden_state", True)),
                use_controller_memory=bool(ckw.get("use_controller_memory", False)),
                controller_memory_dim=ckw.get("controller_memory_dim"),
                memory_zero_init=bool(ckw.get("memory_zero_init", True)),
                memory_use_hidden_write=bool(ckw.get("memory_use_hidden_write", False)),
                memory_write_gain=float(ckw.get("memory_write_gain", 1.0)),
                memory_read_gain=float(ckw.get("memory_read_gain", 1.0)),
                memory_clip=ckw.get("memory_clip"),
                use_raw_input=use_raw_input,
                raw_input_dim=int(model.input_dim) if use_raw_input else None,
                use_transformer_prefilm=bool(ckw.get("use_transformer_prefilm", False)),
                transformer_dim=int(ckw.get("transformer_dim", 128)),
                num_heads=int(ckw.get("num_heads", 4)),
                ffn_dim=int(ckw.get("ffn_dim", 256)),
            )
        elif controller_mode == "transformer_loop":
            if halting_enabled:
                raise ValueError(
                    "controller_kwargs.mode='transformer_loop' is incompatible with "
                    "halting_enabled=True."
                )
            self.controller = LoopTransformerConnector(
                hidden_dim=d,
                raw_input_dim=int(model.input_dim),
                num_iterations=K,
                transformer_dim=int(ckw.get("transformer_dim", 128)),
                num_heads=int(ckw.get("num_heads", 4)),
                ffn_dim=int(ckw.get("ffn_dim", 256)),
                conditioning_dim=int(ckw.get("conditioning_dim", 64)),
                use_size_input=bool(ckw.get("use_size_input", True)),
                use_time_input=bool(ckw.get("use_time_input", False)),
                time_embedding_dim=time_dim,
                zero_init_output=bool(ckw.get("zero_init_output", True)),
            )
        else:
            self.controller = LoopFiLMConditioner(
                **common_kw,
                halting_enabled=halting_enabled,
                init_halt_bias=float(ckw.get("init_halt_bias", -2.0)),
            )

        self.controller_mode = controller_mode
        self.loop_gated_output_enabled = controller_mode == "gated_output"
        self.loop_gated_decode_once = bool(gate_cfg.get("decode_once", True))
        self.loop_gated_use_hidden_mix = bool(gate_cfg.get("use_gated_hidden_mix", False))
        self.loop_gated_final_loss_weight = float(gate_cfg.get("final_loss_weight", 1.0))
        self.loop_gated_level_loss_decay = float(gate_cfg.get("level_loss_decay", 1.0))
        self.loop_gated_supervise_intermediate_levels = bool(
            gate_cfg.get("supervise_intermediate_levels", True)
        )
        if self.loop_gated_decode_once and self.loop_gated_supervise_intermediate_levels:
            self.loop_gated_supervise_intermediate_levels = False
        self._last_gated_aux: Optional[Dict[str, torch.Tensor]] = None

        self.loop_pondernet_enabled = halting_enabled and controller_mode == "ponder"
        self.loop_halt_prior_lambda = float(halt_cfg.get("prior_lambda", 0.4))
        self.loop_halt_reg_weight = float(halt_cfg.get("reg_weight", 0.01))
        self.loop_halt_reg_warmup_steps = int(halt_cfg.get("reg_warmup_steps", 1000))
        self.register_buffer("_ponder_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._last_ponder_aux: Optional[Dict[str, torch.Tensor]] = None

        if freeze_backbone:
            self.freeze_backbone()
        if freeze_decoder:
            self.freeze_decoder()
        else:
            self.unfreeze_decoder()

    def freeze_backbone(self) -> None:
        """Freeze encoder + GNN processor + time embedding."""
        parts = [
            self._model.encoder,
            getattr(self._model, "coord_fourier", None),
            self._model.gnn_blocks,
            getattr(self._model, "global_modules", None),
            getattr(self._model, "time_embedding", None),
        ]
        for part in parts:
            if part is None:
                continue
            for p in part.parameters():
                p.requires_grad_(False)

    def freeze_decoder(self) -> None:
        for p in self._model.decoder.parameters():
            p.requires_grad_(False)

    def unfreeze_decoder(self) -> None:
        for p in self._model.decoder.parameters():
            p.requires_grad_(True)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    @staticmethod
    def _raw_node_features(data) -> torch.Tensor:
        """Raw encoder input features (before MLP / Fourier), matching regression loop."""
        if hasattr(data, "node_types") or (
            hasattr(data, "__contains__") and "inst" in data
        ):
            return data["inst"].x
        return data.x

    def _processor_pass(
        self,
        h: torch.Tensor,
        data,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        time_embed: Optional[torch.Tensor],
        step_idx: Optional[int],
        pos_for_global: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One pass of the frozen GNN processor (matches UnifiedModel.forward)."""
        m = self._model
        size_cond_node = None
        if getattr(m, "size_conditioning_enabled", False):
            size_cond_node = m.size_conditioner(
                batch=getattr(data, "batch", None),
                edge_index=edge_index,
                num_nodes=h.shape[0],
                use_num_edges=getattr(m, "size_conditioning_use_num_edges", False),
            )
        return m.run_homogeneous_processor_blocks(
            h,
            data,
            edge_index,
            edge_attr=edge_attr,
            pos_for_global=pos_for_global,
            time_embed_for_global=time_embed,
            step_idx_for_global=step_idx,
            t_continuous=t_continuous,
            size_cond_node=size_cond_node,
        )

    def _processor_pass_ckpt(
        self,
        h: torch.Tensor,
        data,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        time_embed: Optional[torch.Tensor],
        step_idx: Optional[int],
        use_ckpt: bool,
        pos_for_global: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_ckpt:
            return checkpoint(
                lambda inp: self._processor_pass(
                    inp,
                    data,
                    edge_index,
                    edge_attr,
                    time_embed,
                    step_idx,
                    pos_for_global,
                    t_continuous,
                ),
                h,
                use_reentrant=False,
            )
        return self._processor_pass(
            h,
            data,
            edge_index,
            edge_attr,
            time_embed,
            step_idx,
            pos_for_global,
            t_continuous,
        )

    def _run_loop_iteration(
        self,
        h: torch.Tensor,
        k: int,
        *,
        data,
        batch,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        time_embed: Optional[torch.Tensor],
        step_idx: Optional[int],
        use_ckpt: bool,
        pos_for_global: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, halt_logit = self.controller(
            h,
            iteration=k,
            batch=batch,
            edge_index=edge_index,
            num_nodes=num_nodes,
            time_embed=time_embed,
        )
        h = self._processor_pass_ckpt(
            h,
            data,
            edge_index,
            edge_attr,
            time_embed,
            step_idx,
            use_ckpt,
            pos_for_global,
            t_continuous,
        )
        return h, halt_logit

    def _run_transformer_loop_iteration(
        self,
        h: torch.Tensor,
        k: int,
        *,
        raw_x: torch.Tensor,
        connector_memory: Optional[torch.Tensor],
        data,
        batch,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        time_embed: Optional[torch.Tensor],
        step_idx: Optional[int],
        use_ckpt: bool,
        pos_for_global: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        controller = self.controller
        assert isinstance(controller, LoopTransformerConnector)
        h, connector_memory = controller(
            h,
            raw_x,
            connector_memory,
            iteration=k,
            batch=batch,
            edge_index=edge_index,
            num_nodes=num_nodes,
            time_embed=time_embed,
        )
        h = self._processor_pass_ckpt(
            h,
            data,
            edge_index,
            edge_attr,
            time_embed,
            step_idx,
            use_ckpt,
            pos_for_global,
            t_continuous,
        )
        return h, connector_memory

    def _forward_gated_output(
        self,
        h: torch.Tensor,
        *,
        data,
        batch,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        time_embed: Optional[torch.Tensor],
        step_idx: Optional[int],
        use_ckpt: bool,
        pos_for_global: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = self._model
        controller = self.controller
        assert isinstance(controller, LoopFiLMGatedOutputConditioner)

        h_curr = h
        v_acc: Optional[torch.Tensor] = None
        v_per_level: List[torch.Tensor] = []
        gate_h_list: List[torch.Tensor] = []
        gate_v_list: List[torch.Tensor] = []
        node_batch: Optional[torch.Tensor] = None
        memory: Optional[torch.Tensor] = None
        raw_x = self._raw_node_features(data) if controller.use_raw_input else None
        decode_once = self.loop_gated_decode_once
        use_hidden_mix = self.loop_gated_use_hidden_mix

        for k in range(self.K):
            h_film, per_graph_cond, per_node_cond, node_batch, memory = controller.apply_film(
                h_curr,
                iteration=k,
                batch=batch,
                edge_index=edge_index,
                num_nodes=num_nodes,
                time_embed=time_embed,
                memory=memory,
                raw_x=raw_x,
            )
            num_graphs = int(per_graph_cond.shape[0])
            h_proc = self._processor_pass_ckpt(
                h_film,
                data,
                edge_index,
                edge_attr,
                time_embed,
                step_idx,
                use_ckpt,
                pos_for_global,
                t_continuous,
            )
            if use_hidden_mix:
                h_curr, gate_h = controller.mix_hidden(
                    h_curr,
                    h_proc,
                    per_node_cond,
                    per_graph_cond,
                    node_batch,
                    num_graphs,
                )
                gate_h_list.append(gate_h)
            else:
                h_curr = h_proc

            if not decode_once:
                v_k = m.decoder(h_proc)
                v_acc, gate_v = controller.mix_velocity(
                    v_acc,
                    v_k,
                    per_node_cond,
                    per_graph_cond,
                    node_batch,
                    num_graphs,
                    h_state=h_curr,
                )
                v_per_level.append(v_k)
                if gate_v is not None:
                    gate_v_list.append(gate_v)

            memory = controller.step_memory(memory, per_node_cond, hidden_state=h_curr)

        assert node_batch is not None
        if decode_once:
            v_pred = m.decoder(h_curr)
            self._last_gated_aux = {
                "v_final": v_pred.float(),
                "node_batch": node_batch,
                "gate_h": (
                    torch.stack(gate_h_list, dim=0).float()
                    if gate_h_list
                    else None
                ),
                "gate_v": None,
                "v_per_iter": None,
            }
            return v_pred

        assert v_acc is not None
        self._last_gated_aux = {
            "v_per_iter": torch.stack(v_per_level, dim=0).float(),
            "v_final": v_acc.float(),
            "node_batch": node_batch,
            "gate_h": torch.stack(gate_h_list, dim=0).float(),
            "gate_v": (
                torch.stack(gate_v_list, dim=0).float()
                if gate_v_list
                else None
            ),
        }
        return v_acc

    def _forward_ponder(
        self,
        h: torch.Tensor,
        *,
        data,
        batch,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        time_embed: Optional[torch.Tensor],
        step_idx: Optional[int],
        use_ckpt: bool,
        pos_for_global: Optional[torch.Tensor] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = self._model
        pred_per_iter: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []

        _stats, node_batch, num_graphs = self.controller._graph_size_features(
            batch, edge_index, num_nodes
        )
        remainders = torch.ones(
            num_graphs, device=h.device, dtype=torch.float32
        )

        for k in range(self.K):
            h, halt_logit_k = self._run_loop_iteration(
                h,
                k,
                data=data,
                batch=batch,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                time_embed=time_embed,
                step_idx=step_idx,
                use_ckpt=use_ckpt,
                pos_for_global=pos_for_global,
                t_continuous=t_continuous,
            )
            pred_per_iter.append(m.decoder(h))

            halt_logit_k_fp32 = halt_logit_k.float()
            if k == self.K - 1:
                lam_k = torch.ones_like(halt_logit_k_fp32)
            else:
                lam_k = torch.sigmoid(halt_logit_k_fp32)
            p_k = remainders * lam_k
            halt_probs_list.append(p_k)
            remainders = remainders * (1.0 - lam_k)

        halt_probs = torch.stack(halt_probs_list, dim=0)
        pred_per_iter_t = torch.stack(pred_per_iter, dim=0)
        pred_per_iter_fp32 = pred_per_iter_t.float()
        p_k_node = halt_probs[:, node_batch]
        expected_pred = (p_k_node.unsqueeze(-1) * pred_per_iter_fp32).sum(dim=0)

        self._last_ponder_aux = {
            "halt_probs": halt_probs,
            "pred_per_iter": pred_per_iter_fp32,
            "node_batch": node_batch,
        }
        if self.training:
            self._ponder_step += 1

        return expected_pred.to(pred_per_iter_t.dtype)

    def forward(
        self,
        data,
        z_t: torch.Tensor,
        t_continuous: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        m = self._model
        batch = getattr(data, "batch", None)
        num_nodes = int(z_t.shape[0])

        h, _ = m.forward(
            data,
            x_t=z_t,
            t_continuous=t_continuous,
            return_encoder_output=True,
        )

        time_embed = None
        step_idx = None
        if getattr(m, "time_embedding", None) is not None:
            time_embed = m.time_embedding(t_continuous)
            diff_cfg = m.config.get("diffusion", {})
            num_steps = int(diff_cfg.get("num_steps", 1000))
            t_mean = float(t_continuous.mean().detach().item())
            step_idx = int(round(max(0.0, min(1.0, t_mean)) * max(num_steps - 1, 0)))

        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        use_ckpt = (
            self.gradient_checkpoint
            and self.training
            and torch.is_grad_enabled()
        )

        if self.loop_gated_output_enabled:
            self._last_ponder_aux = None
            v_pred = self._forward_gated_output(
                h,
                data=data,
                batch=batch,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                time_embed=time_embed,
                step_idx=step_idx,
                use_ckpt=use_ckpt,
                pos_for_global=z_t,
                t_continuous=t_continuous,
            )
            return v_pred, None

        if self.loop_pondernet_enabled:
            self._last_gated_aux = None
            v_pred = self._forward_ponder(
                h,
                data=data,
                batch=batch,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                time_embed=time_embed,
                step_idx=step_idx,
                use_ckpt=use_ckpt,
                pos_for_global=z_t,
                t_continuous=t_continuous,
            )
            return v_pred, None

        self._last_ponder_aux = None
        self._last_gated_aux = None
        raw_x = self._raw_node_features(data)
        connector_memory = None
        for k in range(self.K):
            if self.controller_mode == "transformer_loop":
                h, connector_memory = self._run_transformer_loop_iteration(
                    h,
                    k,
                    raw_x=raw_x,
                    connector_memory=connector_memory,
                    data=data,
                    batch=batch,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=num_nodes,
                    time_embed=time_embed,
                    step_idx=step_idx,
                    use_ckpt=use_ckpt,
                    pos_for_global=z_t,
                    t_continuous=t_continuous,
                )
            else:
                h, _ = self._run_loop_iteration(
                    h,
                    k,
                    data=data,
                    batch=batch,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=num_nodes,
                    time_embed=time_embed,
                    step_idx=step_idx,
                    use_ckpt=use_ckpt,
                    pos_for_global=z_t,
                    t_continuous=t_continuous,
                )

        v_pred = m.decoder(h)
        return v_pred, None


def load_fm_looped_denoiser(
    fm_checkpoint_path: str,
    K: int = 6,
    freeze_backbone: bool = True,
    freeze_decoder: bool = False,
    controller_kwargs: Optional[Dict[str, Any]] = None,
    halting_kwargs: Optional[Dict[str, Any]] = None,
    gated_kwargs: Optional[Dict[str, Any]] = None,
    gradient_checkpoint: bool = False,
    device: Optional[torch.device] = None,
) -> FMLoopedDenoiserWrapper:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(fm_checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg: Dict[str, Any] = dict(ckpt.get("config", {}))

    enc_cfg = ckpt_cfg.get("encoder", {})
    if enc_cfg.get("input_dim") in (None, "auto"):
        auto_dim = get_encoder_input_dim(ckpt_cfg.get("dataset", {}))
        ckpt_cfg.setdefault("encoder", {})["input_dim"] = auto_dim

    model = UnifiedModel(ckpt_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[load_fm_looped_denoiser] FM backbone loaded from {fm_checkpoint_path}")

    wrapper = FMLoopedDenoiserWrapper(
        model=model,
        K=K,
        freeze_backbone=freeze_backbone,
        freeze_decoder=freeze_decoder,
        controller_kwargs=controller_kwargs,
        halting_kwargs=halting_kwargs,
        gated_kwargs=gated_kwargs,
        gradient_checkpoint=gradient_checkpoint,
    ).to(device)

    total = sum(p.numel() for p in wrapper.parameters())
    trained = wrapper.trainable_param_count()
    print(
        f"[load_fm_looped_denoiser] Params: total={total:,}  "
        f"trainable={trained:,}  ({100 * trained / total:.1f}% trained)"
    )
    print(f"[load_fm_looped_denoiser] Controller mode: {wrapper.controller_mode}")
    if wrapper.controller_mode == "transformer_loop":
        ctrl = wrapper.controller
        assert isinstance(ctrl, LoopTransformerConnector)
        print(
            f"[load_fm_looped_denoiser] Transformer loop connector "
            f"(d_model={ctrl.transformer_dim}, raw_input_dim={ctrl.feat_mlp[0].in_features}, "
            f"use_time_input={ctrl.use_time_input})"
        )
    if wrapper.loop_gated_output_enabled:
        ctrl = wrapper.controller
        mix_film = getattr(ctrl, "use_mix_film", False)
        use_raw = getattr(ctrl, "use_raw_input", False)
        use_prefilm = getattr(ctrl, "use_transformer_prefilm", False)
        print(
            f"[load_fm_looped_denoiser] Gated output "
            f"(decode_once={wrapper.loop_gated_decode_once}, "
            f"use_gated_hidden_mix={wrapper.loop_gated_use_hidden_mix}, "
            f"mix_film={mix_film}, use_raw_input={use_raw}, "
            f"use_transformer_prefilm={use_prefilm})"
        )
        if not wrapper.loop_gated_decode_once:
            inter = wrapper.loop_gated_supervise_intermediate_levels
            print(
                f"  Legacy v_acc path: supervise_intermediate={inter}, "
                f"level_loss_decay={wrapper.loop_gated_level_loss_decay}, "
                f"final_loss_weight={wrapper.loop_gated_final_loss_weight}"
            )
    if wrapper.loop_pondernet_enabled:
        print(
            f"[load_fm_looped_denoiser] PonderNet halting enabled "
            f"(prior_lambda={wrapper.loop_halt_prior_lambda}, "
            f"reg_weight={wrapper.loop_halt_reg_weight})"
        )
    return wrapper


def load_fm_looped_denoiser_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> FMLoopedDenoiserWrapper:
    """Load a trained looped-FM checkpoint (controller + optional decoder weights)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {checkpoint_path} must contain 'config' and 'model_state_dict'")

    config: Dict[str, Any] = ckpt["config"]
    bkb_cfg = config.get("looped_fm_backbone")
    if not bkb_cfg:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a looped FM checkpoint "
            "(missing looped_fm_backbone in config)."
        )

    denoiser = load_fm_looped_denoiser(
        fm_checkpoint_path=bkb_cfg["checkpoint_path"],
        K=int(bkb_cfg.get("K", 6)),
        freeze_backbone=False,
        freeze_decoder=False,
        controller_kwargs=bkb_cfg.get("controller_kwargs"),
        halting_kwargs=bkb_cfg.get("halting"),
        gated_kwargs=bkb_cfg.get("gated"),
        gradient_checkpoint=bool(bkb_cfg.get("gradient_checkpoint", False)),
        device=torch.device("cpu"),
    )
    denoiser.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[load_fm_looped_denoiser_from_checkpoint] Loaded {checkpoint_path}")
    return denoiser.to(device)


LoopedFMDenoiserWrapper = FMLoopedDenoiserWrapper
load_looped_fm_denoiser = load_fm_looped_denoiser
