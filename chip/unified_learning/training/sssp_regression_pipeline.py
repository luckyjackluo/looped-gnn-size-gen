"""Shared helpers for SSSP-on-static-graphs regression training and evaluation.

This is the SSSP analogue of ``chipgen_regression_pipeline.py``: it wires
``UnifiedModel`` (encoder -> processor -> decoder) onto the static algorithmic
SSSP shards produced by ``generate_algorithmic_benchmarks.sh``, enabling the
exact same "pretrain on small N, finetune (loop + tiny FiLM controller) on
larger N" recipe used in ``configs/chipgen/regression/size_adaptation/``.

Key differences from the chipgen pipeline:
- Inputs are PyG ``Data`` objects loaded from ``.pt`` shards (not pickle).
- No port/coordinate conditioning: ``model(batch, x_t=None)`` is called
  directly. The source node identity is encoded inside ``data.x`` via the
  per-node ``source_flag`` channel produced by
  ``unified_learning/data_preparation/algorithmic/static_graph_tasks.py``.
- The training target is per-node SSSP distance ``data.y_node`` with reachable
  nodes selected by ``data.target_mask``; loss is masked MSE.
- Output dim = 1 (scalar distance per node).
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from unified_learning.data_preparation.algorithmic import load_algorithmic_dataset
from unified_learning.models.model import UnifiedModel
from unified_learning.models.sssp_baselines import CoordinateOnlySSSPMLP
from unified_learning.training.sssp_ablation_transforms import apply_sssp_ablation
from unified_learning.utils.config import load_config, repo_root


def _repo_root() -> Path:
    return repo_root()


def _resolve_paths(data_dir: Path, entries: Optional[Iterable[str]]) -> List[Path]:
    """Resolve ``train_files``/``val_files`` entries to concrete shard paths.

    - Basenames resolve under ``data_dir``.
    - Multi-segment relative paths resolve under the repo root.
    - Absolute paths are returned as-is.
    """
    if not entries:
        return []
    repo_root = _repo_root()
    out: List[Path] = []
    for raw in entries:
        p = Path(raw)
        if p.is_absolute():
            out.append(p)
        elif len(p.parts) > 1:
            out.append((repo_root / p).resolve())
        else:
            out.append((data_dir / p).resolve())
    return out


def _filter_graphs_by_size(
    graphs: List[Data],
    min_nodes: Optional[int],
    max_nodes: Optional[int],
) -> List[Data]:
    if min_nodes is None and max_nodes is None:
        return graphs
    out: List[Data] = []
    for g in graphs:
        n = int(g.num_nodes)
        if min_nodes is not None and n < min_nodes:
            continue
        if max_nodes is not None and n > max_nodes:
            continue
        out.append(g)
    return out


def _sssp_collate(batch: List[Data]) -> Batch:
    return Batch.from_data_list(batch)


def create_dataloader(
    data_path: str | Path,
    batch_size: int,
    shuffle: bool,
    dataset_cfg: dict,
    specific_files: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader from a directory of ``.pt`` shards (or one shard file).

    Honours ``min_graph_size`` / ``max_graph_size`` filters from
    ``dataset_cfg`` so the same shard can be sub-sliced into evaluation bins.
    """
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = (_repo_root() / data_path).resolve()

    if data_path.is_file():
        graphs = load_algorithmic_dataset(
            data_dir=data_path.parent,
            specific_files=[data_path.name],
            max_samples=None,
        )
    elif data_path.is_dir():
        if specific_files:
            files = _resolve_paths(data_path, specific_files)
            graphs = []
            for f in files:
                graphs.extend(
                    load_algorithmic_dataset(
                        data_dir=f.parent,
                        specific_files=[f.name],
                        max_samples=None,
                    )
                )
        else:
            graphs = load_algorithmic_dataset(data_dir=data_path, max_samples=None)
    else:
        raise FileNotFoundError(f"SSSP data path not found: {data_path}")

    graphs = _filter_graphs_by_size(
        graphs,
        dataset_cfg.get("min_graph_size"),
        dataset_cfg.get("max_graph_size"),
    )
    graphs = apply_sssp_ablation(
        graphs,
        ablation=dataset_cfg.get("ablation"),
        seed=int(dataset_cfg.get("ablation_seed", 0)),
    )
    if max_samples is not None and len(graphs) > max_samples:
        graphs = graphs[: int(max_samples)]

    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    persistent_workers = (
        bool(dataset_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    )

    return DataLoader(
        graphs,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=_sssp_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config.get("dataset", {})
    batch_size = int(config.get("batch_size", 32))
    train_dir = dataset_cfg.get("train_dir") or dataset_cfg.get("data_dir")
    val_dir = dataset_cfg.get("val_dir") or dataset_cfg.get("data_dir")
    if not train_dir or not val_dir:
        raise ValueError("dataset.train_dir and dataset.val_dir are required")

    train_loader = create_dataloader(
        train_dir,
        batch_size,
        shuffle=True,
        dataset_cfg=dataset_cfg,
        specific_files=dataset_cfg.get("train_files"),
        max_samples=dataset_cfg.get("max_train_samples"),
    )
    val_loader = create_dataloader(
        val_dir,
        batch_size,
        shuffle=False,
        dataset_cfg=dataset_cfg,
        specific_files=dataset_cfg.get("val_files"),
        max_samples=dataset_cfg.get("max_val_samples"),
    )
    return train_loader, val_loader


def _peek_input_dim(loader: DataLoader) -> int:
    """Pull one batch to discover ``data.x`` feature width."""
    sample = next(iter(loader))
    return int(sample.x.shape[-1])


def _peek_coordinate_mlp_input_dim(
    loader: DataLoader, *, include_node_features: bool
) -> int:
    sample = next(iter(loader))
    dim = 4  # source xy + target xy
    if include_node_features:
        dim += int(sample.x.shape[-1]) * 2
    return dim


def _validate_processor_schedule(config: Dict) -> None:
    """Same gate as the chipgen pipeline: enforce GNN-then-global ordering."""
    if not bool(config.get("enforce_gnn_then_global", False)):
        return

    processor_cfg = config.get("processor", {})
    if processor_cfg.get("type") != "gnn":
        raise ValueError("enforce_gnn_then_global requires processor.type='gnn'")

    num_blocks = processor_cfg.get("num_blocks")
    if not isinstance(num_blocks, int) or num_blocks <= 0:
        raise ValueError("enforce_gnn_then_global requires processor.num_blocks > 0")

    gnn_blocks_cfg = processor_cfg.get("gnn_blocks")
    global_blocks_cfg = processor_cfg.get("global_module_blocks")

    gnn_blocks = (
        list(range(num_blocks)) if gnn_blocks_cfg is None else [int(i) for i in gnn_blocks_cfg]
    )
    global_blocks = (
        list(range(num_blocks))
        if processor_cfg.get("use_global_module", False) and global_blocks_cfg is None
        else ([int(i) for i in global_blocks_cfg] if global_blocks_cfg is not None else [])
    )

    gnn_set = set(gnn_blocks)
    global_set = set(global_blocks)

    invalid_gnn = sorted(i for i in gnn_set if i < 0 or i >= num_blocks)
    invalid_global = sorted(i for i in global_set if i < 0 or i >= num_blocks)
    if invalid_gnn:
        raise ValueError(
            f"processor.gnn_blocks has invalid indices {invalid_gnn}; "
            f"valid range is [0, {num_blocks - 1}]"
        )
    if invalid_global:
        raise ValueError(
            f"processor.global_module_blocks has invalid indices {invalid_global}; "
            f"valid range is [0, {num_blocks - 1}]"
        )

    overlap = sorted(gnn_set & global_set)
    if overlap:
        raise ValueError(
            "enforce_gnn_then_global requires non-interleaved blocks; "
            f"found overlap between gnn/global blocks: {overlap}"
        )

    if gnn_set and global_set and max(gnn_set) >= min(global_set):
        raise ValueError(
            "enforce_gnn_then_global requires all GNN blocks to come before global blocks. "
            f"Got gnn_blocks={sorted(gnn_set)} and global_module_blocks={sorted(global_set)}"
        )


def build_model(config: Dict, *, sample_loader: Optional[DataLoader] = None) -> UnifiedModel:
    """Build the UnifiedModel for SSSP regression.

    If ``encoder.input_dim`` is missing or ``"auto"``, peek one batch from
    ``sample_loader`` to discover the feature width. Same convention as the
    algorithmic trainer, but threaded through here so we can also build the
    model from a saved checkpoint config without needing the dataloader.
    """
    model_type = str(config.get("model_type", "unified")).lower()
    if model_type == "coordinate_mlp":
        baseline_cfg = deepcopy(config.get("coordinate_mlp", {}))
        include_node_features = bool(baseline_cfg.get("include_node_features", False))
        input_dim = baseline_cfg.get("input_dim", "auto")
        if input_dim in (None, "auto"):
            if sample_loader is None:
                raise ValueError(
                    "coordinate_mlp.input_dim is 'auto'/missing and no sample_loader was provided"
                )
            input_dim = _peek_coordinate_mlp_input_dim(
                sample_loader,
                include_node_features=include_node_features,
            )
            baseline_cfg["input_dim"] = input_dim
            config.setdefault("coordinate_mlp", {})["input_dim"] = input_dim
            print(f"[sssp] coordinate_mlp.input_dim auto-set to {input_dim}")
        return CoordinateOnlySSSPMLP(
            input_dim=int(input_dim),
            hidden_dim=int(config.get("hidden_dim", baseline_cfg.get("hidden_dim", 256))),
            output_dim=int(config.get("output_dim", 1)),
            num_layers=int(baseline_cfg.get("num_layers", 3)),
            dropout=float(baseline_cfg.get("dropout", 0.0)),
            activation=str(baseline_cfg.get("activation", "silu")),
            include_node_features=include_node_features,
        )

    _validate_processor_schedule(config)

    model_cfg = deepcopy(config)
    model_cfg["task"] = "regression"
    model_cfg.setdefault("output_dim", 1)

    encoder_cfg = model_cfg.setdefault("encoder", {})
    if encoder_cfg.get("input_dim") in (None, "auto"):
        if sample_loader is None:
            raise ValueError(
                "encoder.input_dim is 'auto'/missing and no sample_loader was "
                "provided; either set encoder.input_dim explicitly or pass a loader."
            )
        encoder_cfg["input_dim"] = _peek_input_dim(sample_loader)
        print(f"[sssp] encoder.input_dim auto-set to {encoder_cfg['input_dim']}")

    return UnifiedModel(model_cfg)


def edge_scale_per_node(batch: Batch) -> torch.Tensor:
    """Per-graph mean edge weight broadcast to each node, shape ``[N_total]``.

    Used to normalize SSSP targets to approximate hop counts so the regression
    task stays scale-invariant across graph sizes — critical for spatial graphs
    (RGG) where edge weights shrink as node count grows.  Always computed in
    float32 regardless of the batch tensor dtype to avoid overflow.
    """
    edge_weight = batch.edge_attr.squeeze(-1).to(torch.float32)   # [E]
    edge_to_graph = batch.batch[batch.edge_index[0]]               # [E]
    num_graphs = int(batch.batch.max().item()) + 1

    scale_sum = torch.zeros(num_graphs, device=edge_weight.device, dtype=torch.float32)
    count = torch.zeros(num_graphs, device=edge_weight.device, dtype=torch.float32)
    scale_sum.scatter_add_(0, edge_to_graph, edge_weight)
    count.scatter_add_(0, edge_to_graph, torch.ones_like(edge_weight))
    scale_per_graph = scale_sum / count.clamp_min(1.0)            # [G]
    return scale_per_graph[batch.batch]                            # [N_total]


def forward_sssp(
    model: UnifiedModel,
    batch: Batch,
    *,
    normalize_targets: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``UnifiedModel`` on an SSSP batch and return ``(pred, target, mask)``.

    All three tensors have shape ``[N_total]``. The model's decoder produces
    ``[N_total, 1]``; we squeeze the trailing 1.

    When ``normalize_targets=True`` the SSSP distance target is divided by the
    per-graph mean edge weight before returning, converting it to approximate
    hop counts.  The model prediction is returned as-is (in the same normalized
    space the model was trained in).  Both pred and target are therefore in
    normalized units — consistent for the loss and for training metrics.

    For evaluation in original distance units, call ``edge_scale_per_node`` on
    the batch and multiply both pred and target by the scale after this call.
    """
    pred, _ = model(batch, x_t=None)
    if pred.dim() == 2:
        if pred.shape[-1] != 1:
            raise ValueError(
                f"forward_sssp expects output_dim=1, got tensor shape {tuple(pred.shape)}"
            )
        pred = pred.squeeze(-1)
    target = batch.y_node
    if target.dim() == 2:
        target = target.squeeze(-1)
    target = target.to(pred.dtype)
    mask = batch.target_mask
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if normalize_targets and batch.edge_attr is not None and batch.edge_attr.numel() > 0:
        # Divide target by per-graph mean edge weight → approximate hop counts.
        # pred is left in the same normalized space the model outputs.
        # Both are now in the same scale so the loss is well-conditioned.
        scale = edge_scale_per_node(batch).to(pred.dtype)         # [N_total]
        target = target / scale.clamp_min(1e-6)

    return pred, target, mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask.to(pred.dtype)
    denom = mask.to(pred.dtype).sum().clamp_min(1.0)
    return diff.square().sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask.to(pred.dtype)
    denom = mask.to(pred.dtype).sum().clamp_min(1.0)
    return diff.sum() / denom


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, float]:
    """Return masked MSE/RMSE/MAE on reachable nodes, plus reachable-node count.

    Always casts to float32 before computing squared differences.  This
    prevents float16 sum overflow under AMP: with batch_size=32 and N≈64,
    there are ~2016 masked nodes; diff² can reach 144 (hop-count targets) and
    the sum 290,304 which exceeds float16's max of 65,504 → inf RMSE without
    the cast.
    """
    pred = pred.detach().float()
    target = target.detach().float()
    mask = mask.detach()
    mse = masked_mse_loss(pred, target, mask)
    mae = masked_mae(pred, target, mask)
    rmse = torch.sqrt(mse.clamp_min(0.0))
    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "num_reachable": int(mask.sum().item()),
    }


def _truncated_geometric_prior(
    K: int, prior_lambda: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Truncated-and-renormalised geometric distribution over ``k = 0..K-1``.

    ``p(k) ~ (1 - prior_lambda)^k * prior_lambda``, renormalised so it sums
    to 1 over the truncated support. Mirrors the helper in
    ``chipgen_regression_pipeline.py`` so PonderNet halting on SSSP uses the
    same prior shape as the existing chipgen path.
    """
    ks = torch.arange(K, device=device, dtype=dtype)
    log_one_minus = torch.log(
        torch.tensor(1.0 - prior_lambda, device=device, dtype=dtype)
    )
    log_lambda = torch.log(
        torch.tensor(prior_lambda, device=device, dtype=dtype)
    )
    log_p = ks * log_one_minus + log_lambda
    p = torch.exp(log_p)
    return p / p.sum().clamp_min(1e-12)


def _masked_per_node_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    """Per-node loss with shape ``[N]`` (masked to reachable nodes)."""
    if loss_type == "mse":
        elem = (pred - target).square()
    elif loss_type == "smooth_l1":
        elem = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type='{loss_type}'. Valid: mse, smooth_l1")
    return elem * mask.to(elem.dtype)


def _gate_reg_terms(
    model,
    *,
    step_counter: Optional[torch.Tensor],
) -> torch.Tensor:
    """Optional L0 / entropy regulariser on the gated-loop conditioner.

    Reads ``model._last_gate_aux`` (populated by the gnn_gated_halt_loop
    forward) and the gate-related config knobs stored on the model. Returns
    a scalar (zero if no gate aux exists or weights are zero). Both regs
    have a linear warmup so they don't drown out the task loss early in
    training.
    """
    aux = getattr(model, "_last_gate_aux", None)
    if aux is None:
        return torch.zeros((), device=next(model.parameters()).device)

    l0_weight = float(getattr(model, "loop_gate_l0_weight", 0.0))
    ent_weight = float(getattr(model, "loop_gate_entropy_weight", 0.0))
    if l0_weight <= 0.0 and ent_weight <= 0.0:
        return torch.zeros(
            (), device=aux["gate_logit_per_iter"].device,
            dtype=aux["gate_logit_per_iter"].dtype,
        )

    step = int(step_counter.item()) if step_counter is not None else 0
    out = torch.zeros(
        (), device=aux["gate_logit_per_iter"].device, dtype=torch.float32
    )
    if l0_weight > 0.0:
        warm = int(getattr(model, "loop_gate_l0_warmup_steps", 0))
        frac = min(1.0, step / max(warm, 1)) if warm > 0 else 1.0
        gate_logits = aux["gate_logit_per_iter"].float()  # [K, N]
        cond = model.loop_conditioner
        if hasattr(cond, "hard_concrete_l0_probability"):
            prob_nonzero = cond.hard_concrete_l0_probability(gate_logits)
        else:
            prob_nonzero = torch.sigmoid(gate_logits)
        out = out + (l0_weight * frac) * prob_nonzero.mean()
    if ent_weight > 0.0:
        warm = int(getattr(model, "loop_gate_entropy_warmup_steps", 0))
        frac = min(1.0, step / max(warm, 1)) if warm > 0 else 1.0
        gate = aux["gate_per_iter"].float().clamp(1e-6, 1.0 - 1e-6)  # [K, N]
        entropy = -(gate * gate.log() + (1.0 - gate) * (1.0 - gate).log())
        # Penalise entropy -> pushes gates toward {0, 1}; matches the
        # "actually halt" objective of the hard-gate variant.
        out = out + (ent_weight * frac) * entropy.mean()
    return out


def regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str = "mse",
    model: Optional[UnifiedModel] = None,
) -> torch.Tensor:
    """Masked regression loss, with PonderNet / iterGNN soft halting on top.

    Plain (no model passed, or no halting enabled):
        Masked MSE or masked smooth-L1.

    iterGNN / PonderNet halting (``model.loop_pondernet_enabled=True``,
    populated by the ``gnn_gated_halt_loop`` forward):
        L_task = sum_k p_k_node * masked_loss(pred_k, target)        (expected loss across k)
        L_dist = KL(p_k || truncated geometric)        if distribution == 'pondernet_geometric'
                 OR  -H(p_k)                            if distribution == 'softmax_over_k'
        L_gate = optional L0 (Hard Concrete) or entropy reg from the gate
        L      = L_task + reg_weight(t) * L_dist + L_gate(t)

    ``reg_weight(t)`` and the gate-reg weights both linearly warm up from
    0 over their configured ``*_warmup_steps`` (read off
    ``model._ponder_step``).
    """
    if model is None or not getattr(model, "loop_pondernet_enabled", False):
        # Plain masked-loss path (no per-iter weighting). Variant A
        # (gated_loop without halting) still falls through here because its
        # gate aux is incorporated below via _gate_reg_terms ONLY if the
        # caller decides to include it - we keep this path bit-identical to
        # the legacy implementation so existing PEFT runs don't regress.
        if loss_type == "mse":
            base = masked_mse_loss(pred, target, mask)
        elif loss_type == "smooth_l1":
            elem = _masked_per_node_loss(pred, target, mask, loss_type="smooth_l1")
            denom = mask.to(elem.dtype).sum().clamp_min(1.0)
            base = elem.sum() / denom
        else:
            raise ValueError(
                f"Unknown loss_type='{loss_type}'. Valid: mse, smooth_l1"
            )
        # Variant A: still pick up gate L0/entropy reg if the gate aux is
        # populated AND weights are set. This is what lets the gated_loop
        # config drive its hard gates toward 0 even without halting.
        if model is not None and getattr(model, "_last_gate_aux", None) is not None:
            step_counter = getattr(model, "_ponder_step", None)
            base = base + _gate_reg_terms(model, step_counter=step_counter)
        return base

    aux = getattr(model, "_last_ponder_aux", None)
    if aux is None:
        raise RuntimeError(
            "model.loop_pondernet_enabled=True but model._last_ponder_aux is "
            "missing; did the forward pass run?"
        )
    halt_probs = aux["halt_probs"].float()              # [K, G]
    pred_per_iter = aux["pred_per_iter"].float()        # [K, N, D] or [K, N]
    node_batch = aux["node_batch"]                       # [N]
    distribution = aux.get(
        "distribution", getattr(model, "loop_halt_distribution", "pondernet_geometric")
    )

    if pred_per_iter.dim() == 3 and pred_per_iter.shape[-1] == 1:
        pred_per_iter = pred_per_iter.squeeze(-1)        # [K, N]
    target_fp32 = target.float()                         # [N]
    mask_bool = mask.bool()

    # Per-iter masked per-node loss, [K, N], summed-then-averaged via halt
    # probs so the result is "expected loss across iterations".
    K = pred_per_iter.shape[0]
    target_expanded = target_fp32.unsqueeze(0).expand(K, -1)
    mask_expanded = mask_bool.unsqueeze(0).expand(K, -1)
    per_iter_per_node = _masked_per_node_loss(
        pred_per_iter, target_expanded, mask_expanded, loss_type=loss_type
    )                                                     # [K, N]
    p_k_node = halt_probs[:, node_batch]                  # [K, N]
    denom = mask_bool.to(per_iter_per_node.dtype).sum().clamp_min(1.0)
    task_loss = (p_k_node * per_iter_per_node).sum() / denom

    eps = 1e-7
    p_clamped = halt_probs.clamp_min(eps)
    if distribution == "pondernet_geometric":
        prior = _truncated_geometric_prior(
            K=K,
            prior_lambda=float(getattr(model, "loop_halt_prior_lambda", 0.4)),
            device=halt_probs.device,
            dtype=halt_probs.dtype,
        )                                                  # [K]
        log_ratio = p_clamped.log() - prior.clamp_min(eps).log().unsqueeze(1)
        dist_reg = (halt_probs * log_ratio).sum(dim=0).mean()
    elif distribution == "softmax_over_k":
        # Negative entropy: penalise high entropy => encourages peaked p_k,
        # which is the "decide which iterations matter" signal.
        entropy = -(p_clamped * p_clamped.log()).sum(dim=0).mean()
        dist_reg = -entropy
    else:
        raise ValueError(
            f"Unknown halt distribution '{distribution}'. Valid: "
            "pondernet_geometric, softmax_over_k"
        )

    reg_weight_target = float(getattr(model, "loop_halt_reg_weight", 0.0))
    warmup_steps = int(getattr(model, "loop_halt_reg_warmup_steps", 0))
    step_counter = getattr(model, "_ponder_step", None)
    step = int(step_counter.item()) if step_counter is not None else 0
    warm_frac = min(1.0, step / max(warmup_steps, 1)) if warmup_steps > 0 else 1.0
    reg_weight = reg_weight_target * warm_frac

    gate_reg = _gate_reg_terms(model, step_counter=step_counter)
    return task_loss + reg_weight * dist_reg + gate_reg


__all__ = [
    "build_model",
    "compute_metrics",
    "create_dataloader",
    "create_dataloaders",
    "edge_scale_per_node",
    "forward_sssp",
    "load_config",
    "masked_mse_loss",
    "masked_mae",
    "regression_loss",
]
