"""Shared helpers for direct ChipGen placement regression."""

from __future__ import annotations

from copy import deepcopy
from functools import partial

# Bump when load_pickle_file_as_data output on disk (Data fields / x layout) changes so
# regression_preloaded_data/*.pt caches are not reused incorrectly.
REGRESSION_CACHE_REVISION = 6
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unified_learning.data_preparation.chipgen.data_loading_homogeneous import (
    HomogeneousGraphDataset,
    SizeBucketBatchSampler,
    collate_homogeneous_graphs,
    get_encoder_input_dim,
    load_pickle_file_as_data,
)
from unified_learning.models.model import UnifiedModel
from unified_learning.utils.config import load_config, repo_root


def _regression_code_repo_root() -> Path:
    return repo_root()


def _resolve_pickle_paths(data_dir: Path, entries: List[str]) -> List[Path]:
    """Resolve `dataset.train_files` / `val_files` entries to concrete pickle paths.

    - Basename only (e.g. ``foo.pickle``) → ``data_dir / basename`` (``train_dir`` / ``val_dir``).
    - Relative path with multiple segments (e.g. ``data/chipgen/.../foo.pickle``) → resolved under
      the **repository root** (directory containing ``unified_learning/``), **not** under ``data_dir``.
      This is how we mix ``training_packing_alg`` (via basenames) and ``training_packing_alg_large``
      (via repo-relative paths) in one config.
    - Absolute path → used as-is.
    """
    repo_root = _regression_code_repo_root()
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


def _graph_num_nodes_item(item: Tuple[Any, Any]) -> int:
    data, _ = item
    return int(data.x.shape[0])


def _subsample_train_list(
    data_list: List,
    max_samples: int,
    dataset_cfg: dict,
    shuffle: bool,
) -> List:
    """Reduce *data_list* to *max_samples* graphs (reproducible via *train_subsample_seed*)."""
    if len(data_list) <= max_samples:
        return data_list
    strategy = dataset_cfg.get("train_subsample_strategy") or "uniform"
    if strategy not in ("uniform", "stratified_focus"):
        raise ValueError(
            "train_subsample_strategy must be 'uniform' or 'stratified_focus', "
            f"got {strategy!r}"
        )
    seed = int(dataset_cfg.get("train_subsample_seed", 42))
    rng = random.Random(seed)

    if strategy == "stratified_focus":
        fm = int(dataset_cfg.get("train_subsample_focus_min_nodes", 500))
        fM = int(dataset_cfg.get("train_subsample_focus_max_nodes", 850))
        frac = float(dataset_cfg.get("train_subsample_focus_fraction", 0.35))
        n = len(data_list)
        focus_idx = [i for i in range(n) if fm <= _graph_num_nodes_item(data_list[i]) <= fM]
        focus_set = set(focus_idx)
        rest_idx = [i for i in range(n) if i not in focus_set]

        k = max_samples
        target_focus = min(int(round(k * frac)), len(focus_idx))
        target_rest = k - target_focus
        if target_rest > len(rest_idx):
            target_rest = len(rest_idx)
            target_focus = min(k - target_rest, len(focus_idx))
        shortfall = k - target_focus - target_rest
        if shortfall > 0:
            more = min(shortfall, len(focus_idx) - target_focus)
            target_focus += more
            shortfall -= more
        if shortfall > 0:
            more = min(shortfall, len(rest_idx) - target_rest)
            target_rest += more

        picked: List[int] = []
        if target_focus:
            picked.extend(rng.sample(focus_idx, target_focus))
        if target_rest:
            picked.extend(rng.sample(rest_idx, target_rest))
        picked_set = set(picked)
        if len(picked) < k:
            remaining = [i for i in range(n) if i not in picked_set]
            need = k - len(picked)
            if remaining:
                picked.extend(rng.sample(remaining, min(need, len(remaining))))
        if len(picked) > k:
            picked = picked[:k]
        out = [data_list[i] for i in picked]
        if bool(dataset_cfg.get("train_subsample_shuffle", False)) and shuffle:
            rng.shuffle(out)
        return out

    if bool(dataset_cfg.get("train_subsample_shuffle", False)) and shuffle:
        return rng.sample(data_list, max_samples)
    return data_list[:max_samples]


def create_dataloader(
    data_path: str,
    batch_size: int,
    shuffle: bool,
    dataset_cfg: dict,
    specific_files=None,
    max_samples=None,
    use_size_bucketing: bool = False,
    max_total_nodes: int | None = None,
) -> DataLoader:
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = (_regression_code_repo_root() / data_path).resolve()
    split_name = "train" if shuffle else "val"
    kwargs = dict(
        normalize_positions=True,
        device="cpu",
        min_graph_size=dataset_cfg.get("min_graph_size"),
        max_graph_size=dataset_cfg.get("max_graph_size"),
        node_features=dataset_cfg.get("node_features"),
        zero_edge_attr=dataset_cfg.get("zero_edge_attr", False),
        normalize_edge_attr_with_coordinates=dataset_cfg.get(
            "normalize_edge_attr_with_coordinates", True
        ),
        eigenvector_dir=dataset_cfg.get("eigenvector_dir"),
        num_eigenvectors=dataset_cfg.get("num_eigenvectors"),
        metis_max_k=int(dataset_cfg.get("metis_max_k", 64)),
        require_aux_features=bool(dataset_cfg.get("require_aux_features", True)),
        lap_eigenvector_stem=dataset_cfg.get("lap_eigenvector_stem"),
        load_graph_lap_eigenvectors=dataset_cfg.get("load_graph_lap_eigenvectors"),
    )
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", True))
    persistent_workers = bool(dataset_cfg.get("persistent_workers", True)) if num_workers > 0 else False

    # Sign-flip augmentation: randomly negate individual Laplacian eigenvector columns
    # per graph at collation time.  Applied to train only (shuffle=True); val is clean.
    # Only active when metis_partition_pe is in node_features; harmless otherwise.
    _augment_signs = shuffle and bool(dataset_cfg.get("augment_metis_signs", True))
    collate_fn = partial(collate_homogeneous_graphs, augment_metis_signs=_augment_signs)

    def _resolve_files(path: Path, files_override):
        if path.is_file():
            return [path]
        if path.is_dir():
            if files_override:
                return _resolve_pickle_paths(path, files_override)
            return sorted(path.glob("*.pickle"))
        return []

    def _build_cache_paths(path: Path, resolved_files):
        use_cache = bool(dataset_cfg.get("cache_preloaded_data", True))
        if not use_cache:
            return None, [], None

        cache_dir = Path(
            dataset_cfg.get(
                "cache_dir",
                "data/chipgen/cache/regression_preloaded_data",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Keep cache keys stable across runs even if filesystem mtimes drift.
        # Dataset/source refresh is still possible by clearing cache files.
        manifest = {
            "split_name": split_name,
            "data_path": str(path.resolve()),
            "specific_files": [str(f) for f in (specific_files or [])],
            "max_samples": max_samples,
            "train_subsample_strategy": dataset_cfg.get("train_subsample_strategy"),
            "train_subsample_shuffle": bool(dataset_cfg.get("train_subsample_shuffle", False)),
            "train_subsample_seed": int(dataset_cfg.get("train_subsample_seed", 42)),
            "train_subsample_focus_min_nodes": dataset_cfg.get("train_subsample_focus_min_nodes"),
            "train_subsample_focus_max_nodes": dataset_cfg.get("train_subsample_focus_max_nodes"),
            "train_subsample_focus_fraction": dataset_cfg.get("train_subsample_focus_fraction"),
            "loader_kwargs": kwargs,
            "cache_revision": REGRESSION_CACHE_REVISION,
            "resolved_files": [],
        }
        for f in resolved_files:
            manifest["resolved_files"].append(str(f.resolve()))

        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()[:16]
        data_file = cache_dir / f"{split_name}_{cache_key}.pt"
        meta_file = cache_dir / f"{split_name}_{cache_key}.json"

        # Backward-compatible fallback: old key format included source file size/mtime.
        legacy_manifest = dict(manifest)
        legacy_manifest["resolved_files"] = []
        for f in resolved_files:
            try:
                st = f.stat()
                legacy_manifest["resolved_files"].append(
                    {
                        "path": str(f.resolve()),
                        "size": int(st.st_size),
                        "mtime_ns": int(st.st_mtime_ns),
                    }
                )
            except FileNotFoundError:
                legacy_manifest["resolved_files"].append({"path": str(f), "missing": True})
        legacy_json = json.dumps(legacy_manifest, sort_keys=True, separators=(",", ":"))
        legacy_key = hashlib.sha256(legacy_json.encode("utf-8")).hexdigest()[:16]
        legacy_data_file = cache_dir / f"{split_name}_{legacy_key}.pt"

        fallback_data_files = [legacy_data_file] if legacy_data_file != data_file else []
        return data_file, fallback_data_files, (meta_file, manifest)

    resolved_files = _resolve_files(data_path, specific_files)
    cache_data_file, fallback_cache_files, cache_meta = _build_cache_paths(data_path, resolved_files)

    cache_candidates = []
    if cache_data_file is not None:
        cache_candidates.append(cache_data_file)
        cache_candidates.extend(fallback_cache_files)

    def _build_loader(dataset: HomogeneousGraphDataset) -> DataLoader:
        """Build a DataLoader, optionally using SizeBucketBatchSampler."""
        if use_size_bucketing:
            sampler = SizeBucketBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                max_total_nodes=max_total_nodes,
                shuffle=shuffle,
                drop_last=False,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    for candidate in cache_candidates:
        if not candidate.exists():
            continue
        print(f"[cache] Loading preloaded {split_name} data from {candidate}")
        try:
            data_list = torch.load(candidate, map_location="cpu", weights_only=False)
            dataset = HomogeneousGraphDataset(data_list)
            return _build_loader(dataset)
        except Exception as e:
            print(f"[cache] Failed to load cache ({e}), trying next candidate if available")

    strategy = dataset_cfg.get("train_subsample_strategy") or "uniform"
    subsample_shuffle = bool(dataset_cfg.get("train_subsample_shuffle", False)) and shuffle
    need_full_load_for_subsample = max_samples is not None and (
        strategy == "stratified_focus" or subsample_shuffle
    )

    if data_path.is_file():
        data_list = load_pickle_file_as_data(str(data_path), **kwargs)
        if max_samples is not None and len(data_list) > max_samples:
            data_list = _subsample_train_list(data_list, max_samples, dataset_cfg, shuffle)
    elif data_path.is_dir():
        files = resolved_files
        data_list = []
        file_iter = tqdm(files, desc=f"Loading {split_name} files", unit="file")
        for file_path in file_iter:
            data_list.extend(load_pickle_file_as_data(str(file_path), **kwargs))
            if max_samples is not None and not need_full_load_for_subsample and len(data_list) >= max_samples:
                data_list = data_list[:max_samples]
                break
        if max_samples is not None and len(data_list) > max_samples:
            data_list = _subsample_train_list(data_list, max_samples, dataset_cfg, shuffle)
    else:
        raise ValueError(f"Data path not found: {data_path}")

    if cache_data_file is not None:
        print(f"[cache] Saving preloaded {split_name} data to {cache_data_file}")
        torch.save(data_list, cache_data_file)
        if cache_meta is not None:
            meta_file, manifest = cache_meta
            meta_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dataset = HomogeneousGraphDataset(data_list)
    return _build_loader(dataset)


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config.get("dataset", {})
    batch_size = int(config.get("batch_size", 32))
    train_dir = dataset_cfg.get("train_dir") or dataset_cfg.get("data_dir")
    val_dir = dataset_cfg.get("val_dir") or dataset_cfg.get("data_dir")
    if not train_dir or not val_dir:
        raise ValueError("dataset.train_dir and dataset.val_dir are required")

    use_size_bucketing = bool(config.get("use_size_bucketing", False))
    max_total_nodes_raw = config.get("max_total_nodes")
    max_total_nodes = int(max_total_nodes_raw) if max_total_nodes_raw is not None else None

    if use_size_bucketing:
        print(
            f"Size-bucket batching enabled: max_total_nodes="
            f"{max_total_nodes or 'auto (median × batch_size)'}"
        )

    train_loader = create_dataloader(
        train_dir,
        batch_size,
        shuffle=True,
        dataset_cfg=dataset_cfg,
        specific_files=dataset_cfg.get("train_files"),
        max_samples=dataset_cfg.get("max_train_samples"),
        use_size_bucketing=use_size_bucketing,
        max_total_nodes=max_total_nodes,
    )
    val_loader = create_dataloader(
        val_dir,
        batch_size,
        shuffle=False,
        dataset_cfg=dataset_cfg,
        specific_files=dataset_cfg.get("val_files"),
        max_samples=dataset_cfg.get("max_val_samples"),
        use_size_bucketing=use_size_bucketing,
        max_total_nodes=max_total_nodes,
    )
    return train_loader, val_loader


def build_model(config: Dict) -> UnifiedModel:
    _validate_processor_schedule(config)

    dataset_cfg = config.get("dataset", {})
    model_cfg = deepcopy(config)
    model_cfg["task"] = "regression"
    model_cfg.setdefault("output_dim", 2)

    encoder_cfg = model_cfg.setdefault("encoder", {})
    if encoder_cfg.get("input_dim") in (None, "auto"):
        encoder_cfg["input_dim"] = get_encoder_input_dim(dataset_cfg)
        print(f"encoder.input_dim set from dataset config: {encoder_cfg['input_dim']}")

    return UnifiedModel(model_cfg)


def _validate_processor_schedule(config: Dict) -> None:
    """Validate optional schedule constraints for regression processor blocks."""
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

    gnn_blocks = list(range(num_blocks)) if gnn_blocks_cfg is None else [int(i) for i in gnn_blocks_cfg]
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


def get_targets(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    target_physical = batch.pos_target if hasattr(batch, "pos_target") else batch.pos
    if (
        hasattr(batch, "mu")
        and batch.mu is not None
        and hasattr(batch, "s")
        and batch.s is not None
    ):
        target_norm = (target_physical - batch.mu) / batch.s
    else:
        target_norm = target_physical
    return target_norm, target_physical


def build_conditioning_coords(batch, target_norm: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros_like(target_norm)
    if mode == "all_nodes":
        return target_norm.clone()
    if mode == "ports_only":
        coords = torch.zeros_like(target_norm)
        if not hasattr(batch, "is_ports") or batch.is_ports is None:
            raise ValueError("conditioning_coords=ports_only requires batch.is_ports")
        coords[batch.is_ports.bool()] = target_norm[batch.is_ports.bool()]
        return coords
    raise ValueError(
        f"Unknown conditioning_coords='{mode}'. Valid: zeros, ports_only, all_nodes"
    )


def forward_direct_regression(model: UnifiedModel, batch, conditioning_mode: str):
    target_norm, target_physical = get_targets(batch)
    cond_coords = build_conditioning_coords(batch, target_norm, conditioning_mode)

    original_x = batch.x
    batch.x = torch.cat([cond_coords, original_x], dim=1)
    try:
        pred_norm, _ = model(batch, x_t=cond_coords)
    finally:
        batch.x = original_x

    if (
        hasattr(batch, "mu")
        and batch.mu is not None
        and hasattr(batch, "s")
        and batch.s is not None
    ):
        pred_physical = pred_norm * batch.s + batch.mu
    else:
        pred_physical = pred_norm

    return pred_norm, pred_physical, target_norm, target_physical


def _per_node_regression_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str
) -> torch.Tensor:
    """Per-node loss with shape ``[N]`` (mean over the output_dim axis)."""
    if loss_type == "mse":
        elem = F.mse_loss(pred, target, reduction="none")
    elif loss_type == "smooth_l1":
        elem = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type='{loss_type}'. Valid: mse, smooth_l1")
    return elem.mean(dim=-1)


def _truncated_geometric_prior(
    K: int, prior_lambda: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Truncated-and-renormalised geometric distribution over k = 0..K-1.

    p(k) ∝ (1 - prior_lambda)^k * prior_lambda
    """
    ks = torch.arange(K, device=device, dtype=dtype)
    log_p = ks * torch.log(torch.tensor(1.0 - prior_lambda, device=device, dtype=dtype))
    log_p = log_p + torch.log(torch.tensor(prior_lambda, device=device, dtype=dtype))
    p = torch.exp(log_p)
    p = p / p.sum().clamp_min(1e-12)
    return p


def regression_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    loss_type: str = "mse",
    *,
    model: Optional[UnifiedModel] = None,
) -> torch.Tensor:
    """Plain regression loss, or PonderNet-weighted loss when ``model`` halts.

    When ``model.loop_pondernet_enabled`` is True, the loss is the per-iter
    expected loss under the halting distribution::

        L_task = sum_k p_k_node * loss(pred_k, target)         (mean over nodes)
        L_kl   = mean over graphs of KL(p_k(g) || prior(k))    (geometric prior)
        L      = L_task + reg_weight(t) * L_kl

    where ``reg_weight(t)`` is linearly warmed up over the first
    ``reg_warmup_steps`` training-mode forward calls.
    """
    if model is None or not getattr(model, "loop_pondernet_enabled", False):
        if loss_type == "mse":
            return F.mse_loss(pred_norm, target_norm)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_norm, target_norm)
        raise ValueError(f"Unknown loss_type='{loss_type}'. Valid: mse, smooth_l1")

    aux = getattr(model, "_last_ponder_aux", None)
    if aux is None:
        raise RuntimeError(
            "model.loop_pondernet_enabled=True but model._last_ponder_aux is "
            "missing; did the forward pass run?"
        )
    # Pin loss math to fp32 regardless of AMP state. The model's ponder
    # branch already returns fp32 halt_probs / pred_per_iter, but coercing
    # here too makes the loss robust if a future caller skips that.
    halt_probs = aux["halt_probs"].float()          # [K, G]
    pred_per_iter = aux["pred_per_iter"].float()    # [K, N, D]
    node_batch = aux["node_batch"]                  # [N]
    target_norm_fp32 = target_norm.float()          # [N, D]

    K = pred_per_iter.shape[0]
    target_expanded = target_norm_fp32.unsqueeze(0).expand(K, -1, -1)  # [K, N, D]
    per_iter_per_node_loss = _per_node_regression_loss(
        pred_per_iter, target_expanded, loss_type
    )  # [K, N], fp32
    p_k_node = halt_probs[:, node_batch]  # [K, N], fp32
    task_loss = (p_k_node * per_iter_per_node_loss).sum(dim=0).mean()

    # KL(p || prior) per graph, averaged over graphs. Halt probs sum to 1
    # along k by construction (the last lambda is forced to 1). Clamp to a
    # small floor to keep log finite in case a sigmoid saturates and a
    # tail-prob underflows.
    eps = 1e-7
    prior = _truncated_geometric_prior(
        K=K,
        prior_lambda=float(getattr(model, "loop_halt_prior_lambda", 0.4)),
        device=halt_probs.device,
        dtype=halt_probs.dtype,
    )  # [K]
    p = halt_probs.clamp_min(eps)
    log_ratio = p.log() - prior.clamp_min(eps).log().unsqueeze(1)
    kl_per_graph = (p * log_ratio).sum(dim=0)  # [G]
    kl_loss = kl_per_graph.mean()

    # Linear warmup of the KL regulariser. Only training-mode forwards bump
    # the step counter; eval-mode loss uses the fully-warmed weight.
    reg_weight_target = float(getattr(model, "loop_halt_reg_weight", 0.0))
    warmup_steps = int(getattr(model, "loop_halt_reg_warmup_steps", 0))
    step = int(model._ponder_step.item()) if hasattr(model, "_ponder_step") else 0
    if warmup_steps > 0:
        warm_frac = min(1.0, step / max(warmup_steps, 1))
    else:
        warm_frac = 1.0
    reg_weight = reg_weight_target * warm_frac

    return task_loss + reg_weight * kl_loss


def compute_metrics(
    pred_norm: torch.Tensor,
    pred_physical: torch.Tensor,
    target_norm: torch.Tensor,
    target_physical: torch.Tensor,
) -> Dict[str, float]:
    mse_norm = F.mse_loss(pred_norm, target_norm)
    mae_norm = F.l1_loss(pred_norm, target_norm)
    mse_physical = F.mse_loss(pred_physical, target_physical)
    mae_physical = F.l1_loss(pred_physical, target_physical)
    return {
        "rmse_norm": float(torch.sqrt(mse_norm).item()),
        "mae_norm": float(mae_norm.item()),
        "rmse_physical": float(torch.sqrt(mse_physical).item()),
        "mae_physical": float(mae_physical.item()),
    }
