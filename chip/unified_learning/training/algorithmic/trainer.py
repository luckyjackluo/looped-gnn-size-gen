"""Shared training helpers for static algorithmic graph benchmarks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from unified_learning.data_preparation.algorithmic import load_algorithmic_dataset
from unified_learning.models.model import UnifiedModel


def load_algorithmic_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _collate_fn(batch):
    return Batch.from_data_list(batch)


def create_algorithmic_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_graphs = load_algorithmic_dataset(
        data_dir=dataset_cfg["train_dir"],
        specific_files=dataset_cfg.get("train_files"),
        max_samples=dataset_cfg.get("max_train_samples"),
    )
    val_graphs = load_algorithmic_dataset(
        data_dir=dataset_cfg["val_dir"],
        specific_files=dataset_cfg.get("val_files"),
        max_samples=dataset_cfg.get("max_val_samples"),
    )
    if not train_graphs:
        raise ValueError("No training graphs found for algorithmic benchmark")
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader


def build_algorithmic_backbone(config: Dict, sample_batch: Batch) -> UnifiedModel:
    """Build the shared backbone, inferring encoder input_dim from the dataset when needed."""
    model_cfg = deepcopy(config)
    model_cfg["task"] = "regression"
    model_cfg.setdefault("output_dim", 1)

    encoder_cfg = model_cfg.setdefault("encoder", {})
    if encoder_cfg.get("input_dim") in (None, "auto"):
        encoder_cfg["input_dim"] = int(sample_batch.x.shape[-1])
    return UnifiedModel(model_cfg)


def _resolve_source_global_idx(batch: Batch) -> torch.Tensor | None:
    if not hasattr(batch, "source_idx"):
        return None
    source_local = batch.source_idx
    if source_local.dim() == 2 and source_local.size(-1) == 1:
        source_local = source_local.squeeze(-1)
    if source_local.dim() == 0:
        source_local = source_local.unsqueeze(0)
    ptr = batch.ptr
    return ptr[:-1] + source_local.to(ptr.device)


def _compute_loss_and_metrics(
    pred: torch.Tensor,
    batch: Batch,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    supervision_cfg = config.get("supervision", {})
    level = supervision_cfg.get("level", "node")
    task_type = supervision_cfg.get("type", "regression")
    target_key = supervision_cfg.get("target_key", "y_node" if level == "node" else "y_graph")
    mask_key = supervision_cfg.get("mask_key", "target_mask" if level == "node" else None)

    target = getattr(batch, target_key)
    mask = getattr(batch, mask_key) if mask_key and hasattr(batch, mask_key) else None

    if task_type == "regression":
        target = target.to(pred.dtype)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        diff = pred - target
        if mask is not None:
            if mask.dim() == 1 and diff.dim() == 2:
                mask = mask.unsqueeze(-1)
            diff = diff * mask.to(diff.dtype)
            denom = mask.to(diff.dtype).sum().clamp_min(1.0)
        else:
            denom = torch.tensor(float(diff.numel()), device=diff.device, dtype=diff.dtype).clamp_min(1.0)
        mse = (diff.square().sum() / denom)
        rmse = torch.sqrt(mse.clamp_min(0.0))
        mae = diff.abs().sum() / denom
        return mse, {"mse": float(mse.item()), "rmse": float(rmse.item()), "mae": float(mae.item())}

    if task_type == "classification":
        target = target.long()
        if mask is not None:
            mask = mask.bool()
            pred = pred[mask]
            target = target[mask]
        if target.numel() == 0:
            zero = pred.sum() * 0.0
            return zero, {"loss": 0.0, "accuracy": 0.0}
        loss = torch.nn.functional.cross_entropy(pred, target)
        accuracy = (pred.argmax(dim=-1) == target).float().mean()
        return loss, {"loss": float(loss.item()), "accuracy": float(accuracy.item())}

    raise ValueError(f"Unsupported supervision type '{task_type}'")


def _run_algorithmic_epoch(
    *,
    backbone: UnifiedModel,
    head: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: dict,
) -> dict:
    training = optimizer is not None
    backbone.train(training)
    head.train(training)
    totals = None
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            node_emb, _ = backbone(batch, x_t=None, t_continuous=None, return_encoder_output=True)
            supervision_cfg = config.get("supervision", {})
            level = supervision_cfg.get("level", "node")
            kwargs = {"batch": batch.batch}
            source_global_idx = _resolve_source_global_idx(batch)
            if source_global_idx is not None:
                kwargs["source_global_idx"] = source_global_idx

            if level == "node":
                pred = head(node_emb, **kwargs)
            elif level == "graph":
                pred = head(node_emb, batch=batch.batch)
            else:
                raise ValueError(f"Unsupported supervision level '{level}'")

            loss, metrics = _compute_loss_and_metrics(pred, batch, config)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()

        if totals is None:
            totals = {k: 0.0 for k in metrics}
        for key, value in metrics.items():
            totals[key] += float(value)
        num_batches += 1

    if totals is None:
        return {"loss": float("nan")}
    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def train_algorithmic_epoch(
    *,
    backbone: UnifiedModel,
    head: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict:
    return _run_algorithmic_epoch(
        backbone=backbone,
        head=head,
        loader=loader,
        optimizer=optimizer,
        device=device,
        config=config,
    )


@torch.no_grad()
def evaluate_algorithmic_epoch(
    *,
    backbone: UnifiedModel,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict:
    return _run_algorithmic_epoch(
        backbone=backbone,
        head=head,
        loader=loader,
        optimizer=None,
        device=device,
        config=config,
    )
