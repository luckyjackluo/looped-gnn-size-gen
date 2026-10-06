#!/usr/bin/env python3
"""Evaluate a common direct ChipGen placement regression model."""

import sys
import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unified_learning.training.chipgen_regression_pipeline import (
    build_model,
    compute_metrics,
    create_dataloader,
    forward_direct_regression,
    load_config,
)


@torch.no_grad()
@torch.no_grad()
def evaluate(model, loader, device: torch.device, conditioning_mode: str):
    model.eval()

    total_rmse_norm = 0.0
    total_mae_norm = 0.0
    total_rmse_physical = 0.0
    total_mae_physical = 0.0
    # Predicted-only metrics: error over just the nodes the model must PLACE (pos_mask / non-fixed),
    # excluding ports & macros whose positions are given as input. This is the fair number to compare
    # anchored vs non-anchored datasets, where anchored tiles add ~20% trivially-correct port nodes.
    total_rmse_norm_pred = 0.0
    total_mae_norm_pred = 0.0
    num_batches = 0
    num_pred_batches = 0
    num_graphs = 0

    for batch in tqdm(loader, desc="Eval"):
        batch = batch.to(device)
        pred_norm, pred_physical, target_norm, target_physical = forward_direct_regression(
            model=model,
            batch=batch,
            conditioning_mode=conditioning_mode,
        )
        metrics = compute_metrics(pred_norm, pred_physical, target_norm, target_physical)
        total_rmse_norm += metrics["rmse_norm"]
        total_mae_norm += metrics["mae_norm"]
        total_rmse_physical += metrics["rmse_physical"]
        total_mae_physical += metrics["mae_physical"]

        # predicted-node mask: prefer pos_mask; else everything that isn't a port or macro
        if getattr(batch, "pos_mask", None) is not None:
            pmask = batch.pos_mask.bool()
        else:
            is_p = batch.is_ports.bool() if getattr(batch, "is_ports", None) is not None else torch.zeros(pred_norm.shape[0], dtype=torch.bool, device=pred_norm.device)
            is_m = batch.is_macro.bool() if getattr(batch, "is_macro", None) is not None else torch.zeros_like(is_p)
            pmask = ~(is_p | is_m)
        if pmask.any():
            mm = compute_metrics(pred_norm[pmask], pred_physical[pmask], target_norm[pmask], target_physical[pmask])
            total_rmse_norm_pred += mm["rmse_norm"]
            total_mae_norm_pred += mm["mae_norm"]
            num_pred_batches += 1

        num_batches += 1
        num_graphs += int(getattr(batch, "num_graphs", 1))

    if num_batches == 0:
        raise ValueError("Evaluation dataloader produced zero batches.")

    npb = max(num_pred_batches, 1)
    return {
        "rmse_norm": total_rmse_norm / num_batches,
        "mae_norm": total_mae_norm / num_batches,
        "rmse_physical": total_rmse_physical / num_batches,
        "mae_physical": total_mae_physical / num_batches,
        "rmse_norm_predicted": total_rmse_norm_pred / npb,
        "mae_norm_predicted": total_mae_norm_pred / npb,
        "num_batches": num_batches,
        "num_graphs": num_graphs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct ChipGen placement regression model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="val", choices=["train", "val"], help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save metrics json")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda:0" or "cpu"')
    args = parser.parse_args()

    file_config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})

    # Start from checkpoint config (architecture + training-time dataset). Merge eval YAML
    # `dataset` keys on top so partial overrides (e.g. only val_dir) cannot drop node_features,
    # metis_max_k, require_aux_features, etc.
    if isinstance(checkpoint_config, dict) and checkpoint_config:
        config = deepcopy(checkpoint_config)
        file_ds = file_config.get("dataset")
        if isinstance(file_ds, dict):
            merged_ds = dict(config.get("dataset", {}))
            merged_ds.update(file_ds)
            config["dataset"] = merged_ds

        # Honor eval-time top-level batch knobs from the eval YAML. These otherwise default from the
        # checkpoint's TRAINING config (e.g. batch_size=512, use_size_bucketing on) which OOMs on
        # dense DEHNN data; the eval driver sets batch_size=1 / bucketing off and must be respected.
        for _k in ("batch_size", "use_size_bucketing", "max_total_nodes"):
            if _k in file_config:
                config[_k] = file_config[_k]

        # Also honor eval-time overrides for `processor.global_module.params`
        # (e.g. raising `max_k` for size-extrapolation bins). Only learnable-shape
        # -agnostic knobs should ever be bumped here; see the ablation driver for
        # the single known use (max_k).
        file_proc = file_config.get("processor")
        if isinstance(file_proc, dict):
            file_gm = file_proc.get("global_module")
            if isinstance(file_gm, dict):
                merged_proc = dict(config.get("processor", {}))
                merged_gm = dict(merged_proc.get("global_module", {}))
                file_params = file_gm.get("params")
                if isinstance(file_params, dict):
                    merged_params = dict(merged_gm.get("params", {}))
                    merged_params.update(file_params)
                    merged_gm["params"] = merged_params
                for k, v in file_gm.items():
                    if k != "params":
                        merged_gm[k] = v
                merged_proc["global_module"] = merged_gm
                config["processor"] = merged_proc
    else:
        config = deepcopy(file_config)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    dataset_cfg = config.get("dataset", {})
    data_path = dataset_cfg.get("val_dir") if args.dataset == "val" else dataset_cfg.get("train_dir")
    if data_path is None:
        data_path = dataset_cfg.get("data_dir")
    if data_path is None:
        raise ValueError("Dataset path is missing in config (train_dir/val_dir/data_dir)")

    specific_files = dataset_cfg.get("val_files") if args.dataset == "val" else dataset_cfg.get("train_files")
    loader = create_dataloader(
        data_path=data_path,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        dataset_cfg=dataset_cfg,
        specific_files=specific_files,
        max_samples=dataset_cfg.get("max_val_samples" if args.dataset == "val" else "max_train_samples"),
    )

    conditioning_mode = config.get("conditioning_coords", "ports_only")
    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        conditioning_mode=conditioning_mode,
    )

    print(
        f"rmse_norm={metrics['rmse_norm']:.6f}, "
        f"rmse_norm_predicted={metrics['rmse_norm_predicted']:.6f}, "
        f"mae_norm={metrics['mae_norm']:.6f}, "
        f"rmse_physical={metrics['rmse_physical']:.6f}, "
        f"mae_physical={metrics['mae_physical']:.6f}, "
        f"num_batches={metrics['num_batches']}, "
        f"num_graphs={metrics['num_graphs']}"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.get("save_dir", "data/chipgen/checkpoints/regression_baseline"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"regression_eval_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
