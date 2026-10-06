#!/usr/bin/env python3
"""Train a common direct ChipGen placement regression model."""

import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import List

import contextlib
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
    _AUTOCAST_HAS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AUTOCAST_HAS_DEVICE_TYPE = False
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unified_learning.training.chipgen_regression_pipeline import (
    build_model,
    compute_metrics,
    create_dataloaders,
    forward_direct_regression,
    load_config,
    regression_loss,
)
from unified_learning.training.finetune_utils import configure_trainable_components


def _remap_checkpoint_state_dict(state_dict: dict, remap_mode: str, log_print) -> dict:
    """Optionally remap checkpoint keys for additive-module finetuning experiments."""
    if not remap_mode:
        return state_dict

    remapped = {}
    remapped_count = 0
    dropped_count = 0

    if remap_mode == "late_global_insert_local_blocks_4to6":
        for key, value in state_dict.items():
            new_key = key

            if key.startswith("global_modules.4."):
                new_key = key.replace("global_modules.4.", "global_modules.6.", 1)
                remapped_count += 1
            elif key.startswith("global_modules.5."):
                new_key = key.replace("global_modules.5.", "global_modules.7.", 1)
                remapped_count += 1
            elif key.startswith("gnn_blocks."):
                parts = key.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    # Old blocks 4..5 were never executed in the late-global baseline.
                    # Drop them so the inserted local blocks are truly new parameters.
                    if 12 <= layer_idx <= 17:
                        dropped_count += 1
                        continue

            remapped[new_key] = value

    elif remap_mode == "late_global_insert_local_blocks_4to8":
        # Insert 4 new local GNN blocks before the global stage.
        # Old model: 4 GNN blocks (layers 0–11) + 2 global blocks (positions 4,5).
        # New model: 8 GNN blocks (layers 0–23) + 2 global blocks (positions 8,9).
        # - Keep gnn_blocks[0..11] (original 4 GNN blocks × 3 layers).
        # - Drop gnn_blocks[12..23]: these are new blocks — initialized fresh.
        # - Remap global_modules.4 → global_modules.8, global_modules.5 → global_modules.9.
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("global_modules.4."):
                new_key = key.replace("global_modules.4.", "global_modules.8.", 1)
                remapped_count += 1
            elif key.startswith("global_modules.5."):
                new_key = key.replace("global_modules.5.", "global_modules.9.", 1)
                remapped_count += 1
            elif key.startswith("gnn_blocks."):
                parts = key.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    if 12 <= layer_idx <= 23:
                        dropped_count += 1
                        continue
            remapped[new_key] = value

    elif remap_mode == "late_global_insert_local_blocks_8to12":
        # Insert 4 new local GNN blocks before the global stage.
        # Old model: 8 GNN blocks (layers 0–23) + 2 global blocks (positions 8,9).
        # New model: 12 GNN blocks (layers 0–35) + 2 global blocks (positions 12,13).
        # - Keep gnn_blocks[0..23] (original 8 GNN blocks × 3 layers).
        # - Drop gnn_blocks[24..35]: these are new blocks — initialized fresh.
        # - Remap global_modules.8 → global_modules.12, global_modules.9 → global_modules.13.
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("global_modules.8."):
                new_key = key.replace("global_modules.8.", "global_modules.12.", 1)
                remapped_count += 1
            elif key.startswith("global_modules.9."):
                new_key = key.replace("global_modules.9.", "global_modules.13.", 1)
                remapped_count += 1
            elif key.startswith("gnn_blocks."):
                parts = key.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    if 24 <= layer_idx <= 35:
                        dropped_count += 1
                        continue
            remapped[new_key] = value

    elif remap_mode == "add_gnn_blocks_2to4":
        # Size-adaptation: double the pure-GNN block count before a single global block.
        # Old model: 2 GNN blocks + 1 global block
        #   processor.num_blocks=3, gnn_blocks=[0,1], global_module_blocks=[2]
        #   => gnn_blocks.0..5 executed, gnn_blocks.6..8 dead dummies, global_modules.2 = transformer.
        # New model: 4 GNN blocks + 1 global block
        #   processor.num_blocks=5, gnn_blocks=[0,1,2,3], global_module_blocks=[4]
        #   => gnn_blocks.0..11 all executed, gnn_blocks.12..14 dead, global_modules.4 = transformer.
        # Mapping:
        #   - keep gnn_blocks.0..5 at same indices (original 2 GNN blocks).
        #   - drop gnn_blocks.6..8 (old dead block-2 slots).
        #   - remap global_modules.2.* -> global_modules.4.*.
        #   - new gnn_blocks.6..11 (2 new GNN blocks) and 12..14 (new dead slots) init fresh.
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("global_modules.2."):
                new_key = key.replace("global_modules.2.", "global_modules.4.", 1)
                remapped_count += 1
            elif key.startswith("gnn_blocks."):
                parts = key.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    if 6 <= layer_idx <= 8:
                        dropped_count += 1
                        continue
            remapped[new_key] = value

    elif remap_mode == "late_global_expand_local_layers_per_block_3to4":
        for key, value in state_dict.items():
            new_key = key

            if key.startswith("gnn_blocks."):
                parts = key.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    old_idx = int(parts[1])
                    old_block = old_idx // 3
                    old_layer = old_idx % 3
                    new_idx = old_block * 4 + old_layer
                    new_key = key.replace(f"gnn_blocks.{old_idx}.", f"gnn_blocks.{new_idx}.", 1)
                    if new_key != key:
                        remapped_count += 1

            remapped[new_key] = value
    else:
        raise ValueError(f"Unknown checkpoint_remap='{remap_mode}'")

    log_print(
        f"Applied checkpoint remap '{remap_mode}': "
        f"{remapped_count} remapped keys, {dropped_count} dropped keys"
    )
    return remapped
def run_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    conditioning_mode: str,
    loss_type: str,
    scaler: GradScaler,
    use_amp: bool,
    training: bool,
):
    model.train() if training else model.eval()

    def _autocast_ctx():
        enabled = use_amp and device.type == "cuda"
        if _AUTOCAST_HAS_DEVICE_TYPE:
            return autocast(device_type="cuda", enabled=enabled)
        return autocast(enabled=enabled)

    total_loss = 0.0
    total_rmse_norm = 0.0
    total_rmse_physical = 0.0
    num_batches = 0

    iterator = tqdm(loader, desc="Train" if training else "Val")
    for batch in iterator:
        batch = batch.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_ctx():
                pred_norm, pred_physical, target_norm, target_physical = forward_direct_regression(
                    model=model,
                    batch=batch,
                    conditioning_mode=conditioning_mode,
                )
                loss = regression_loss(
                    pred_norm, target_norm, loss_type=loss_type, model=model
                )

            if training:
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        metrics = compute_metrics(pred_norm, pred_physical, target_norm, target_physical)
        total_loss += float(loss.item())
        total_rmse_norm += metrics["rmse_norm"]
        total_rmse_physical += metrics["rmse_physical"]
        num_batches += 1
        iterator.set_postfix(
            loss=f"{total_loss / num_batches:.5f}",
            rmse_norm=f"{total_rmse_norm / num_batches:.5f}",
            rmse_phys=f"{total_rmse_physical / num_batches:.5f}",
        )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "rmse_norm": total_rmse_norm / n,
        "rmse_physical": total_rmse_physical / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train direct ChipGen placement regression model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--device", default=None, help='Device, e.g. "cuda:0" or "cpu"')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    save_dir = Path(config.get("save_dir", "data/chipgen/checkpoints/regression_baseline"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training.log"
    num_epochs = int(config.get("num_epochs", 100))
    bootstrap_ckpt = (config.get("checkpoint") or "").strip() or None
    local_resume = save_dir / "best_model.pt"
    cli_ckpt = (args.checkpoint or "").strip() or None
    resume_from_local_progress = False
    resume_done_epochs = None
    if not cli_ckpt and local_resume.exists():
        try:
            stub = torch.load(local_resume, map_location="cpu", weights_only=False)
            done = int(stub.get("epoch", 0))
            if done < num_epochs:
                resume_from_local_progress = True
                resume_done_epochs = done
        except Exception:
            pass

    log_fp = open(log_file, "a" if resume_from_local_progress else "w", encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_fp, flush=True)

    try:
        log_print(f"Using device: {device}")
        log_print(f"Training log: {log_file.resolve()}")
        if resume_from_local_progress:
            log_print("(Appending to existing training log — resuming run.)")

        model = build_model(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        log_print(f"Parameters: {total_params:,}")

        train_loader, val_loader = create_dataloaders(config)
        conditioning_mode = config.get("conditioning_coords", "ports_only")
        loss_type = config.get("loss_type", "mse")

        use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
        if _AUTOCAST_HAS_DEVICE_TYPE:
            scaler = GradScaler("cuda", enabled=use_amp)
        else:
            scaler = GradScaler(enabled=use_amp)
        log_print(f"AMP enabled: {use_amp}")
        log_print(f"conditioning_coords: {conditioning_mode}")

        start_epoch = 0
        best_val_loss = float("inf")
        checkpoint_path = cli_ckpt
        if checkpoint_path:
            pass
        elif resume_from_local_progress:
            checkpoint_path = str(local_resume.resolve())
            log_print(
                f"Resuming from in-run checkpoint {checkpoint_path} "
                f"(completed epochs={resume_done_epochs}, target num_epochs={num_epochs})"
            )
        if not checkpoint_path and bootstrap_ckpt:
            checkpoint_path = bootstrap_ckpt
        elif not checkpoint_path and local_resume.exists() and not resume_from_local_progress:
            checkpoint_path = str(local_resume.resolve())
            log_print(f"Auto-resuming from {checkpoint_path}")

        reset_epochs_on_checkpoint = bool(config.get("reset_epochs_on_checkpoint", False))
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"]
            remap_mode = config.get("checkpoint_remap")
            if remap_mode and not resume_from_local_progress:
                state_dict = _remap_checkpoint_state_dict(state_dict, remap_mode, log_print)
            load_result = model.load_state_dict(state_dict, strict=False)
            log_print(f"Loaded model weights from {checkpoint_path}")
            log_print(
                f"Checkpoint load summary: {len(load_result.missing_keys)} missing keys, "
                f"{len(load_result.unexpected_keys)} unexpected keys"
            )
            if load_result.missing_keys:
                for key in sorted(load_result.missing_keys)[:20]:
                    log_print(f"  missing: {key}")
                if len(load_result.missing_keys) > 20:
                    log_print(f"  ... and {len(load_result.missing_keys) - 20} more missing keys")
            if load_result.unexpected_keys:
                for key in sorted(load_result.unexpected_keys)[:20]:
                    log_print(f"  unexpected: {key}")
                if len(load_result.unexpected_keys) > 20:
                    log_print(
                        f"  ... and {len(load_result.unexpected_keys) - 20} more unexpected keys"
                    )

        trainability_info = configure_trainable_components(
            model,
            config,
            log_print,
            force_trainable_components=(
                ["decoder"]
                if bool(checkpoint_path)
                and bool(config.get("force_decoder_trainable_on_finetune", True))
                else []
            ),
        )
        is_transfer_learning = (
            trainability_info["has_partial_training"] or reset_epochs_on_checkpoint
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.get("learning_rate", 2.5e-4)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        scheduler = None
        if config.get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )

        trainable_params = trainability_info["trainable_params"]
        total_params = trainability_info["total_params"]
        frozen_params = trainability_info["frozen_params"]
        trainable_pct = 100.0 * trainable_params / max(total_params, 1)
        log_print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_pct:.2f}%)"
        )
        log_print(f"Frozen parameters: {frozen_params:,} / {total_params:,}")

        if checkpoint_path:
            if resume_from_local_progress:
                start_epoch = int(checkpoint.get("epoch", 0))
                best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
                log_print(
                    f"Continuing training at epoch {start_epoch} "
                    f"(best val_loss so far {best_val_loss:.6f})"
                )
                if not trainability_info["has_partial_training"]:
                    if "optimizer_state_dict" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if scheduler is not None and "scheduler_state_dict" in checkpoint:
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                elif trainability_info["has_partial_training"]:
                    log_print(
                        "Skipping optimizer/scheduler state load because partial training is enabled"
                    )
            elif is_transfer_learning:
                if reset_epochs_on_checkpoint:
                    log_print("Resetting epoch counter because reset_epochs_on_checkpoint=True")
                if trainability_info["has_partial_training"]:
                    log_print(
                        "Skipping optimizer/scheduler state load because partial training is enabled"
                    )
            else:
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler is not None and "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = int(checkpoint.get("epoch", 0))
                best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
                log_print(f"Resumed from {checkpoint_path} at epoch {start_epoch}")
        for epoch in range(start_epoch, num_epochs):
            log_print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_stats = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                conditioning_mode=conditioning_mode,
                loss_type=loss_type,
                scaler=scaler,
                use_amp=use_amp,
                training=True,
            )
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                conditioning_mode=conditioning_mode,
                loss_type=loss_type,
                scaler=scaler,
                use_amp=use_amp,
                training=False,
            )

            if scheduler is not None:
                scheduler.step(val_stats["loss"])

            log_print(
                f"  train_loss={train_stats['loss']:.6f}  train_rmse_norm={train_stats['rmse_norm']:.6f}"
                f"  train_rmse_physical={train_stats['rmse_physical']:.6f}"
            )
            log_print(
                f"  val_loss={val_stats['loss']:.6f}  val_rmse_norm={val_stats['rmse_norm']:.6f}"
                f"  val_rmse_physical={val_stats['rmse_physical']:.6f}"
            )

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_stats["loss"],
                "config": config,
            }
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save(checkpoint, save_dir / "best_model.pt")
                log_print(f"  Saved best model  val_loss={best_val_loss:.6f}")

            if (epoch + 1) % int(config.get("save_every", 10)) == 0:
                torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        log_print("Training complete.")
    except Exception as e:
        log_print(f"Training failed with error: {e}")
        log_print(traceback.format_exc())
        raise
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
