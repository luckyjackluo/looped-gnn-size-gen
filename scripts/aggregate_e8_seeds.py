#!/usr/bin/env python
"""Aggregate the 3-seed chip-placement reruns (UnifiedLearning) into the
paper table: mean +- sd of rmse_physical per bucket per variant, plus the
original (unseeded) run for reference. MLCAD/DEHNN column included when done.
"""
import glob, json, os
import numpy as np

SRC = "chip/data/chipgen/test_outputs_regression"
BUCKETS = ["400_600", "800_1000", "1800_2000", "2400_2800"]
VARIANTS = {  # table row -> variant base name
    "unadapted (Tier 0/1)": "pretrain_baseline_0_500",
    "FT (full fine-tune)": "full_finetune_500_1000",
    "frozen + deeper unroll": "loop_no_ctrl_k2_500_1000",
    "FS, K=2": "loop_peft_500_1000",
    "FS, K=6": "loop_peft_k6_500_1000",
}
DEHNN = {"zero-shot": "zeroshot_pretrain", "FT": "full_finetune_dehnn",
         "FS (K=6)": "loop_triple_peft_k6_dehnn"}

def val(path, key="rmse_physical"):
    return json.load(open(path))[key] if os.path.exists(path) else None

print("== ChipGen OOD buckets (rmse_physical): original | mean+-sd over seeds ==")
for row, base in VARIANTS.items():
    cells = []
    for b in BUCKETS:
        orig = val(f"{SRC}/{base}/test_filtered_{b}/regression_eval_val.json")
        seeds = [v for s in (1, 2, 3)
                 if (v := val(f"{SRC}/{base}_seed{s}/test_filtered_{b}/regression_eval_val.json")) is not None]
        if base == "pretrain_baseline_0_500":  # not rerun: deterministic eval of the fixed pretrain
            cells.append(f"{orig:.2f}" if orig else "-")
        elif seeds:
            cells.append(f"{np.mean(seeds):.2f}+-{np.std(seeds):.2f} (n={len(seeds)}; orig {orig:.2f})")
        else:
            cells.append(f"orig {orig:.2f} (no seeds yet)" if orig else "-")
    print(f"{row:26s} | " + " | ".join(cells))

print("\n== MLCAD / DEHNN (rmse_physical) ==")
for row, base in DEHNN.items():
    orig = val(f"{SRC}/dehnn_finetune/{base}/regression_eval_val.json")
    seeds = [v for s in (1, 2, 3)
             if (v := val(f"{SRC}/dehnn_finetune/{base}_seed{s}/regression_eval_val.json")) is not None]
    if base == "zeroshot_pretrain":
        print(f"{row:12s} | {orig:.2f} (seed-independent)")
    elif seeds:
        print(f"{row:12s} | {np.mean(seeds):.2f}+-{np.std(seeds):.2f} (n={len(seeds)}; orig {orig:.2f})")
    else:
        print(f"{row:12s} | orig {orig:.2f} (no seeds yet)" if orig else f"{row:12s} | -")
