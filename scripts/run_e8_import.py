#!/usr/bin/env python
"""E8 — import real-world (chip placement) results from UnifiedLearning.

Aggregates the size-adaptation OOD evaluations already produced in the main
repo (pretrain 0-500 -> adapt 500-1000 -> test up to N=2800) into the paper's
real-world table.  Scheme mapping onto the paper's tiers:

  pretrain_baseline_0_500 : Tier 0/1 pretrained backbone (fixed arch)
  full_finetune_500_1000  : Scheme FT
  loop_no_ctrl_k{2,6}     : Tier 1 (frozen deeper unroll, decoder-only tune)
  loop_peft_k{2,4,6,8}    : Scheme FS (frozen processor + FiLM controller)
  (plus extended PEFT variants for the appendix)

Usage:
    python scripts/run_e8_import.py \
        [--src chip/data/chipgen/test_outputs_regression]
"""

import argparse
import csv
import glob
import json
import os
import re

DEFAULT_SRC = "chip/data/chipgen/test_outputs_regression"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e8")

MAIN_VARIANTS = [
    "pretrain_baseline_0_500",
    "full_finetune_500_1000",
    "loop_no_ctrl_k2_500_1000",
    "loop_no_ctrl_k6_500_1000",
    "loop_peft_500_1000",
    "loop_peft_k4_500_1000",
    "loop_peft_k6_500_1000",
    "loop_peft_k8_500_1000",
]


def bucket_key(name: str):
    m = re.match(r"test_filtered_(\d+)_(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--metric", default="rmse_physical",
                    choices=["rmse_physical", "rmse_norm", "mae_physical", "mae_norm"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = {}
    buckets = set()
    for path in sorted(glob.glob(os.path.join(
            args.src, "*", "test_filtered_*", "regression_eval_val.json"))):
        parts = path.split(os.sep)
        variant, bucket = parts[-3], parts[-2]
        bk = bucket_key(bucket)
        if bk is None:
            continue
        with open(path) as f:
            d = json.load(f)
        rows.setdefault(variant, {})[bk] = d.get(args.metric)
        buckets.add(bk)

    if not rows:
        print(f"No eval JSONs found under {args.src}")
        return

    buckets = sorted(buckets)
    header = ["variant"] + [f"{a}-{b}" for a, b in buckets]

    def emit(variants, fname):
        csv_path = os.path.join(OUT_DIR, fname + ".csv")
        md_path = os.path.join(OUT_DIR, fname + ".md")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for v in variants:
                if v not in rows:
                    continue
                w.writerow([v] + [
                    f"{rows[v].get(b, float('nan')):.3f}" if rows[v].get(b) is not None else ""
                    for b in buckets
                ])
        with open(md_path, "w") as f:
            f.write(f"# ChipGen size-adaptation OOD table ({args.metric})\n\n")
            f.write("Pretrain N=0-500, adapt N=500-1000, test buckets by node count.\n\n")
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "---|" * len(header) + "\n")
            for v in variants:
                if v not in rows:
                    continue
                cells = [f"{rows[v].get(b):.3f}" if rows[v].get(b) is not None else "-"
                         for b in buckets]
                f.write("| " + " | ".join([v] + cells) + " |\n")
        print(f"wrote {md_path}")

    emit(MAIN_VARIANTS, f"e8_main_{args.metric}")
    emit(sorted(rows), f"e8_all_{args.metric}")

    # ---- DEHNN / MLCAD real-netlist table (single val set, no buckets) ----
    dehnn = {}
    for path in sorted(glob.glob(os.path.join(
            args.src, "dehnn_finetune", "*", "regression_eval_val.json"))):
        variant = path.split(os.sep)[-2]
        with open(path) as f:
            dehnn[variant] = json.load(f)
    if dehnn:
        md = os.path.join(OUT_DIR, "e8_dehnn_real_netlists.md")
        with open(md, "w") as f:
            f.write("# Real MLCAD netlists (DEHNN features): adapt from "
                    "synthetic pretrain\n\n")
            f.write("| variant | rmse_norm | rmse_physical | mae_physical |\n")
            f.write("|---|---|---|---|\n")
            order = ["zeroshot_pretrain", "full_finetune_dehnn",
                     "loop_triple_peft_k6_dehnn"]
            for v in order + sorted(set(dehnn) - set(order)):
                if v in dehnn:
                    d = dehnn[v]
                    f.write(f"| {v} | {d['rmse_norm']:.4f} | "
                            f"{d['rmse_physical']:.3f} | "
                            f"{d['mae_physical']:.3f} |\n")
        print(f"wrote {md}")

    # quick console view of the main table
    print("\n" + " | ".join(header))
    for v in MAIN_VARIANTS:
        if v in rows:
            print(" | ".join([v.ljust(28)] + [
                f"{rows[v].get(b):.3f}" if rows[v].get(b) is not None else "  -  "
                for b in buckets]))


if __name__ == "__main__":
    main()
