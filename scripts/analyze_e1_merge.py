#!/usr/bin/env python
"""Merge multi-run E1 JSONs (base + seed-offset runs), report mean ± 95% CI,
paired FT-vs-FS statistics, and regenerate the headline figure.

Usage:
    python scripts/analyze_e1_merge.py --prefix e1_pagerank_smooth_a0.05_rgg_d2
    python scripts/analyze_e1_merge.py --prefix e1_pagerank_smooth_a0.05_rgg_d2_adapt9
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply_style
apply_style()
import numpy as np

E1_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e1")
SCHEMES = ["tier0", "tier1", "ft", "fs"]


def load_runs(prefix):
    """Collect runs from '<prefix>.json' and '<prefix>_soff*.json' only
    (exact match — avoid pulling in _adapt9/_anchored variants)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(E1_DIR, prefix + "*.json"))):
        stem = os.path.basename(path)[:-5]
        suffix = stem[len(prefix):]
        if suffix and not re.fullmatch(r"_soff\d+", suffix):
            continue
        with open(path) as f:
            d = json.load(f)
        runs.extend(d["runs"])
        print(f"  + {stem}: seeds {[r['seed'] for r in d['runs']]}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    runs = load_runs(args.prefix)
    if not runs:
        print("no runs found"); sys.exit(1)
    S = len(runs)
    sizes = sorted(runs[0]["risk"].keys(), key=int)
    print(f"\n{args.prefix}: {S} seeds, sizes {sizes}\n")

    # per-scheme matrix (seeds x sizes)
    mats = {sch: np.array([[r["risk"][n][sch] for n in sizes] for r in runs])
            for sch in SCHEMES}

    # table: mean +/- 95% CI
    t95 = 2.776 if S == 5 else (2.571 if S == 6 else 2.0)  # small-S t approx
    hdr = ["N"] + SCHEMES + ["fs/ft ratio", "fs<ft seeds"]
    print(" | ".join(h.ljust(10) for h in hdr))
    rows = []
    for j, n in enumerate(sizes):
        cells = [n]
        for sch in SCHEMES:
            v = mats[sch][:, j]
            cells.append(f"{v.mean():.5f}±{t95*v.std(ddof=1)/np.sqrt(S):.5f}")
        ratio = mats["fs"][:, j] / mats["ft"][:, j]
        wins = int((mats["fs"][:, j] < mats["ft"][:, j]).sum())
        cells += [f"{np.exp(np.mean(np.log(ratio))):.2f}", f"{wins}/{S}"]
        rows.append(cells)
        print(" | ".join(str(c).ljust(10) for c in cells))

    # paired FT-vs-FS across all cells
    diffs = np.log(mats["fs"]) - np.log(mats["ft"])
    print(f"\npaired log-ratio fs/ft over all cells: "
          f"{diffs.mean():+.3f} ± {diffs.std(ddof=1)/np.sqrt(diffs.size):.3f} "
          f"(negative = FS better); FS wins {int((diffs<0).sum())}/{diffs.size} cells")
    print(f"d_R = {runs[0]['d_R']}, d_phi = {runs[0]['d_phi']} "
          f"({100*runs[0]['d_phi']/runs[0]['d_R']:.1f}%)")

    # headline figure with 95% CI bands
    fig, ax = plt.subplots(figsize=(6, 4.2))
    styles = {"tier0": ("Fixed", "tab:red"),
              "tier1": ("Loop", "tab:orange"),
              "ft": ("Loop-Tune", "tab:blue"),
              "fs": ("LFS", "tab:green")}
    ns = [int(n) for n in sizes]
    for sch, (label, color) in styles.items():
        m = mats[sch].mean(0)
        ci = t95 * mats[sch].std(0, ddof=1) / np.sqrt(S)
        ax.loglog(ns, m, "-o", color=color, label=f"{label}", ms=4)
        ax.fill_between(ns, m - ci, m + ci, color=color, alpha=0.15)
    ax.set_xlabel("deployment size N")
    ax.set_ylabel("risk (per-node MSE)")
    nice = {"e1_pagerank_smooth_a0.05_rgg_d2":
                "Tiered schemes, damped PageRank (rho=0.95) on RGG-d2 — generous adaptation (100 graphs)",
            "e1_pagerank_smooth_a0.05_rgg_d2_adapt9":
                "Tiered schemes, damped PageRank (rho=0.95) on RGG-d2 — scarce adaptation (9 graphs)"}
    fig.tight_layout()
    out = os.path.join(E1_DIR, f"{args.prefix}_merged{S}seeds.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"figure: {out}")

    with open(os.path.join(E1_DIR, f"{args.prefix}_merged.json"), "w") as f:
        json.dump({"prefix": args.prefix, "seeds": S,
                   "sizes": sizes,
                   "mean": {s: mats[s].mean(0).tolist() for s in SCHEMES},
                   "ci95": {s: (t95*mats[s].std(0, ddof=1)/np.sqrt(S)).tolist()
                            for s in SCHEMES}}, f, indent=1)


if __name__ == "__main__":
    main()
