#!/usr/bin/env python
"""E5 figure with the Fixed and Loop schemes included.

Left: paired Delta eps_B vs the frozen operator (as before).
Right: absolute eps_B(N), trajectory mean, for ALL schemes including
Tier 0 (per-layer probe over its K_fix distinct blocks) and Tier 1
(the frozen theta_0 itself).  Writes results/e5/e5_epsB_paired.pdf and
paper_export/fig_e5_epsB_paired.pdf.
"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply_style
apply_style()
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_export")
path = os.path.join(R, "e5", "e5_pagerank_smooth_a0.05_rgg_d2_adapt10.json")
d = json.load(open(path)); runs = d["runs"]
sizes = sorted({int(n) for r in runs for n in r["probes"]})
have_t0 = "tier0" in runs[0]["probes"][str(sizes[0])]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 3.6))
# ---- left: paired deltas vs frozen (unchanged semantics) ----------------
pairs = {"ft_single": ("Loop-Tune single $-$ frozen", "#1baf7a", "o"),
         "ft_spread": ("Loop-Tune spread $-$ frozen", "#2a78d6", "s"),
         "fs": ("LFS $-$ frozen", "#eda100", "D")}
for name, (label, color, mk) in pairs.items():
    by_n = [[np.mean(r["probes"][str(n)][name]["alignment_mse"])
             - np.mean(r["probes"][str(n)]["tier1"]["alignment_mse"])
             for r in runs if "alignment_mse" in r["probes"][str(n)][name]]
            for n in sizes]
    m = [np.mean(c) for c in by_n]
    ci = [1.96 * np.std(c) / max(np.sqrt(len(c)), 1) for c in by_n]
    axl.errorbar(sizes, m, yerr=ci, fmt="-" + mk, color=color, ms=4,
                 capsize=3, label=label, mec="white", mew=0.5)
    deltas = [x for c in by_n for x in c]
    print(f"{name}: mean {np.mean(deltas):+.6f}, {sum(x>0 for x in deltas)}/{len(deltas)} cells > 0")
axl.axhline(0, color="k", lw=0.8)
axl.set_xscale("log"); axl.set_xlabel("graph size $N$")
axl.set_ylabel(r"paired $\Delta\epsilon_B$")
axl.set_title("Adaptation shift, paired per (seed, $N$)", fontsize=15)
axl.legend(fontsize=7, frameon=False)
# ---- right: absolute eps_B for every tier -------------------------------
schemes = ([("tier0", "Fixed (per-layer, $K_{\\mathrm{fix}}$ blocks)", "#9a9a95", "^")] if have_t0 else []) + [
    ("tier1", "Loop (frozen $\\theta_0$)", "#eb6834", "v"),
    ("ft_spread", "Loop-Tune (size-spread)", "#2a78d6", "s"),
    ("ft_single", "Loop-Tune (single-size)", "#1baf7a", "o"),
    ("fs", "LFS (frozen + controller)", "#eda100", "D")]
for name, label, color, mk in schemes:
    vals = [[np.mean(r["probes"][str(n)][name]["alignment_mse"]) for r in runs
             if "alignment_mse" in r["probes"][str(n)].get(name, {})]
            for n in sizes]
    if not any(vals): continue
    m = [np.mean(c) for c in vals]
    ci = [1.96 * np.std(c) / max(np.sqrt(len(c)), 1) for c in vals]
    axr.errorbar(sizes, m, yerr=ci, fmt="-" + mk, color=color, ms=4,
                 capsize=3, label=label, mec="white", mew=0.5)
axr.set_xscale("log"); axr.set_yscale("log")
axr.set_xlabel("graph size $N$")
axr.set_ylabel(r"absolute $\epsilon_B$")
axr.set_title("Absolute operator alignment, all schemes", fontsize=15)
axr.legend(fontsize=7, frameon=False)
fig.tight_layout()
for f in (os.path.join(R, "e5", "e5_epsB_paired.pdf"),
          os.path.join(OUT, "fig_e5_epsB_paired.pdf")):
    fig.savefig(f, dpi=300, bbox_inches="tight")
print("wrote e5 epsB figure (tier0 included:", have_t0, ")")
