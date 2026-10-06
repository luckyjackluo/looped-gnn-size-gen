#!/usr/bin/env python
"""Round-2 evaluator figure fixes (all replots from cached JSONs).

1. E4/E6: titles state FINDINGS not hypotheses (old titles contradicted data).
2. E5: paired per-cell delta-epsB panel (the +10% uniform shift, with sign test).
3. E3a depth-law: legend/theory lines use the rho_eff RECIPE (the quantity the
   text quotes), not the nominal-rho envelope (which is 2x off by design).
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply_style
apply_style()
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------- E4/E6
def fix_e4e6():
    path = os.path.join(R, "e4_e6", "e4e6_pagerank_smooth_a0.05_rgg_d2.json")
    d = json.load(open(path))
    runs = d["runs"]
    gaps = sorted({int(g) for r in runs for g in r["e4"]})
    budgets = sorted({int(b) for r in runs for b in r["e6"]})
    arms = {"ft_single": ("Loop-Tune (single-size)", "tab:red"),
            "ft_spread": ("Loop-Tune (size-spread)", "tab:blue"),
            "fs": ("LFS (frozen + controller)", "tab:green")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    for name, (label, color) in arms.items():
        vals = np.array([[r["e4"][str(g)][name]["risk"] for g in gaps] for r in runs])
        ax.loglog(gaps, vals.mean(0), "-o", color=color, label=label, ms=4)
        ax.fill_between(gaps, vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), color=color, alpha=0.15)
    ax.axvline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"extrapolation gap $g = N_{OOD}/N_{adapt}$")
    ax.set_ylabel("risk (per-node MSE)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, (label, color) in arms.items():
        vals = np.array([[r["e6"][str(b)][name]["risk"] for b in budgets]
                         for r in runs])
        ax.loglog(budgets, vals.mean(0), "-o", color=color, label=label, ms=4)
        ax.fill_between(budgets, vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), color=color, alpha=0.15)
    ax.set_xlabel("adaptation budget (graphs)")
    ax.set_ylabel(r"risk at $N{=}5000$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(R, "e4_e6", "e4e6_pagerank_smooth_a0.05_rgg_d2.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("e4e6 replotted (findings-titled)")


# ---------------------------------------------------------------- E5 paired
def fix_e5_paired():
    path = sorted(glob.glob(os.path.join(
        R, "e5", "e5_pagerank_smooth_a0.05_rgg_d2_adapt10*.json")))[-1]
    d = json.load(open(path))
    runs = d["runs"]
    sizes = sorted({int(n) for r in runs for n in r["probes"]})
    pairs = {"ft_single": ("Loop-Tune single − frozen", "tab:red"),
             "ft_spread": ("Loop-Tune spread − frozen", "tab:blue"),
             "fs": ("LFS − frozen", "tab:green")}
    fig, ax = plt.subplots(figsize=(6, 4))
    all_deltas = {}
    for name, (label, color) in pairs.items():
        deltas = []  # one per (seed, N) cell
        by_n = []
        for n in sizes:
            cell = [np.mean(r["probes"][str(n)][name]["alignment_mse"])
                    - np.mean(r["probes"][str(n)]["tier1"]["alignment_mse"])
                    for r in runs
                    if "alignment_mse" in r["probes"][str(n)][name]]
            by_n.append(cell)
            deltas.extend(cell)
        m = [np.mean(c) for c in by_n]
        ci = [1.96 * np.std(c) / max(np.sqrt(len(c)), 1) for c in by_n]
        ax.errorbar(sizes, m, yerr=ci, fmt="-o", color=color, ms=4,
                    capsize=3, label=label)
        all_deltas[name] = deltas
        pos = sum(1 for x in deltas if x > 0)
        print(f"  {name}: mean delta {np.mean(deltas):+.5f}, "
              f"{pos}/{len(deltas)} cells > 0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("graph size N")
    ax.set_ylabel(r"paired $\Delta\epsilon_B$ vs frozen operator")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "e5", "e5_epsB_paired.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("e5 paired-delta figure written")


# ---------------------------------------------------------------- depth law
def fix_depth_law():
    for fam_tag in ["rgg_d2", "sparse_er"]:
        path = os.path.join(R, "e3", f"summary_{fam_tag}.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        keys = [k for k in s if k.startswith("pagerank_a") and "depth_fits" in s[k]]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(keys)))
        tau = "0.01"
        for k, c in zip(sorted(keys, key=lambda x: float(x.split("_a")[1])), colors):
            fits = s[k]["depth_fits"].get(tau)
            if not fits:
                continue
            pts = fits["points"]
            ns = np.array([p[0] for p in pts]); ls = np.array([p[1] for p in pts])
            rho_eff = s[k]["rho_eff"]
            recipe = 1.0 / (2.0 * abs(np.log(rho_eff)))
            ax.semilogx(ns, ls, "o", color=c, ms=4)
            a, b = np.polyfit(np.log(ns), ls, 1)
            ax.semilogx(ns, a * np.log(ns) + b, "-", color=c,
                        label=(rf"$\rho_{{eff}}$={rho_eff:.2f}: "
                               rf"fit {a:.2f}, recipe {recipe:.2f}"))
            ax.semilogx(ns, recipe * (np.log(ns) - np.log(ns[0])) + ls[0],
                        "--", color=c, lw=0.9, alpha=0.8)
        ax.set_xlabel("graph size N")
        ax.set_ylabel(rf"task-side depth L(N) at $\tau$={tau} (sum norm)")
        ax.set_title(f"Depth law on {fam_tag}: fit vs constant-free recipe "
                     r"$1/(2|\ln\rho_{eff}|)$ (dashed)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(R, "e3", f"depth_law_recipe_{fam_tag}.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"depth-law recipe figure written for {fam_tag}")


if __name__ == "__main__":
    fix_e4e6()
    fix_e5_paired()
    fix_depth_law()
