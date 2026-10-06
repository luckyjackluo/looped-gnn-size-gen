#!/usr/bin/env python
"""E4 + E6 — FS-vs-FT crossover and adaptation-budget sweep (Props. 17-18).

Adaptation arms (all from the SAME Tier-1 checkpoint, same epochs/lr):
  ft_single : fine-tune ALL params on graphs of ONE size at ONE depth —
              the literal premise of (A14): single-size fine-tuning leaves
              deployment-size alignment unconstrained.
  ft_spread : fine-tune ALL params on a size spread (0.7/1.0/1.4 x N_adapt)
              with matched depth range — the strongest fair FT.
  fs        : frozen operator + zero-init size-conditioned controller on the
              same spread data.

E6 (budget sweep): |D_adapt| in {5, 10, 30, 100, 300} graphs; deploy at
10 x N_adapt.  Prediction: FS's advantage is largest at small budgets
(estimation term ~ sqrt(d/n), Gamma = sqrt((d_R+d_phi)/d_phi)).

E4 (gap sweep): with the default-budget adapters, deploy at
g x N_adapt for g in {1, 2, 4, 10, 40}.  Prediction: ft_single degrades
with g (drift, A14); fs and ft_spread hold; fs vs ft_spread ordering shows
whether the freeze advantage survives multi-size FT.

Deployment depth per (scheme, N): selected on a 2-graph OOD val split.

Usage:
    python scripts/run_e4_e6_budget.py --task pagerank_smooth_a0.05 \
        --k_train_max 24 [--anchored] [--quick]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.schemes import adapt_fs, adapt_ft, deploy_depth_rule, pretrain_tier1  # noqa: E402
from sizegen.training import make_dataset  # noqa: E402
from sizegen.training.train import evaluate_risk  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e4_e6")


def eval_with_k_selection(model, ds, k_cands, device):
    """Select K on 2 val graphs, report risk on the rest."""
    ds_val, ds_test = ds[:2], ds[2:]
    vals = [evaluate_risk(model, ds_val, K=k, device=device)["mse"]
            for k in k_cands]
    k_sel = k_cands[int(np.argmin(vals))]
    return evaluate_risk(model, ds_test, K=k_sel, device=device)["mse"], k_sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pagerank_smooth_a0.05")
    ap.add_argument("--family", default="rgg_d2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--k_train_max", type=int, default=24)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--n_adapt", type=int, default=500)
    ap.add_argument("--rho_eff", type=float, default=0.918)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 1
        pre_ep, ad_ep, gps, og = 30, 20, 15, 6
        gaps = [1, 4, 10]
        budgets = [5, 30]
        budget_default = 30
    else:
        pre_ep, ad_ep, gps, og = 150, 60, 40, 10
        gaps = [1, 2, 4, 10, 40]
        budgets = [3, 5, 10, 30, 100, 300]
        budget_default = 100

    os.makedirs(RESULTS_DIR, exist_ok=True)
    k_adapt = deploy_depth_rule(args.n_adapt, args.rho_eff)
    adapt_sizes = [int(0.7 * args.n_adapt), args.n_adapt, int(1.4 * args.n_adapt)]
    k_range = (deploy_depth_rule(adapt_sizes[0], args.rho_eff),
               deploy_depth_rule(adapt_sizes[-1], args.rho_eff))
    runs = []
    for seed in range(args.seeds):
        d_train = make_dataset(args.family, args.task, list(range(50, 101, 10)),
                               gps, seed=seed)
        d_val = make_dataset(args.family, args.task, [100], 8, seed=seed + 1000)
        d_adapt_val = make_dataset(args.family, args.task, [args.n_adapt], 8,
                                   seed=seed + 3000)
        tier1 = pretrain_tier1(d_train, d_val, in_dim=2,
                               hidden_dim=args.hidden_dim,
                               k_min=2, k_max=args.k_train_max,
                               anchored=args.anchored,
                               epochs=pre_ep, device=args.device, seed=seed)

        out = {"seed": seed, "k_adapt": k_adapt, "e4": {}, "e6": {}}

        # ---- E6: budget sweep, deploy at 10 x N_adapt ---------------------
        n_ood_e6 = 10 * args.n_adapt
        d_ood_e6 = make_dataset(args.family, args.task, [n_ood_e6], og,
                                seed=seed + 4000)
        k_rule_e6 = deploy_depth_rule(n_ood_e6, args.rho_eff)
        k_cands_e6 = sorted(set([12, 24, k_adapt, k_rule_e6]))
        keep = {}
        for budget in budgets:
            # exact budget parity: distribute `budget` graphs across the
            # spread sizes with remainder (5 -> [2,2,1]), so ft_single and
            # the spread arms see the SAME total graph count
            base, rem = divmod(budget, len(adapt_sizes))
            d_spread = []
            for i, sz in enumerate(adapt_sizes):
                cnt = base + (1 if i < rem else 0)
                if cnt > 0:
                    d_spread += make_dataset(args.family, args.task, [sz],
                                             cnt, seed=seed + 2000)
            d_single = make_dataset(args.family, args.task, [args.n_adapt],
                                    budget, seed=seed + 2000)
            common = dict(epochs=ad_ep, device=args.device, seed=seed,
                          batch_size=16)  # deep-unroll K~37 backprop: keep node count per batch modest
            arms = {
                "ft_single": adapt_ft(tier1["model"], d_single, d_adapt_val,
                                      k_adapt=k_adapt, k_range=None, **common),
                "ft_spread": adapt_ft(tier1["model"], d_spread, d_adapt_val,
                                      k_adapt=k_adapt, k_range=k_range, **common),
                "fs": adapt_fs(tier1["model"], d_spread, d_adapt_val,
                               k_adapt=k_adapt, k_range=k_range, **common),
            }
            out["e6"][budget] = {}
            for name, arm in arms.items():
                risk, k_sel = eval_with_k_selection(
                    arm["model"], d_ood_e6, k_cands_e6, args.device)
                out["e6"][budget][name] = {"risk": risk, "k_sel": k_sel}
            if budget == budget_default:
                keep = arms
            r = out["e6"][budget]
            print(f"[seed {seed}] E6 budget={budget}: "
                  + " ".join(f"{n} {v['risk']:.5f}" for n, v in r.items()),
                  flush=True)

        # ---- E4: gap sweep with default-budget arms -----------------------
        for g in gaps:
            n_ood = g * args.n_adapt
            d_ood = make_dataset(args.family, args.task, [n_ood], og,
                                 seed=seed + 5000)
            k_rule = deploy_depth_rule(n_ood, args.rho_eff)
            k_cands = sorted(set([12, 24, k_adapt, k_rule]))
            out["e4"][g] = {"n_ood": n_ood}
            for name, arm in keep.items():
                risk, k_sel = eval_with_k_selection(
                    arm["model"], d_ood, k_cands, args.device)
                out["e4"][g][name] = {"risk": risk, "k_sel": k_sel}
            r = out["e4"][g]
            print(f"[seed {seed}] E4 gap={g} (N={n_ood}): "
                  + " ".join(f"{n} {v['risk']:.5f}" for n, v in r.items()
                             if isinstance(v, dict)), flush=True)
        runs.append(out)

    tag = (f"{args.task}_{args.family}"
           + ("_anchored" if args.anchored else "")
           + ("_quick" if args.quick else ""))
    with open(os.path.join(RESULTS_DIR, f"e4e6_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "runs": runs}, f, indent=1)

    # ---- figures ----------------------------------------------------------
    arms_style = {"ft_single": ("FT (single-size)", "tab:red"),
                  "ft_spread": ("FT (size-spread)", "tab:blue"),
                  "fs": ("FS", "tab:green")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    for name, (label, color) in arms_style.items():
        vals = np.array([[r["e4"][g][name]["risk"] for g in gaps] for r in runs])
        ax.loglog(gaps, vals.mean(0), "-o", color=color, label=label, ms=4)
        ax.fill_between(gaps, vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), color=color, alpha=0.15)
    ax.set_xlabel(r"extrapolation gap $g = N_{OOD}/N_{adapt}$")
    ax.set_ylabel("risk")
    ax.set_title("E4: crossover — single-size FT drifts (A14)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, (label, color) in arms_style.items():
        vals = np.array([[r["e6"][b][name]["risk"] for b in budgets]
                         for r in runs])
        ax.loglog(budgets, vals.mean(0), "-o", color=color, label=label, ms=4)
        ax.fill_between(budgets, vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), color=color, alpha=0.15)
    ax.set_xlabel(r"adaptation budget (graphs)")
    ax.set_ylabel(f"risk at N={10*args.n_adapt}")
    ax.set_title("E6: estimation term — FS wins when data is scarce", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e4e6_{tag}.png"), dpi=160)
    print(f"Saved to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
