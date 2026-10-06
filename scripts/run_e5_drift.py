#!/usr/bin/env python
"""E5 — size-transfer preservation vs operator drift (paper A14 / Prop. 18).

Directly probes the draft's red-flagged assumption (A14): does single-size
fine-tuning inflate the operator's misalignment at deployment sizes, while
freezing preserves the cross-size calibration?

Protocol: pretrain Tier-1 -> adapt FT and FS at N_adapt (same budget) ->
at each N in {N_train, N_adapt, N_OOD...} measure
  (a) hidden-trajectory per-step residuals + fitted contraction rate,
  (b) risk-vs-depth curves (drift = risk turning up beyond K_adapt).

Outcomes feed the paper either way:
  - if FT's contraction/risk degrades off-N_adapt while FS's doesn't -> A14
    validated with a mechanism figure;
  - if not, we have the characterization the red note asks for (when A14
    fails, FT is fine — the honest scope boundary).

Usage:
    python scripts/run_e5_drift.py [--quick] [--task pagerank]
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

from sizegen.eval import risk_vs_depth, trajectory_contraction  # noqa: E402
from sizegen.eval.drift import operator_alignment  # noqa: E402
from sizegen.schemes import (adapt_fs, adapt_ft, deploy_depth_rule,  # noqa: E402
                             pretrain_tier0, pretrain_tier1)
from sizegen.training import make_dataset  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pagerank")
    ap.add_argument("--family", default="rgg_d2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--k_train_max", type=int, default=24)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--adapt_graphs_override", type=int, default=None)
    ap.add_argument("--n_adapt", type=int, default=500)
    ap.add_argument("--rho_eff", type=float, default=0.804)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 1
        pre_ep, ad_ep, gps, ag, og = 30, 20, 15, 40, 5
        probe_sizes = [100, 500, 2000, 5000]
        k_probe = 24
    else:
        pre_ep, ad_ep, gps, ag, og = 150, 60, 40, 100, 8
    if args.adapt_graphs_override is not None:
        ag = args.adapt_graphs_override
        probe_sizes = [100, 500, 2000, 5000, 10000, 20000]
        k_probe = 40

    os.makedirs(RESULTS_DIR, exist_ok=True)
    k_grid = list(range(1, k_probe + 1))
    all_out = []
    for seed in range(args.seeds):
        d_train = make_dataset(args.family, args.task, list(range(50, 101, 10)),
                               gps, seed=seed)
        d_val = make_dataset(args.family, args.task, [100], 8, seed=seed + 1000)
        adapt_sizes = [int(0.7 * args.n_adapt), args.n_adapt,
                       int(1.4 * args.n_adapt)]
        k_range = (deploy_depth_rule(adapt_sizes[0], args.rho_eff),
                   deploy_depth_rule(adapt_sizes[-1], args.rho_eff))
        d_single = make_dataset(args.family, args.task, [args.n_adapt], ag,
                                seed=seed + 2000)
        d_spread = make_dataset(args.family, args.task, adapt_sizes,
                                max(ag // 3, 2), seed=seed + 2000)
        d_adapt_val = make_dataset(args.family, args.task, [args.n_adapt], 8,
                                   seed=seed + 3000)

        tier0 = pretrain_tier0(d_train, d_val, in_dim=2,
                               hidden_dim=args.hidden_dim, k_fix=4,
                               epochs=pre_ep, device=args.device, seed=seed)
        tier1 = pretrain_tier1(d_train, d_val, in_dim=2,
                               hidden_dim=args.hidden_dim,
                               k_min=2, k_max=args.k_train_max,
                               anchored=args.anchored,
                               epochs=pre_ep, device=args.device, seed=seed)
        k_adapt = deploy_depth_rule(args.n_adapt, args.rho_eff)
        common = dict(epochs=ad_ep, device=args.device, seed=seed)
        ft_single = adapt_ft(tier1["model"], d_single, d_adapt_val,
                             k_adapt=k_adapt, k_range=None, **common)
        ft_spread = adapt_ft(tier1["model"], d_spread, d_adapt_val,
                             k_adapt=k_adapt, k_range=k_range, **common)
        fs = adapt_fs(tier1["model"], d_spread, d_adapt_val,
                      k_adapt=k_adapt, k_range=k_range, **common)

        models = {"tier0": tier0["model"], "tier1": tier1["model"],
                  "ft_single": ft_single["model"],
                  "ft_spread": ft_spread["model"], "fs": fs["model"]}
        out = {"seed": seed, "k_adapt": k_adapt, "probes": {}}
        for n in probe_sizes:
            ds = make_dataset(args.family, args.task, [n], og, seed=seed + 4000)
            out["probes"][n] = {}
            alpha = (float(args.task.rsplit("_a", 1)[1])
                     if "_a" in args.task else None)
            for name, model in models.items():
                traj = trajectory_contraction(model, ds, K=k_probe,
                                              device=args.device)
                # Tier 0 has no depth knob: one risk value, replicated
                if name == "tier0":
                    from sizegen.training.train import evaluate_risk
                    r0 = evaluate_risk(model, ds, K=None, device=args.device)["mse"]
                    risks = [r0] * len(k_grid)
                else:
                    risks = risk_vs_depth(model, ds, k_grid, device=args.device)
                out["probes"][n][name] = {
                    "contraction_rate": traj["rate"],
                    "residuals": traj["residuals"],
                    "risk_vs_K": risks,
                }
                rate_s = f"{traj['rate']:.3f}" if traj['rate'] is not None else "n/a"
                msg = (f"[seed {seed}] N={n} {name}: rate={rate_s} "
                       f"min_risk={min(risks):.5f} "
                       f"argmin_K={k_grid[int(np.argmin(risks))]}")
                if alpha is not None and "pagerank" in args.task:
                    eb = operator_alignment(model, ds, K=k_probe, alpha=alpha,
                                            device=args.device)
                    out["probes"][n][name]["alignment_mse"] = eb["alignment_mse"]
                    msg += f" epsB_mean={np.mean(eb['alignment_mse']):.6f}"
                print(msg, flush=True)
        all_out.append(out)

    tag = (f"{args.task}_{args.family}"
           + ("_anchored" if args.anchored else "")
           + (f"_adapt{args.adapt_graphs_override}" if args.adapt_graphs_override else "")
           + ("_quick" if args.quick else ""))
    with open(os.path.join(RESULTS_DIR, f"e5_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "runs": all_out}, f, indent=1)

    # ---- fig: risk-vs-K per scheme at each probe size --------------------
    schemes = {"tier0": ("Tier 0 (fixed depth)", "tab:purple"),
               "tier1": ("frozen theta0", "tab:orange"),
               "ft_single": ("FT single-size", "tab:red"),
               "ft_spread": ("FT size-spread", "tab:blue"),
               "fs": ("frozen + controller", "tab:green")}
    ncols = len(probe_sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.4), sharey=False)
    for ax, n in zip(np.atleast_1d(axes), probe_sizes):
        for name, (label, color) in schemes.items():
            risks = np.mean(
                [r["probes"][n][name]["risk_vs_K"] for r in all_out], axis=0
            )
            ax.semilogy(k_grid, risks, color=color, label=label, lw=1.2)
        ax.axvline(all_out[0]["k_adapt"], color="k", ls=":", lw=0.8)
        ax.set_title(f"N={n}", fontsize=9)
        ax.set_xlabel("depth K")
    np.atleast_1d(axes)[0].set_ylabel("risk")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("E5: operator drift — risk vs depth (dotted: K_adapt)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e5_riskdepth_{tag}.png"), dpi=160)

    # ---- fig: contraction rate vs N per scheme ---------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (label, color) in schemes.items():
        rates = [
            np.mean([r["probes"][n][name]["contraction_rate"] for r in all_out])
            for n in probe_sizes
        ]
        ax.semilogx(probe_sizes, rates, "-o", color=color, label=label, ms=4)
    ax.axvline(args.n_adapt, color="k", ls=":", lw=0.8)
    ax.set_xlabel("graph size N")
    ax.set_ylabel("hidden-trajectory contraction rate")
    ax.set_title("E5: learned operator contraction across sizes", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e5_contraction_{tag}.png"), dpi=160)
    print(f"Saved to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
