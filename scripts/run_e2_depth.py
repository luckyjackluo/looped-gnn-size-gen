#!/usr/bin/env python
"""E2 — optimal deployment depth K*(N) (paper Lemma 14 / Prop. 15).

Takes a pretrained Tier-1 looped model, sweeps deployment depth K at every
OOD size, and locates K*(N) = argmin_K risk(N, K).  Three claims:

 (a) K*(N) grows ~ log N (fit K* vs ln N);
 (b) the growth rate scales with 1/|log rho|  — swept via the PageRank
     damping dial (--alphas);
 (c) risk at K* keeps improving with N-matched depth while any FIXED K
     saturates/degrades (Prop. 15: iteration removes the Tier-0 ceiling).

Usage:
    python scripts/run_e2_depth.py [--quick] [--family rgg_d2]
                                   [--alphas 0.5 0.3 0.15]
"""

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.schemes import pretrain_tier1  # noqa: E402
from sizegen.tasks import horizon_T  # noqa: E402
from sizegen.training import make_dataset  # noqa: E402
from sizegen.training.data import _ITEROP_RE  # noqa: E402
from sizegen.training.train import evaluate_risk  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e2")


def task_name(alpha: float, seed_mode: str = "iid", fmt: str = None) -> str:
    if fmt:  # e.g. "iterop_log_a{alpha:g}" (constructed-horizon targets)
        return fmt.format(alpha=alpha)
    if seed_mode == "smooth":
        return f"pagerank_smooth_a{alpha:g}"
    return "pagerank" if abs(alpha - 0.15) < 1e-9 else f"pagerank_a{alpha:g}"


def run_alpha(args, alpha: float, sizes_ood, k_grid, device: str):
    task = task_name(alpha, args.seed_mode, args.task_fmt)
    rho = 1.0 - alpha
    print(f"\n=== alpha={alpha} (rho={rho}) ===", flush=True)
    per_seed = []
    for seed in range(args.seeds):
        step = max(args.n_train // 10, 1)
        train_sizes = list(range(args.n_train // 2, args.n_train + 1, step))
        d_train = make_dataset(args.family, task, train_sizes,
                               args.graphs_per_size, seed=seed)
        d_val = make_dataset(args.family, task, [args.n_train], 8, seed=seed + 1000)
        tier1 = pretrain_tier1(
            d_train, d_val, in_dim=int(d_train[0].x.shape[1]),
            hidden_dim=args.hidden_dim,
            k_min=2, k_max=args.k_train_max, anchored=args.anchored,
            traj_sup=args.traj_sup,
            epochs=args.pretrain_epochs, device=device, seed=seed,
        )
        model = tier1["model"]

        curves = {}  # n -> [risk at each K in k_grid]
        for n in sizes_ood:
            ds = make_dataset(args.family, task, [n], args.ood_graphs,
                              seed=seed + 4000)
            risks = [evaluate_risk(model, ds, K=k, device=device)["mse"]
                     for k in k_grid]
            curves[n] = risks
            k_star = k_grid[int(np.argmin(risks))]
            print(f"[a={alpha} seed={seed}] N={n}: K*={k_star} "
                  f"risk(K*)={min(risks):.5f}", flush=True)
        per_seed.append(curves)
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="rgg_d2")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.3, 0.15])
    ap.add_argument("--seed_mode", default="iid", choices=["iid", "smooth"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--k_train_max", type=int, default=8)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--traj_sup", action="store_true")
    ap.add_argument("--task_fmt", default=None,
                    help='task name template with {alpha}, e.g. "iterop_log_a{alpha:g}"')
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 1
        args.pretrain_epochs, args.graphs_per_size, args.ood_graphs = 30, 15, 5
        sizes_ood = [500, 1000, 2000, 5000]
        k_grid = list(range(1, 25))
    else:
        args.pretrain_epochs, args.graphs_per_size, args.ood_graphs = 150, 40, 10
        sizes_ood = [500, 1000, 2000, 5000, 10000, 20000]
        k_grid = list(range(1, 65))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    for alpha in args.alphas:
        all_results[alpha] = run_alpha(args, alpha, sizes_ood, k_grid, args.device)

    tag = (args.family + ("_anchored" if args.anchored else "")
           + ("_traj" if args.traj_sup else "") + ("_quick" if args.quick else ""))
    tag += "_smooth" if args.seed_mode == "smooth" else ""
    tag += "_al" + "-".join(f"{a:g}" for a in args.alphas)
    if args.task_fmt:
        tag += "_" + args.task_fmt.split("_a{")[0]
    if args.n_train != 100:
        tag += f"_ntr{args.n_train}"
    # constructed-horizon overlay: T*(N) is the dependency radius by construction
    t_star = None
    m = _ITEROP_RE.match(task_name(args.alphas[0], args.seed_mode, args.task_fmt))
    if m is not None:
        d = int(args.family.split("_d", 1)[1]) if "_d" in args.family else 2
        t_star = {n: horizon_T(m["horizon"], n, d,
                               k0=int(m["k0"]) if m["k0"] else 5,
                               n_ref=int(m["nref"]) if m["nref"] else 200)
                  for n in sizes_ood}
    payload = {
        "args": vars(args), "sizes_ood": sizes_ood, "k_grid": k_grid,
        "results": {
            str(a): [{str(n): v for n, v in c.items()} for c in seeds_c]
            for a, seeds_c in all_results.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, f"e2_{tag}.json"), "w") as f:
        json.dump(payload, f, indent=1)

    # ---- fig 1: K* vs N (semilog-x), one line per alpha ------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(args.alphas)))
    fits = {}
    for alpha, c in zip(args.alphas, colors):
        k_stars = []
        for n in sizes_ood:
            per_seed = [k_grid[int(np.argmin(sc[n]))] for sc in all_results[alpha]]
            k_stars.append(np.mean(per_seed))
        a_fit, b_fit = np.polyfit(np.log(sizes_ood), k_stars, 1)
        fits[alpha] = (float(a_fit), float(b_fit))
        rho = 1.0 - alpha
        ax.semilogx(sizes_ood, k_stars, "o", color=c)
        ax.semilogx(sizes_ood, a_fit * np.log(sizes_ood) + b_fit, "-", color=c,
                    label=rf"$\rho$={rho:.2f}: slope {a_fit:.2f}")
    if t_star is not None:
        ax.semilogx(sizes_ood, [t_star[n] for n in sizes_ood], "k--", lw=1,
                    label=r"$T^*(N)$ (constructed radius)")
    ax.set_xlabel("deployment size N")
    ax.set_ylabel(r"optimal depth $K^*(N)$")
    ax.set_title(f"E2: K* ~ log N, rate set by rho — {args.family}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e2_kstar_{tag}.png"), dpi=160)

    # ---- fig 2: risk vs K curves for the largest alpha (dial visual) -----
    alpha0 = args.alphas[-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(sizes_ood)))
    for n, c in zip(sizes_ood, colors):
        risks = np.mean([sc[n] for sc in all_results[alpha0]], axis=0)
        ax.semilogy(k_grid, risks, "-o", ms=2.5, color=c, label=f"N={n}")
        ax.axvline(k_grid[int(np.argmin(risks))], color=c, ls=":", lw=0.8)
    ax.set_xlabel("deployment depth K")
    ax.set_ylabel("risk (per-node MSE)")
    ax.set_title(
        f"E2: risk vs depth, minimum shifts right with N (alpha={alpha0})",
        fontsize=10,
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e2_riskcurves_{tag}.png"), dpi=160)

    with open(os.path.join(RESULTS_DIR, f"e2_fits_{tag}.json"), "w") as f:
        json.dump({"fits": {str(a): {"slope": v[0], "intercept": v[1],
                                     "theory_note": "slope ~ 1/(2|log rho_eff|) envelope"}
                            for a, v in fits.items()},
                   "T_star": {str(n): t for n, t in t_star.items()} if t_star else None},
                  f, indent=1)
    print("\nFitted K* slopes:", {a: round(v[0], 2) for a, v in fits.items()})
    print(f"Saved to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
