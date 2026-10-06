#!/usr/bin/env python
"""E10 — Tier-0 adaptation variants (the standard transfer-learning recipes).

Two schemes built on the SAME pretrained fixed-depth model as E1's Tier 0
(same seeds, same datasets, so results merge row-wise with the E1 runs):

  ft_fix : fine-tune all parameters on D_adapt, deploy at the fixed depth
           (the most common ML practice; also what the chip-placement FT is)
  fs_fix : freeze the model; train only a zero-init FiLM adapter inserted
           between the fixed layers (Tier-0 mirror of freeze-and-steer)

Usage:
    python scripts/run_e10_tier0_variants.py --task iterop_log_a0.3 \
        --family rgg_d2 --n_train 200 --k_fix 5 --seeds 5 [--quick]
"""
import argparse, json, os, sys, time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.schemes import adapt_fs_fix, adapt_ft_fix, pretrain_tier0  # noqa: E402
from sizegen.training import make_dataset  # noqa: E402
from sizegen.training.train import evaluate_risk  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e10")


def run_seed(args, seed, sizes_ood, device):
    t0 = time.time()
    step = max(args.n_train // 10, 1)
    train_sizes = list(range(args.n_train // 2, args.n_train + 1, step))
    d_train = make_dataset(args.family, args.task, train_sizes,
                           args.graphs_per_size, seed=seed)
    d_val = make_dataset(args.family, args.task, [args.n_train], 8, seed=seed + 1000)
    adapt_sizes = [int(0.7 * args.n_adapt), args.n_adapt, int(1.4 * args.n_adapt)]
    d_adapt = make_dataset(args.family, args.task, adapt_sizes,
                           max(args.adapt_graphs // len(adapt_sizes), 2),
                           seed=seed + 2000)
    d_adapt_val = make_dataset(args.family, args.task, [args.n_adapt], 8,
                               seed=seed + 3000)
    in_dim = int(d_train[0].x.shape[1])

    # identical call order and seeds as run_e1_tiers -> same tier0 weights
    tier0 = pretrain_tier0(d_train, d_val, in_dim, hidden_dim=args.hidden_dim,
                           k_fix=args.k_fix, epochs=args.pretrain_epochs,
                           device=device, seed=seed, traj_sup=args.traj_sup)
    common = dict(epochs=args.adapt_epochs, device=device, seed=seed,
                  traj_sup=args.traj_sup)
    ftf = adapt_ft_fix(tier0["model"], d_adapt, d_adapt_val, **common)
    fsf = adapt_fs_fix(tier0["model"], d_adapt, d_adapt_val, **common)
    print(f"[seed {seed}] adapt done: ft_fix val {ftf['info']['best_val']:.5f} "
          f"fs_fix val {fsf['info']['best_val']:.5f} "
          f"(d_R={fsf['d_R']}, d_phi={fsf['d_phi']})", flush=True)

    out = {"seed": seed, "d_R": fsf["d_R"], "d_phi": fsf["d_phi"], "risk": {}}
    for n in sizes_ood:
        ds = make_dataset(args.family, args.task, [n], args.ood_graphs,
                          seed=seed + 4000)
        ds_test = ds[2:]  # same test split as E1 (first 2 are the K-selection split)
        out["risk"][n] = {
            "tier0": evaluate_risk(tier0["model"], ds_test, K=None, device=device)["mse"],
            "ft_fix": evaluate_risk(ftf["model"], ds_test, K=None, device=device)["mse"],
            "fs_fix": evaluate_risk(fsf["model"], ds_test, K=None, device=device)["mse"],
        }
        r = out["risk"][n]
        print(f"[seed {seed}] N={n}  tier0 {r['tier0']:.5f} | "
              f"ft_fix {r['ft_fix']:.5f} | fs_fix {r['fs_fix']:.5f}", flush=True)
    print(f"[seed {seed}] done ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="iterop_log_a0.3")
    ap.add_argument("--family", default="rgg_d2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--k_fix", type=int, default=5)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_adapt", type=int, default=500)
    ap.add_argument("--traj_sup", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 1
        args.pretrain_epochs, args.adapt_epochs = 30, 20
        args.graphs_per_size, args.adapt_graphs, args.ood_graphs = 15, 40, 5
        sizes_ood = [1000, 2000, 5000]
    else:
        args.pretrain_epochs, args.adapt_epochs = 150, 60
        args.graphs_per_size, args.adapt_graphs, args.ood_graphs = 40, 100, 10
        sizes_ood = [1000, 2000, 5000, 10000, 20000]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs = [run_seed(args, s, sizes_ood, args.device) for s in range(args.seeds)]
    tag = (f"{args.task}_{args.family}_ntr{args.n_train}"
           + ("_traj" if args.traj_sup else "") + ("_quick" if args.quick else ""))
    with open(os.path.join(RESULTS_DIR, f"e10_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "runs": runs}, f, indent=1)
    print(f"Saved results/e10/e10_{tag}.json")


if __name__ == "__main__":
    main()
