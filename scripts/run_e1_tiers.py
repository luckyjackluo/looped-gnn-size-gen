#!/usr/bin/env python
"""E1 — Tiered dominance (paper Theorem 20): the headline experiment.

Protocol (paper A12: genuine extrapolation, N_train <= N_adapt << N_OOD):
  1. Pretrain Tier-0 (fixed depth) and Tier-1 (looped) on small graphs.
  2. Adapt FT (fine-tune all) and FS (freeze + controller) from the SAME
     Tier-1 checkpoint on the SAME intermediate-size adaptation set with the
     SAME depth rule and budget — only the freeze sub-choice differs.
  3. Deploy every scheme at each OOD size; Tier-1/FT/FS use the closed-form
     depth rule K(N) = ceil(log N / 2|log rho_eff|) with rho_eff measured in
     E3; Tier 0 is architecturally stuck at K_fix.

Expected (F2 targets): E[L0] >= E[L1] >= E[L_FT] >= E[L_FS], gaps growing
with N_OOD.  Control (F1 target, e.g. --task one_hop_mean): all gaps vanish.

Usage:
    python scripts/run_e1_tiers.py --task pagerank --family rgg_d2 [--quick]
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

from sizegen.schemes import (  # noqa: E402
    adapt_fs,
    adapt_ft,
    deploy_depth_rule,
    pretrain_tier0,
    pretrain_tier1,
)
from sizegen.tasks import horizon_T  # noqa: E402
from sizegen.training import make_dataset  # noqa: E402
from sizegen.training.data import _ITEROP_RE  # noqa: E402
from sizegen.training.train import evaluate_risk  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e1")


def _family_dim(family: str) -> int:
    return int(family.split("_d", 1)[1]) if "_d" in family else 2


def make_depth_rule(task: str, family: str, rho_eff: float):
    """Deployment depth K(N).  For constructed-horizon (iterop) targets the
    dependency radius is T*(N) by construction, so the closed-form rule is
    the horizon itself; otherwise the Lemma-14 rule with the measured rate."""
    m = _ITEROP_RE.match(task)
    if m is None:
        return lambda n: deploy_depth_rule(n, rho_eff)
    d = _family_dim(family)
    k0 = int(m["k0"]) if m["k0"] else 5
    n_ref = int(m["nref"]) if m["nref"] else 200
    return lambda n: horizon_T(m["horizon"], n, d, k0=k0, n_ref=n_ref)


def load_rho_eff(family: str, task: str) -> float:
    """Measured effective rate from the E3 summary; conservative fallback.
    Constructed-horizon targets have an exact rate rho = 1 - alpha."""
    m = _ITEROP_RE.match(task)
    if m is not None:
        return 1.0 - float(m["alpha"])
    path = os.path.join(
        os.path.dirname(__file__), "..", "results", "e3", f"summary_{family}.json"
    )
    key = {"pagerank": "pagerank_a0.15", "labelprop": "labelprop_a0.85"}.get(task)
    if key is None and "pagerank" in task and "_a" in task:
        # smooth variants share the plain-task operator (same contraction rate)
        key = f"pagerank_a{float(task.rsplit('_a', 1)[1]):g}"
    if key and os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        if key in s and "rho_eff" in s[key]:
            return float(s[key]["rho_eff"])
    return 0.85


def run_seed(args, seed: int, rho_eff: float, sizes_ood, device: str):
    t0 = time.time()
    depth_rule = make_depth_rule(args.task, args.family, rho_eff)
    hp = dict(hidden_dim=args.hidden_dim, epochs=args.pretrain_epochs,
              device=device, seed=seed)

    # ---- datasets --------------------------------------------------------
    step = max(args.n_train // 10, 1)
    train_sizes = list(range(args.n_train // 2, args.n_train + 1, step))
    d_train = make_dataset(args.family, args.task, train_sizes,
                           args.graphs_per_size, seed=seed)
    d_val = make_dataset(args.family, args.task, [args.n_train], 8, seed=seed + 1000)
    in_dim = int(d_train[0].x.shape[1])
    # Adaptation over a size SPREAD around N_adapt so the size-conditioned
    # controller learns a trend rather than one OOD-fragile point.
    adapt_sizes = [int(0.7 * args.n_adapt), args.n_adapt, int(1.4 * args.n_adapt)]
    adapt_family = args.adapt_family or args.family
    d_adapt = make_dataset(adapt_family, args.task, adapt_sizes,
                           max(args.adapt_graphs // len(adapt_sizes), 2),
                           seed=seed + 2000)
    d_adapt_val = make_dataset(adapt_family, args.task, [args.n_adapt], 8,
                               seed=seed + 3000)
    d_ood = {
        n: make_dataset(args.family, args.task, [n], args.ood_graphs,
                        seed=seed + 4000)
        for n in sizes_ood
    }
    print(f"[seed {seed}] data ready ({time.time()-t0:.0f}s)", flush=True)

    # ---- pretraining -----------------------------------------------------
    tier0 = pretrain_tier0(d_train, d_val, in_dim, k_fix=args.k_fix, **hp)
    tier1 = pretrain_tier1(d_train, d_val, in_dim, k_min=2, k_max=args.k_train_max,
                           anchored=args.anchored, traj_sup=args.traj_sup, **hp)
    print(f"[seed {seed}] pretrain done: tier0 val {tier0['info']['best_val']:.5f} "
          f"tier1 val {tier1['info']['best_val']:.5f}", flush=True)

    # ---- adaptation (identical budgets; only the freeze choice differs) --
    k_adapt = depth_rule(args.n_adapt)
    k_range = (depth_rule(adapt_sizes[0]), depth_rule(adapt_sizes[-1]))
    ad = dict(k_adapt=k_adapt, k_range=k_range, epochs=args.adapt_epochs,
              device=device, seed=seed, traj_sup=args.traj_sup)
    ft = adapt_ft(tier1["model"], d_adapt, d_adapt_val, **ad)
    fs = adapt_fs(tier1["model"], d_adapt, d_adapt_val, **ad)
    print(f"[seed {seed}] adapt done at K={k_adapt}: "
          f"ft val {ft['info']['best_val']:.5f} fs val {fs['info']['best_val']:.5f} "
          f"(d_R={fs['d_R']}, d_phi={fs['d_phi']})", flush=True)

    # ---- deployment ------------------------------------------------------
    # K is a deployment-time knob (paper §4.2): each looped scheme selects
    # its depth on a small OOD val split (2 graphs), tested on the rest.
    # The closed-form rule K(N) supplies the candidate ceiling.
    out = {"seed": seed, "k_adapt": k_adapt,
           "d_R": fs["d_R"], "d_phi": fs["d_phi"], "risk": {}}
    for n, ds in d_ood.items():
        k_rule = depth_rule(n)
        k_cands = sorted(set(
            [4, 8, 12, 16, 24, 32] + [k_adapt, k_rule, int(1.25 * k_rule),
                                      int(1.5 * k_rule)]
        ))
        ds_val, ds_test = ds[:2], ds[2:]
        out["risk"][n] = {"K_rule": k_rule, "K_sel": {}}
        out["risk"][n]["k_cands"] = k_cands
        out["risk"][n]["val_curves"] = {}
        for name, entry in (("tier1", tier1), ("ft", ft), ("fs", fs)):
            vals = [evaluate_risk(entry["model"], ds_val, K=k, device=device)["mse"]
                    for k in k_cands]
            k_sel = k_cands[int(np.argmin(vals))]
            out["risk"][n][name] = evaluate_risk(
                entry["model"], ds_test, K=k_sel, device=device)["mse"]
            out["risk"][n]["K_sel"][name] = k_sel
            out["risk"][n]["val_curves"][name] = vals
        # diagnostic: FS at FT's selected depth (isolates "steering deeper
        # into a misaligned operator" from the freeze choice itself)
        out["risk"][n]["fs_at_ftK"] = evaluate_risk(
            fs["model"], ds_test, K=out["risk"][n]["K_sel"]["ft"],
            device=device)["mse"]
        out["risk"][n]["tier0"] = evaluate_risk(
            tier0["model"], ds_test, K=None, device=device)["mse"]
        r = out["risk"][n]
        print(f"[seed {seed}] N={n} K_rule={k_rule} K_sel={r['K_sel']}  "
              f"tier0 {r['tier0']:.5f} | tier1 {r['tier1']:.5f} | "
              f"ft {r['ft']:.5f} | fs {r['fs']:.5f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pagerank")
    ap.add_argument("--family", default="rgg_d2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--k_fix", type=int, default=4)
    ap.add_argument("--k_train_max", type=int, default=8)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--traj_sup", action="store_true",
                    help="supervise iterate k toward the task truncation m_k (A6)")
    ap.add_argument("--adapt_graphs_override", type=int, default=None,
                    help="scarce-adaptation regime: total adaptation graphs")
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_adapt", type=int, default=500)
    ap.add_argument("--n_train", type=int, default=100,
                    help="pretraining sizes span [n_train/2, n_train]")
    ap.add_argument("--adapt_family", default=None,
                    help="graph family for the ADAPTATION set only (A14 bias probe)")
    ap.add_argument("--tag_suffix", default="",
                    help="extra tag so reruns do not overwrite existing results")
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
    if args.adapt_graphs_override is not None:
        args.adapt_graphs = args.adapt_graphs_override

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rho_eff = load_rho_eff(args.family, args.task)
    depth_rule = make_depth_rule(args.task, args.family, rho_eff)
    print(f"rho_eff = {rho_eff:.3f} -> K(N): "
          + ", ".join(f"{n}:{depth_rule(n)}" for n in sizes_ood))

    runs = [run_seed(args, s, rho_eff, sizes_ood, args.device)
            for s in range(args.seed_offset, args.seed_offset + args.seeds)]

    tag = (f"{args.task}_{args.family}"
           + ("_anchored" if args.anchored else "")
           + ("_traj" if args.traj_sup else "")
           + (f"_adapt{args.adapt_graphs}" if args.adapt_graphs_override else "")
           + (f"_soff{args.seed_offset}" if args.seed_offset else "")
           + (f"_ntr{args.n_train}" if args.n_train != 100 else "")
           + (f"_adaptfam_{args.adapt_family}" if args.adapt_family else "")
           + (f"_{args.tag_suffix}" if args.tag_suffix else "")
           + ("_quick" if args.quick else ""))
    with open(os.path.join(RESULTS_DIR, f"e1_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "rho_eff": rho_eff, "runs": runs}, f, indent=1)

    # ---- figure: risk vs N_OOD, four curves ------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    styles = {"tier0": ("Tier 0 (fixed)", "tab:red"),
              "tier1": ("Tier 1 (loop)", "tab:orange"),
              "ft": ("FT", "tab:blue"),
              "fs": ("FS (ours)", "tab:green")}
    for scheme, (label, color) in styles.items():
        ns = sorted(runs[0]["risk"].keys(), key=int)
        vals = np.array([[r["risk"][n][scheme] for n in ns] for r in runs])
        mean, std = vals.mean(0), vals.std(0)
        ns_i = [int(n) for n in ns]
        ax.loglog(ns_i, mean, "-o", color=color, label=label, ms=4)
        ax.fill_between(ns_i, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("deployment size N")
    ax.set_ylabel("risk (per-node MSE)")
    ax.set_title(f"E1 tiered dominance — {args.task} on {args.family}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e1_{tag}.png"), dpi=160)
    print(f"\nSaved results/e1/e1_{tag}.{{json,png}}")


if __name__ == "__main__":
    main()
