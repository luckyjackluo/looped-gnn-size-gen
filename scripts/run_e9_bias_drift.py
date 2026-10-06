#!/usr/bin/env python
"""E9 — manufactured (A14) bias: does a structurally biased adaptation set
cause cross-size operator drift under FT, and does freezing prevent it?

Protocol: pretrain Tier-1 (target-supervised) on rgg_d2 (deg 8); adapt FT
and FS on (i) an UNBIASED set (rgg_d2, deg 8) and (ii) a BIASED set
(rgg_d2_deg4: same geometry, half the average degree) at N ~ 500; probe
eps_B (operator alignment, iterop form) and risk across deployment sizes
on the UNBIASED carrier.

Prediction (A14 as a condition on the adaptation distribution): the biased
set pulls the fine-tuned operator off its cross-size calibration
(size-inflating eps_B) while FS's frozen operator is immune -> the freeze
gain appears/increases exactly under bias.

Usage: python scripts/run_e9_bias_drift.py [--task iterop_diam_a0.3] [--quick]
"""
import argparse, json, os, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.eval.drift import operator_alignment_iterop  # noqa: E402
from sizegen.schemes import adapt_fs, adapt_ft, pretrain_tier1  # noqa: E402
from sizegen.tasks import horizon_T  # noqa: E402
from sizegen.training import make_dataset  # noqa: E402
from sizegen.training.data import _ITEROP_RE  # noqa: E402
from sizegen.training.train import evaluate_risk  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e9")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="iterop_diam_a0.3")
    ap.add_argument("--family", default="rgg_d2")
    ap.add_argument("--bias_family", default="rgg_d2_deg4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_adapt", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    m = _ITEROP_RE.match(args.task); alpha = float(m["alpha"]); d = 2
    hor = m["horizon"]

    if args.quick:
        args.seeds = 1
        pre_ep, ad_ep, gps, ag, og = 30, 20, 15, 40, 5
        probe_sizes = [500, 2000, 5000]
    else:
        pre_ep, ad_ep, gps, ag, og = 150, 60, 40, 100, 8
        probe_sizes = [500, 2000, 5000, 20000]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_out = []
    for seed in range(args.seeds):
        t0 = time.time()
        step = max(args.n_train // 10, 1)
        train_sizes = list(range(args.n_train // 2, args.n_train + 1, step))
        d_train = make_dataset(args.family, args.task, train_sizes, gps, seed=seed)
        d_val = make_dataset(args.family, args.task, [args.n_train], 8, seed=seed + 1000)
        adapt_sizes = [int(0.7 * args.n_adapt), args.n_adapt, int(1.4 * args.n_adapt)]
        k_adapt = horizon_T(hor, args.n_adapt, d)
        k_range = (horizon_T(hor, adapt_sizes[0], d), horizon_T(hor, adapt_sizes[-1], d))

        tier1 = pretrain_tier1(d_train, d_val, in_dim=int(d_train[0].x.shape[1]),
                               hidden_dim=args.hidden_dim, k_min=2, k_max=8,
                               anchored=True, epochs=pre_ep,
                               device=args.device, seed=seed)
        models = {"tier1": tier1["model"]}
        for tag, fam in (("unbiased", args.family), ("biased", args.bias_family)):
            d_ad = make_dataset(fam, args.task, adapt_sizes, max(ag // 3, 2),
                                seed=seed + 2000)
            d_adv = make_dataset(fam, args.task, [args.n_adapt], 8, seed=seed + 3000)
            common = dict(k_adapt=k_adapt, k_range=k_range, epochs=ad_ep,
                          device=args.device, seed=seed)
            models[f"ft_{tag}"] = adapt_ft(tier1["model"], d_ad, d_adv, **common)["model"]
            models[f"fs_{tag}"] = adapt_fs(tier1["model"], d_ad, d_adv, **common)["model"]

        out = {"seed": seed, "k_adapt": k_adapt, "probes": {}}
        for n in probe_sizes:
            ds = make_dataset(args.family, args.task, [n], og, seed=seed + 4000)
            k_probe = max(horizon_T(hor, n, d) + 4, 12)
            out["probes"][n] = {}
            for name, model in models.items():
                eb = operator_alignment_iterop(model, ds, K=k_probe, alpha=alpha,
                                               device=args.device)
                risk = evaluate_risk(model, ds, K=horizon_T(hor, n, d),
                                     device=args.device)["mse"]
                out["probes"][n][name] = {
                    "epsB_mean": float(np.mean(eb["alignment_mse"])),
                    "risk_at_Tstar": risk,
                }
                print(f"[seed {seed}] N={n} {name}: epsB={np.mean(eb['alignment_mse']):.6f} "
                      f"risk@T*={risk:.5f}", flush=True)
        all_out.append(out)
        print(f"[seed {seed}] done in {time.time()-t0:.0f}s", flush=True)

    tag = f"{args.task}_{args.family}_vs_{args.bias_family}" + ("_quick" if args.quick else "")
    with open(os.path.join(RESULTS_DIR, f"e9_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "runs": all_out}, f, indent=1)

    # figure: paired delta eps_B vs frozen, biased vs unbiased; risk ratio
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {"ft_unbiased": ("#1baf7a", "o", "FT, unbiased adapt"),
              "ft_biased": ("#eb6834", "s", "FT, biased adapt (deg 4)"),
              "fs_unbiased": ("#2a78d6", "D", "FS, unbiased adapt"),
              "fs_biased": ("#eda100", "^", "FS, biased adapt (deg 4)")}
    for name, (c, mk, lab) in styles.items():
        de = np.array([[r["probes"][n][name]["epsB_mean"]
                        - r["probes"][n]["tier1"]["epsB_mean"]
                        for n in probe_sizes] for r in all_out])
        axl.errorbar(probe_sizes, de.mean(0), yerr=de.std(0), fmt="-" + mk,
                     color=c, ms=4, capsize=3, label=lab, mec="white", mew=0.5)
    axl.axhline(0, color="k", lw=0.8); axl.set_xscale("log")
    axl.set_xlabel("deployment size $N$")
    axl.set_ylabel(r"$\Delta\epsilon_B$ vs frozen operator")
    axl.legend(fontsize=7, frameon=False)
    for tag2, c, mk in (("unbiased", "#2a78d6", "o"), ("biased", "#eb6834", "s")):
        lr = np.array([[np.log(r["probes"][n][f"fs_{tag2}"]["risk_at_Tstar"]
                               / r["probes"][n][f"ft_{tag2}"]["risk_at_Tstar"])
                        for n in probe_sizes] for r in all_out])
        axr.errorbar(probe_sizes, np.exp(lr.mean(0)),
                     yerr=np.exp(lr.mean(0)) * lr.std(0), fmt="-" + mk, color=c,
                     ms=4, capsize=3, label=f"{tag2} adaptation", mec="white", mew=0.5)
    axr.axhline(1.0, color="k", lw=0.8, ls="--"); axr.set_xscale("log")
    axr.set_xlabel("deployment size $N$"); axr.set_ylabel("FS/FT risk (<1: FS better)")
    axr.legend(fontsize=7, frameon=False)
    fig.suptitle(f"E9 (A14 manufactured): structural adaptation bias -> FT drift -> freeze gain ({args.task})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"e9_{tag}.png"), dpi=160)
    print("saved", RESULTS_DIR)


if __name__ == "__main__":
    main()
