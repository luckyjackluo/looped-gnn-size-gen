#!/usr/bin/env python
"""E3b — cross-size transfer rate W1 ~ N^{-1/d} (paper Lemma 6 / P2).

Model-free probe of the size-transfer primitive: the distribution over nodes
of a radius-k statistic (the k-step task iterate) converges, as N grows, to
its local-weak limit at rate ~ N^{-1/d} for locality-preserving carriers in
R^d — the dominant finite-size effect being the boundary fraction
(perimeter/volume ~ N^{-1/d}).

Method: pool node values across S seeds at each N (pushes the empirical-W1
sampling floor ~ (S N)^{-1/2} below the signal), compute 1-D W1 against the
pooled reference at N_ref, fit the log-log slope, compare to -1/d.

Usage:
    python scripts/run_e3b_transfer.py [--quick]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figstyle import apply_style
apply_style()
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.graphs import rgg  # noqa: E402
from sizegen.tasks import pagerank  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e3")


def pooled_values(d, n, seeds, k_stat, alpha):
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(77_000 + 1000 * s + n + 13 * d)
        g = rgg(n, d=d, avg_degree=8.0, rng=rng, largest_component=True)
        res = pagerank(g, alpha=alpha, k_max=k_stat, rng=rng)
        vals.append(res.iterates[k_stat])
    return np.concatenate(vals)


def w1_quantile(a, b, q=2000):
    grid = np.linspace(0.0, 1.0, q)
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--k_stat", type=int, default=2)
    args = ap.parse_args()

    if args.quick:
        dims, sizes, n_ref, seeds = [2], [500, 1000, 2000], 5000, 3
    else:
        # k-hop boundary layer must be thin relative to the box: at k_stat=4,
        # N=500 (side ~22) the layer covers >80% of nodes — pre-asymptotic.
        # Use larger sizes + k_stat=2 (default via CLI) for the clean regime.
        dims, sizes, n_ref, seeds = [2, 3], [1000, 2000, 4000, 8000, 16000], 64000, 16

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"alpha": args.alpha, "k_stat": args.k_stat, "n_ref": n_ref,
           "seeds": seeds, "dims": {}}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = {2: "tab:blue", 3: "tab:green"}
    for d in dims:
        ref = pooled_values(d, n_ref, seeds, args.k_stat, args.alpha)
        w1s, w1_ci = [], []
        for n in sizes:
            # pooled point estimate + spread over random quarter-splits
            full = pooled_values(d, n, seeds, args.k_stat, args.alpha)
            halves = np.array_split(np.random.default_rng(0).permutation(full), 4)
            sub_w1 = [w1_quantile(h, ref) for h in halves]
            w1s.append(w1_quantile(full, ref))
            w1_ci.append(1.96 * np.std(sub_w1) / 2.0)
        slope = np.polyfit(np.log(sizes), np.log(w1s), 1)[0]
        out["dims"][d] = {"sizes": sizes, "w1": w1s, "slope": float(slope),
                          "theory_slope": -1.0 / d}
        print(f"d={d}: W1={['%.4f' % w for w in w1s]} slope={slope:.3f} "
              f"(theory {-1.0/d:.3f})", flush=True)
        ax.errorbar(sizes, w1s, yerr=w1_ci, fmt="-o", color=colors.get(d, None),
                    ms=4, capsize=3,
                    label=rf"d={d}: slope {slope:.2f} (theory {-1.0/d:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        anchor = w1s[0] * (np.array(sizes) / sizes[0]) ** (-1.0 / d)
        ax.loglog(sizes, anchor, "--", color=colors.get(d, None), lw=0.8)
    ax.set_xlabel("graph size N")
    ax.set_ylabel(rf"$W_1$ of radius-{args.k_stat} statistic vs pooled "
                  rf"$N_{{ref}}$={n_ref} reference")
    ax.set_title("E3b: cross-size transfer rate ~ $N^{-1/d}$ (Lemma 6)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    tag = "_quick" if args.quick else ""
    fig.savefig(os.path.join(RESULTS_DIR, f"transfer_rate_w1{tag}.pdf"), dpi=160)
    with open(os.path.join(RESULTS_DIR, f"transfer_rate_w1{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
