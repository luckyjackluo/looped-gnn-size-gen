#!/usr/bin/env python
"""E3c — model-free certification of the constructed-horizon task family.

For every (horizon family, carrier dimension d, damping alpha), with NO
training involved:

  (a) resolution profile eps^2(k) = ||f* - m_k||^2 vs k for several N:
      geometric decay at rate rho^{2k} (Lemma P1) up to the constructed
      radius T*(N), then exactly zero — the class signature;
  (b) the class signature: per-node residual at the pretraining depth K0,
      r(N) = mean_v |f*(v) - m_{K0}(v)|^2 — exactly zero on F1 (const),
      positive and growing with N on F2/F3 (the receptive-field ceiling of
      prop:ceiling), plus the graph-sum far field C(N) = eps^2(0);
  (c) the depth law: T*(N) (by construction) alongside the measured
      L(N) = least k with eps(k) <= tau, checked against the recipe slope
      1/(2 |log rho|) on the log family.

Usage:
    python scripts/run_e3c_iterop.py [--d 2 3] [--alphas 0.1 0.2 0.3 0.5]
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.graphs import rgg  # noqa: E402
from sizegen.tasks import HORIZONS, iterop  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e3c")


def profiles(horizon, d, alpha, sizes, seeds, k_max, k0=5):
    out = {}
    for n in sizes:
        curves, curves_node, t_stars = [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(1000 * n + s)
            g = rgg(n, d=d, avg_degree=8.0, rng=rng, largest_component=True)
            res = iterop(g, horizon=horizon, alpha=alpha, k_max=k_max, rng=rng)
            curves.append(res.truncation_mse(reduction="sum"))
            curves_node.append(res.truncation_mse(reduction="mean"))
            t_stars.append(res.meta["T_star"])
        out[n] = {"eps2": np.mean(curves, axis=0), "T_star": int(np.median(t_stars)),
                  "resid_k0": float(np.mean([c[k0] for c in curves_node]))}
    return out


def fit_decay(eps2, t_star):
    """Late-range log-slope of eps^2(k) on 2 <= k < T*; None if too short."""
    k = np.arange(2, t_star)
    if k.size < 3:
        return None
    y = np.log(np.maximum(eps2[k], 1e-300))
    return float(np.polyfit(k, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[200, 500, 1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1e-2)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {}
    for d in args.d:
        fig, axes = plt.subplots(len(args.alphas), len(HORIZONS),
                                 figsize=(3.4 * len(HORIZONS), 2.8 * len(args.alphas)),
                                 sharex=True)
        axes = np.atleast_2d(axes)
        for ai, alpha in enumerate(args.alphas):
            rho = 1.0 - alpha
            for hi, horizon in enumerate(HORIZONS):
                t0 = time.time()
                k_max = max(64, 2 * max(
                    int(round(5 * (n / 200) ** (1.0 / d))) for n in args.sizes))
                prof = profiles(horizon, d, alpha, args.sizes, args.seeds, k_max)
                ax = axes[ai, hi]
                colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(args.sizes)))
                rec = {"T_star": {}, "L_tau": {}, "C_N": {}, "decay_slope": {},
                       "resid_k0": {}}
                for n, c in zip(args.sizes, colors):
                    e2, ts = prof[n]["eps2"], prof[n]["T_star"]
                    kk = np.arange(len(e2))
                    ax.semilogy(kk[:ts + 1], np.maximum(e2[:ts + 1], 1e-30), "-",
                                color=c, lw=1, label=f"N={n}")
                    ax.axvline(ts, color=c, ls=":", lw=0.6)
                    rec["T_star"][n] = ts
                    rec["C_N"][n] = float(e2[0])
                    rec["resid_k0"][n] = prof[n]["resid_k0"]
                    hit = np.where(np.sqrt(e2) <= args.tau)[0]
                    rec["L_tau"][n] = int(hit[0]) if hit.size else -1
                    rec["decay_slope"][n] = fit_decay(e2, ts)
                # fits: C(N) ~ N^gamma ; measured slope vs 2 log rho
                ns = np.array(args.sizes, float)
                cn = np.array([rec["C_N"][n] for n in args.sizes])
                rec["C_growth_exp"] = float(np.polyfit(np.log(ns), np.log(cn), 1)[0])
                slopes = [v for v in rec["decay_slope"].values() if v is not None]
                rec["decay_slope_mean"] = float(np.mean(slopes)) if slopes else None
                rec["decay_slope_theory"] = float(2 * np.log(rho))
                ts_arr = np.array([rec["T_star"][n] for n in args.sizes], float)
                rec["T_star_logN_slope"] = float(np.polyfit(np.log(ns), ts_arr, 1)[0])
                rec["recipe_slope_logN"] = float(1.0 / (2.0 * abs(np.log(rho))))
                summary[f"{horizon}_d{d}_a{alpha:g}"] = rec
                ax.set_title(f"{horizon}  d={d}  rho={rho:.2f}\n"
                             f"slope {rec['decay_slope_mean'] or float('nan'):.2f} "
                             f"(th. {rec['decay_slope_theory']:.2f}); "
                             f"C(N)~N^{rec['C_growth_exp']:.2f}", fontsize=8)
                if hi == 0:
                    ax.set_ylabel(r"$\varepsilon^2(k)$ (graph sum)")
                if ai == len(args.alphas) - 1:
                    ax.set_xlabel("radius k")
                if ai == 0 and hi == 0:
                    ax.legend(fontsize=6)
                print(f"[d={d} a={alpha} {horizon}] {time.time()-t0:.1f}s  "
                      f"decay {rec['decay_slope_mean']:.2f} vs {rec['decay_slope_theory']:.3f}; "
                      f"resid@K0 {rec['resid_k0'][args.sizes[0]]:.1e}->"
                      f"{rec['resid_k0'][args.sizes[-1]]:.1e}; T*: "
                      + ", ".join(f"{n}:{rec['T_star'][n]}" for n in args.sizes),
                      flush=True)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"e3c_profiles_rgg_d{d}.png"), dpi=150)

        # depth-law figure: T*(N) per family (one panel per alpha is redundant —
        # T* is alpha-independent by construction), plus measured L_tau.
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
        a_mid = args.alphas[len(args.alphas) // 2]
        for horizon, c in zip(HORIZONS, ["tab:gray", "tab:blue", "tab:orange", "tab:red"]):
            rec = summary[f"{horizon}_d{d}_a{a_mid:g}"]
            ax.semilogx(args.sizes, [rec["T_star"][n] for n in args.sizes], "-o",
                        color=c, ms=3, label=f"{horizon}: T*(N)")
            ax2.semilogx(args.sizes, [rec["resid_k0"][n] * 1e3 for n in args.sizes],
                         "-o", color=c, ms=3, label=horizon)
        ax.set_xlabel("N"); ax.set_ylabel("dependency radius T*(N)")
        ax.set_title(f"Constructed horizons on RGG d={d} (all = K0 at N_ref=200)",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax2.set_xlabel("N"); ax2.set_ylabel(r"per-node residual at $K_0$  ($\times 10^{-3}$)")
        ax2.set_title(f"Receptive-field ceiling at K0=5 (rho={1-a_mid:.2f}): "
                      "0 on F1, grows on F2/F3", fontsize=9)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"e3c_horizons_rgg_d{d}.png"), dpi=150)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=1, default=float)
    print(f"Saved to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
