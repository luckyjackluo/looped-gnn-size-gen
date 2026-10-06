#!/usr/bin/env python
"""E3(a) — model-free assumption certification (paper Lemmas 4 & 14).

Produces, per task family, with NO training involved:

1. resolution_profile_*.png : log10 eps^2(k) (graph-level sum norm) vs depth k,
   one curve per graph size N, with the analytic contraction envelope
   slope 2*log10(rho).  Certifies (A2)/(A3): geometric decay, far field C(N)
   growing with N on F2, and the exact-cutoff contrast on F1.

2. depth_law_*.png : L(N) = least k with eps(k) <= tau, plotted against N on a
   semilog-x axis, one line per contraction dial alpha.  Certifies the
   task-side depth law L(N) ~ log N / (2*|log rho|) that grounds
   K*_eff = Theta(log N) (Lemma 14), including the 1/|log rho| slope.

3. summary.json : fitted slopes/intercepts for the paper's tables.

Usage:
    python scripts/run_e3_assumptions.py [--quick]
"""

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figstyle import apply_style
apply_style()
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sizegen.graphs import rgg, sparse_er  # noqa: E402
from sizegen.tasks import degree_target, labelprop, pagerank, sssp_hops  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e3")


def graph_factory(family, n, d, rng):
    if family == "rgg":
        return rgg(n, d=d, avg_degree=8.0, rng=rng, largest_component=True)
    if family == "sparse_er":
        return sparse_er(n, avg_degree=8.0, rng=rng, largest_component=True)
    raise ValueError(family)


def collect_profiles(family, d, task_name, alpha, sizes, seeds, k_max):
    """-> {n: mean eps^2(k) curve (sum norm)}, {n: [L(n) per tau]}, taus"""
    taus = [1.0, 0.1, 0.01]
    curves = {}
    depths = {n: {tau: [] for tau in taus} for n in sizes}
    for n in sizes:
        per_seed = []
        for s in range(seeds):
            rng = np.random.default_rng(10_000 * s + n)
            g = graph_factory(family, n, d, rng)
            if task_name == "pagerank":
                res = pagerank(g, alpha=alpha, k_max=k_max, rng=rng)
            elif task_name == "labelprop":
                res = labelprop(g, alpha=alpha, k_max=k_max, rng=rng)
            elif task_name == "degree":
                res = degree_target(g, rng=rng)
            elif task_name == "sssp":
                res = sssp_hops(g, k_max=k_max, rng=rng)
            else:
                raise ValueError(task_name)
            per_seed.append(res.truncation_mse(reduction="sum"))
            for tau in taus:
                depths[n][tau].append(res.depth_to_tolerance(tau, reduction="sum"))
        k_len = min(len(c) for c in per_seed)
        curves[n] = np.mean([c[:k_len] for c in per_seed], axis=0)
    return curves, depths, taus


def plot_resolution_profile(curves, rho, title, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(curves)))
    for (n, eps2), c in zip(sorted(curves.items()), colors):
        ks = np.arange(len(eps2))
        pos = eps2 > 0
        ax.plot(ks[pos], np.log10(eps2[pos]), "-o", ms=2.5, color=c, label=f"N={n}")
    if rho is not None:
        # analytic contraction envelope, anchored at the largest-N curve start
        n_ref = max(curves)
        e0 = curves[n_ref][0]
        ks = np.arange(len(curves[n_ref]))
        ax.plot(
            ks,
            np.log10(e0) + 2.0 * np.log10(rho) * ks,
            "k--",
            lw=1,
            label=r"envelope $\rho^{2k}$",
        )
    ax.set_xlabel("depth k")
    ax.set_ylabel(r"$\log_{10}\,\varepsilon^2(k)$  (sum norm)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit_effective_rate(curves):
    """Fit the late-range decay rate rho_eff from the largest-N profile.

    The sup-norm contraction rho is only an envelope; the observed L2 decay
    is steeper at finite k (subdominant spectral modes).  The paper's
    constant-free recipe uses the *measured* rate: fitting eps^2(k) ~
    rho_eff^{2k} over the late range, then predicting the depth-law slope
    1/(2 |log rho_eff|) with no free constants.
    """
    n_ref = max(curves)
    eps2 = curves[n_ref]
    ks = np.arange(len(eps2))
    # Fit only the pre-floor geometric regime: skip the k<5 transient and cut
    # once the relative error reaches 1e-10 of the initial value (the exact
    # fixed point itself is only resolved to ~1e-13 sup-norm, so entries below
    # that are numerical-floor artifacts that flatten the fit).
    valid = (eps2 > eps2[0] * 1e-10) & (eps2 > 0)
    # adaptive transient skip: fast-decaying curves (expanders: rho_eff ~
    # rho * lambda2(P) << rho) may have <10 usable points, so skipping a
    # fixed 5 leaves too few and the fit goes rogue (sparse-ER a=0.5 gave a
    # non-monotone rho_eff before this fix)
    skip = min(5, max(2, int(valid.sum()) // 4))
    valid[:skip] = False
    if valid.sum() < 4:
        valid = eps2 > 0
        valid[:2] = False
    kk, ll = ks[valid], np.log(eps2[valid])
    slope = np.polyfit(kk, ll, 1)[0]
    return float(np.exp(slope / 2.0))


def fit_depth_law(depths, taus):
    """Fit L(N) = a * ln(N) + b per tau; return {tau: (a, b)}."""
    fits = {}
    for tau in taus:
        ns, ls = [], []
        for n in sorted(depths):
            vals = [v for v in depths[n][tau] if v >= 0]
            if vals:
                ns.append(n)
                ls.append(np.mean(vals))
        if len(ns) >= 3:
            a, b = np.polyfit(np.log(ns), ls, 1)
            fits[tau] = (float(a), float(b), list(zip(ns, ls)))
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, 2 seeds")
    ap.add_argument("--family", default="rgg", choices=["rgg", "sparse_er"])
    ap.add_argument("--d", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.quick:
        sizes, seeds, k_max = [500, 1000, 2000, 4000], 2, 48
    else:
        sizes, seeds, k_max = [1000, 2000, 5000, 10000, 20000], 5, 64

    summary = {"family": args.family, "d": args.d, "sizes": sizes, "seeds": seeds}
    tag = f"{args.family}_d{args.d}" if args.family == "rgg" else args.family

    # ---- F2 tasks with contraction dial ---------------------------------
    alpha_grid = {
        "pagerank": [0.5, 0.3, 0.15, 0.05],
        "labelprop": [0.85],
    }
    for task_name, alphas in alpha_grid.items():
        depth_lines = {}
        for alpha in alphas:
            t0 = time.time()
            curves, depths, taus = collect_profiles(
                args.family, args.d, task_name, alpha, sizes, seeds, k_max
            )
            rho = (1.0 - alpha) if task_name == "pagerank" else alpha
            plot_resolution_profile(
                curves,
                rho,
                f"{task_name} (alpha={alpha}, rho={rho}) on {tag} — F2",
                os.path.join(
                    RESULTS_DIR, f"resolution_profile_{task_name}_a{alpha}_{tag}.pdf"
                ),
            )
            fits = fit_depth_law(depths, taus)
            rho_eff = fit_effective_rate(curves)
            pred_slope = 1.0 / (2.0 * abs(np.log(rho_eff)))
            depth_lines[alpha] = (rho, fits)
            summary[f"{task_name}_a{alpha}"] = {
                "rho": rho,
                "rho_eff": rho_eff,
                "predicted_depth_slope_from_rho_eff": pred_slope,
                "depth_fits": {
                    str(tau): {"slope_vs_lnN": a, "intercept": b, "points": pts}
                    for tau, (a, b, pts) in fits.items()
                },
                "eps2_curves": {str(n): c.tolist() for n, c in curves.items()},
            }
            print(
                f"[{task_name} a={alpha}] {time.time()-t0:.1f}s; "
                f"rho={rho:.2f} rho_eff={rho_eff:.3f}; depth slopes: "
                + ", ".join(
                    f"tau={tau}: {a:.2f}" for tau, (a, b, _) in fits.items()
                )
                + f" | recipe predicts {pred_slope:.2f} (envelope {1.0/(2*abs(np.log(rho))):.2f})"
            )

        # depth-law figure: L(N) vs N (semilog-x), one line per alpha
        fig, ax = plt.subplots(figsize=(6, 4.2))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(depth_lines)))
        tau_plot = 0.1
        for (alpha, (rho, fits)), c in zip(sorted(depth_lines.items()), colors):
            if tau_plot not in fits:
                continue
            a, b, pts = fits[tau_plot]
            ns = np.array([p[0] for p in pts])
            ls = np.array([p[1] for p in pts])
            ax.semilogx(ns, ls, "o", color=c)
            ax.semilogx(
                ns,
                a * np.log(ns) + b,
                "-",
                color=c,
                label=(
                    rf"$\rho$={rho:.2f}: slope {a:.2f} "
                    rf"(theory {1.0/(2*abs(np.log(rho))):.2f})"
                ),
            )
        ax.set_xlabel("graph size N")
        ax.set_ylabel(rf"L(N) at $\tau$={tau_plot}")
        ax.set_title(
            f"{task_name} on {tag}: depth law L(N) ~ log N / (2|log rho|)",
            fontsize=10,
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"depth_law_{task_name}_{tag}.pdf"), dpi=160)
        plt.close(fig)

    # ---- F1 control (exact cutoff) + SSSP boundary (no geometric decay) --
    for task_name in ["degree", "sssp"]:
        curves, depths, taus = collect_profiles(
            args.family, args.d, task_name, None, sizes, seeds, k_max
        )
        plot_resolution_profile(
            curves,
            None,
            f"{task_name} on {tag} — "
            + ("F1 control: exact at k=1" if task_name == "degree" else "outside F^op"),
            os.path.join(RESULTS_DIR, f"resolution_profile_{task_name}_{tag}.pdf"),
        )
        summary[task_name] = {
            "eps2_curves": {str(n): c.tolist() for n, c in curves.items()}
        }
        print(f"[{task_name}] done")

    with open(os.path.join(RESULTS_DIR, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nAll outputs in {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
