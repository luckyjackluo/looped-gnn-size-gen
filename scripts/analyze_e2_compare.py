#!/usr/bin/env python
"""Compare E2 K*(N) between the plain and anchored looped operators.

Question: does the contractive-by-construction (anchored) operator remove the
trained-depth stability horizon that compressed the plain model's K* slopes?

Reads results/e2/e2_rgg_d2.json (plain, iid tag) and
results/e2/e2_rgg_d2_anchored_smooth.json, plus the E3 rho_eff values,
and emits a slope table: measured plain / measured anchored / recipe
prediction 1/(2|log rho_eff|).
"""

import glob
import json
import os
import sys

import numpy as np

E2_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e2")
E3_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "results", "e3",
                          "summary_rgg_d2.json")


def kstar_slopes(payload):
    """Per-seed slope fits, then aggregate (evaluator fix: pooling seeds with
    different intercepts destroys the fit — anchored a=0.05 per-seed slopes
    were 0.65/1.89 but the pooled fit reported a meaningless 1.27)."""
    k_grid = payload["k_grid"]
    sizes = payload["sizes_ood"]
    logn = np.log(sizes)
    out = {}
    for a_str, seeds_curves in payload["results"].items():
        per_seed_slopes = []
        per_seed_kstars = []
        for sc in seeds_curves:
            ks = [k_grid[int(np.argmin(sc[str(n)]))] for n in sizes]
            per_seed_kstars.append(ks)
            per_seed_slopes.append(float(np.polyfit(logn, ks, 1)[0]))
        r_first = np.mean([min(sc[str(sizes[0])]) for sc in seeds_curves])
        r_last = np.mean([min(sc[str(sizes[-1])]) for sc in seeds_curves])
        out[float(a_str)] = {
            "slope": float(np.mean(per_seed_slopes)),
            "slope_std": float(np.std(per_seed_slopes)),
            "per_seed_slopes": per_seed_slopes,
            "per_seed_kstars": per_seed_kstars,
            "k_stars": np.mean(per_seed_kstars, axis=0).tolist(),
            "risk_at_kstar_growth": float(r_last / max(r_first, 1e-12)),
        }
    return out, sizes


def main():
    rho_eff = {}
    if os.path.exists(E3_SUMMARY):
        s = json.load(open(E3_SUMMARY))
        for k, v in s.items():
            if isinstance(v, dict) and "rho_eff" in v and k.startswith("pagerank_a"):
                rho_eff[float(k.split("_a")[1])] = v["rho_eff"]

    files = {os.path.basename(p): p for p in glob.glob(os.path.join(E2_DIR, "e2_rgg_d2*.json"))
             if "fits" not in p and "quick" not in p}
    print(f"found: {sorted(files)}\n")
    table = {}
    for name, path in sorted(files.items()):
        payload = json.load(open(path))
        slopes, sizes = kstar_slopes(payload)
        table[name] = slopes

    alphas = sorted({a for v in table.values() for a in v})
    hdr = ["alpha", "rho_eff", "recipe 1/(2|ln rho_eff|)"] + list(table.keys())
    print(" | ".join(h[:36] for h in hdr))
    rows = []
    for a in alphas:
        re_ = rho_eff.get(a)
        recipe = 1.0 / (2.0 * abs(np.log(re_))) if re_ else float("nan")
        row = [f"{a:g}", f"{re_:.3f}" if re_ else "-", f"{recipe:.2f}"]
        for name in table:
            v = table[name].get(a)
            row.append(f"{v['slope']:.2f}±{v['slope_std']:.2f} "
                       f"(x{v['risk_at_kstar_growth']:.1f})" if v else "-")
        rows.append(row)
        print(" | ".join(row))
    print("\n(xR = risk(K*) growth factor from smallest to largest N; "
          "R >> 1 at slow contraction = stability-horizon symptom)")

    with open(os.path.join(E2_DIR, "e2_slope_comparison.json"), "w") as f:
        json.dump({"rho_eff": rho_eff, "table": table}, f, indent=1)

    # ---- improved K* figure: per-seed points + theory overlay -------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from figstyle import apply_style
    apply_style()

    for name, path in sorted(files.items()):
        payload = json.load(open(path))
        slopes, sizes = kstar_slopes(payload)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(slopes)))
        for (a, v), c in zip(sorted(slopes.items()), colors):
            for ks in v["per_seed_kstars"]:
                ax.semilogx(sizes, ks, "o", color=c, alpha=0.45, ms=4)
            mean_ks = np.array(v["k_stars"])
            b = float(np.mean(mean_ks - v["slope"] * np.log(sizes)))
            ax.semilogx(sizes, v["slope"] * np.log(sizes) + b, "-", color=c,
                        label=(rf"$\rho$={1-a:.2f}: {v['slope']:.2f}"
                               rf"$\pm${v['slope_std']:.2f}"))
            re_ = rho_eff.get(a)
            if re_:
                th = 1.0 / (2.0 * abs(np.log(re_)))
                ax.semilogx(sizes, th * (np.log(sizes) - np.log(sizes[0]))
                            + mean_ks[0], "--", color=c, lw=0.9, alpha=0.7)
        ax.set_xlabel("deployment size N")
        ax.set_ylabel(r"optimal depth $K^*(N)$")
        ax.set_title(f"E2 K* per-seed + theory (dashed) — {name}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(
            E2_DIR, name.replace(".json", "") + "_perseed.pdf"), dpi=160)
        plt.close(fig)
    print("per-seed figures written")


if __name__ == "__main__":
    main()
