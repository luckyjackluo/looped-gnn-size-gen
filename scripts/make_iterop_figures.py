#!/usr/bin/env python
"""Paper figures for the constructed-horizon study (Section 6, synthetic).

fig_hier_tiers.pdf   : risk vs N, four schemes, one panel per horizon family
                       (target-supervised pretraining, alpha=0.3, d=2; 5 seeds, +-1 sd)
fig_hier_kstar.pdf   : optimal deployment depth K*(N) vs the constructed radius
                       T*(N), per family, both pretraining objectives, alpha=0.1
fig_hier_fsft.pdf    : FS/FT risk ratio vs N across families/alphas (target-sup):
                       the extrapolation-regime crossover of prop:freeze
Reads results/e1 and results/e2 JSONs; writes to paper_export/.
"""
import glob, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sizegen.tasks import HORIZONS, horizon_T  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_export")
# validated categorical palette (dataviz reference instance, light mode), fixed order
COL = {"tier0": "#2a78d6", "tier1": "#eb6834", "ft": "#1baf7a", "fs": "#eda100"}
MK = {"tier0": "s", "tier1": "^", "ft": "o", "fs": "D"}
LBL = {"tier0": "Fixed (depth $K_{\\mathrm{fix}}$)", "tier1": "Loop (zero-shot $K$)",
       "ft": "Loop-Tune (fine-tune)", "fs": "LFS (freeze-and-steer)"}
FAM_LBL = {"const": r"const: $T^*=K_0$  ($F_1$)",
           "log": r"log: $T^*\propto\log N$  ($F_2^{\log}$)",
           "poly": r"poly: $T^*\propto N^{1/2d}$  ($F_2\backslash F_2^{\log}$)",
           "diam": r"diam: $T^*\propto N^{1/d}$  ($F_3$)"}
from figstyle import apply_style
apply_style()


def e1_load(hor, a, fam="rgg_d2", traj=False):
    f = f"{R}/e1/e1_iterop_{hor}_a{a}_{fam}_anchored{'_traj' if traj else ''}_ntr200.json"
    return json.load(open(f)) if os.path.exists(f) else None


def fig_tiers(a="0.3", fam="rgg_d2"):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=False)
    for ax, hor in zip(axes, HORIZONS):
        r = e1_load(hor, a, fam)
        if r is None: continue
        ns = sorted(r["runs"][0]["risk"], key=int); x = [int(n) for n in ns]
        for s in ["tier0", "tier1", "ft", "fs"]:
            v = np.array([[run["risk"][n][s] for n in ns] for run in r["runs"]])
            m, sd = v.mean(0), v.std(0)
            ax.plot(x, m, "-", color=COL[s], lw=2.2, marker=MK[s], ms=7, mec="white", mew=0.8, label=LBL[s])
            ax.fill_between(x, m - sd, m + sd, color=COL[s], alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(FAM_LBL[hor], fontsize=17)
        ax.set_xlabel("deployment size $N$")
        ax.set_xticks([1000, 5000, 20000]); ax.set_xticklabels(["1k", "5k", "20k"])
    axes[0].set_ylabel("risk (per-node MSE)")
    axes[0].legend(fontsize=15, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hier_tiers.pdf", dpi=300, bbox_inches="tight"); plt.close(fig)


def e2_cells(hor, fam, traj):
    pat = f"{R}/e2/e2_{fam}_anchored{'_traj' if traj else ''}_al*_iterop_{hor}_ntr200.json"
    out = {}
    for f in glob.glob(pat):
        r = json.load(open(f))
        for a, seeds in r["results"].items():
            out[a] = (r["sizes_ood"], r["k_grid"], seeds)
    return out


def fig_kstar(a="0.1", fam="rgg_d2"):
    d = int(fam[-1])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    for ax, hor in zip(axes, ["log", "poly", "diam"]):
        for traj, col, mk, lab in [(False, COL["tier0"], "s", "target-supervised pretraining"),
                                   (True, COL["tier1"], "o", "operator-aligned pretraining")]:
            cells = e2_cells(hor, fam, traj)
            if a not in cells: continue
            sizes, kg, seeds = cells[a]
            ks = np.array([[kg[int(np.argmin(s[str(n)]))] for n in sizes] for s in seeds])
            m, sd = ks.mean(0), ks.std(0)
            ax.plot(sizes, m, "-", color=col, lw=2.2, marker=mk, ms=7, mec="white", mew=0.8, label=lab)
            ax.fill_between(sizes, m - sd, m + sd, color=col, alpha=0.12, lw=0)
        sizes = [500, 1000, 2000, 5000, 10000, 20000]
        ax.plot(sizes, [horizon_T(hor, n, d) for n in sizes], "--", color="#3a3a37", lw=1.8,
                label=r"prescribed horizon $T^*(N)$")
        ax.axhline(8, color="#9a9a95", lw=1.4, ls=":")
        ax.set_xscale("log"); ax.set_title(FAM_LBL[hor], fontsize=17)
        ax.set_xlabel("deployment size $N$")
        ax.set_xticks([500, 2000, 20000]); ax.set_xticklabels(["500", "2k", "20k"])
    axes[0].set_ylabel(r"risk-minimizing depth $K^*(N)$")
    axes[0].legend(fontsize=15, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hier_kstar.pdf", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_fsft(fam="rgg_d2"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series = [("log", "0.1"), ("log", "0.3"), ("poly", "0.1"), ("poly", "0.3"),
              ("diam", "0.1"), ("diam", "0.3"), ("diam", "0.5"), ("const", "0.1")]
    cols = ["#2a78d6", "#2a78d6", "#1baf7a", "#1baf7a", "#eb6834", "#eb6834",
            "#eb6834", "#9a9a95"]
    mks = ["o", "s", "o", "s", "o", "s", "^", "D"]
    for (hor, a), c, mk in zip(series, cols, mks):
        r = e1_load(hor, a, fam)
        if r is None: continue
        ns = sorted(r["runs"][0]["risk"], key=int); x = [int(n) for n in ns]
        lr = np.array([[np.log(run["risk"][n]["fs"] / run["risk"][n]["ft"]) for n in ns] for run in r["runs"]])
        m = np.exp(lr.mean(0)); lo = np.exp(lr.mean(0) - lr.std(0)); hi = np.exp(lr.mean(0) + lr.std(0))
        ax.plot(x, m, "-", color=c, lw=2.0, marker=mk, ms=7, mec="white", mew=0.8,
                label=fr"{hor}, $\rho={1-float(a):.1f}$")
        ax.fill_between(x, lo, hi, color=c, alpha=0.08, lw=0)
    ax.axhline(1.0, color="#3a3a37", lw=1.0, ls="--")
    ax.text(1050, 1.02, "LFS = Loop-Tune", fontsize=15, color="#3a3a37")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1000, 5000, 20000]); ax.set_xticklabels(["1k", "5k", "20k"])
    ax.set_yticks([0.7, 1.0, 1.4]); ax.set_yticklabels(["0.7", "1.0", "1.4"]); ax.set_yticks([], minor=True)
    ax.set_xlabel("deployment size $N$  ($N_{\\mathrm{adapt}}\\approx 500$)")
    ax.set_ylabel("LFS / Loop-Tune risk ratio")
    ax.set_title("Freeze gain grows with the extrapolation gap\n(target-supervised pretraining, RGG $d=2$, 5 seeds, $\\pm$1 sd)", fontsize=16)
    ax.set_ylim(0.62, 1.55)
    ax.legend(fontsize=14, frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hier_fsft.pdf", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_phase(fam="rgg_d2"):
    """FS-vs-FT phase diagram: extrapolation gap x contraction rate ->
    FS/FT risk ratio, annotated per cell (log and diam families only --
    the two with complete 4-alpha grids)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "fsft", ["#2a78d6", "#f4f4f2", "#eb6834"])
    fams = ["log", "diam"]
    alphas = ["0.1", "0.2", "0.3", "0.5"]  # rho = 0.9 ... 0.5, top to bottom
    fig, axes = plt.subplots(1, len(fams), figsize=(11.5, 4.2), sharey=True)
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    im = None
    yl = [fr"$\rho={1-float(a):.1f}$" for a in alphas]
    for ax, hor in zip(axes, fams):
        rows = np.full((len(alphas), 5), np.nan)
        ns = None
        for i, a in enumerate(alphas):
            r = e1_load(hor, a, fam)
            if r is None: continue
            ns = sorted(r["runs"][0]["risk"], key=int)
            rows[i] = np.array([[np.log2(run["risk"][n]["fs"] / run["risk"][n]["ft"])
                                 for n in ns] for run in r["runs"]]).mean(0)
        x = np.array([int(n) / 500.0 for n in ns])
        xm = np.sqrt(x[1:] * x[:-1])
        xe = np.concatenate([[x[0] ** 2 / xm[0]], xm, [x[-1] ** 2 / xm[-1]]])
        # row 0 (rho=0.9) at TOP: flip rows for pcolormesh, keep labels in order
        rows_plot = rows[::-1]
        im = ax.pcolormesh(xe, np.arange(len(alphas) + 1), rows_plot,
                           cmap=cmap, norm=norm, shading="flat",
                           edgecolors="white", linewidth=1.5)
        for i in range(len(alphas)):          # annotate with the plain ratio
            for j, xv in enumerate(x):
                v = rows_plot[i, j]
                if np.isfinite(v):
                    ax.text(xv, i + 0.5, f"{2**v:.2f}", ha="center",
                            va="center", fontsize=16, color="#1a1a19")
        ax.set_xscale("log")
        ax.set_yticks(np.arange(len(alphas)) + 0.5)
        ax.set_yticklabels(yl[::-1])
        ax.set_xticks([2, 4, 10, 20, 40]); ax.set_xticklabels(["2", "4", "10", "20", "40"])
        ax.minorticks_off()
        ax.set_xlabel(r"deployment size / adaptation size")
        ax.set_title(FAM_LBL[hor], fontsize=17)
        ax.grid(False)
    axes[0].set_ylabel("contraction rate")
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02,
                      ticks=[np.log2(0.7), 0.0, np.log2(1.4)])
    cb.ax.set_yticklabels(["0.7", "1.0", "1.4"])
    cb.set_label("LFS / Loop-Tune risk ratio", fontsize=16)
    cb.ax.text(0.5, 1.04, "Loop-Tune better", transform=cb.ax.transAxes, ha="center", fontsize=14)
    cb.ax.text(0.5, -0.08, "LFS better", transform=cb.ax.transAxes, ha="center", fontsize=14)
    fig.savefig(f"{OUT}/fig_hier_phase.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_tiers(); fig_kstar(); fig_fsft(); fig_phase()
    print("wrote", [f for f in os.listdir(OUT) if f.startswith("fig_hier")])
