#!/usr/bin/env python
"""fig_e9_bias.pdf — manufactured (A14) bias: drift probe + freeze gain."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply_style
apply_style()
import numpy as np
R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_export")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})
fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 3.4))
# left: E9 delta eps_B vs frozen
d = json.load(open(f"{R}/e9/e9_iterop_diam_a0.3_rgg_d2_vs_rgg_d2_deg4.json"))
sizes = sorted({int(n) for run in d["runs"] for n in run["probes"]})
styles = {"ft_unbiased": ("#1baf7a", "o", "Loop-Tune, unbiased adapt"),
          "ft_biased": ("#eb6834", "s", "Loop-Tune, biased adapt"),
          "fs_unbiased": ("#2a78d6", "D", "LFS, unbiased adapt"),
          "fs_biased": ("#eda100", "^", "LFS, biased adapt")}
for name, (c, mk, lab) in styles.items():
    de = np.array([[r["probes"][str(n)][name]["epsB_mean"] - r["probes"][str(n)]["tier1"]["epsB_mean"]
                    for n in sizes] for r in d["runs"]])
    axl.errorbar(sizes, de.mean(0) * 1e3, yerr=de.std(0) * 1e3, fmt="-" + mk, color=c,
                 ms=4, capsize=3, label=lab, mec="white", mew=0.5)
axl.axhline(0, color="k", lw=0.8)
axl.set_xscale("log"); axl.set_xlabel("probe size $N$")
axl.set_ylabel(r"$\Delta\epsilon_B$ ($\times 10^{-3}$)")
axl.set_title("Operator drift (diam family, $\\rho=0.7$; 3 seeds)", fontsize=15)
axl.legend(fontsize=7, frameon=False)
# right: FS/FT ratio, biased vs unbiased adaptation (5-seed E1 cells)
series = [("log", "", "#2a78d6", "o", "log, unbiased"),
          ("log", "_adaptfam_rgg_d2_deg4", "#2a78d6", "s", "log, biased"),
          ("diam", "", "#eb6834", "o", "diam, unbiased"),
          ("diam", "_adaptfam_rgg_d2_deg4", "#eb6834", "s", "diam, biased")]
for hor, suf, c, mk, lab in series:
    r = json.load(open(f"{R}/e1/e1_iterop_{hor}_a0.3_rgg_d2_anchored_ntr200{suf}.json"))
    ns = sorted(r["runs"][0]["risk"], key=int); x = [int(n) for n in ns]
    lr = np.array([[np.log(run["risk"][n]["fs"] / run["risk"][n]["ft"]) for n in ns]
                   for run in r["runs"]])
    m = np.exp(lr.mean(0))
    axr.plot(x, m, "-" if suf else "--", color=c, lw=1.5, marker=mk, ms=4,
             mec="white", mew=0.5, label=lab)
    axr.fill_between(x, np.exp(lr.mean(0) - lr.std(0)), np.exp(lr.mean(0) + lr.std(0)),
                     color=c, alpha=0.08, lw=0)
axr.axhline(1.0, color="#3a3a37", lw=1.0, ls=":")
axr.set_xscale("log"); axr.set_yscale("log")
axr.set_yticks([0.2, 0.5, 1.0]); axr.set_yticklabels(["0.2", "0.5", "1.0"])
axr.set_yticks([], minor=True)
axr.set_xlabel("deployment size $N$")
axr.set_ylabel("LFS / Loop-Tune risk")
axr.set_title("Freeze gain under adaptation bias (5 seeds, $\\pm$1 sd)", fontsize=15)
axr.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_e9_bias.pdf", dpi=300, bbox_inches="tight")
print("wrote fig_e9_bias.pdf")
