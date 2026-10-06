#!/usr/bin/env python
"""fig_hier_signature.pdf — constructed horizons T*(N) + per-node truncation
error at K0, replotted from results/e3c/summary.json (RGG d=2, alpha=0.3)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
SUM = os.path.join(HERE, "..", "results", "e3c", "summary.json")
OUT = os.path.join(HERE, "..", "paper_export")

from figstyle import apply_style
apply_style()

COL = {"const": "#9a9a95", "log": "#2a78d6", "poly": "#1baf7a", "diam": "#eb6834"}
MK = {"const": "D", "log": "o", "poly": "s", "diam": "^"}

summary = json.load(open(SUM))["summary"]
fams = ["const", "log", "poly", "diam"]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 4.3))
for fam in fams:
    rec = summary[f"{fam}_d2_a0.3"]
    ns = sorted(int(n) for n in rec["T_star"])
    axl.semilogx(ns, [rec["T_star"][str(n)] for n in ns], "-", color=COL[fam],
                 marker=MK[fam], mec="white", mew=0.8, label=fam)
    axr.semilogx(ns, [rec["resid_k0"][str(n)] * 1e3 for n in ns], "-",
                 color=COL[fam], marker=MK[fam], mec="white", mew=0.8, label=fam)
axl.set_xlabel("graph size $N$")
axl.set_ylabel(r"prescribed horizon $T^*(N)$")
axl.legend(frameon=False, loc="upper left")
axr.set_xlabel("graph size $N$")
axr.set_ylabel(r"truncation error at $K_0$  ($\times 10^{-3}$)")
axr.legend(frameon=False, loc="upper left")
for ax in (axl, axr):
    ax.set_xticks([200, 1000, 5000, 20000])
    ax.set_xticklabels(["200", "1k", "5k", "20k"])
    ax.minorticks_off()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hier_signature.pdf"), dpi=450, bbox_inches="tight")
print("wrote fig_hier_signature.pdf")
