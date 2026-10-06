#!/usr/bin/env python
"""Summarize the constructed-horizon study (E1 tiers + E2 depth) across all
cells and both pretraining objectives; writes results/iterop_summary.{md,json}."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sizegen.tasks import horizon_T  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
out_md, out_js = [], {"e1": {}, "e2": {}}

# ---------------- E1 ----------------
out_md.append("## E1 — tiered risk (per-node MSE, mean over seeds), FS/FT ratio, FS-win cells\n")
out_md.append("| cell | obj | N | tier0 | tier1 | FT | FS | FS/FT | FS<FT seeds | tier1<tier0 seeds |")
out_md.append("|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{R}/e1/e1_iterop_*_anchored*_ntr200.json")):
    if "quick" in f: continue
    m = re.search(r"e1_(iterop_[a-z]+_a[0-9.]+)_(rgg_d\d)_anchored(_traj)?_ntr200", f)
    cell, fam, traj = m.group(1), m.group(2), bool(m.group(3))
    r = json.load(open(f)); runs = r["runs"]
    ns = sorted(runs[0]["risk"], key=int)
    rec = {}
    for n in ns:
        v = {s: np.array([run["risk"][n][s] for run in runs]) for s in ["tier0", "tier1", "ft", "fs"]}
        rec[n] = {s: float(v[s].mean()) for s in v}
        rec[n]["fs_ft_ratio"] = float(np.exp(np.mean(np.log(v["fs"] / v["ft"]))))
        rec[n]["fs_wins"] = int((v["fs"] < v["ft"]).sum()); rec[n]["t1_wins"] = int((v["tier1"] < v["tier0"]).sum())
        rec[n]["K_sel_tier1"] = [run["risk"][n]["K_sel"]["tier1"] for run in runs]
        rec[n]["K_rule"] = runs[0]["risk"][n]["K_rule"]
        if n in (ns[0], ns[-1]):
            x = rec[n]
            out_md.append(f"| {cell} {fam} | {'op-aligned' if traj else 'target'} | {n} | {x['tier0']:.5f} | {x['tier1']:.5f} | {x['ft']:.5f} | {x['fs']:.5f} | {x['fs_ft_ratio']:.2f} | {x['fs_wins']}/{len(runs)} | {x['t1_wins']}/{len(runs)} |")
    out_js["e1"][f"{cell}_{fam}_{'traj' if traj else 'tgt'}"] = rec

# ---------------- E2 ----------------
out_md.append("\n## E2 — optimal deployment depth K*(N) vs constructed radius T*(N)\n")
out_md.append("| cell | obj | alpha | N: K* per seed (T*) | slope dK*/dlnN | risk(K*) N_min→N_max | risk(K=T*)/risk(K*) at N_max |")
out_md.append("|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{R}/e2/e2_rgg_d*_anchored*_iterop_*_ntr200.json")):
    if "quick" in f or "fits" in f: continue
    m = re.search(r"e2_(rgg_d\d)_anchored(_traj)?_al([0-9.-]+)_iterop_([a-z]+)_ntr200", f)
    fam, traj, hor = m.group(1), bool(m.group(2)), m.group(4); d = int(fam[-1])
    r = json.load(open(f)); kg = r["k_grid"]; sizes = r["sizes_ood"]
    for a, seeds in r["results"].items():
        ks = {n: [kg[int(np.argmin(s[str(n)]))] for s in seeds] for n in sizes}
        kbar = [np.mean(ks[n]) for n in sizes]
        slope = float(np.polyfit(np.log(sizes), kbar, 1)[0])
        tstar = {n: horizon_T(hor, n, d) for n in sizes}
        Rm = {n: np.mean([s[str(n)] for s in seeds], axis=0) for n in sizes}
        rk = lambda n: float(Rm[n].min())
        nmax = sizes[-1]; kt = min(tstar[nmax], kg[-1])
        ratio = float(Rm[nmax][kg.index(kt)] / Rm[nmax].min())
        cellk = f"{hor}_{fam}_a{a}_{'traj' if traj else 'tgt'}"
        out_js["e2"][cellk] = {"K_star": ks, "T_star": tstar, "slope": slope,
                               "risk_at_Kstar": {n: rk(n) for n in sizes}, "risk_T_over_Kstar_Nmax": ratio}
        kstr = "; ".join(f"{n}:{ks[n]}({tstar[n]})" for n in [sizes[0], sizes[len(sizes)//2], sizes[-1]])
        out_md.append(f"| {hor} {fam} | {'op-aligned' if traj else 'target'} | {a} | {kstr} | {slope:+.2f} | {rk(sizes[0]):.5f}→{rk(nmax):.5f} | {ratio:.2f} |")

open(f"{R}/iterop_summary.md", "w").write("\n".join(out_md) + "\n")
json.dump(out_js, open(f"{R}/iterop_summary.json", "w"), indent=1, default=float)
print("\n".join(out_md))
