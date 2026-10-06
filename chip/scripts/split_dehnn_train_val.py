"""Design-level train/val split for the DEHNN MLCAD sub-graph dataset.

Holds out whole DESIGNS for validation (not just variants/tiles), so val measures generalization to
*unseen netlists* -- the meaningful test for a placement foundation model. Creates two directories of
symlinks (no data duplication) that the regression pipeline can glob via dataset.train_dir/val_dir.

Files are named ``dehnn_mlcad_d{num}_{variant}_batch{idx}.pickle``; a design's every file goes
entirely to train or val (no leakage of a design across the split).

Usage:
    python scripts/dataset_generation/split_dehnn_train_val.py \
        --src data/chipgen/dehnn_mlcad --out data/chipgen/dehnn_mlcad_split --val_every 10
"""

import argparse
import os
import re
from pathlib import Path

PAT = re.compile(r"dehnn_mlcad_d(\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/chipgen/dehnn_mlcad")
    ap.add_argument("--out", default="data/chipgen/dehnn_mlcad_split")
    ap.add_argument("--val_every", type=int, default=10,
                    help="Hold out every Nth design (by sorted design number) for val.")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    files = sorted(src.glob("dehnn_mlcad_d*.pickle"))
    if not files:
        raise SystemExit(f"No DEHNN pickles found in {src}")

    by_design = {}
    for f in files:
        m = PAT.search(f.name)
        if not m:
            print(f"  [warn] unparseable filename, skipping: {f.name}"); continue
        by_design.setdefault(int(m.group(1)), []).append(f)

    designs = sorted(by_design)
    val_designs = set(designs[:: args.val_every])
    train_designs = [d for d in designs if d not in val_designs]

    out = Path(args.out).resolve()
    train_dir = out / "train"; val_dir = out / "val"
    for d in (train_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.pickle"):
            old.unlink()  # symlinks; safe to clear and rebuild

    n_train = n_val = 0
    for d, flist in by_design.items():
        dst = val_dir if d in val_designs else train_dir
        for f in flist:
            (dst / f.name).symlink_to(f)
            if dst is val_dir:
                n_val += 1
            else:
                n_train += 1

    print(f"designs: {len(designs)} total -> train {len(train_designs)} / val {len(val_designs)}")
    print(f"val design numbers: {sorted(val_designs)}")
    print(f"files: train {n_train} | val {n_val}")
    print(f"train_dir: {train_dir}")
    print(f"val_dir:   {val_dir}")


if __name__ == "__main__":
    main()
