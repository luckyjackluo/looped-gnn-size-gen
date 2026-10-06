# Controlled Reuse for Size-Generalizable Looped GNNs — code

Anonymous code release for the paper *"Controlled Reuse for
Size-Generalizable Looped GNNs."* It contains the complete synthetic
study (top level) and the chip-placement pipeline (`chip/`). All
datasets regenerate deterministically from seeds; the raw result JSONs
behind every table and figure are included under `results/`.

## Layout

- `sizegen/` — library: graph generators, constructed-horizon tasks,
  models (looped GNN, fixed-depth GNN, FiLM controller), the six
  adaptation schemes, training, evaluation.
- `scripts/` — experiment runners and figure scripts (see the table
  below).
- `configs/` — per-experiment settings.
- `tests/` — solver correctness and scheme-equivalence tests
  (`python -m pytest tests/`).
- `results/` — raw result JSONs for every reported number.
- `chip/` — chip-placement pipeline: `unified_learning/` (models,
  training, data preparation), `scripts/` (train / eval / driver),
  `configs/` (one YAML per table row).

## Environment

Python ≥ 3.8 with `torch`, `torch_geometric`, `numpy`, `scipy`,
`networkx`, `matplotlib`, `scienceplots`, `pyyaml` (synthetic study);
`chip/requirements.txt` for the chip pipeline. One GPU suffices for any
single run.

## Schemes

`Fixed`, `Fixed-Tune`, `Fixed-Steer`, `Loop`, `Loop-Tune`, `LFS`
(Loop-Freeze-Steer) as defined in Section 3 of the paper. In the code,
the internal keys are `tier0`, `ft_fix`, `fs_fix`, `tier1`, `ft`, `fs`
respectively. All adapted schemes train the readout; LFS and Fixed-Steer
additionally train the FiLM controller (`sizegen/models/controller.py`)
and freeze everything else.

## Reproducing the paper

Synthetic study (each cell: one GPU, minutes to ~1 h; 5 seeds):

| Paper artifact | Command |
|---|---|
| Fig. 2 (horizons + truncation error) | `python scripts/run_e3c_iterop.py` then `python scripts/make_signature_figure.py` |
| Fig. 3 (risk-minimizing depth) | `python scripts/run_e2_depth.py --task_fmt iterop_{family}_a{alpha:g} --alphas 0.1 --family rgg_d2 --n_train 200 --anchored --seeds 3 [--traj_sup]` per family, then `python scripts/make_iterop_figures.py` |
| Fig. 4 (LFS vs Loop-Tune phase map), Figs. 6–7 | `python scripts/run_e1_tiers.py --task iterop_{family}_a{alpha} --family rgg_d2 --n_train 200 --k_fix 5 --anchored --seeds 5` over families × α ∈ {0.1, 0.2, 0.3, 0.5}, then `python scripts/make_iterop_figures.py` |
| Tables 2–3 (six schemes) | `python scripts/run_e10_tier0_variants.py --task iterop_{family}_a0.3 --family rgg_d2 --n_train 200 --k_fix 5 --seeds 5` (Table 2) and `... --task iterop_{log,diam}_a0.1 --traj_sup` (Table 3), combined with the matching `run_e1_tiers.py` cells |
| Fig. 8 (damped PageRank) | `python scripts/run_e1_tiers.py --task pagerank_smooth_a0.05 --family rgg_d2 --seeds 5` then `python scripts/analyze_e1_merge.py --prefix e1_pagerank_smooth_a0.05_rgg_d2` |
| Fig. 9 (size gap / budget) | `python scripts/run_e4_e6_budget.py --task pagerank_smooth_a0.05 --family rgg_d2 --seeds 3` then `python scripts/polish_figures.py` |
| Fig. 10 (operator drift) | `python scripts/run_e5_drift.py --task pagerank_smooth_a0.05 --family rgg_d2 --seeds 3 --adapt_graphs_override 10 --rho_eff 0.804` then `python scripts/analyze_e5_tier01.py` |
| Fig. 11 (deliberate bias) | `python scripts/run_e1_tiers.py --task iterop_{log,diam}_a0.3 ... --adapt_family rgg_d2_deg4` and `python scripts/run_e9_bias_drift.py`, then `python scripts/make_e9_figure.py` |
| Fig. 12 (SSSP failure case) | `python scripts/run_e1_tiers.py --task sssp --family rgg_d2 --seeds 3` then `python scripts/analyze_e1_merge.py --prefix e1_sssp_rgg_d2` |
| Training-free checks (Fig. 5) | `python scripts/run_e3_assumptions.py` and `python scripts/run_e3b_transfer.py` |

Chip placement (Table 1): from `chip/`,

1. Pretrain: `python scripts/train_chipgen_regression.py --config
   configs/size_adaptation/<variant>.yaml` with the
   `pretrain_baseline_0_500` config, then each adaptation variant config
   (`full_finetune_500_1000`, `loop_tune_k{2,6}_500_1000`,
   `loop_peft[_k6]_500_1000`).
2. Evaluate the size sweep:
   `./scripts/run_test_chipgen_regression_size_adaptation_batch.sh <config>`.
3. Real MLCAD netlists: `./scripts/run_dehnn_finetune.sh cuda:0` with
   `configs/dehnn_finetune/` (zero-shot, full fine-tune, Loop-Tune, LFS
   at K ∈ {2, 6}).
4. Aggregate into the paper table: `python ../scripts/run_e8_import.py`
   and `python ../scripts/aggregate_e8_seeds.py`.

The synthetic netlist generator is included under
`chip/unified_learning/data_generation/`; the real-netlist experiments
use the public MLCAD contest benchmark.
