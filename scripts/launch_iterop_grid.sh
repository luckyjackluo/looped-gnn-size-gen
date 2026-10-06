#!/bin/bash
# Full constructed-horizon grid (paper §sec:synthetic): E1 tiers + E2 depth
# across horizon families x carrier dimension x damping.  Jobs are queued
# over 7 GPUs (GPU 4 is left to its current tenant), 2 jobs per GPU.
#
#   E1 (5 seeds): all 4 families at alpha=0.3 for d in {2,3}   -> 8 jobs
#                 alpha in {0.1,0.2,0.5} for log & diam at d=2  -> 6 jobs
#   E2 (3 seeds): all 4 families, alphas {0.1,0.3,0.5}, d=2     -> 4 jobs
#                 all 4 families, alpha 0.3, d=3                 -> 4 jobs
#
# Usage: bash scripts/launch_iterop_grid.sh [max_parallel]
set -u
cd "$(dirname "$0")/.."
mkdir -p results/grid_logs
P=${1:-14}
GPUS=(0 1 2 3 5 6 7)

jobs=()
for d in 2 3; do
  for h in const log poly diam; do
    jobs+=("e1|python scripts/run_e1_tiers.py --task iterop_${h}_a0.3 --family rgg_d${d} --n_train 200 --k_fix 5 --anchored --seeds 5|e1_iterop_${h}_a0.3_rgg_d${d}")
  done
done
for a in 0.1 0.2 0.5; do
  for h in log diam; do
    jobs+=("e1|python scripts/run_e1_tiers.py --task iterop_${h}_a${a} --family rgg_d2 --n_train 200 --k_fix 5 --anchored --seeds 5|e1_iterop_${h}_a${a}_rgg_d2")
  done
done
for h in const log poly diam; do
  jobs+=("e2|python scripts/run_e2_depth.py --task_fmt iterop_${h}_a{alpha:g} --alphas 0.1 0.3 0.5 --family rgg_d2 --n_train 200 --anchored --seeds 3|e2_iterop_${h}_rgg_d2")
  jobs+=("e2|python scripts/run_e2_depth.py --task_fmt iterop_${h}_a{alpha:g} --alphas 0.3 --family rgg_d3 --n_train 200 --anchored --seeds 3|e2_iterop_${h}_rgg_d3")
done

printf '%s\n' "${jobs[@]}" | nl -v0 | while IFS=$'\t' read -r idx line; do
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  cmd=$(echo "$line" | cut -d'|' -f2)
  tag=$(echo "$line" | cut -d'|' -f3)
  echo "CUDA_VISIBLE_DEVICES=$gpu $cmd > results/grid_logs/$tag.log 2>&1; echo \"[done] $tag \$(date +%H:%M)\""
done > results/grid_logs/jobs.txt

echo "$(wc -l < results/grid_logs/jobs.txt) jobs, $P parallel; logs in results/grid_logs/"
xargs -P "$P" -I{} bash -c '{}' < results/grid_logs/jobs.txt
echo "ALL DONE $(date)"
