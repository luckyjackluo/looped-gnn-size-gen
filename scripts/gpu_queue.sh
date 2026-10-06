#!/bin/bash
# Memory-aware job runner: reads job lines (bash commands containing
# CUDA_VISIBLE_DEVICES=<n>) from $1, appends allowed at any time; launches
# each on the first GPU (excluding $EXCLUDE) with >= $MIN_FREE MiB free.
# Usage: bash scripts/gpu_queue.sh results/grid_logs/pending.txt
set -u
Q=$1; MIN_FREE=${MIN_FREE:-24000}; EXCLUDE=${EXCLUDE:-0}; DONE="$Q.launched"
touch "$Q" "$DONE"
while true; do
  line=$(grep -vxFf "$DONE" "$Q" | head -1)
  if [ -z "$line" ]; then sleep 60; continue; fi
  gpu=""
  while [ -z "$gpu" ]; do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | awk -F', ' -v m="$MIN_FREE" -v ex="$EXCLUDE" '$1!=ex && $2>=m {print $1; exit}')
    [ -z "$gpu" ] && sleep 60
  done
  cmd=$(echo "$line" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]/CUDA_VISIBLE_DEVICES=$gpu/")
  echo "[gpu_queue] $(date +%H:%M) launching on GPU $gpu: $(echo "$cmd" | grep -oE 'grid_logs/[^ ]+\.log')"
  setsid nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
  echo "$line" >> "$DONE"
  sleep 120   # let the new job's memory settle before placing the next one
done
