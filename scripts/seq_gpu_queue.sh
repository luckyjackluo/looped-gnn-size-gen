#!/bin/bash
# Strictly sequential single-GPU queue: runs each job line from $2 on GPU $1,
# starting the next only when the GPU has no compute processes.
GPU=$1; Q=$2; UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
while IFS= read -r line; do
  while [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $UUID)" -gt 0 ]; do sleep 60; done
  cmd=$(echo "$line" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$GPU/")
  echo "[seq gpu$GPU] $(date +%H:%M) start $(echo "$cmd" | grep -oE 'grid_logs/[^ ]+\.log')"
  bash -c "$cmd"
done < "$Q"
echo "[seq gpu$GPU] queue finished $(date +%H:%M)"
