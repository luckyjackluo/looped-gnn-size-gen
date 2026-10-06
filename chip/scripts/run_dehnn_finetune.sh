#!/usr/bin/env bash
# Iterative-finetune-with-controller on REAL DEHNN MLCAD placement (placement regression).
#
# Stages (auto-resumes; skips finished work):
#   0. Build a design-level train/val split of data/chipgen/dehnn_mlcad (val = unseen netlists).
#   1. ZERO-SHOT eval: the synthetic pretrain checkpoint on DEHNN val (lower bound, no finetune).
#   2. TRAIN the loop-controller finetune (frozen backbone, K=6 triple-FiLM controller + decoder).
#   2. TRAIN the full-finetune baseline (unfreeze everything) for comparison.
#   3. EVAL both finetuned checkpoints on DEHNN val.
#
# Usage:  bash scripts/experiment_running/run_dehnn_finetune.sh [cuda:N]
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON_BIN:-python3}"

DEVICE_ARGS=()
[ $# -gt 0 ] && DEVICE_ARGS=(--device "$1") && echo "Device: $1"

TRAIN="$ROOT/scripts/experiment_running/train_chipgen_regression.py"
TEST="$ROOT/scripts/experiment_running/test_chipgen_regression.py"
# Overridable via env so the same driver runs the anchored variant:
#   DEHNN_SRC=data/chipgen/dehnn_mlcad_anchored SPLIT_DIR=...anchored_split \
#   CFG_DIR=configs/chipgen/regression/dehnn_finetune_anchored \
#   OUT_ROOT=.../dehnn_finetune_anchored  bash run_dehnn_finetune.sh cuda:0
CFG_DIR="${CFG_DIR:-configs/chipgen/regression/dehnn_finetune}"
PRETRAIN_CKPT="data/chipgen/checkpoints/size_adaptation/pretrain_baseline_0_500/best_model.pt"
DEHNN_SRC="${DEHNN_SRC:-data/chipgen/dehnn_mlcad}"
SPLIT_DIR="${SPLIT_DIR:-data/chipgen/dehnn_mlcad_split}"
OUT_ROOT="${OUT_ROOT:-data/chipgen/test_outputs_regression/dehnn_finetune}"

mapfile -t CONFIGS < <(ls "$ROOT/$CFG_DIR"/*.yaml 2>/dev/null)
[ ${#CONFIGS[@]} -gt 0 ] || { echo "No configs in $CFG_DIR" >&2; exit 1; }

# --- Stage 0: split -------------------------------------------------------
if [ ! -d "$SPLIT_DIR/train" ] || [ -z "$(ls -A "$SPLIT_DIR/train" 2>/dev/null)" ]; then
  echo "### Stage 0: building design-level train/val split"
  "$PY" scripts/dataset_generation/split_dehnn_train_val.py --src "$DEHNN_SRC" --out "$SPLIT_DIR"
else
  echo "### Stage 0: split exists ($SPLIT_DIR) -- skipping"
fi

[ -f "$PRETRAIN_CKPT" ] || { echo "Missing pretrain checkpoint: $PRETRAIN_CKPT" >&2; exit 1; }

save_dir_of() { "$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['save_dir'])" "$1"; }

# --- Stage 1: zero-shot lower bound --------------------------------------
ZS_OUT="$OUT_ROOT/zeroshot_pretrain"
if [ ! -f "$ROOT/$ZS_OUT/regression_eval_val.json" ]; then
  echo "### Stage 1: zero-shot eval of synthetic pretrain on DEHNN val"
  TMP_ZS="$(mktemp /tmp/dehnn_zeroshot.XXXXXX.yaml)"
  "$PY" - "$ROOT/$PRETRAIN_CKPT" "$SPLIT_DIR/val" "$TMP_ZS" <<'PY'
import sys, yaml, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ckpt["config"]
d = dict(cfg.get("dataset", {}))
d["val_dir"] = sys.argv[2]; d["val_files"] = None
d["min_graph_size"] = 50; d["max_graph_size"] = 4000; d["max_val_samples"] = 1000
cfg["dataset"] = d; cfg["batch_size"] = 1
cfg["use_size_bucketing"] = False; cfg["max_total_nodes"] = None  # eval one graph at a time (DEHNN is dense; bucketing OOMs)
yaml.safe_dump(cfg, open(sys.argv[3], "w"), sort_keys=False)
PY
  TQDM_DISABLE=1 "$PY" "$TEST" --config "$TMP_ZS" --checkpoint "$ROOT/$PRETRAIN_CKPT" \
    --dataset val --output_dir "$ZS_OUT" "${DEVICE_ARGS[@]}" || true
  rm -f "$TMP_ZS"
else
  echo "### Stage 1: zero-shot result exists -- skipping"
fi

# --- Stage 2: train -------------------------------------------------------
for cfg in "${CONFIGS[@]}"; do
  SD="$(save_dir_of "$cfg")"
  echo ""
  echo "### Stage 2 TRAIN: $cfg"
  if [ -f "$ROOT/$SD/training.log" ] && grep -q "Training complete\." "$ROOT/$SD/training.log" 2>/dev/null; then
    echo "  complete -- skipping"
  else
    "$PY" "$TRAIN" --config "$cfg" "${DEVICE_ARGS[@]}"
  fi
done

# --- Stage 3: eval finetuned variants on DEHNN val ------------------------
for cfg in "${CONFIGS[@]}"; do
  SD="$(save_dir_of "$cfg")"
  NAME="$(basename "$SD")"
  CKPT="$ROOT/$SD/best_model.pt"
  OUT="$OUT_ROOT/$NAME"
  [ -f "$CKPT" ] || { echo "  no checkpoint for $NAME, skipping eval"; continue; }
  if [ -f "$ROOT/$OUT/regression_eval_val.json" ]; then
    echo "### Stage 3 EVAL: $NAME -- exists, skipping"; continue
  fi
  echo "### Stage 3 EVAL: $NAME"
  TMP_E="$(mktemp /tmp/dehnn_eval.XXXXXX.yaml)"
  "$PY" - "$CKPT" "$SPLIT_DIR/val" "$TMP_E" <<'PY'
import sys, yaml, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ckpt["config"]
d = dict(cfg.get("dataset", {}))
d["val_dir"] = sys.argv[2]; d["val_files"] = None
d["min_graph_size"] = 50; d["max_graph_size"] = 4000; d["max_val_samples"] = 1000
cfg["dataset"] = d; cfg["batch_size"] = 1
cfg["use_size_bucketing"] = False; cfg["max_total_nodes"] = None  # eval one graph at a time (DEHNN is dense; bucketing OOMs)
yaml.safe_dump(cfg, open(sys.argv[3], "w"), sort_keys=False)
PY
  TQDM_DISABLE=1 "$PY" "$TEST" --config "$TMP_E" --checkpoint "$CKPT" \
    --dataset val --output_dir "$OUT" "${DEVICE_ARGS[@]}" || true
  rm -f "$TMP_E"
done

echo ""
echo "### DONE. Results under $OUT_ROOT/{zeroshot_pretrain,loop_triple_peft_k6_dehnn,full_finetune_dehnn}/"
