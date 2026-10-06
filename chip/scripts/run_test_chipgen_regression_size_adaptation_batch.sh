#!/usr/bin/env bash
#
# Size-adaptation study: batch-test the default finetune variants on progressively
# larger OOD graph sizes, mirroring `run_test_chipgen_regression_batch.sh` but
# pointed at configs in `configs/chipgen/regression/size_adaptation/`.
#
# Variants evaluated by default:
#   - full_finetune_1000_2000   (all params trainable, no size conditioning)
#   - global_peft_1000_2000     (global modules + decoder trainable)
#   - local_peft_1000_2000      (pure-local GNN layers + decoder trainable)
#   - size_film_1000_2000       (FiLM size-conditioning on top of full finetune)
#   - film_decoder_peft_1000_2000 (FiLM/size pathway + decoder only; backbone frozen)
#
# All variants were finetuned on N = 1000-2000 from the transformer baseline.
# Evaluation shards span below / inside / above the training range so we can
# see both interpolation and extrapolation behaviour.
#
# Default behavior:
# - Resolve each checkpoint from its config's `save_dir/best_model.pt`.
# - Skip a variant if `best_model.pt` is missing.
# - Write one merged log per checkpoint variant (not one log per dataset shard).
# - Evaluation uses a checkpoint-native config with only dataset fields overridden.
# - Skip-if-done: a shard's eval is reused from its regression_eval_val.json if
#   present. The cached metrics are still summarised in the new merged log so
#   the merged log stays complete across all shards. Pass --force to re-run
#   every shard from scratch.
#
# Usage:
#   ./run_test_chipgen_regression_size_adaptation_batch.sh
#     Evaluate all default size_adaptation variants on the full dataset sweep.
#     Only shards without a cached regression_eval_val.json are actually run.
#
#   ./run_test_chipgen_regression_size_adaptation_batch.sh --force
#     Re-run every shard for every variant; overwrite cached results.
#
#   ./run_test_chipgen_regression_size_adaptation_batch.sh <config-or-checkpoint> [...]
#     Run only the explicitly supplied config paths and/or checkpoint paths.
#
#   ./run_test_chipgen_regression_size_adaptation_batch.sh --packing-alg-n-values <n_min> <n_max> [<config-or-checkpoint> ...]
#     Restrict filtered-dataset preparation input files to packing_alg_nXXX_batch*.pickle
#     where XXX is in the inclusive range [n_min, n_max].
#
# Notes:
# - Evaluation uses only raw filtered shards under test_filtered_packing_alg (no Metis mirror dir).
# - Evaluation uses batch_size=1 for stable large-graph testing.
# - With dataset.require_aux_features=true (default), missing aux feature files stop
#   the run at data load.
#

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_chipgen_regression.py"
PREPARE_PACKING_ALG_SCRIPT="$REPO_ROOT/scripts/dataset_generation/prepare_filtered_packing_alg_dataset.py"
GENERATE_PACKING_ALG_SCRIPT="$REPO_ROOT/scripts/dataset_generation/generate_packing_alg_dataset.sh"
PYTHONPATH_WITH_REPO="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CONFIG_DIR="configs/chipgen/regression/size_adaptation"
LOG_DIR="data/chipgen/test_logs"
OUTPUT_ROOT="data/chipgen/test_outputs_regression"
TRAINING_PACKING_ALG_DIR="data/chipgen/raw/training_packing_alg"
FILTERED_DATASET_DIR="data/chipgen/raw/test_filtered_packing_alg"

DEFAULT_CONFIGS=(
  "$CONFIG_DIR/full_finetune_1000_2000.yaml"
  "$CONFIG_DIR/global_peft_1000_2000.yaml"
  "$CONFIG_DIR/local_peft_1000_2000.yaml"
  "$CONFIG_DIR/size_film_1000_2000.yaml"
  "$CONFIG_DIR/film_decoder_peft_1000_2000.yaml"
)

DEFAULT_CHECKPOINTS=()

DATASETS=(
  "${FILTERED_DATASET_DIR}/test_filtered_200_400.pickle"
  "${FILTERED_DATASET_DIR}/test_filtered_400_600.pickle"
  "${FILTERED_DATASET_DIR}/test_filtered_800_1000.pickle"
  "${FILTERED_DATASET_DIR}/test_filtered_1400_1600.pickle"
  "${FILTERED_DATASET_DIR}/test_filtered_1800_2000.pickle"
  # 2400_2800 replaces the earlier 2400_2600 shard: same lower bound but a
  # wider window so we get 20 graphs (the 2400_2600 shard only held 3 because
  # raw source files have a natural gap in 2450-2600).
  "${FILTERED_DATASET_DIR}/test_filtered_2400_2800.pickle"
)

PACKING_ALG_N_MIN=""
PACKING_ALG_N_MAX=""
# Skip per-shard eval if regression_eval_val.json already exists.
# Use --force to re-run every shard unconditionally.
FORCE=false
USER_INPUTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --force)
      FORCE=true
      shift
      continue
      ;;
    --packing-alg-n-values|--packing_alg_n_values)
      if [ $# -lt 3 ]; then
        echo "Error: --packing-alg-n-values requires two integer values: <n_min> <n_max>." >&2
        exit 1
      fi
      if [[ ! "$2" =~ ^[0-9]+$ ]] || [[ ! "$3" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid n-range values '$2' '$3' (expected integers)." >&2
        exit 1
      fi
      PACKING_ALG_N_MIN="$2"
      PACKING_ALG_N_MAX="$3"
      if [ "$PACKING_ALG_N_MIN" -gt "$PACKING_ALG_N_MAX" ]; then
        echo "Error: invalid n-range: min ($PACKING_ALG_N_MIN) must be <= max ($PACKING_ALG_N_MAX)." >&2
        exit 1
      fi
      shift 3
      continue
      ;;
    --packing-alg-n-values=*|--packing_alg_n_values=*)
      csv_values="${1#*=}"
      IFS=',' read -r -a parsed_values <<< "$csv_values"
      if [ "${#parsed_values[@]}" -ne 2 ]; then
        echo "Error: --packing-alg-n-values=<n_min>,<n_max> expects exactly two values." >&2
        exit 1
      fi
      if [[ ! "${parsed_values[0]}" =~ ^[0-9]+$ ]] || [[ ! "${parsed_values[1]}" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid n-range values '${parsed_values[0]}' '${parsed_values[1]}' (expected integers)." >&2
        exit 1
      fi
      PACKING_ALG_N_MIN="${parsed_values[0]}"
      PACKING_ALG_N_MAX="${parsed_values[1]}"
      if [ "$PACKING_ALG_N_MIN" -gt "$PACKING_ALG_N_MAX" ]; then
        echo "Error: invalid n-range: min ($PACKING_ALG_N_MIN) must be <= max ($PACKING_ALG_N_MAX)." >&2
        exit 1
      fi
      shift
      continue
      ;;
    *)
      USER_INPUTS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$REPO_ROOT/$LOG_DIR" "$REPO_ROOT/$OUTPUT_ROOT"

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "Error: test script not found: $TEST_SCRIPT" >&2
  exit 1
fi

generate_raw_packing_alg_data() {
  echo "=============================================="
  echo "Raw packing_alg data missing or empty. Generating..."
  echo "=============================================="
  if [ ! -f "$GENERATE_PACKING_ALG_SCRIPT" ]; then
    echo "Error: generate script not found: $GENERATE_PACKING_ALG_SCRIPT" >&2
    exit 1
  fi
  (
    cd "$REPO_ROOT" && \
    bash "$GENERATE_PACKING_ALG_SCRIPT" \
      --output_dir "$TRAINING_PACKING_ALG_DIR" \
      --sweep_start 500 \
      --sweep_end 1000 \
      --sweep_step 500 \
      --graphs_per_target 50 \
      --graphs_per_batch 100 && \
    bash "$GENERATE_PACKING_ALG_SCRIPT" \
      --output_dir "$TRAINING_PACKING_ALG_DIR" \
      --sweep_start 2000 \
      --sweep_end 10000 \
      --sweep_step 1000 \
      --graphs_per_target 50 \
      --graphs_per_batch 100
  )
  echo ""
}

resolve_repo_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$path"
  fi
}

resolve_checkpoint_from_config() {
  local config_path="$1"
  "$PYTHON_BIN" - "$config_path" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open() as f:
    config = yaml.safe_load(f)

save_dir = config.get("save_dir")
if not save_dir:
    raise SystemExit(f"Config is missing save_dir: {config_path}")

print(Path(save_dir) / "best_model.pt")
PY
}

CHECKPOINTS=()
CONFIG_SOURCES=()
if [ ${#USER_INPUTS[@]} -gt 0 ]; then
  for item in "${USER_INPUTS[@]}"; do
    if [[ "$item" == *.yaml || "$item" == *.yml ]]; then
      CONFIG_PATH="$(resolve_repo_path "$item")"
      if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: config not found: $item" >&2
        exit 1
      fi
      CHECKPOINTS+=("$(resolve_checkpoint_from_config "$CONFIG_PATH")")
      CONFIG_SOURCES+=("${CONFIG_PATH#$REPO_ROOT/}")
    else
      CHECKPOINTS+=("$item")
      CONFIG_SOURCES+=("")
    fi
  done
else
  for checkpoint_path in "${DEFAULT_CHECKPOINTS[@]}"; do
    CHECKPOINTS+=("$checkpoint_path")
    CONFIG_SOURCES+=("")
  done
  for config_path in "${DEFAULT_CONFIGS[@]}"; do
    ABS_CONFIG_PATH="$(resolve_repo_path "$config_path")"
    if [ ! -f "$ABS_CONFIG_PATH" ]; then
      echo "Error: default config not found: $config_path" >&2
      exit 1
    fi
    CHECKPOINTS+=("$(resolve_checkpoint_from_config "$ABS_CONFIG_PATH")")
    CONFIG_SOURCES+=("$config_path")
  done
fi

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "Error: no regression configs or checkpoints selected." >&2
  exit 1
fi

MISSING_DATASETS=()
for dataset in "${DATASETS[@]}"; do
  base="$REPO_ROOT/$dataset"
  if [[ ! -f "$base" ]]; then
    MISSING_DATASETS+=("$dataset")
  fi
done

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
  raw_pickles=( "$REPO_ROOT/$TRAINING_PACKING_ALG_DIR"/packing_alg_*.pickle )
  if [ ${#raw_pickles[@]} -eq 0 ]; then
    generate_raw_packing_alg_data
  fi

  echo "=============================================="
  echo "Some packing_alg filtered test datasets are missing."
  echo "Preparing missing datasets from DATASETS array..."
  echo "=============================================="
  if [ ! -f "$PREPARE_PACKING_ALG_SCRIPT" ]; then
    echo "Error: packing_alg preparation script not found: $PREPARE_PACKING_ALG_SCRIPT" >&2
    exit 1
  fi

  PREPARE_SOURCE_ARGS=()
  if [ -n "$PACKING_ALG_N_MIN" ] && [ -n "$PACKING_ALG_N_MAX" ]; then
    PREPARE_SOURCE_ARGS=( --n_values "$PACKING_ALG_N_MIN" "$PACKING_ALG_N_MAX" )
    echo "Using source n-range: [$PACKING_ALG_N_MIN, $PACKING_ALG_N_MAX]"
  fi

  RANGE_ARGS=()
  FAILED_DATASET_GENERATION=0
  for dataset in "${MISSING_DATASETS[@]}"; do
    dataset_file="$(basename "$dataset")"
    range_part="${dataset_file#test_filtered_}"
    range_part="${range_part%.pickle}"

    if [[ ! "$range_part" =~ ^([0-9]+)_([0-9]+)$ ]]; then
      echo "Warning: dataset filename does not match test_filtered_<min>_<max>.pickle, skipping: $dataset" >&2
      FAILED_DATASET_GENERATION=1
      continue
    fi

    RANGE_ARGS+=("$range_part")
    echo "Queued missing dataset: $dataset_file (range ${BASH_REMATCH[1]}-${BASH_REMATCH[2]})"
  done

  if [ ${#RANGE_ARGS[@]} -gt 0 ]; then
    (
      cd "$REPO_ROOT" && \
      PYTHONPATH="$PYTHONPATH_WITH_REPO" "$PYTHON_BIN" "$PREPARE_PACKING_ALG_SCRIPT" \
        --input_dir "$TRAINING_PACKING_ALG_DIR" \
        --output_dir "$FILTERED_DATASET_DIR" \
        --max_graphs 20 \
        "${PREPARE_SOURCE_ARGS[@]}" \
        --ranges "${RANGE_ARGS[@]}"
    )
  fi

  if [ "$FAILED_DATASET_GENERATION" -ne 0 ]; then
    echo "Warning: one or more missing datasets could not be generated due to invalid names." >&2
  fi
  echo ""
fi

AVAILABLE_DATASETS=()
for dataset in "${DATASETS[@]}"; do
  base="$REPO_ROOT/$dataset"
  if [[ -f "$base" ]]; then
    AVAILABLE_DATASETS+=("$dataset")
  fi
done

if [ ${#AVAILABLE_DATASETS[@]} -eq 0 ]; then
  echo "Error: no filtered packing_alg test datasets are available." >&2
  exit 1
fi

TMP_CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/regression-chipgen-eval.XXXXXX")"
cleanup() {
  rm -rf "$TMP_CONFIG_DIR"
}
trap cleanup EXIT

echo "Checkpoint variants: ${#CHECKPOINTS[@]}"
echo "Datasets per variant: ${#AVAILABLE_DATASETS[@]}"
echo "Filtered test dir: $FILTERED_DATASET_DIR"
echo "Selection mode: config-derived checkpoints with optional explicit overrides"
echo "Logging mode: one merged log per checkpoint variant"
echo "Evaluation mode: checkpoint-native config with dataset-only override"
echo "Inference mode: direct regression forward pass"
if [ "$FORCE" = "true" ]; then
  echo "Resume mode:  --force (re-run every shard)"
else
  echo "Resume mode:  skip-if-done (reuse regression_eval_val.json when present; --force to override)"
fi
echo ""

LOG_FILES_WRITTEN=()

for idx in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[$idx]}"
  CONFIG_SOURCE="${CONFIG_SOURCES[$idx]}"
  CHECKPOINT_PATH="$checkpoint"
  if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH="$REPO_ROOT/$CHECKPOINT_PATH"
  fi

  if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Skipping missing checkpoint: $checkpoint" >&2
    continue
  fi

  MODEL_NAME="$(basename "$(dirname "$CHECKPOINT_PATH")")"
  MERGED_LOG="$REPO_ROOT/$LOG_DIR/test_chipgen_regression_${MODEL_NAME}_MERGED.log"
  LOG_FILES_WRITTEN+=("${MERGED_LOG#$REPO_ROOT/}")

  echo "=============================================="
  echo "VARIANT: $MODEL_NAME"
  echo "Checkpoint: ${CHECKPOINT_PATH#$REPO_ROOT/}"
  if [ -n "$CONFIG_SOURCE" ]; then
    echo "Config: $CONFIG_SOURCE"
  fi
  echo "Datasets: ${#AVAILABLE_DATASETS[@]}"
  echo "Merged log: ${MERGED_LOG#$REPO_ROOT/}"
  echo "=============================================="

  {
    echo "=============================================="
    echo "Batch test started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Variant: $MODEL_NAME"
    echo "Checkpoint: ${CHECKPOINT_PATH#$REPO_ROOT/}"
    if [ -n "$CONFIG_SOURCE" ]; then
      echo "Config: $CONFIG_SOURCE"
    fi
    echo "Evaluation: checkpoint-native config with dataset-only override"
    echo "Inference: direct regression forward pass"
    echo "Datasets: ${#AVAILABLE_DATASETS[@]}"
    echo "=============================================="
    echo ""
  } > "$MERGED_LOG"

  run_idx=0
  for dataset in "${AVAILABLE_DATASETS[@]}"; do
    run_idx=$((run_idx + 1))
    DATASET_SLUG="$(basename "$dataset" .pickle)"
    OUTPUT_DIR="$OUTPUT_ROOT/${MODEL_NAME}/${DATASET_SLUG}"
    TEMP_CONFIG="$TMP_CONFIG_DIR/${MODEL_NAME}_${DATASET_SLUG}.yaml"
    CACHED_METRICS="$REPO_ROOT/$OUTPUT_DIR/regression_eval_val.json"

    # Skip-if-done: reuse cached metrics and emit a compact summary to the merged log.
    # Keeps the merged log complete across all shards while only spending GPU time
    # on shards whose JSON doesn't exist yet (e.g. the new 2400_2800 shard).
    if [ -f "$CACHED_METRICS" ] && [ "$FORCE" != "true" ]; then
      echo "  RUN $run_idx/${#AVAILABLE_DATASETS[@]}: $(basename "$dataset") -- SKIP (cached)"
      {
        echo "----------------------------------------------"
        echo "RUN $run_idx/${#AVAILABLE_DATASETS[@]}: cached"
        echo "Dataset: $dataset"
        echo "Output dir: $OUTPUT_DIR"
        echo "Status: SKIP (regression_eval_val.json already exists; --force to re-run)"
        echo "----------------------------------------------"
        PYTHONPATH="$PYTHONPATH_WITH_REPO" "$PYTHON_BIN" - "$CACHED_METRICS" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
keys = ["rmse_physical", "mae_physical", "rmse_norm", "mae_norm", "num_graphs", "num_batches"]
width = max(len(k) for k in keys if k in m)
for k in keys:
    if k in m:
        v = m[k]
        if isinstance(v, float):
            print(f"  {k:<{width}} = {v:.6f}")
        else:
            print(f"  {k:<{width}} = {v}")
PY
        echo ""
      } >> "$MERGED_LOG"
      continue
    fi

    (
      cd "$REPO_ROOT" && \
      PYTHONPATH="$PYTHONPATH_WITH_REPO" "$PYTHON_BIN" - "$CHECKPOINT_PATH" "$dataset" "$TEMP_CONFIG" <<'PY'
from pathlib import Path
import sys
import torch
import yaml

checkpoint_path = Path(sys.argv[1])
dataset_path = sys.argv[2]
output_config_path = Path(sys.argv[3])

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError(f"Checkpoint is missing a usable config: {checkpoint_path}")

dataset_cfg = dict(config.get("dataset", {}))
eval_path = Path(dataset_path)
dataset_cfg["val_dir"] = str(eval_path)
dataset_cfg["val_files"] = None
dataset_cfg["min_graph_size"] = 0
dataset_cfg["max_graph_size"] = None
dataset_cfg["max_val_samples"] = 20
config["dataset"] = dataset_cfg
config["batch_size"] = 1

output_config_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_config_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
    )

    echo "  RUN $run_idx/${#AVAILABLE_DATASETS[@]}: $(basename "$dataset")"
    echo "    Output dir: $OUTPUT_DIR"

    {
      echo "----------------------------------------------"
      echo "RUN $run_idx/${#AVAILABLE_DATASETS[@]}"
      echo "Dataset: $dataset"
      echo "Output dir: $OUTPUT_DIR"
      echo "Temp config: ${TEMP_CONFIG}"
      echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "----------------------------------------------"
      echo ""
    } >> "$MERGED_LOG"

    (
      cd "$REPO_ROOT" && \
      PYTHONPATH="$PYTHONPATH_WITH_REPO" TQDM_DISABLE=1 "$PYTHON_BIN" "$TEST_SCRIPT" \
        --config "$TEMP_CONFIG" \
        --checkpoint "$CHECKPOINT_PATH" \
        --dataset val \
        --output_dir "$OUTPUT_DIR"
    ) 2>&1 | tee -a "$MERGED_LOG"

    {
      echo ""
      echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
      echo ""
    } >> "$MERGED_LOG"
  done

  {
    echo "=============================================="
    echo "Variant finished $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
  } >> "$MERGED_LOG"
done

echo ""
echo "Merged log files written:"
for log_file in "${LOG_FILES_WRITTEN[@]}"; do
  echo "  - $log_file"
done
