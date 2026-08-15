#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
TORCHRUN="$ROOT/.venv/bin/torchrun"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DRY_RUN=0
PREPARE_ASSETS=1
AUTO_RESUME=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-assets) PREPARE_ASSETS=0 ;;
    --no-resume) AUTO_RESUME=0 ;;
    *)
      printf 'usage: %s [--dry-run] [--skip-assets] [--no-resume]\n' "$0" >&2
      exit 2
      ;;
  esac
  shift
done

for executable in "$PYTHON" "$TORCHRUN"; do
  if [[ ! -x "$executable" ]]; then
    printf 'Missing .venv executable: %s\n' "$executable" >&2
    exit 1
  fi
done
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NPROC_PER_NODE must be a positive integer, got %s\n' "$NPROC_PER_NODE" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/artifacts/hf_cache/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/artifacts/runtime_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$ROOT/outputs/logs" "$XDG_CACHE_HOME"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_stage() {
  local stage="$1"
  shift
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage"
  print_command "$@"
  if (( DRY_RUN )); then
    return
  fi
  "$@" > >(tee -a "$ROOT/outputs/logs/${stage}.log") 2>&1
}

skip_stage() {
  printf '\n[%s] %s: complete, skipping\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

all_files_exist() {
  local path
  for path in "$@"; do
    [[ -f "$path" ]] || return 1
  done
}

data_assets_complete() {
  all_files_exist \
    outputs/base/qwen3_14b_full_attnres/checkpoint_manifest.json \
    outputs/data/splits.json outputs/data/data_manifest.json \
    outputs/data/mbpp_full_preprocessed/preprocessing_manifest.json || return 1
  "$PYTHON" -c '
import json
from pathlib import Path
report = json.loads(Path("outputs/data/data_manifest.json").read_text())
mbpp = json.loads(Path("outputs/data/mbpp_full_preprocessed/preprocessing_manifest.json").read_text())
assert set(report["tasks"]) == {"math", "multihop", "code"}
assert report["leakage_audit"]["status"] == "PASS"
assert report["counts"]["stage2_discovery"] == {"math": 1000, "multihop": 1000, "code": 200}
assert report["counts"]["stage3_adapter_train"] == 1000
assert report["tasks"]["math"]["selected"]["stage3_adapter_train"] == 1000
assert report["tasks"]["multihop"]["selected"]["stage3_adapter_train"] == 1000
assert report["tasks"]["code"]["selected"]["stage3_adapter_train"] == 200
assert report["tasks"]["math"]["pool_official_splits"] == ["test", "train"]
assert report["tasks"]["multihop"]["pool_official_splits"] == ["test", "train", "validation"]
assert report["tasks"]["code"]["pool_official_splits"] == ["test", "train", "validation"]
assert report["leakage_audit"]["allowed_cross_stage_reuse"] == [["stage2_discovery", "stage3_adapter_train"]]
assert mbpp["status"] == "PASS"
' >/dev/null 2>&1
}

probe_complete() {
  all_files_exist \
    outputs/formal/probe/probe_manifest.json \
    outputs/formal/probe/probe_head.safetensors || return 1
  "$PYTHON" -c '
import json
from pathlib import Path
manifest = json.loads(Path("outputs/formal/probe/probe_manifest.json").read_text())
assert manifest["class_mapping"] == {"0": "math", "1": "multihop", "2": "code"}
assert manifest["confidence_threshold"] == 0.5
' >/dev/null 2>&1
}

probe_features_complete() {
  all_files_exist \
    outputs/formal/probe/feature_manifest.json \
    outputs/formal/probe/train_features.safetensors \
    outputs/formal/probe/validation_features.safetensors || return 1
  "$PYTHON" -c '
import json
from pathlib import Path
manifest = json.loads(Path("outputs/formal/probe/feature_manifest.json").read_text())
assert manifest["train_examples_by_class"] == {"math": 200, "multihop": 200, "code": 100}
assert manifest["validation_examples_by_class"] == {"math": 500, "multihop": 500, "code": 45}
' >/dev/null 2>&1
}

evaluation_complete() {
  all_files_exist \
    outputs/formal/evaluation/predictions.jsonl \
    outputs/formal/evaluation/evaluation_results.json \
    outputs/formal/evaluation/efficiency.csv \
    outputs/formal/evaluation/environment.json || return 1
  "$PYTHON" -c '
import json
from pathlib import Path
result = json.loads(Path("outputs/formal/evaluation/evaluation_results.json").read_text())
assert all("code" in scores for scores in result["primary_scores"].values())
' >/dev/null 2>&1
}

if (( PREPARE_ASSETS )); then
  if data_assets_complete; then
    skip_stage "00_prepare_assets"
  else
    run_stage "00_prepare_assets" "$ROOT/scripts/prepare_assets.sh"
  fi
fi

if (( ! DRY_RUN )); then
  "$PYTHON" - <<'PY'
from src.adapter.train_query import validate_adapter_config
from src.adapter.train_fixed_query import validate_fixed_adapter_config
from src.common import load_yaml
from src.discovery.run_all import validate_discovery_config
from src.evaluation.run_evaluation import validate_evaluation_config
from src.probe.extract_features import validate_probe_config

validate_discovery_config(load_yaml("configs/stage2_discovery.yaml"))
validate_adapter_config(load_yaml("configs/stage3_adapter.yaml"))
validate_fixed_adapter_config(load_yaml("configs/stage3_fixed_adapter.yaml"))
validate_probe_config(load_yaml("configs/probe.yaml"))
validate_evaluation_config(load_yaml("configs/evaluation.yaml"))
print("POST_TRAINING_CONFIG_PREFLIGHT_PASS")
PY
fi

stage2_complete=1
for task in math multihop code; do
  all_files_exist \
    "outputs/formal/discovery/$task/partition.json" \
    "outputs/formal/discovery/$task/discovery_result.json" || stage2_complete=0
done
if (( stage2_complete )); then
  skip_stage "01_discovery"
else
  stage2_command=(
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE"
    -m src.discovery.run_all
    --config configs/stage2_discovery.yaml
  )
  if (( AUTO_RESUME )) && [[ -d outputs/formal/discovery ]]; then
    stage2_command+=(--resume outputs/formal/discovery)
  fi
  run_stage "01_discovery" "${stage2_command[@]}"
fi

for task in math multihop code; do
  if all_files_exist \
    "outputs/formal/adapter/$task/query_manifest.json" \
    "outputs/formal/adapter/$task/final_query.safetensors"; then
    skip_stage "02_query_${task}"
  else
    stage3_command=(
      "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE"
      -m src.adapter.train_query --task "$task"
      --config configs/stage3_adapter.yaml
    )
    if (( AUTO_RESUME )); then
      stage3_command+=(--resume auto)
    fi
    run_stage "02_query_${task}" "${stage3_command[@]}"
  fi
done

if all_files_exist \
  outputs/formal/adapter/fixed/partition.json \
  outputs/formal/adapter/fixed/query_manifest.json \
  outputs/formal/adapter/fixed/final_query.safetensors; then
  skip_stage "02_query_fixed"
else
  fixed_stage3_command=(
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE"
    -m src.adapter.train_fixed_query
    --config configs/stage3_fixed_adapter.yaml
  )
  if (( AUTO_RESUME )); then
    fixed_stage3_command+=(--resume auto)
  fi
  run_stage "02_query_fixed" "${fixed_stage3_command[@]}"
fi

if probe_complete; then
  skip_stage "03_probe"
else
  if probe_features_complete; then
    skip_stage "03_probe_extract"
  else
    run_stage "03_probe_extract" \
      "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
      -m src.probe.extract_features --config configs/probe.yaml
  fi
  run_stage "03_probe_train" \
    "$PYTHON" -m src.probe.train_probe --config configs/probe.yaml
fi

if evaluation_complete; then
  skip_stage "04_evaluation"
else
  run_stage "04_evaluation" \
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m src.evaluation.run_evaluation --config configs/evaluation.yaml
fi

if (( DRY_RUN )); then
  printf '\nPOST_TRAINING_PIPELINE_DRY_RUN_PASS\n'
else
  printf '\nPOST_TRAINING_PIPELINE_COMPLETE\n'
fi
