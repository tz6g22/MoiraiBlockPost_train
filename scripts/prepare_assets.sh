#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing .venv. Run scripts/setup_venv.sh first.\n' >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/artifacts/hf_cache/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -m src.modeling.prepare_hf_checkpoint \
  --config configs/model_source.yaml "$@"
if [[ ! -f outputs/data/mbpp_full_preprocessed/preprocessing_manifest.json ]]; then
  "$PYTHON" -m src.data.prepare_mbpp
fi
"$PYTHON" -m src.data.prepare_training_sources --config configs/data.yaml
"$PYTHON" -m src.data.prepare_post_data --config configs/data.yaml
