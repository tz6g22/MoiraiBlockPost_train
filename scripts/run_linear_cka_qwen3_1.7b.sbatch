#!/usr/bin/env bash
# Experimental ordinary-residual Linear CKA Discovery only.
# This job does not run post-training or modify the cosine Discovery outputs.
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=moirai-cka-discovery
#SBATCH --output=outputs/formal/qwen3_1.7b/discovery_linear_cka_job_%j.out
#SBATCH --error=outputs/formal/qwen3_1.7b/discovery_linear_cka_job_%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/artifacts/hf_cache/datasets}"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/artifacts/runtime_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec .venv/bin/python -u -m src.discovery.run_all \
  --config configs/qwen3_1.7b_discovery_linear_cka.yaml
