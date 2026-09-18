#!/usr/bin/env bash
# Two-GPU formal Task-Adaptive Qwen3-1.7B post-training.
# Discovery artifacts are supplied by the run-scoped directory; this stage
# starts training only and never performs an implicit Discovery fallback.
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=taibar-formal-retrain
#SBATCH --output=outputs/formal_retrain/slurm/main_%j.out
#SBATCH --error=outputs/formal_retrain/slurm/main_%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export QWEN3_1_7B_PATH="${QWEN3_1_7B_PATH:-artifacts/models/Qwen3-1.7B}"
export QWEN3_FORMAL_MASTER_PORT="${QWEN3_FORMAL_MASTER_PORT:-29528}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec .venv/bin/python -u -m src.formal.pipeline \
  --config qwen3_1.7b_config.yaml \
  --stage train
