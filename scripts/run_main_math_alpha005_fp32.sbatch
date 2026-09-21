#!/usr/bin/env bash
# Math-only Alpha=0.05 run with FP32 Query/Alpha parameters.
# Discovery artifacts are reused; no Multi-hop stage is enabled.
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=main-math-alpha005-fp32
#SBATCH --output=outputs/formal_retrain/slurm/main_math_alpha005_fp32_%j.out
#SBATCH --error=outputs/formal_retrain/slurm/main_math_alpha005_fp32_%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export QWEN3_1_7B_PATH="${QWEN3_1_7B_PATH:-artifacts/models/Qwen3-1.7B}"
export QWEN3_FORMAL_MASTER_PORT="${QWEN3_FORMAL_MASTER_PORT:-29559}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec .venv/bin/python -u -m src.formal.pipeline \
  --config outputs/formal_retrain/main_math_alpha005_fp32_retry1/candidate_config.yaml \
  --stage train \
  --max-steps 2706
