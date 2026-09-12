#!/usr/bin/env bash
#SBATCH --job-name=moirai-formal-query
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=outputs/logs/formal-query-%j.out
#SBATCH --error=outputs/logs/formal-query-%j.err

set -Eeuo pipefail
ROOT=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
TORCHRUN="$ROOT/.venv/bin/torchrun"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/artifacts/hf_cache/datasets}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/artifacts/runtime_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${MODE:?MODE must be math, multihop, code, or fixed}"
if [[ "$MODE" == fixed ]]; then
  exec srun --ntasks=1 --ntasks-per-node=1 "$TORCHRUN" --standalone --nproc_per_node=2 \
    -m src.adapter.train_fixed_query --config configs/stage3_fixed_adapter_lr3e-5.yaml
else
  exec srun --ntasks=1 --ntasks-per-node=1 "$TORCHRUN" --standalone --nproc_per_node=2 \
    -m src.adapter.train_query --task "$MODE" --config configs/stage3_adapter_lr3e-5.yaml
fi
