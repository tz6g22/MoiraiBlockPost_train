#!/usr/bin/env bash
# Bounded Qwen3-14B formal-stage connectivity on a two-GPU A100 node.
# This is not a formal token-budget training launch.
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --job-name=moirai-formal-connect
#SBATCH --output=outputs/formal/formal_train_connectivity_%j.log
#SBATCH --error=outputs/formal/formal_train_connectivity_%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

export QWEN3_14B_PATH="${QWEN3_14B_PATH:-artifacts/models/Qwen3-14B}"
export QWEN3_FORMAL_MASTER_PORT="${QWEN3_FORMAL_MASTER_PORT:-29622}"
export TOKENIZERS_PARALLELISM=false

exec .venv/bin/python -u -m torch.distributed.run \
  --rdzv_backend static \
  --master_addr 127.0.0.1 \
  --master_port "$QWEN3_FORMAL_MASTER_PORT" \
  --nproc_per_node 2 \
  -m src.formal.pipeline \
  --config qwen3_14b_config.yaml \
  --stage train \
  --max-steps 2
