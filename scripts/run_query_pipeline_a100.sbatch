#!/usr/bin/env bash
#SBATCH --job-name=moirai-query
#SBATCH --partition=a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
TORCHRUN="$ROOT/.venv/bin/torchrun"
export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/artifacts/hf_cache/datasets}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/artifacts/runtime_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

distributed_stage() {
  srun --ntasks=2 --ntasks-per-node=1 bash -c '
    exec "$1" --nnodes=2 --nproc_per_node=2 \
      --node_rank="$SLURM_PROCID" \
      --master_addr="$2" --master_port="$3" "${@:4}"
  ' bash "$TORCHRUN" "$MASTER_ADDR" "$MASTER_PORT" "$@"
}

distributed_stage -m src.adapter.train_query --task math \
  --config configs/stage3_adapter.yaml --resume auto
distributed_stage -m src.adapter.train_query --task multihop \
  --config configs/stage3_adapter.yaml --resume auto
distributed_stage -m src.adapter.train_query --task code \
  --config configs/stage3_adapter.yaml --resume auto
distributed_stage -m src.adapter.train_fixed_query \
  --config configs/stage3_fixed_adapter.yaml --resume auto
distributed_stage -m src.probe.extract_features --config configs/probe.yaml
srun --ntasks=1 --ntasks-per-node=1 "$PYTHON" -m src.probe.train_probe \
  --config configs/probe.yaml
distributed_stage -m src.evaluation.run_evaluation --config configs/evaluation.yaml
