#!/usr/bin/env bash
# Continue an existing formal Moirai query for one additional 1000-example pass.
#SBATCH --job-name=moirai-query-2x
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=60:00:00
#SBATCH --output=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/query-2x-%j.out
#SBATCH --error=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/query-2x-%j.err

set -euo pipefail
: "${TASK:?TASK must be math or multihop}"
case "$TASK" in math|multihop) ;; *) exit 2 ;; esac
ROOT=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train
START="$ROOT/outputs/formal/adapter_lr3e-5/$TASK/final_query.safetensors"
OUT="$ROOT/outputs/formal/adapter_lr3e-5_2x/$TASK"
CACHE=/iridisfs/scratch/tz6g22/kimi/KimiBlockAttnRes/outputs/formal/task_specific/multihop_tokenmatched/moirai_training_inputs.pt
mkdir -p "$ROOT/outputs/logs"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ARGS=(--task "$TASK" --start-query "$START" --output "$OUT")
if [[ "$TASK" == multihop ]]; then ARGS+=(--multihop-cache "$CACHE"); fi
torchrun --standalone --nproc_per_node=2 scripts/continue_query_training.py "${ARGS[@]}"
