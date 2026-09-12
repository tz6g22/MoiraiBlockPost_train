#!/bin/bash
# 30-case canonical MoiraiBlock evaluation using lr=3e-5 query bundles.
#SBATCH --job-name=moirai-eval-lr3e5
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/eval-rerun-lr3e5-%j.out
#SBATCH --error=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/eval-rerun-lr3e5-%j.err
set -euo pipefail
cd /iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train
mkdir -p outputs/logs outputs/formal/evaluation_rerun_lr3e-5
source .venv/bin/activate
export PYTHONPATH="$PWD"
torchrun --standalone --nproc_per_node=2 -m src.evaluation.run_evaluation \
  --config configs/evaluation.yaml \
  --output-dir outputs/formal/evaluation_rerun_lr3e-5
