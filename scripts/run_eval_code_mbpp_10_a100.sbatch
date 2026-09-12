#!/bin/bash
# Independent 10-case MBPP Code evaluation using the canonical MoiraiBlock path.
#SBATCH --job-name=moirai-code10
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/code10-moirai-%j.out
#SBATCH --error=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/code10-moirai-%j.err
set -euo pipefail
cd /iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train
mkdir -p outputs/logs outputs/formal/evaluation_code_mbpp_10
source .venv/bin/activate
export PYTHONPATH="$PWD"
torchrun --standalone --nproc_per_node=2 scripts/eval_code_mbpp_10.py
