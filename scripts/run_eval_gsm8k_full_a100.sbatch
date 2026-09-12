#!/bin/bash
# Full local GSM8K evaluation using the canonical MoiraiBlock path.
#SBATCH --job-name=moirai-gsm8k-full
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/gsm8k-full-moirai-%j.out
#SBATCH --error=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/gsm8k-full-moirai-%j.err
set -euo pipefail
cd /iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train
mkdir -p outputs/logs outputs/formal/evaluation_gsm8k_full
source .venv/bin/activate
export PYTHONPATH="$PWD"
torchrun --standalone --nproc_per_node=2 scripts/eval_moirai_gsm8k_full_once.py
