#!/bin/bash
# Native Qwen3-14B MBPP validation on the same 10-case manifest.
#SBATCH --job-name=native-mbpp-val10
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/native-mbpp-val10-%j.out
#SBATCH --error=/iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train/outputs/logs/native-mbpp-val10-%j.err

set -euo pipefail
cd /iridisfs/scratch/tz6g22/kimi/MoiraiBlockPost_train
mkdir -p outputs/logs
source .venv/bin/activate
export PYTHONPATH="$PWD"
torchrun --standalone --nproc_per_node=2 scripts/eval_native_qwen3_mbpp_validation_10.py
