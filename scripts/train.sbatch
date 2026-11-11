#!/bin/bash
#SBATCH --job-name=dpo-ranker
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/dpo_raker
mkdir -p logs
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
/home/yxfeng/miniconda3/envs/www27/bin/python scripts/run.py configs/math_qwen7b.yaml
