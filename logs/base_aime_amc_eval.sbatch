#!/bin/bash
#SBATCH --job-name=base-aime-amc
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodelist=node4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/base-aime-amc-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/base-aime-amc-%j.err
set -euo pipefail
cd /home/yxfeng/cscwd27_iraker
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
/home/yxfeng/miniconda3/envs/verl1/bin/python scripts/eval.py \
  --model /data3/yxfeng/models/Qwen2.5/Qwen2.5-3B \
  --sets aime24,aime25,aime26,amc23 \
  --n-sampling 4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 3072 \
  --gpu 0 \
  --output /data3/yxfeng/cscwd27_iraker/run1/base_aime_amc.json
echo "BASE_EVAL_DONE"
