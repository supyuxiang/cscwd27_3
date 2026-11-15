#!/bin/bash
#SBATCH --job-name=sft-sanity4
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodelist=node4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/sft-sanity4-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/sft-sanity4-%j.err
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
/home/yxfeng/miniconda3/envs/verl1/bin/python /home/yxfeng/cscwd27_iraker/logs/sft_sanity4.py
