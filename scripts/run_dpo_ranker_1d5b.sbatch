#!/bin/bash
#SBATCH --job-name=dpo-1.5b
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=160G
#SBATCH --nodelist=node5
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# Same ranker-DPO loop as the 3B run, starting from the Qwen2.5-1.5B base.
# Iteration 0 samples this model. It does not reuse the 3B trajectories or ranker.
# A same-protocol base eval is written first so the gain is avg@4 on the five sets.
set -euo pipefail
cd /home/yxfeng/cscwd27_iraker
mkdir -p logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export BASE=/data3/yxfeng/models/Qwen2.5/Qwen2.5-1.5B
export TRAIN_JSON=/home/yxfeng/datasets/sft_qwen2.5-3B/en/train_sft_Qwen_math7b_instruct_correct.json
export ROOT=/data3/yxfeng/cscwd27_iraker/dpo_ranker_1d5b
export REUSE=/data3/yxfeng/cscwd27_iraker/dpo_ranker_1d5b/no_reuse
export ITERS=3

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
mkdir -p "$ROOT"
IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ ! -s "${ROOT}/base_eval.json" ]]; then
  echo "[base] eval ${BASE}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" scripts/eval.py \
    --model "$BASE" --output "${ROOT}/base_eval.json" \
    --sets math500,aime24,aime25,aime26,amc23 \
    --n-sampling 4 --temperature 0.7 --top-p 0.95 --max-new-tokens 3072 --gpu 0
fi

bash scripts/run_dpo_ranker.sbatch
