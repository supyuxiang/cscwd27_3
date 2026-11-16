#!/bin/bash
#SBATCH --job-name=grpo-eval
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --mem=64G
#SBATCH --nodelist=node5
#SBATCH --time=06:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# Merge the finished GRPO FSDP actor (global_step_30) to HuggingFace weights,
# then vLLM avg@4 on math500 / aime24 / aime25 / aime26 / amc23.
# Prompts are raw build_prompt strings. No chat template, no HF generate.
set -euo pipefail
cd /home/yxfeng/cscwd27_iraker

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/home/yxfeng/cscwd27_iraker/verl:/home/yxfeng/cscwd27_iraker${PYTHONPATH:+:$PYTHONPATH}
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
ACTOR=/data3/yxfeng/cscwd27_iraker/run1/verl_grpo/global_step_30/actor
MERGED=/data3/yxfeng/cscwd27_iraker/run1/verl_grpo/global_step_30/hf
OUT=/data3/yxfeng/cscwd27_iraker/run1/grpo_eval.json

if [[ ! -f "${MERGED}/model.safetensors" && ! -f "${MERGED}/model.safetensors.index.json" ]]; then
  rm -rf "${MERGED}"
  "$PY" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR}" \
    --target_dir "${MERGED}" \
    --use_cpu_initialization
fi

"$PY" scripts/eval.py \
  --model "${MERGED}" \
  --output "${OUT}" \
  --sets math500,aime24,aime25,aime26,amc23 \
  --n-sampling 4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 3072 \
  --gpu 0

echo "[grpo-eval] wrote ${OUT}"
