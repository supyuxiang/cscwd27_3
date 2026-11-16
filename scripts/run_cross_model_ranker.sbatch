#!/bin/bash
#SBATCH --job-name=ranker-xmodel
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:3
#SBATCH --mem=120G
#SBATCH --nodelist=node5
#SBATCH --time=08:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# Cross-model ranker eval on the 3 node5 4090s not used by verl GRPO.
# One GPU replays one source model. The test set is the INLP draw:
# 500 correct + 500 incorrect per language, seed 42.
set -euo pipefail
cd /home/yxfeng/cscwd27_iraker

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if ((${#GPUS[@]} < 3)); then
  echo "need 3 visible GPUs, got ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

SOURCES=(Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B)
pids=()
for i in 0 1 2; do
  src="${SOURCES[$i]}"
  gpu="${GPUS[$i]}"
  echo "[launch] gpu=${gpu} source=${src}"
  CUDA_VISIBLE_DEVICES="${gpu}" "$PY" scripts/cross_model_ranker.py --source "${src}" \
    > "logs/ranker-xmodel-${src}.out" 2> "logs/ranker-xmodel-${src}.err" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
exit "${fail}"
