#!/bin/bash
#SBATCH --job-name=iraker
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=128G
#SBATCH --nodelist=node5
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# Full IRAKer run on 2x 4090:
#   stage1 sample -> stage2 features -> stage3 ranker
#   -> route A: rejection-sampling SFT   -> eval
#   -> route B: GRPO with ranker advantage -> eval
# 4 GPUs. Three train the policy, the fourth (index 3) is reserved for the vLLM
# sampler so generation never competes with the policy for GPU memory.
set -euo pipefail
cd /home/yxfeng/cscwd27_iraker
mkdir -p logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
TORCHRUN=/home/yxfeng/miniconda3/envs/verl1/bin/torchrun
NPROC=${NPROC:-3}
SAMPLER_GPU=${SAMPLER_GPU:-3}

MODEL=${MODEL:-/data3/yxfeng/models/Qwen2.5/Qwen2.5-3B}
TRAIN_JSON=${TRAIN_JSON:-/home/yxfeng/datasets/sft_qwen2.5-3B/en/train_sft_Qwen_math7b_instruct_correct.json}
OUT=${OUT:-/data3/yxfeng/cscwd27_iraker/run1}

# Sampling size. K=8 over all 2474 questions is 19792 completions.
LIMIT=${LIMIT:-0}
K=${K:-8}
DEPTH=${DEPTH:-0.6}

mkdir -p "$OUT"
echo "model=$MODEL"
echo "train_json=$TRAIN_JSON"
echo "out=$OUT limit=$LIMIT k=$K depth=$DEPTH"

candidates_ready() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local want="$LIMIT"
  if [[ "$want" -le 0 ]]; then
    want=$(/home/yxfeng/miniconda3/envs/verl1/bin/python -c "import json,sys; print(len(json.load(open('$TRAIN_JSON'))))")
  fi
  local have
  have=$(wc -l < "$path")
  [[ "$have" -ge "$want" ]]
}

# ------------------------------------------------------------------ stage 1
if ! candidates_ready "$OUT/candidates.jsonl" || [[ "${FORCE_SAMPLE:-0}" == "1" ]]; then
  echo "===== stage1: sample ${K} candidates per question ====="
  "$PY" scripts/stage1_sample.py \
    --model "$MODEL" \
    --train-json "$TRAIN_JSON" \
    --output "$OUT/candidates.jsonl" \
    --limit "$LIMIT" \
    --k "$K" \
    --max-new-tokens 2048 \
    --temperature 0.8 \
    --top-p 0.95 \
    --chunk 8 \
    --tensor-parallel 2
else
  echo "[skip] stage1 -> $OUT/candidates.jsonl"
fi

features_ready() {
  local feat="$1"
  local cand="$2"
  [[ -s "$feat" ]] || return 1
  local want have
  want=$(wc -l < "$cand")
  have=$(/home/yxfeng/miniconda3/envs/verl1/bin/python -c "
import torch
d = torch.load('$feat', map_location='cpu', weights_only=False)
print(len(d['groups']))
" 2>/dev/null) || return 1
  [[ "$have" -ge "$want" ]]
}

# ------------------------------------------------------------------ stage 2
if ! features_ready "$OUT/features.pt" "$OUT/candidates.jsonl" || [[ "${FORCE_FEATURES:-0}" == "1" ]]; then
  echo "===== stage2: extract probe-layer features ====="
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/stage2_features.py \
    --model "$MODEL" \
    --candidates "$OUT/candidates.jsonl" \
    --output "$OUT/features.pt" \
    --depth "$DEPTH" \
    --batch-size 2
else
  echo "[skip] stage2 -> $OUT/features.pt"
fi

# ------------------------------------------------------------------ stage 3
if [[ ! -s "$OUT/ranker.pt" ]] || [[ "${FORCE_RANKER:-0}" == "1" ]]; then
  echo "===== stage3: train the lightweight ranker ====="
  "$PY" scripts/stage3_ranker.py \
    --features "$OUT/features.pt" \
    --output "$OUT/ranker.pt" \
    --kind listwise \
    --d-proj 128 \
    --epochs 40 \
    --lr 1e-3
else
  echo "[skip] stage3 -> $OUT/ranker.pt"
fi

# ------------------------------------------------------------- pool analysis
echo "===== analysis: pass@k and ranker top-1 selection ====="
"$PY" scripts/analyse_pool.py \
  --candidates "$OUT/candidates.jsonl" \
  --features "$OUT/features.pt" \
  --ranker "$OUT/ranker.pt" \
  --output "$OUT/analysis_pool.json"

# -------------------------------------------------------- route A: rejection SFT
echo "===== route A: rejection-sampling SFT ====="
if [[ -s "$OUT/sft/policy_sft/config.json" && "${FORCE_ROUTE_A:-0}" != "1" ]]; then
  echo "[skip] route A train -> reuse $OUT/sft/policy_sft, eval only"
  "$PY" scripts/stage4_rejection_sft.py \
    --model "$MODEL" \
    --features "$OUT/features.pt" \
    --ranker "$OUT/ranker.pt" \
    --output-dir "$OUT/sft" \
    --eval \
    --eval-sampling 4 \
    --sampler-gpu "${SAMPLER_GPU}" \
    --skip-train
else
  "$TORCHRUN" --standalone --nproc_per_node=3 \
    scripts/stage4_rejection_sft.py \
    --model "$MODEL" \
    --features "$OUT/features.pt" \
    --ranker "$OUT/ranker.pt" \
    --output-dir "$OUT/sft" \
    --keep 1 \
    --epochs 3 \
    --micro-batch 1 \
    --grad-accum 8 \
    --lr 1e-4 \
    --lora-r 16 \
    --max-length 4096 \
    --eval \
    --eval-sampling 4 \
    --sampler-gpu "${SAMPLER_GPU}"
fi

# ----------------------------------------------------- route B: ranker-advantage GRPO
echo "===== route B: GRPO with ranker advantage ====="
"$TORCHRUN" --standalone --nproc_per_node="${NPROC}" \
  scripts/stage5_grpo.py \
  --model "$MODEL" \
  --train-json "$TRAIN_JSON" \
  --ranker "$OUT/ranker.pt" \
  --output-dir "$OUT/grpo" \
  --limit "$LIMIT" \
  --iterations 30 \
  --questions-per-iter 8 \
  --group-size 8 \
  --inner-epochs 2 \
  --lr 1e-6 \
  --sampler-gpu "${SAMPLER_GPU}" \
  --lora-r 16 \
  --max-new-tokens 2048 \
  --depth "$DEPTH" \
  --eval \
  --eval-sampling 4

# ------------------------------------------------------ baseline A: plain SFT
echo "===== baseline A: SFT on teacher golden_res ====="
"$TORCHRUN" --standalone --nproc_per_node=3 \
  scripts/baseline_sft.py \
  --model "$MODEL" \
  --train-json "$TRAIN_JSON" \
  --output-dir "$OUT/baseline_sft" \
  --epochs 3 \
  --micro-batch 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --lora-r 16 \
  --max-length 4096 \
  --eval \
  --eval-sampling 4 \
  --sampler-gpu "${SAMPLER_GPU}"

# --------------------------------------------------- baseline B: verifier GRPO
echo "===== baseline B: GRPO with verifier advantage ====="
"$TORCHRUN" --standalone --nproc_per_node="${NPROC}" \
  scripts/baseline_grpo.py \
  --model "$MODEL" \
  --train-json "$TRAIN_JSON" \
  --output-dir "$OUT/baseline_grpo" \
  --limit "$LIMIT" \
  --iterations 30 \
  --questions-per-iter 8 \
  --group-size 8 \
  --inner-epochs 2 \
  --lr 1e-6 \
  --sampler-gpu "${SAMPLER_GPU}" \
  --lora-r 16 \
  --max-new-tokens 2048 \
  --eval \
  --eval-sampling 4

echo "ALL_DONE $(date)"

