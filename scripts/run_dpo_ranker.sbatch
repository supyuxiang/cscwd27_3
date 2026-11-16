#!/bin/bash
#SBATCH --job-name=dpo-ranker
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=160G
#SBATCH --nodelist=node7
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# Iterative ranker-DPO, starting from base Qwen2.5-3B.
#   sample K=8 -> 1/0 labels -> listwise ranker (probe depth 0.6)
#   resample K=8 -> ranker scores -> (highest, lowest) pair per question
#   LoRA DPO (beta 0.1), merge into a full HF policy
#   vLLM avg@4 on math500 / aime24 / aime25 / aime26 / amc23
# Iteration 0 reuses the existing base-model pool, features and ranker.
# Prompts are build_prompt. No chat template.
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

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
BASE=${BASE:-/data3/yxfeng/models/Qwen2.5/Qwen2.5-3B}
TRAIN_JSON=${TRAIN_JSON:-/home/yxfeng/datasets/sft_qwen2.5-3B/en/train_sft_Qwen_math7b_instruct_correct.json}
ROOT=${ROOT:-/data3/yxfeng/cscwd27_iraker/dpo_ranker}
REUSE=${REUSE:-/data3/yxfeng/cscwd27_iraker/run1}
ITERS=${ITERS:-3}
K=8

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NSHARD=${#GPU_LIST[@]}
EXPECT=$("$PY" -c "import json; print(len(json.load(open('${TRAIN_JSON}'))))")
mkdir -p "$ROOT"
echo "base=$BASE iters=$ITERS questions=$EXPECT gpus=${GPU_LIST[*]} root=$ROOT"

# First argument is the per-shard output prefix. Remaining arguments are the
# command, which must not already contain --output / --shard / --shards.
run_shards() {
  local prefix=$1
  shift
  local fail=0
  local -a pids=()
  local shard
  for shard in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$shard]}" "$@" \
      --shard "$shard" --shards "$NSHARD" \
      --output "${prefix}.shard${shard}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "a shard failed" >&2
    exit 1
  fi
}

shard_inputs() {
  local prefix=$1
  local shard
  for shard in "${!GPU_LIST[@]}"; do
    printf '%s\n' "${prefix}.shard${shard}"
  done
}

write_summary() {
  "$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("iter_*/eval.json")):
    evaluation = json.loads(path.read_text())
    stats_path = path.parent / "pair_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    compact = {}
    for key, value in evaluation.items():
        compact[key] = value.get("mean_acc") if isinstance(value, dict) else value
    rows.append({"iter": path.parent.name, "eval": compact, "pairs": stats})
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
}

for ((iter=0; iter<ITERS; iter++)); do
  dir=$(printf '%s/iter_%02d' "$ROOT" "$iter")
  mkdir -p "${dir}/shards"
  if [[ "$iter" -eq 0 ]]; then
    model="$BASE"
  else
    model=$(printf '%s/iter_%02d/policy' "$ROOT" "$((iter - 1))")
  fi
  echo
  echo "========== iter ${iter} model=${model} =========="
  if [[ -f "${dir}/_EVAL_DONE" && -s "${dir}/eval.json" ]]; then
    echo "[skip] iter ${iter} already evaluated"
    continue
  fi

  if [[ ! -s "${dir}/ranker.pt" ]]; then
    reuse_n=0
    if [[ -s "${REUSE}/candidates.jsonl" ]]; then
      reuse_n=$(wc -l < "${REUSE}/candidates.jsonl")
    fi
    if [[ "$iter" -eq 0 && "$model" == "$BASE" && "$reuse_n" -eq "$EXPECT" && -s "${REUSE}/features.pt" && -s "${REUSE}/ranker.pt" ]]; then
      ln -f "${REUSE}/candidates.jsonl" "${dir}/label_candidates.jsonl"
      ln -f "${REUSE}/features.pt" "${dir}/label_features.pt"
      ln -f "${REUSE}/ranker.pt" "${dir}/ranker.pt"
      echo "[iter 0] reuse base label sample, features and ranker"
    else
      if [[ ! -s "${dir}/label_candidates.jsonl" ]]; then
        run_shards "${dir}/shards/label.jsonl" \
          "$PY" scripts/dpo_ranker_loop.py sample \
          --model "$model" --train-json "$TRAIN_JSON" --limit 0 \
          --k "$K" --seed "$((42 + iter))" --temperature 0.8 --top-p 0.95 \
          --max-new-tokens 2048 --chunk 4
        mapfile -t inputs < <(shard_inputs "${dir}/shards/label.jsonl")
        "$PY" scripts/dpo_ranker_loop.py merge-samples \
          --inputs "${inputs[@]}" --output "${dir}/label_candidates.jsonl" \
          --k "$K" --expect "$EXPECT"
      fi
      if [[ ! -s "${dir}/label_features.pt" ]]; then
        run_shards "${dir}/shards/label_features.pt" \
          "$PY" scripts/dpo_ranker_loop.py features \
          --model "$model" --candidates "${dir}/label_candidates.jsonl" \
          --depth 0.6 --batch-size 2 --max-length 4096
        mapfile -t inputs < <(shard_inputs "${dir}/shards/label_features.pt")
        "$PY" scripts/dpo_ranker_loop.py merge-features \
          --inputs "${inputs[@]}" --output "${dir}/label_features.pt" --expect "$EXPECT"
      fi
      "$PY" scripts/dpo_ranker_loop.py ranker \
        --features "${dir}/label_features.pt" --output "${dir}/ranker.pt" \
        --epochs 40 --lr 1e-3 --seed 42
    fi
  fi

  if [[ ! -s "${dir}/reward_candidates.jsonl" ]]; then
    run_shards "${dir}/shards/reward.jsonl" \
      "$PY" scripts/dpo_ranker_loop.py sample \
      --model "$model" --train-json "$TRAIN_JSON" --limit 0 \
      --k "$K" --seed "$((10042 + iter))" --temperature 0.8 --top-p 0.95 \
      --max-new-tokens 2048 --chunk 4
    mapfile -t inputs < <(shard_inputs "${dir}/shards/reward.jsonl")
    "$PY" scripts/dpo_ranker_loop.py merge-samples \
      --inputs "${inputs[@]}" --output "${dir}/reward_candidates.jsonl" \
      --k "$K" --expect "$EXPECT"
  fi

  if [[ ! -s "${dir}/reward_features.pt" ]]; then
    run_shards "${dir}/shards/reward_features.pt" \
      "$PY" scripts/dpo_ranker_loop.py features \
      --model "$model" --candidates "${dir}/reward_candidates.jsonl" \
      --depth 0.6 --batch-size 2 --max-length 4096
    mapfile -t inputs < <(shard_inputs "${dir}/shards/reward_features.pt")
    "$PY" scripts/dpo_ranker_loop.py merge-features \
      --inputs "${inputs[@]}" --output "${dir}/reward_features.pt" --expect "$EXPECT"
  fi

  if [[ ! -s "${dir}/pairs.jsonl" ]]; then
    "$PY" scripts/dpo_ranker_loop.py pairs \
      --features "${dir}/reward_features.pt" --ranker "${dir}/ranker.pt" \
      --output "${dir}/pairs.jsonl" --stats "${dir}/pair_stats.json"
  fi

  if [[ ! -s "${dir}/policy/config.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" scripts/dpo_ranker_loop.py dpo \
      --model "$model" --pairs "${dir}/pairs.jsonl" --output-dir "$dir" \
      --beta 0.1 --lr 5e-6 --epochs 1 --grad-accum 16 --max-length 4096 \
      --swan-name "$(printf 'dpo-ranker-iter%02d' "$iter")"
  fi

  if [[ ! -s "${dir}/eval.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" scripts/eval.py \
      --model "${dir}/policy" --output "${dir}/eval.json" \
      --sets math500,aime24,aime25,aime26,amc23 \
      --n-sampling 4 --temperature 0.7 --top-p 0.95 --max-new-tokens 3072 --gpu 0
  fi
  touch "${dir}/_EVAL_DONE"
  write_summary
  echo "[done] iter ${iter}"
done

echo "===== dpo-ranker finished ====="
write_summary
