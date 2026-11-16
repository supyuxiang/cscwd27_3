#!/bin/bash
#SBATCH --job-name=dpo-rand-1.5b
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=160G
#SBATCH --nodelist=node4
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# Random positive/negative DPO on Qwen2.5-1.5B, the pair-selection control
# for the ranker run in dpo_ranker_1d5b.
# Each question contributes one pair: a uniformly random correct response
# and a uniformly random incorrect response. Questions missing either side
# are skipped. DPO and the avg@4 eval match the ranker run.
# Iteration 0 reuses that run's base reward sample, so the first comparison
# is the same 8 responses per question with a different pair rule.
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
BASE=/data3/yxfeng/models/Qwen2.5/Qwen2.5-1.5B
TRAIN_JSON=/home/yxfeng/datasets/sft_qwen2.5-3B/en/train_sft_Qwen_math7b_instruct_correct.json
ROOT=/data3/yxfeng/cscwd27_iraker/dpo_random_1d5b
RANKER_POOL=/data3/yxfeng/cscwd27_iraker/dpo_ranker_1d5b/iter_00/reward_candidates.jsonl
BASE_EVAL=/data3/yxfeng/cscwd27_iraker/dpo_ranker_1d5b/base_eval.json
ITERS=3
K=8

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NSHARD=${#GPU_LIST[@]}
EXPECT=$("$PY" -c "import json; print(len(json.load(open('${TRAIN_JSON}'))))")
mkdir -p "$ROOT"
if [[ -s "$BASE_EVAL" && ! -s "${ROOT}/base_eval.json" ]]; then
  cp -a "$BASE_EVAL" "${ROOT}/base_eval.json"
fi
echo "base=$BASE iters=$ITERS questions=$EXPECT gpus=${GPU_LIST[*]} root=$ROOT"

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
base_path = root / "base_eval.json"
base = json.loads(base_path.read_text()) if base_path.is_file() else {}
base_macro = base.get("macro_avg")
for path in sorted(root.glob("iter_*/eval.json")):
    evaluation = json.loads(path.read_text())
    stats_path = path.parent / "pair_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    compact = {}
    for key, value in evaluation.items():
        compact[key] = value.get("mean_acc") if isinstance(value, dict) else value
    row = {"iter": path.parent.name, "eval": compact, "pairs": stats}
    if base_macro is not None and "macro_avg" in compact:
        row["gain_vs_base"] = compact["macro_avg"] - base_macro
    rows.append(row)
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

  if [[ ! -s "${dir}/candidates.jsonl" ]]; then
    if [[ "$iter" -eq 0 && -s "$RANKER_POOL" ]]; then
      cp -a "$RANKER_POOL" "${dir}/candidates.jsonl"
      echo "[iter 0] reused ranker reward sample ${RANKER_POOL}"
    else
      run_shards "${dir}/shards/sample.jsonl" \
        "$PY" scripts/dpo_ranker_loop.py sample \
        --model "$model" --train-json "$TRAIN_JSON" --limit 0 \
        --k "$K" --seed "$((10042 + iter))" --temperature 0.8 --top-p 0.95 \
        --max-new-tokens 2048 --chunk 4
      mapfile -t inputs < <(shard_inputs "${dir}/shards/sample.jsonl")
      "$PY" scripts/dpo_ranker_loop.py merge-samples \
        --inputs "${inputs[@]}" --output "${dir}/candidates.jsonl" \
        --k "$K" --expect "$EXPECT"
    fi
  fi

  if [[ ! -s "${dir}/pairs.jsonl" ]]; then
    "$PY" scripts/dpo_ranker_loop.py random-pairs \
      --candidates "${dir}/candidates.jsonl" \
      --output "${dir}/pairs.jsonl" --stats "${dir}/pair_stats.json" \
      --seed "$((42 + iter))"
  fi

  if [[ ! -s "${dir}/policy/config.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" scripts/dpo_ranker_loop.py dpo \
      --model "$model" --pairs "${dir}/pairs.jsonl" --output-dir "$dir" \
      --beta 0.1 --lr 5e-6 --epochs 1 --grad-accum 16 --max-length 4096 \
      --swan-name "$(printf 'dpo-random-1d5b-iter%02d' "$iter")"
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

echo "===== random dpo finished ====="
write_summary
