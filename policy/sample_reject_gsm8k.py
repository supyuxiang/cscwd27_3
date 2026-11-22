"""Sample GSM8K answers with Qwen2.5-Math-7B-Instruct, keep the correct ones.

The teacher is the Instruct checkpoint. There is no local Math-7B base.
Each kept row is a question plus one correct completion, for MySFTDataset_Base.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

TRAIN_JSON = Path("/data3/yxfeng/datasets/gsm8k/sft_base/train.json")
MODEL = "/data3/yxfeng/models/Qwen2.5/Qwen2.5-Math-7B-Instruct"
OUT_DIR = Path("/data3/yxfeng/cscwd27_ranker/gsm8k_rej")
N_SAMPLES = 8
MAX_TOKENS = 768
EXTRACT = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
USER = (
    "Please solve the given math problem step by step and present the answer "
    'in the following format: "\\boxed{{X}}", where X is the answer.\n\n'
    "Question:\n{question}"
)


def _offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)


def grade_one(prediction: str, gold: str) -> bool:
    gold_parsed = parse(gold, EXTRACT)
    pred_parsed = parse((prediction or "").strip(), EXTRACT)
    if not gold_parsed or not pred_parsed:
        return False
    try:
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False


def sample_shard(shard: int, nshards: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    mine = [row for index, row in enumerate(rows) if index % nshards == shard]
    out_path = OUT_DIR / f"samples_shard{shard}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    if out_path.exists():
        done = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
    pending = mine[done:]
    print(f"shard {shard} total={len(mine)} done={done} pending={len(pending)}", flush=True)
    if not pending:
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = []
    for row in pending:
        messages = [
            {"role": "system", "content": "You are a math expert."},
            {"role": "user", "content": USER.format(question=row["question"])},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    params = SamplingParams(
        n=N_SAMPLES,
        temperature=0.7,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>"],
        seed=42 + shard,
    )
    chunk = 64
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), chunk):
            batch_rows = pending[start : start + chunk]
            batch_prompts = prompts[start : start + chunk]
            outputs = llm.generate(batch_prompts, params)
            for row, output in zip(batch_rows, outputs):
                completions = [item.text for item in output.outputs]
                handle.write(
                    json.dumps(
                        {
                            "question": row["question"],
                            "gold": row["golden_res"],
                            "completions": completions,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            print(f"shard {shard} {done + start + len(batch_rows)}/{len(mine)}", flush=True)


def _records(path: Path):
    # json.dumps leaves U+2028 in strings, and splitlines() treats it as a break.
    text = path.read_text(encoding="utf-8").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    for line in text.split("\n"):
        if line.strip():
            yield json.loads(line)


def filter_correct() -> None:
    kept = []
    n_completions = 0
    n_correct = 0
    n_questions = 0
    n_questions_hit = 0
    for path in sorted(OUT_DIR.glob("samples_shard*.jsonl")):
        for row in _records(path):
            n_questions += 1
            hit = False
            for text in row["completions"]:
                n_completions += 1
                if not isinstance(text, str) or not text.strip():
                    continue
                if grade_one(text, row["gold"]):
                    n_correct += 1
                    hit = True
                    kept.append({"question": row["question"], "golden_res": text.strip()})
            if hit:
                n_questions_hit += 1
    out_path = OUT_DIR / "train_correct.json"
    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "n_questions": n_questions,
        "n_questions_with_correct": n_questions_hit,
        "n_completions": n_completions,
        "n_correct": n_correct,
        "kept": len(kept),
    }
    (OUT_DIR / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"filter questions={n_questions} hit={n_questions_hit} "
        f"correct={n_correct}/{n_completions} kept={len(kept)}",
        flush=True,
    )


def main() -> None:
    _offline()
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--filter", action="store_true")
    args = parser.parse_args()
    if args.filter:
        filter_correct()
    else:
        if args.shard is None:
            raise SystemExit("need --shard")
        sample_shard(args.shard, args.nshards)


if __name__ == "__main__":
    main()
