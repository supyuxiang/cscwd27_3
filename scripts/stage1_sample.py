"""Stage 1: vLLM samples K candidates per question and grades them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.core import write_jsonl  # noqa: E402
from ranker.sampling import sample_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]
    problems = [{"question": str(r["question"]), "answer": str(r["answer"])} for r in rows]
    print(f"[stage1] questions={len(problems)} k={args.k}", flush=True)

    out = sample_candidates(
        args.model,
        problems,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk=args.chunk,
        output=args.output,
        seed=args.seed,
        tensor_parallel=args.tensor_parallel,
    )
    total = sum(len(r["responses"]) for r in out)
    correct = sum(sum(r["labels"]) for r in out)
    print(f"[stage1] saved {args.output} responses={total} correct={correct} ({correct / total:.1%})", flush=True)


if __name__ == "__main__":
    main()
