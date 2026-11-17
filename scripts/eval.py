"""Stage 6: evaluate any checkpoint on math500 / aime24 / aime25 / aime26 / amc23."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.core import write_json  # noqa: E402
from ranker.evaluate import DEFAULT_SETS, evaluate_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model or a saved checkpoint dir")
    parser.add_argument("--output", default="")
    parser.add_argument("--sets", default=",".join(DEFAULT_SETS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-sampling", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    results = evaluate_model(
        args.model,
        datasets=tuple(x.strip() for x in args.sets.split(",") if x.strip()),
        n_sampling=args.n_sampling,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        gpu=args.gpu,
        top_p=args.top_p,
    )
    if args.output:
        write_json(args.output, results)
        print(f"[eval] wrote {args.output}")
    else:
        print(results)


if __name__ == "__main__":
    main()
