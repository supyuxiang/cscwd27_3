"""vLLM-only evaluation sampler: n draws per prompt on one GPU.

Reads a json list of question strings, samples n responses each with vLLM,
and writes the completions as json. Grading happens in the caller so this
worker never imports HF generate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iraker.core import build_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-sampling", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stop-token-ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    prompts = [build_prompt(question) for question in questions]
    llm = LLM(
        args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(
        n=args.n_sampling,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=args.stop_token_ids,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    groups = [[item.text.strip() for item in result.outputs] for result in outputs]
    if not groups or not all(len(g) >= args.n_sampling for g in groups):
        raise RuntimeError(
            f"vLLM returned incomplete output: {len(groups)} prompts, "
            f"expected {args.n_sampling} each"
        )
    nonempty = sum(1 for g in groups for t in g if t.strip())
    print(f"[vllm-eval] sampled {len(groups)} prompts x {args.n_sampling}, nonempty={nonempty}", flush=True)
    print(f"[vllm-eval] sample[0][0]={groups[0][0][:200]!r}", flush=True)
    Path(args.output).write_text(json.dumps(groups, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[vllm-eval] wrote {args.output} prompts={len(groups)} n={args.n_sampling}", flush=True)


if __name__ == "__main__":
    main()
