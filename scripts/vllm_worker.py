"""Standalone vLLM sampler used as a subprocess by the GRPO scripts.

Running vLLM in its own process on a dedicated GPU avoids fighting the policy
model for memory on the trainer's GPU. The worker is told which GPU to use via
CUDA_VISIBLE_DEVICES, reads a json list of questions, and writes the sampled
groups as json.
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
    parser.add_argument("--questions", required=True, help="json list of question strings")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
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
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    groups = []
    for prompt, result in zip(prompts, outputs):
        groups.append({"prompt": prompt, "responses": [item.text.strip() for item in result.outputs]})
    Path(args.output).write_text(json.dumps(groups, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[vllm-worker] wrote {args.output} groups={len(groups)}", flush=True)


if __name__ == "__main__":
    main()
