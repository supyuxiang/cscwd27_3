from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

from generator.prompt import build_prompt





def load_data(data_path: str | Path):
    pass



def sampling(
    model: LLM,
    sp: SamplingParams,
    question_ls: list[str]
):
    pass
    

def save_sampled(sampled):
    pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument('--limit',type=int, default=-1)
    args = parser.parse_args()

    question_ls = load_data(args.data_path)
    if args.limit > 0:
        question_ls = question_ls[: args.limit]

    llm = LLM(args.model_path, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    sampled = sampling(
        llm,
        sp,
        question_ls,
    )
    save_sampled(sampled)

if __name__ == "__main__":
    main()
