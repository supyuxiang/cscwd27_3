from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

from generator.prompt import build_prompt


def load_data(data_path: str | Path) -> list[str]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(data_path)
    question_ls = []
    answer_ls = []
    for item in data:
        question_ls.append(item['question'])
        answer_ls.append(item['answer'])
    return question_ls, answer_ls


def sampling(
    llm: LLM,
    sp: SamplingParams,
    question_ls: list[str]
):
    prompt_ls = [build_prompt(q) for q in question_ls]

    outputs = llm.generate(
        prompt_ls,
        sp
    )
    # output[i].outputs[j].text

    res_ls = [
        [group[j].text for j in range(len(group))]
        for group in outputs
    ]

    out = []
    for q, group in zip(question_ls, res_ls):
        for res in group:
            out.append(
                {
                    'question': q,
                    'res': res
                }
            )
    return out

def save_sampled(sampled, save_path):
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='/data3/yxfeng/models/Qwen2.5/Qwen2.5-Math-1.5B')
    parser.add_argument("--data_path", default='/data3/yxfeng/datasets/math500/math500.json')
    parser.add_argument('--save_path', default='/data3/yxfeng/cscwd27_ranker/eval/Qwen2.5-Math-1.5B/math500')
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument('--limit',type=int, default=-1)
    args = parser.parse_args()

    question_ls, _ = load_data(args.data_path)
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
    save_sampled(sampled, save_path=args.save_path)

if __name__ == "__main__":
    main()
