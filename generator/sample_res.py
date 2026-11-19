from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from eval.labeling import grade_text
from generator.prompt import build_prompt
from generator.utils import write_jsonl


def sample_candidates(
    model_path: str,
    problems: list[dict],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    chunk: int = 32,
    output: str | Path,
    seed: int = 42,
    tensor_parallel: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 6144,
) -> list[dict]:
    from vllm import LLM, SamplingParams

    # vLLM here only samples the *starting* policy: the offline pool, the ranker
    # training data and route A all come from this checkpoint.
    llm = LLM(
        model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    rows: list[dict] = []
    prompts = [build_prompt(item["question"]) for item in problems]

    # Resume: keep questions that already have every response.
    done: dict[int, dict] = {}
    if Path(output).is_file():
        for line in Path(output).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if len(row.get("responses") or []) == k:
                    done[int(row["qid"])] = row
        rows = [done[qid] for qid in sorted(done)]
        print(f"[sample] resume {len(rows)}/{len(prompts)} questions from {output}", flush=True)

    def qid_of(index: int) -> int:
        problem = problems[index]
        return int(problem["qid"]) if "qid" in problem else index

    for start in range(0, len(prompts), chunk):
        chunk_rows = [start + offset for offset in range(min(chunk, len(prompts) - start))]
        todo = [index for index in chunk_rows if qid_of(index) not in done]
        if not todo:
            continue
        chunk_prompts = [prompts[index] for index in todo]
        outputs = llm.generate(chunk_prompts, params, use_tqdm=False)
        for offset, result in enumerate(outputs):
            index = todo[offset]
            problem = problems[index]
            qid = qid_of(index)
            texts = [item.text.strip() for item in result.outputs]
            labels = [grade_text(text, str(problem["answer"])) for text in texts]
            done[qid] = {
                "qid": qid,
                "question": problem["question"],
                "answer": str(problem["answer"]),
                "prompt": prompts[index],
                "responses": texts,
                "labels": labels,
            }
        rows = [done[qid] for qid in sorted(done)]
        write_jsonl(output, rows)
        print(f"[sample] {len(rows)}/{len(prompts)} questions", flush=True)
    return rows


def main() -> None:
    import argparse

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
    problems = [{"question": str(row["question"]), "answer": str(row["answer"])} for row in rows]
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
    total = sum(len(row["responses"]) for row in out)
    correct = sum(sum(row["labels"]) for row in out)
    rate = correct / total if total else 0.0
    print(f"[stage1] saved {args.output} responses={total} correct={correct} ({rate:.1%})", flush=True)


if __name__ == "__main__":
    main()
