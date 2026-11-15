"""Sample K candidates per question with vLLM and grade them.

vLLM writes text only. Feature extraction happens later in features.py because
the probe layer must be read from the same policy checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from .core import build_prompt, grade_text, write_jsonl  # noqa: E402


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

    for start in range(0, len(prompts), chunk):
        chunk_rows = [start + offset for offset in range(min(chunk, len(prompts) - start))]
        todo = [index for index in chunk_rows if index not in done]
        if not todo:
            continue
        chunk_prompts = [prompts[index] for index in todo]
        outputs = llm.generate(chunk_prompts, params, use_tqdm=False)
        for offset, result in enumerate(outputs):
            index = todo[offset]
            problem = problems[index]
            texts = [item.text.strip() for item in result.outputs]
            labels = [grade_text(text, str(problem["answer"])) for text in texts]
            done[index] = {
                "qid": index,
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
