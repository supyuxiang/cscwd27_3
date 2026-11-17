"""Rejection-sampling SFT: sample many, keep what the ranker likes, then SFT."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402

from .core import build_prompt, grade_text, read_jsonl, write_jsonl  # noqa: E402
from .ranker_data import load_feature_file, pad_groups  # noqa: E402
from .ranker_model import load_ranker  # noqa: E402


def select_by_ranker(ranker_path: str, groups: list[dict], keep: int = 1) -> list[dict]:
    model, _ = load_ranker(ranker_path, device="cpu")
    chosen = []
    for group in groups:
        batch = [group]
        inst, resp, labels = pad_groups(batch, torch.device("cpu"))
        with torch.no_grad():
            scores = model(inst, resp)[0][: group["response"].size(0)]
        top = torch.topk(scores, k=min(keep, scores.numel())).indices.tolist()
        for index in top:
            chosen.append(
                {
                    "qid": group["qid"],
                    "prompt": group["prompt"],
                    "response": group["responses"][index],
                    "label": int(group["labels"][index].item()),
                    "score": float(scores[index]),
                }
            )
    return chosen


def sample_for_rejection(
    model_path: str,
    problems: list[dict],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    chunk: int,
    output: str,
    tensor_parallel: int = 1,
) -> list[dict]:
    """Sample without grading: the ranker decides what is kept."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=6144,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(n=k, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    prompts = [build_prompt(item["question"]) for item in problems]
    rows: list[dict] = []
    for start in range(0, len(prompts), chunk):
        batch = prompts[start:start + chunk]
        outputs = llm.generate(batch, params, use_tqdm=False)
        for offset, output_item in enumerate(outputs):
            problem = problems[start + offset]
            rows.append(
                {
                    "qid": start + offset,
                    "question": problem["question"],
                    "answer": str(problem["answer"]),
                    "prompt": batch[offset],
                    "responses": [item.text.strip() for item in output_item.outputs],
                }
            )
        write_jsonl(output, rows)
        print(f"[reject-sample] {min(start + chunk, len(prompts))}/{len(prompts)}", flush=True)
    return rows
