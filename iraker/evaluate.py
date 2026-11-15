"""math_verify-based evaluation on math500 / aime24 / aime25 / aime26 / amc23.

All generation goes through vLLM (never HF ``generate``) so eval is fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path("/data3/yxfeng/datasets")
DEFAULT_SETS = ("math500", "aime24", "aime25", "aime26", "amc23")
STOP_WORDS = ("</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<|EOT|>")


def clean(text: str) -> str:
    text = (text or "").rstrip()
    for word in STOP_WORDS:
        if word in text:
            text = text.split(word)[0].strip()
    return text


def _grader():
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    extract = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())

    def grade(pred: str, gold: str) -> bool:
        gold_parsed = parse(gold, extract)
        pred_parsed = parse(pred or "", extract)
        if not gold_parsed or not pred_parsed:
            return False
        try:
            return bool(verify(gold_parsed, pred_parsed))
        except Exception:
            return False

    return grade


def load_set(name: str, lang: str = "en", limit: int = 0) -> list[dict]:
    path = DATA_ROOT / name / f"{name}_{lang}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows[:limit] if limit > 0 else rows


def _vllm_sample(
    model_path: str,
    questions: list[str],
    *,
    n_sampling: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gpu: int | None,
    seed: int = 1234,
) -> list[list[str]]:
    """Sample n completions per question on one GPU. Returns [per question][n] texts."""
    workdir = Path(tempfile.mkdtemp(prefix="iraker-eval-"))
    questions_path = workdir / "questions.json"
    output_path = workdir / "groups.json"
    questions_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if gpu is not None:
        visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            entries = [item for item in visible.split(",") if item.strip()]
            env["CUDA_VISIBLE_DEVICES"] = entries[gpu] if gpu < len(entries) else str(gpu)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "vllm_eval_worker.py"),
        "--model", model_path,
        "--questions", str(questions_path),
        "--output", str(output_path),
        "--n-sampling", str(n_sampling),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-new-tokens", str(max_new_tokens),
        "--seed", str(seed),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"vLLM eval worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout[-3000:]}\nstderr:\n{completed.stderr[-3000:]}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def evaluate_model(
    model_path: str,
    *,
    datasets=DEFAULT_SETS,
    n_sampling: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 3072,
    limit: int = 0,
    gpu: int | None = 0,
    log=print,
) -> dict:
    """avg@n accuracy with vLLM sampling. Prompts are raw strings, matching SFT/ranker."""
    grade = _grader()
    results: dict = {}
    for name in datasets:
        rows = load_set(name, limit=limit)
        questions = [row["question"] for row in rows]
        golds = [str(row["answer"]) for row in rows]
        groups = _vllm_sample(
            model_path,
            questions,
            n_sampling=n_sampling,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            gpu=gpu,
        )
        score_matrix = np.zeros((len(rows), n_sampling))
        for index in range(len(rows)):
            for draw in range(n_sampling):
                text = clean(groups[index][draw] if index < len(groups) and draw < len(groups[index]) else "")
                score_matrix[index, draw] = float(grade(text, golds[index]))
        per_draw = score_matrix.mean(axis=0) * 100
        results[name] = {
            "n": len(rows),
            "n_sampling": n_sampling,
            "acc": float(per_draw[0]),
            "all_acc": [float(x) for x in per_draw],
            "mean_acc": float(score_matrix.mean() * 100),
        }
        log(f"[eval] {name:8s} n={len(rows)} avg@{n_sampling}={results[name]['mean_acc']:.2f}")
    results["macro_avg"] = float(np.mean([r["mean_acc"] for r in results.values() if isinstance(r, dict)]))
    log(f"[eval] macro avg@{n_sampling} = {results['macro_avg']:.2f}")
    return results
