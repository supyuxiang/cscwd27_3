"""Run the vLLM sampler as a subprocess on a chosen GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sample_with_worker(
    model_path: str,
    questions: list[str],
    *,
    group_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gpu: int | None,
) -> list[list[dict]]:
    """Return [{prompt, responses}] per question, generated on ``gpu`` when given."""
    workdir = Path(tempfile.mkdtemp(prefix="iraker-sampler-"))
    questions_path = workdir / "questions.json"
    output_path = workdir / "groups.json"
    questions_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

    env = dict(os.environ)
    if gpu is not None:
        # ``gpu`` is an index into the GPUs this job can see, so a job whose
        # CUDA_VISIBLE_DEVICES is remapped by Slurm still picks the right card.
        visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            entries = [item for item in visible.split(",") if item.strip()]
            env["CUDA_VISIBLE_DEVICES"] = entries[gpu] if gpu < len(entries) else str(gpu)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "vllm_worker.py"),
        "--model", model_path,
        "--questions", str(questions_path),
        "--output", str(output_path),
        "--group-size", str(group_size),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-new-tokens", str(max_new_tokens),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"vLLM worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    groups = json.loads(output_path.read_text(encoding="utf-8"))
    return [[{"prompt": item["prompt"], "response": text, "_score": 0.0} for text in item["responses"]]
            for item in groups]
