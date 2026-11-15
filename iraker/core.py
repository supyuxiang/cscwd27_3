"""Shared helpers: prompts, grading, distributed setup, io."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

# Exactly the English prompt used by exp1_math eval (prompt_type=language).
SYSTEM_EXPERT = "You are a math expert.\nYou must answer in English."
USER_TEMPLATE = (
    "Question: {question} Please reason step by step and put your final answer "
    "within \\boxed{{}}."
)


def build_prompt(question: str) -> str:
    """Base-model prompt: plain text, no chat template."""
    return f"{SYSTEM_EXPERT}\n\n{USER_TEMPLATE.format(question=question)}"


# --------------------------------------------------------------------------- io
def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.is_file():
        return rows
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(p)


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


# ------------------------------------------------------------------- grading
def _boxed(text: str) -> str | None:
    key = r"\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    index = start + len(key)
    depth = 1
    chars: list[str] = []
    while index < len(text) and depth:
        char = text[index]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth:
                chars.append(char)
        else:
            chars.append(char)
        index += 1
    return "".join(chars).strip() if depth == 0 else None


def _normalize(value: str) -> str:
    import re

    text = value.strip().replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace(",", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def grade_text(response: str, gold: str) -> int:
    """1 when the extracted \\boxed{} answer matches the reference."""
    pred = _boxed(response)
    if pred is None:
        return 0
    left, right = _normalize(pred), _normalize(gold)
    if left == right:
        return 1
    try:
        return int(abs(float(left) - float(right)) < 1e-6)
    except ValueError:
        return 0


# ------------------------------------------------------------------ distributed
def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        dist.init_process_group(backend="nccl")
        return rank, world, local
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(world: int) -> None:
    if world > 1:
        import torch.distributed as dist

        dist.barrier()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Log:
    """Rank-0 logger."""

    def __init__(self, rank: int):
        self.rank = rank

    def __call__(self, *args) -> None:
        if self.rank == 0:
            print(*args, flush=True)


def probe_layer_index(depth: float, n_layers: int) -> int:
    """The paper finds ~60% depth best; layers are 1-indexed in logs, 0-based here."""
    index = int(round(depth * (n_layers - 1)))
    return min(n_layers - 1, max(0, index))
