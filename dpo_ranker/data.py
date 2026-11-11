"""One shared prompt pool, split once into labeled and hidden-answer subsets."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Problem:
    qid: int
    question: str
    answer: str
    labeled: bool


def load_problems(path: str | Path) -> list[Problem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for index, row in enumerate(raw):
        problems.append(
            Problem(
                qid=index,
                question=str(row["question"]),
                answer=str(row.get("answer") or row.get("solution") or ""),
                labeled=True,
            )
        )
    return problems


def split_pool(problems: list[Problem], n_prompts: int, n_labeled: int, seed: int) -> list[Problem]:
    if n_labeled > n_prompts:
        raise ValueError(f"n_labeled={n_labeled} exceeds n_prompts={n_prompts}")
    rng = random.Random(seed)
    chosen = list(problems)
    rng.shuffle(chosen)
    chosen = chosen[:n_prompts]
    for offset, problem in enumerate(chosen):
        problem.labeled = offset < n_labeled
    return chosen


def save_split(problems: list[Problem], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(problem) for problem in problems], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_split(path: str | Path) -> list[Problem]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Problem(**row) for row in rows]
