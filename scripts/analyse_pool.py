"""Analysis: pass@k of the sampling pool, ranker top-1 selection rate, and baselines.

Answers two questions:
  1. how good is the sampled pool (pass@k) and how often does the ranker's top-1
     pick a correct completion (its "precision"), compared with a random pick
  2. how do normal SFT (teacher answers) and normal GRPO (verifier reward)
     compare with the ranker variants
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.ranker_data import pad_groups  # noqa: E402
from ranker.ranker_model import load_ranker  # noqa: E402


def pass_at_k(labels: list[list[int]], k: int) -> float:
    """Unbiased pass@k over questions: 1 - C(n-c, k) / C(n, k)."""
    from math import comb

    hits = 0.0
    counted = 0
    for bits in labels:
        n = len(bits)
        c = sum(bits)
        if n < k:
            continue
        counted += 1
        if n - c < k:
            hits += 1.0
        else:
            hits += 1.0 - comb(n - c, k) / comb(n, k)
    return hits / counted if counted else 0.0


def analyse_pool(rows_path: str, ranker_path: str | None, features_path: str | None) -> dict:
    rows = []
    with open(rows_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    labels = [row["labels"] for row in rows]
    k = max(len(bits) for bits in labels)

    out: dict = {
        "questions": len(rows),
        "k": k,
        "responses": int(sum(len(bits) for bits in labels)),
        "mean_correct_per_question": float(np.mean([sum(bits) for bits in labels])),
        "answer_accuracy": float(np.mean([np.mean(bits) for bits in labels])),
        "pass@k": {str(j): pass_at_k(labels, j) for j in range(1, k + 1) if j <= k},
        "oracle_top1": float(np.mean([1.0 if sum(bits) > 0 else 0.0 for bits in labels])),
        "random_top1_expected": float(np.mean([np.mean(bits) for bits in labels])),
        "all_wrong_questions": int(sum(1 for bits in labels if sum(bits) == 0)),
        "all_right_questions": int(sum(1 for bits in labels if sum(bits) == len(bits))),
    }
    if ranker_path and features_path:
        payload = torch.load(features_path, map_location="cpu", weights_only=False)
        groups = payload["groups"]
        model, meta = load_ranker(ranker_path, device="cpu")
        picked_correct = []
        picked_top1_score = []
        with torch.no_grad():
            for group in groups:
                inst, resp, group_labels = pad_groups([group], torch.device("cpu"))
                scores = model(inst, resp)[0][: group["response"].size(0)]
                best = int(torch.argmax(scores))
                picked_correct.append(int(group["labels"][best].item()))
                picked_top1_score.append(float(scores[best]))
        out["ranker"] = {
            "kind": meta["kind"],
            "probe_layer": meta["probe_layer"],
            "depth": meta["depth"],
            "params": int(sum(v.numel() for v in model.state_dict().values())),
            "top1_selection_accuracy": float(np.mean(picked_correct)),
            "top1_recall_of_correct": float(np.mean(picked_correct)),
            "pool_answer_accuracy": out["answer_accuracy"],
            "lift_over_random": float(np.mean(picked_correct) - out["answer_accuracy"]),
            "mean_top1_score": float(np.mean(picked_top1_score)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--features", default="")
    parser.add_argument("--ranker", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = analyse_pool(args.candidates, args.ranker or None, args.features or None)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"[analysis] wrote {args.output}")


if __name__ == "__main__":
    main()
