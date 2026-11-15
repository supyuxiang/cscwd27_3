"""Build ranker training groups from sampled candidates + cached features."""

from __future__ import annotations

import random

import torch


def load_feature_file(path: str) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["groups"]


def build_groups(rows: list[dict], inst: torch.Tensor, resp: torch.Tensor, keep_k: int, seed: int) -> list[dict]:
    """Slice the flat (inst, resp) tensors back into per-question groups.

    ``rows`` is the sampling output; responses are stored question-major so the
    flat index is a running offset.
    """
    groups: list[dict] = []
    offset = 0
    rng = random.Random(seed)
    for row in rows:
        count = len(row["responses"])
        group_inst = inst[offset]
        group_resp = resp[offset:offset + count]
        labels = torch.tensor(row["labels"], dtype=torch.float32)
        offset += count
        if count > keep_k:
            order = rng.sample(range(count), keep_k)
            order.sort()
            group_resp = group_resp[order]
            labels = labels[order]
        groups.append(
            {
                "qid": int(row["qid"]),
                "prompt": row["prompt"],
                "responses": [row["responses"][i] for i in range(len(row["responses"]))] if count <= keep_k else None,
                "instruction": group_inst,
                "response": group_resp,
                "labels": labels,
                "has_pos": bool((labels > 0.5).any()),
                "has_neg": bool((labels < 0.5).any()),
            }
        )
    return groups


def split_groups(groups: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    informative = [g for g in groups if g["has_pos"] and g["has_neg"]]
    rng = random.Random(seed)
    rng.shuffle(informative)
    n_val = max(1, int(round(len(informative) * val_ratio))) if len(informative) > 1 else 0
    return informative[n_val:], informative[:n_val]


def pad_groups(groups: list[dict], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack groups into (B, d), (B, K, d), (B, K) with K = max group size."""
    k = max(int(g["response"].size(0)) for g in groups)
    d = int(groups[0]["instruction"].numel())
    inst = torch.zeros((len(groups), d))
    resp = torch.zeros((len(groups), k, d))
    labels = torch.zeros((len(groups), k))
    for row, group in enumerate(groups):
        n = int(group["response"].size(0))
        inst[row] = group["instruction"]
        resp[row, :n] = group["response"]
        labels[row, :n] = group["labels"]
    return inst.to(device), resp.to(device), labels.to(device)
