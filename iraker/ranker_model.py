"""Lightweight rankers over cached instruction/response features.

Pointwise: shared projection + MLP on (instruction, response), score by cosine.
Listwise: the same projection, one transformer block over [instruction, r1..rK],
then cosine between the transformed instruction and each transformed response.

Both are binary classifiers (correct / incorrect) and stay well under 1M
parameters, matching the Language Ranker design.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SCALE_INIT = 10.0  # cosine logits multiplied by this start near +-10


class PointwiseRanker(nn.Module):
    def __init__(self, d_in: int, d_proj: int = 128, d_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_proj),
            nn.Linear(d_proj, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_proj),
        )
        self.scale = nn.Parameter(torch.tensor(SCALE_INIT))

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.proj(features)), dim=-1)

    def forward(self, instruction: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """instruction (B, d); responses (B, K, d) -> logits (B, K)."""
        inst = self._encode(instruction)  # B, d_proj
        resp = self._encode(responses)  # B, K, d_proj
        cosine = (resp * inst.unsqueeze(1)).sum(-1)
        return self.scale * cosine


class _Block(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class ListwiseRanker(nn.Module):
    def __init__(self, d_in: int, d_proj: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_proj, bias=False)
        self.block = _Block(d_proj, n_heads, dropout)
        self.scale = nn.Parameter(torch.tensor(SCALE_INIT))

    def forward(self, instruction: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([instruction.unsqueeze(1), responses], dim=1)  # B, 1+K, d
        tokens = F.normalize(self.block(self.proj(tokens)), dim=-1)
        inst = tokens[:, 0]  # B, d_proj
        resp = tokens[:, 1:]
        return self.scale * (resp * inst.unsqueeze(1)).sum(-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def batch_loss(kind: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary labels: BCE for pointwise, KL-to-target-distribution for listwise."""
    if kind == "pointwise":
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    target = labels.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    log_prob = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_prob, target, reduction="batchmean")


@torch.no_grad()
def pairwise_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of (correct, incorrect) pairs inside a group that are ordered right."""
    hits = total = 0
    for scores, bits in zip(logits.tolist(), labels.tolist()):
        pos = [s for s, y in zip(scores, bits) if y >= 0.5]
        neg = [s for s, y in zip(scores, bits) if y < 0.5]
        for p in pos:
            for n in neg:
                total += 1
                hits += int(p > n)
    return hits / total if total else 0.0


def save_ranker(
    path: str | Path,
    kind: str,
    model: nn.Module,
    d_in: int,
    d_proj: int,
    probe_layer: int,
    depth: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "kind": kind,
        "d_in": d_in,
        "d_proj": d_proj,
        "probe_layer": probe_layer,
        "depth": depth,
        "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": extra or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_ranker(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload["kind"]
    cls = PointwiseRanker if kind == "pointwise" else ListwiseRanker
    model = cls(d_in=int(payload["d_in"]), d_proj=int(payload["d_proj"]))
    model.load_state_dict(payload["state"])
    model.to(device).eval()
    return model, payload


def ranker_summary(path: str | Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return json.dumps({key: payload[key] for key in ("kind", "d_in", "d_proj", "probe_layer", "depth")})
