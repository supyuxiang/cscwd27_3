"""Bradley--Terry ranker on cached candidate features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Ranker(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 0):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class Standardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, features: torch.Tensor) -> "Standardizer":
        return cls(features.mean(0), features.std(0).clamp(min=1e-3))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean.to(features.device)) / self.std.to(features.device)


def fit_ranker(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    input_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    prompt_ids: list[int] | None = None,
) -> tuple[Ranker, Standardizer]:
    if not pairs:
        raise ValueError("Ranker training needs at least one correct-incorrect pair.")
    pos = torch.stack([item[0] for item in pairs])
    neg = torch.stack([item[1] for item in pairs])
    stats = Standardizer.fit(torch.cat([pos, neg], dim=0))
    pos = stats(pos).to(device)
    neg = stats(neg).to(device)
    owners = torch.tensor(prompt_ids if prompt_ids is not None else list(range(len(pairs))), device=device)
    model = Ranker(input_dim, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        pair_loss = -F.logsigmoid(model(pos) - model(neg))
        loss = torch.stack([pair_loss[owners == owner].mean() for owner in owners.unique()]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, stats


@torch.no_grad()
def score_features(model: Ranker, stats: Standardizer, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    return model(stats(features).to(device)).float().cpu()
