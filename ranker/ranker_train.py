"""Fit the lightweight ranker on cached features."""

from __future__ import annotations

import random

import torch

from .ranker_data import pad_groups, split_groups
from .ranker_structure import (
    ListwiseRanker,
    PointwiseRanker,
    batch_loss,
    count_parameters,
    pairwise_accuracy,
    save_ranker,
)


def train_ranker(
    groups: list[dict],
    *,
    kind: str = "listwise",
    d_proj: int = 128,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    output: str | None = None,
    probe_layer: int = 0,
    depth: float = 0.6,
    log=print,
):
    torch.manual_seed(seed)
    random.seed(seed)
    train, val = split_groups(groups, val_ratio, seed)
    if not train:
        raise SystemExit("no groups contain both a correct and an incorrect response")
    d_in = int(train[0]["instruction"].numel())
    model = (PointwiseRanker if kind == "pointwise" else ListwiseRanker)(d_in=d_in, d_proj=d_proj).to(device)
    log(f"[ranker] kind={kind} params={count_parameters(model)} d_in={d_in} train={len(train)} val={len(val)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        random.shuffle(train)
        model.train()
        total = 0.0
        for start in range(0, len(train), batch_size):
            chunk = train[start:start + batch_size]
            if len(chunk) < 2:
                continue
            inst, resp, labels = pad_groups(chunk, device)
            logits = model(inst, resp)
            loss = batch_loss(kind, logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        message = f"[ranker] epoch {epoch}/{epochs} loss={total / max(1, len(train) - 1):.4f}"
        if val:
            model.eval()
            with torch.no_grad():
                inst, resp, labels = pad_groups(val, device)
                logits = model(inst, resp)
                accuracy = pairwise_accuracy(logits.cpu(), labels.cpu())
            message += f" val_pair_acc={accuracy:.4f}"
            if accuracy > best:
                best = accuracy
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(message)

    if best_state is not None:
        model.load_state_dict(best_state)
    if output:
        save_ranker(
            output,
            kind,
            model,
            d_in=d_in,
            d_proj=d_proj,
            probe_layer=probe_layer,
            depth=depth,
            extra={"val_pair_acc": best, "epochs": epochs},
        )
        log(f"[ranker] saved {output} (best val_pair_acc={best:.4f})")
    return model, {"val_pair_acc": best}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", default="listwise", choices=("listwise", "pointwise"))
    parser.add_argument("--d-proj", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    groups = payload["groups"]
    print(f"[stage3] groups={len(groups)} d_in={payload['d_in']} probe_layer={payload['probe_layer']}", flush=True)
    _, metrics = train_ranker(
        groups,
        kind=args.kind,
        d_proj=args.d_proj,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=torch.device("cpu"),
        output=args.output,
        probe_layer=int(payload["probe_layer"]),
        depth=float(payload["depth"]),
    )
    print(f"[stage3] done {args.output} {metrics}", flush=True)


if __name__ == "__main__":
    main()
