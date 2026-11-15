"""Stage 3: train the lightweight ranker on the cached features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iraker.ranker_train import train_ranker  # noqa: E402


def main() -> None:
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
