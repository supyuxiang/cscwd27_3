"""Stage 2: replay prompt+candidate through the policy and cache probe-layer features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iraker.core import read_jsonl  # noqa: E402
from iraker.features import decoder_layers, extract_features  # noqa: E402
from iraker.policy import load_policy, load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    if not rows:
        raise SystemExit(f"no candidates in {args.candidates}")

    tokenizer = load_tokenizer(args.model)
    model = load_policy(args.model, lora_r=0, dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    n_layers = len(decoder_layers(model))
    layer = min(n_layers - 1, max(0, int(round(args.depth * (n_layers - 1)))))
    print(f"[stage2] layers={n_layers} probe_layer={layer} ({args.depth:.0%} depth)", flush=True)

    inst_all, resp_all = [], []
    done = 0
    for row in rows:
        prompt = row["prompt"]
        responses = row["responses"]
        inst, resp = extract_features(
            model,
            tokenizer,
            [prompt] * len(responses),
            responses,
            layer_index=layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        inst_all.append(inst[0])
        resp_all.append(resp)
        done += len(responses)
        if done % 200 == 0:
            print(f"[stage2] {done} responses", flush=True)

    payload = {
        "groups": [
            {
                "qid": int(row["qid"]),
                "prompt": row["prompt"],
                "response": resp,
                "instruction": inst,
                "labels": torch.tensor(row["labels"], dtype=torch.float32),
                "responses": row["responses"],
                "has_pos": bool(any(row["labels"])),
                "has_neg": bool(any(1 - label for label in row["labels"])),
            }
            for row, inst, resp in zip(rows, inst_all, resp_all)
        ],
        "probe_layer": layer,
        "depth": args.depth,
        "d_in": int(inst_all[0].numel()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"[stage2] saved {out} groups={len(payload['groups'])} d_in={payload['d_in']}", flush=True)


if __name__ == "__main__":
    main()
