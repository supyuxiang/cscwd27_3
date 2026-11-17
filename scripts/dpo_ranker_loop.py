"""One phase of an iterative ranker-DPO loop.

iter i:
  1. sample K trajectories from the current policy and grade them 1/0
  2. train a listwise ranker on those labels
  3. sample a fresh set of trajectories and score them with that ranker
  4. for each question, DPO on (highest score, lowest score)
  5. the caller evaluates the merged policy, then the next iter starts from it

Each subcommand is a fresh process so vLLM teardown cannot leak into training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.core import read_jsonl, write_json, write_jsonl  # noqa: E402
from ranker.features import decoder_layers, extract_features  # noqa: E402
from ranker.policy import load_policy, load_tokenizer  # noqa: E402
from ranker.ranker_model import load_ranker  # noqa: E402
from ranker.ranker_train import train_ranker  # noqa: E402


def _train_rows(path: str, limit: int) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    return rows


def cmd_sample(args: argparse.Namespace) -> None:
    from ranker.sampling import sample_candidates

    rows = _train_rows(args.train_json, args.limit)
    picked = [i for i in range(len(rows)) if i % args.shards == args.shard]
    problems = [
        {"qid": i, "question": str(rows[i]["question"]), "answer": str(rows[i]["answer"])}
        for i in picked
    ]
    print(
        f"[sample] shard {args.shard}/{args.shards} questions={len(problems)} "
        f"k={args.k} seed={args.seed}",
        flush=True,
    )
    sample_candidates(
        args.model,
        problems,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk=args.chunk,
        output=args.output,
        seed=args.seed,
        tensor_parallel=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # vLLM's worker teardown can sit forever after the jsonl is already complete.
    os._exit(0)


def cmd_merge_samples(args: argparse.Namespace) -> None:
    rows = []
    seen: set[int] = set()
    for path in args.inputs:
        for row in read_jsonl(path):
            qid = int(row["qid"])
            if qid in seen:
                raise SystemExit(f"duplicate qid {qid} while merging samples")
            if len(row.get("responses") or []) != args.k:
                raise SystemExit(f"qid {qid} has {len(row.get('responses') or [])} responses, want {args.k}")
            seen.add(qid)
            rows.append(row)
    rows.sort(key=lambda row: int(row["qid"]))
    if args.expect > 0 and len(rows) != args.expect:
        raise SystemExit(f"merged {len(rows)} questions, expected {args.expect}")
    write_jsonl(args.output, rows)
    correct = sum(sum(row["labels"]) for row in rows)
    total = sum(len(row["labels"]) for row in rows)
    print(f"[merge-samples] {args.output} questions={len(rows)} correct={correct}/{total}", flush=True)


def _extract(args: argparse.Namespace, rows: list[dict]) -> dict:
    tokenizer = load_tokenizer(args.model)
    model = load_policy(args.model, lora_r=0, dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    n_layers = len(decoder_layers(model))
    layer = min(n_layers - 1, max(0, int(round(args.depth * (n_layers - 1)))))
    print(f"[features] layers={n_layers} probe_layer={layer} rows={len(rows)}", flush=True)
    batch_size = args.batch_size
    inst_all, resp_all = [], []
    done = 0
    for row in rows:
        responses = row["responses"]
        while True:
            try:
                inst, resp = extract_features(
                    model,
                    tokenizer,
                    [row["prompt"]] * len(responses),
                    responses,
                    layer_index=layer,
                    batch_size=batch_size,
                    max_length=args.max_length,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(f"[features] oom, retry batch_size={batch_size}", flush=True)
        inst_all.append(inst[0])
        resp_all.append(resp)
        done += len(responses)
        if done % 400 == 0 or done == len(rows) * len(rows[0]["responses"]):
            print(f"[features] {done} responses", flush=True)
    return {
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
        "d_in": int(inst_all[0].numel()) if inst_all else 0,
    }


def cmd_features(args: argparse.Namespace) -> None:
    rows = [row for row in read_jsonl(args.candidates) if int(row["qid"]) % args.shards == args.shard]
    if not rows:
        raise SystemExit(f"shard {args.shard} has no rows")
    payload = _extract(args, rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"[features] saved {out} groups={len(payload['groups'])}", flush=True)


def cmd_merge_features(args: argparse.Namespace) -> None:
    groups = []
    meta = None
    for path in args.inputs:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if meta is None:
            meta = payload
        elif payload["probe_layer"] != meta["probe_layer"] or payload["d_in"] != meta["d_in"]:
            raise SystemExit(f"feature meta mismatch in {path}")
        groups.extend(payload["groups"])
    groups.sort(key=lambda group: int(group["qid"]))
    qids = [int(group["qid"]) for group in groups]
    if len(qids) != len(set(qids)):
        raise SystemExit("duplicate qid in feature shards")
    if args.expect > 0 and len(groups) != args.expect:
        raise SystemExit(f"merged {len(groups)} groups, expected {args.expect}")
    out = {
        "groups": groups,
        "probe_layer": meta["probe_layer"],
        "depth": meta["depth"],
        "d_in": meta["d_in"],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(out, tmp)
    tmp.replace(path)
    print(f"[merge-features] {path} groups={len(groups)} d_in={out['d_in']} layer={out['probe_layer']}", flush=True)


def cmd_ranker(args: argparse.Namespace) -> None:
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    _, metrics = train_ranker(
        payload["groups"],
        kind="listwise",
        d_proj=args.d_proj,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=16,
        val_ratio=0.1,
        seed=args.seed,
        device=torch.device("cpu"),
        output=args.output,
        probe_layer=int(payload["probe_layer"]),
        depth=float(payload["depth"]),
    )
    print(f"[ranker] {args.output} {metrics}", flush=True)


def cmd_pairs(args: argparse.Namespace) -> None:
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    ranker, meta = load_ranker(args.ranker, device=torch.device("cpu"))
    if int(meta["d_in"]) != int(payload["d_in"]) or int(meta["probe_layer"]) != int(payload["probe_layer"]):
        raise SystemExit(
            f"ranker d_in/layer {meta['d_in']}/{meta['probe_layer']} "
            f"!= features {payload['d_in']}/{payload['probe_layer']}"
        )
    pairs = []
    skip = {"tie": 0, "same_text": 0, "empty": 0, "nan": 0}
    chosen_correct = rejected_correct = 0
    agree = disagree = both_right = both_wrong = 0
    gaps = []
    ranker.eval()
    with torch.no_grad():
        for group in payload["groups"]:
            responses = list(group["responses"])
            labels = [int(x) for x in group["labels"].tolist()]
            if len(responses) != int(group["response"].size(0)):
                raise SystemExit(f"qid {group['qid']} feature/response length mismatch")
            inst = group["instruction"].unsqueeze(0)
            resp = group["response"].unsqueeze(0)
            scores = ranker(inst, resp)[0].float()
            if not torch.isfinite(scores).all():
                skip["nan"] += 1
                continue
            high = int(torch.argmax(scores))
            low = int(torch.argmin(scores))
            if high == low or float(scores[high]) == float(scores[low]):
                skip["tie"] += 1
                continue
            chosen = responses[high].strip()
            rejected = responses[low].strip()
            if not chosen or not rejected:
                skip["empty"] += 1
                continue
            if chosen == rejected:
                skip["same_text"] += 1
                continue
            pairs.append(
                {
                    "qid": int(group["qid"]),
                    "prompt": group["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_score": float(scores[high]),
                    "rejected_score": float(scores[low]),
                    "chosen_label": labels[high],
                    "rejected_label": labels[low],
                    "score_gap": float(scores[high] - scores[low]),
                }
            )
            chosen_correct += labels[high]
            rejected_correct += labels[low]
            if labels[high] == 1 and labels[low] == 0:
                agree += 1
            elif labels[high] == 0 and labels[low] == 1:
                disagree += 1
            elif labels[high] == 1 and labels[low] == 1:
                both_right += 1
            else:
                both_wrong += 1
            gaps.append(float(scores[high] - scores[low]))
    if not pairs:
        raise SystemExit("ranker produced no preference pairs")
    write_jsonl(args.output, pairs)
    stats = {
        "questions": len(payload["groups"]),
        "pairs": len(pairs),
        "skip": skip,
        "chosen_correct_rate": chosen_correct / len(pairs),
        "rejected_correct_rate": rejected_correct / len(pairs),
        "ranker_agrees_with_label": agree,
        "ranker_disagrees_with_label": disagree,
        "both_correct": both_right,
        "both_incorrect": both_wrong,
        "mean_score_gap": sum(gaps) / len(gaps),
        "ranker_val_pair_acc": meta.get("config", {}).get("val_pair_acc"),
    }
    write_json(args.stats, stats)
    print(f"[pairs] {args.output} {json.dumps(stats)}", flush=True)


def cmd_random_pairs(args: argparse.Namespace) -> None:
    """One DPO pair per question: a uniformly random correct response and a random incorrect one."""
    import random

    rows = read_jsonl(args.candidates)
    rng = random.Random(args.seed)
    pairs = []
    skip = {"no_pos": 0, "no_neg": 0, "same_text": 0}
    chosen_correct = rejected_correct = 0
    for row in rows:
        responses = list(row["responses"])
        labels = [int(x) for x in row["labels"]]
        if len(responses) != len(labels):
            raise SystemExit(f"qid {row['qid']} response/label length mismatch")
        positive = [i for i, label in enumerate(labels) if label == 1 and responses[i].strip()]
        negative = [i for i, label in enumerate(labels) if label == 0 and responses[i].strip()]
        if not positive:
            skip["no_pos"] += 1
            continue
        if not negative:
            skip["no_neg"] += 1
            continue
        chosen_i = rejected_i = None
        for _ in range(16):
            high = rng.choice(positive)
            low = rng.choice(negative)
            if responses[high].strip() != responses[low].strip():
                chosen_i, rejected_i = high, low
                break
        if chosen_i is None:
            skip["same_text"] += 1
            continue
        pairs.append(
            {
                "qid": int(row["qid"]),
                "prompt": row["prompt"],
                "chosen": responses[chosen_i].strip(),
                "rejected": responses[rejected_i].strip(),
                "chosen_label": labels[chosen_i],
                "rejected_label": labels[rejected_i],
            }
        )
        chosen_correct += labels[chosen_i]
        rejected_correct += labels[rejected_i]
    if not pairs:
        raise SystemExit("random pairing produced no preference pairs")
    write_jsonl(args.output, pairs)
    stats = {
        "selection": "random_positive_negative",
        "seed": args.seed,
        "questions": len(rows),
        "pairs": len(pairs),
        "skip": skip,
        "chosen_correct_rate": chosen_correct / len(pairs),
        "rejected_correct_rate": rejected_correct / len(pairs),
    }
    write_json(args.stats, stats)
    print(f"[random-pairs] {args.output} {json.dumps(stats)}", flush=True)


def cmd_dpo(args: argparse.Namespace) -> None:
    from ranker.dpo import train_dpo
    from ranker.swan import SwanLog

    rows = read_jsonl(args.pairs)
    swan = SwanLog(
        "iraker",
        args.swan_name,
        config={
            "beta": args.beta,
            "lr": args.lr,
            "epochs": args.epochs,
            "pairs": len(rows),
            "model": args.model,
        },
    )
    metrics = train_dpo(
        args.model,
        rows,
        args.output_dir,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        seed=args.seed,
        swan=swan,
    )
    write_json(Path(args.output_dir) / "dpo_metrics.json", metrics)
    swan.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample = sub.add_parser("sample")
    sample.add_argument("--model", required=True)
    sample.add_argument("--train-json", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--shard", type=int, required=True)
    sample.add_argument("--shards", type=int, required=True)
    sample.add_argument("--limit", type=int, default=0)
    sample.add_argument("--k", type=int, default=8)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-p", type=float, default=0.95)
    sample.add_argument("--max-new-tokens", type=int, default=2048)
    sample.add_argument("--chunk", type=int, default=4)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    sample.set_defaults(func=cmd_sample)

    merge_s = sub.add_parser("merge-samples")
    merge_s.add_argument("--inputs", nargs="+", required=True)
    merge_s.add_argument("--output", required=True)
    merge_s.add_argument("--k", type=int, default=8)
    merge_s.add_argument("--expect", type=int, default=0)
    merge_s.set_defaults(func=cmd_merge_samples)

    feat = sub.add_parser("features")
    feat.add_argument("--model", required=True)
    feat.add_argument("--candidates", required=True)
    feat.add_argument("--output", required=True)
    feat.add_argument("--shard", type=int, default=0)
    feat.add_argument("--shards", type=int, default=1)
    feat.add_argument("--depth", type=float, default=0.6)
    feat.add_argument("--batch-size", type=int, default=2)
    feat.add_argument("--max-length", type=int, default=4096)
    feat.set_defaults(func=cmd_features)

    merge_f = sub.add_parser("merge-features")
    merge_f.add_argument("--inputs", nargs="+", required=True)
    merge_f.add_argument("--output", required=True)
    merge_f.add_argument("--expect", type=int, default=0)
    merge_f.set_defaults(func=cmd_merge_features)

    rank = sub.add_parser("ranker")
    rank.add_argument("--features", required=True)
    rank.add_argument("--output", required=True)
    rank.add_argument("--d-proj", type=int, default=128)
    rank.add_argument("--epochs", type=int, default=40)
    rank.add_argument("--lr", type=float, default=1e-3)
    rank.add_argument("--seed", type=int, default=42)
    rank.set_defaults(func=cmd_ranker)

    pairs = sub.add_parser("pairs")
    pairs.add_argument("--features", required=True)
    pairs.add_argument("--ranker", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--stats", required=True)
    pairs.set_defaults(func=cmd_pairs)

    random_pairs = sub.add_parser("random-pairs")
    random_pairs.add_argument("--candidates", required=True)
    random_pairs.add_argument("--output", required=True)
    random_pairs.add_argument("--stats", required=True)
    random_pairs.add_argument("--seed", type=int, default=42)
    random_pairs.set_defaults(func=cmd_random_pairs)

    dpo = sub.add_parser("dpo")
    dpo.add_argument("--model", required=True)
    dpo.add_argument("--pairs", required=True)
    dpo.add_argument("--output-dir", required=True)
    dpo.add_argument("--beta", type=float, default=0.1)
    dpo.add_argument("--lr", type=float, default=5e-6)
    dpo.add_argument("--epochs", type=int, default=1)
    dpo.add_argument("--grad-accum", type=int, default=16)
    dpo.add_argument("--max-length", type=int, default=4096)
    dpo.add_argument("--seed", type=int, default=42)
    dpo.add_argument("--swan-name", default="dpo-ranker")
    dpo.set_defaults(func=cmd_dpo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
