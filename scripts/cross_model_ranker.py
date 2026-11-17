"""Cross-model generalization of the Qwen2.5-3B ranker.

Replay another model's already-sampled MATH500 responses (and their math-verify
labels) through Qwen2.5-3B, read the ranker's probe layer, and score them with
the frozen ranker. One process, one GPU, one source model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FDU = Path("/home/yxfeng/fdu")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FDU))

from exp1_math.language import Language  # noqa: E402
from ranker.features import extract_features  # noqa: E402
from ranker.policy import load_tokenizer  # noqa: E402
from ranker.ranker_model import load_ranker  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

SAMPLED_ROOT = Path("/data3/yxfeng/fdu/exp1_math/sampled_res")
JUDGE_ROOT = Path("/data3/yxfeng/fdu/exp1_math/judge")
NAME_RE = re.compile(
    r"reward_math500_(?P<lang>[a-z]+)_n(?P<n>\d+)_t(?P<temp>[0-9.]+)_tokens(?P<tokens>\d+)\.pt$"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def balanced_row_index(reward: torch.Tensor, n_pos: int, n_neg: int, seed: int) -> torch.Tensor:
    """Same draw as exp1_math/inlp_runner.py: shuffle with RandomState(seed), keep n_pos then n_neg."""
    labels = reward.detach().float().view(-1).cpu()
    pos = torch.where(labels >= 0.5)[0].numpy()
    neg = torch.where(labels < 0.5)[0].numpy()
    rng = np.random.RandomState(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if len(pos) < n_pos or len(neg) < n_neg:
        raise RuntimeError(
            f"need {n_pos} positive and {n_neg} negative, got {len(pos)}/{len(neg)}"
        )
    chosen = np.concatenate([pos[:n_pos], neg[:n_neg]])
    rng.shuffle(chosen)
    return torch.from_numpy(chosen.astype(np.int64))


def paired_settings(source: str) -> list[dict]:
    judge_dir = JUDGE_ROOT / source
    if not judge_dir.is_dir():
        raise SystemExit(f"no judge dir: {judge_dir}")
    settings = []
    for reward_path in sorted(judge_dir.glob("reward_math500_*.pt")):
        match = NAME_RE.search(reward_path.name)
        if match is None:
            log(f"[skip] unrecognized reward name {reward_path.name}")
            continue
        info = match.groupdict()
        res_name = (
            f"res_math500_{info['lang']}_n{info['n']}_t{info['temp']}_tokens{info['tokens']}.json"
        )
        res_path = SAMPLED_ROOT / source / res_name
        if not res_path.is_file():
            raise SystemExit(f"reward has no sampled responses: {res_path}")
        settings.append(
            {
                "lang": info["lang"],
                "n": int(info["n"]),
                "temperature": float(info["temp"]),
                "tokens": int(info["tokens"]),
                "res": res_path,
                "reward": reward_path,
            }
        )
    if not settings:
        raise SystemExit(f"no reward tensors in {judge_dir}")
    return settings


def load_inlp_groups(source: str, lang: str, files: list[dict], n_pos: int, n_neg: int, seed: int) -> list[dict]:
    """Pool every judged file of one language, then keep the INLP 500+500 draw."""
    rows: list[dict] = []
    label_parts: list[torch.Tensor] = []
    temps: list[float] = []
    for setting in files:
        part = json.loads(setting["res"].read_text(encoding="utf-8"))
        labels = torch.load(setting["reward"], map_location="cpu", weights_only=False).view(-1).float()
        if len(part) != int(labels.numel()):
            raise SystemExit(
                f"length mismatch {setting['res'].name}: rows={len(part)} labels={int(labels.numel())}"
            )
        rows.extend(part)
        label_parts.append(labels)
        temps.extend([float(setting["temperature"])] * len(part))
    labels = torch.cat(label_parts, dim=0)
    n_avail_pos = int((labels >= 0.5).sum())
    n_avail_neg = int((labels < 0.5).sum())
    use_pos, use_neg = n_pos, n_neg
    if n_avail_pos < use_pos or n_avail_neg < use_neg:
        use_pos = min(use_pos, n_avail_pos)
        use_neg = min(use_neg, n_avail_neg)
        # Stay balanced when one class is short of the INLP 500.
        use_pos = use_neg = min(use_pos, use_neg)
        log(
            f"[subsample] {source} {lang} only {n_avail_pos} correct / {n_avail_neg} incorrect; "
            f"keep {use_pos}+{use_neg} instead of {n_pos}+{n_neg}"
        )
    chosen = balanced_row_index(labels, use_pos, use_neg, seed)
    language = Language(lang)
    buckets: OrderedDict[str, dict] = OrderedDict()
    for index in chosen.tolist():
        row = rows[index]
        question = row["question"]
        bucket = buckets.get(question)
        if bucket is None:
            bucket = {
                "qid": len(buckets),
                "lang": lang,
                "prompt": language.get_prompt(question),
                "responses": [],
                "labels": [],
                "temps": [],
                "n_pos_kept": use_pos,
                "n_neg_kept": use_neg,
                "n_pos_available": n_avail_pos,
                "n_neg_available": n_avail_neg,
            }
            buckets[question] = bucket
        bucket["responses"].append(row["res"])
        bucket["labels"].append(float(labels[index]))
        bucket["temps"].append(temps[index])
    groups = []
    for bucket in buckets.values():
        bucket["labels"] = torch.tensor(bucket["labels"], dtype=torch.float32)
        bucket["temps"] = torch.tensor(bucket["temps"], dtype=torch.float32)
        groups.append(bucket)
    log(
        f"[subsample] {source} {lang} keep {use_pos} pos + {use_neg} neg / {len(labels)} "
        f"-> {len(groups)} questions"
    )
    return groups


def load_encoder(path: str, device: torch.device):
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"
    except Exception:
        attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    model.config.use_cache = False
    model.to(device).eval()
    log(f"[encoder] {path} attn={attn} device={device}")
    return model


def extract_groups(model, tokenizer, groups: list[dict], layer: int, batch_size: int, max_length: int):
    prompts: list[str] = []
    responses: list[str] = []
    for group in groups:
        prompts.extend([group["prompt"]] * len(group["responses"]))
        responses.extend(group["responses"])
    bs = batch_size
    while True:
        try:
            inst, resp = extract_features(
                model,
                tokenizer,
                prompts,
                responses,
                layer_index=layer,
                batch_size=bs,
                max_length=max_length,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            log(f"[oom] retry extract batch_size={bs}")
    if inst.size(-1) == 0:
        raise RuntimeError("empty instruction features")
    offset = 0
    for group in groups:
        k = len(group["responses"])
        group["instruction"] = inst[offset]
        group["response"] = resp[offset:offset + k]
        offset += k
    if offset != inst.size(0):
        raise RuntimeError("feature offset mismatch")


def score_groups(ranker, groups: list[dict], device: torch.device) -> None:
    ranker.eval()
    with torch.no_grad():
        for group in groups:
            inst = group["instruction"].unsqueeze(0).to(device)
            resp = group["response"].unsqueeze(0).to(device)
            group["scores"] = ranker(inst, resp)[0].float().cpu()


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    y = labels.astype(np.int32)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    y = labels.astype(np.int32)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return None
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y) + 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def _flat(groups: list[dict], field: str) -> torch.Tensor:
    return torch.cat([g[field].view(-1) for g in groups], dim=0)


def summarize(groups: list[dict], *, key: str) -> dict:
    scores = _flat(groups, "scores").numpy()
    labels = _flat(groups, "labels").numpy()
    pred = scores > 0
    truth = labels >= 0.5
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    pos_rate = float(truth.mean()) if len(truth) else 0.0
    majority = max(pos_rate, 1.0 - pos_rate)

    pair_hits = pair_total = 0
    top1 = []
    top1_when_possible = []
    for group in groups:
        bits = group["labels"].tolist()
        sc = group["scores"].tolist()
        pos = [s for s, y in zip(sc, bits) if y >= 0.5]
        neg = [s for s, y in zip(sc, bits) if y < 0.5]
        for p in pos:
            for n in neg:
                pair_total += 1
                pair_hits += int(p > n)
        best = int(torch.argmax(group["scores"]))
        hit = int(group["labels"][best].item() >= 0.5)
        top1.append(hit)
        if bool((group["labels"] >= 0.5).any()):
            top1_when_possible.append(hit)

    pair = pair_hits / pair_total if pair_total else None
    both = [g for g in groups if bool((g["labels"] >= 0.5).any()) and bool((g["labels"] < 0.5).any())]
    return {
        "split": key,
        "questions": len(groups),
        "responses": int(labels.size),
        "mean_k": float(np.mean([int(g["labels"].numel()) for g in groups])) if groups else 0.0,
        "label_rate": pos_rate,
        "majority_acc": majority,
        "judge_acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": _auroc(scores, truth.astype(np.int32)),
        "average_precision": _average_precision(scores, truth.astype(np.int32)),
        "pairwise_acc": pair,
        "pairwise_questions": len(both),
        "top1_acc": float(np.mean(top1)) if top1 else None,
        "top1_acc_if_any_correct": float(np.mean(top1_when_possible)) if top1_when_possible else None,
        "lift_top1_over_label_rate": (float(np.mean(top1)) - pos_rate) if top1 else None,
        "score_mean_correct": float(scores[truth].mean()) if truth.any() else None,
        "score_mean_incorrect": float(scores[~truth].mean()) if (~truth).any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="sampled model dir name, e.g. Qwen2.5-0.5B")
    parser.add_argument("--encoder", default="/data3/yxfeng/models/Qwen2.5/Qwen2.5-3B")
    parser.add_argument("--ranker", default="/data3/yxfeng/cscwd27_iraker/run1/ranker.pt")
    parser.add_argument("--output-dir", default="/data3/yxfeng/cscwd27_iraker/run1/cross_model_inlp1k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--n-pos", type=int, default=500)
    parser.add_argument("--n-neg", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = paired_settings(args.source)
    by_lang: dict[str, list[dict]] = {}
    for setting in settings:
        by_lang.setdefault(setting["lang"], []).append(setting)
    log(
        f"[data] {args.source} files={len(settings)} langs={list(by_lang)} "
        f"draw={args.n_pos}+{args.n_neg} seed={args.seed}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker, meta = load_ranker(args.ranker, device=device)
    layer = int(meta["probe_layer"])
    log(
        f"[ranker] kind={meta['kind']} d_in={meta['d_in']} probe_layer={layer} "
        f"depth={meta['depth']} val_pair_acc={meta.get('config', {}).get('val_pair_acc')}"
    )

    feature_path = out_dir / "features.pt"
    if feature_path.is_file():
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        groups = payload["groups"]
        if (
            int(payload.get("probe_layer", -1)) != layer
            or payload.get("encoder") != args.encoder
            or int(payload.get("n_pos", -1)) != args.n_pos
            or int(payload.get("seed", -1)) != args.seed
        ):
            raise SystemExit(f"existing features do not match this run: {feature_path}")
        log(f"[resume] loaded {len(groups)} groups from {feature_path}")
    else:
        tokenizer = load_tokenizer(args.encoder)
        model = load_encoder(args.encoder, device)
        groups = []
        for lang, files in by_lang.items():
            part = load_inlp_groups(args.source, lang, files, args.n_pos, args.n_neg, args.seed)
            log(
                f"[extract] {args.source} {lang} "
                f"questions={len(part)} responses={sum(len(g['responses']) for g in part)}"
            )
            extract_groups(model, tokenizer, part, layer, args.batch_size, args.max_length)
            d_in = int(part[0]["instruction"].numel())
            if d_in != int(meta["d_in"]):
                raise SystemExit(f"feature dim {d_in} != ranker d_in {meta['d_in']}")
            for group in part:
                group.pop("responses", None)
            groups.extend(part)
            log(f"[extract] done {lang} running_groups={len(groups)}")
        tmp = feature_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "source": args.source,
                "encoder": args.encoder,
                "probe_layer": layer,
                "depth": float(meta["depth"]),
                "n_pos": args.n_pos,
                "n_neg": args.n_neg,
                "seed": args.seed,
                "groups": groups,
            },
            tmp,
        )
        tmp.replace(feature_path)
        log(f"[extract] saved {feature_path} groups={len(groups)}")
        del model
        torch.cuda.empty_cache()

    score_groups(ranker, groups, device)
    by_key: dict[str, list] = {"all": groups}
    for group in groups:
        by_key.setdefault(f"lang={group['lang']}", []).append(group)

    report = {
        "source": args.source,
        "encoder": args.encoder,
        "ranker": args.ranker,
        "ranker_kind": meta["kind"],
        "probe_layer": layer,
        "depth": float(meta["depth"]),
        "d_in": int(meta["d_in"]),
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "seed": args.seed,
        "decision": "score > 0 is predicted correct",
        "note": "Test rows are the INLP draw: 500 correct + 500 incorrect per language, seed 42, taken from every judged file of that language. Hidden states are Qwen2.5-3B at the ranker's probe layer. Listwise scores use the kept responses of one question.",
        "splits": [summarize(rows, key=key) for key, rows in by_key.items()],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(text + "\n", encoding="utf-8")
    log(text)
    log(f"[done] {metrics_path}")


if __name__ == "__main__":
    main()
