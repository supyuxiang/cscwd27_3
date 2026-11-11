"""Iso-prompt DPO-Ranker round: cache, fit, build pairs, update the policy."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_problems, load_split, save_split, split_pool
from .optim import train_dpo, train_sft
from .pairs import PreferencePair, ScoredCandidate, pairs_for_prompt, select_budget
from .ranker import fit_ranker, score_features
from .sample import format_prompt, sample_problem
from .verifier import is_correct


def load_policy(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    if config.get("lora", True):
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(config.get("lora_r", 16)),
                lora_alpha=int(config.get("lora_alpha", 32)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    model.to(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    return model, tokenizer


def prepare_pool(config: dict) -> list:
    output = Path(config["output_dir"])
    split_path = output / "split.json"
    if split_path.is_file() and not config.get("resplit", False):
        return load_split(split_path)
    problems = split_pool(
        load_problems(config["train_json"]),
        int(config["n_prompts"]),
        int(config["n_labeled"]),
        int(config["seed"]),
    )
    save_split(problems, split_path)
    return problems


def _kind(candidates, labeled: bool) -> str:
    if not labeled:
        return "reference_free"
    labels = {item.label for item in candidates}
    if labels == {0, 1}:
        return "informative"
    if labels == {1}:
        return "all_correct"
    return "all_wrong"


def _weak_pairs(grouped: dict[int, list]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int]]:
    pairs = []
    owners = []
    for qid, candidates in grouped.items():
        good = [item.feature for item in candidates if item.label == 1]
        bad = [item.feature for item in candidates if item.label == 0]
        for positive in good:
            for negative in bad:
                pairs.append((positive, negative))
                owners.append(qid)
    return pairs, owners


def collect(model, tokenizer, problems, config: dict):
    grouped = defaultdict(list)
    for problem in problems:
        grouped[problem.qid] = sample_problem(
            model,
            tokenizer,
            problem,
            k=int(config["k"]),
            max_new_tokens=int(config["max_new_tokens"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            layer_depth=float(config["layer_depth"]),
            pooling=str(config["pooling"]),
            verifier=is_correct,
        )
    return grouped


def build_rank_pairs(model, tokenizer, problems, config: dict, device: torch.device) -> tuple[list[PreferencePair], dict]:
    supervision = collect(model, tokenizer, problems, config)
    labeled = {qid: items for qid, items in supervision.items() if any(item.label is not None for item in items)}
    weak, owners = _weak_pairs(labeled)
    hidden = int(config.get("ranker_hidden", 256))
    if not weak:
        raise RuntimeError("No correct-incorrect pair in D_L; the ranker has nothing to fit.")
    ranker, stats = fit_ranker(
        weak,
        input_dim=weak[0][0].numel(),
        hidden=hidden,
        epochs=int(config.get("ranker_epochs", 20)),
        lr=float(config.get("ranker_lr", 1e-3)),
        weight_decay=float(config.get("ranker_weight_decay", 1e-4)),
        device=device,
        prompt_ids=owners,
    )
    fresh = collect(model, tokenizer, problems, config)
    by_id = {problem.qid: problem for problem in problems}
    margins = []
    drafted = []
    for qid, candidates in fresh.items():
        problem = by_id[qid]
        features = torch.stack([item.feature for item in candidates])
        scores = score_features(ranker, stats, features, device)
        scored = [
            ScoredCandidate(item.text, item.label, float(score))
            for item, score in zip(candidates, scores)
        ]
        if len(scored) >= 2:
            margins.append(max(item.score for item in scored) - min(item.score for item in scored))
        drafted.append((problem, scored, _kind(candidates, problem.labeled)))
    if margins:
        quantile = float(config.get("margin_quantile", 0.5))
        ordered = sorted(margins)
        tau = ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))]
    else:
        tau = 0.0
    pairs = []
    for problem, scored, kind in drafted:
        pair = pairs_for_prompt(
            problem.qid,
            problem.question,
            scored,
            kind,
            tau,
            int(config.get("delta_len", 400)),
        )
        if pair is not None:
            pairs.append(pair)
    selected = select_budget(
        pairs,
        int(config.get("pair_budget", len(pairs))),
        float(config.get("ranker_fraction", 0.5)),
    )
    report = {
        "weak_pairs": len(weak),
        "tau": tau,
        "yield": {source: sum(pair.source == source for pair in selected) for source in ("hard_negative", "saturated", "reference_free")},
        "n_pairs": len(selected),
    }
    return selected, report


def run(config: dict) -> None:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"]))
    problems = prepare_pool(config)
    model, tokenizer = load_policy(config)
    device = model.device
    correct = []
    for problem in problems:
        if not problem.labeled:
            continue
        sampled = sample_problem(
            model,
            tokenizer,
            problem,
            k=int(config["k"]),
            max_new_tokens=int(config["max_new_tokens"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            layer_depth=float(config["layer_depth"]),
            pooling=str(config["pooling"]),
            verifier=is_correct,
        )
        correct.extend((format_prompt(tokenizer, item.prompt), item.text) for item in sampled if item.label == 1)
    train_sft(
        model,
        tokenizer,
        correct,
        epochs=int(config.get("sft_epochs", 1)),
        lr=float(config.get("sft_lr", 1e-5)),
        max_length=int(config.get("max_length", 2048)),
    )
    history = []
    for round_index in range(1, int(config.get("rounds", 1)) + 1):
        pairs, report = build_rank_pairs(model, tokenizer, problems, config, device)
        metrics = train_dpo(
            model,
            tokenizer,
            pairs,
            beta=float(config.get("beta", 0.1)),
            epochs=int(config.get("dpo_epochs", 1)),
            lr=float(config.get("dpo_lr", 5e-7)),
            batch_size=int(config.get("batch_size", 2)),
            max_length=int(config.get("max_length", 2048)),
        )
        record = {"round": round_index, **report, **metrics}
        history.append(record)
        (output / "pairs_round{}.json".format(round_index)).write_text(
            json.dumps([asdict(pair) for pair in pairs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(record), flush=True)
    model.save_pretrained(output / "policy")
    tokenizer.save_pretrained(output / "policy")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
