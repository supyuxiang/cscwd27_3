"""Baseline B: standard GRPO. The advantage is the z-scored verifier reward, not the ranker."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iraker.core import barrier, build_prompt, grade_text, init_distributed, set_seed, write_json  # noqa: E402
from iraker.evaluate import DEFAULT_SETS, evaluate_model  # noqa: E402
from iraker.grpo import group_advantages, grpo_step_verifier  # noqa: E402
from iraker.policy import load_policy, load_tokenizer, save_policy, wrap_ddp  # noqa: E402
from iraker.swan import SwanLog  # noqa: E402


def sample_group(model_path: str, questions: list[str], args, sampler_gpu: int | None) -> list[list[dict]]:
    """Sample G completions per question on a dedicated GPU via a subprocess."""
    from iraker.sampler import sample_with_worker

    return sample_with_worker(
        model_path,
        questions,
        group_size=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gpu=sampler_gpu,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--questions-per-iter", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--inner-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--sampler-gpu", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-sets", default=",".join(DEFAULT_SETS))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-sampling", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, local = init_distributed()
    set_seed(args.seed)
    log = print if rank == 0 else (lambda *a, **k: None)
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    problems = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    if args.limit > 0:
        problems = problems[: args.limit]
    questions = [str(p["question"]) for p in problems]
    golds = [str(p["answer"]) for p in problems]

    tokenizer = load_tokenizer(args.model)
    policy = load_policy(args.model, lora_r=args.lora_r)
    reference = copy.deepcopy(policy)
    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()
    device = torch.device(f"cuda:{local}")
    policy.to(device)
    reference.to(device)
    policy = wrap_ddp(policy, local, world)

    optimizer = torch.optim.AdamW(
        [p for p in (policy.module if hasattr(policy, "module") else policy).parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )
    swan = (
        SwanLog(
            "iraker",
            "baselineB-grpo-verifier",
            config={
                "route": "baseline_grpo", "advantage": "verifier", "iterations": args.iterations,
                "group_size": args.group_size, "lr": args.lr, "lora_r": args.lora_r,
            },
            enabled=rank == 0,
        )
        if rank == 0
        else None
    )
    history = []
    for iteration in range(args.iterations):
        start = (iteration * args.questions_per_iter) % len(questions)
        picks = [(start + i) % len(questions) for i in range(args.questions_per_iter)]
        payload = []
        if rank == 0:
            groups = sample_group(args.model, [questions[i] for i in picks], args, args.sampler_gpu)
            for qid, group in zip(picks, groups):
                bits = [grade_text(item["response"], golds[qid]) for item in group]
                scores = torch.tensor(bits, dtype=torch.float32)
                advantage = (
                    group_advantages(scores.unsqueeze(0)).squeeze(0)
                    if len(group) > 1
                    else torch.zeros_like(scores)
                )
                for item, bit, adv in zip(group, bits, advantage):
                    item["qid"] = qid
                    item["label"] = int(bit)
                    item["_adv"] = float(adv)
                payload.append(group)
        if world > 1:
            holder = [payload]
            dist.broadcast_object_list(holder, src=0)
            payload = holder[0]
        rows = [item for group in payload for item in group]
        if not rows:
            continue
        mean_label = sum(item["label"] for item in rows) / len(rows)
        stats = grpo_step_verifier(
            policy,
            reference,
            rows,
            tokenizer,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            max_length=args.max_length,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        stats["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in (policy.module if hasattr(policy, "module") else policy).parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        history.append(
            {
                "iteration": iteration,
                "samples": len(rows),
                "mean_correct": mean_label,
                "policy_loss": stats["policy_loss"],
                "kl": stats["kl"],
            }
        )
        if rank == 0:
            log(f"[baseline-grpo] iter {iteration} correct={mean_label:.3f} kl={stats['kl']:.4f}")
            write_json(out_dir / "grpo_history.json", history)
            swan.log(
                {
                    "train/correct": mean_label,
                    "train/policy_loss": stats["policy_loss"],
                    "train/kl": stats["kl"],
                },
                step=iteration,
            )
        barrier(world)

    if rank == 0:
        save_policy(out_dir / "policy_grpo_verifier", policy, tokenizer)
        log(f"[baseline-grpo] saved {out_dir / 'policy_grpo_verifier'}")
    barrier(world)
    if args.eval and rank == 0:
        results = evaluate_model(
            str(out_dir / "policy_grpo_verifier"),
            datasets=tuple(x.strip() for x in args.eval_sets.split(",") if x.strip()),
            n_sampling=args.eval_sampling,
            limit=args.eval_limit,
            gpu=args.sampler_gpu,
            log=log,
        )
        write_json(out_dir / "eval_grpo_verifier.json", results)
        for name, value in results.items():
            if isinstance(value, dict):
                swan.log({f"eval/{name}": value["mean_acc"]})
        swan.log({"eval/macro_avg": results["macro_avg"]})
    if swan is not None:
        swan.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
