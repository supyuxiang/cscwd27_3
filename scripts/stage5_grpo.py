"""Stage 5 (route B): GRPO with the ranker replacing the advantage estimate.

Each iteration:
  1. rank 0 samples G completions per question from the current policy (vLLM)
  2. rank 0 extracts probe-layer features and scores them with the frozen ranker,
     then z-scores those scores inside each question group to get advantages
  3. the sampled completions are broadcast to every rank
  4. the policy takes `inner_epochs` PPO-clipped steps on those completions
     with a KL penalty to a frozen copy of the starting policy
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iraker.core import barrier, build_prompt, grade_text, init_distributed, set_seed, write_json  # noqa: E402
from iraker.evaluate import DEFAULT_SETS, evaluate_model  # noqa: E402
from iraker.features import decoder_layers, extract_features  # noqa: E402
from iraker.grpo import grpo_step, group_advantages  # noqa: E402
from iraker.policy import load_policy, load_tokenizer, save_policy, wrap_ddp  # noqa: E402
from iraker.ranker_model import load_ranker  # noqa: E402
from iraker.swan import SwanLog  # noqa: E402


def sample_group(model_path: str, questions: list[str], args, sampler_gpu: int | None) -> list[list[dict]]:
    """Sample G completions per question.

    When ``sampler_gpu`` is set we hand the work to a separate process pinned to
    that GPU, so the trainer's GPUs keep their memory for the policy. Otherwise
    vLLM runs in-process on whatever device is visible.
    """
    from iraker.sampler import sample_with_worker

    groups = sample_with_worker(
        model_path,
        questions,
        group_size=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gpu=sampler_gpu,
    )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--ranker", required=True)
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
    parser.add_argument("--depth", type=float, default=0.6)
    parser.add_argument(
        "--sampler-gpu",
        type=int,
        default=None,
        help="physical GPU index reserved for vLLM sampling; None samples in-process",
    )
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
    # The last visible GPU is reserved for the vLLM sampler; training uses the rest.
    sampler_gpu = args.sampler_gpu
    if rank == 0:
        log(f"[grpo] world={world} sampler_gpu={sampler_gpu}")

    import json

    problems = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    if args.limit > 0:
        problems = problems[: args.limit]
    questions = [str(p["question"]) for p in problems]
    golds = [str(p["answer"]) for p in problems]

    tokenizer = load_tokenizer(args.model)
    policy = load_policy(args.model, lora_r=args.lora_r)
    reference = copy.deepcopy(policy) if args.lr > 0 else None
    if reference is not None:
        for param in reference.parameters():
            param.requires_grad = False
        reference.eval()
    device = torch.device(f"cuda:{local}")
    policy.to(device)
    if reference is not None:
        reference.to(device)
    policy = wrap_ddp(policy, local, world)

    ranker, _ = load_ranker(args.ranker, device=device)
    n_layers = len(decoder_layers(policy.module if hasattr(policy, "module") else policy))
    probe_layer = min(n_layers - 1, max(0, int(round(args.depth * (n_layers - 1)))))
    log(f"[grpo] layers={n_layers} probe_layer={probe_layer} ranker={args.ranker}")

    optimizer = torch.optim.AdamW(
        [p for p in (policy.module if hasattr(policy, "module") else policy).parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    swan = (
        SwanLog(
            "iraker",
            "routeB-grpo-ranker",
            config={
                "route": "B", "advantage": "ranker", "iterations": args.iterations,
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
        iteration_questions = [questions[i] for i in picks]

        payload = []
        if rank == 0:
            groups = sample_group(args.model, iteration_questions, args, sampler_gpu)
            raw = policy.module if hasattr(policy, "module") else policy
            raw.eval()
            scored_groups = []
            for qid, group in zip(picks, groups):
                prompts = [item["prompt"] for item in group]
                responses = [item["response"] for item in group]
                inst, resp = extract_features(
                    raw,
                    tokenizer,
                    prompts,
                    responses,
                    layer_index=probe_layer,
                    batch_size=2,
                    max_length=args.max_length,
                )
                with torch.no_grad():
                    scores = ranker(inst.to(device), resp.unsqueeze(1).to(device)).squeeze(-1)
                if len(group) > 1:
                    advantage = group_advantages(scores.unsqueeze(0)).squeeze(0)
                else:
                    advantage = torch.zeros_like(scores)
                for item, inst_row, resp_row, score, adv in zip(
                    group, inst, resp, scores, advantage
                ):
                    item["_inst"] = inst_row
                    item["_resp"] = resp_row
                    item["_score"] = float(score)
                    item["_adv"] = float(adv)
                    item["qid"] = qid
                    item["label"] = int(grade_text(item["response"], golds[qid]))
                scored_groups.append(group)
            payload = [[item for item in group] for group in scored_groups]

        if world > 1:
            holder = [payload]
            dist.broadcast_object_list(holder, src=0)
            payload = holder[0]
        rows = [item for group in payload for item in group]
        if not rows:
            continue
        mean_score = sum(item["_score"] for item in rows) / len(rows)
        mean_label = sum(item["label"] for item in rows) / len(rows)

        policy.train()
        for inner in range(args.inner_epochs):
            stats = grpo_step(
                policy,
                reference if reference is not None else policy,
                ranker,
                rows,
                tokenizer,
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef if reference is not None else 0.0,
                max_length=args.max_length,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in (policy.module if hasattr(policy, "module") else policy).parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
        record = {
            "iteration": iteration,
            "questions": len(payload),
            "samples": len(rows),
            "mean_ranker_score": mean_score,
            "mean_correct": mean_label,
            "policy_loss": stats["policy_loss"],
            "kl": stats["kl"],
            "mean_adv": stats["mean_adv"],
        }
        history.append(record)
        if rank == 0:
            log(f"[grpo] iter {iteration} samples={len(rows)} ranker={mean_score:.3f} correct={mean_label:.3f} "
                f"kl={stats['kl']:.4f}")
            write_json(out_dir / "grpo_history.json", history)
            swan.log(
                {
                    "train/ranker_score": mean_score,
                    "train/correct": mean_label,
                    "train/policy_loss": stats["policy_loss"],
                    "train/kl": stats["kl"],
                    "train/advantage": stats["mean_adv"],
                },
                step=iteration,
            )
        barrier(world)

    if rank == 0:
        save_policy(out_dir / "policy_grpo", policy, tokenizer)
        log(f"[grpo] saved {out_dir / 'policy_grpo'}")
    barrier(world)
    done_marker = out_dir / ".eval_done"
    if args.eval and rank == 0:
        results = evaluate_model(
            str(out_dir / "policy_grpo"),
            datasets=tuple(x.strip() for x in args.eval_sets.split(",") if x.strip()),
            n_sampling=args.eval_sampling,
            limit=args.eval_limit,
            gpu=args.sampler_gpu,
            log=log,
        )
        write_json(out_dir / "eval_grpo.json", results)
        for name, value in results.items():
            if isinstance(value, dict):
                swan.log({f"eval/{name}": value["mean_acc"]})
        swan.log({"eval/macro_avg": results["macro_avg"]})
        done_marker.touch()
    elif args.eval:
        while not done_marker.exists():
            time.sleep(5)
    if swan is not None:
        swan.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
