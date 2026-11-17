"""Stage 4 (route A): rejection-sampling SFT.

The ranker scores every sampled candidate in ``--features``. For each question
the highest-scoring completion is kept, and the policy is SFT-trained on
prompt + that completion (loss on the answer tokens only).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.core import barrier, init_distributed, set_seed, write_json  # noqa: E402
from ranker.evaluate import DEFAULT_SETS, evaluate_model  # noqa: E402
from ranker.policy import load_policy, load_tokenizer, run_sft, save_policy, wrap_ddp  # noqa: E402
from ranker.ranker_model import load_ranker  # noqa: E402
from ranker.ranker_data import pad_groups  # noqa: E402
from ranker.swan import SwanLog  # noqa: E402


def select_best(ranker_path: str, groups: list[dict], keep: int) -> list[dict]:
    """Pick the top-k non-empty responses per question by ranker score.

    Empty completions get loss on only the EOS token and teach the policy to
    answer with EOS right after the prompt, so they are filtered out.
    """
    model, _ = load_ranker(ranker_path, device="cpu")
    chosen: list[dict] = []
    with torch.no_grad():
        for group in groups:
            inst, resp, labels = pad_groups([group], torch.device("cpu"))
            scores = model(inst, resp)[0][: group["response"].size(0)]
            responses = group["responses"]
            nonempty = [i for i in range(len(responses)) if str(responses[i]).strip()]
            if not nonempty:
                continue
            order = sorted(nonempty, key=lambda i: float(scores[i]), reverse=True)
            for index in order[:keep]:
                chosen.append(
                    {
                        "qid": int(group["qid"]),
                        "prompt": group["prompt"],
                        "response": responses[index],
                        "label": int(group["labels"][index].item()),
                        "score": float(scores[index]),
                    }
                )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-sets", default=",".join(DEFAULT_SETS))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-sampling", type=int, default=4)
    parser.add_argument("--sampler-gpu", type=int, default=0, help="GPU index for the vLLM eval sampler")
    parser.add_argument("--skip-train", action="store_true", help="use an already-saved policy_sft and only run eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, local = init_distributed()
    set_seed(args.seed)
    log = print if rank == 0 else (lambda *a, **k: None)
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier(world)

    if args.skip_train:
        if rank == 0 and args.eval:
            results = evaluate_model(
                str(out_dir / "policy_sft"),
                datasets=tuple(x.strip() for x in args.eval_sets.split(",") if x.strip()),
                n_sampling=args.eval_sampling,
                limit=args.eval_limit,
                gpu=args.sampler_gpu,
                log=log,
            )
            write_json(out_dir / "eval_sft.json", results)
            log(f"[sft] eval done macro={results['macro_avg']:.2f}")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    groups = payload["groups"]
    log(f"[sft] groups={len(groups)} ranker={args.ranker}")
    chosen = select_best(args.ranker, groups, args.keep) if rank == 0 else []
    # Broadcast-free: every rank selects from the same data, so results match.
    if rank != 0:
        chosen = select_best(args.ranker, groups, args.keep)
    correct = sum(row["label"] for row in chosen)
    log(f"[sft] selected={len(chosen)} correct_labels={correct} ({correct / max(1, len(chosen)):.1%})")
    if rank == 0:
        write_json(out_dir / "selected_sft.json", chosen)

    tokenizer = load_tokenizer(args.model)
    model = load_policy(args.model, lora_r=args.lora_r)
    device = torch.device(f"cuda:{local}")
    model.to(device)
    model = wrap_ddp(model, local, world)

    swan = (
        SwanLog(
            "iraker",
            "routeA-rejection-sft",
            config={"route": "A", "keep": args.keep, "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r},
            enabled=rank == 0,
        )
        if rank == 0
        else None
    )
    stats = run_sft(
        model,
        tokenizer,
        chosen,
        rank=rank,
        world=world,
        local_rank=local,
        device=device,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        log=log,
        swan=swan,
    )
    barrier(world)
    if rank == 0:
        save_policy(out_dir / "policy_sft", model, tokenizer)
        write_json(out_dir / "sft_stats.json", stats)
        log(f"[sft] saved {out_dir / 'policy_sft'}")
    barrier(world)

    # Eval runs only on rank 0 (vLLM on the sampler GPU). The other ranks must
    # not block in a barrier for the full eval (~10 min > NCCL 600s timeout),
    # so they poll for the result file instead and only join a barrier after.
    done_marker = out_dir / ".eval_done"
    if args.eval and rank == 0:
        results = evaluate_model(
            str(out_dir / "policy_sft"),
            datasets=tuple(x.strip() for x in args.eval_sets.split(",") if x.strip()),
            n_sampling=args.eval_sampling,
            limit=args.eval_limit,
            gpu=args.sampler_gpu,
            log=log,
        )
        write_json(out_dir / "eval_sft.json", results)
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
    barrier(world)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
