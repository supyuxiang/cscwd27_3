from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.math_ev import DEFAULT_SETS, evaluate_model  # noqa: E402
from generator.prompt import build_prompt  # noqa: E402
from generator.utils import barrier, init_distributed, set_seed, write_json  # noqa: E402
from policy.policy import load_policy, load_tokenizer, run_sft, save_policy, wrap_ddp  # noqa: E402
from policy.swan import SwanLog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, local = init_distributed()
    set_seed(args.seed)
    log = print if rank == 0 else (lambda *a, **k: None)
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    rows = [
        {"prompt": build_prompt(str(row["question"])), "response": str(row["golden_res"])}
        for row in raw_rows
    ]
    log(f"[baseline-sft] rows={len(rows)} epochs={args.epochs} lr={args.lr}")

    tokenizer = load_tokenizer(args.model)
    model = load_policy(args.model, lora_r=args.lora_r)
    device = torch.device(f"cuda:{local}")
    model.to(device)
    model = wrap_ddp(model, local, world)

    swan = (
        SwanLog(
            "iraker",
            "baselineA-teacher-sft",
            config={"route": "baseline_sft", "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r},
            enabled=rank == 0,
        )
        if rank == 0
        else None
    )
    stats = run_sft(
        model,
        tokenizer,
        rows,
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
        save_policy(out_dir / "policy_sft_teacher", model, tokenizer)
        write_json(out_dir / "sft_stats.json", stats)
        log(f"[baseline-sft] saved {out_dir / 'policy_sft_teacher'}")
    barrier(world)

    done_marker = out_dir / ".eval_done"
    if args.eval and rank == 0:
        results = evaluate_model(
            str(out_dir / "policy_sft_teacher"),
            datasets=tuple(x.strip() for x in args.eval_sets.split(",") if x.strip()),
            n_sampling=args.eval_sampling,
            limit=args.eval_limit,
            gpu=args.sampler_gpu,
            log=log,
        )
        write_json(out_dir / "eval_sft_teacher.json", results)
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
