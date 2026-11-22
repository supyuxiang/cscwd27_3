"""Full SFT of Qwen2.5-7B on 4 GPUs. One L40S cannot hold the Adam states."""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from my_sft_trainer_v1 import (
    DatasetPadMode,
    MySFTDataset_Base,
    SFTTensorCollator,
    masked_lm_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.global_batch % (args.micro * world) != 0:
        raise SystemExit("global batch must divide by micro * world")
    accum = args.global_batch // (args.micro * world)

    class Config(dict):
        def get(self, key, default=None):
            return self[key] if key in self else default

    data_config = Config(max_length=args.max_length, shuffle=False, seed=1)
    train_data = MySFTDataset_Base(args.train, args.model, data_config)
    val_data = MySFTDataset_Base(args.val, args.model, data_config)
    train_sampler = DistributedSampler(train_data, shuffle=True, drop_last=True, seed=1)
    train_loader = DataLoader(
        train_data,
        batch_size=args.micro,
        sampler=train_sampler,
        collate_fn=SFTTensorCollator(DatasetPadMode.RIGHT),
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.micro,
        shuffle=False,
        collate_fn=SFTTensorCollator(DatasetPadMode.RIGHT),
        num_workers=0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer})
    model = FSDP(
        model,
        auto_wrap_policy=wrap,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // accum
    total_steps = steps_per_epoch * args.epochs
    warmup = int(0.03 * total_steps)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_steps,
    )
    if rank == 0:
        print(
            f"[7b] samples={len(train_data)} steps/epoch={steps_per_epoch} "
            f"total={total_steps} accum={accum}",
            flush=True,
        )

    def evaluate() -> None:
        model.eval()
        totals = torch.zeros(2, device="cuda")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].cuda(non_blocking=True)
                attention_mask = batch["attention_mask"].cuda(non_blocking=True)
                loss_mask = batch["loss_mask"].cuda(non_blocking=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss, n_tok, _ = masked_lm_loss(logits, input_ids, loss_mask)
                totals[0] += loss.detach() * n_tok
                totals[1] += n_tok
        dist.all_reduce(totals)
        if rank == 0:
            mean = float(totals[0] / totals[1].clamp_min(1))
            print(f"[7b] val_loss={mean:.4f} n_tok={float(totals[1]):.0f}", flush=True)
        model.train()

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for data in train_loader:
            input_ids = data["input_ids"].cuda(non_blocking=True)
            attention_mask = data["attention_mask"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss, _, _ = masked_lm_loss(logits, input_ids, loss_mask)
            (loss / accum).backward()
            pending += 1
            if pending < accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            global_step += 1
            if rank == 0 and global_step % 10 == 0:
                print(
                    f"[7b] step {global_step}/{total_steps} loss={float(loss.detach()):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            evaluate()
            break
        evaluate()

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
    if rank == 0:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        bare = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        bare.load_state_dict(state)
        bare.save_pretrained(out)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
        print(f"[7b] saved {out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
