"""Policy loading, generation and the shared SFT trainer (DDP, LoRA or full FT)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_policy(path: str, lora_r: int = 16, lora_alpha: int = 32, dropout: float = 0.05, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if _has_flash() else "sdpa",
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if lora_r > 0:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    model.config.use_cache = False
    return model


def _has_flash() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


# ------------------------------------------------------------------ SFT data
class CompletionDataset(Dataset):
    """Each row is a prompt plus the answer text the policy should learn."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 4096):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt_ids = self.tokenizer.encode(row["prompt"], add_special_tokens=False)
        response_ids = self.tokenizer.encode(row["response"], add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            response_ids = response_ids + [eos]
        ids = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
            mask = mask[: self.max_length]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float32),
        }


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(item["input_ids"].numel() for item in batch)
    input_ids, loss_mask, attention = [], [], []
    for item in batch:
        ids = item["input_ids"]
        mask = item["loss_mask"]
        pad = width - ids.numel()
        input_ids.append(F.pad(ids, (0, pad), value=pad_id))
        loss_mask.append(F.pad(mask, (0, pad), value=0.0))
        attention.append(F.pad(torch.ones_like(ids), (0, pad), value=0))
    return {
        "input_ids": torch.stack(input_ids),
        "loss_mask": torch.stack(loss_mask),
        "attention_mask": torch.stack(attention),
    }


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    """Cross entropy on the response tokens only."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    shift_mask = mask[:, 1:].float()
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)
    denom = shift_mask.sum().clamp(min=1.0)
    return (loss * shift_mask).sum() / denom, denom


def run_sft(
    model,
    tokenizer,
    rows: list[dict],
    *,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
    epochs: int = 3,
    micro_batch: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    warmup_ratio: float = 0.03,
    max_length: int = 4096,
    max_grad_norm: float = 1.0,
    log=print,
    swan=None,
    seed: int = 0,
) -> dict:
    dataset = CompletionDataset(rows, tokenizer, max_length)
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed)
        if world > 1
        else None
    )
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: collate(batch, pad_id),
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    params = trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1 + math.cos(math.pi * progress))

    model.train()
    global_step = 0
    running = 0.0
    seen = 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        accum = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            sync = (accum + 1) == grad_accum
            context = model.no_sync() if (world > 1 and not sync) else _null()
            with context:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss, ntokens = masked_lm_loss(logits, batch["input_ids"], batch["loss_mask"])
                (loss / grad_accum).backward()
            running += float(loss.detach())
            seen += 1
            accum += 1
            if not sync:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr_at(global_step)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            global_step += 1
            if global_step % 10 == 0:
                mean_loss = running / max(seen, 1)
                mean_ntok = float(ntokens)
                log(f"[sft] step {global_step}/{total_steps} loss={mean_loss:.4f} lr={lr_at(global_step):.2e}")
                if swan is not None:
                    swan.log(
                        {
                            "train/loss": mean_loss,
                            "train/lr": lr_at(global_step),
                            "train/epoch": epoch + 1,
                            "train/ntokens": mean_ntok,
                        },
                        step=global_step,
                    )
                running = 0.0
                seen = 0
    return {"steps": global_step, "epochs": epochs, "lr": lr}


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def wrap_ddp(model, local_rank: int, world: int):
    if world <= 1:
        return model
    return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


def save_policy(path: str, model, tokenizer) -> None:
    """Merge LoRA adapters into the base weights so plain AutoModel can load it."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    if hasattr(raw, "merge_and_unload"):
        raw = raw.merge_and_unload()
    raw.save_pretrained(target)
    tokenizer.save_pretrained(target)


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float, do_sample: bool):
    """Plain generation for the base model: prompts are raw strings, no chat template."""
    device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    out = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5) if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(out[:, prompt_width:], skip_special_tokens=True)
