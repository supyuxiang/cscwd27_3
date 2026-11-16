"""Length-sum DPO on one chosen/rejected pair per question.

The reference policy is the checkpoint DPO starts from. Its sequence logprobs
are cached before LoRA training so the reference weights are not kept on GPU.
Log-softmax runs over short token chunks: a full fp32 [seq, vocab] tensor is
about 1.8GB at seq=3200 and deadlocks a 24GB card.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .policy import load_policy, load_tokenizer, save_policy, trainable_params


def token_logprob_sums(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    """Sum of token logprobs. logits [B, S, V], labels/mask [B, S] -> [B]."""
    total = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)
    vocab_index = labels.clamp(min=0).unsqueeze(-1)
    for start in range(0, labels.size(1), chunk):
        end = min(start + chunk, labels.size(1))
        logp = torch.log_softmax(logits[:, start:end, :].float(), dim=-1)
        gathered = logp.gather(-1, vocab_index[:, start:end]).squeeze(-1)
        total = total + (gathered * mask[:, start:end].float()).sum(dim=-1)
    return total


def _encode_pair(tokenizer, prompt: str, response: str, max_length: int) -> dict | None:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        response_ids = response_ids + [eos]
    if not response_ids:
        return None
    ids = prompt_ids + response_ids
    if len(ids) > max_length:
        return None
    mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "loss_mask": torch.tensor(mask, dtype=torch.float32),
    }


def _batch(item: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": item["input_ids"].unsqueeze(0).to(device),
        "loss_mask": item["loss_mask"].unsqueeze(0).to(device),
        "attention_mask": torch.ones(1, item["input_ids"].numel(), dtype=torch.long, device=device),
    }


def sequence_logprob(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return token_logprob_sums(logits[:, :-1, :], batch["input_ids"][:, 1:], batch["loss_mask"][:, 1:])[0]


def prepare_pairs(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    prepared = []
    dropped = 0
    for row in rows:
        chosen = _encode_pair(tokenizer, row["prompt"], row["chosen"], max_length)
        rejected = _encode_pair(tokenizer, row["prompt"], row["rejected"], max_length)
        if chosen is None or rejected is None:
            dropped += 1
            continue
        prepared.append({"chosen": chosen, "rejected": rejected, "qid": int(row["qid"])})
    print(f"[dpo] pairs kept={len(prepared)} dropped_overlong_or_empty={dropped}", flush=True)
    if not prepared:
        raise SystemExit("no DPO pairs left after the length filter")
    return prepared


@torch.no_grad()
def cache_reference(model, pairs: list[dict], device: torch.device) -> torch.Tensor:
    """Return [N, 2] reference sequence logprobs (chosen, rejected)."""
    model.eval()
    out = torch.empty(len(pairs), 2, dtype=torch.float32)
    for index, pair in enumerate(pairs):
        chosen = sequence_logprob(model, _batch(pair["chosen"], device))
        rejected = sequence_logprob(model, _batch(pair["rejected"], device))
        out[index, 0] = chosen.float().cpu()
        out[index, 1] = rejected.float().cpu()
        if index % 200 == 0:
            print(f"[dpo] reference {index}/{len(pairs)}", flush=True)
    if not torch.isfinite(out).all():
        raise SystemExit("reference logprobs contain non-finite values")
    return out


def train_dpo(
    model_path: str,
    rows: list[dict],
    output_dir: str,
    *,
    beta: float = 0.1,
    lr: float = 5e-6,
    epochs: int = 1,
    grad_accum: int = 16,
    max_length: int = 4096,
    lora_r: int = 16,
    seed: int = 42,
    swan=None,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_path)
    pairs = prepare_pairs(rows, tokenizer, max_length)

    ref_path = Path(output_dir) / "ref_logprobs.pt"
    if ref_path.is_file():
        reference = torch.load(ref_path, map_location="cpu", weights_only=False)["logprobs"]
        if reference.shape[0] != len(pairs):
            raise SystemExit(f"cached reference has {reference.shape[0]} rows, pairs={len(pairs)}")
        print(f"[dpo] loaded reference logprobs {tuple(reference.shape)}", flush=True)
    else:
        reference_model = load_policy(model_path, lora_r=0).to(device).eval()
        if hasattr(reference_model, "gradient_checkpointing_disable"):
            reference_model.gradient_checkpointing_disable()
        reference = cache_reference(reference_model, pairs, device)
        del reference_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logprobs": reference}, ref_path)
        print(f"[dpo] saved {ref_path}", flush=True)

    policy = load_policy(model_path, lora_r=lora_r).to(device)
    params = trainable_params(policy)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = max(1, math.ceil(len(pairs) / grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * 0.03))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1 + math.cos(math.pi * progress))

    order = list(range(len(pairs)))
    global_step = 0
    policy.train()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(order), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        window_loss = 0.0
        window_acc = 0.0
        window_margin = 0.0
        for position, index in enumerate(perm):
            pair = pairs[index]
            pi_w = sequence_logprob(policy, _batch(pair["chosen"], device))
            pi_l = sequence_logprob(policy, _batch(pair["rejected"], device))
            ref_w = reference[index, 0].to(device)
            ref_l = reference[index, 1].to(device)
            margin = beta * ((pi_w - pi_l) - (ref_w - ref_l))
            loss = -F.logsigmoid(margin)
            (loss / grad_accum).backward()
            accum += 1
            window_loss += float(loss.detach())
            window_acc += float(margin.detach() > 0)
            window_margin += float(margin.detach())
            last = position + 1 == len(perm)
            if accum < grad_accum and not last:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr_at(global_step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 10 == 0 or last:
                denom = max(accum, 1)
                message = (
                    f"[dpo] step {global_step}/{total_steps} "
                    f"loss={window_loss / denom:.4f} "
                    f"acc={window_acc / denom:.3f} "
                    f"margin={window_margin / denom:.3f} "
                    f"lr={lr_at(global_step):.2e}"
                )
                print(message, flush=True)
                if swan is not None:
                    swan.log(
                        {
                            "train/loss": window_loss / denom,
                            "train/acc": window_acc / denom,
                            "train/margin": window_margin / denom,
                            "train/lr": lr_at(global_step),
                        },
                        step=global_step,
                    )
            accum = 0
            window_loss = 0.0
            window_acc = 0.0
            window_margin = 0.0

    policy.config.use_cache = True
    save_policy(str(Path(output_dir) / "policy"), policy, tokenizer)
    metrics = {"steps": global_step, "pairs": len(pairs), "beta": beta, "lr": lr, "epochs": epochs}
    print(f"[dpo] saved policy {metrics}", flush=True)
    return metrics
