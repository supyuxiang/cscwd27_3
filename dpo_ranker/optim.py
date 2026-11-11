"""SFT initialization and one round of reference-fixed or moving-reference DPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pairs import PreferencePair
from .sample import format_prompt


def _token_ids(tokenizer, prompt: str, completion: str, max_length: int) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    return input_ids, min(len(prompt_ids), len(input_ids))


def _completion_logprob(model, tokenizer, prompt: str, completion: str, max_length: int) -> torch.Tensor:
    ids, prompt_len = _token_ids(tokenizer, prompt, completion, max_length)
    input_ids = torch.tensor([ids], device=model.device)
    logits = model(input_ids=input_ids).logits[:, :-1]
    target = input_ids[:, 1:]
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    response = torch.arange(token_logp.size(1), device=model.device) >= max(prompt_len - 1, 0)
    return token_logp[0, response].sum()


def dpo_loss(policy_pos, policy_neg, ref_pos, ref_neg, beta: float) -> torch.Tensor:
    logits = beta * ((policy_pos - ref_pos) - (policy_neg - ref_neg))
    return -F.logsigmoid(logits).mean()


def train_sft(model, tokenizer, rows: list[tuple[str, str]], epochs: int, lr: float, max_length: int) -> None:
    if not rows:
        return
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    model.train()
    for _ in range(epochs):
        for prompt, completion in rows:
            ids, prompt_len = _token_ids(tokenizer, prompt, completion, max_length)
            input_ids = torch.tensor([ids], device=model.device)
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def train_dpo(
    model,
    tokenizer,
    pairs: list[PreferencePair],
    beta: float,
    epochs: int,
    lr: float,
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    if not pairs:
        return {"loss": 0.0, "chosen_logratio": 0.0, "rejected_logratio": 0.0}
    reference = []
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            prompt = format_prompt(tokenizer, pair.prompt)
            reference.append(
                (
                    _completion_logprob(model, tokenizer, prompt, pair.chosen, max_length).detach(),
                    _completion_logprob(model, tokenizer, prompt, pair.rejected, max_length).detach(),
                )
            )
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    totals = []
    chosen_gap = []
    rejected_gap = []
    model.train()
    for _ in range(epochs):
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            refs = reference[start : start + batch_size]
            pos, neg, ref_pos, ref_neg = [], [], [], []
            for pair, (old_pos, old_neg) in zip(batch, refs):
                prompt = format_prompt(tokenizer, pair.prompt)
                pos.append(_completion_logprob(model, tokenizer, prompt, pair.chosen, max_length))
                neg.append(_completion_logprob(model, tokenizer, prompt, pair.rejected, max_length))
                ref_pos.append(old_pos)
                ref_neg.append(old_neg)
            loss = dpo_loss(torch.stack(pos), torch.stack(neg), torch.stack(ref_pos), torch.stack(ref_neg), beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
            chosen_gap.append(float((torch.stack(pos) - torch.stack(ref_pos)).mean().detach()))
            rejected_gap.append(float((torch.stack(neg) - torch.stack(ref_neg)).mean().detach()))
    return {
        "loss": sum(totals) / max(len(totals), 1),
        "chosen_logratio": sum(chosen_gap) / max(len(chosen_gap), 1),
        "rejected_logratio": sum(rejected_gap) / max(len(rejected_gap), 1),
    }


