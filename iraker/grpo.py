"""GRPO-style policy gradient, two advantage sources.

  * ranker advantage (``grpo_step``): the frozen ranker scores each completion and
    the z-scored ranker score inside a question group is the advantage.
  * verifier advantage (``grpo_step_verifier``): the standard GRPO baseline where
    the advantage is the z-scored 0/1 correctness bit.

Both share the same PPO-clipped surrogate and KL penalty, so the only difference
between the two runs is where the advantage comes from.
"""

from __future__ import annotations

import torch


def group_advantages(scores: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True).clamp(min=eps)
    return (scores - mean) / std


def token_logprobs(model, input_ids: torch.Tensor, attention: torch.Tensor, prompt_lens: torch.Tensor, micro: int = 0):
    """Per-token logprob of the completion part, plus a mask of real tokens.

    ``micro`` chunks the batch over rows so a (batch, 4096, vocab) fp32 logits
    tensor never has to live in 24GB at once. Chunking is over rows so the
    prompt_lens mask is unchanged.
    """
    if micro and micro < input_ids.size(0):
        logps, masks = [], []
        for start in range(0, input_ids.size(0), micro):
            logp, mask = token_logprobs(
                model,
                input_ids[start:start + micro],
                attention[start:start + micro],
                prompt_lens[start:start + micro],
                micro=0,
            )
            logps.append(logp)
            masks.append(mask)
        width = input_ids.size(1) - 1
        logp = torch.zeros(input_ids.size(0), width, dtype=torch.float32, device=input_ids.device)
        mask = torch.zeros(input_ids.size(0), width, dtype=torch.float32, device=input_ids.device)
        row = 0
        for logp_chunk, mask_chunk in zip(logps, masks):
            n = logp_chunk.size(0)
            logp[row:row + n, : logp_chunk.size(1)] = logp_chunk
            mask[row:row + n, : mask_chunk.size(1)] = mask_chunk
            row += n
        return logp, mask

    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, input_ids.size(1), device=input_ids.device).unsqueeze(0)
    mask = (positions >= prompt_lens.unsqueeze(1)) & (attention[:, 1:] > 0)
    return logp, mask.float()


def _encode_batch(rows: list[dict], tokenizer, max_length: int, device):
    seqs, prompt_lens = [], []
    for row in rows:
        p_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        r_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        seq = (p_ids + r_ids)[:max_length]
        seqs.append(seq)
        prompt_lens.append(min(len(p_ids), len(seq)))

    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
    for index, seq in enumerate(seqs):
        input_ids[index, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention[index, : len(seq)] = 1
    prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    return input_ids, attention, prompt_lens_t


def _loss_from_advantage(
    policy,
    reference,
    rows: list[dict],
    advantage: torch.Tensor,
    tokenizer,
    *,
    clip_eps: float,
    kl_coef: float,
    max_length: int,
    device,
    logprob_micro: int = 8,
) -> dict:
    input_ids, attention, prompt_lens = _encode_batch(rows, tokenizer, max_length, device)

    with torch.no_grad():
        old_logp, mask = token_logprobs(policy, input_ids, attention, prompt_lens, micro=logprob_micro)
        ref_logp, _ = token_logprobs(reference, input_ids, attention, prompt_lens, micro=logprob_micro)

    new_logp, _ = token_logprobs(policy, input_ids, attention, prompt_lens, micro=logprob_micro)
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantage.unsqueeze(1)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage.unsqueeze(1)
    policy_loss = -(torch.min(unclipped, clipped) * mask).sum() / mask.sum().clamp(min=1.0)

    kl = (torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1.0).clamp(min=0)
    kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1.0)
    loss = policy_loss + kl_coef * kl_loss
    return {
        "loss": loss,
        "policy_loss": float(policy_loss.detach()),
        "kl": float(kl_loss.detach()),
        "mean_adv": float(advantage.mean()),
    }


def grpo_step(
    policy,
    reference,
    ranker,
    batch_rows: list[dict],
    tokenizer,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    max_length: int = 4096,
    device: torch.device,
) -> dict:
    """Advantage = z-scored ranker score inside each question group."""
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(batch_rows):
        groups.setdefault(int(row["qid"]), []).append(index)

    advantage = torch.zeros(len(batch_rows), device=device)
    ranker.eval()
    with torch.no_grad():
        for members in groups.values():
            if len(members) < 2:
                continue
            inst = torch.stack([batch_rows[i]["_inst"] for i in members]).to(device)
            resp = torch.stack([batch_rows[i]["_resp"] for i in members]).to(device)
            scores = ranker(inst, resp.unsqueeze(1)).squeeze(-1)
            advantage[members] = group_advantages(scores.unsqueeze(0)).squeeze(0)

    return _loss_from_advantage(
        policy,
        reference,
        batch_rows,
        advantage,
        tokenizer,
        clip_eps=clip_eps,
        kl_coef=kl_coef,
        max_length=max_length,
        device=device,
    )


def grpo_step_verifier(
    policy,
    reference,
    batch_rows: list[dict],
    tokenizer,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    max_length: int = 4096,
    device: torch.device,
) -> dict:
    """Advantage = z-scored correctness bit written into each row as ``_adv``."""
    advantage = torch.tensor([float(row["_adv"]) for row in batch_rows], device=device)
    return _loss_from_advantage(
        policy,
        reference,
        batch_rows,
        advantage,
        tokenizer,
        clip_eps=clip_eps,
        kl_coef=kl_coef,
        max_length=max_length,
        device=device,
    )
