"""Sample candidates and keep the hidden states produced by that generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Problem
from .features import ResponseHiddenCache, decoder_layers, layer_index, response_lengths


@dataclass
class Candidate:
    qid: int
    prompt: str
    text: str
    label: int | None
    feature: torch.Tensor


def format_prompt(tokenizer, question: str) -> str:
    message = question + "\nPut the final answer in \\boxed{}."
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return message


@torch.no_grad()
def sample_problem(
    model,
    tokenizer,
    problem: Problem,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    layer_depth: float,
    pooling: str,
    verifier,
) -> list[Candidate]:
    model.eval()
    layers = decoder_layers(model)
    layer = layers[layer_index(len(layers), layer_depth)]
    prompt = format_prompt(tokenizer, problem.question)
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_len = int(encoded.input_ids.shape[1])
    input_ids = encoded.input_ids.repeat(k, 1).to(model.device)
    attention = torch.ones_like(input_ids)
    cache = ResponseHiddenCache(layer)
    try:
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        cache.close()
    lengths = response_lengths(sequences, prompt_len, int(tokenizer.eos_token_id))
    features = cache.pooled(lengths, pooling)
    texts = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
    candidates = []
    for index, text in enumerate(texts):
        label = int(verifier(text, problem.answer)) if problem.labeled else None
        candidates.append(Candidate(problem.qid, problem.question, text, label, features[index].cpu()))
    return candidates
