"""Extract the probe-layer features the ranker consumes.

For every candidate we need two vectors at the same hidden layer:
  * instruction feature  - hidden state of the final prompt token
  * response feature     - hidden state of the final response token

Sampling is done separately (vLLM). This module only replays prompt+candidate
through the model and reads one layer, which is much cheaper than hooking the
decoder during generation and gives identical vectors.
"""

from __future__ import annotations

import torch
from torch import nn


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = base.model if hasattr(base, "model") else base
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return inner.model.layers
    if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
        return inner.decoder.layers
    raise AttributeError("Cannot find decoder layers on the policy.")


class LayerTap:
    """Read one decoder layer's output for the whole batch."""

    def __init__(self, layer: nn.Module):
        self.hidden: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.hidden = hidden.detach()

    def close(self) -> None:
        self._handle.remove()


def _last_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.to(dtype=torch.long).sum(dim=-1) - 1


@torch.no_grad()
def encode_batch(model, tokenizer, prompts: list[str], responses: list[str], dtype: torch.dtype):
    """Return (instruction, response) features plus next-token logprobs of each response.

    Prompt and response are tokenized separately: the prompt feature is the last
    prompt token, the response feature is the last response token, and the
    prompt is never followed by a chat template.
    """
    device = next(model.parameters()).device
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    response_ids = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    sequences = [p + r for p, r in zip(prompt_ids, response_ids)]
    prompt_lens = torch.tensor([len(p) for p in prompt_ids], device=device)
    max_len = max(len(s) for s in sequences)

    ignore = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((len(sequences), max_len), ignore, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for row, seq in enumerate(sequences):
        input_ids[row, : len(seq)] = torch.tensor(seq, device=device)
        attention[row, : len(seq)] = 1

    layers = decoder_layers(model)
    return layers, prompt_lens, input_ids, attention


@torch.no_grad()
def extract_features(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    layer_index: int,
    batch_size: int = 4,
    max_length: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch over (prompt, response) and return (instruction, response) features."""
    device = next(model.parameters()).device
    layers = decoder_layers(model)
    tap = LayerTap(layers[layer_index])
    inst_parts: list[torch.Tensor] = []
    resp_parts: list[torch.Tensor] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk_p = prompts[start:start + batch_size]
            chunk_r = responses[start:start + batch_size]
            p_ids = [tokenizer.encode(p, add_special_tokens=False) for p in chunk_p]
            r_ids = [tokenizer.encode(r, add_special_tokens=False) for r in chunk_r]
            seqs = []
            for p, r in zip(p_ids, r_ids):
                seq = p + r
                if len(seq) > max_length:
                    seq = seq[:max_length]
                seqs.append(seq)
            prompt_lens = [min(len(p), len(s)) for p, s in zip(p_ids, seqs)]
            lengths = [len(s) for s in seqs]

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            max_len = max(lengths)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
            attention = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
            for row, seq in enumerate(seqs):
                input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                attention[row, : len(seq)] = 1

            model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            hidden = tap.hidden
            if hidden is None:
                raise RuntimeError("Layer tap captured nothing; check the layer index.")

            rows = torch.arange(len(seqs), device=device)
            inst_parts.append(hidden[rows, torch.tensor(prompt_lens, device=device) - 1].float().cpu())
            resp_parts.append(hidden[rows, torch.tensor(lengths, device=device) - 1].float().cpu())
    finally:
        tap.close()

    return torch.cat(inst_parts, dim=0), torch.cat(resp_parts, dim=0)


@torch.no_grad()
def score_responses(model, tokenizer, prompts: list[str], responses: list[str], max_length: int = 4096):
    """Mean next-token logprob of each response under the model."""
    device = next(model.parameters()).device
    out = []
    for prompt, response in zip(prompts, responses):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(response, add_special_tokens=False)
        ids = (p_ids + r_ids)[:max_length]
        if len(ids) <= len(p_ids):
            out.append(float("-inf"))
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
        logp = torch.log_softmax(logits[0, len(p_ids) - 1: -1], dim=-1)
        targets = torch.tensor(ids[len(p_ids):], dtype=torch.long, device=device)
        out.append(float(logp.gather(-1, targets[:, None]).mean()))
    return out


def main() -> None:
    import argparse
    from pathlib import Path

    from policy.policy import load_policy, load_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    from generator.utils import read_jsonl

    rows = read_jsonl(args.candidates)
    if not rows:
        raise SystemExit(f"no candidates in {args.candidates}")

    tokenizer = load_tokenizer(args.model)
    model = load_policy(args.model, lora_r=0, dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    n_layers = len(decoder_layers(model))
    layer = min(n_layers - 1, max(0, int(round(args.depth * (n_layers - 1)))))
    print(f"[stage2] layers={n_layers} probe_layer={layer} ({args.depth:.0%} depth)", flush=True)

    inst_all, resp_all = [], []
    done = 0
    for row in rows:
        prompt = row["prompt"]
        responses = row["responses"]
        inst, resp = extract_features(
            model,
            tokenizer,
            [prompt] * len(responses),
            responses,
            layer_index=layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        inst_all.append(inst[0])
        resp_all.append(resp)
        done += len(responses)
        if done % 200 == 0:
            print(f"[stage2] {done} responses", flush=True)

    payload = {
        "groups": [
            {
                "qid": int(row["qid"]),
                "prompt": row["prompt"],
                "response": resp,
                "instruction": inst,
                "labels": torch.tensor(row["labels"], dtype=torch.float32),
                "responses": row["responses"],
                "has_pos": bool(any(row["labels"])),
                "has_neg": bool(any(1 - label for label in row["labels"])),
            }
            for row, inst, resp in zip(rows, inst_all, resp_all)
        ],
        "probe_layer": layer,
        "depth": args.depth,
        "d_in": int(inst_all[0].numel()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"[stage2] saved {out} groups={len(payload['groups'])} d_in={payload['d_in']}", flush=True)


if __name__ == "__main__":
    main()
