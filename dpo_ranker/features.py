"""Hidden states cached while the sampling policy generates a candidate."""

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
    raise AttributeError("Cannot find decoder layers on the policy.")


class ResponseHiddenCache:
    """Record response-token states at one layer during cached decoding.

    The first forward with sequence length greater than one is prompt prefill
    and is ignored. Later forwards are the generated tokens.
    """

    def __init__(self, layer: nn.Module):
        self.steps: list[torch.Tensor] = []
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.size(1) != 1:
            return
        self.steps.append(hidden[:, 0, :].detach().float().cpu())

    def close(self) -> None:
        self._handle.remove()

    def pooled(self, lengths: torch.Tensor, pooling: str) -> torch.Tensor:
        if not self.steps:
            raise RuntimeError("No response hidden states were cached.")
        states = torch.stack(self.steps, dim=1)
        limit = min(states.size(1), int(lengths.max()))
        states = states[:, :limit]
        positions = torch.arange(limit).unsqueeze(0)
        valid = positions < lengths.cpu().unsqueeze(1)
        if pooling == "last":
            index = (lengths.cpu() - 1).clamp(min=0, max=limit - 1)
            return states[torch.arange(states.size(0)), index]
        valid = valid.unsqueeze(-1).to(states.dtype)
        return (states * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def response_lengths(sequences: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    tail = sequences[:, prompt_len:]
    is_eos = tail == eos_id
    has = is_eos.any(dim=1)
    first = is_eos.int().argmax(dim=1)
    return torch.where(has, first + 1, torch.full_like(first, tail.size(1))).clamp(min=1)


def layer_index(n_layers: int, depth: float) -> int:
    return min(n_layers - 1, max(0, int(round(depth * (n_layers - 1)))))
