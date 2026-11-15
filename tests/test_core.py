"""Quick offline checks: grading, ranker shapes, loss math. Run: python -m tests.test_core"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from iraker.core import build_prompt, grade_text, probe_layer_index
from iraker.grpo import group_advantages
from iraker.ranker_model import ListwiseRanker, PointwiseRanker, count_parameters, pairwise_accuracy


def test_prompt_matches_eval():
    prompt = build_prompt("What is 1+1?")
    assert prompt.startswith("You are a math expert.\nYou must answer in English.\n\n")
    assert prompt.endswith("within \\boxed{}.")
    assert "Question: What is 1+1?" in prompt


def test_grade():
    assert grade_text(r"so the answer is \boxed{6}", "6") == 1
    assert grade_text(r"\boxed{\frac{1}{2}}", "\\frac{1}{2}") == 1
    assert grade_text(r"\boxed{7}", "6") == 0
    assert grade_text("no box here", "6") == 0


def test_layer_index():
    assert probe_layer_index(0.6, 36) == 21
    assert probe_layer_index(0.0, 36) == 0
    assert probe_layer_index(1.0, 36) == 35


def test_ranker_shapes():
    d, k, batch = 2048, 8, 3
    inst = torch.randn(batch, d)
    resp = torch.randn(batch, k, d)
    for cls in (PointwiseRanker, ListwiseRanker):
        model = cls(d_in=d)
        logits = model(inst, resp)
        assert logits.shape == (batch, k)
        assert count_parameters(model) < 500_000
        print(f"{cls.__name__}: {count_parameters(model)} params")


def test_advantage():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    adv = group_advantages(scores)
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert adv[0, 2] > adv[0, 0]


def test_pairwise_accuracy():
    logits = torch.tensor([[0.9, 0.1, 0.5]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    assert pairwise_accuracy(logits, labels) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
