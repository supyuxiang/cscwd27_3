"""CPU checks for the verifier, the pair rules, and the ranker loss."""

import torch

from dpo_ranker.pairs import ScoredCandidate, pairs_for_prompt, select_budget
from dpo_ranker.ranker import fit_ranker, score_features
from dpo_ranker.verifier import answers_match, extract_answer, is_correct


def test_verifier_reads_boxed_and_hash():
    assert extract_answer(r"reason \boxed{2}") == "2"
    assert is_correct("#### 1,000", "1000")
    assert answers_match(r"\left(3\right)", "(3)")
    assert not is_correct("no answer here", "2")


def test_informative_pair_picks_the_highest_scoring_wrong_answer():
    candidates = [
        ScoredCandidate("right-easy", 1, 0.2),
        ScoredCandidate("right-hard", 1, 0.9),
        ScoredCandidate("wrong-easy", 0, 0.1),
        ScoredCandidate("wrong-hard", 0, 0.8),
    ]
    pair = pairs_for_prompt(1, "q", candidates, "informative", tau=10.0, delta_len=100)
    assert pair is not None
    assert pair.chosen == "right-hard"
    assert pair.rejected == "wrong-hard"
    assert pair.source == "hard_negative"


def test_saturated_pair_needs_a_margin_and_all_wrong_is_dropped():
    close = [ScoredCandidate("a", 1, 0.5), ScoredCandidate("b", 1, 0.55)]
    assert pairs_for_prompt(1, "q", close, "all_correct", tau=0.2, delta_len=100) is None
    wide = [ScoredCandidate("a", None, 1.0), ScoredCandidate("b" * 5, None, 0.0)]
    kept = pairs_for_prompt(1, "q", wide, "reference_free", tau=0.5, delta_len=100)
    assert kept is not None and kept.source == "reference_free"
    wrong = [ScoredCandidate("a", 0, 1.0), ScoredCandidate("b", 0, 0.0)]
    assert pairs_for_prompt(1, "q", wrong, "all_wrong", tau=0.0, delta_len=100) is None


def test_budget_keeps_verifier_pairs_from_being_crowded_out():
    from dpo_ranker.pairs import PreferencePair

    pairs = [
        PreferencePair(i, "q", "c", "r", "reference_free", 10.0 - i) for i in range(6)
    ] + [PreferencePair(100, "q", "c", "r", "hard_negative", 0.1)]
    selected = select_budget(pairs, budget=4, ranker_fraction=0.5)
    assert len(selected) == 4
    assert sum(pair.source == "hard_negative" for pair in selected) == 1
    assert sum(pair.source == "reference_free" for pair in selected) == 3


def test_ranker_separates_planted_pairs():
    torch.manual_seed(0)
    positive = [torch.tensor([1.0, 0.0]) + 0.01 * torch.randn(2) for _ in range(32)]
    negative = [torch.tensor([0.0, 1.0]) + 0.01 * torch.randn(2) for _ in range(32)]
    model, stats = fit_ranker(list(zip(positive, negative)), 2, hidden=0, epochs=40, lr=0.1, weight_decay=0.0, device=torch.device("cpu"))
    scores = score_features(model, stats, torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]), torch.device("cpu"))
    assert float(scores[0]) > float(scores[1])
