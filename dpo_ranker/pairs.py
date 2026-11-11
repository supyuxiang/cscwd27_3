"""Turn verifier bits and ranker scores into a fixed budget of DPO pairs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredCandidate:
    text: str
    label: int | None
    score: float


@dataclass
class PreferencePair:
    qid: int
    prompt: str
    chosen: str
    rejected: str
    source: str
    margin: float

    @property
    def length_gap(self) -> int:
        return abs(len(self.chosen) - len(self.rejected))


def _length_ok(chosen: str, rejected: str, delta_len: int) -> bool:
    return abs(len(chosen) - len(rejected)) <= delta_len


def pairs_for_prompt(
    qid: int,
    prompt: str,
    candidates: list[ScoredCandidate],
    kind: str,
    tau: float,
    delta_len: int,
) -> PreferencePair | None:
    if kind == "informative":
        correct = [item for item in candidates if item.label == 1]
        wrong = [item for item in candidates if item.label == 0]
        if not correct or not wrong:
            return None
        chosen = max(correct, key=lambda item: item.score)
        rejected = max(wrong, key=lambda item: item.score)
        source = "hard_negative"
    elif kind in {"all_correct", "reference_free"}:
        chosen = max(candidates, key=lambda item: item.score)
        rejected = min(candidates, key=lambda item: item.score)
        source = "saturated" if kind == "all_correct" else "reference_free"
    else:
        return None
    margin = chosen.score - rejected.score
    if kind != "informative" and margin < tau:
        return None
    if not _length_ok(chosen.text, rejected.text, delta_len):
        return None
    if chosen.text == rejected.text:
        return None
    return PreferencePair(qid, prompt, chosen.text, rejected.text, source, margin)


def select_budget(pairs: list[PreferencePair], budget: int, ranker_fraction: float) -> list[PreferencePair]:
    if budget <= 0:
        return []
    ranker_cap = int(budget * ranker_fraction)
    informed = sorted((pair for pair in pairs if pair.source == "hard_negative"), key=lambda pair: pair.margin, reverse=True)
    ranker = sorted((pair for pair in pairs if pair.source != "hard_negative"), key=lambda pair: pair.margin, reverse=True)
    chosen = ranker[:ranker_cap]
    room = budget - len(chosen)
    chosen.extend(informed[:room])
    if len(chosen) < budget:
        used = {(pair.qid, pair.chosen, pair.rejected) for pair in chosen}
        rest = [pair for pair in ranker[ranker_cap:] if (pair.qid, pair.chosen, pair.rejected) not in used]
        chosen.extend(rest[: budget - len(chosen)])
    return chosen[:budget]
