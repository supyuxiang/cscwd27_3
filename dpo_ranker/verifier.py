"""Exact-match verifier for a final answer already stored beside the problem."""

from __future__ import annotations

import re


def _boxed(text: str) -> str | None:
    key = r"\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    chars: list[str] = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return "".join(chars).strip() if depth == 0 else None


def extract_answer(text: str) -> str | None:
    boxed = _boxed(text)
    if boxed:
        return boxed
    hashes = re.findall(r"####\s*(.+)", text)
    if hashes:
        return hashes[-1].strip()
    return None


def normalize(text: str) -> str:
    value = text.strip()
    value = value.replace("$", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = value.replace(",", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value.lower()


def answers_match(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    pred = normalize(prediction)
    ref = normalize(gold)
    if pred == ref:
        return True
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return False


def is_correct(response: str, gold: str) -> bool:
    return answers_match(extract_answer(response), gold)
