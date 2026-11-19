"""Extract and grade a \\boxed{} answer."""

from __future__ import annotations


def _boxed(text: str) -> str | None:
    key = r"\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    index = start + len(key)
    depth = 1
    chars: list[str] = []
    while index < len(text) and depth:
        char = text[index]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth:
                chars.append(char)
        else:
            chars.append(char)
        index += 1
    return "".join(chars).strip() if depth == 0 else None


def _normalize(value: str) -> str:
    import re

    text = value.strip().replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace(",", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def grade_text(response: str, gold: str) -> int:
    """1 when the extracted \\boxed{} answer matches the reference."""
    pred = _boxed(response)
    if pred is None:
        return 0
    left, right = _normalize(pred), _normalize(gold)
    if left == right:
        return 1
    try:
        return int(abs(float(left) - float(right)) < 1e-6)
    except ValueError:
        return 0
