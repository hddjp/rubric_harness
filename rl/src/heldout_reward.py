"""Rule-based scoring for held-out validation sets mixed into VERL validation."""

from __future__ import annotations

import re

_CHOICE_RE = re.compile(
    r"(?:\\boxed\s*\{\s*\(?([A-D])\)?\s*\}|(?:final answer|answer)\s*(?:is)?\s*[:：]?\s*\(?([A-D])\)?\b)",
    re.IGNORECASE,
)


def extract_choice(text: str) -> str | None:
    matches = _CHOICE_RE.findall(text or "")
    if not matches:
        return None
    a, b = matches[-1]
    return (a or b).upper()


def gpqa_score(solution_str: str, ground_truth: str) -> dict:
    """1.0 iff the last boxed/stated option letter equals the gold letter."""
    predicted = extract_choice(solution_str)
    # Same keys as the rubric rewards: VERL merges reward_extra_info across rows.
    return {"score": float(predicted == ground_truth), "judge_failed": 0.0, "judge_error": ""}
