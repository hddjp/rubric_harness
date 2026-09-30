from __future__ import annotations

import json
import re


def extract_json(text: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        starts = [i for i in (candidate.find("["), candidate.find("{")) if i >= 0]
        if not starts:
            raise ValueError(f"judge 未返回 JSON：{text[:240]}")
        start = min(starts)
        end = max(candidate.rfind("]"), candidate.rfind("}")) + 1
        return json.loads(candidate[start:end])


def first_number(text: str, low: float, high: float) -> float:
    match = re.search(r"(?<![\d.])(10(?:\.0+)?|[0-9](?:\.\d+)?)(?![\d.])", text)
    if not match:
        raise ValueError(f"judge 未返回数值分数：{text[:240]}")
    return min(high, max(low, float(match.group(1))))

