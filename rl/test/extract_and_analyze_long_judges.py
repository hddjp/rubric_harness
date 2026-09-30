#!/data/sty/miniconda3/envs/ssttyy/bin/python
"""抽取 finish_reason=length 的 judge 记录并分析其输出。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from analyze_judge_token_limit import (
    LOG_PATH,
    TOKENIZER_NAME,
    count_request_tokens,
    count_response_tokens,
    iter_records,
    request_messages,
    response_message,
)
from transformers import AutoTokenizer


OUTPUT_PATH = Path(__file__).resolve().parent / "judge_finish_length_records.json"


def expected_judge_items(messages: list[dict[str, Any]]) -> int | None:
    """从 user prompt 中读取要求 judge 返回的对象数量。"""
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        match = re.search(r"exactly\s+(\d+)\s+objects", content, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def response_content(message: dict[str, Any] | None) -> str:
    if not message or not isinstance(message.get("content"), str):
        return ""
    return message["content"]


def candidate_json_values(text: str) -> list[Any]:
    """找出 response 中完整可解析的 JSON 值，包括 fenced/嵌入 JSON。"""
    decoder = json.JSONDecoder()
    values = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (TypeError, json.JSONDecodeError):
            value = None
        if value is not None:
            values.append(value)

    for match in re.finditer(r"[\[]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            values.append(value)
    return values


def judge_json_status(
    content: str,
    expected_count: int | None,
    finish_reason: Any,
) -> dict[str, Any]:
    values = candidate_json_values(content)
    arrays = [value for value in values if isinstance(value, list)]
    exact_json = False
    complete_array = False
    actual_count = None

    try:
        exact_value = json.loads(content.strip())
        exact_json = isinstance(exact_value, list)
    except json.JSONDecodeError:
        pass

    matching = arrays
    if expected_count is not None:
        matching = [value for value in arrays if len(value) == expected_count]
    if matching:
        selected = matching[-1]
        actual_count = len(selected)
        complete_array = all(isinstance(item, dict) for item in selected)
    elif arrays:
        actual_count = len(arrays[-1])

    return {
        "exact_json_array": exact_json,
        "complete_expected_array": complete_array,
        "expected_items": expected_count,
        "actual_items": actual_count,
        "finish_reason": finish_reason,
        "possibly_truncated": finish_reason == "length",
    }


def main() -> None:
    if not LOG_PATH.is_file():
        raise SystemExit(f"找不到日志文件：{LOG_PATH}")

    print(f"正在加载 tokenizer：{TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    estimated_total = LOG_PATH.read_text(encoding="utf-8").count("\n\n")

    selected = []
    for record in tqdm(
        iter_records(LOG_PATH),
        total=estimated_total or None,
        desc="扫描记录",
        unit="record",
    ):
        response_obj = record.get("response")
        finish_reason = response_obj.get("finish_reason") if isinstance(response_obj, dict) else None
        if finish_reason != "length":
            continue

        messages = request_messages(record)
        message = response_message(record)
        request_count = count_request_tokens(tokenizer, messages)
        response_count = count_response_tokens(tokenizer, messages, message) if message else 0
        combined_count = request_count + response_count
        status = judge_json_status(
            response_content(message),
            expected_judge_items(messages),
            finish_reason,
        )
        selected.append({
            "request": record.get("request"),
            "response": response_obj,
            "request_tokens": request_count,
            "response_tokens": response_count,
            "combined_tokens": combined_count,
            "judge_json": status,
        })

    OUTPUT_PATH.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bins = Counter((item["response_tokens"] // 200) * 200 for item in selected)
    complete = sum(item["judge_json"]["complete_expected_array"] for item in selected)
    exact = sum(item["judge_json"]["exact_json_array"] for item in selected)
    truncated = sum(item["judge_json"]["possibly_truncated"] for item in selected)

    print(f"\n抽取文件：{OUTPUT_PATH}")
    print(f"抽取记录数：{len(selected)}")
    print("\n输出长度分布（response tokens，每 200 一档）：")
    for lower in sorted(bins):
        print(f"  {lower:5d}-{lower + 199:5d}: {bins[lower]}")

    if selected:
        print("\njudge JSON 分析：")
        print(f"  response 本身是完整 JSON array：{exact}/{len(selected)} ({exact / len(selected):.2%})")
        print(f"  包含符合期望数量的完整 judge JSON：{complete}/{len(selected)} ({complete / len(selected):.2%})")
        print(f"  finish_reason=length：{truncated}/{len(selected)} ({truncated / len(selected):.2%})")


if __name__ == "__main__":
    main()
