#!/data/sty/miniconda3/envs/ssttyy/bin/python
"""统计 judge 日志中 request+response 达到 10000 tokens 的比例。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator
from tqdm import tqdm

# The serving setup already uses this environment for Qwen/vLLM.
SSTTYY_SITE = Path("/data/sty/miniconda3/envs/ssttyy/lib/python3.12/site-packages")
if SSTTYY_SITE.is_dir() and str(SSTTYY_SITE) not in sys.path:
    sys.path.insert(0, str(SSTTYY_SITE))

from transformers import AutoTokenizer


LOG_PATH = Path(__file__).resolve().parents[1] / "log.json"
TOKENIZER_NAME = "Qwen/Qwen3.5-4B"
LIMIT = 10_000


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """读取多个连续的、以空行分隔的 JSON 对象。"""
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return
        value, position = decoder.raw_decode(text, position)
        if not isinstance(value, dict):
            raise ValueError(f"日志中发现非对象记录，位置：{position}")
        yield value


def _count_rendered_messages(tokenizer, messages: list[dict[str, Any]]) -> int:
    """渲染并编码消息，返回 token 数而不是 BatchEncoding 字段数。"""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def request_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    request = record.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    return messages if isinstance(messages, list) else []


def response_message(record: dict[str, Any]) -> dict[str, Any] | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    message = response.get("message")
    if isinstance(message, dict):
        return message
    # 兼容 response 仍保留完整 OpenAI choices 外层的日志格式。
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return None


def count_request_tokens(tokenizer, messages: list[dict[str, Any]]) -> int:
    """统计 request.messages 本身的 token 数。"""
    return _count_rendered_messages(tokenizer, messages)


def count_response_tokens(
    tokenizer,
    request_messages_value: list[dict[str, Any]],
    message: dict[str, Any],
) -> int:
    """统计追加 response.message 后新增的 token，包括 assistant 标记。"""
    request_count = count_request_tokens(tokenizer, request_messages_value)
    full_count = _count_rendered_messages(
        tokenizer,
        request_messages_value + [message],
    )
    return max(full_count - request_count, 0)


def main() -> None:
    if not LOG_PATH.is_file():
        raise SystemExit(f"找不到日志文件：{LOG_PATH}")

    print(f"正在加载 tokenizer：{TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Each record is separated by a blank line in this pretty-printed log.
    estimated_total = LOG_PATH.read_text(encoding="utf-8").count("\n\n")

    total = 0
    reached = 0
    request_sum = 0
    response_sum = 0
    combined_sum = 0
    combined_min = None
    combined_max = None

    for record in tqdm(
        iter_records(LOG_PATH),
        total=estimated_total or None,
        desc="分析记录",
        unit="record",
    ):
        messages = request_messages(record)
        request_count = count_request_tokens(tokenizer, messages)
        message = response_message(record)
        response_count = count_response_tokens(tokenizer, messages, message) if message else 0
        combined_count = request_count + response_count

        total += 1
        reached += combined_count >= LIMIT
        request_sum += request_count
        response_sum += response_count
        combined_sum += combined_count
        combined_min = combined_count if combined_min is None else min(combined_min, combined_count)
        combined_max = combined_count if combined_max is None else max(combined_max, combined_count)

    if total == 0:
        print("没有找到任何记录。")
        return

    print(f"日志：{LOG_PATH}")
    print(f"记录数：{total}")
    print(f"阈值：request + response >= {LIMIT} tokens")
    print(f"达到阈值：{reached}/{total} ({reached / total:.2%})")
    print(f"request 平均 tokens：{request_sum / total:.1f}")
    print(f"response 平均 tokens：{response_sum / total:.1f}")
    print(f"合计平均 tokens：{combined_sum / total:.1f}")
    print(f"合计范围：{combined_min}..{combined_max}")


if __name__ == "__main__":
    main()
