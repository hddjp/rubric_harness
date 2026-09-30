#!/usr/bin/env python3
"""Plot the token-count distribution of response.message.content in rl/log.json.

The log is a sequence of pretty-printed JSON objects rather than one JSON array.
This script parses it incrementally, so it does not load the 200+ MB log at once.

Example:
    python rl/test/plot_response_token_counts.py \
        --tokenizer Qwen/Qwen3.8-27B \
        --bin-size 256
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "rl" / "log.json"
DEFAULT_PLOT = ROOT / "rl" / "test" / "response_token_counts.png"
DEFAULT_CSV = ROOT / "rl" / "test" / "response_token_counts.csv"


def iter_json_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Yield concatenated JSON objects from a file without reading it all at once."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position == len(buffer):
                    buffer = ""
                    break

                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("log.json 末尾存在无法解析的 JSON")
                    # Keep the incomplete object for the next chunk.
                    buffer = buffer[position:]
                    break

                if not isinstance(value, dict):
                    raise ValueError(f"发现非 JSON object 记录：{type(value).__name__}")
                yield value
                position = end

            if eof and not buffer:
                return


def response_content(record: dict[str, Any]) -> str | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None

    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]

    # Compatibility with an OpenAI choices-style response wrapper.
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.8-27B",
        help="用于分词的 Hugging Face tokenizer；应与服务端模型一致。",
    )
    parser.add_argument("--bin-size", type=int, default=256, help="每个条形区间的 token 宽度。")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_size <= 0:
        raise SystemExit("--bin-size 必须为正数")
    if not args.log.is_file():
        raise SystemExit(f"找不到日志文件：{args.log}")

    print(f"加载 tokenizer：{args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    counts: list[int] = []
    for index, record in enumerate(iter_json_objects(args.log), start=1):
        if args.max_records is not None and index > args.max_records:
            break
        content = response_content(record)
        if content is None:
            continue
        counts.append(len(tokenizer.encode(content, add_special_tokens=False)))

    if not counts:
        raise SystemExit("没有找到 response.message.content 文本")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_index", "response_token_count"])
        for index, count in enumerate(counts, start=1):
            writer.writerow([index, count])

    maximum = max(counts)
    upper = max(args.bin_size, math.ceil((maximum + 1) / args.bin_size) * args.bin_size)
    edges = list(range(0, upper + args.bin_size, args.bin_size))
    frequencies = [0] * (len(edges) - 1)
    for count in counts:
        bucket = min(count // args.bin_size, len(frequencies) - 1)
        frequencies[bucket] += 1

    labels = [f"{edges[i]}–{edges[i + 1] - 1}" for i in range(len(frequencies))]
    plt.figure(figsize=(14, 6))
    plt.bar(labels, frequencies, width=0.85)
    plt.xlabel("response.message token 数")
    plt.ylabel("样本数")
    plt.title(f"Response token 数分布（n={len(counts)}）")
    plt.xticks(rotation=60, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot, dpi=160)
    plt.close()

    ordered = sorted(counts)
    percentile = lambda p: ordered[min(len(ordered) - 1, int(len(ordered) * p))]
    print(f"记录数：{len(counts)}")
    print(f"min/mean/median/max：{min(counts)}/{sum(counts) / len(counts):.1f}/{percentile(0.5)}/{max(counts)}")
    print(f"图像：{args.plot}")
    print(f"明细：{args.csv}")


if __name__ == "__main__":
    main()
