#!/usr/bin/env python3
"""Download and normalize the two downstream evaluation datasets.

HealthBench is loaded from OpenAI's public Hugging Face dataset and GPQA from
the original GPQA dataset. The paper's private-to-the-paper HealthBench-1k ID
file can be supplied with --healthbench-ids; without it, a deterministic
prompt_id-sorted 1,000-example subset is written and marked approximate.
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from pathlib import Path
from typing import Any

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_healthbench(output: Path, limit: int, ids_path: Path | None) -> None:
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        repo_id="openai/healthbench",
        filename="2025-05-07-06-14-12_oss_eval.jsonl",
        repo_type="dataset",
    )
    rows = []
    with open(source, encoding="utf-8") as handle:
      source_rows = [json.loads(line) for line in handle if line.strip()]
    for row in source_rows:
        prompt_id = str(row["prompt_id"])
        rubrics = row["rubrics"]
        if isinstance(rubrics, str):
            rubrics = json.loads(rubrics)
        rows.append({
            "id": prompt_id,
            "messages": row["prompt"],
            "rubrics": rubrics,
            "tags": row.get("example_tags", []),
        })
    selected_by = "prompt_id_sorted_fallback"
    if ids_path:
        raw = ids_path.read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.splitlines()
        if isinstance(parsed, dict):
            parsed = parsed.get("ids", parsed.get("prompt_ids", []))
        if not isinstance(parsed, list):
            raise ValueError(f"{ids_path}: expected a JSON list/object or one ID per line")
        wanted = {str(item.get("id", item) if isinstance(item, dict) else item) for item in parsed}
        rows = [row for row in rows if row["id"] in wanted]
        selected_by = str(ids_path)
    else:
        rows.sort(key=lambda row: row["id"])
        rows = rows[:limit]
    if ids_path and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError("HealthBench selection is empty")
    _write_jsonl(output / "healthbench_1k.jsonl", rows)
    (output / "healthbench_selection.json").write_text(
        json.dumps({"count": len(rows), "selected_by": selected_by}, indent=2),
        encoding="utf-8",
    )
    print(f"{output / 'healthbench_1k.jsonl'}: {len(rows)} examples ({selected_by})")


def download_gpqa(output: Path, config: str) -> None:
    try:
        from datasets import load_dataset

        source_name = "Idavidrein/gpqa"
        dataset = load_dataset("Idavidrein/gpqa", config, split="train")
    except Exception as original_error:
        # The original GPQA repository is gated in some environments. Download
        # the public mirror's parquet directly so this fallback does not depend
        # on datasets' schema inference.
        source_name = "extraordinarylab/gpqa-diamond"
        try:
            import pyarrow.parquet as pq

            url = "https://huggingface.co/datasets/extraordinarylab/gpqa-diamond/resolve/main/data/test-00000-of-00001.parquet?download=true"
            output.mkdir(parents=True, exist_ok=True)
            parquet_path = output / ".gpqa_diamond.download.parquet"
            request = urllib.request.Request(url, headers={"User-Agent": "rubric-harness-eval"})
            with urllib.request.urlopen(request, context=ssl._create_unverified_context(), timeout=120) as response:
                parquet_path.write_bytes(response.read())
            dataset = pq.read_table(parquet_path).to_pylist()
            parquet_path.unlink()
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not load GPQA from Idavidrein/gpqa ({original_error}) "
                f"or fallback {source_name} ({fallback_error})"
            ) from fallback_error
    rows = []
    for index, row in enumerate(dataset):
        if "choices" in row:
            choices = [str(choice) for choice in row["choices"]]
            answer_index = int(row.get("answer_index", 0))
            answer = str(row.get("answer", choices[answer_index]))
            question = str(row["question"])
            question_id = row.get("id", index)
        else:
            choices = [
                str(row.get("Correct Answer", "")),
                str(row.get("Incorrect Answer 1", "")),
                str(row.get("Incorrect Answer 2", "")),
                str(row.get("Incorrect Answer 3", "")),
            ]
            answer_index = 0
            answer = choices[0]
            question = str(row["Question"])
            question_id = row.get("Question ID", index)
        rows.append({
            "id": str(question_id),
            "question": question,
            "choices": choices,
            "answer": answer,
            "answer_index": answer_index,
            "source": source_name,
        })
    _write_jsonl(output / "gpqa_diamond.jsonl", rows)
    print(f"{output / 'gpqa_diamond.jsonl'}: {len(rows)} examples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default="./eval")
    parser.add_argument("--healthbench-limit", type=int, default=1000)
    parser.add_argument("--healthbench-ids", type=Path, default=None)
    parser.add_argument("--gpqa-config", default="gpqa_diamond")
    parser.add_argument("--skip-healthbench", action="store_true")
    parser.add_argument("--skip-gpqa", action="store_true")
    args = parser.parse_args()
    if not args.skip_healthbench:
        download_healthbench(args.output_dir, args.healthbench_limit, args.healthbench_ids)
    if not args.skip_gpqa:
        download_gpqa(args.output_dir, args.gpqa_config)


if __name__ == "__main__":
    main()
