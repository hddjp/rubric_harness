#!/usr/bin/env python3
"""Convert the authors' untouched Parquet files to VERL's canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset


def convert(source: Path, destination: Path, data_source: str, split: str, rubric_file: Path | None = None) -> None:
    dataset = load_dataset("parquet", data_files=str(source), split="train")
    generated_rubrics = {}
    if rubric_file is not None:
        with rubric_file.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                generated_rubrics[int(record["index"])] = record["rubrics"]

    def map_row(row: dict, index: int) -> dict:
        rubrics = generated_rubrics.get(index, row.get("rubric"))
        if rubric_file is not None and index not in generated_rubrics:
            raise ValueError(f"{rubric_file}: missing rubric for row {index}")
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError(f"{source}: row {index} has no rubric list")
        question = row["question"]
        reference = row["reference_answer"]
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "medicine" if data_source == "rar_medicine" else "science",
            "reward_model": {"style": "model", "ground_truth": reference},
            "extra_info": {
                "index": index,
                "split": split,
                "question": question,
                "reference_answer": reference,
                "question_source": row.get("question_source", ""),
                "rubrics": rubrics,
            },
        }

    converted = dataset.map(map_row, with_indices=True, remove_columns=dataset.column_names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(str(destination))
    print(f"{source} -> {destination}: {len(converted)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/verl"))
    parser.add_argument(
        "--rubrics-dir",
        type=Path,
        default=None,
        help="Use generated rubric JSONL files from this directory instead of released rubrics.",
    )
    args = parser.parse_args()
    for dirname, source_name in (("RaR-Medicine", "rar_medicine"), ("RaR-Science", "rar_science")):
        for split in ("train", "val", "test"):
            convert(
                args.raw_dir / dirname / f"{split}-00000-of-00001.parquet",
                args.output_dir / dirname / f"{split}.parquet",
                source_name,
                split,
                args.rubrics_dir / dirname / f"{split}.jsonl" if args.rubrics_dir else None,
            )


if __name__ == "__main__":
    main()
