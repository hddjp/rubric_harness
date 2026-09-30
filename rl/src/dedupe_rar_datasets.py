#!/usr/bin/env python3
"""Find and optionally remove duplicate RaR-Medicine/Science questions.

The VERL parquet files contain the canonical ``extra_info.question`` field.
This script compares questions after Unicode normalisation, case-folding and
whitespace normalisation.  It reports exact-normalised duplicates both within
each domain and across the two domains.  Original files are never modified.

The script is intentionally runnable without arguments:

    python rl/src/dedupe_rar_datasets.py

It reads ``rl/data/verl`` and writes the deduplicated files to
``rl/data/verl-new``.  If a duplicate appears in train and an evaluation
split, the evaluation row is kept and the train row is removed. Cross-domain
clashes otherwise keep the Medicine row.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "verl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "verl-newnew"
DEFAULT_DOMAINS = ("RaR-Medicine", "RaR-Science")
_SPACE_RE = re.compile(r"\s+")


@dataclass
class Row:
    domain: str
    split: str
    index: int
    record: dict[str, Any]

    @property
    def question(self) -> str:
        extra = self.record.get("extra_info")
        if isinstance(extra, dict) and extra.get("question") is not None:
            return str(extra["question"])
        value = self.record.get("question")
        return "" if value is None else str(value)


def normalise_question(text: str) -> str:
    """Return a conservative key: NFKC + casefold + collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    return _SPACE_RE.sub(" ", text).strip().casefold()


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Reading parquet requires the 'datasets' package; install it or use the "
            "same environment as the VERL data preparation scripts."
        ) from exc
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in dataset]


def load_rows(input_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for domain in DEFAULT_DOMAINS:
        domain_dir = input_dir / domain
        for path in sorted(domain_dir.glob("*.parquet")):
            for index, record in enumerate(_load_parquet(path)):
                rows.append(Row(domain, path.stem, index, record))
    if not rows:
        raise SystemExit(f"No parquet files found below {input_dir}")
    return rows


def group_by_key(rows: Iterable[Row]) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        key = normalise_question(row.question)
        if key:
            groups[key].append(row)
    return dict(groups)


def _location(row: Row) -> str:
    return f"{row.domain}/{row.split}[{row.index}]"


def summarise(rows: list[Row]) -> dict[str, Any]:
    groups = group_by_key(rows)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    cross_groups = [
        group for group in duplicate_groups
        if len({row.domain for row in group}) > 1
    ]
    within_groups = [
        group for group in duplicate_groups
        if len({row.domain for row in group}) == 1
    ]
    return {
        "n_rows": len(rows),
        "n_unique_questions": len(groups),
        "n_duplicate_rows": sum(len(group) - 1 for group in duplicate_groups),
        "n_duplicate_groups": len(duplicate_groups),
        "n_cross_domain_groups": len(cross_groups),
        "n_cross_domain_duplicate_rows": sum(len(group) - 1 for group in cross_groups),
        "n_within_domain_groups": len(within_groups),
        "duplicate_groups": [
            {
                "question": group[0].question,
                "locations": [_location(row) for row in group],
            }
            for group in sorted(duplicate_groups, key=lambda group: _location(group[0]))
        ],
    }


def write_deduped(rows: list[Row], output_dir: Path, keep_domain: str) -> dict[str, int]:
    """Write parquet files, preferring evaluation rows over train rows."""
    groups = group_by_key(rows)
    kept_ids: set[tuple[str, str, int]] = set()
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda row: (
                1 if row.split == "train" else 0,
                0 if row.domain == keep_domain else 1,
                _location(row),
            ),
        )
        kept = ordered[0]
        kept_ids.add((kept.domain, kept.split, kept.index))

    # Import lazily so report-only mode does not need the Dataset writer path.
    from datasets import Dataset

    counts: dict[str, int] = {}
    for domain in DEFAULT_DOMAINS:
        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        for split in sorted({row.split for row in rows if row.domain == domain}):
            selected = [
                row.record for row in rows
                if row.domain == domain
                and row.split == split
                and (row.domain, row.split, row.index) in kept_ids
            ]
            destination = domain_dir / f"{split}.parquet"
            Dataset.from_list(selected).to_parquet(str(destination))
            counts[f"{domain}/{split}"] = len(selected)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="write deduplicated parquet files here",
    )
    parser.add_argument(
        "--keep-domain", choices=DEFAULT_DOMAINS, default="RaR-Medicine",
        help="domain retained when the same question appears in both domains",
    )
    parser.add_argument("--report", type=Path, default=None,
                        help="optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input_dir)
    report = summarise(rows)
    report["input_dir"] = str(args.input_dir)
    report["normalisation"] = "Unicode NFKC + casefold + collapsed whitespace"
    print(f"总数据条数: {report['n_rows']}")
    print(f"去重后数据条数: {report['n_unique_questions']}")
    print(f"重复条数: {report['n_duplicate_rows']}")
    print(f"重复组数: {report['n_duplicate_groups']}")
    print(f"跨 Medicine/Science 重复组数: {report['n_cross_domain_groups']}")
    print(f"跨 Medicine/Science 重复条数: {report['n_cross_domain_duplicate_rows']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = write_deduped(rows, args.output_dir, args.keep_domain)
    print(f"去重数据已写入: {args.output_dir}")
    print(json.dumps({"written_rows": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
