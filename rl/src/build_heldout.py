#!/usr/bin/env python3
"""Build the held-out validation files used by rl/run_fsdp.sh.

    rl/data/heldout/RaR-Science_val500.parquet   fixed 500-question sample of RaR val (seed 0)
    rl/data/heldout/RaR-Medicine_val500.parquet
    rl/data/heldout/gpqa_diamond.parquet         198 GPQA-Diamond questions, choices shuffled per
                                                 question (the source lists the gold answer first)

Inputs: rl/data/verl-newnew/RaR-*/val.parquet and rl/eval/data/gpqa_diamond.jsonl
(`python rl/eval/download_eval_data.py --output-dir rl/eval/data --skip-healthbench`).
GPQA is not redistributed in this repo; build it locally.
"""

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RL = Path(__file__).resolve().parents[1]
OUT = RL / "data" / "heldout"


def build_gpqa() -> None:
    rows = []
    with open(RL / "eval" / "data" / "gpqa_diamond.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            order = list(range(4))
            random.Random(f"gpqa:{r['id']}").shuffle(order)
            choices = [r["choices"][k].strip() for k in order]
            gold = chr(65 + order.index(int(r["answer_index"])))
            labels = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(choices))
            prompt = (f"{r['question'].strip()}\n\n{labels}\n\nSolve the problem and end with "
                      "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.")
            rows.append({
                "data_source": "gpqa_diamond",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {"index": i, "split": "heldout", "question": prompt, "answer": gold},
            })
    pq.write_table(pa.Table.from_pylist(rows), OUT / "gpqa_diamond.parquet")
    print(f"gpqa_diamond: {len(rows)} rows")


def build_val500() -> None:
    for domain in ("RaR-Science", "RaR-Medicine"):
        table = pq.read_table(RL / "data" / "verl-newnew" / domain / "val.parquet")
        idx = sorted(random.Random(0).sample(range(table.num_rows), 500))
        pq.write_table(table.take(idx), OUT / f"{domain}_val500.parquet")
        print(f"{domain}_val500: 500 of {table.num_rows}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_val500()
    build_gpqa()
