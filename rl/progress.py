#!/usr/bin/env python3
"""Training progress for the RaR rubric runs, parsed from logs/train_*.log.

    python rl/progress.py              # terminal summary
    python rl/progress.py --json out   # also dump every series (for the dashboard)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.environ.get("RAR_LOGS", os.path.join(ROOT, "logs"))
TOTAL_STEPS = 300
ARMS = ("agentic", "baseline", "shipped")
DATASETS = ("science", "medicine")
TRAIN_KEYS = {
    "critic/score/mean": "train_reward",
    "response_length/mean": "resp_len",
    "response_length/clip_ratio": "clip_ratio",
    "actor/entropy": "entropy",
    "actor/kl_loss": "kl_to_ref",
    "actor/grad_norm": "grad_norm",
    "timing_s/step": "step_s",
}
STEP_RE = re.compile(r"step:(\d+) - (.*)")
NUM_RE = re.compile(r"np\.float(?:32|64)\(([^)]*)\)")


def parse_run(arm: str, ds: str) -> dict | None:
    files = sorted(glob.glob(os.path.join(LOGS, f"train_{arm}_{ds}.attempt*.log")),
                   key=lambda p: int(re.search(r"attempt(\d+)", p).group(1)))
    if not files:
        return None
    steps: dict[int, dict] = {}
    for path in files:  # later attempts overwrite resumed steps
        with open(path, errors="ignore") as handle:
            for line in handle:
                m = STEP_RE.search(line)
                if not m:
                    continue
                row = {}
                for kv in m.group(2).split(" - "):
                    key, _, val = kv.partition(":")
                    val = NUM_RE.sub(r"\1", val).strip()
                    try:
                        row[key.strip()] = float(val)
                    except ValueError:
                        pass
                steps.setdefault(int(m.group(1)), {}).update(row)
    train = {k: [] for k in TRAIN_KEYS.values()}
    val = defaultdict(list)
    for s in sorted(steps):
        row = steps[s]
        for key, name in TRAIN_KEYS.items():
            if key in row:
                train[name].append([s, row[key]])
        for key, v in row.items():
            m = re.match(r"val-core/([^/]+)/reward/mean@\d+", key)
            if m:
                val[m.group(1)].append([s, v])
    last = max((s for s in steps if s > 0), default=0)
    recent = [v for _, v in train["step_s"][-10:]]
    eta_s = (TOTAL_STEPS - last) * (sum(recent) / len(recent)) if recent else None
    return {
        "arm": arm, "dataset": ds, "step": last, "attempts": len(files),
        "log_mtime": max(os.path.getmtime(p) for p in files),
        "eta_hours": None if eta_s is None else eta_s / 3600,
        "train": train, "val": dict(val),
    }


def bar(frac: float, width: int = 24) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def mean_last(series, n=10):
    vals = [v for _, v in series[-n:]]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write all series to this JSON file")
    args = ap.parse_args()
    runs = [r for a in ARMS for d in DATASETS if (r := parse_run(a, d))]
    now = time.time()

    print(f"RaR rubric RL progress  ({time.strftime('%Y-%m-%d %H:%M %Z')})\n")
    for r in runs:
        stale = now - r["log_mtime"] > 1800
        eta = "done" if r["step"] >= TOTAL_STEPS else (
            f"ETA {r['eta_hours']:.1f}h" if r["eta_hours"] is not None else "starting")
        flag = "  ! log idle >30min" if stale and r["step"] < TOTAL_STEPS else ""
        print(f"{r['arm']:>8} {r['dataset']:<8} {bar(r['step'] / TOTAL_STEPS)} "
              f"{r['step']:>3}/{TOTAL_STEPS} {100 * r['step'] / TOTAL_STEPS:5.1f}%  {eta}"
              f"  restarts={r['attempts'] - 1}{flag}")

    for ds in DATASETS:
        group = [r for r in runs if r["dataset"] == ds]
        if not group:
            continue
        sets = sorted({name for r in group for name in r["val"]})
        print(f"\n== {ds}: held-out validation (higher is better)")
        for name in sets:
            steps = sorted({s for r in group for s, _ in r["val"].get(name, [])})
            print(f"  {name}")
            print("    step " + "".join(f"{r['arm']:>10}" for r in group))
            for s in steps:
                cells = []
                for r in group:
                    v = dict((int(a), b) for a, b in r["val"].get(name, [])).get(s)
                    cells.append(f"{v:>10.3f}" if v is not None else f"{'':>10}")
                print(f"    {s:>4} " + "".join(cells))
        print(f"\n== {ds}: training health (mean of last 10 steps)")
        print("    " + f"{'':>10}" + "".join(f"{r['arm']:>10}" for r in group))
        for key in ("train_reward", "clip_ratio", "resp_len", "entropy", "kl_to_ref", "grad_norm"):
            print(f"    {key:>10}" + "".join(f"{mean_last(r['train'][key]):>10.3f}" for r in group))
    print("\nNote: train_reward uses each arm's own rubric, so it is not comparable across arms;"
          " compare the held-out validation instead.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"generated_at": now, "total_steps": TOTAL_STEPS, "runs": runs}, fh)


if __name__ == "__main__":
    main()
