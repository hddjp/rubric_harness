"""VERL reward hook that judges responses with released baseline rubrics."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
RL_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import shipped_reward as _shipped_reward  # noqa: E402

_judge_baseline = _shipped_reward._judge_baseline
_write_reward_log = _shipped_reward._write_reward_log
_write_reward_time = _shipped_reward._write_reward_time

DISK_LOGGING_ENABLED = os.environ.get("VERL_REWARD_ENABLE_DISK_LOGGING", "0").lower() in {
    "1", "true", "yes", "on"
}

# Keep baseline diagnostics separate from the shipped-rubric run.  The judge
# implementation is shared, but its module-level logging paths are redirected
# for this worker process.
_shipped_reward.REWARD_LOG_PATH = (
    str(RL_DIR / "outputs" / "baseline_reward_log.jsonl")
    if DISK_LOGGING_ENABLED
    else ""
)
_shipped_reward.REWARD_TIME_PATH = (
    RL_DIR / "outputs" / "baseline_reward_time.txt"
    if DISK_LOGGING_ENABLED
    else None
)
_shipped_reward.MODEL_LOG_PATH = (
    RL_DIR / "outputs" / "baseline_model_log.json"
    if DISK_LOGGING_ENABLED
    else None
)

# Keep baseline independently configurable.  It also honors
# RUBRIC_WEIGHTING=raw_weight at process launch.
WEIGHTING_MODE = os.environ.get("RUBRIC_WEIGHTING", "paper_explicit").strip().lower()
if WEIGHTING_MODE not in _shipped_reward.VALID_WEIGHTING_MODES:
    raise ValueError(
        f"unsupported RUBRIC_WEIGHTING={WEIGHTING_MODE!r}; "
        f"choose one of {sorted(_shipped_reward.VALID_WEIGHTING_MODES)}"
    )

RUBRIC_DIR = RL_DIR / "data" / "rubrics"
ERROR_LOG_PATH = RL_DIR / "outputs" / "error.txt" if DISK_LOGGING_ENABLED else None
_RUBRIC_LOCK = threading.Lock()
_ERROR_LOCK = threading.Lock()
_RUBRICS_LOADED = False
_RUBRICS: dict[tuple[str, str], list[dict]] = {}


def _write_error(message: str, exc: BaseException | None = None) -> None:
    if not DISK_LOGGING_ENABLED or ERROR_LOG_PATH is None:
        return
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        detail = f"{message}: {type(exc).__name__}: {exc}" if exc else message
        if exc is not None:
            detail += "\n" + "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).rstrip()
        with _ERROR_LOCK, ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(detail.rstrip() + "\n\n")
    except OSError:
        pass


def _load_rubrics() -> None:
    global _RUBRICS_LOADED
    with _RUBRIC_LOCK:
        if _RUBRICS_LOADED:
            return
        for path in sorted(RUBRIC_DIR.glob("*_baseline_*.jsonl")):
            try:
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        domain = str(record.get("domain") or "")
                        question = record.get("question")
                        rubric = record.get("rubric", record.get("rubrics"))
                        if domain and isinstance(question, str) and isinstance(rubric, list):
                            _RUBRICS[(domain, question)] = rubric
                        else:
                            _write_error(f"invalid baseline rubric record {path}:{line_number}")
            except Exception as exc:
                _write_error(f"failed to load baseline rubrics from {path}", exc)
        _RUBRICS_LOADED = True


def _get_rubric(data_source: str, question: str) -> list[dict]:
    _load_rubrics()
    rubric = _RUBRICS.get((data_source, question))
    if not rubric:
        raise KeyError(
            f"baseline rubric not found for data_source={data_source!r}, question={question!r}"
        )
    return rubric


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Judge one response against the matching released baseline rubric."""
    if data_source == "gpqa_diamond":
        from heldout_reward import gpqa_score
        return gpqa_score(solution_str, ground_truth)
    started = time.perf_counter()
    info = extra_info or {}
    question = info.get("question")
    rubric = None
    result = None
    try:
        if data_source not in {"rar_medicine", "rar_science"}:
            raise ValueError(f"unsupported data_source: {data_source!r}")
        if not isinstance(question, str) or not question:
            raise ValueError("extra_info.question is missing")
        if info.get("split") == "train":
            rubric = _get_rubric(data_source, question)
        else:
            # Validation: most val questions have no generated rubric, so every
            # arm is scored with the shipped rubric to share one yardstick.
            rubric = info.get("rubrics")
            if not isinstance(rubric, list) or not rubric:
                raise ValueError("no shipped rubric for validation row")
        score, result, verdicts = _judge_baseline(
            question, solution_str, rubric, info, weighting_mode=WEIGHTING_MODE
        )
        result = dict(result)
        result["baseline_verdicts"] = verdicts
        reward = {"score": score, "judge_failed": 0.0, "judge_error": ""}
    except Exception as exc:
        _write_error(
            f"baseline reward failed for data_source={data_source!r}, index={info.get('index')!r}",
            exc,
        )
        reward = {"score": 0.0, "judge_failed": 1.0, "judge_error": str(exc)}

    _write_reward_time(started)
    _write_reward_log(
        question=question,
        model_output=solution_str,
        rubrics=rubric or [],
        judge_response=result,
        reward_result=reward,
    )
    return reward
