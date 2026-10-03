"""VERL reward hook using the shipped rubrics carried by the VERL dataset."""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import http.client
import zlib
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
for path in (SRC_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harness.schema import Criterion, Polarity  # noqa: E402

JUDGE_BASE_URL = "http://127.0.0.1:8000/v1"
# Comma-separated replicas (rl/serve_judge_dp.sh serves ports 8000-8003).
JUDGE_BASE_URLS = [u.strip() for u in os.environ.get("JUDGE_BASE_URLS", JUDGE_BASE_URL).split(",") if u.strip()]
JUDGE_OUTAGE_SECONDS = float(os.environ.get("JUDGE_OUTAGE_SECONDS", "900"))
JUDGE_MODEL = "Qwen/Qwen3.8-27B"
JUDGE_TIMEOUT_SECONDS = float(os.environ.get("JUDGE_TIMEOUT_SECONDS", "2000"))
JUDGE_MAX_RETRIES = 2
JUDGE_MAX_CONCURRENT = int(os.environ.get("JUDGE_MAX_CONCURRENT", "8"))
BASELINE_JUDGE_MAX_TOKENS = 8192 #12288
# Select how criterion weights are aggregated into the reward.  This can be
# overridden at launch with RUBRIC_WEIGHTING=raw_weight.
WEIGHTING_MODE = os.environ.get("RUBRIC_WEIGHTING", "paper_explicit").strip().lower()
VALID_WEIGHTING_MODES = {"paper_explicit", "raw_weight"}
_DISK_LOGGING_ENABLED = os.environ.get("VERL_REWARD_ENABLE_DISK_LOGGING", "0").lower() in {
    "1", "true", "yes", "on"
}
REWARD_LOG_PATH = str(SRC_DIR.parent / "outputs" / "verl_rar_reward_log.jsonl") if _DISK_LOGGING_ENABLED else ""
MODEL_LOG_PATH = SRC_DIR.parent / "log.json" if _DISK_LOGGING_ENABLED else None
REWARD_TIME_PATH = SRC_DIR.parent / "outputs" / "reward_time.txt" if _DISK_LOGGING_ENABLED else None

_JUDGE_SEMAPHORE = threading.BoundedSemaphore(JUDGE_MAX_CONCURRENT)
_REWARD_LOG_LOCK = threading.Lock()
_MODEL_LOG_LOCK = threading.Lock()
_REWARD_TIME_LOCK = threading.Lock()

BASELINE_JUDGE_SYSTEM = (
    "You are a meticulous, impartial grader. You are given a question, a candidate response, "
    "and a numbered list of statements about that response. For EACH statement, decide "
    "independently whether the statement is LITERALLY TRUE of the candidate response.\n\n"
    "Critical: you are NOT judging whether the response is good. You are only judging whether "
    "each statement accurately describes the response. Some statements describe desirable "
    "things and some describe mistakes; treat them identically and report only truth.\n\n"
    "Rules:\n"
    "- Judge ONLY the candidate response as written. Do not credit what it seems to intend.\n"
    "- Judge each statement independently.\n"
    "- Ignore formatting, length and eloquence unless explicitly mentioned.\n\n"
    "Output ONLY a JSON array, one object per statement, in the order given:\n"
    '[{"id": 1, "true": true, "checkable": true, "why": "<=20 words"}, ...]\n'
    "No prose outside the JSON array."
)


def _build_prompt(question: str, response: str, criteria: list[Criterion]) -> str:
    statements = []
    for index, criterion in enumerate(criteria, start=1):
        title = f" [{criterion.title}]" if criterion.title else ""
        statements.append(f"{index}.{title} {criterion.description}")
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<candidate_response>\n{response}\n</candidate_response>\n\n"
        f"<statements>\n{chr(10).join(statements)}\n</statements>\n\n"
        f"Return the JSON array with exactly {len(criteria)} objects, ids 1..{len(criteria)}."
    )


def _parse_judge_array(text: str, expected: int) -> list[dict]:
    """Extract the final JSON array even when the model prefixes reasoning."""
    candidates = [m.group(1) for m in re.finditer(
        r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
    )]
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (TypeError, json.JSONDecodeError):
            value = None
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]

    # Qwen may emit analysis before the final JSON without a code fence.
    # raw_decode() lets us parse an array embedded in that surrounding text.
    embedded = []
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, list) and all(isinstance(entry, dict) for entry in value):
            embedded.append(value)
    if embedded:
        exact = [value for value in embedded if len(value) == expected]
        value = exact[-1] if exact else embedded[-1]
        return value
    raise ValueError(f"baseline judge did not return a JSON array for {expected} criteria")


def _criterion_weight(criterion: Criterion, weighting_mode: str) -> float:
    if weighting_mode == "paper_explicit":
        category_weights = {
            "Essential": 1.0,
            "Important": 0.7,
            "Optional": 0.3,
            "Pitfall": 0.9,
        }
        return category_weights.get(criterion.category.value, 0.7)
    if weighting_mode == "raw_weight":
        try:
            return abs(float(criterion.weight)) or 1.0
        except (TypeError, ValueError):
            return 1.0
    raise ValueError(
        f"unsupported RUBRIC_WEIGHTING={weighting_mode!r}; "
        f"choose one of {sorted(VALID_WEIGHTING_MODES)}"
    )


def _explicit_score(
    criteria: list[Criterion],
    verdicts: list[dict],
    weighting_mode: str = WEIGHTING_MODE,
) -> float:
    by_id = {}
    for position, entry in enumerate(verdicts, start=1):
        try:
            item_id = int(entry.get("id", position))
        except (TypeError, ValueError):
            item_id = position
        by_id.setdefault(item_id, entry)
    numerator = denominator = 0.0
    for position, criterion in enumerate(criteria, start=1):
        literal = by_id.get(position, {}).get("true", by_id.get(position, {}).get("met"))
        if isinstance(literal, str):
            literal = literal.strip().lower() in {"true", "1", "yes"}
        met = bool(literal)
        if criterion.polarity is Polarity.NEGATIVE:
            met = not met
        weight = _criterion_weight(criterion, weighting_mode)
        denominator += weight
        if met:
            numerator += weight
    return numerator / denominator if denominator else 0.0


def _jsonable(value):
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _judge_content(response):
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _write_model_log(record: dict) -> None:
    if not _DISK_LOGGING_ENABLED or MODEL_LOG_PATH is None:
        return
    def as_json(value):
        if value is None or not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value

    response = as_json(record.get("response"))
    if isinstance(response, dict) and isinstance(response.get("choices"), list):
        response = response["choices"][0] if response["choices"] else None
    payload = {"request": as_json(record.get("request")), "response": response}
    try:
        MODEL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _MODEL_LOG_LOCK, MODEL_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n\n")
            handle.flush()
    except (OSError, TypeError, ValueError):
        pass


def _request_payload(request: urllib.request.Request) -> dict:
    body = request.data
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return {
        "url": request.full_url,
        "method": request.method or "POST",
        "headers": dict(request.header_items()),
        "body": body,
    }


def _write_reward_time(start_time: float) -> None:
    if not _DISK_LOGGING_ENABLED or REWARD_TIME_PATH is None:
        return
    try:
        REWARD_TIME_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _REWARD_TIME_LOCK, REWARD_TIME_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.perf_counter() - start_time:.6f}\n")
    except OSError:
        pass


def _write_reward_log(*, question, model_output, rubrics, judge_response, reward_result):
    if REWARD_LOG_PATH.lower() in {"", "0", "false", "no", "off", "none"}:
        return
    record = {
        "question": _jsonable(question),
        "model_output": _jsonable(model_output),
        "rubrics": _jsonable(rubrics),
        "verifier_score": _jsonable(reward_result.get("score")),
        "verifier_judge_failed": _jsonable(reward_result.get("judge_failed")),
        "verifier_output": _judge_content(judge_response),
    }
    if reward_result.get("judge_error"):
        record["verifier_error"] = reward_result["judge_error"]
    try:
        path = Path(REWARD_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _REWARD_LOG_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _judge_baseline(
    question: str,
    solution: str,
    rubrics: list[dict],
    info: dict,
    weighting_mode: str = WEIGHTING_MODE,
):
    criteria = []
    for item in rubrics:
        if not isinstance(item, dict) or not str(item.get("description") or "").strip():
            continue
        criteria.append(Criterion(
            title=str(item.get("title") or "").strip(),
            description=str(item.get("description") or "").strip(),
            weight=item.get("weight", 3),
            category=item.get("category"),
            polarity=item.get("polarity"),
            provenance={"stage": "shipped"},
        ))
    if not criteria:
        raise ValueError("shipped rubric has no usable criteria")

    order = list(range(len(criteria)))
    random.Random(f"7:{question}:{info.get('index', 'response')}").shuffle(order)
    shown = [criteria[index] for index in order]
    body = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": BASELINE_JUDGE_SYSTEM},
            {"role": "user", "content": _build_prompt(question, solution, shown)},
        ],
        "temperature": 0,
        "max_tokens": BASELINE_JUDGE_MAX_TOKENS,
    }).encode("utf-8")
    # Spread requests over the judge replicas; on an outage (connection refused,
    # 5xx, timeout) move to the next replica and keep retrying for up to
    # JUDGE_OUTAGE_SECONDS, so a judge restart delays rewards instead of
    # silently turning them into 0. Bad outputs still get JUDGE_MAX_RETRIES.
    first = zlib.crc32(f"{question}:{info.get('index', '')}:{solution[:64]}".encode()) % len(JUDGE_BASE_URLS)
    deadline = time.monotonic() + JUDGE_OUTAGE_SECONDS
    last_error = None
    bad_outputs = 0
    attempt = 0
    while True:
        base = JUDGE_BASE_URLS[(first + attempt) % len(JUDGE_BASE_URLS)]
        attempt += 1
        request = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_payload = _request_payload(request)
        raw_response = None
        try:
            with _JUDGE_SEMAPHORE:
                with urllib.request.urlopen(request, timeout=JUDGE_TIMEOUT_SECONDS) as response:
                    raw_response = response.read()
            result = json.loads(raw_response)
            text = result["choices"][0]["message"]["content"]
            verdicts = _parse_judge_array(text, len(criteria))
            by_id = {int(v.get("id", i)): v for i, v in enumerate(verdicts, start=1)}
            ordered = [by_id.get(i, {"id": i}) for i in range(1, len(criteria) + 1)]
            _write_model_log({
                "request": request_payload["body"],
                "response": raw_response.decode("utf-8", errors="replace"),
            })
            return _explicit_score(shown, ordered, weighting_mode), result, verdicts
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500:  # e.g. 400 prompt longer than the context: not retryable
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(2 ** min(attempt, 5), 30))
        except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException) as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(min(2 ** min(attempt, 5), 30))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            _write_model_log({
                "request": request_payload["body"],
                "response": (
                    raw_response.decode("utf-8", errors="replace")
                    if isinstance(raw_response, bytes) else None
                ),
            })
            bad_outputs += 1
            if bad_outputs > JUDGE_MAX_RETRIES:
                break
            time.sleep(min(2 ** bad_outputs, 8))
    raise RuntimeError(f"baseline judge failed after {attempt} attempt(s): {last_error}")


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Score one response against the shipped rubric stored in VERL data."""
    if data_source == "gpqa_diamond":
        from heldout_reward import gpqa_score
        return gpqa_score(solution_str, ground_truth)
    if data_source not in {"rar_medicine", "rar_science"}:
        raise ValueError(f"unsupported data_source: {data_source!r}")
    info = extra_info or {}
    question = info["question"]
    rubrics = info.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError(f"no shipped rubrics for index={info.get('index')!r}")
    request_start = time.perf_counter()
    try:
        score, result, verdicts = _judge_baseline(
            question, solution_str, rubrics, info, weighting_mode=WEIGHTING_MODE
        )
        result = dict(result)
        result["baseline_verdicts"] = verdicts
        reward_result = {"score": score, "judge_failed": 0.0, "judge_error": ""}
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
        result = None
        reward_result = {"score": 0.0, "judge_failed": 1.0, "judge_error": str(exc)}
    _write_reward_log(
        question=question,
        model_output=solution_str,
        rubrics=rubrics,
        judge_response=result,
        reward_result=reward_result,
    )
    _write_reward_time(request_start)
    return reward_result
