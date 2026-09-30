#!/usr/bin/env python3
"""Evaluate a policy on HealthBench-1k and/or GPQA-Diamond.

The policy is started locally with vLLM for the duration of the evaluation.
The judge is an already-running OpenAI-compatible chat-completion endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


HEALTH_GRADER_PROMPT = """Your job is to look at a conversation and a rubric item, and score the last assistant response on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric}

# Instructions
Return a JSON object with fields \"explanation\" and \"criteria_met\". The latter must be a boolean. If the criterion has multiple requirements, it is met only when all requirements are met. Return only the JSON object in a markdown code block."""


GPQA_VERIFIER_PROMPT = """Determine whether the following model response matches the ground truth answer.
## Ground truth answer##: Option {correct_answer} or {correct_answer_text}
## Model Response ##: {response_text}

A response is considered correct if its final answer is the correct option letter (A, B, C, or D), or has the correct answer text.
Please respond with only \"Yes\" or \"No\" (without quotes). Do not include a rationale."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _post(base_url: str, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, timeout: float) -> str:
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return str(payload["choices"][0]["message"]["content"])


def _model_base_url(args: argparse.Namespace) -> str:
    return f"http://{args.model_host}:{args.model_port}/v1"


def _start_model_server(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.model_host,
        "--port",
        str(args.model_port),
        "--gpu-memory-utilization",
        "0.9",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "8192",
    ]
    environment = os.environ.copy()
    environment["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    health_url = f"http://{args.model_host}:{args.model_port}/v1/models"
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"model server exited during startup with code {process.returncode}")
            try:
                with urllib.request.urlopen(health_url, timeout=5):
                    return process
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"model server did not become ready at {health_url}")
    except Exception:
        _stop_model_server(process)
        raise


def _stop_model_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _json_object(text: str) -> dict[str, Any]:
    for candidate in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE) + [text]:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"invalid JSON response: {text[:300]!r}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no", "0"}:
        return False
    raise ValueError(f"criteria_met is not boolean: {value!r}")


def _conversation(messages: list[dict[str, str]], answer: str) -> str:
    return "\n\n".join(f"{item['role']}: {item['content']}" for item in messages + [{"role": "assistant", "content": answer}])


def evaluate_healthbench(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}
    cache_lock = threading.Lock()

    def grade(row: dict[str, Any], answer: str, item: dict[str, Any]) -> dict[str, Any]:
        rubric = {"criterion": item["criterion"], "points": item["points"], "tags": item.get("tags", [])}
        key = hashlib.sha256(json.dumps([row["id"], answer, rubric], sort_keys=True).encode()).hexdigest()
        with cache_lock:
            if key in cache:
                return cache[key]
        prompt = HEALTH_GRADER_PROMPT.format(
            conversation=_conversation(row["messages"], answer),
            rubric=json.dumps(rubric, ensure_ascii=False),
        )
        try:
            text = _post(args.judge_base_url, args.judge_model, [{"role": "user", "content": prompt}], 0, 256, args.timeout)
            result = _json_object(text)
            result = {"criteria_met": _as_bool(result["criteria_met"]), "explanation": str(result.get("explanation", ""))}
        except Exception as exc:  # keep one bad judge call from losing the run
            result = {"criteria_met": False, "explanation": f"judge_error: {exc}"}
        with cache_lock:
            cache[key] = result
        return result

    output = []
    progress_path = args.results_dir / "healthbench_results.jsonl"
    with progress_path.open("w", encoding="utf-8") as progress_file:
        progress = tqdm(rows[:args.limit or None], desc="HealthBench", unit="example")
        for row in progress:
            answer = _post(_model_base_url(args), args.model, row["messages"], 0, args.max_tokens, args.timeout)
            futures = []
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                for item in row["rubrics"]:
                    futures.append(pool.submit(grade, row, answer, item))
                grades = [future.result() for future in futures]
            positive_total = sum(float(item["points"]) for item in row["rubrics"] if float(item["points"]) > 0)
            achieved = sum(float(item["points"]) for item, result in zip(row["rubrics"], grades) if result["criteria_met"])
            score = achieved / positive_total if positive_total else 0.0
            result = {"id": row["id"], "answer": answer, "score": score, "grades": grades}
            output.append(result)
            _write_jsonl_record(progress_file, result)
        progress.close()
    score = sum(item["score"] for item in output) / len(output) if output else 0.0
    return {"task": "medicine", "n": len(output), "overall_score": max(0.0, min(1.0, score)), "examples": output}


def _choice_prompt(row: dict[str, Any], choices: list[str]) -> list[dict[str, str]]:
    labels = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    return [{"role": "user", "content": f"{row['question']}\n\n{labels}\n\nSolve the problem and end with \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}."}]


def _extract_choice(text: str, choices: list[str]) -> str | None:
    matches = re.findall(r"(?:\\boxed\s*\{\s*([A-D])\s*\}|(?:final answer|answer)\s*[:：]?\s*\(?([A-D])\)?)", text, re.I)
    if matches:
        a, b = matches[-1]
        return (a or b).upper()
    for i, choice in enumerate(choices):
        if choice.strip() and choice.strip().lower() in text.lower():
            return chr(65 + i)
    return None


def _verify_choice_with_judge(
    row: dict[str, Any],
    correct_answer: str,
    response: str,
    args: argparse.Namespace,
) -> str | None:
    prompt = GPQA_VERIFIER_PROMPT.format(
        correct_answer=correct_answer,
        correct_answer_text=row["choices"][int(row["answer_index"])],
        response_text=response,
    )
    try:
        verdict = _post(
            args.judge_base_url,
            args.judge_model,
            [{"role": "user", "content": prompt}],
            0,
            16,
            args.timeout,
        ).strip().lower()
    except Exception:
        return None
    if re.search(r"\byes\b", verdict):
        return correct_answer
    if re.search(r"\bno\b", verdict):
        return None
    return None


def evaluate_gpqa(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = rows[:args.limit or None]
    runs = []
    progress_path = args.results_dir / "gpqa_results.jsonl"
    with progress_path.open("w", encoding="utf-8") as progress_file:
        progress = tqdm(total=len(rows) * args.repeats, desc="GPQA", unit="example")
        for repeat in range(args.repeats):
            rng = random.Random(args.seed + repeat)
            correct = 0
            details = []
            for row in rows:
                order = list(range(4))
                rng.shuffle(order)
                choices = [row["choices"][i] for i in order]
                answer_label = chr(65 + order.index(int(row["answer_index"])))
                text = _post(_model_base_url(args), args.model, _choice_prompt(row, choices), 0, args.max_tokens, args.timeout)
                predicted = _extract_choice(text, choices)
                if predicted is None:
                    predicted = _verify_choice_with_judge(row, answer_label, text, args)
                is_correct = predicted == answer_label
                correct += int(is_correct)
                result = {"id": row["id"], "repeat": repeat, "predicted": predicted, "target": answer_label, "correct": is_correct, "answer": text}
                details.append({key: value for key, value in result.items() if key != "repeat"})
                _write_jsonl_record(progress_file, result)
                progress.update(1)
            runs.append({"repeat": repeat, "accuracy": correct / len(rows) if rows else 0.0, "examples": details})
        progress.close()
    return {"task": "science", "n": len(rows), "repeats": args.repeats, "accuracy_mean": sum(run["accuracy"] for run in runs) / len(runs) if runs else 0.0, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("medicine", "science", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default="./eval/data")
    parser.add_argument("--results-dir", type=Path, default="./eval/results")
    parser.add_argument("--model-host", default="127.0.0.1")
    parser.add_argument("--model-port", type=int, default=8001)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--judge-base-url", default="http://10.176.56.53:8000/v1")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model_server = None
    try:
        model_server = _start_model_server(args)
        summary = {}
        if args.task in {"medicine", "both"}:
            path = args.data_dir / "healthbench_1k.jsonl"
            summary["medicine"] = evaluate_healthbench(_read_jsonl(path), args)
            (args.results_dir / "healthbench_results.json").write_text(json.dumps(summary["medicine"], ensure_ascii=False, indent=2), encoding="utf-8")
        if args.task in {"science", "both"}:
            path = args.data_dir / "gpqa_diamond.jsonl"
            summary["science"] = evaluate_gpqa(_read_jsonl(path), args)
            (args.results_dir / "gpqa_results.json").write_text(json.dumps(summary["science"], ensure_ascii=False, indent=2), encoding="utf-8")
        (args.results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: {k: v for k, v in value.items() if k not in {"examples", "runs"}} for key, value in summary.items()}, indent=2))
    finally:
        _stop_model_server(model_server)


if __name__ == "__main__":
    main()
