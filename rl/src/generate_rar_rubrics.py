#!/usr/bin/env python3
"""Generate question-specific RaR rubrics using the paper's Appendix A prompt."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from tqdm import tqdm

MAX_GENERATION_ATTEMPTS = 5


MEDICAL_PROMPT = """You are an expert rubric writer. Your job is to generate a self-contained set of evaluation criteria (“rubrics”) for judging how good a response is to a given question. Rubrics can cover aspects of a response such as, but not limited to, factual correctness, ideal-response characteristics, style, completeness, helpfulness, harmlessness, patient-centeredness, depth of reasoning, contextual relevance, and empathy. Each item must be self-contained -- non expert readers should not need to infer anything or consult external information. Begin each description with its category: “Essential Criteria: …”, “Important Criteria: …”, “Optional Criteria: …”, or “Pitfall Criteria: Does not mention …”.

Inputs:
• question: The full question text.
• reference_answer: The ideal answer, including any specific facts, explanations, or advice.

Total items:
• Choose 7–20 rubric items based on the complexity of the question.

Each rubric item:
• title (2–4 words).
• description: One sentence starting with its category prefix that explicitly states exactly what to look for. For example:
  • Essential Criteria: Identifies non-contrast helical CT scan as the most sensitive modality for ureteric stones.
  • Pitfall Criteria: Does not mention identifying (B) as the correct answer.
  • Important Criteria: Explains that non-contrast helical CT detects stones of varying sizes and compositions.
  • Optional Criteria: States “The final answer is (B)” or similar answer choice formatting.
• weight: For Essential/Important/Optional, use 1–5 (5 = most important); for Pitfall, use –1 or –2.

Category guidance:
• Essential: Critical facts or safety checks; if missing, the response is invalid (weight 5).
• Important: Key reasoning, completeness, or clarity; strongly affects quality (weight 3–4).
• Optional: Helpful style or extra depth; nice to have but not deal-breaking (weight 1–2).
• Pitfall: Common mistakes or omissions specific to this prompt—identify things a respondent often forgets or misstates. Each Pitfall description must begin with “Pitfall Criteria: Does not mention …” or “Pitfall Criteria: Recommends …” and use weight –1 or –2.

To ensure self-contained guidance:
• When referring to answer choices, explicitly say “Identifies (A)”, “Identifies (B)”, etc., rather than vague phrasing.
• If the format requires a conclusion like “The final answer is (B)”, include a rubric item such as:
  • Essential Criteria: Includes a clear statement “The final answer is (B)”.
• If reasoning should precede the answer, include a rubric like:
  • Important Criteria: Presents the explanation before stating the final answer.
• If brevity is valued, include a rubric like:
  • Optional Criteria: Remains concise and avoids unnecessary detail.
• If the question context demands mention of specific findings, include that explicitly (e.g., “Essential Criteria: Mentions that CT does not require contrast”).

Output:
Provide a JSON array of rubric objects. Each object must contain exactly three keys—title, description, and weight. Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category prefix, and no extra keys are allowed.

Now, given the question and reference_answer, generate the rubric as described. The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.

question:
{question}

reference_answer:
{reference_answer}"""


SCIENCE_PROMPT = """You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. Your job is to generate a self-contained set of evaluation criteria (“rubrics”) for judging how good a response is to a given question in one of these domains. Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. Each rubric item must be fully self-contained so that non-expert readers need not consult any external information.

Inputs:
• question: The full question text.
• reference_answer: The ideal answer, including any key facts or explanations.

Total items:
• Choose 7–20 rubric items based on the complexity of the question.

Each rubric item must include exactly three keys:
• title (2–4 words)
• description: One sentence beginning with its category prefix, explicitly stating what to look for. For example:
  • Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential) before the event equals the total mechanical energy after the event.
  • Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic energy plus initial potential energy equals final kinetic energy plus final potential energy.
  • Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential energy, to illustrate how energy shifts within the system.
  • Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying conservation of mechanical energy.
• weight: For Essential/Important/Optional, use 1–5 (5 = most important); for Pitfall, use –1 or –2.

Category guidance:
• Essential: Critical facts or safety checks; omission invalidates the response.
• Important: Key reasoning or completeness; strongly affects quality.
• Optional: Nice-to-have style or extra depth.
• Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
• When referring to answer choices, explicitly say “Identifies (A)”, “Identifies (B)”, etc.
• If a clear conclusion is required (e.g. “The final answer is (B)”), include an Essential Criteria for it.
• If reasoning should precede the final answer, include an Important Criteria to that effect.
• If brevity is valued, include an Optional Criteria about conciseness.

Output:  
Provide a JSON array of rubric objects. Each object must contain exactly three keys—title, description, and weight. Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category prefix, and no extra keys are allowed.

Now, given the question and reference_answer, generate the rubric as described. The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.

question:
{question}

reference_answer:
{reference_answer}"""


def extract_array(text: str) -> list[dict]:
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates.append(text[text.find("[") : text.rfind("]") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, list):
            return value
    raise ValueError(f"generator did not return a JSON array: {text[:500]!r}")


def validate_rubrics(value: list[dict]) -> list[dict]:
    if not 7 <= len(value) <= 20:
        raise ValueError(f"expected 7-20 rubrics, got {len(value)}")
    result = []
    for item in value:
        if set(item) != {"title", "description", "weight"}:
            raise ValueError(f"rubric keys must be title, description, weight: {item!r}")
        title = str(item["title"]).strip()
        description = str(item["description"]).strip()
        weight = float(item["weight"])
        if not title or not description or not (-2 <= weight <= 5):
            raise ValueError(f"invalid rubric item: {item!r}")
        if not re.match(r"^(Essential|Important|Optional|Pitfall) Criteria:", description):
            raise ValueError(f"description has no category prefix: {description!r}")
        result.append({"title": title, "description": description, "weight": int(weight) if weight.is_integer() else weight})
    return result


def request_rubric(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
    *,
    system: str | None = None,
    max_tokens: int = 16384,
    request_hook: Callable[[dict], None] | None = None,
) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request_record = {
        "messages": messages,
        "request": {
            "endpoint": f"{base_url.rstrip('/')}/chat/completions",
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    if request_hook is not None:
        request_hook({"event": "request", **request_record})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        text = payload["choices"][0]["message"]["content"]
    except Exception as exc:
        if request_hook is not None:
            request_hook({"event": "error", **request_record, "error": repr(exc)})
        raise
    if request_hook is not None:
        request_hook({
            "event": "response",
            **request_record,
            "response": {"role": "assistant", "content": text},
        })
    return validate_rubrics(extract_array(text))


def main() -> None:
    # Keep this dependency local so online_reward.py can reuse the paper's
    # prompt templates without requiring the dataset-processing package.
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "rl/data/generated_rubrics")
    parser.add_argument("--base-url", default="http://10.176.56.53:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    for dirname in ("RaR-Medicine", "RaR-Science"):
        template = MEDICAL_PROMPT if dirname == "RaR-Medicine" else SCIENCE_PROMPT
        for split in ("train", "val", "test"):
            source = args.raw_dir / dirname / f"{split}-00000-of-00001.parquet"
            output = args.output_dir / dirname / f"{split}.jsonl"
            dataset = load_dataset("parquet", data_files=str(source), split="train")
            output.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if output.exists():
                with output.open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        existing[int(record["index"])] = record
            pending = [
                (index, row) for index, row in enumerate(dataset)
                if index not in existing or not existing[index].get("rubrics")
            ]
            lock = threading.Lock()
            error_output = output.with_suffix(".errors.jsonl")

            def generate(item):
                index, row = item
                prompt = template.format(
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                )
                last_error = None
                for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
                    try:
                        attempt_prompt = prompt
                        if attempt > 1:
                            attempt_prompt += (
                                "\n\nIMPORTANT: Your previous output was invalid because it did not contain "
                                "a JSON array. Output the JSON array immediately. Do not include reasoning, "
                                "analysis, markdown fences, or any text before or after the array."
                            )
                        rubrics = request_rubric(
                            args.base_url,
                            args.model,
                            attempt_prompt,
                            args.timeout,
                            temperature=0.2 if attempt > 1 else 0,
                        )
                        return {
                            "index": index,
                            "question": row["question"],
                            "rubrics": rubrics,
                        }
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < MAX_GENERATION_ATTEMPTS:
                            time.sleep(min(2 ** (attempt - 1), 8))
                return {
                    "index": index,
                    "question": row["question"],
                    "rubrics": [],
                    "error": last_error,
                    "attempts": MAX_GENERATION_ATTEMPTS,
                }

            with output.open("a", encoding="utf-8") as handle:
                with error_output.open("a", encoding="utf-8") as error_handle:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        future_to_index = {
                            pool.submit(generate, item): item[0] for item in pending
                        }
                        progress = tqdm(
                            as_completed(future_to_index),
                            total=len(future_to_index),
                            desc=f"{dirname}/{split}",
                            unit="rubric",
                        )
                        failures = 0
                        for future in progress:
                            index = future_to_index[future]
                            try:
                                record = future.result()
                            except Exception as exc:
                                failures += 1
                                error_record = {
                                    "index": index,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                error_handle.write(
                                    json.dumps(error_record, ensure_ascii=False) + "\n"
                                )
                                error_handle.flush()
                                tqdm.write(
                                    f"{dirname}/{split}: failed index {index}: {exc}"
                                )
                                continue
                            with lock:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                handle.flush()
                                if record.get("error"):
                                    error_handle.write(
                                        json.dumps(record, ensure_ascii=False) + "\n"
                                    )
                                    error_handle.flush()
                                    failures += 1
                                    tqdm.write(
                                        f"{dirname}/{split}: failed index {index} "
                                        f"after {MAX_GENERATION_ATTEMPTS} attempts: "
                                        f"{record['error']}"
                                    )
                        progress.close()
            print(f"{output}: {len(existing) + len(pending) - failures} records")
            if failures:
                print(f"{error_output}: {failures} failures")


if __name__ == "__main__":
    main()
