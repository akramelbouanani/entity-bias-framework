"""Completion-API client and resumable experiment execution."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests
from tqdm import tqdm

from .config import TaskSpec
from .prompts import build_prompt, category_tokens
from .server import discover_model


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -1e9
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def aggregate_token_probabilities(
    top_logprobs: Mapping[str, float],
    token_map: Mapping[str, Sequence[str]],
    exact: bool = False,
) -> dict[str, float]:
    """Aggregate first-token log probabilities into configured categories."""
    grouped = {category: [] for category in token_map}
    for token, log_probability in top_logprobs.items():
        normalized = token.strip().casefold()
        for category, candidates in token_map.items():
            matches = (candidate.strip().casefold() for candidate in candidates)
            if any(normalized == candidate if exact else normalized.startswith(candidate) for candidate in matches):
                grouped[category].append(float(log_probability))

    unnormalized = {
        category: math.exp(logsumexp(probabilities)) if probabilities else 0.0
        for category, probabilities in grouped.items()
    }
    total = sum(unnormalized.values())
    if total == 0:
        return {category: float("nan") for category in token_map}
    return {category: value / total for category, value in unnormalized.items()}


class CompletionClient:
    """Small client for OpenAI-compatible local completion endpoints."""

    def __init__(
        self,
        model: str | None = None,
        port: int = 9715,
        host: str = "localhost",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
        request_options: Mapping[str, Any] | None = None,
    ):
        self.model = model or discover_model(host, port, min(timeout, 10.0))
        self.url = f"http://{host}:{port}/v1/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.request_options = dict(request_options or {})

    def classify(
        self,
        prompts: str | Sequence[str],
        token_map: Mapping[str, Sequence[str]],
        exact: bool = False,
    ) -> dict[str, float] | list[dict[str, float]]:
        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)
        payload = {
            "model": self.model,
            "prompt": prompt_list,
            "seed": 42,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "logprobs": 20,
            **self.request_options,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                choices = response.json().get("choices", [])
                if len(choices) != len(prompt_list):
                    raise RuntimeError(
                        f"Completion API returned {len(choices)} choices for {len(prompt_list)} prompts"
                    )
                if all(isinstance(choice.get("index"), int) for choice in choices):
                    choices = sorted(choices, key=lambda choice: choice["index"])
                outputs = []
                for choice in choices:
                    top = choice.get("logprobs", {}).get("top_logprobs", [{}])[0] or {}
                    outputs.append(aggregate_token_probabilities(top, token_map, exact))
                return outputs[0] if single else outputs
            except (requests.RequestException, KeyError, TypeError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        print(f"Completion request failed after {self.max_retries} attempts: {last_error}")
        missing = {category: float("nan") for category in token_map}
        return missing if single else [missing.copy() for _ in prompt_list]


def output_path(
    output_dir: Path,
    task: TaskSpec,
    language: str,
    model_name: str,
    few_shot: bool,
    numerical: bool,
    variant: str,
) -> Path:
    """Build the stable output layout used by execution and processing."""
    if task.suite == "validation":
        filename = "original_results.csv" if variant == "original" else "synthetic_results.csv"
        return output_dir / task.id / model_name / language / filename
    filename = f"output_{'fs' if few_shot else 'nofs'}_{'num' if numerical else 'text'}.csv"
    return output_dir / task.entity_set / task.id / language / model_name / filename


def run_dataframe(
    frame: pd.DataFrame,
    task: TaskSpec,
    client: CompletionClient,
    output_file: Path,
    language: str,
    few_shot: bool = False,
    numerical: bool = False,
    request_batch_size: int = 500,
    flush_rows: int = 5000,
) -> Path:
    """Classify an expanded frame, append checkpoints, and resume by row identity."""
    token_map = category_tokens(task, language, numerical)
    probability_columns = list(token_map)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed: set[tuple[int, int]] = set()
    if output_file.exists():
        previous = pd.read_csv(output_file)
        required = {"entity_index", "sentence_index", *probability_columns}
        if not required.issubset(previous.columns):
            missing = ", ".join(sorted(required - set(previous.columns)))
            raise ValueError(f"Cannot resume {output_file}: missing columns {missing}")
        complete = previous.dropna(subset=probability_columns, how="any")
        processed = set(zip(complete["entity_index"].astype(int), complete["sentence_index"].astype(int)))
        print(f"Resuming {output_file}: {len(processed):,} completed rows found")

    if processed:
        row_keys = pd.MultiIndex.from_frame(frame[["entity_index", "sentence_index"]].astype(int))
        completed_keys = pd.MultiIndex.from_tuples(processed, names=["entity_index", "sentence_index"])
        pending = frame.loc[~row_keys.isin(completed_keys)]
    else:
        pending = frame
    print(f"Processing {len(pending):,} rows for {task.id}/{language}")

    buffered: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffered
        if not buffered:
            return
        pd.DataFrame.from_records(buffered).to_csv(
            output_file,
            mode="a",
            index=False,
            header=not output_file.exists(),
        )
        buffered = []

    for start in tqdm(range(0, len(pending), request_batch_size), desc=f"Classifying {task.id}"):
        batch = pending.iloc[start : start + request_batch_size]
        prompts = [
            build_prompt(
                task,
                sentence=str(row.sentence),
                language=language,
                few_shot=few_shot,
                numerical=numerical,
                target=str(row.target),
            )
            for row in batch.itertuples(index=False)
        ]
        probabilities = client.classify(prompts, token_map, exact=numerical)
        for row, probability in zip(batch.itertuples(index=False), probabilities):
            buffered.append(
                {
                    "entity_index": int(row.entity_index),
                    "sentence_index": int(row.sentence_index),
                    **probability,
                }
            )
        if len(buffered) >= flush_rows:
            flush()
    flush()
    return output_file
