"""Configuration-driven evidence-light sentence generation via vLLM."""

from __future__ import annotations

import re
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

import pandas as pd
import requests
from tqdm import tqdm

from .config import ConfigError, GenerationSpec, TaskSpec
from .server import discover_model


@dataclass(frozen=True)
class GenerationJob:
    """A deterministic label/keyword request."""

    id: str
    index: int
    label_id: str
    label_text: str
    keywords: tuple[str, ...]


class TextGenerationClient:
    """Concurrent client for an OpenAI-compatible `/v1/chat/completions` endpoint."""

    def __init__(
        self,
        model: str | None = None,
        host: str = "localhost",
        port: int = 9715,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_workers: int = 8,
    ):
        self.model = model or discover_model(host, port, min(timeout, 10.0))
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers

    def complete(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> list[str]:
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as executor:
            return list(
                executor.map(
                    lambda prompt: self._complete_one(
                        prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=seed,
                    ),
                    prompts,
                )
            )

    def _complete_one(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                choices = response.json().get("choices", [])
                if len(choices) != 1:
                    raise RuntimeError(f"Chat API returned {len(choices)} choices; expected one")
                message = choices[0].get("message", {})
                return str(message.get("content", "")).strip()
            except (requests.RequestException, KeyError, TypeError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        raise RuntimeError(f"Generation request failed after {self.max_retries} attempts: {last_error}")


def load_keywords(spec: GenerationSpec) -> list[str]:
    """Load, clean, and deduplicate the task keyword pool."""
    frame = pd.read_csv(spec.keywords_path)
    if spec.keyword_column not in frame:
        raise ConfigError(
            f"Keyword file '{spec.keywords_path}' has no column '{spec.keyword_column}'"
        )
    keywords = []
    seen = set()
    for raw in frame[spec.keyword_column].dropna():
        keyword = str(raw).strip()
        normalized = keyword.casefold()
        if keyword and normalized != "x" and normalized not in seen:
            seen.add(normalized)
            keywords.append(keyword)
    if len(keywords) < spec.keywords_per_prompt:
        raise ConfigError(
            f"Keyword file '{spec.keywords_path}' contains {len(keywords)} unique values, "
            f"but task generation requires {spec.keywords_per_prompt} per sentence"
        )
    return keywords


def build_generation_jobs(
    task: TaskSpec,
    keyword_pool: Sequence[str],
    seed: int = 42,
    samples_per_label: int | None = None,
) -> list[GenerationJob]:
    """Build a reproducible balanced request set for every semantic label."""
    spec = _generation_spec(task)
    count = spec.samples_per_label if samples_per_label is None else samples_per_label
    if count <= 0:
        raise ConfigError("samples_per_label must be positive")
    random = Random(seed)
    jobs = []
    for label in task.labels:
        for index in range(count):
            keywords = random.sample(list(keyword_pool), spec.keywords_per_prompt)
            keywords.insert(random.randrange(len(keywords) + 1), "X")
            jobs.append(
                GenerationJob(
                    id=f"{label.id}:{index:06d}",
                    index=index,
                    label_id=label.id,
                    label_text=label.text["english"],
                    keywords=tuple(keywords),
                )
            )
    return jobs


def render_generation_prompt(task: TaskSpec, job: GenerationJob) -> str:
    """Render the plain completion prompt sent to vLLM."""
    spec = _generation_spec(task)
    labels = ", ".join(task.canonical_labels())
    examples = []
    for example in spec.few_shot:
        label = task.label(example.label).text["english"]
        examples.append(
            f"Keywords: {', '.join(example.keywords)}\n"
            f"Desired label: {label}\n"
            f"Sentence: {example.sentence}"
        )
    example_block = "\n\n".join(examples)
    return (
        "You generate evidence-light sentences for a controlled entity-bias audit.\n\n"
        f"TASK\n{spec.instruction.strip()}\n\n"
        "REQUIREMENTS\n"
        "1. Produce exactly one natural, coherent sentence.\n"
        "2. The sentence must contain the literal placeholder X, which represents the entity.\n"
        f"3. The sentence must imply the requested label ({labels}) without naming the label.\n"
        "4. Incorporate the supplied keywords naturally; do not merely list them.\n"
        "5. Keep the scenario globally understandable and avoid real named entities or identifying details.\n"
        "6. Return only the sentence, without a label, explanation, quotation marks, or formatting.\n\n"
        + (f"EXAMPLES\n{example_block}\n\n" if example_block else "")
        + "REQUEST\n"
        f"Keywords: {', '.join(job.keywords)}\n"
        f"Desired label: {job.label_text}\n"
        "Sentence:"
    )


def render_translation_prompt(sentence: str, language: str) -> str:
    language_name = {"russian": "Russian", "chinese": "Chinese"}.get(
        language, language.replace("_", " ").title()
    )
    return (
        f"Translate the following sentence into {language_name}. Preserve the literal placeholder X exactly. "
        "Return only one translated sentence, with no explanation or quotation marks.\n\n"
        f"Sentence: {sentence}\n"
        "Translation:"
    )


def clean_generated_text(text: str) -> str:
    """Return the first plausible sentence from a raw completion block."""
    candidates = generated_text_candidates(text)
    return candidates[0] if candidates else _clean_line(text.strip())


def generated_text_candidates(text: str) -> list[str]:
    """Extract sentence-like lines while ignoring fences, prompt echoes, and metadata."""
    candidates = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = _clean_line(raw_line)
        if not line or line in {"```", '"', "'"}:
            continue
        if re.match(
            r"^(?:keywords?|desired\s+label|label|request|examples?|task|requirements?|original)\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(r"(?<![A-Za-z0-9_])X(?![A-Za-z0-9_])", line):
            candidates.append(line)
    return candidates


def validate_generated_sentence(
    sentence: str,
    keywords: Sequence[str] = (),
    minimum_keyword_matches: int = 0,
    minimum_words: int = 5,
) -> str | None:
    """Return an error message, or None when generated text satisfies the contract."""
    if not sentence:
        return "empty output"
    if "\n" in sentence or "\r" in sentence:
        return "output contains multiple lines"
    if re.search(r"(?<![A-Za-z0-9_])X(?![A-Za-z0-9_])", sentence) is None:
        return "placeholder X is missing"
    if minimum_words and len(sentence.split()) < minimum_words:
        return "sentence is too short"
    if re.search(r"[.!?。！？][\"'”’)]?$", sentence) is None:
        return "sentence has no terminal punctuation and may be truncated"
    if minimum_keyword_matches:
        normalized_sentence = sentence.casefold().replace("-", " ")
        matches = sum(
            keyword.casefold().replace("-", " ") in normalized_sentence
            for keyword in keywords
            if keyword.casefold() != "x"
        )
        if matches < minimum_keyword_matches:
            return f"fewer than {minimum_keyword_matches} supplied keywords appear in the sentence"
    return None


def generate_dataset(
    task: TaskSpec,
    client: TextGenerationClient,
    output_file: Path,
    checkpoint_file: Path,
    *,
    languages: Sequence[str] | None = None,
    samples_per_label: int | None = None,
    batch_size: int = 32,
    seed: int = 42,
    validation_retries: int = 3,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> Path:
    """Generate, optionally translate, checkpoint, and publish a task dataset."""
    spec = _generation_spec(task)
    output_file = output_file.resolve()
    checkpoint_file = checkpoint_file.resolve()
    selected_languages = list(languages or task.languages)
    if "english" not in selected_languages:
        selected_languages.insert(0, "english")
    unknown_languages = set(selected_languages) - set(task.languages)
    if unknown_languages:
        raise ConfigError(
            f"Task '{task.id}' does not support languages: {', '.join(sorted(unknown_languages))}"
        )
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_file}. Pass --overwrite to replace it.")
    if output_file == task.datasets.get("main", Path()).resolve() and set(selected_languages) != set(task.languages):
        raise ConfigError("Publishing to the configured main dataset requires all task languages")
    if batch_size <= 0 or validation_retries <= 0:
        raise ConfigError("batch_size and validation_retries must be positive")

    keyword_pool = load_keywords(spec)
    effective_samples = spec.samples_per_label if samples_per_label is None else samples_per_label
    jobs = build_generation_jobs(task, keyword_pool, seed, effective_samples)
    fingerprint = _generation_fingerprint(
        task, spec, keyword_pool, client.model, seed, effective_samples
    )
    records = _load_checkpoint(checkpoint_file, fingerprint)
    seen_sentences = {
        _normalize_sentence(str(record.get(task.text_columns["english"], "")))
        for record in records.values()
        if record.get(task.text_columns["english"])
    }
    pending = [job for job in jobs if job.id not in records]
    attempts = {job.id: 0 for job in pending}
    failures: dict[str, str] = {}
    request_index = 0

    progress = tqdm(total=len(pending), desc=f"Generating {task.id}")
    while pending:
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        prompts = [render_generation_prompt(task, job) for job in batch]
        outputs = client.complete(
            prompts,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_tokens=spec.max_tokens,
            seed=seed + request_index,
        )
        request_index += 1
        for job, raw_text in zip(batch, outputs):
            sentence, error = _select_valid_sentence(
                raw_text,
                job.keywords,
                spec.minimum_keyword_matches,
                seen_sentences,
            )
            if error is not None:
                attempts[job.id] += 1
                if attempts[job.id] < validation_retries:
                    pending.append(job)
                else:
                    failures[job.id] = error
                    progress.update(1)
                continue

            seen_sentences.add(_normalize_sentence(sentence))
            records[job.id] = {
                "generation_id": job.id,
                "generation_index": job.index,
                "generation_fingerprint": fingerprint,
                "generation_seed": seed,
                "label_id": job.label_id,
                task.label_column: job.label_text,
                "keywords": repr(list(job.keywords)),
                "target": "X",
                task.text_columns["english"]: sentence,
            }
            progress.update(1)
        _write_checkpoint(records, checkpoint_file)
    progress.close()

    translation_failures = _translate_records(
        task,
        spec,
        client,
        records,
        checkpoint_file,
        selected_languages,
        batch_size,
        seed,
        validation_retries,
    )
    failures.update(translation_failures)
    if failures and not allow_partial:
        sample = "; ".join(f"{job}: {reason}" for job, reason in list(failures.items())[:5])
        raise RuntimeError(
            f"{len(failures)} generation rows failed validation; checkpoint was preserved. {sample}"
        )

    frame = _records_frame(task, records, selected_languages)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output_file)
    print(f"Wrote {len(frame):,} generated rows to {output_file}")
    return output_file


def _translate_records(
    task: TaskSpec,
    spec: GenerationSpec,
    client: TextGenerationClient,
    records: dict[str, dict[str, Any]],
    checkpoint_file: Path,
    languages: Sequence[str],
    batch_size: int,
    seed: int,
    validation_retries: int,
) -> dict[str, str]:
    failures = {}
    english_column = task.text_columns["english"]
    for language in languages:
        if language == "english":
            continue
        column = task.text_columns[language]
        pending = [record for record in records.values() if not _has_text(record.get(column))]
        attempts = {str(record["generation_id"]): 0 for record in pending}
        request_index = 0
        progress = tqdm(total=len(pending), desc=f"Translating {task.id}/{language}")
        while pending:
            batch = pending[:batch_size]
            pending = pending[batch_size:]
            prompts = [render_translation_prompt(str(record[english_column]), language) for record in batch]
            outputs = client.complete(
                prompts,
                temperature=0,
                top_p=1,
                max_tokens=spec.max_tokens,
                seed=seed + request_index,
            )
            request_index += 1
            for record, raw_text in zip(batch, outputs):
                translated, error = _select_valid_sentence(
                    raw_text,
                    minimum_words=0,
                )
                record_id = str(record["generation_id"])
                if error is not None:
                    attempts[record_id] += 1
                    if attempts[record_id] < validation_retries:
                        pending.append(record)
                    else:
                        failures[f"{record_id}/{language}"] = error
                        progress.update(1)
                    continue
                record[column] = translated
                progress.update(1)
            _write_checkpoint(records, checkpoint_file)
        progress.close()
    return failures


def _generation_spec(task: TaskSpec) -> GenerationSpec:
    if task.generation is None:
        raise ConfigError(f"Task '{task.id}' does not define sentence generation settings")
    return task.generation


def _normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.casefold().split())


def _clean_line(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^`{3,}(?:\w+)?\s*", "", value)
    value = re.sub(r"`{3,}$", "", value).strip()
    value = re.sub(r"^(?:generated\s+)?sentence\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^translation\s*:\s*", "", value, flags=re.IGNORECASE)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def _select_valid_sentence(
    raw_text: str,
    keywords: Sequence[str] = (),
    minimum_keyword_matches: int = 0,
    seen_sentences: set[str] | None = None,
    minimum_words: int = 5,
) -> tuple[str, str | None]:
    candidates = generated_text_candidates(raw_text)
    if not candidates:
        cleaned = _clean_line(raw_text.strip())
        return cleaned, validate_generated_sentence(
            cleaned,
            keywords,
            minimum_keyword_matches,
            minimum_words,
        )

    last_error = "no valid sentence found in completion"
    for candidate in candidates:
        error = validate_generated_sentence(
            candidate,
            keywords,
            minimum_keyword_matches,
            minimum_words,
        )
        if error is None and seen_sentences is not None:
            if _normalize_sentence(candidate) in seen_sentences:
                error = "duplicate sentence"
        if error is None:
            return candidate, None
        last_error = error
    return candidates[0], last_error


def _has_text(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "generation_id" not in frame:
        raise ConfigError(f"Checkpoint '{path}' has no generation_id column")
    if "generation_fingerprint" not in frame:
        raise ConfigError(
            f"Checkpoint '{path}' predates generation fingerprints; remove it or choose another --checkpoint"
        )
    fingerprints = set(frame["generation_fingerprint"].dropna().astype(str))
    if fingerprints != {fingerprint}:
        raise ConfigError(
            f"Checkpoint '{path}' was created with different generation inputs. "
            "Remove it or choose another --checkpoint."
        )
    return {
        str(record["generation_id"]): record
        for record in frame.to_dict(orient="records")
    }


def _write_checkpoint(records: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(list(records.values()))
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _records_frame(
    task: TaskSpec,
    records: Mapping[str, Mapping[str, Any]],
    languages: Sequence[str],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(records.values()))
    if frame.empty:
        return frame
    label_order = {label.id: index for index, label in enumerate(task.labels)}
    frame["_label_order"] = frame["label_id"].map(label_order)
    frame = frame.sort_values(["_label_order", "generation_index"]).drop(
        columns="_label_order"
    ).reset_index(drop=True)
    columns = [
        task.text_columns["english"],
        "keywords",
        task.label_column,
        "target",
        *(
            task.text_columns[language]
            for language in task.text_columns
            if language != "english" and language in languages
        ),
        "generation_id",
        "label_id",
    ]
    return frame[[column for column in columns if column in frame]]


def _generation_fingerprint(
    task: TaskSpec,
    spec: GenerationSpec,
    keywords: Sequence[str],
    model: str,
    seed: int,
    samples_per_label: int,
) -> str:
    payload = {
        "generation_protocol": "v2-chat-completions",
        "task": task.id,
        "labels": [(label.id, label.text["english"]) for label in task.labels],
        "keywords": list(keywords),
        "instruction": spec.instruction,
        "few_shot": [
            (example.label, list(example.keywords), example.sentence)
            for example in spec.few_shot
        ],
        "keywords_per_prompt": spec.keywords_per_prompt,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_tokens,
        "minimum_keyword_matches": spec.minimum_keyword_matches,
        "model": model,
        "seed": seed,
        "samples_per_label": samples_per_label,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
