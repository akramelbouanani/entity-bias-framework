"""Shared configuration and I/O helpers for result processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from bias_audit.analysis import normalize_probability_columns
from bias_audit.config import TaskRegistry, TaskSpec


PAPER_MODELS = [
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-70B",
    "Meta-Llama-3-8B-Instruct",
    "aya-expanse-32b",
    "Qwen2.5-72B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-32B",
    "Qwen3-14B",
    "Qwen3-8B",
    "Mistral-7B-Instruct-v0.2",
    "Qwen3-8B-Base",
    "Qwen3-14B-Base",
    "Llama-3.3-70B-Instruct",
    "Meta-Llama-3-8B",
]
PAPER_LANGUAGES = ["russian", "english", "chinese"]
LANGUAGE_CODES = {"english": "en", "russian": "ru", "chinese": "zh"}
SETTING_IDS = {(False, False): 1, (True, False): 2, (False, True): 3, (True, True): 4}


def select_tasks(registry: TaskRegistry, task_ids: Iterable[str] | None) -> list[TaskSpec]:
    tasks = registry.tasks(suite="generalization")
    if task_ids is None:
        return tasks
    selected = set(task_ids)
    unknown = selected - {task.id for task in tasks}
    if unknown:
        raise ValueError(f"Unknown generalization tasks: {', '.join(sorted(unknown))}")
    return [task for task in tasks if task.id in selected]


def raw_output_path(
    outputs_dir: Path,
    task: TaskSpec,
    language: str,
    model: str,
    few_shot: bool,
    numerical: bool,
) -> Path:
    return (
        outputs_dir
        / task.entity_set
        / task.id
        / language
        / model
        / f"output_{'fs' if few_shot else 'nofs'}_{'num' if numerical else 'text'}.csv"
    )


def load_probabilities(
    outputs_dir: Path,
    task: TaskSpec,
    language: str,
    model: str,
    few_shot: bool,
    numerical: bool,
) -> pd.DataFrame:
    path = raw_output_path(outputs_dir, task, language, model, few_shot, numerical)
    frame = pd.read_csv(path)
    frame = normalize_probability_columns(frame, task, language, numerical)
    labels = task.canonical_labels()
    # Failed rows may be retried and appended later. Keep the most complete copy.
    frame["_completeness"] = frame[labels].notna().sum(axis=1)
    frame = (
        frame.sort_values("_completeness")
        .drop_duplicates(["entity_index", "sentence_index"], keep="last")
        .drop(columns="_completeness")
    )
    return frame


def setting_column(
    prefix: str,
    task: TaskSpec,
    language: str,
    model: str,
    model_index: int,
    few_shot: bool,
    numerical: bool,
) -> str:
    language_code = LANGUAGE_CODES.get(language, language[:2].lower())
    setting = SETTING_IDS[(few_shot, numerical)]
    return f"{prefix}_{language_code}_s{setting}_m{model_index}_{task.id}"


def base_entity_frame(registry: TaskRegistry, entity_set_id: str, current_path: Path | None = None) -> pd.DataFrame:
    if current_path is not None and current_path.exists():
        return pd.read_csv(current_path)
    return pd.read_csv(registry.entity_set(entity_set_id).path)


def task_groups(tasks: Iterable[TaskSpec]) -> dict[str, list[TaskSpec]]:
    groups: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        groups.setdefault(task.entity_set, []).append(task)
    return groups


def write_processing_metadata(
    result_path: Path,
    metric: str,
    models: Iterable[str],
    languages: Iterable[str],
    tasks: Iterable[TaskSpec],
) -> Path:
    """Write the model-index contract consumed by the plotting framework."""
    model_list = list(models)
    metadata = {
        "schema_version": 1,
        "metric": metric,
        "models": {str(index): model for index, model in enumerate(model_list, start=1)},
        "languages": list(languages),
        "tasks": [task.id for task in tasks],
        "settings": {
            "1": {"few_shot": False, "numerical": False},
            "2": {"few_shot": True, "numerical": False},
            "3": {"few_shot": False, "numerical": True},
            "4": {"few_shot": True, "numerical": True},
        },
    }
    metadata_path = Path(result_path).with_suffix(".meta.json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(metadata_path)
    return metadata_path
