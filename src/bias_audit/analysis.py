"""Configuration-aware normalization shared by processing pipelines."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import ConfigError, TaskSpec


INDEX_COLUMNS = ("entity_index", "sentence_index")


def source_labels(task: TaskSpec, language: str, numerical: bool) -> list[str]:
    """Return configured output columns in semantic label order."""
    if numerical:
        if not task.supports_numerical:
            raise ConfigError(f"Task '{task.id}' does not support numerical labels")
        return [str(label.numeric) for label in task.labels]
    return [label.text[language] for label in task.labels]


def normalize_probability_columns(
    frame: pd.DataFrame,
    task: TaskSpec,
    language: str,
    numerical: bool,
    canonical_language: str = "english",
) -> pd.DataFrame:
    """Rename localized/numeric probability columns to canonical semantic labels."""
    canonical = task.canonical_labels(canonical_language)
    configured = source_labels(task, language, numerical)
    frame = frame.copy()

    if set(configured).issubset(frame.columns):
        return frame.rename(columns=dict(zip(configured, canonical)))

    # Compatibility with early output files whose columns only encoded position.
    probability_columns = [column for column in frame.columns if column not in INDEX_COLUMNS]
    if len(probability_columns) != len(canonical):
        raise ConfigError(
            f"Output for task '{task.id}' has {len(probability_columns)} probability columns; "
            f"expected {len(canonical)} ({', '.join(configured)})"
        )
    return frame.rename(columns=dict(zip(probability_columns, canonical)))


def task_groups(tasks: Iterable[TaskSpec]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for task in tasks:
        groups.setdefault(task.entity_set, []).append(task.id)
    return {entity_set: sorted(ids) for entity_set, ids in groups.items()}

