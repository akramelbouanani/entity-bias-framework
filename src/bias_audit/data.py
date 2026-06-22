"""Load and expand evidence-light sentence templates over entity tables."""

from __future__ import annotations

import re

import pandas as pd
from tqdm import tqdm

from .config import ConfigError, EntitySetSpec, TaskSpec


def substitute_entity(sentence: str, name: str, placeholder: str = "X") -> str:
    """Replace a placeholder while preserving hashtag/mention compaction behavior."""
    pattern = rf"[#@]?\w*{re.escape(placeholder)}"

    def replacer(match: re.Match[str]) -> str:
        full = match.group(0)
        prefix = full.split(placeholder, 1)[0]
        replacement = name.replace(" ", "") if "#" in prefix or "@" in prefix else name
        return full.replace(placeholder, replacement)

    return re.sub(pattern, replacer, sentence)


def expand_frames(
    task: TaskSpec,
    entity_set: EntitySetSpec,
    templates: pd.DataFrame,
    entities: pd.DataFrame,
    language: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Create the Cartesian product of templates and entities for one language."""
    if language not in task.languages:
        raise ConfigError(f"Task '{task.id}' does not support language '{language}'")
    sentence_column = task.text_columns[language]
    name_column = entity_set.name_columns[language]
    rows = []
    iterator = templates.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(templates), desc=f"Expanding {task.id}/{language}")

    for sentence_index, template in iterator:
        sentence = str(template[sentence_column])
        for entity_index, entity in entities.iterrows():
            localized_name = str(entity[name_column])
            rows.append(
                {
                    "sentence_index": sentence_index,
                    "entity_index": entity_index,
                    "sentence": substitute_entity(sentence, localized_name, entity_set.placeholder),
                    "label": template.get(task.label_column),
                    "target": localized_name,
                    "entity": entity.get(entity_set.name_columns.get("english", name_column), localized_name),
                }
            )
    return pd.DataFrame.from_records(rows)


def expand_task(
    task: TaskSpec,
    entity_set: EntitySetSpec,
    language: str = "english",
    variant: str = "main",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load configured CSVs and return their fully expanded experiment frame."""
    try:
        dataset_path = task.datasets[variant]
    except KeyError as exc:
        raise ConfigError(
            f"Task '{task.id}' has no dataset variant '{variant}'. Available: {', '.join(task.variants)}"
        ) from exc
    templates = pd.read_csv(dataset_path)
    entities = pd.read_csv(entity_set.path)
    return expand_frames(task, entity_set, templates, entities, language, show_progress)

