#!/usr/bin/env python3
"""Compute configuration-driven per-entity macro-F1 scores."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from bias_audit.config import DEFAULT_CONFIG_DIR, TaskRegistry, TaskSpec
from processing.common import (
    PAPER_LANGUAGES,
    PAPER_MODELS,
    base_entity_frame,
    load_probabilities,
    select_tasks,
    setting_column,
    task_groups,
    write_processing_metadata,
)


def f1_setting(
    outputs_dir: Path,
    task: TaskSpec,
    language: str,
    model: str,
    few_shot: bool,
    numerical: bool,
    min_entities: int,
    min_sentences: int,
) -> pd.Series:
    frame = load_probabilities(outputs_dir, task, language, model, few_shot, numerical)
    labels = task.canonical_labels()
    frame[labels] = frame[labels].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=labels, how="all")

    templates = pd.read_csv(task.datasets["main"])
    gold = templates[task.label_column].astype(str).to_dict()
    frame["gold"] = frame["sentence_index"].map(gold)
    frame = frame.dropna(subset=["gold"])
    sentence_coverage = frame.groupby("sentence_index")["entity_index"].nunique()
    frame = frame[frame["sentence_index"].isin(sentence_coverage[sentence_coverage >= min_entities].index)]
    entity_coverage = frame.groupby("entity_index")["sentence_index"].nunique()
    frame = frame[frame["entity_index"].isin(entity_coverage[entity_coverage >= min_sentences].index)]
    if frame.empty:
        return pd.Series(dtype=float)

    probabilities = frame[labels].to_numpy()
    valid = ~np.isnan(probabilities).all(axis=1)
    frame = frame.loc[valid].copy()
    frame["prediction"] = np.asarray(labels)[np.nanargmax(probabilities[valid], axis=1)]
    values = {
        entity_index: f1_score(
            group["gold"],
            group["prediction"],
            labels=labels,
            average="macro",
            zero_division=0,
        )
        for entity_index, group in frame.groupby("entity_index")
    }
    return pd.Series(values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--processed-outputs-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--models", nargs="+", default=PAPER_MODELS)
    parser.add_argument("--languages", nargs="+", default=PAPER_LANGUAGES)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--min-entities", type=int, default=100)
    parser.add_argument("--min-sentences", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = TaskRegistry(args.config_dir)
    tasks = select_tasks(registry, args.tasks)
    args.processed_outputs_dir.mkdir(parents=True, exist_ok=True)

    for entity_set, entity_tasks in task_groups(tasks).items():
        destination = args.processed_outputs_dir / f"{entity_set}_f1.csv"
        result = base_entity_frame(registry, entity_set, destination)
        jobs = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for task in entity_tasks:
                for language in args.languages:
                    for model_index, model in enumerate(args.models, start=1):
                        for few_shot in (False, True):
                            for numerical in (False, True):
                                column = setting_column("f1", task, language, model, model_index, few_shot, numerical)
                                if column in result.columns:
                                    continue
                                future = executor.submit(
                                    f1_setting,
                                    args.outputs_dir,
                                    task,
                                    language,
                                    model,
                                    few_shot,
                                    numerical,
                                    args.min_entities,
                                    args.min_sentences,
                                )
                                jobs.append((future, column))
            for future, column in tqdm(jobs, desc=f"F1 {entity_set}"):
                try:
                    scores = future.result()
                except FileNotFoundError:
                    continue
                result[column] = result.index.to_series().map(scores)
        result.to_csv(destination, index=False)
        write_processing_metadata(destination, "f1", args.models, args.languages, entity_tasks)


if __name__ == "__main__":
    main()
