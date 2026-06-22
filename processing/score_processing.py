#!/usr/bin/env python3
"""Compute configuration-driven normalized bias scores per entity."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
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


def score_setting(
    outputs_dir: Path,
    task: TaskSpec,
    language: str,
    model: str,
    few_shot: bool,
    numerical: bool,
    standardize: bool,
    min_entities: int,
    min_sentences: int,
) -> pd.DataFrame:
    frame = load_probabilities(outputs_dir, task, language, model, few_shot, numerical)
    labels = task.canonical_labels()
    weights = task.weights()
    frame[labels] = frame[labels].apply(pd.to_numeric, errors="coerce")
    nan_ratio = frame.groupby("entity_index")[labels].apply(lambda values: values.isna().sum().sum() / values.size)
    frame = frame.dropna(subset=labels, how="any")

    sentence_coverage = frame.groupby("sentence_index")["entity_index"].nunique()
    frame = frame[frame["sentence_index"].isin(sentence_coverage[sentence_coverage >= min_entities].index)]
    entity_coverage = frame.groupby("entity_index")["sentence_index"].nunique()
    frame = frame[frame["entity_index"].isin(entity_coverage[entity_coverage >= min_sentences].index)]

    frame["raw_score"] = sum(frame[label] * weights[label] for label in labels)
    sentence_stats = frame.groupby("sentence_index")["raw_score"].agg(["mean", "std"])
    frame = frame.join(sentence_stats, on="sentence_index")
    frame["deviation"] = frame["raw_score"] - frame["mean"]
    if standardize:
        frame["deviation"] = frame["deviation"] / frame["std"].replace(0, float("nan"))
    result = frame.groupby("entity_index")["deviation"].mean().to_frame("score")
    result["nan_ratio"] = nan_ratio.reindex(result.index)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--processed-outputs-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--models", nargs="+", default=PAPER_MODELS)
    parser.add_argument("--languages", nargs="+", default=PAPER_LANGUAGES)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--min-entities", type=int, default=100)
    parser.add_argument("--min-sentences", type=int, default=50)
    parser.add_argument("--no-std", dest="standardize", action="store_false")
    parser.set_defaults(standardize=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = TaskRegistry(args.config_dir)
    tasks = select_tasks(registry, args.tasks)
    args.processed_outputs_dir.mkdir(parents=True, exist_ok=True)

    for entity_set, entity_tasks in task_groups(tasks).items():
        legacy_input = args.processed_outputs_dir / f"{entity_set}.csv"
        destination = args.processed_outputs_dir / f"{entity_set}_2.csv"
        current_input = destination if destination.exists() else legacy_input
        result = base_entity_frame(registry, entity_set, current_input)
        jobs = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for task in entity_tasks:
                for language in args.languages:
                    for model_index, model in enumerate(args.models, start=1):
                        for few_shot in (False, True):
                            for numerical in (False, True):
                                if numerical and not task.supports_numerical:
                                    continue
                                column = setting_column("score", task, language, model, model_index, few_shot, numerical)
                                if column in result.columns:
                                    continue
                                future = executor.submit(
                                    score_setting,
                                    args.outputs_dir,
                                    task,
                                    language,
                                    model,
                                    few_shot,
                                    numerical,
                                    args.standardize,
                                    args.min_entities,
                                    args.min_sentences,
                                )
                                jobs.append((future, column))
            for future, column in tqdm(jobs, desc=f"Scoring {entity_set}"):
                try:
                    scores = future.result()
                except FileNotFoundError:
                    continue
                result[column] = result.index.to_series().map(scores["score"])
                result[column.replace("score_", "p_", 1)] = result.index.to_series().map(scores["nan_ratio"])
        result.to_csv(destination, index=False)
        write_processing_metadata(destination, "score", args.models, args.languages, entity_tasks)


if __name__ == "__main__":
    main()
