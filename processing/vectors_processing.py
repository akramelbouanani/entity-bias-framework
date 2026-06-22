#!/usr/bin/env python3
"""Build per-entity vectors of configured task predictions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
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


def vector_setting(
    outputs_dir: Path,
    task: TaskSpec,
    language: str,
    model: str,
    few_shot: bool,
    numerical: bool,
    min_non_nan: int,
) -> pd.Series:
    frame = load_probabilities(outputs_dir, task, language, model, few_shot, numerical)
    labels = task.canonical_labels()
    frame[labels] = frame[labels].apply(pd.to_numeric, errors="coerce")
    frame["prediction"] = frame[labels].idxmax(axis=1).map({label: index for index, label in enumerate(labels)})
    frame = frame.sort_values("prediction", na_position="last").drop_duplicates(
        ["entity_index", "sentence_index"], keep="first"
    )
    coverage = frame.groupby("entity_index")["prediction"].count()
    frame = frame[frame["entity_index"].isin(coverage[coverage >= min_non_nan].index)]
    pivot = frame.pivot(index="entity_index", columns="sentence_index", values="prediction")
    return pivot.apply(lambda row: json.dumps(row.tolist(), allow_nan=True), axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--processed-outputs-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--models", nargs="+", default=PAPER_MODELS)
    parser.add_argument("--languages", nargs="+", default=PAPER_LANGUAGES)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--min-non-nan", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = TaskRegistry(args.config_dir)
    tasks = select_tasks(registry, args.tasks)
    args.processed_outputs_dir.mkdir(parents=True, exist_ok=True)

    for entity_set, entity_tasks in task_groups(tasks).items():
        destination = args.processed_outputs_dir / f"{entity_set}_vs.csv"
        result = base_entity_frame(registry, entity_set, destination)
        jobs = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for task in entity_tasks:
                for language in args.languages:
                    for model_index, model in enumerate(args.models, start=1):
                        for few_shot in (False, True):
                            for numerical in (False, True):
                                column = setting_column("v", task, language, model, model_index, few_shot, numerical)
                                if column in result.columns:
                                    continue
                                future = executor.submit(
                                    vector_setting,
                                    args.outputs_dir,
                                    task,
                                    language,
                                    model,
                                    few_shot,
                                    numerical,
                                    args.min_non_nan,
                                )
                                jobs.append((future, column))
            for future, column in tqdm(jobs, desc=f"Vectors {entity_set}"):
                try:
                    vectors = future.result()
                except FileNotFoundError:
                    continue
                result[column] = result.index.to_series().map(vectors)
        result.to_csv(destination, index=False)
        write_processing_metadata(destination, "vector", args.models, args.languages, entity_tasks)


if __name__ == "__main__":
    main()
