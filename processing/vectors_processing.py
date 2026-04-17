#!/usr/bin/env python3
"""
Build per-entity prediction vectors (integer-encoded argmax labels) across tasks/settings and
append them as columns to processed *_vs.csv tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# -----------------------------
# Defaults / configuration (kept close to original)
# -----------------------------

COMPANY_TASKS = ["product", "risk", "violation"]
POLITICIAN_TASKS = ["leadership", "intent", "misconduct"]
COUNTRY_TASKS = ["law", "urgency", "credibility"]
TASKS = COMPANY_TASKS + POLITICIAN_TASKS + COUNTRY_TASKS

MODELS = [
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
]

LAN = ["russian", "english", "chinese"]
FS_MAP = {False: "nofs", True: "fs"}
NUM_MAP = {False: "text", True: "num"}
MODEL_MAP = {model: i + 1 for i, model in enumerate(MODELS)}

LANG_MAP = {"english": "en", "russian": "ru", "chinese": "zh"}

SETTING_MAP = {
    ("nofs", "text"): 1,
    ("fs", "text"): 2,
    ("nofs", "num"): 3,
    ("fs", "num"): 4,
}


@dataclass(frozen=True)
class PathsConfig:
    """Store base directories used for data loading and output writing."""
    data_dir: Path
    outputs_dir: Path
    processed_outputs_dir: Path


# -----------------------------
# IO helpers
# -----------------------------


def load_entities_df(paths: PathsConfig, entity_type: str) -> pd.DataFrame:
    return pd.read_csv(paths.data_dir / entity_type / "entities.csv")


def load_sentences_df(paths: PathsConfig, entity_type: str, task: str) -> pd.DataFrame:
    return pd.read_csv(paths.data_dir / entity_type / f"{task}.csv")


def load_output(
    paths: PathsConfig,
    entity_type: str,
    task: str,
    language: str,
    model: str,
    fs_flag: str,
    label_type: str,
) -> pd.DataFrame:
    """
    Load model output probability file.

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.
        language: Language folder.
        model: Model folder.
        fs_flag: 'fs' or 'nofs'.
        label_type: 'text' or 'num'.

    Returns:
        Output DataFrame.
    """
    path = (
        paths.outputs_dir
        / entity_type
        / task
        / language
        / model
        / f"output_{fs_flag}_{label_type}.csv"
    )
    return pd.read_csv(path)


# -----------------------------
# Label normalization
# -----------------------------


def infer_canonical_labels(paths: PathsConfig, entity_type: str, task: str) -> List[str]:
    """
    Infer canonical label columns from English/text/nofs reference output.

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.

    Returns:
        List of canonical label column names.
    """
    df = load_output(
        paths=paths,
        entity_type=entity_type,
        task=task,
        language="english",
        model=list(MODEL_MAP.keys())[0],
        fs_flag="nofs",
        label_type="text",
    )
    return [c for c in df.columns if c not in ["entity_index", "sentence_index"]]


def remap_to_canonical_labels(df: pd.DataFrame, canonical_labels: Sequence[str]) -> pd.DataFrame:
    """
    Rename probability columns to match canonical label names.

    Args:
        df: Output DataFrame with probability columns.
        canonical_labels: Canonical label ordering/names.

    Returns:
        DataFrame with renamed columns.
    """
    df = df.copy()
    prob_cols = [c for c in df.columns if c not in ["entity_index", "sentence_index"]]
    rename_map = {old: new for old, new in zip(prob_cols, canonical_labels)}
    return df.rename(columns=rename_map)


# -----------------------------
# Vector computation
# -----------------------------


def compute_prediction_vectors_for_setting(
    paths: PathsConfig,
    entity_type: str,
    task: str,
    language: str,
    model: str,
    fs_flag: str,
    label_type: str,
    canonical_labels: Sequence[str],
    min_non_nan: int = 100,
) -> pd.DataFrame:
    """
    Compute predicted label vectors for a single configuration.

    Returns a pivot DataFrame:
        index   -> entity_index
        columns -> sentence_index
        values  -> integer-encoded predicted labels (argmax)

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.
        language: Language name.
        model: Model name.
        fs_flag: 'fs' or 'nofs'.
        label_type: 'text' or 'num'.
        canonical_labels: Canonical label list.
        min_non_nan: Minimum non-NaN predictions required per entity.

    Returns:
        Pivot DataFrame entity_index x sentence_index of predicted label integers.
    """
    outputs_df = load_output(
        paths=paths,
        entity_type=entity_type,
        task=task,
        language=language,
        model=model,
        fs_flag=fs_flag,
        label_type=label_type,
    )

    outputs_df["entity_index"] = outputs_df["entity_index"].astype(int)
    outputs_df["sentence_index"] = outputs_df["sentence_index"].astype(int)

    if not (language == "english" and label_type == "text"):
        outputs_df = remap_to_canonical_labels(outputs_df, canonical_labels)

    prob_cols = list(canonical_labels)
    probs = outputs_df[prob_cols]
    predicted_label = probs.idxmax(axis=1)

    label_to_int = {lab: i for i, lab in enumerate(canonical_labels)}
    predicted_int = predicted_label.map(label_to_int)

    outputs_df["predicted_int"] = predicted_int
    outputs_df = (
        outputs_df.sort_values("predicted_int", na_position="last")
        .drop_duplicates(subset=["entity_index", "sentence_index"], keep="first")
    )

    valid_entities = (
        outputs_df["predicted_int"].notna().groupby(outputs_df["entity_index"]).sum() >= min_non_nan
    )
    valid_entities = valid_entities[valid_entities].index
    outputs_df = outputs_df[outputs_df["entity_index"].isin(valid_entities)]

    pivot = outputs_df.pivot(
        index="entity_index",
        columns="sentence_index",
        values="predicted_int",
    )
    return pivot


def vector_column_name(language: str, fs_flag: str, label_type: str, model: str, task: str) -> str:
    """
    Construct standardized vector column name for a configuration.

    Args:
        language: Language name.
        fs_flag: 'fs' or 'nofs'.
        label_type: 'text' or 'num'.
        model: Model name.
        task: Task name.

    Returns:
        Column name string.
    """
    lan = LANG_MAP[language]
    setting = SETTING_MAP[(fs_flag, label_type)]
    m = MODEL_MAP[model]
    return f"v_{lan}_s{setting}_m{m}_{task}"


def compute_all_entity_prediction_vectors(
    paths: PathsConfig,
    input_dfs: Mapping[str, pd.DataFrame],
    company_tasks: Sequence[str],
    politician_tasks: Sequence[str],
    country_tasks: Sequence[str],
    models: Sequence[str],
    languages: Sequence[str],
    num_flags: Sequence[bool],
    fs_flags: Sequence[bool] = (True, False),
    max_workers: int = 6,
    min_non_nan: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Compute prediction vectors for all entity types and merge into input DataFrames.

    Args:
        paths: Path configuration.
        input_dfs: Mapping from entity_type -> existing DataFrame (index should align to entity_index).
        company_tasks: Company tasks list.
        politician_tasks: Politician tasks list.
        country_tasks: Country tasks list.
        models: Model list.
        languages: Language list.
        num_flags: Iterable of booleans indicating label-type (True->num, False->text).
        fs_flags: Iterable of booleans indicating few-shot.
        max_workers: ThreadPoolExecutor worker count.
        min_non_nan: Minimum non-NaN predictions per entity before keeping it.

    Returns:
        Mapping from entity_type -> updated DataFrame.
    """
    entity_tasks_map = {
        "companies": list(company_tasks),
        "politicians": list(politician_tasks),
        "countries": list(country_tasks),
    }

    result_dfs: Dict[str, pd.DataFrame] = {}

    for entity_type, tasks in entity_tasks_map.items():
        final_df = input_dfs[entity_type].copy()
        existing_columns = set(final_df.columns)

        jobs: List[Tuple[str, str, str, str, str, List[str], str]] = []
        for task in tasks:
            canonical_labels = infer_canonical_labels(paths, entity_type, task)
            for language in languages:
                for model in models:
                    for fs_bool in fs_flags:
                        for num_bool in num_flags:

                            fs_flag = FS_MAP[fs_bool]
                            label_type = NUM_MAP[num_bool]
                            col_name = vector_column_name(language, fs_flag, label_type, model, task)

                            if col_name in existing_columns:
                                continue

                            jobs.append((task, language, model, fs_flag, label_type, canonical_labels, col_name))

        if not jobs:
            result_dfs[entity_type] = final_df
            continue

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    compute_prediction_vectors_for_setting,
                    paths,
                    entity_type,
                    task,
                    language,
                    model,
                    fs_flag,
                    label_type,
                    canonical_labels,
                    min_non_nan,
                ): col_name
                for (task, language, model, fs_flag, label_type, canonical_labels, col_name) in jobs
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {entity_type} vectors",
            ):
                col_name = futures[future]
                try:
                    pivot = future.result()
                    vectors = pivot.apply(lambda r: r.values.tolist(), axis=1)
                    final_df[col_name] = final_df.index.map(vectors)
                except FileNotFoundError:
                    pass

        result_dfs[entity_type] = final_df

    return result_dfs


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed args namespace.
    """
    p = argparse.ArgumentParser(
        description="Compute per-entity prediction vectors and update processed *_vs.csv files."
    )
    p.add_argument("--data-dir", type=Path, required=True, help="Base data directory.")
    p.add_argument("--outputs-dir", type=Path, required=True, help="Base outputs directory.")
    p.add_argument("--processed-outputs-dir", type=Path, required=True, help="Directory for processed *_vs.csv.")
    p.add_argument("--max-workers", type=int, default=6, help="ThreadPoolExecutor worker count.")
    p.add_argument("--min-non-nan", type=int, default=100, help="Minimum non-NaN predictions per entity.")
    return p.parse_args()


def main() -> None:
    """
    Load processed *_vs.csv, compute missing vector columns, and overwrite *_vs.csv files.
    """
    args = parse_args()
    paths = PathsConfig(
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        processed_outputs_dir=args.processed_outputs_dir,
    )

    companies_df = pd.read_csv(paths.processed_outputs_dir / "companies_vs.csv")
    politicians_df = pd.read_csv(paths.processed_outputs_dir / "politicians_vs.csv")
    countries_df = pd.read_csv(paths.processed_outputs_dir / "countries_vs.csv")

    dfs_in = {
        "companies": companies_df,
        "politicians": politicians_df,
        "countries": countries_df,
    }

    dfs_out = compute_all_entity_prediction_vectors(
        paths=paths,
        input_dfs=dfs_in,
        company_tasks=COMPANY_TASKS,
        politician_tasks=POLITICIAN_TASKS,
        country_tasks=COUNTRY_TASKS,
        models=MODELS,
        languages=LAN,
        num_flags=[True, False],
        fs_flags=[True, False],
        max_workers=args.max_workers,
        min_non_nan=args.min_non_nan,
    )

    dfs_out["companies"].to_csv(paths.processed_outputs_dir / "companies_vs.csv", index=False)
    dfs_out["politicians"].to_csv(paths.processed_outputs_dir / "politicians_vs.csv", index=False)
    dfs_out["countries"].to_csv(paths.processed_outputs_dir / "countries_vs.csv", index=False)


if __name__ == "__main__":
    main()
