#!/usr/bin/env python3
"""
Compute per-entity macro-F1 scores across entity types, tasks, languages,
models, and prompting/label settings, and update processed output CSV files.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm


DEFAULT_COMPANY_TASKS = ["product", "risk", "violation", "sentimentCom"]
DEFAULT_POLITICIAN_TASKS = ["leadership", "intent", "misconduct", "sentimentP"]
DEFAULT_COUNTRY_TASKS = ["law", "urgency", "credibility", "sentimentCou"]

DEFAULT_MODELS = [
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

DEFAULT_LANGUAGES = ["russian", "english", "chinese"]

FS_MAP = {False: "nofs", True: "fs"}
NUM_MAP = {False: "text", True: "num"}
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


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def load_sentences_df(paths: PathsConfig, entity_type: str, task: str) -> pd.DataFrame:
    return read_csv_required(paths.data_dir / entity_type / f"{task}.csv")


def load_output_df(
    paths: PathsConfig,
    entity_type: str,
    task: str,
    language: str,
    model: str,
    fs_flag: str,
    label_type: str,
) -> pd.DataFrame:
    """
    Load model output probabilities for a specific configuration.

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.
        language: Language name.
        model: Model name.
        fs_flag: Few-shot flag ('fs' or 'nofs').
        label_type: Label format ('text' or 'num').

    Returns:
        DataFrame with prediction probabilities.
    """
    path = (
        paths.outputs_dir
        / entity_type
        / task
        / language
        / model
        / f"output_{fs_flag}_{label_type}.csv"
    )
    return read_csv_required(path)


def infer_canonical_labels(
    paths: PathsConfig,
    entity_type: str,
    task: str,
    reference_model: str,
) -> List[str]:
    """
    Infer canonical label column names from a reference configuration.

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.
        reference_model: Model used to extract label schema.

    Returns:
        List of canonical label column names.
    """
    df = load_output_df(
        paths,
        entity_type,
        task,
        language="english",
        model=reference_model,
        fs_flag="nofs",
        label_type="text",
    )
    return [c for c in df.columns if c not in ["entity_index", "sentence_index"]]


def remap_to_canonical_labels(
    df: pd.DataFrame,
    canonical_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Rename probability columns to match canonical label names.

    Args:
        df: DataFrame with probability columns.
        canonical_labels: Target label ordering.

    Returns:
        DataFrame with renamed columns.
    """
    df = df.copy()
    prob_cols = [c for c in df.columns if c not in ["entity_index", "sentence_index"]]
    rename_map = {old: new for old, new in zip(prob_cols, canonical_labels)}
    return df.rename(columns=rename_map)


def apply_filters(
    df: pd.DataFrame,
    min_entities_per_sentence: int,
    min_sentences_per_entity: int,
) -> pd.DataFrame:
    """
    Filter rows based on minimum coverage constraints.

    Args:
        df: Merged DataFrame with entity and sentence indices.
        min_entities_per_sentence: Minimum distinct entities per sentence.
        min_sentences_per_entity: Minimum distinct sentences per entity.

    Returns:
        Filtered DataFrame.
    """
    sent_counts = df.groupby("sentence_index")["entity_index"].nunique()
    valid_sents = sent_counts[sent_counts >= min_entities_per_sentence].index
    df = df[df["sentence_index"].isin(valid_sents)]

    ent_counts = df.groupby("entity_index")["sentence_index"].nunique()
    valid_ents = ent_counts[ent_counts >= min_sentences_per_entity].index
    return df[df["entity_index"].isin(valid_ents)]


def f1_column_name(
    language: str,
    fs_flag: str,
    label_type: str,
    model: str,
    task: str,
    model_map: Mapping[str, int],
) -> str:
    """
    Construct standardized column name for an F1 score.

    Args:
        language: Language name.
        fs_flag: Few-shot flag.
        label_type: Label format.
        model: Model name.
        task: Task name.
        model_map: Mapping from model name to index.

    Returns:
        Column name string.
    """
    lan = LANG_MAP[language]
    setting = SETTING_MAP[(fs_flag, label_type)]
    m = model_map[model]
    return f"f1_{lan}_s{setting}_m{m}_{task}"


def compute_f1_for_setting(
    paths: PathsConfig,
    entity_type: str,
    task: str,
    language: str,
    model: str,
    fs_flag: str,
    label_type: str,
    canonical_labels: Sequence[str],
    min_entities: int,
    min_sentences: int,
) -> pd.DataFrame:
    """
    Compute macro-F1 per entity for a single configuration.

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.
        language: Language name.
        model: Model name.
        fs_flag: Few-shot flag.
        label_type: Label format.
        canonical_labels: Canonical label list.
        min_entities: Minimum entities per sentence.
        min_sentences: Minimum sentences per entity.

    Returns:
        DataFrame indexed by entity_index with a single column 'f1'.
    """
    outputs_df = load_output_df(
        paths, entity_type, task, language, model, fs_flag, label_type
    )
    sentences_df = load_sentences_df(paths, entity_type, task)

    outputs_df["entity_index"] = outputs_df["entity_index"].astype(int)
    outputs_df["sentence_index"] = outputs_df["sentence_index"].astype(int)
    sentences_df.index = sentences_df.index.astype(int)
    sentences_df["label"] = sentences_df["label"].astype(str)

    if not (language == "english" and label_type == "text"):
        outputs_df = remap_to_canonical_labels(outputs_df, canonical_labels)

    outputs_df[canonical_labels] = outputs_df[canonical_labels].astype(float)
    outputs_df = outputs_df.dropna(subset=canonical_labels, how="all")
    if outputs_df.empty:
        return pd.DataFrame({"f1": []})

    sentence_label_map = sentences_df["label"].to_dict()
    outputs_df["label"] = outputs_df["sentence_index"].map(sentence_label_map)
    outputs_df = outputs_df.dropna(subset=["label"])
    if outputs_df.empty:
        return pd.DataFrame({"f1": []})

    outputs_df = apply_filters(outputs_df, min_entities, min_sentences)
    if outputs_df.empty:
        return pd.DataFrame({"f1": []})

    probs = outputs_df[canonical_labels].to_numpy()
    idxs = np.nanargmax(probs, axis=1)
    outputs_df["predicted_label"] = np.array(canonical_labels)[idxs]

    f1s = {}
    for ent, g in outputs_df.groupby("entity_index"):
        f1s[ent] = f1_score(
            g["label"],
            g["predicted_label"],
            labels=canonical_labels,
            average="macro",
            zero_division=0,
        )

    return pd.DataFrame({"f1": f1s})


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute per-entity macro-F1 scores and update processed CSV files."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Base data directory.")
    parser.add_argument("--outputs-dir", type=Path, required=True, help="Model outputs directory.")
    parser.add_argument(
        "--processed-outputs-dir",
        type=Path,
        required=True,
        help="Directory containing processed *_f1.csv files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--min-entities",
        type=int,
        default=100,
        help="Minimum entities per sentence for filtering.",
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=100,
        help="Minimum sentences per entity for filtering.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Execute the full F1 computation pipeline from CLI arguments.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    paths = PathsConfig(
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        processed_outputs_dir=args.processed_outputs_dir,
    )

    model_map = {model: i + 1 for i, model in enumerate(DEFAULT_MODELS)}

    input_dfs = {
        "companies": read_csv_optional(paths.processed_outputs_dir / "companies_f1.csv")
        or pd.DataFrame(),
        "politicians": read_csv_optional(paths.processed_outputs_dir / "politicians_f1.csv")
        or pd.DataFrame(),
        "countries": read_csv_optional(paths.processed_outputs_dir / "countries_f1.csv")
        or pd.DataFrame(),
    }

    entity_tasks_map = {
        "companies": DEFAULT_COMPANY_TASKS,
        "politicians": DEFAULT_POLITICIAN_TASKS,
        "countries": DEFAULT_COUNTRY_TASKS,
    }

    for entity_type, tasks in entity_tasks_map.items():
        final_df = input_dfs[entity_type].copy()
        existing_columns = set(final_df.columns)

        jobs: List[Tuple] = []
        for task in tasks:
            canonical_labels = infer_canonical_labels(
                paths, entity_type, task, DEFAULT_MODELS[0]
            )
            for language in DEFAULT_LANGUAGES:
                for model in DEFAULT_MODELS:
                    for fs_bool in [True, False]:
                        for num_bool in [True, False]:
                            fs_flag = FS_MAP[fs_bool]
                            label_type = NUM_MAP[num_bool]
                            col = f1_column_name(
                                language, fs_flag, label_type, model, task, model_map
                            )
                            if col in existing_columns:
                                continue
                            jobs.append(
                                (
                                    entity_type,
                                    task,
                                    language,
                                    model,
                                    fs_flag,
                                    label_type,
                                    canonical_labels,
                                    col,
                                )
                            )

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(
                    compute_f1_for_setting,
                    paths,
                    et,
                    task,
                    language,
                    model,
                    fs_flag,
                    label_type,
                    canonical_labels,
                    args.min_entities,
                    args.min_sentences,
                ): col
                for (et, task, language, model, fs_flag, label_type, canonical_labels, col) in jobs
            }

            for future in tqdm(as_completed(future_map), total=len(future_map)):
                col = future_map[future]
                try:
                    f1_df = future.result()
                    final_df = final_df.merge(
                        f1_df["f1"],
                        left_index=True,
                        right_index=True,
                        how="left",
                    ).rename(columns={"f1": col})
                except FileNotFoundError:
                    continue

        final_df.to_csv(
            paths.processed_outputs_dir / f"{entity_type}_f1.csv", index=False
        )


if __name__ == "__main__":
    main()
