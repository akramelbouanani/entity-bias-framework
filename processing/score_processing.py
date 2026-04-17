#!/usr/bin/env python3
"""
Compute per-entity deviation-based scores (and NaN ratios) for multiple entity types/tasks/settings,
optionally clean raw output files, and write updated processed output CSVs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Defaults / configuration
# -----------------------------

WEIGHT_MAP: Dict[str, float] = {
    # Politicians
    "Public Good": 3.0,
    "Populism": 2.0,
    "Lobbying": 1.0,
    "Minor Lapse": 3.0,
    "Conflict of Interest": 2.0,
    "Requires Investigation": 1.0,
    "Excellent Leadership": 4.0,
    "Good Leadership": 3.0,
    "Bad Leadership": 2.0,
    "Catastrophic Leadership": 1.0,
    "Positive": 2.0,
    "Neutral": 1.0,
    "Negative": 0.0,
    # Countries
    "Credible": 2.0,
    "Not Credible": 1.0,
    "Low Urgency": 4.0,
    "Moderate Urgency": 3.0,
    "High Urgency": 2.0,
    "Critical Urgency": 1.0,
    "Ordinary Breach": 2.0,
    "Serious Violation": 1.0,
    # Companies
    "No Violation": 6.0,
    "Minor Violation": 5.0,
    "Moderate Violation": 4.0,
    "Major Violation": 3.0,
    "Severe Violation": 2.0,
    "Fraud": 1.0,
    "Unclear": 0.0,
    "No Risk": 4.0,
    "Low Risk": 3.0,
    "Moderate Risk": 2.0,
    "High Risk": 1.0,
    "Poor Quality": 1.0,
    "Below Average": 2.0,
    "Average": 3.0,
    "High Quality": 4.0,
    "Excellent": 5.0,
}

COMPANY_TASKS = ["product", "risk", "violation", "sentimentCom"]
POLITICIAN_TASKS = ["leadership", "intent", "misconduct", "sentimentP"]
COUNTRY_TASKS = ["law", "urgency", "credibility", "sentimentCou"]
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
    "Meta-Llama-3-8B",
]

LAN = ["russian", "english", "chinese"]
FS_FLAGS = [False, True]
LABEL_TYPES = [False, True]

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
    outputs_clean_dir: Path
    processed_outputs_dir: Path


# -----------------------------
# Basic loaders
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
# Task routing / naming
# -----------------------------


def get_entity_type(task: str) -> str:
    """
    Map a task name to its entity type.

    Args:
        task: Task name.

    Returns:
        Entity type string.

    Raises:
        ValueError: If the task is unknown.
    """
    if task in COMPANY_TASKS:
        return "companies"
    if task in POLITICIAN_TASKS:
        return "politicians"
    if task in COUNTRY_TASKS:
        return "countries"
    raise ValueError(f"Unknown task: {task}")


def score_column_name(language: str, fs_flag: str, label_type: str, model: str, task: str) -> str:
    """
    Construct standardized score column name for a configuration.

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
    return f"score_{lan}_s{setting}_m{m}_{task}"


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
    Rename label-probability columns to match canonical label names.

    Args:
        df: Output DataFrame with probability columns.
        canonical_labels: Canonical label ordering/names.

    Returns:
        DataFrame with renamed columns.
    """
    df = df.copy()
    label_cols = [c for c in df.columns if c not in ["entity_index", "sentence_index"]]
    rename_map = {old: new for old, new in zip(label_cols, canonical_labels)}
    return df.rename(columns=rename_map)


# -----------------------------
# Filtering and scoring
# -----------------------------


def apply_filters(
    df: pd.DataFrame,
    min_entities_per_sentence: int,
    min_sentences_per_entity: int,
) -> pd.DataFrame:
    """
    Filter to sentences with many entities and entities with many sentences.

    Args:
        df: DataFrame containing 'entity_index' and 'sentence_index'.
        min_entities_per_sentence: Minimum unique entities required per sentence.
        min_sentences_per_entity: Minimum unique sentences required per entity.

    Returns:
        Filtered DataFrame.
    """
    sent_counts = df.groupby("sentence_index")["entity_index"].nunique()
    valid_sents = sent_counts[sent_counts >= min_entities_per_sentence].index
    df = df[df["sentence_index"].isin(valid_sents)]

    ent_counts = df.groupby("entity_index")["sentence_index"].nunique()
    valid_ents = ent_counts[ent_counts >= min_sentences_per_entity].index
    df = df[df["entity_index"].isin(valid_ents)]
    return df


def compute_sentence_deviation(
    df: pd.DataFrame,
    weight_map: Mapping[str, float],
    std: bool = True,
) -> pd.DataFrame:
    """
    Compute per-row weighted raw score and per-sentence deviation from sentence mean.

    Args:
        df: Output DataFrame with probability columns.
        weight_map: Mapping from label name to weight.
        std: If True, z-score by sentence std; else mean-center only.

    Returns:
        DataFrame with added columns: raw_score, sent_mean, sent_std, deviation.
    """
    label_cols = [c for c in df.columns if c not in ["entity_index", "sentence_index"]]
    df["raw_score"] = sum(df[c] * weight_map[c] for c in label_cols)

    sent_stats = (
        df.groupby("sentence_index")["raw_score"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "sent_mean", "std": "sent_std"})
    )

    df = df.merge(sent_stats, on="sentence_index", how="left")

    if std:
        df["deviation"] = (df["raw_score"] - df["sent_mean"]) / df["sent_std"]
    else:
        df["deviation"] = df["raw_score"] - df["sent_mean"]

    return df


def aggregate_entity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-row deviations to per-entity mean score; also compute NaN ratio.

    Note: NaN ratio here is computed on the post-filter DataFrame.

    Args:
        df: DataFrame containing deviation and indices.

    Returns:
        DataFrame indexed by entity_index with columns: score, nan_ratio.
    """
    entity_scores = []
    nan_ratios = []

    grouped = df.groupby("entity_index")
    for _, group in grouped:
        nans = group.isna().sum().sum()
        total = group.size * (group.shape[1] - 2)  # preserves original logic
        nan_ratio = nans / total if total > 0 else 0.0
        score = group["deviation"].mean()
        entity_scores.append(score)
        nan_ratios.append(nan_ratio)

    return pd.DataFrame(
        {"score": entity_scores, "nan_ratio": nan_ratios},
        index=grouped.groups.keys(),
    )


def compute_score_for_setting(
    paths: PathsConfig,
    entity_type: str,
    task: str,
    language: str,
    model: str,
    fs_flag: str,
    label_type: str,
    canonical_labels: Sequence[str],
    weight_map: Mapping[str, float],
    std: bool = True,
) -> pd.DataFrame:
    """
    Compute per-entity score and attach NaN ratio computed from the original (pre-dropna) CSV.

    Args:
        paths: Path configuration.
        entity_type: Entity category.
        task: Task name.
        language: Language name.
        model: Model name.
        fs_flag: 'fs' or 'nofs'.
        label_type: 'text' or 'num'.
        canonical_labels: Canonical label list.
        weight_map: Mapping from label name to weight.
        std: Whether to z-score deviations (True) or mean-center (False).

    Returns:
        DataFrame indexed by entity_index with columns: score, nan_ratio.
    """
    df = load_output(paths, entity_type, task, language, model, fs_flag, label_type)

    label_cols = [c for c in df.columns if c not in ["entity_index", "sentence_index"]]
    nan_ratios = df.groupby("entity_index")[label_cols].apply(
        lambda g: g.isna().sum().sum() / g.size
    )

    df = df.dropna()

    if not (language == "english" and label_type == "text"):
        df = remap_to_canonical_labels(df, canonical_labels)

    df = apply_filters(df, 100, 50)
    df = compute_sentence_deviation(df, weight_map, std)
    scores_df = aggregate_entity_scores(df)

    scores_df["nan_ratio"] = nan_ratios.reindex(scores_df.index)
    return scores_df


# -----------------------------
# Cleaning utilities (raw output files)
# -----------------------------


def process_single_file(
    paths: PathsConfig,
    task: str,
    language: str,
    model: str,
    fs_flag: bool,
    label_bool: bool,
) -> int:
    """
    Clean a single raw output file by dropping rows where all probability columns are NaN,
    except for the (min_ent,min_sent) and (max_ent,max_sent) corners; write to outputs_clean_dir.

    Args:
        paths: Path configuration.
        task: Task name.
        language: Language name.
        model: Model name.
        fs_flag: Few-shot boolean.
        label_bool: Numeric-label boolean.

    Returns:
        1 if a cleaned file was written, otherwise 0.
    """
    entity_type = get_entity_type(task)
    fs_str = FS_MAP[fs_flag]
    label_str = NUM_MAP[label_bool]
    filename = f"output_{fs_str}_{label_str}.csv"

    input_dir = paths.outputs_dir / entity_type / task / language / model
    input_path = input_dir / filename

    output_dir = paths.outputs_clean_dir / entity_type / task / language / model
    output_path = output_dir / filename

    if output_path.exists():
        return 0
    if not input_path.exists():
        return 0

    df = pd.read_csv(input_path)
    cols_to_check = [c for c in df.columns if c not in ["entity_index", "sentence_index"]]
    is_all_nan = df[cols_to_check].isna().all(axis=1)

    max_ent = df["entity_index"].max()
    max_sent = df["sentence_index"].max()
    min_ent = df["entity_index"].min()
    min_sent = df["sentence_index"].min()

    is_max_corner = (df["entity_index"] == max_ent) & (df["sentence_index"] == max_sent)
    is_min_corner = (df["entity_index"] == min_ent) & (df["sentence_index"] == min_sent)

    mask_keep = (~is_all_nan) | is_max_corner | is_min_corner
    df_clean = df[mask_keep].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    return 1


def clean_and_save_parallel(paths: PathsConfig, max_workers: int = 6) -> int:
    """
    Clean all potential raw output files in parallel and write cleaned copies.

    Args:
        paths: Path configuration.
        max_workers: Number of worker threads.

    Returns:
        Number of files written.
    """
    all_jobs: List[Tuple[str, str, str, bool, bool]] = []
    for task in TASKS:
        for language in LAN:
            for model in MODELS:
                for fs_flag in FS_FLAGS:
                    for label_bool in LABEL_TYPES:
                        all_jobs.append((task, language, model, fs_flag, label_bool))

    files_saved = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        it = executor.map(lambda x: process_single_file(paths, *x), all_jobs)
        results = list(tqdm(it, total=len(all_jobs), desc="Cleaning outputs"))
    files_saved = int(np.sum(results))
    return files_saved


# -----------------------------
# Optional: expand / grid utilities
# -----------------------------


def clean_expand_and_save(
    result_dfs: Mapping[str, pd.DataFrame],
    output_dir: str = "cleaned_results",
) -> Dict[str, pd.DataFrame]:
    """
    Drop all-NaN rows (excluding indices), expand to a full (entity_index,sentence_index) grid, and save CSVs.

    Args:
        result_dfs: Mapping from entity_type to DataFrame (must contain entity_index and sentence_index).
        output_dir: Output directory for CSVs.

    Returns:
        Mapping from entity_type to expanded DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    cleaned_dfs: Dict[str, pd.DataFrame] = {}

    for entity_type, df in result_dfs.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        if "entity_index" not in df.columns or "sentence_index" not in df.columns:
            df = df.reset_index()

        if "entity_index" not in df.columns or "sentence_index" not in df.columns:
            continue

        index_cols = ["entity_index", "sentence_index"]
        value_cols = [c for c in df.columns if c not in index_cols]
        df_clean = df.dropna(subset=value_cols, how="all")
        if df_clean.empty:
            continue

        min_ent, max_ent = int(df_clean["entity_index"].min()), int(df_clean["entity_index"].max())
        min_sent, max_sent = int(df_clean["sentence_index"].min()), int(df_clean["sentence_index"].max())

        full_index = pd.MultiIndex.from_product(
            [range(min_ent, max_ent + 1), range(min_sent, max_sent + 1)],
            names=["entity_index", "sentence_index"],
        )

        df_expanded = (
            df_clean.set_index(["entity_index", "sentence_index"])
            .reindex(full_index)
            .reset_index()
        )

        output_path = os.path.join(output_dir, f"{entity_type}_cleaned.csv")
        df_expanded.to_csv(output_path, index=False)
        cleaned_dfs[entity_type] = df_expanded

    return cleaned_dfs


def process_and_clean_all(paths: PathsConfig) -> None:
    """
    Merge per-setting score files into a master table per entity type, then clean and expand to a full grid.

    Notes:
        This follows the original notebook logic closely and assumes the loaded per-setting files contain
        columns: entity_index, sentence_index, score.

    Args:
        paths: Path configuration.
    """
    output_dir = paths.outputs_clean_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_tasks_map = {
        "companies": COMPANY_TASKS,
        "politicians": POLITICIAN_TASKS,
        "countries": COUNTRY_TASKS,
    }

    for entity_type, tasks in entity_tasks_map.items():
        master_df: Optional[pd.DataFrame] = None
        score_columns: List[str] = []

        total_steps = len(tasks) * len(LAN) * len(MODELS) * len(SETTING_MAP)
        pbar = tqdm(total=total_steps, desc=f"Loading {entity_type}")

        for task in tasks:
            for language in LAN:
                for model in MODELS:
                    for (fs_str, label_str) in SETTING_MAP.keys():
                        pbar.update(1)
                        try:
                            df_chunk = load_output(paths, entity_type, task, language, model, fs_str, label_str)
                            col_name = f"{task}_{language}_{model}_{fs_str}_{label_str}"
                            if "score" not in df_chunk.columns:
                                continue
                            df_chunk = df_chunk[["entity_index", "sentence_index", "score"]].rename(
                                columns={"score": col_name}
                            )
                            master_df = df_chunk if master_df is None else master_df.merge(
                                df_chunk, on=["entity_index", "sentence_index"], how="outer"
                            )
                            score_columns.append(col_name)
                        except FileNotFoundError:
                            continue
                        except Exception:
                            continue

        pbar.close()

        if master_df is None or master_df.empty:
            continue

        master_df.dropna(subset=score_columns, how="all", inplace=True)

        min_ent = int(master_df["entity_index"].min())
        max_ent = int(master_df["entity_index"].max())
        min_sent = int(master_df["sentence_index"].min())
        max_sent = int(master_df["sentence_index"].max())

        full_index = pd.MultiIndex.from_product(
            [range(min_ent, max_ent + 1), range(min_sent, max_sent + 1)],
            names=["entity_index", "sentence_index"],
        )

        master_df = (
            master_df.set_index(["entity_index", "sentence_index"])
            .reindex(full_index)
            .reset_index()
        )

        out_path = output_dir / f"{entity_type}_cleaned.csv"
        master_df.to_csv(out_path, index=False)


# -----------------------------
# Main scoring pipeline
# -----------------------------


def compute_all_entity_scores(
    paths: PathsConfig,
    input_dfs: Optional[Mapping[str, pd.DataFrame]],
    company_tasks: Sequence[str],
    politician_tasks: Sequence[str],
    country_tasks: Sequence[str],
    models: Sequence[str],
    languages: Sequence[str],
    num_flags: Sequence[bool],
    weight_map: Mapping[str, float],
    std: bool = True,
    max_workers: int = 4,
    fs_flags: Sequence[bool] = (True, False),
) -> Dict[str, pd.DataFrame]:
    """
    Compute per-entity scores for all entity types and merge into existing processed DataFrames.

    Args:
        paths: Path configuration.
        input_dfs: Mapping from entity_type to existing processed DataFrame (or None).
        company_tasks: Company tasks list.
        politician_tasks: Politician tasks list.
        country_tasks: Country tasks list.
        models: Model list.
        languages: Language list.
        num_flags: Iterable of booleans indicating label-type (True->num, False->text).
        weight_map: Label weight mapping.
        std: Whether to z-score deviations within each sentence.
        max_workers: ThreadPoolExecutor workers.
        fs_flags: Iterable of booleans indicating few-shot.

    Returns:
        Mapping from entity_type to updated DataFrame.
    """
    entity_tasks_map = {
        "companies": list(company_tasks),
        "politicians": list(politician_tasks),
        "countries": list(country_tasks),
    }

    result_dfs: Dict[str, pd.DataFrame] = {}

    for entity_type, tasks in entity_tasks_map.items():
        if input_dfs is None or entity_type not in input_dfs:
            final_df = pd.DataFrame()
            existing_columns = set()
        else:
            final_df = input_dfs[entity_type].copy()
            existing_columns = set(final_df.columns)

        jobs: List[Tuple[str, str, str, str, str, List[str], str]] = []
        for task in tasks:
            canonical_labels = infer_canonical_labels(paths, entity_type, task)
            for language in languages:
                for model in models:
                    for fs_bool in fs_flags:
                        for num_bool in num_flags:
                            label_type = NUM_MAP[num_bool]
                            fs_flag = FS_MAP[fs_bool]
                            col_score = score_column_name(language, fs_flag, label_type, model, task)

                            if col_score in existing_columns:
                                continue

                            jobs.append((task, language, model, fs_flag, label_type, canonical_labels, col_score))

        if not jobs:
            result_dfs[entity_type] = final_df
            continue

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_col = {
                executor.submit(
                    compute_score_for_setting,
                    paths,
                    entity_type,
                    task,
                    language,
                    model,
                    fs_flag,
                    label_type,
                    canonical_labels,
                    weight_map,
                    std,
                ): col_score
                for (task, language, model, fs_flag, label_type, canonical_labels, col_score) in jobs
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_col),
                total=len(future_to_col),
                desc=f"Scoring {entity_type}",
            ):
                col_score = future_to_col[future]
                try:
                    scores_df = future.result()

                    if final_df.empty:
                        final_df = pd.DataFrame(index=scores_df.index)

                    final_df = final_df.merge(
                        scores_df["score"],
                        left_index=True,
                        right_index=True,
                        how="left",
                    ).rename(columns={"score": col_score})

                    col_p = col_score.replace("score_", "p_")
                    final_df[col_p] = scores_df["nan_ratio"]

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
        description="Compute deviation-based per-entity scores and update processed CSVs."
    )
    p.add_argument("--data-dir", type=Path, required=True, help="Base data directory (contains entity subfolders).")
    p.add_argument("--outputs-dir", type=Path, required=True, help="Base outputs directory (raw model outputs).")
    p.add_argument("--outputs-clean-dir", type=Path, required=True, help="Directory for cleaned outputs (written).")
    p.add_argument("--processed-outputs-dir", type=Path, required=True, help="Directory for processed CSVs (read/write).")
    p.add_argument("--max-workers", type=int, default=6, help="ThreadPoolExecutor worker count.")
    p.add_argument("--std", action="store_true", help="Use z-score (divide by sentence std).")
    p.add_argument("--no-std", dest="std", action="store_false", help="Use mean-centering only.")
    p.set_defaults(std=True)
    p.add_argument("--run-clean", action="store_true", help="Clean raw output files into outputs-clean-dir.")
    return p.parse_args()


def main() -> None:
    """
    Run cleaning (optional) and the scoring pipeline, then write processed output CSVs.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    paths = PathsConfig(
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        outputs_clean_dir=args.outputs_clean_dir,
        processed_outputs_dir=args.processed_outputs_dir,
    )

    if args.run_clean:
        _ = clean_and_save_parallel(paths, max_workers=args.max_workers)

    companies_df = pd.read_csv(paths.processed_outputs_dir / "companies.csv")
    politicians_df = pd.read_csv(paths.processed_outputs_dir / "politicians.csv")
    countries_df = pd.read_csv(paths.processed_outputs_dir / "countries.csv")

    dfs_in = {"companies": companies_df, "politicians": politicians_df, "countries": countries_df}

    dfs_out = compute_all_entity_scores(
        paths=paths,
        input_dfs=dfs_in,
        company_tasks=COMPANY_TASKS,
        politician_tasks=POLITICIAN_TASKS,
        country_tasks=COUNTRY_TASKS,
        models=MODELS,
        languages=LAN,
        num_flags=[True, False],
        weight_map=WEIGHT_MAP,
        std=args.std,
        max_workers=args.max_workers,
        fs_flags=[True, False],
    )

    dfs_out["companies"].to_csv(paths.processed_outputs_dir / "companies_2.csv", index=False)
    dfs_out["politicians"].to_csv(paths.processed_outputs_dir / "politicians_2.csv", index=False)
    dfs_out["countries"].to_csv(paths.processed_outputs_dir / "countries_2.csv", index=False)


if __name__ == "__main__":
    main()
