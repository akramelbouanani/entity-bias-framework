"""Entity-table validation, audit execution, and report construction."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from bias_audit.data import substitute_entity

from .clients import ClassificationClient
from .generation import LANGUAGES
from .store import ProjectStore
from .tokens import label_token_overrides as _label_token_overrides
from .tokens import text_token_map as _text_token_map


@dataclass(frozen=True)
class Setting:
    language: str

    @property
    def id(self) -> str:
        return self.language

    @property
    def label(self) -> str:
        return LANGUAGES[self.language]["label"]


class ExperimentPipeline:
    def __init__(self, store: ProjectStore, batch_size: int = 300):
        self.store = store
        self.batch_size = batch_size

    def configure_entities(
        self,
        project_id: str,
        name_columns: Mapping[str, str],
    ) -> dict:
        project = self.store.get(project_id)
        if not project.get("generation"):
            raise ValueError("Generate the task before configuring entities")
        path = self.store.artifact(project_id, "entities.csv")
        if not path.is_file():
            raise ValueError("Upload an entity CSV first")
        frame = pd.read_csv(path)
        required_languages = project["design"]["languages"]
        for language in required_languages:
            column = name_columns.get(language)
            if not column or column not in frame.columns:
                raise ValueError(f"Choose an entity-name column for {LANGUAGES[language]['label']}")
            if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
                raise ValueError(f"Entity-name column '{column}' contains empty values")
        english_column = name_columns["english"]
        if frame[english_column].astype(str).str.strip().duplicated().any():
            raise ValueError("The English entity-name column contains duplicates")
        mapped_names = set(name_columns.values())
        analysis_columns = []
        filter_columns = []
        for column in frame.columns:
            values = frame[column].astype(str).str.strip()
            unique = sorted(values.unique().tolist())
            is_metadata = (
                column not in mapped_names
                and not frame[column].isna().any()
                and not (values == "").any()
                and 2 <= len(unique) <= 250
                and len(unique) < len(frame)
            )
            if is_metadata:
                filter_columns.append({"name": column, "values": unique})
            if (
                is_metadata
                and 2 <= len(unique) <= 8
            ):
                analysis_columns.append({"name": column, "values": unique})
        if not analysis_columns:
            raise ValueError(
                "The CSV needs at least one non-empty categorical metadata column with 2–8 values"
            )
        entities = {
            "path": str(path),
            "row_count": len(frame),
            "name_columns": dict(name_columns),
            "analysis_columns": analysis_columns,
            "filter_columns": filter_columns,
        }
        project["entities"] = entities
        project["stage"] = "entities_configured"
        self.store.save(project)
        return entities

    def preflight_tokens(
        self,
        project_id: str,
        config: dict,
        host: str,
        port: int,
    ) -> dict:
        """Verify every category token on a known example before full inference."""
        project = self.store.get(project_id)
        if not project.get("entities"):
            raise ValueError("Configure the entity table before checking category tokens")
        settings = self._settings(project, config)
        design = project["design"]
        labels = [label["name"] for label in design["labels"]]
        token_overrides = _label_token_overrides(design)
        with Path(project["generation"]["locales"]).open(encoding="utf-8") as handle:
            locales = json.load(handle)
        client = ClassificationClient(host, port)

        prompts = []
        expectations = []
        for setting in settings:
            token_map, exact = _token_map(labels, token_overrides)
            examples = locales[setting.language]["examples"]
            for label in labels:
                example = next(
                    (item for item in examples if item.get("label") == label), None
                )
                if example is None:
                    raise ValueError(
                        f"No {LANGUAGES[setting.language]['label']} preflight example exists for '{label}'"
                    )
                prompts.append(
                    self._classification_prompt(
                        design, locales[setting.language], setting, str(example["sentence"])
                    )
                )
                expectations.append(
                    {
                        "label": label,
                        "token": token_map[label][0].casefold(),
                        "exact": exact,
                        "setting": _setting_dict(setting),
                    }
                )

        observed_outputs = client.top_tokens(prompts)
        checks = []
        for expected, observed in zip(expectations, observed_outputs):
            candidate = expected["token"]
            matches = [
                token
                for token in observed
                if (token == candidate if expected["exact"] else token.startswith(candidate))
            ]
            ranked = sorted(observed.items(), key=lambda item: item[1], reverse=True)
            checks.append(
                {
                    "label": expected["label"],
                    "token": candidate,
                    "setting": expected["setting"],
                    "passed": bool(matches),
                    "matched": matches[:3],
                    "observed": [token for token, _ in ranked[:8]],
                }
            )

        passed = all(check["passed"] for check in checks)
        result = {
            "id": uuid.uuid4().hex[:12],
            "passed": passed,
            "model": client.model,
            "host": host,
            "port": port,
            "config": config,
            "tokens": {
                label: candidates[0]
                for label, candidates in _text_token_map(labels, token_overrides).items()
            },
            "check_count": len(checks),
            "failed_count": sum(not check["passed"] for check in checks),
            "checks": checks,
        }
        project["token_preflight"] = result
        self.store.save(project)
        return result

    def run(
        self,
        project_id: str,
        config: dict,
        host: str,
        port: int,
        progress: Callable[[float, str], None],
    ) -> dict:
        started = time.monotonic()
        project = self.store.get(project_id)
        if not project.get("entities"):
            raise ValueError("Configure the entity table before running the audit")
        settings = self._settings(project, config)
        design = project["design"]
        labels = [label["name"] for label in design["labels"]]
        token_overrides = _label_token_overrides(design)
        weights = {label["name"]: float(label["weight"]) for label in design["labels"]}
        _text_token_map(labels, token_overrides)

        sentences = pd.read_csv(project["generation"]["dataset"])
        entities = pd.read_csv(project["entities"]["path"])
        with Path(project["generation"]["locales"]).open(encoding="utf-8") as handle:
            locales = json.load(handle)
        client = ClassificationClient(host, port)
        checked = project.get("token_preflight") or {}
        if checked.get("model") != client.model:
            raise ValueError("The served model changed after token preflight; check tokens again")
        total = len(sentences) * len(entities) * len(settings)
        completed = 0
        score_records = []
        contexts = []
        warnings = []

        for setting_index, setting in enumerate(settings):
            progress(2 + 94 * completed / max(total, 1), f"Preparing {setting.label}…")
            raw_scores = np.full((len(entities), len(sentences)), np.nan, dtype=np.float32)
            label_sums = np.zeros(len(labels), dtype=np.float64)
            label_counts = np.zeros(len(labels), dtype=np.int64)
            token_map, exact = _token_map(labels, token_overrides)
            for start in range(0, len(sentences) * len(entities), self.batch_size):
                stop = min(start + self.batch_size, len(sentences) * len(entities))
                coordinates = [divmod(flat_index, len(entities)) for flat_index in range(start, stop)]
                prompts = [
                    self._classification_prompt(
                        design,
                        locales[setting.language],
                        setting,
                        substitute_entity(
                            str(sentences.iloc[sentence_index][LANGUAGES[setting.language]["column"]]),
                            str(entities.iloc[entity_index][project["entities"]["name_columns"][setting.language]]),
                        ),
                    )
                    for sentence_index, entity_index in coordinates
                ]
                probabilities = client.classify(prompts, token_map, exact=exact)
                for (sentence_index, entity_index), output in zip(coordinates, probabilities):
                    values = np.asarray([output.get(label, np.nan) for label in labels], dtype=float)
                    if np.isfinite(values).all():
                        raw_scores[entity_index, sentence_index] = float(
                            sum(values[index] * weights[label] for index, label in enumerate(labels))
                        )
                        label_sums += values
                        label_counts += 1
                completed += len(coordinates)
                progress(
                    2 + 94 * completed / max(total, 1),
                    f"{setting.label} · {stop:,}/{len(sentences) * len(entities):,}",
                )

            means = np.nanmean(raw_scores, axis=0)
            standard_deviations = np.nanstd(raw_scores, axis=0, ddof=1)
            coverage = np.isfinite(raw_scores).sum(axis=0)
            valid = (coverage >= max(2, math.ceil(len(entities) * 0.8))) & (standard_deviations > 1e-12)
            valid_sentence_ids = np.flatnonzero(valid)
            usable = len(valid_sentence_ids) >= 3
            if usable:
                z_vectors = (
                    raw_scores[:, valid] - means[valid][None, :]
                ) / standard_deviations[valid][None, :]
                minimum_sentences = max(3, math.ceil(valid.sum() * 0.5))
                entity_coverage = np.isfinite(z_vectors).sum(axis=1)
                entity_scores = np.nanmean(z_vectors, axis=1)
                entity_scores[entity_coverage < minimum_sentences] = np.nan
                finite_scores = np.isfinite(entity_scores)
                if finite_scores.any():
                    entity_scores[finite_scores] -= np.mean(entity_scores[finite_scores])
            else:
                z_vectors = np.empty((len(entities), 0), dtype=np.float32)
                entity_coverage = np.zeros(len(entities), dtype=int)
                entity_scores = np.full(len(entities), np.nan, dtype=np.float32)
                warnings.append(
                    f"{setting.label} produced fewer than three entity-varying sentences and "
                    "is unavailable for normalized-bias plots."
                )
            reference_label_probabilities = label_sums / np.maximum(label_counts, 1)

            context_name = _safe_setting_name(setting.id) + ".npz"
            context_path = self.store.artifact(project_id, "contexts", context_name)
            np.savez_compressed(
                context_path,
                means=means[valid].astype(np.float32),
                standard_deviations=standard_deviations[valid].astype(np.float32),
                sentence_ids=valid_sentence_ids.astype(np.int32),
                z_vectors=z_vectors.astype(np.float32),
                entity_scores=entity_scores.astype(np.float32),
                reference_label_probabilities=reference_label_probabilities.astype(np.float32),
            )
            contexts.append(
                {
                    "setting": _setting_dict(setting),
                    "path": str(context_path),
                    "usable": usable,
                    "variable_sentence_count": int(len(valid_sentence_ids)),
                }
            )
            for entity_index, score in enumerate(entity_scores):
                score_records.append(
                    {
                        "entity_index": entity_index,
                        "entity": str(
                            entities.iloc[entity_index][project["entities"]["name_columns"]["english"]]
                        ).strip(),
                        "language": setting.language,
                        "setting": setting.id,
                        "score": _finite(score),
                        "coverage": round(100 * entity_coverage[entity_index] / max(valid.sum(), 1), 2),
                    }
                )

        scores_path = self.store.artifact(project_id, "entity_scores.csv")
        pd.DataFrame.from_records(score_records).to_csv(scores_path, index=False)
        report = {
            "project_id": project_id,
            "task_name": design["name"],
            "direction": {
                "higher": design["higher_weight_meaning"],
                "lower": design["lower_weight_meaning"],
            },
            "model": client.model,
            "category_tokens": {
                label: candidates[0]
                for label, candidates in _text_token_map(labels, token_overrides).items()
            },
            "entity_count": len(entities),
            "sentence_count": len(sentences),
            "setting_count": len(settings),
            "prediction_count": total,
            "analysis_columns": project["entities"]["analysis_columns"],
            "entity_groups": {
                item["name"]: entities[item["name"]].astype(str).str.strip().tolist()
                for item in project["entities"]["analysis_columns"]
            },
            "filter_columns": project["entities"]["filter_columns"],
            "entity_filters": {
                item["name"]: entities[item["name"]].astype(str).str.strip().tolist()
                for item in project["entities"]["filter_columns"]
            },
            "settings": [_setting_dict(setting) for setting in settings],
            "scores": score_records,
            "warnings": warnings,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        report_path = self.store.artifact(project_id, "report.json")
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False)
        project["experiment"] = {
            "config": config,
            "model": client.model,
            "contexts": contexts,
            "scores": str(scores_path),
        }
        project["report"] = str(report_path)
        project["stage"] = "completed"
        self.store.save(project)
        progress(99, "Preparing the interactive report…")
        return report

    def run_additional_entity(
        self,
        project_id: str,
        names: Mapping[str, str],
        reference_filters: Mapping[str, str],
        host: str,
        port: int,
        progress: Callable[[float, str], None],
    ) -> dict:
        project = self.store.get(project_id)
        if project.get("stage") != "completed" or not project.get("experiment"):
            raise ValueError("Complete the reference audit before adding an entity")
        for language in project["experiment"]["config"]["languages"]:
            if not str(names.get(language, "")).strip():
                raise ValueError(f"Provide a name for {LANGUAGES[language]['label']}")
        design = project["design"]
        labels = [label["name"] for label in design["labels"]]
        token_overrides = _label_token_overrides(design)
        weights = {label["name"]: float(label["weight"]) for label in design["labels"]}
        sentences = pd.read_csv(project["generation"]["dataset"])
        entities = pd.read_csv(project["entities"]["path"])
        subset_indices, applied_filters = _select_reference_cohort(
            entities, project["entities"].get("filter_columns", []), reference_filters
        )
        with Path(project["generation"]["locales"]).open(encoding="utf-8") as handle:
            locales = json.load(handle)
        client = ClassificationClient(host, port)
        context_items = project["experiment"]["contexts"]
        total = len(sentences) * len(context_items)
        completed = 0
        setting_results = []
        new_vectors = []
        reference_vectors = []

        for item in context_items:
            setting = Setting(language=item["setting"]["language"])
            context = np.load(item["path"])
            token_map, exact = _token_map(labels, token_overrides)
            raw_scores = np.full(len(sentences), np.nan, dtype=np.float32)
            label_sums = np.zeros(len(labels), dtype=np.float64)
            label_count = 0
            for start in range(0, len(sentences), self.batch_size):
                stop = min(start + self.batch_size, len(sentences))
                prompts = [
                    self._classification_prompt(
                        design,
                        locales[setting.language],
                        setting,
                        substitute_entity(
                            str(sentences.iloc[index][LANGUAGES[setting.language]["column"]]),
                            str(names[setting.language]).strip(),
                        ),
                    )
                    for index in range(start, stop)
                ]
                outputs = client.classify(prompts, token_map, exact=exact)
                for offset, output in enumerate(outputs):
                    values = np.asarray([output.get(label, np.nan) for label in labels], dtype=float)
                    if np.isfinite(values).all():
                        raw_scores[start + offset] = sum(
                            values[index] * weights[label] for index, label in enumerate(labels)
                        )
                        label_sums += values
                        label_count += 1
                completed += stop - start
                progress(
                    5 + 90 * completed / max(total, 1),
                    f"Additional entity · {setting.label} · {stop}/{len(sentences)}",
                )
            sentence_ids = context["sentence_ids"].astype(int)
            global_z_vector = (
                (raw_scores[sentence_ids] - context["means"]) / context["standard_deviations"]
                if len(sentence_ids)
                else np.asarray([], dtype=float)
            )
            reference_base = np.asarray(context["z_vectors"], dtype=float)[subset_indices]
            if reference_base.shape[1]:
                cohort_means = np.nanmean(reference_base, axis=0)
                cohort_deviations = np.nanstd(reference_base, axis=0, ddof=1)
                cohort_coverage = np.isfinite(reference_base).sum(axis=0)
                usable = (
                    (cohort_coverage >= max(2, math.ceil(len(subset_indices) * 0.8)))
                    & (cohort_deviations > 1e-12)
                )
            else:
                usable = np.asarray([], dtype=bool)
                cohort_means = np.asarray([], dtype=float)
                cohort_deviations = np.asarray([], dtype=float)
            if usable.sum() >= 3:
                z_vector = (
                    global_z_vector[usable] - cohort_means[usable]
                ) / cohort_deviations[usable]
                cohort_vectors = (
                    reference_base[:, usable] - cohort_means[usable][None, :]
                ) / cohort_deviations[usable][None, :]
                minimum_sentences = max(3, math.ceil(usable.sum() * 0.5))
                cohort_entity_coverage = np.isfinite(cohort_vectors).sum(axis=1)
                cohort_entity_scores = np.nanmean(cohort_vectors, axis=1)
                cohort_entity_scores[cohort_entity_coverage < minimum_sentences] = np.nan
                finite_cohort_scores = np.isfinite(cohort_entity_scores)
                cohort_center = (
                    float(np.mean(cohort_entity_scores[finite_cohort_scores]))
                    if finite_cohort_scores.any()
                    else 0.0
                )
                cohort_entity_scores[finite_cohort_scores] -= cohort_center
                vector_sentence_ids = sentence_ids[usable]
            else:
                z_vector = np.asarray([], dtype=float)
                cohort_vectors = np.empty((len(subset_indices), 0), dtype=float)
                cohort_entity_scores = np.full(len(subset_indices), np.nan, dtype=float)
                cohort_center = 0.0
                vector_sentence_ids = np.asarray([], dtype=int)
            score = (
                float(np.nanmean(z_vector) - cohort_center)
                if len(z_vector) and np.isfinite(z_vector).any()
                else None
            )
            reference_scores = pd.Series(cohort_entity_scores).dropna()
            setting_results.append(
                {
                    "setting": _setting_dict(setting),
                    "score": _finite(score),
                    "coverage": round(100 * np.isfinite(z_vector).sum() / max(len(z_vector), 1), 2),
                    "reference_count": len(reference_scores),
                    "reference_scores": [
                        _finite(value) for value in cohort_entity_scores
                    ],
                    "labels": [
                        {
                            "label": label,
                            "probability": _finite(label_sums[index] / max(label_count, 1)),
                        }
                        for index, label in enumerate(labels)
                    ],
                }
            )
            if len(vector_sentence_ids):
                prefix = setting.id + ":"
                new_vectors.append(
                    pd.Series(z_vector, index=[f"{prefix}{value}" for value in vector_sentence_ids])
                )
                reference_vectors.append(
                    pd.DataFrame(
                        cohort_vectors,
                        index=subset_indices,
                        columns=[f"{prefix}{value}" for value in vector_sentence_ids],
                    )
                )

        views = _additional_views(setting_results)
        if new_vectors:
            combined_new = pd.concat(new_vectors)
            combined_reference = pd.concat(reference_vectors, axis=1)
            correlations = _correlations(
                combined_new,
                combined_reference,
                entities[project["entities"]["name_columns"]["english"]],
            )
            group_correlations = _additional_group_correlations(
                combined_new,
                combined_reference,
                entities,
                project["entities"]["analysis_columns"],
            )
        else:
            correlations = {"highest": [], "lowest": [], "compared_entities": 0}
            group_correlations = []
        result = {
            "entity": str(names["english"]).strip(),
            "model": client.model,
            "views": views,
            "overall": views[0],
            "correlations": correlations,
            "group_correlations": group_correlations,
            "prompt_count": total,
            "reference_filters": applied_filters,
            "reference_count": len(subset_indices),
        }
        destination = self.store.artifact(
            project_id, "additional_entities", f"{int(time.time())}_{_safe_setting_name(names['english'])}.json"
        )
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        return result

    def _settings(self, project: dict, config: dict) -> list[Setting]:
        languages = list(dict.fromkeys(config.get("languages", [])))
        if not languages:
            raise ValueError("Select at least one language")
        if not set(languages).issubset(project["design"]["languages"]):
            raise ValueError("Experiment languages must have been generated first")
        return [Setting(language) for language in languages]

    def _classification_prompt(
        self,
        design: dict,
        locale: dict,
        setting: Setting,
        sentence: str,
    ) -> str:
        labels = [label["name"] for label in design["labels"]]
        parts = [
            locale["instruction"].strip(),
            "\n\n",
            f"Classify the sentence and answer with exactly one label: {', '.join(labels)}.\n",
            "\nExamples:\n",
        ]
        for example in locale["examples"]:
            parts.append(f"Sentence: {example['sentence']}\nAnswer: {example['label']}\n\n")
        parts.append(f"Sentence: {sentence}\nAnswer:")
        return "".join(parts)


def inspect_entity_csv(path: Path) -> dict:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The uploaded CSV is empty")
    if len(frame) < 2:
        raise ValueError("Upload at least two reference entities")
    if len(frame) > 5000:
        raise ValueError("The local workbench currently supports at most 5,000 entities")
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("CSV column names must be unique")
    return {
        "row_count": len(frame),
        "columns": [
            {
                "name": column,
                "unique": int(frame[column].nunique(dropna=True)),
                "missing": int(frame[column].isna().sum()),
                "sample": [str(value) for value in frame[column].dropna().head(3)],
            }
            for column in frame.columns
        ],
        "preview": frame.head(5).fillna("").astype(str).to_dict("records"),
    }


def _select_reference_cohort(
    entities: pd.DataFrame,
    filter_columns: Sequence[Mapping],
    requested: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, str]]:
    allowed = {str(item["name"]): {str(value) for value in item["values"]} for item in filter_columns}
    filters = {
        str(column): str(value).strip()
        for column, value in requested.items()
        if str(value).strip()
    }
    unknown = sorted(set(filters) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported reference filter column(s): {', '.join(unknown)}")
    for column, value in filters.items():
        if value not in allowed[column]:
            raise ValueError(f"Unknown value '{value}' for reference filter '{column}'")
    mask = np.ones(len(entities), dtype=bool)
    for column, value in filters.items():
        mask &= entities[column].astype(str).str.strip().to_numpy() == value
    indices = np.flatnonzero(mask)
    if len(indices) < 3:
        criteria = " AND ".join(f"{column} = {value}" for column, value in filters.items())
        raise ValueError(
            f"The selected reference cohort has {len(indices)} entities; at least 3 are required"
            + (f" ({criteria})" if criteria else "")
        )
    return indices, filters


def _token_map(
    labels: Sequence[str],
    overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, list[str]], bool]:
    return _text_token_map(labels, overrides), False


def _setting_dict(setting: Setting) -> dict:
    return {
        "language": setting.language,
        "id": setting.id,
        "label": setting.label,
    }


def _additional_views(results: Sequence[dict]) -> list[dict]:
    definitions = [{"id": "all", "group": "Average", "label": "All settings", "items": list(results)}]
    dimensions = [("language", "Language")]
    for field, group in dimensions:
        values = list(dict.fromkeys(item["setting"][field] for item in results))
        if len(values) > 1:
            for value in values:
                definitions.append(
                    {
                        "id": f"{field}:{value}",
                        "group": group,
                        "label": value.replace("_", " ").title(),
                        "items": [item for item in results if item["setting"][field] == value],
                    }
                )
    views = []
    for definition in definitions:
        items = definition.pop("items")
        available_scores = [item["score"] for item in items if item["score"] is not None]
        score = float(np.mean(available_scores)) if available_scores else None
        reference_matrix = np.asarray(
            [
                [np.nan if value is None else float(value) for value in item["reference_scores"]]
                for item in items
            ],
            dtype=float,
        )
        required_settings = max(1, math.ceil(len(items) / 2))
        reference_coverage = np.isfinite(reference_matrix).sum(axis=0)
        reference_totals = np.nansum(reference_matrix, axis=0)
        reference_values = reference_totals / np.maximum(reference_coverage, 1)
        reference_scores = pd.Series(
            reference_values[reference_coverage >= required_settings]
        ).dropna()
        reference_center = float(reference_scores.mean()) if len(reference_scores) else 0.0
        reference_scores = reference_scores - reference_center
        if score is not None:
            score -= reference_center
        labels = []
        for label in items[0]["labels"]:
            name = label["label"]
            labels.append(
                {
                    "label": name,
                    "probability": _finite(np.mean([
                        next(value["probability"] for value in item["labels"] if value["label"] == name)
                        for item in items
                    ])),
                }
            )
        views.append(
            {
                **definition,
                "score": _finite(score),
                "percentile": _finite(_percentile(reference_scores, score)),
                "coverage": _finite(np.mean([item["coverage"] for item in items])),
                "setting_count": len(items),
                "reference_count": int(len(reference_scores)),
                "histogram": _histogram(reference_scores, score),
                "labels": labels,
            }
        )
    return views


def _correlations(new: pd.Series, reference: pd.DataFrame, names: pd.Series) -> dict:
    new = new.reindex(reference.columns)
    right = new.to_numpy(dtype=float)
    minimum = max(10, math.ceil(np.isfinite(right).sum() * 0.5))
    values = []
    for index, row in reference.iterrows():
        left = row.to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() < minimum or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
            continue
        correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
        if np.isfinite(correlation):
            values.append(
                {
                    "name": str(names.iloc[int(index)]).strip(),
                    "correlation": round(correlation, 4),
                    "overlap": int(valid.sum()),
                }
            )
    values.sort(key=lambda value: value["correlation"], reverse=True)
    return {
        "highest": values[:3],
        "lowest": list(reversed(values[-3:])),
        "compared_entities": len(values),
    }


def _additional_group_correlations(
    new: pd.Series,
    reference: pd.DataFrame,
    entities: pd.DataFrame,
    analysis_columns: Sequence[Mapping],
) -> list[dict]:
    """Correlate a submitted entity with each category's average response profile."""
    new = new.reindex(reference.columns)
    submitted = new.to_numpy(dtype=float)
    minimum = max(10, math.ceil(np.isfinite(submitted).sum() * 0.5))
    entity_indices = np.asarray(reference.index, dtype=int)
    results = []
    for column_spec in analysis_columns:
        column = str(column_spec["name"])
        metadata = entities.iloc[entity_indices][column].astype(str).str.strip().to_numpy()
        values = []
        for group in column_spec["values"]:
            positions = np.flatnonzero(metadata == str(group))
            if not len(positions):
                continue
            selected = reference.iloc[positions].to_numpy(dtype=float)
            counts = np.isfinite(selected).sum(axis=0)
            totals = np.nansum(selected, axis=0)
            profile = np.divide(
                totals,
                counts,
                out=np.full(selected.shape[1], np.nan, dtype=float),
                where=counts > 0,
            )
            valid = np.isfinite(profile) & np.isfinite(submitted)
            if (
                valid.sum() < minimum
                or np.std(profile[valid]) <= 1e-12
                or np.std(submitted[valid]) <= 1e-12
            ):
                continue
            correlation = float(np.corrcoef(profile[valid], submitted[valid])[0, 1])
            if np.isfinite(correlation):
                values.append(
                    {
                        "group": str(group),
                        "correlation": round(correlation, 4),
                        "size": int(len(positions)),
                        "overlap": int(valid.sum()),
                    }
                )
        if len(values) < 2:
            continue
        values.sort(key=lambda value: value["correlation"], reverse=True)
        results.append(
            {
                "column": column,
                "highest": values[:2],
                "lowest": list(reversed(values[-2:])),
                "compared_groups": len(values),
            }
        )
    return results


def _percentile(reference: pd.Series, score: float) -> float:
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values) or score is None or not np.isfinite(score):
        return float("nan")
    return float(100 * (np.sum(values < score) + 0.5 * np.sum(np.isclose(values, score))) / len(values))


def _histogram(reference: pd.Series, marker: float) -> dict:
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"counts": [], "edges": [], "marker": _finite(marker), "underflow": 0, "overflow": 0}
    bins = min(18, max(10, int(math.sqrt(len(values)))))
    low, high = _robust_histogram_range(values)
    clipped = np.clip(values, low, high)
    counts, edges = np.histogram(clipped, bins=np.linspace(low, high, bins + 1))
    return {
        "counts": [int(value) for value in counts],
        "edges": [round(float(value), 4) for value in edges],
        "marker": _finite(marker),
        "underflow": int(np.sum(values < low)),
        "overflow": int(np.sum(values > high)),
    }


def _robust_histogram_range(values: np.ndarray) -> tuple[float, float]:
    """Keep reference outliers from flattening the visible cohort distribution."""
    values = np.asarray(values, dtype=float)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    spread = q3 - q1
    if spread > 1e-12:
        low = max(float(np.min(values)), float(q1 - 2.5 * spread))
        high = min(float(np.max(values)), float(q3 + 2.5 * spread))
    else:
        center = float(np.median(values))
        fallback = max(float(np.std(values)) * 3, abs(center) * 0.08, 0.05)
        low, high = center - fallback, center + fallback
    if high - low <= 1e-12:
        low, high = low - 0.05, high + 0.05
    return low, high


def _finite(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None and np.isfinite(value) else None


def _safe_setting_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:100]
