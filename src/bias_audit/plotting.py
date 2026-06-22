"""Reusable plots for processed entity-bias audit results."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_DIR, ConfigError, TaskRegistry, TaskSpec


METRIC_FILENAMES = {"score": "{entity_set}_2.csv", "f1": "{entity_set}_f1.csv"}
DIMENSIONS = ("language", "model", "setting")


@dataclass(frozen=True)
class EntityAnalysisSpec:
    """Default grouping and labels for one entity set's figures."""

    id: str
    title: str
    category_column: str
    category_order: tuple[str, ...]
    category_labels: Mapping[str, str]


@dataclass(frozen=True)
class AnalysisSpec:
    """Small declarative layer for figure labels and entity groupings."""

    entity_sets: Mapping[str, EntityAnalysisSpec]
    language_labels: Mapping[str, str]
    setting_labels: Mapping[int, str]
    task_labels: Mapping[str, str]

    def entity_set(self, entity_set_id: str) -> EntityAnalysisSpec:
        try:
            return self.entity_sets[entity_set_id]
        except KeyError as exc:
            raise ConfigError(
                f"Entity set '{entity_set_id}' has no entry in configs/analysis.json"
            ) from exc

    def task_label(self, task_id: str) -> str:
        return self.task_labels.get(task_id, task_id.replace("_", " ").title())


def load_analysis_spec(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> AnalysisSpec:
    """Load and validate ``configs/analysis.json``."""
    path = Path(config_dir).resolve() / "analysis.json"
    if not path.is_file():
        raise ConfigError(f"Analysis configuration does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise ConfigError(f"{path}: unsupported schema_version '{data.get('schema_version')}'")

    raw_entities = data.get("entity_sets")
    if not isinstance(raw_entities, dict) or not raw_entities:
        raise ConfigError(f"{path}: 'entity_sets' must be a non-empty object")
    entities: dict[str, EntityAnalysisSpec] = {}
    for entity_id, raw in raw_entities.items():
        if not isinstance(raw, dict) or not raw.get("category_column"):
            raise ConfigError(f"{path}: entity set '{entity_id}' needs a category_column")
        entities[entity_id] = EntityAnalysisSpec(
            id=entity_id,
            title=str(raw.get("title", entity_id.replace("_", " ").title())),
            category_column=str(raw["category_column"]),
            category_order=tuple(str(value) for value in raw.get("category_order", [])),
            category_labels={str(key): str(value) for key, value in raw.get("category_labels", {}).items()},
        )
    return AnalysisSpec(
        entity_sets=entities,
        language_labels={str(key): str(value) for key, value in data.get("language_labels", {}).items()},
        setting_labels={int(key): str(value) for key, value in data.get("setting_labels", {}).items()},
        task_labels={str(key): str(value) for key, value in data.get("task_labels", {}).items()},
    )


def validate_analysis_spec(spec: AnalysisSpec, registry: TaskRegistry) -> None:
    """Validate configured entity/category references against the registry data."""
    errors = []
    registered = {entity.id for entity in registry.entity_sets()}
    for entity_id, entity_spec in spec.entity_sets.items():
        if entity_id not in registered:
            errors.append(f"analysis references unknown entity set '{entity_id}'")
            continue
        path = registry.entity_set(entity_id).path
        if path.is_file():
            columns = set(pd.read_csv(path, nrows=0).columns)
            if entity_spec.category_column not in columns:
                errors.append(
                    f"entity file '{path}' has no analysis category column "
                    f"'{entity_spec.category_column}'"
                )
    generalization_tasks = {task.id for task in registry.tasks(suite="generalization")}
    unknown_labels = set(spec.task_labels) - generalization_tasks
    if unknown_labels:
        errors.append(f"analysis has labels for unknown tasks: {', '.join(sorted(unknown_labels))}")
    if errors:
        raise ConfigError("Analysis configuration validation failed:\n- " + "\n- ".join(errors))


def processed_result_path(processed_outputs_dir: Path, entity_set: str, metric: str) -> Path:
    try:
        filename = METRIC_FILENAMES[metric].format(entity_set=entity_set)
    except KeyError as exc:
        raise ConfigError(f"Unsupported plot metric '{metric}'") from exc
    return Path(processed_outputs_dir).resolve() / filename


def load_processed_results(path: Path, category_column: str) -> pd.DataFrame:
    """Load a processed wide table and enforce the columns plots depend on."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed result file does not exist: {path}. Run processing.score_processing "
            "or processing.f1_processing first."
        )
    frame = pd.read_csv(path)
    if category_column not in frame:
        raise ConfigError(f"Processed file '{path}' has no category column '{category_column}'")
    if "name" in frame:
        frame = frame[frame["name"].astype(str).str.strip() != "X"].copy()
    return frame


def load_model_names(result_path: Path, override: Sequence[str] | None = None) -> dict[int, str]:
    """Resolve m1/m2/... using a processing sidecar or an explicit legacy override."""
    if override:
        return {index: name for index, name in enumerate(override, start=1)}
    metadata_path = Path(result_path).with_suffix(".meta.json")
    if not metadata_path.is_file():
        return {}
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid processing metadata in {metadata_path}: {exc}") from exc
    return {int(index): str(name) for index, name in metadata.get("models", {}).items()}


def metric_long_frame(
    frame: pd.DataFrame,
    metric: str,
    tasks: Sequence[TaskSpec],
    category_column: str,
    model_names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Convert score/F1 columns into a stable tidy plotting table."""
    task_ids = {task.id for task in tasks}
    pattern = re.compile(rf"^{re.escape(metric)}_([^_]+)_s(\d+)_m(\d+)_(.+)$")
    metadata: dict[str, tuple[str, int, int, str]] = {}
    for column in frame.columns:
        match = pattern.fullmatch(column)
        if match and match.group(4) in task_ids:
            metadata[column] = (
                match.group(1),
                int(match.group(2)),
                int(match.group(3)),
                match.group(4),
            )
    if not metadata:
        expected = f"{metric}_<language>_s<setting>_m<model>_<task>"
        raise ConfigError(f"No {metric} columns matched the expected pattern '{expected}'")

    work = frame.reset_index(drop=True).copy()
    work["entity_row"] = np.arange(len(work))
    long = work[["entity_row", category_column, *metadata]].melt(
        id_vars=["entity_row", category_column],
        value_vars=list(metadata),
        var_name="metric_column",
        value_name="value",
    )
    expanded = pd.DataFrame(
        long["metric_column"].map(metadata).tolist(),
        columns=["language", "setting", "model_index", "task"],
        index=long.index,
    )
    long = pd.concat([long.drop(columns="metric_column"), expanded], axis=1)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=[category_column, "value"])
    long[category_column] = long[category_column].astype(str).str.strip()
    names = dict(model_names or {})
    long["model"] = long["model_index"].map(
        lambda index: names.get(int(index), f"m{int(index)}")
    )
    return long


def plot_ellipse_report(
    registry: TaskRegistry,
    analysis: AnalysisSpec,
    processed_outputs_dir: Path,
    entity_set: str,
    output: Path,
    *,
    task_ids: Sequence[str] | None = None,
    dimensions: Sequence[str] = DIMENSIONS,
    category_column: str | None = None,
    category_order: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    confidence: float = 0.95,
    center: bool = False,
    dpi: int = 200,
) -> Path:
    """Plot mean normalized bias and confidence ellipses by configuration dimension."""
    if not 0 < confidence < 1:
        raise ConfigError("confidence must be between 0 and 1")
    unknown_dimensions = set(dimensions) - {"language", "model", "setting", "task"}
    if unknown_dimensions:
        raise ConfigError(f"Unknown ellipse dimensions: {', '.join(sorted(unknown_dimensions))}")
    entity_spec = analysis.entity_set(entity_set)
    category = category_column or entity_spec.category_column
    order = tuple(category_order) if category_order is not None else entity_spec.category_order
    tasks = _select_entity_tasks(registry, entity_set, task_ids)
    result_path = processed_result_path(processed_outputs_dir, entity_set, "score")
    frame = load_processed_results(result_path, category)
    long = metric_long_frame(
        frame, "score", tasks, category, load_model_names(result_path, model_names)
    )

    plt, ellipse_class = _plot_dependencies(ellipse=True)
    figure, axes = plt.subplots(
        1, len(dimensions), figsize=(7 * len(dimensions), max(5, 0.65 * long[category].nunique())),
        sharey=True, squeeze=False,
    )
    for index, dimension in enumerate(dimensions):
        _draw_ellipse_axis(
            axes[0, index], long, category, dimension, order, entity_spec.category_labels,
            analysis, confidence, center, ellipse_class,
        )
    figure.suptitle(entity_spec.title, fontsize=16, fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, output, dpi, plt)


def plot_radar_report(
    registry: TaskRegistry,
    analysis: AnalysisSpec,
    processed_outputs_dir: Path,
    entity_sets: Sequence[str],
    output: Path,
    *,
    task_ids: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    dpi: int = 200,
) -> Path:
    """Plot one radar panel per entity set, with one line per registered task."""
    entity_sets = list(entity_sets)
    task_ids_by_entity: dict[str, list[str]] = {}
    if task_ids is not None:
        for task_id in task_ids:
            task = registry.task(task_id)
            if task.suite != "generalization":
                raise ConfigError(f"Radar plots only support generalization tasks ('{task_id}')")
            task_ids_by_entity.setdefault(task.entity_set, []).append(task_id)
        entity_sets = [entity_set for entity_set in entity_sets if entity_set in task_ids_by_entity]
        if not entity_sets:
            raise ConfigError("None of the selected tasks belong to the selected entity sets")
    plt, _ = _plot_dependencies()
    figure, axes = plt.subplots(
        1, len(entity_sets), figsize=(8 * len(entity_sets), 8),
        subplot_kw={"projection": "polar"}, squeeze=False,
    )
    for index, entity_set in enumerate(entity_sets):
        entity_spec = analysis.entity_set(entity_set)
        local_task_ids = task_ids_by_entity.get(entity_set) if task_ids is not None else None
        tasks = _select_entity_tasks(registry, entity_set, local_task_ids)
        result_path = processed_result_path(processed_outputs_dir, entity_set, "score")
        frame = load_processed_results(result_path, entity_spec.category_column)
        long = metric_long_frame(
            frame,
            "score",
            tasks,
            entity_spec.category_column,
            load_model_names(result_path, model_names),
        )
        _draw_radar_axis(axes[0, index], long, entity_spec, tasks, analysis, plt)
    figure.tight_layout()
    return _save_figure(figure, output, dpi, plt)


def plot_boxplot_report(
    registry: TaskRegistry,
    analysis: AnalysisSpec,
    processed_outputs_dir: Path,
    entity_set: str,
    output: Path,
    *,
    metric: str = "score",
    task_ids: Sequence[str] | None = None,
    category_column: str | None = None,
    category_order: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    center: bool | None = None,
    dpi: int = 200,
) -> Path:
    """Plot per-entity distributions for every task, grouped by entity metadata."""
    entity_spec = analysis.entity_set(entity_set)
    category = category_column or entity_spec.category_column
    configured_order = tuple(category_order) if category_order is not None else entity_spec.category_order
    tasks = _select_entity_tasks(registry, entity_set, task_ids)
    result_path = processed_result_path(processed_outputs_dir, entity_set, metric)
    frame = load_processed_results(result_path, category)
    long = metric_long_frame(
        frame, metric, tasks, category, load_model_names(result_path, model_names)
    )
    should_center = metric == "f1" if center is None else center

    plt, _ = _plot_dependencies()
    columns = min(2, len(tasks))
    rows = math.ceil(len(tasks) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(8 * columns, 5 * rows), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        _draw_boxplot_axis(
            axis, long, task, category, configured_order, entity_spec.category_labels,
            analysis, metric, should_center,
        )
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    figure.suptitle(entity_spec.title, fontsize=16, fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, output, dpi, plt)


def _select_entity_tasks(
    registry: TaskRegistry, entity_set: str, task_ids: Sequence[str] | None
) -> list[TaskSpec]:
    available = registry.tasks(suite="generalization", entity_set=entity_set)
    if task_ids is None:
        return available
    selected = set(task_ids)
    unknown = selected - {task.id for task in available}
    if unknown:
        raise ConfigError(
            f"Tasks do not belong to generalization/{entity_set}: {', '.join(sorted(unknown))}"
        )
    return [task for task in available if task.id in selected]


def _category_order(
    data: pd.DataFrame, category: str, configured: Sequence[str]
) -> list[str]:
    present = [str(value) for value in data[category].dropna().unique()]
    ordered = [value for value in configured if value in present]
    remaining = [value for value in present if value not in ordered and value != "Unknown"]
    if not configured and remaining:
        medians = data[data[category].astype(str).isin(remaining)].groupby(category)["value"].median()
        remaining = [str(value) for value in medians.sort_values(ascending=False).index]
    if "Unknown" in present:
        remaining.append("Unknown")
    return ordered + remaining


def _dimension_order(long: pd.DataFrame, dimension: str) -> list[object]:
    if dimension == "language":
        preferred = ["en", "ru", "zh"]
    elif dimension == "setting":
        preferred = [1, 2, 3, 4]
    elif dimension == "model":
        index_names = long[["model_index", "model"]].drop_duplicates().sort_values("model_index")
        return index_names["model"].tolist()
    else:
        return long[dimension].drop_duplicates().tolist()
    present = set(long[dimension].dropna().unique())
    return [value for value in preferred if value in present] + sorted(present - set(preferred))


def _group_label(value: object, dimension: str, analysis: AnalysisSpec) -> str:
    if dimension == "language":
        return analysis.language_labels.get(str(value), str(value))
    if dimension == "setting":
        return analysis.setting_labels.get(int(value), f"s{value}")
    if dimension == "task":
        return analysis.task_label(str(value))
    return str(value)


def _draw_ellipse_axis(
    axis,
    long: pd.DataFrame,
    category: str,
    dimension: str,
    configured_order: Sequence[str],
    category_labels: Mapping[str, str],
    analysis: AnalysisSpec,
    confidence: float,
    center: bool,
    ellipse_class,
) -> None:
    group_order = _dimension_order(long, dimension)
    category_order = _category_order(long, category, configured_order)
    summary = (
        long.groupby([dimension, category], observed=True)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"])
    if center:
        summary["mean"] = summary["mean"] - summary.groupby(dimension)["mean"].transform("mean")
    z_value = NormalDist().inv_cdf((1 + confidence) / 2)
    palette = _palette(len(group_order))
    y_lookup = {value: index for index, value in enumerate(category_order)}
    total = len(group_order)
    for group_index, (group, color) in enumerate(zip(group_order, palette)):
        subset = summary[summary[dimension] == group]
        offset = (group_index - (total - 1) / 2) * (0.5 / max(total, 1))
        label = _group_label(group, dimension, analysis)
        plotted = False
        for _, row in subset.iterrows():
            raw_category = str(row[category])
            if raw_category not in y_lookup:
                continue
            y_value = y_lookup[raw_category] + offset
            ci = z_value * float(row["sem"])
            axis.add_patch(
                ellipse_class(
                    (float(row["mean"]), y_value), width=max(2 * ci, 1e-10), height=0.12,
                    facecolor=color, edgecolor=color, alpha=0.22,
                )
            )
            axis.scatter(float(row["mean"]), y_value, color=color, s=24, zorder=3,
                         label=label if not plotted else None)
            plotted = True
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(category_order)))
    axis.set_yticklabels([category_labels.get(value, value) for value in category_order])
    axis.set_xlabel("Average bias score")
    axis.set_title(dimension.title())
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.legend(fontsize=7, loc="best")


def _draw_radar_axis(
    axis,
    long: pd.DataFrame,
    entity_spec: EntityAnalysisSpec,
    tasks: Sequence[TaskSpec],
    analysis: AnalysisSpec,
    plt,
) -> None:
    category = entity_spec.category_column
    category_order = _category_order(long, category, entity_spec.category_order)
    if len(category_order) < 3:
        raise ConfigError(
            f"Radar plot for '{entity_spec.id}' needs at least three populated categories in '{category}'"
        )
    aggregate = long.groupby([category, "task"], observed=True)["value"].mean()
    angles = np.linspace(0, 2 * np.pi, len(category_order), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    finite_values = aggregate.to_numpy(dtype=float)
    data_min, data_max = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
    padding = max(0.05, (data_max - data_min) * 0.12)
    radial_min, radial_max = min(data_min - padding, 0), max(data_max + padding, 0)

    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    circle = np.linspace(0, 2 * np.pi, 300)
    if radial_min < 0:
        axis.fill_between(circle, radial_min, 0, color="#d95f02", alpha=0.05)
    if radial_max > 0:
        axis.fill_between(circle, 0, radial_max, color="#1b9e77", alpha=0.05)
    palette = _palette(len(tasks))
    for task, color in zip(tasks, palette):
        values = np.array(
            [aggregate.get((category_value, task.id), np.nan) for category_value in category_order],
            dtype=float,
        )
        axis.plot(closed_angles, np.append(values, values[0]), linewidth=2,
                  label=analysis.task_label(task.id), color=color)
    axis.plot(circle, np.zeros_like(circle), color="black", linestyle="--", linewidth=1, alpha=0.5)
    axis.set_xticks(angles)
    axis.set_xticklabels(
        [entity_spec.category_labels.get(value, value) for value in category_order], fontsize=9
    )
    axis.set_ylim(radial_min, radial_max)
    axis.tick_params(axis="y", labelsize=8, colors="grey")
    axis.set_title(entity_spec.title, fontsize=15, fontweight="bold", pad=22)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=9)


def _draw_boxplot_axis(
    axis,
    long: pd.DataFrame,
    task: TaskSpec,
    category: str,
    configured_order: Sequence[str],
    category_labels: Mapping[str, str],
    analysis: AnalysisSpec,
    metric: str,
    center: bool,
) -> None:
    subset = long[long["task"] == task.id]
    entity_values = subset.groupby(["entity_row", category], observed=True)["value"].mean().reset_index()
    if center:
        entity_values["value"] = entity_values["value"] - entity_values["value"].mean()
    order = _category_order(entity_values, category, configured_order)
    distributions = [
        entity_values.loc[entity_values[category].astype(str) == value, "value"].dropna().to_numpy()
        for value in order
    ]
    keep = [(value, values) for value, values in zip(order, distributions) if len(values)]
    if not keep:
        axis.set_visible(False)
        return
    order, distributions = map(list, zip(*keep))
    artists = axis.boxplot(distributions, patch_artist=True, showfliers=False)
    colors = _palette(len(distributions))
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(1, len(order) + 1))
    axis.set_xticklabels([category_labels.get(value, value) for value in order], rotation=35, ha="right")
    axis.set_title(analysis.task_label(task.id))
    if metric == "f1":
        axis.set_ylabel("Average F1 deviation" if center else "Average macro-F1")
    else:
        axis.set_ylabel("Average bias score")
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def _palette(count: int) -> list[object]:
    _, _, colormaps = _plot_dependencies(colormaps=True)
    color_map = colormaps.get_cmap("tab20")
    return [color_map(index / max(count - 1, 1)) for index in range(count)]


def _plot_dependencies(ellipse: bool = False, colormaps: bool = False):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps as matplotlib_colormaps
        from matplotlib.patches import Ellipse
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: pip install -e '.[analysis]'"
        ) from exc
    if colormaps:
        return plt, Ellipse if ellipse else None, matplotlib_colormaps
    return plt, Ellipse if ellipse else None


def _save_figure(figure, output: Path, dpi: int, plt) -> Path:
    output = Path(output).resolve()
    if output.suffix.lower() not in {".pdf", ".png", ".svg"}:
        raise ConfigError("Plot output must end in .pdf, .png, or .svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote plot to {output}")
    return output
