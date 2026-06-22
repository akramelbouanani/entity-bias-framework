"""Load and validate declarative entity-set and task specifications."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when framework configuration is missing or inconsistent."""


def _required(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ConfigError(f"{context}: missing required field '{key}'")
    return data[key]


def _resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class EntitySetSpec:
    """An entity table and the localized columns containing entity names."""

    id: str
    path: Path
    name_columns: Mapping[str, str]
    placeholder: str = "X"
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], root: Path, source: Path) -> "EntitySetSpec":
        context = str(source)
        return cls(
            id=str(_required(data, "id", context)),
            path=_resolve(_required(data, "path", context), root),
            name_columns=dict(_required(data, "name_columns", context)),
            placeholder=str(data.get("placeholder", "X")),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class LabelSpec:
    """A stable outcome label with localized renderings and an analysis weight."""

    id: str
    text: Mapping[str, str]
    tokens: Mapping[str, tuple[str, ...]]
    numeric: str | None = None
    weight: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], context: str) -> "LabelSpec":
        raw_tokens = _required(data, "tokens", context)
        return cls(
            id=str(_required(data, "id", context)),
            text=dict(_required(data, "text", context)),
            tokens={lang: tuple(values) for lang, values in raw_tokens.items()},
            numeric=None if data.get("numeric") is None else str(data["numeric"]),
            weight=None if data.get("weight") is None else float(data["weight"]),
        )


@dataclass(frozen=True)
class GenerationExample:
    """One keyword/label/sentence demonstration for sentence generation."""

    keywords: tuple[str, ...]
    label: str
    sentence: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], context: str) -> "GenerationExample":
        return cls(
            keywords=tuple(str(value) for value in _required(data, "keywords", context)),
            label=str(_required(data, "label", context)),
            sentence=str(_required(data, "sentence", context)),
        )


@dataclass(frozen=True)
class GenerationSpec:
    """Declarative inputs and prompting settings for evidence-light generation."""

    keywords_path: Path
    instruction: str
    few_shot: tuple[GenerationExample, ...]
    samples_per_label: int
    keywords_per_prompt: int
    keyword_column: str = "keyword"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 160
    minimum_keyword_matches: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], root: Path, context: str) -> "GenerationSpec":
        return cls(
            keywords_path=_resolve(_required(data, "keywords_file", context), root),
            keyword_column=str(data.get("keyword_column", "keyword")),
            instruction=str(_required(data, "instruction", context)),
            few_shot=tuple(
                GenerationExample.from_dict(example, f"{context} example #{index + 1}")
                for index, example in enumerate(data.get("few_shot", []))
            ),
            samples_per_label=int(_required(data, "samples_per_label", context)),
            keywords_per_prompt=int(_required(data, "keywords_per_prompt", context)),
            temperature=float(data.get("temperature", 1.0)),
            top_p=float(data.get("top_p", 1.0)),
            max_tokens=int(data.get("max_tokens", 160)),
            minimum_keyword_matches=int(data.get("minimum_keyword_matches", 1)),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Complete definition of a task, its data variants, labels, and prompts."""

    id: str
    suite: str
    entity_set: str
    datasets: Mapping[str, Path]
    text_columns: Mapping[str, str]
    labels: tuple[LabelSpec, ...]
    locales: Mapping[str, Mapping[str, Any]]
    label_column: str = "label"
    include_target: bool = False
    description: str = ""
    generation: GenerationSpec | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], root: Path, source: Path) -> "TaskSpec":
        context = str(source)
        dataset = _required(data, "dataset", context)
        labels = tuple(
            LabelSpec.from_dict(item, f"{context} label #{index + 1}")
            for index, item in enumerate(_required(data, "labels", context))
        )
        generation_data = data.get("generation")
        return cls(
            id=str(_required(data, "id", context)),
            suite=str(_required(data, "suite", context)),
            entity_set=str(_required(data, "entity_set", context)),
            datasets={name: _resolve(path, root) for name, path in _required(dataset, "variants", context).items()},
            text_columns=dict(_required(dataset, "text_columns", context)),
            label_column=str(dataset.get("label_column", "label")),
            labels=labels,
            locales=dict(_required(_required(data, "prompt", context), "locales", context)),
            include_target=bool(data["prompt"].get("include_target", False)),
            description=str(data.get("description", "")),
            generation=(
                GenerationSpec.from_dict(generation_data, root, f"{context} generation")
                if generation_data is not None
                else None
            ),
        )

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self.locales)

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(self.datasets)

    @property
    def supports_numerical(self) -> bool:
        return all(label.numeric is not None for label in self.labels)

    def label(self, label_id: str) -> LabelSpec:
        try:
            return next(label for label in self.labels if label.id == label_id)
        except StopIteration as exc:
            raise ConfigError(f"Task '{self.id}' has no label id '{label_id}'") from exc

    def canonical_labels(self, language: str = "english") -> list[str]:
        return [label.text[language] for label in self.labels]

    def weights(self, language: str = "english") -> dict[str, float]:
        missing = [label.id for label in self.labels if label.weight is None]
        if missing:
            raise ConfigError(f"Task '{self.id}' has labels without weights: {', '.join(missing)}")
        return {label.text[language]: float(label.weight) for label in self.labels}


class TaskRegistry:
    """Discover task files and provide the single routing source for the framework."""

    def __init__(self, config_dir: str | Path = DEFAULT_CONFIG_DIR, project_root: str | Path = PROJECT_ROOT):
        self.config_dir = Path(config_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self._entities: dict[str, EntitySetSpec] = {}
        self._tasks: dict[str, TaskSpec] = {}
        self._load()

    def _load_json(self, path: Path) -> Mapping[str, Any]:
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

    def _load(self) -> None:
        entity_dir = self.config_dir / "entities"
        task_dir = self.config_dir / "tasks"
        if not entity_dir.is_dir() or not task_dir.is_dir():
            raise ConfigError(f"Expected '{entity_dir}' and '{task_dir}' configuration directories")

        for path in sorted(entity_dir.glob("*.json")):
            data = self._load_json(path)
            if data.get("schema_version") != 1:
                raise ConfigError(f"{path}: unsupported schema_version '{data.get('schema_version')}'")
            spec = EntitySetSpec.from_dict(data, self.project_root, path)
            if spec.id in self._entities:
                raise ConfigError(f"Duplicate entity-set id '{spec.id}'")
            self._entities[spec.id] = spec

        for path in sorted(task_dir.rglob("*.json")):
            data = self._load_json(path)
            if data.get("schema_version") != 1:
                raise ConfigError(f"{path}: unsupported schema_version '{data.get('schema_version')}'")
            spec = TaskSpec.from_dict(data, self.project_root, path)
            if spec.id in self._tasks:
                raise ConfigError(f"Duplicate task id '{spec.id}'")
            self._tasks[spec.id] = spec

        self.validate(check_data=False)

    def task(self, task_id: str) -> TaskSpec:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._tasks))
            raise ConfigError(f"Unknown task '{task_id}'. Available tasks: {available}") from exc

    def entity_set(self, entity_set_id: str) -> EntitySetSpec:
        try:
            return self._entities[entity_set_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._entities))
            raise ConfigError(f"Unknown entity set '{entity_set_id}'. Available: {available}") from exc

    def tasks(self, suite: str | None = None, entity_set: str | None = None) -> list[TaskSpec]:
        values: Iterable[TaskSpec] = self._tasks.values()
        if suite is not None:
            values = (task for task in values if task.suite == suite)
        if entity_set is not None:
            values = (task for task in values if task.entity_set == entity_set)
        return sorted(values, key=lambda task: task.id)

    def entity_sets(self) -> list[EntitySetSpec]:
        return sorted(self._entities.values(), key=lambda item: item.id)

    def validate(self, check_data: bool = True) -> None:
        """Validate cross-references and optionally inspect CSV headers and paths."""
        errors: list[str] = []
        safe_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
        for entity in self._entities.values():
            if not safe_id.fullmatch(entity.id):
                errors.append(f"Entity-set id '{entity.id}' may only contain letters, numbers, '_' and '-'")
        for task in self._tasks.values():
            if not safe_id.fullmatch(task.id):
                errors.append(f"Task id '{task.id}' may only contain letters, numbers, '_' and '-'")
            if task.entity_set not in self._entities:
                errors.append(f"Task '{task.id}' references unknown entity set '{task.entity_set}'")
                continue
            if task.suite not in {"generalization", "validation"}:
                errors.append(f"Task '{task.id}' has unsupported suite '{task.suite}'")
            if not task.labels:
                errors.append(f"Task '{task.id}' defines no labels")
            label_ids = [label.id for label in task.labels]
            if len(label_ids) != len(set(label_ids)):
                errors.append(f"Task '{task.id}' has duplicate label ids")
            if not task.datasets:
                errors.append(f"Task '{task.id}' defines no dataset variants")
            if "english" not in task.locales:
                errors.append(f"Task '{task.id}' must define an English locale for canonical analysis labels")
            numerical_count = sum(label.numeric is not None for label in task.labels)
            if numerical_count not in {0, len(task.labels)}:
                errors.append(f"Task '{task.id}' must define numerical values for either all labels or none")
            numerical_values = [label.numeric for label in task.labels if label.numeric is not None]
            if len(numerical_values) != len(set(numerical_values)):
                errors.append(f"Task '{task.id}' has duplicate numerical labels")
            if task.suite == "generalization" and any(label.weight is None for label in task.labels):
                errors.append(f"Generalization task '{task.id}' must assign a weight to every label")
            if task.generation is not None:
                generation = task.generation
                if task.suite != "generalization":
                    errors.append(f"Only generalization tasks can define generation settings ('{task.id}')")
                if generation.samples_per_label <= 0:
                    errors.append(f"Task '{task.id}' generation samples_per_label must be positive")
                if generation.keywords_per_prompt <= 0:
                    errors.append(f"Task '{task.id}' generation keywords_per_prompt must be positive")
                if generation.max_tokens <= 0:
                    errors.append(f"Task '{task.id}' generation max_tokens must be positive")
                if not 0 <= generation.top_p <= 1:
                    errors.append(f"Task '{task.id}' generation top_p must be between 0 and 1")
                for example in generation.few_shot:
                    if example.label not in label_ids:
                        errors.append(
                            f"Task '{task.id}' generation example uses unknown label '{example.label}'"
                        )
                    if not example.keywords:
                        errors.append(f"Task '{task.id}' has a generation example without keywords")
                    if "X" not in example.sentence:
                        errors.append(f"Task '{task.id}' generation example must contain placeholder 'X'")
            for language, locale in task.locales.items():
                if language not in task.text_columns:
                    errors.append(f"Task '{task.id}' has no text column for '{language}'")
                if language not in self._entities[task.entity_set].name_columns:
                    errors.append(f"Entity set '{task.entity_set}' has no name column for '{language}'")
                for field in ("instruction", "text_scale"):
                    if not locale.get(field):
                        errors.append(f"Task '{task.id}' locale '{language}' is missing '{field}'")
                if task.supports_numerical and not locale.get("numerical_scale"):
                    errors.append(f"Task '{task.id}' locale '{language}' is missing 'numerical_scale'")
                for label in task.labels:
                    if language not in label.text or language not in label.tokens:
                        errors.append(f"Task '{task.id}' label '{label.id}' is incomplete for '{language}'")
                    elif not label.tokens[language]:
                        errors.append(f"Task '{task.id}' label '{label.id}' has no tokens for '{language}'")
                texts = [label.text.get(language) for label in task.labels]
                if len(texts) != len(set(texts)):
                    errors.append(f"Task '{task.id}' has duplicate text labels for '{language}'")
                for example in locale.get("few_shot", []):
                    if example.get("label") not in label_ids:
                        errors.append(
                            f"Task '{task.id}' locale '{language}' uses unknown few-shot label "
                            f"'{example.get('label')}'"
                        )

            if check_data:
                self._validate_data(task, errors)

        if errors:
            raise ConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))

    def _validate_data(self, task: TaskSpec, errors: list[str]) -> None:
        import pandas as pd

        entity = self._entities[task.entity_set]
        if not entity.path.is_file():
            errors.append(f"Entity file for '{entity.id}' does not exist: {entity.path}")
        else:
            entity_columns = set(pd.read_csv(entity.path, nrows=0).columns)
            for language in task.languages:
                column = entity.name_columns[language]
                if column not in entity_columns:
                    errors.append(f"Entity file '{entity.path}' has no column '{column}'")

        for variant, path in task.datasets.items():
            if not path.is_file():
                errors.append(f"Task '{task.id}' dataset '{variant}' does not exist: {path}")
                continue
            columns = set(pd.read_csv(path, nrows=0).columns)
            required = {task.label_column, *(task.text_columns[lang] for lang in task.languages)}
            missing = sorted(required - columns)
            if missing:
                errors.append(f"Dataset '{path}' is missing columns: {', '.join(missing)}")

        if task.generation is not None:
            keywords_path = task.generation.keywords_path
            if not keywords_path.is_file():
                errors.append(f"Task '{task.id}' keyword file does not exist: {keywords_path}")
            else:
                keyword_columns = set(pd.read_csv(keywords_path, nrows=0).columns)
                if task.generation.keyword_column not in keyword_columns:
                    errors.append(
                        f"Keyword file '{keywords_path}' has no column "
                        f"'{task.generation.keyword_column}'"
                    )
