"""Generic prompt rendering and model-output token schemas."""

from __future__ import annotations

from typing import Mapping

from .config import ConfigError, TaskSpec


def build_prompt(
    task: TaskSpec,
    sentence: str,
    language: str = "english",
    few_shot: bool = False,
    numerical: bool = False,
    target: str | None = None,
) -> str:
    """Render any configured task without entity-specific Python dispatch."""
    if language not in task.locales:
        raise ConfigError(f"Task '{task.id}' does not support language '{language}'")
    if numerical and not task.supports_numerical:
        raise ConfigError(f"Task '{task.id}' does not define numerical labels")

    locale = task.locales[language]
    scale_key = "numerical_scale" if numerical else "text_scale"
    chunks = [str(locale["instruction"]), str(locale[scale_key])]

    sentence_prefix = str(locale.get("sentence_prefix", "Sentence: "))
    label_prefix = str(locale.get("label_prefix", "Label: "))
    target_prefix = str(locale.get("target_prefix", "Target: "))

    if few_shot:
        examples = []
        for example in locale.get("few_shot", []):
            label = task.label(str(example["label"]))
            rendered_label = label.numeric if numerical else label.text[language]
            examples.append(
                f"{sentence_prefix}{example['sentence']}\n{label_prefix}{rendered_label}"
            )
        if examples:
            chunks.append("\n\n".join(examples) + "\n\n")

    if task.include_target and target:
        chunks.append(f"{target_prefix}{target}\n")
    chunks.append(f"{sentence_prefix}{sentence}\n{label_prefix}")
    return "".join(chunks)


def category_tokens(task: TaskSpec, language: str = "english", numerical: bool = False) -> Mapping[str, list[str]]:
    """Return ordered output-column names and tokens for one-token classification."""
    if language not in task.languages:
        raise ConfigError(f"Task '{task.id}' does not support language '{language}'")
    if numerical:
        if not task.supports_numerical:
            raise ConfigError(f"Task '{task.id}' does not define numerical labels")
        return {str(label.numeric): [str(label.numeric)] for label in task.labels}
    return {label.text[language]: list(label.tokens[language]) for label in task.labels}


def output_labels(task: TaskSpec, language: str, numerical: bool) -> list[str]:
    """Return probability-column names in their configured semantic order."""
    return list(category_tokens(task, language, numerical))

