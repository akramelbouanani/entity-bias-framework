"""Configuration-driven entity bias auditing framework."""

from .config import (
    ConfigError,
    EntitySetSpec,
    GenerationExample,
    GenerationSpec,
    LabelSpec,
    TaskRegistry,
    TaskSpec,
)
from .data import expand_task
from .prompts import build_prompt, category_tokens
from .server import discover_model, list_served_models, model_output_name

__all__ = [
    "ConfigError",
    "EntitySetSpec",
    "GenerationExample",
    "GenerationSpec",
    "LabelSpec",
    "TaskRegistry",
    "TaskSpec",
    "build_prompt",
    "category_tokens",
    "expand_task",
    "discover_model",
    "list_served_models",
    "model_output_name",
]

__version__ = "0.1.0"
