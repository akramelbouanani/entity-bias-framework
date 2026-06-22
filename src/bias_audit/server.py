"""Inspect models exposed by an OpenAI-compatible inference server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import requests


def list_served_models(
    host: str = "localhost",
    port: int = 9715,
    timeout: float = 10.0,
) -> list[Mapping[str, Any]]:
    """Return model records from the server's `/v1/models` endpoint."""
    url = f"http://{host}:{port}/v1/models"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Could not query models from {url}: {exc}") from exc

    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError(f"Invalid model-list response from {url}: expected a data array")
    valid = [model for model in models if isinstance(model, dict) and model.get("id")]
    if not valid:
        raise RuntimeError(f"The server at {url} exposes no models")
    return valid


def discover_model(
    host: str = "localhost",
    port: int = 9715,
    timeout: float = 10.0,
) -> str:
    """Return the unique model ID exposed by a server."""
    models = list_served_models(host, port, timeout)
    if len(models) != 1:
        ids = ", ".join(str(model["id"]) for model in models)
        raise RuntimeError(
            f"Server at http://{host}:{port} exposes {len(models)} models ({ids}). "
            "Pass --model explicitly to choose one."
        )
    return str(models[0]["id"])


def model_output_name(model_id: str) -> str:
    """Convert a path or namespaced model ID into a stable output-folder name."""
    normalized = model_id.rstrip("/")
    return Path(normalized).name or normalized.replace("/", "_")

