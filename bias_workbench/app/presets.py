"""Static, validated entity cohorts bundled with the local workbench."""

from __future__ import annotations

import json
from pathlib import Path

from .store import WORKBENCH_ROOT


PRESET_DIR = WORKBENCH_ROOT / "data" / "entity_presets"
MANIFEST_PATH = PRESET_DIR / "manifest.json"


def list_entity_presets() -> list[dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise RuntimeError("Entity preset manifest is malformed")
    return presets


def entity_preset_path(preset_id: str) -> Path:
    matching = [item for item in list_entity_presets() if item.get("id") == preset_id]
    if not matching:
        raise KeyError(f"Unknown entity preset '{preset_id}'")
    filename = str(matching[0].get("file", ""))
    path = (PRESET_DIR / filename).resolve()
    if path.parent != PRESET_DIR.resolve() or not path.is_file():
        raise RuntimeError(f"Entity preset '{preset_id}' is unavailable")
    return path
