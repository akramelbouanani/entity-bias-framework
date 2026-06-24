"""Small persistent project store for local workbench runs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = WORKBENCH_ROOT / "runs"


class ProjectStore:
    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def create(self, design: dict[str, Any]) -> dict[str, Any]:
        project_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        project = {
            "id": project_id,
            "stage": "design",
            "created_at": now,
            "updated_at": now,
            "design": design,
            "generation": None,
            "entities": None,
            "experiment": None,
            "report": None,
        }
        self.project_dir(project_id).mkdir(parents=True, exist_ok=False)
        self.save(project)
        return project

    def get(self, project_id: str) -> dict[str, Any]:
        path = self.project_dir(project_id) / "project.json"
        if not path.is_file():
            raise KeyError(f"Unknown project '{project_id}'")
        with self.lock, path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, project: dict[str, Any]) -> dict[str, Any]:
        project = dict(project)
        project["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self.project_dir(project["id"]) / "project.json"
        temporary = path.with_suffix(".json.tmp")
        with self.lock, temporary.open("w", encoding="utf-8") as handle:
            json.dump(project, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
        return project

    def update(self, project_id: str, **values: Any) -> dict[str, Any]:
        with self.lock:
            project = self.get(project_id)
            project.update(values)
            return self.save(project)

    def project_dir(self, project_id: str) -> Path:
        if not project_id.isalnum():
            raise ValueError("Invalid project id")
        return self.runs_dir / project_id

    def artifact(self, project_id: str, *parts: str) -> Path:
        base = self.project_dir(project_id).resolve()
        path = base.joinpath(*parts).resolve()
        if base not in path.parents and path != base:
            raise ValueError("Artifact path escapes project directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
