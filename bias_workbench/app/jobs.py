"""A local single-GPU job queue with progress snapshots."""

from __future__ import annotations

import copy
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from typing import Callable


class JobManager:
    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bias-workbench")
        self.jobs: dict[str, dict] = {}
        self.lock = RLock()

    def submit(self, project_id: str, stage: str, runner: Callable) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "project_id": project_id,
                "stage": stage,
                "status": "queued",
                "progress": 0,
                "message": "Waiting for the local GPU…",
                "result": None,
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        self.executor.submit(self._run, job_id, runner)
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            value = self.jobs.get(job_id)
            return copy.deepcopy(value) if value else None

    def _run(self, job_id: str, runner: Callable) -> None:
        self._update(job_id, status="running", progress=1, message="Starting…")

        def progress(percentage: float, message: str) -> None:
            self._update(
                job_id,
                progress=max(1, min(99, int(round(percentage)))),
                message=message,
            )

        try:
            result = runner(progress)
            self._update(
                job_id,
                status="completed",
                progress=100,
                message="Done ✦",
                result=result,
            )
        except Exception as exc:
            traceback.print_exc()
            self._update(
                job_id,
                status="failed",
                message="This stage could not be completed",
                error=str(exc),
            )

    def _update(self, job_id: str, **values) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(values)
