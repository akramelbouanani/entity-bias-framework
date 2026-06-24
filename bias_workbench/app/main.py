"""FastAPI application for the local end-to-end bias workbench."""

from __future__ import annotations

from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .assist import AssistPipeline
from .experiment import ExperimentPipeline, inspect_entity_csv
from .generation import LANGUAGES, GenerationPipeline, validate_design
from .jobs import JobManager
from .presets import entity_preset_path, list_entity_presets
from .store import ProjectStore, WORKBENCH_ROOT


STATIC_DIR = WORKBENCH_ROOT / "static"
store = ProjectStore()
generation = GenerationPipeline(store)
experiments = ExperimentPipeline(store)
assist = AssistPipeline(store)
jobs = JobManager(max_workers=1)
app = FastAPI(title="Bias Audit Workbench", version="0.1.0")


class LabelInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    weight: float = Field(ge=-5, le=5)
    token: str | None = Field(default=None, max_length=40)

    @field_validator("token")
    @classmethod
    def clean_token(cls, value: str | None) -> str | None:
        value = value.strip() if value else ""
        return value or None


class ExampleInput(BaseModel):
    sentence: str = Field(min_length=3, max_length=500)
    label: str


class DesignInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=20, max_length=2000)
    instruction: str = Field(min_length=20, max_length=3000)
    higher_weight_meaning: str = Field(min_length=2, max_length=120)
    lower_weight_meaning: str = Field(min_length=2, max_length=120)
    labels: list[LabelInput] = Field(min_length=2, max_length=8)
    examples: list[ExampleInput] = Field(min_length=2, max_length=40)
    languages: list[str] = Field(min_length=1)

    @field_validator("languages")
    @classmethod
    def unique_languages(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class CreateProjectRequest(BaseModel):
    design: DesignInput
    host: str = "localhost"
    port: int = Field(default=9714, ge=1, le=65535)


class TaskAssistRequest(BaseModel):
    brief: str = Field(min_length=15, max_length=3000)
    languages: list[str] = Field(min_length=1)
    host: str = "localhost"
    port: int = Field(default=9714, ge=1, le=65535)


class EntityAssistRequest(BaseModel):
    brief: str = Field(min_length=15, max_length=3000)
    count: int = Field(default=60, ge=6, le=100)
    host: str = "localhost"
    port: int = Field(default=9714, ge=1, le=65535)


class EntityConfiguration(BaseModel):
    name_columns: dict[str, str]


class ExperimentRequest(BaseModel):
    languages: list[str] = Field(min_length=1)
    host: str = "localhost"
    port: int = Field(default=9714, ge=1, le=65535)
    preflight_id: str | None = None


class AdditionalEntityRequest(BaseModel):
    names: dict[str, str]
    reference_filters: dict[str, str] = Field(default_factory=dict)
    host: str = "localhost"
    port: int = Field(default=9714, ge=1, le=65535)


@app.get("/api/options")
def options() -> dict:
    return {
        "languages": [
            {"id": key, **value} for key, value in LANGUAGES.items()
        ],
        "sentences_per_label": generation.sentences_per_label,
        "task_keyword_count": generation.task_keyword_count,
        "general_keyword_count": len(generation.general_keywords),
        "entity_presets": list_entity_presets(),
    }


@app.post("/api/projects", status_code=202)
def create_project(request: CreateProjectRequest) -> dict:
    design = request.design.model_dump()
    try:
        validate_design(design)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = store.create(design)
    job_id = jobs.submit(
        project["id"],
        "generation",
        lambda progress: generation.run(
            project["id"], request.host, request.port, progress
        ),
    )
    return {"project_id": project["id"], "job_id": job_id}


@app.post("/api/assist/task", status_code=202)
def assist_task(request: TaskAssistRequest) -> dict:
    job_id = jobs.submit(
        "task_assist",
        "task_assist",
        lambda progress: assist.design_task(
            request.brief, request.languages, request.host, request.port, progress
        ),
    )
    return {"job_id": job_id}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return store.get(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


@app.post("/api/projects/{project_id}/entities", status_code=201)
async def upload_entities(project_id: str, file: UploadFile = File(...)) -> dict:
    try:
        store.get(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not file.filename or not file.filename.casefold().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Entity CSV must be smaller than 25 MB")
    path = store.artifact(project_id, "entities.csv")
    path.write_bytes(content)
    try:
        inspection = inspect_entity_csv(path)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return inspection


@app.post("/api/projects/{project_id}/entities/preset/{preset_id}", status_code=201)
def use_entity_preset(project_id: str, preset_id: str) -> dict:
    try:
        store.get(project_id)
        source = entity_preset_path(preset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = store.artifact(project_id, "entities.csv")
    shutil.copyfile(source, path)
    try:
        return inspect_entity_csv(path)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Bundled preset is invalid: {exc}") from exc


@app.post("/api/projects/{project_id}/assist/entities", status_code=202)
def assist_entities(project_id: str, request: EntityAssistRequest) -> dict:
    try:
        store.get(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job_id = jobs.submit(
        project_id,
        "entity_assist",
        lambda progress: assist.generate_entities(
            project_id, request.brief, request.count, request.host, request.port, progress
        ),
    )
    return {"project_id": project_id, "job_id": job_id}


@app.post("/api/projects/{project_id}/entity-configuration")
def configure_entities(project_id: str, request: EntityConfiguration) -> dict:
    try:
        return experiments.configure_entities(
            project_id, request.name_columns
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/experiment", status_code=202)
def run_experiment(project_id: str, request: ExperimentRequest) -> dict:
    values = request.model_dump()
    preflight_id = values.pop("preflight_id")
    host, port = values.pop("host"), values.pop("port")
    try:
        project = store.get(project_id)
        experiments._settings(project, values)
        checked = project.get("token_preflight") or {}
        if not preflight_id or checked.get("id") != preflight_id or not checked.get("passed"):
            raise ValueError("Run a successful category-token check for these settings first")
        if checked.get("config") != values or checked.get("host") != host or checked.get("port") != port:
            raise ValueError("Experiment settings changed; run the category-token check again")
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = jobs.submit(
        project_id,
        "experiment",
        lambda progress: experiments.run(project_id, values, host, port, progress),
    )
    return {"project_id": project_id, "job_id": job_id}


@app.post("/api/projects/{project_id}/token-preflight")
def token_preflight(project_id: str, request: ExperimentRequest) -> dict:
    values = request.model_dump()
    values.pop("preflight_id")
    host, port = values.pop("host"), values.pop("port")
    try:
        return experiments.preflight_tokens(project_id, values, host, port)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/report")
def get_report(project_id: str) -> dict:
    try:
        project = store.get(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not project.get("report") or not Path(project["report"]).is_file():
        raise HTTPException(status_code=404, detail="This project has no completed report")
    import json

    with Path(project["report"]).open(encoding="utf-8") as handle:
        return json.load(handle)


@app.post("/api/projects/{project_id}/additional-entity", status_code=202)
def additional_entity(project_id: str, request: AdditionalEntityRequest) -> dict:
    try:
        store.get(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job_id = jobs.submit(
        project_id,
        "additional_entity",
        lambda progress: experiments.run_additional_entity(
            project_id,
            request.names,
            request.reference_filters,
            request.host,
            request.port,
            progress,
        ),
    )
    return {"project_id": project_id, "job_id": job_id}


@app.get("/api/projects/{project_id}/download/{artifact}")
def download_artifact(project_id: str, artifact: str) -> FileResponse:
    allowed = {
        "sentences": ("generated_sentences.csv", "generated_sentences.csv"),
        "scores": ("entity_scores.csv", "entity_scores.csv"),
        "entities": ("entities.csv", "entities.csv"),
    }
    if artifact not in allowed:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = store.artifact(project_id, allowed[artifact][0])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is not available yet")
    return FileResponse(path, filename=allowed[artifact][1])


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
