"""Stateless FastAPI control plane for runs, models, results, and artifacts."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Annotated, Any, Callable
from urllib.parse import quote

from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import OptimizationConfig, SimulationConfig
from .database import get_db
from .models import ArtifactRecord, ModelRecord, RunRecord, utcnow
from .queue import execute_run
from .schemas import (
    ArtifactResponse,
    ErrorResponse,
    HealthResponse,
    ModelResponse,
    OptimizationResults,
    OptimizationRunRequest,
    RunResponse,
    RunStatus,
    RunType,
    SimulationResults,
    SimulationRunRequest,
    TrainingResults,
    TrainingRunRequest,
)
from .storage import S3Storage, get_storage

app = FastAPI(title="Reactor Lab API", version="1.0.0")
API_PREFIX = "/api/v1"
RunResults = Annotated[
    SimulationResults | TrainingResults | OptimizationResults,
    Field(discriminator="type"),
]


def get_run_queue() -> Any:
    return execute_run


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


def _artifact_response(artifact: ArtifactRecord) -> ArtifactResponse:
    return ArtifactResponse(
        id=artifact.id,
        name=artifact.name,
        media_type=artifact.media_type,
        size_bytes=artifact.size_bytes,
        download_url=f"{API_PREFIX}/artifacts/{artifact.id}/download",
    )


def _run_response(run: RunRecord) -> RunResponse:
    return RunResponse(
        id=run.id,
        type=RunType(run.type),
        status=RunStatus(run.status),
        created_at=run.created_at,
        updated_at=run.updated_at,
        error=run.error,
        model_id=run.model_id,
        config=run.config,
        metrics=run.metrics,
        artifacts=[_artifact_response(artifact) for artifact in run.artifacts],
    )


def _model_response(model: ModelRecord) -> ModelResponse:
    return ModelResponse(
        id=model.id,
        training_run_id=model.training_run_id,
        created_at=model.created_at,
        config=model.config,
        metrics=model.metrics,
    )


def _get_run(run_id: uuid.UUID, db: Session) -> RunRecord:
    run = db.get(RunRecord, run_id)
    if run is None:
        raise _error(status.HTTP_404_NOT_FOUND, "run_not_found", "Run not found")
    return run


def _get_model(model_id: uuid.UUID, db: Session) -> ModelRecord:
    model = db.get(ModelRecord, model_id)
    if model is None:
        raise _error(status.HTTP_404_NOT_FOUND, "model_not_found", "Model not found")
    return model


def _validated_config(factory: Callable[[], Any]) -> Any:
    try:
        return factory()
    except (TypeError, ValueError) as error:
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_config", str(error)) from error


def _submit(run: RunRecord, db: Session, queue: Any) -> RunResponse:
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        queue.send(str(run.id))
    except Exception as error:
        run.status = RunStatus.failed.value
        run.error = f"Queue submission failed: {type(error).__name__}: {error}"
        run.updated_at = utcnow()
        db.commit()
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "queue_unavailable",
            "The run could not be submitted to the worker queue",
        ) from error
    return _run_response(run)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse()


@app.post(
    f"{API_PREFIX}/simulation-runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["runs"],
)
def submit_simulation(
    request: SimulationRunRequest,
    db: Session = Depends(get_db),
    queue: Any = Depends(get_run_queue),
) -> RunResponse:
    config: SimulationConfig = _validated_config(request.to_domain)
    run = RunRecord(
        id=uuid.uuid4(),
        type=RunType.simulation.value,
        status=RunStatus.queued.value,
        config=asdict(config),
    )
    return _submit(run, db, queue)


@app.post(
    f"{API_PREFIX}/training-runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["runs"],
)
def submit_training(
    request: TrainingRunRequest,
    db: Session = Depends(get_db),
    queue: Any = Depends(get_run_queue),
) -> RunResponse:
    sampling_config = _validated_config(request.sampling_domain)
    training_config = _validated_config(request.training_domain)
    run = RunRecord(
        id=uuid.uuid4(),
        type=RunType.training.value,
        status=RunStatus.queued.value,
        config={
            "sampling_config": asdict(sampling_config),
            "training_config": asdict(training_config),
        },
    )
    return _submit(run, db, queue)


@app.post(
    f"{API_PREFIX}/optimization-runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["runs"],
)
def submit_optimization(
    request: OptimizationRunRequest,
    db: Session = Depends(get_db),
    queue: Any = Depends(get_run_queue),
) -> RunResponse:
    model = _get_model(request.model_id, db)
    training_run = db.get(RunRecord, model.training_run_id)
    if training_run is None or training_run.status != RunStatus.succeeded.value:
        raise _error(
            status.HTTP_409_CONFLICT,
            "model_not_ready",
            "The model's training run has not succeeded",
        )
    config: OptimizationConfig = _validated_config(request.to_domain)
    run = RunRecord(
        id=uuid.uuid4(),
        type=RunType.optimization.value,
        status=RunStatus.queued.value,
        model_id=model.id,
        config={
            "model_id": str(model.id),
            "optimization_config": asdict(config),
        },
    )
    return _submit(run, db, queue)


@app.get(f"{API_PREFIX}/runs", response_model=list[RunResponse], tags=["runs"])
def list_runs(
    run_type: RunType | None = Query(default=None, alias="type"),
    run_status: RunStatus | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
) -> list[RunResponse]:
    statement = select(RunRecord).order_by(RunRecord.created_at.desc())
    if run_type is not None:
        statement = statement.where(RunRecord.type == run_type.value)
    if run_status is not None:
        statement = statement.where(RunRecord.status == run_status.value)
    return [_run_response(run) for run in db.scalars(statement).all()]


@app.get(
    f"{API_PREFIX}/runs/{{run_id}}",
    response_model=RunResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["runs"],
)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> RunResponse:
    return _run_response(_get_run(run_id, db))


@app.get(
    f"{API_PREFIX}/runs/{{run_id}}/results",
    response_model=RunResults,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["runs"],
)
def get_run_results(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: S3Storage = Depends(get_storage),
) -> dict[str, Any]:
    run = _get_run(run_id, db)
    artifact = db.scalar(
        select(ArtifactRecord).where(
            ArtifactRecord.run_id == run.id, ArtifactRecord.name == "results.json"
        )
    )
    if run.status != RunStatus.succeeded.value or artifact is None:
        message = (
            f"Results are unavailable while the run is {run.status}"
            if run.status != RunStatus.succeeded.value
            else "The result projection is unavailable"
        )
        raise _error(
            status.HTTP_409_CONFLICT,
            "results_unavailable",
            message,
        )
    try:
        payload = storage.get_bytes(artifact.object_key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            raise _error(
                status.HTTP_409_CONFLICT,
                "results_unavailable",
                "The result projection is unavailable",
            ) from error
        raise
    try:
        result = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise _error(
            status.HTTP_409_CONFLICT,
            "results_unavailable",
            "The result projection is unavailable",
        ) from error
    if not isinstance(result, dict) or result.get("type") != run.type:
        raise _error(
            status.HTTP_409_CONFLICT,
            "results_unavailable",
            "The result projection does not match the run type",
        )
    return result


@app.get(f"{API_PREFIX}/models", response_model=list[ModelResponse], tags=["models"])
def list_models(db: Session = Depends(get_db)) -> list[ModelResponse]:
    statement = select(ModelRecord).order_by(ModelRecord.created_at.desc())
    return [_model_response(model) for model in db.scalars(statement).all()]


@app.get(
    f"{API_PREFIX}/models/{{model_id}}",
    response_model=ModelResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["models"],
)
def get_model(model_id: uuid.UUID, db: Session = Depends(get_db)) -> ModelResponse:
    return _model_response(_get_model(model_id, db))


@app.get(
    f"{API_PREFIX}/artifacts/{{artifact_id}}/download",
    responses={404: {"model": ErrorResponse}},
    tags=["artifacts"],
)
def download_artifact(
    artifact_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: S3Storage = Depends(get_storage),
) -> StreamingResponse:
    artifact = db.get(ArtifactRecord, artifact_id)
    if artifact is None:
        raise _error(
            status.HTTP_404_NOT_FOUND, "artifact_not_found", "Artifact not found"
        )
    response = storage.get_object(artifact.object_key)
    body = response["Body"]

    def content() -> Any:
        try:
            yield from body.iter_chunks(chunk_size=64 * 1024)
        finally:
            body.close()

    headers = {
        "Content-Length": str(artifact.size_bytes),
        "Content-Disposition": (
            f"attachment; filename*=UTF-8''{quote(artifact.name)}"
        ),
    }
    return StreamingResponse(
        content(), media_type=artifact.media_type, headers=headers
    )
