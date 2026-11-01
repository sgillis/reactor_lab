"""Dramatiq worker implementation using the shared scientific workflows."""

from __future__ import annotations

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session, sessionmaker

from ..services import run_optimization, run_simulation, run_training, write_json
from .database import get_session_factory
from .models import ArtifactRecord, ModelRecord, RunRecord, utcnow
from .queue import execute_run
from .schemas import (
    OptimizationRunRequest,
    RunStatus,
    RunType,
    SimulationRunRequest,
    TrainingRunRequest,
)
from .storage import get_storage, media_type_for

__all__ = ["execute_run", "execute_run_by_id"]


def _simulation_projection(outcome: Any) -> dict[str, Any]:
    trajectory = outcome.trajectory
    return {
        "type": RunType.simulation.value,
        "summary": outcome.summary,
        "trajectory": {
            "time": np.asarray(trajectory.time).tolist(),
            "temperature": np.asarray(trajectory.temperature).tolist(),
            "oxide_fraction": np.asarray(trajectory.oxide_fraction).tolist(),
            "gas_temperature": np.asarray(trajectory.gas_temperature).tolist(),
            "reached_melting": bool(trajectory.reached_melting),
        },
    }


def _training_projection(outcome: Any) -> dict[str, Any]:
    return {
        "type": RunType.training.value,
        "metrics": outcome.metrics,
        "history": outcome.result.history,
        "test": {
            "simulated": np.asarray(outcome.simulated["test"]).tolist(),
            "predicted": np.asarray(outcome.predicted["test"]).tolist(),
        },
        "ood": {
            "simulated": np.asarray(outcome.simulated["ood"]).tolist(),
            "predicted": np.asarray(outcome.predicted["ood"]).tolist(),
        },
    }


def _optimization_projection(outcome: Any) -> dict[str, Any]:
    return {"type": RunType.optimization.value, "report": outcome.report}


def _restore_checkpoint(
    run: RunRecord, db: Session, storage: Any, directory: Path
) -> tuple[Any, ModelRecord]:
    if run.model_id is None:
        raise ValueError("optimization run has no model lineage")
    model_record = db.get(ModelRecord, run.model_id)
    if model_record is None:
        raise ValueError("referenced model does not exist")
    artifacts = {
        artifact.name: artifact
        for artifact in db.scalars(
            select(ArtifactRecord).where(
                ArtifactRecord.run_id == model_record.training_run_id,
                ArtifactRecord.name.in_(("model.eqx", "config.json")),
            )
        ).all()
    }
    missing = {"model.eqx", "config.json"}.difference(artifacts)
    if missing:
        raise ValueError(f"model checkpoint is incomplete: missing {', '.join(sorted(missing))}")
    for name, artifact in artifacts.items():
        storage.download_file(artifact.object_key, directory / name)
    from ..surrogate import load_surrogate

    return load_surrogate(directory), model_record


def _store_artifacts(
    run: RunRecord, output_directory: Path, db: Session, storage: Any
) -> None:
    existing = {
        artifact.name: artifact
        for artifact in db.scalars(
            select(ArtifactRecord).where(ArtifactRecord.run_id == run.id)
        ).all()
    }
    for path in sorted(item for item in output_directory.rglob("*") if item.is_file()):
        name = path.relative_to(output_directory).as_posix()
        key = f"runs/{run.id}/{name}"
        media_type = media_type_for(path)
        size_bytes = storage.put_file(key, path, media_type)
        artifact = existing.get(name)
        if artifact is None:
            artifact = ArtifactRecord(
                id=uuid.uuid4(),
                run_id=run.id,
                name=name,
                media_type=media_type,
                size_bytes=size_bytes,
                object_key=key,
            )
            db.add(artifact)
        else:
            artifact.media_type = media_type
            artifact.size_bytes = size_bytes
            artifact.object_key = key


def _execute_claimed_run(run: RunRecord, db: Session, storage: Any) -> None:
    with TemporaryDirectory(prefix=f"reactor-run-{run.id}-") as temporary:
        root = Path(temporary)
        output = root / "output"
        output.mkdir()
        model_record: ModelRecord | None = None
        if run.type == RunType.simulation.value:
            config = SimulationRunRequest.model_validate(run.config).to_domain()
            outcome = run_simulation(config, output)
            projection = _simulation_projection(outcome)
            metrics = outcome.summary
        elif run.type == RunType.training.value:
            request = TrainingRunRequest.model_validate(run.config)
            outcome = run_training(
                request.sampling_domain(), request.training_domain(), output
            )
            projection = _training_projection(outcome)
            metrics = outcome.metrics
            model_record = db.scalar(
                select(ModelRecord).where(ModelRecord.training_run_id == run.id)
            )
            if model_record is None:
                model_record = ModelRecord(
                    id=uuid.uuid4(),
                    training_run_id=run.id,
                    config=run.config,
                    metrics=outcome.metrics,
                )
                db.add(model_record)
            else:
                model_record.config = run.config
                model_record.metrics = outcome.metrics
            run.model_id = model_record.id
        elif run.type == RunType.optimization.value:
            request = OptimizationRunRequest.model_validate(run.config)
            model, lineage = _restore_checkpoint(run, db, storage, root / "checkpoint")
            outcome = run_optimization(
                model,
                request.to_domain(),
                output,
                model_checkpoint=f"model:{lineage.id}",
            )
            projection = _optimization_projection(outcome)
            metrics = {
                "surrogate_feasible_count": outcome.result.surrogate_feasible_count,
                "validated_candidate_count": len(outcome.result.validated_candidates),
                "has_recommendation": outcome.result.recommendation is not None,
            }
        else:
            raise ValueError(f"unsupported run type: {run.type}")

        write_json(output / "results.json", projection)
        _store_artifacts(run, output, db, storage)
        run.metrics = metrics
        run.error = None
        run.status = RunStatus.succeeded.value
        run.updated_at = utcnow()
        db.commit()


def execute_run_by_id(
    run_id: str | uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
    storage: Any | None = None,
) -> None:
    """Atomically claim and execute a queued run; terminal redeliveries are no-ops."""
    run_uuid = uuid.UUID(str(run_id))
    factory = session_factory or get_session_factory()
    artifact_storage = storage or get_storage()
    with factory() as db:
        claimed = cast(
            CursorResult[Any],
            db.execute(
                update(RunRecord)
                .where(
                    RunRecord.id == run_uuid,
                    RunRecord.status == RunStatus.queued.value,
                )
                .values(status=RunStatus.running.value, error=None, updated_at=utcnow())
            ),
        )
        db.commit()
        if claimed.rowcount != 1:
            return
        run = db.get(RunRecord, run_uuid)
        if run is None:
            return
        try:
            _execute_claimed_run(run, db, artifact_storage)
        except Exception as error:
            db.rollback()
            failed_run = db.get(RunRecord, run_uuid)
            if failed_run is not None:
                failed_run.status = RunStatus.failed.value
                failed_run.error = f"{type(error).__name__}: {error}"
                failed_run.updated_at = utcnow()
                db.commit()
