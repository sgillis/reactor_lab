from __future__ import annotations

import json
import uuid
from dataclasses import asdict

from sqlalchemy import select

from reactor_lab.platform.models import ArtifactRecord, ModelRecord, RunRecord
from reactor_lab.platform.schemas import (
    OptimizationRunRequest,
    RunStatus,
    RunType,
    SimulationRunRequest,
    TrainingRunRequest,
)
from reactor_lab.platform.worker import execute_run_by_id


def _add_run(session_factory, run_type: RunType, config: dict) -> uuid.UUID:
    run_id = uuid.uuid4()
    with session_factory() as session:
        session.add(
            RunRecord(
                id=run_id,
                type=run_type.value,
                status=RunStatus.queued.value,
                config=config,
            )
        )
        session.commit()
    return run_id


def _result(storage, run_id: uuid.UUID) -> dict:
    return json.loads(storage.get_bytes(f"runs/{run_id}/results.json"))


def test_simulation_worker_persists_projection_and_is_redelivery_safe(
    session_factory, fake_storage, monkeypatch
) -> None:
    request = SimulationRunRequest.model_validate({"solver": {"num_steps": 3}})
    config = asdict(request.to_domain())
    run_id = _add_run(session_factory, RunType.simulation, config)

    execute_run_by_id(run_id, session_factory=session_factory, storage=fake_storage)

    with session_factory() as session:
        run = session.get(RunRecord, run_id)
        assert run.status == RunStatus.succeeded.value
        assert run.error is None
        assert run.metrics["schema_version"] == 1
        artifact_count = len(session.scalars(select(ArtifactRecord)).all())
    result = _result(fake_storage, run_id)
    assert len(result["trajectory"]["time"]) == 4
    assert len(result["trajectory"]["temperature"]) == 4
    assert result["summary"]["reached_melting"] == result["trajectory"]["reached_melting"]

    def unexpected_reexecution(*args, **kwargs):
        raise AssertionError("a terminal run must not be recomputed")

    monkeypatch.setattr("reactor_lab.platform.worker.run_simulation", unexpected_reexecution)
    execute_run_by_id(run_id, session_factory=session_factory, storage=fake_storage)
    with session_factory() as session:
        assert len(session.scalars(select(ArtifactRecord)).all()) == artifact_count


def test_worker_records_full_error_for_invalid_stored_config(
    session_factory, fake_storage
) -> None:
    request = SimulationRunRequest()
    config = request.model_dump()
    config["solver"]["num_steps"] = 0
    run_id = _add_run(session_factory, RunType.simulation, config)

    execute_run_by_id(run_id, session_factory=session_factory, storage=fake_storage)

    with session_factory() as session:
        run = session.get(RunRecord, run_id)
        assert run.status == RunStatus.failed.value
        assert "ValueError" in run.error
        assert "num_steps must be at least one" in run.error


def test_training_worker_emits_chart_arrays_and_optimization_reloads_checkpoint(
    session_factory, fake_storage
) -> None:
    training_request = TrainingRunRequest.model_validate(
        {
            "sampling_config": {
                "train_samples": 4,
                "validation_samples": 4,
                "test_samples": 4,
                "ood_samples": 4,
                "solver": {"num_steps": 2},
            },
            "training_config": {
                "hidden_width": 4,
                "hidden_depth": 1,
                "batch_size": 2,
                "epochs": 1,
            },
        }
    )
    training_config = {
        "sampling_config": asdict(training_request.sampling_domain()),
        "training_config": asdict(training_request.training_domain()),
    }
    training_run_id = _add_run(session_factory, RunType.training, training_config)

    execute_run_by_id(
        training_run_id, session_factory=session_factory, storage=fake_storage
    )

    with session_factory() as session:
        training_run = session.get(RunRecord, training_run_id)
        assert training_run.status == RunStatus.succeeded.value
        assert training_run.model_id is not None
        model_id = training_run.model_id
        model = session.get(ModelRecord, model_id)
        assert model.training_run_id == training_run_id
        assert model.config == training_config
        names = {
            artifact.name
            for artifact in session.scalars(
                select(ArtifactRecord).where(ArtifactRecord.run_id == training_run_id)
            ).all()
        }
        assert {"model.eqx", "config.json", "dataset.npz", "results.json"} <= names

    training_result = _result(fake_storage, training_run_id)
    for split_name in ("test", "ood"):
        split = training_result[split_name]
        assert len(split["simulated"]) == 4
        assert len(split["predicted"]) == 4
        assert all(len(targets) == 2 for targets in split["predicted"])
    assert len(training_result["history"]["train_loss"]) == 1

    optimization_request = OptimizationRunRequest.model_validate(
        {
            "model_id": model_id,
            "optimization_config": {
                "candidate_count": 4,
                "validation_count": 2,
                "solver": {"num_steps": 2},
            },
        }
    )
    optimization_config = {
        "model_id": str(model_id),
        "optimization_config": asdict(optimization_request.to_domain()),
    }
    optimization_run_id = uuid.uuid4()
    with session_factory() as session:
        session.add(
            RunRecord(
                id=optimization_run_id,
                type=RunType.optimization.value,
                status=RunStatus.queued.value,
                model_id=model_id,
                config=optimization_config,
            )
        )
        session.commit()

    execute_run_by_id(
        optimization_run_id, session_factory=session_factory, storage=fake_storage
    )

    with session_factory() as session:
        optimization_run = session.get(RunRecord, optimization_run_id)
        assert optimization_run.status == RunStatus.succeeded.value
        assert optimization_run.metrics["validated_candidate_count"] <= 2
    optimization_result = _result(fake_storage, optimization_run_id)
    assert optimization_result["type"] == "optimization"
    assert optimization_result["report"]["model_checkpoint"] == f"model:{model_id}"
    assert len(optimization_result["report"]["validated_candidates"]) <= 2
