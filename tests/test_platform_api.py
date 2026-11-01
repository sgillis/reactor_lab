from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from reactor_lab.platform.api import app
from reactor_lab.platform.database import get_db
from reactor_lab.platform.models import ArtifactRecord, ModelRecord, RunRecord
from reactor_lab.platform.schemas import RunStatus, RunType
from reactor_lab.platform.storage import get_storage


class FakeQueue:
    def __init__(self) -> None:
        self.run_ids: list[str] = []
        self.error: Exception | None = None

    def send(self, run_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.run_ids.append(run_id)


@pytest.fixture
def api_context(session_factory, fake_storage):
    queue = FakeQueue()

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    from reactor_lab.platform.api import get_run_queue

    app.dependency_overrides[get_run_queue] = lambda: queue
    with TestClient(app) as client:
        yield client, queue, session_factory, fake_storage
    app.dependency_overrides.clear()


def test_health_and_simulation_submission_do_not_compute(api_context) -> None:
    client, queue, session_factory, _ = api_context
    assert client.get("/api/v1/health").json() == {"status": "ok"}

    response = client.post("/api/v1/simulation-runs", json={})

    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "queued"
    assert run["type"] == "simulation"
    assert run["config"]["reactor"]["peak_temperature"] == 2_200.0
    assert run["config"]["solver"]["num_steps"] == 1_000
    assert queue.run_ids == [run["id"]]
    assert client.get("/api/v1/runs", params={"type": "simulation"}).json()[0]["id"] == run["id"]
    assert client.get("/api/v1/runs", params={"status": "succeeded"}).json() == []
    with session_factory() as session:
        assert session.get(RunRecord, uuid.UUID(run["id"])).status == "queued"


def test_structured_404_409_and_domain_validation(api_context) -> None:
    client, _, _, _ = api_context
    missing = client.get(f"/api/v1/runs/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "run_not_found"

    created = client.post("/api/v1/simulation-runs", json={}).json()
    unavailable = client.get(f"/api/v1/runs/{created['id']}/results")
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"]["code"] == "results_unavailable"

    invalid = client.post(
        "/api/v1/simulation-runs", json={"solver": {"num_steps": 0}}
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "invalid_config"


def test_queue_failure_is_persistent_and_structured(api_context) -> None:
    client, queue, session_factory, _ = api_context
    queue.error = RuntimeError("redis unavailable")
    response = client.post("/api/v1/simulation-runs", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "queue_unavailable"
    run = client.get("/api/v1/runs").json()[0]
    assert run["status"] == "failed"
    assert "redis unavailable" in run["error"]
    with session_factory() as session:
        assert session.get(RunRecord, uuid.UUID(run["id"])).status == "failed"


def test_results_and_artifact_download_use_recorded_object_key(api_context) -> None:
    client, _, session_factory, storage = api_context
    run_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    projection = {
        "type": "simulation",
        "summary": {
            "schema_version": 1,
            "synthetic": True,
            "final_oxide_fraction": 0.25,
            "maximum_temperature": 900.0,
            "reached_melting": False,
        },
        "trajectory": {
            "time": [0.0, 1.0],
            "temperature": [300.0, 900.0],
            "oxide_fraction": [0.5, 0.25],
            "gas_temperature": [300.0, 1_000.0],
            "reached_melting": False,
        },
    }
    payload = json.dumps(projection).encode()
    key = f"runs/{run_id}/results.json"
    storage.put_bytes(key, payload, "application/json")
    with session_factory() as session:
        session.add(
            RunRecord(
                id=run_id,
                type=RunType.simulation.value,
                status=RunStatus.succeeded.value,
                config={},
                metrics=projection["summary"],
            )
        )
        session.add(
            ArtifactRecord(
                id=artifact_id,
                run_id=run_id,
                name="results.json",
                media_type="application/json",
                size_bytes=len(payload),
                object_key=key,
            )
        )
        session.commit()

    result = client.get(f"/api/v1/runs/{run_id}/results")
    assert result.status_code == 200
    assert result.json()["trajectory"]["temperature"] == [300.0, 900.0]
    download = client.get(f"/api/v1/artifacts/{artifact_id}/download")
    assert download.status_code == 200
    assert download.content == payload
    assert download.headers["content-length"] == str(len(payload))


def test_optimization_submission_preserves_model_lineage(api_context) -> None:
    client, queue, session_factory, _ = api_context
    model_id = uuid.uuid4()
    training_run_id = uuid.uuid4()
    with session_factory() as session:
        session.add(
            RunRecord(
                id=training_run_id,
                type=RunType.training.value,
                status=RunStatus.succeeded.value,
                config={},
            )
        )
        session.add(
            ModelRecord(
                id=model_id,
                training_run_id=training_run_id,
                config={},
                metrics={},
            )
        )
        session.commit()

    response = client.post(
        "/api/v1/optimization-runs",
        json={"model_id": str(model_id), "optimization_config": {"candidate_count": 8}},
    )
    assert response.status_code == 202
    run = response.json()
    assert run["model_id"] == str(model_id)
    assert run["config"]["optimization_config"]["candidate_count"] == 8
    assert run["config"]["optimization_config"]["solver"]["num_steps"] == 200
    assert queue.run_ids == [run["id"]]
    models = client.get("/api/v1/models").json()
    assert models[0]["training_run_id"] == str(training_run_id)
