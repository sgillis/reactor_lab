from __future__ import annotations

import io
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from reactor_lab.platform.models import Base


class FakeStreamingBody:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)
        self.closed = False

    def read(self) -> bytes:
        return self._stream.read()

    def iter_chunks(self, chunk_size: int):
        while chunk := self._stream.read(chunk_size):
            yield chunk

    def close(self) -> None:
        self.closed = True
        self._stream.close()


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.media_types: dict[str, str] = {}

    def put_file(self, key: str, path: str | Path, media_type: str | None = None) -> int:
        payload = Path(path).read_bytes()
        self.objects[key] = payload
        self.media_types[key] = media_type or "application/octet-stream"
        return len(payload)

    def put_bytes(self, key: str, payload: bytes, media_type: str) -> int:
        self.objects[key] = payload
        self.media_types[key] = media_type
        return len(payload)

    def download_file(self, key: str, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.objects[key])

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key]

    def get_object(self, key: str):
        return {"Body": FakeStreamingBody(self.objects[key])}


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def fake_storage():
    return FakeStorage()
