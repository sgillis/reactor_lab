"""SQLAlchemy records for authoritative run, model, and artifact metadata."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(UTC)


JsonDocument = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_type_status_created_at", "type", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    error: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("models.id"), nullable=True
    )
    config: Mapped[dict[str, Any]] = mapped_column(JsonDocument, nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JsonDocument)
    artifacts: Mapped[list[ArtifactRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("runs.id", ondelete="RESTRICT"), unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    config: Mapped[dict[str, Any]] = mapped_column(JsonDocument, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JsonDocument, nullable=False)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "name", name="uq_artifacts_run_id_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    run: Mapped[RunRecord] = relationship(back_populates="artifacts")
