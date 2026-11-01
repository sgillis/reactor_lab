"""Create run, model lineage, and artifact metadata tables.

Revision ID: 20261101_0001
Revises:
Create Date: 2026-11-01
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20261101_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint("type IN ('simulation', 'training', 'optimization')", name="ck_runs_type"),
        sa.CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')", name="ck_runs_status"),
    )
    op.create_index(
        "ix_runs_type_status_created_at",
        "runs",
        ["type", "status", "created_at"],
    )
    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["training_run_id"], ["runs.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("training_run_id", name="uq_models_training_run_id"),
    )
    op.create_foreign_key(
        "fk_runs_model_id_models", "runs", "models", ["model_id"], ["id"]
    )
    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "name", name="uq_artifacts_run_id_name"),
        sa.UniqueConstraint("object_key", name="uq_artifacts_object_key"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_artifacts_size_bytes"),
    )
    op.create_index("ix_artifacts_run_id", "artifacts", ["run_id"])


def downgrade() -> None:
    op.drop_table("artifacts")
    op.drop_constraint("fk_runs_model_id_models", "runs", type_="foreignkey")
    op.drop_table("models")
    op.drop_index("ix_runs_type_status_created_at", table_name="runs")
    op.drop_table("runs")
