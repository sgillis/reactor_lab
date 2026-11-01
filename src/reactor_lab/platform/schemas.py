"""Typed FastAPI boundary and numerical result projections."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..config import (
    Bounds,
    InputBounds,
    MaterialConfig,
    OptimizationConfig,
    ParticleConfig,
    ReactorConfig,
    SamplingConfig,
    SimulationConfig,
    SolverConfig,
    TrainingConfig,
)


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunType(str, Enum):
    simulation = "simulation"
    training = "training"
    optimization = "optimization"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ReactorSchema(Schema):
    peak_temperature: float = 2_200.0
    residence_time: float = 0.5
    hydrogen_concentration: float = 50.0

    def to_domain(self) -> ReactorConfig:
        return ReactorConfig(**self.model_dump())


class ParticleSchema(Schema):
    diameter: float = 50e-6
    initial_temperature: float = 300.0
    initial_oxide_fraction: float = 0.9

    def to_domain(self) -> ParticleConfig:
        return ParticleConfig(**self.model_dump())


class MaterialSchema(Schema):
    density: float = 5_000.0
    heat_capacity: float = 500.0
    heat_transfer_coefficient: float = 100.0
    emissivity: float = 0.8
    surroundings_temperature: float = 300.0
    melting_temperature: float = 1_800.0
    reaction_pre_exponential: float = 0.05
    activation_energy: float = 30_000.0

    def to_domain(self) -> MaterialConfig:
        return MaterialConfig(**self.model_dump())


class SolverSchema(Schema):
    num_steps: int = 1_000

    def to_domain(self) -> SolverConfig:
        return SolverConfig(**self.model_dump())


class BoundsSchema(Schema):
    lower: float
    upper: float

    def to_domain(self) -> Bounds:
        return Bounds(**self.model_dump())


class InputBoundsSchema(Schema):
    peak_temperature: BoundsSchema = Field(
        default_factory=lambda: BoundsSchema(lower=1_800.0, upper=2_800.0)
    )
    residence_time: BoundsSchema = Field(
        default_factory=lambda: BoundsSchema(lower=0.2, upper=0.8)
    )
    hydrogen_concentration: BoundsSchema = Field(
        default_factory=lambda: BoundsSchema(lower=20.0, upper=100.0)
    )
    diameter: BoundsSchema = Field(
        default_factory=lambda: BoundsSchema(lower=30e-6, upper=80e-6)
    )
    initial_oxide_fraction: BoundsSchema = Field(
        default_factory=lambda: BoundsSchema(lower=0.6, upper=1.0)
    )

    def to_domain(self) -> InputBounds:
        return InputBounds(
            peak_temperature=self.peak_temperature.to_domain(),
            residence_time=self.residence_time.to_domain(),
            hydrogen_concentration=self.hydrogen_concentration.to_domain(),
            diameter=self.diameter.to_domain(),
            initial_oxide_fraction=self.initial_oxide_fraction.to_domain(),
        )


class SamplingConfigSchema(Schema):
    sampling_bounds: InputBoundsSchema = Field(default_factory=InputBoundsSchema)
    ood_bounds: InputBoundsSchema = Field(
        default_factory=lambda: InputBoundsSchema(
            peak_temperature=BoundsSchema(lower=2_800.0, upper=3_400.0)
        )
    )
    initial_temperature: float = 300.0
    material: MaterialSchema = Field(default_factory=MaterialSchema)
    solver: SolverSchema = Field(default_factory=lambda: SolverSchema(num_steps=200))
    train_samples: int = 128
    validation_samples: int = 32
    test_samples: int = 32
    ood_samples: int = 32
    seed: int = 2_025

    def to_domain(self) -> SamplingConfig:
        values = self.model_dump()
        values["sampling_bounds"] = self.sampling_bounds.to_domain()
        values["ood_bounds"] = self.ood_bounds.to_domain()
        values["material"] = self.material.to_domain()
        values["solver"] = self.solver.to_domain()
        return SamplingConfig(**values)


class TrainingConfigSchema(Schema):
    hidden_width: int = 32
    hidden_depth: int = 2
    learning_rate: float = 0.003
    batch_size: int = 64
    epochs: int = 50
    seed: int = 2_025

    def to_domain(self) -> TrainingConfig:
        return TrainingConfig(**self.model_dump())


class OptimizationConfigSchema(Schema):
    sampling_bounds: InputBoundsSchema = Field(default_factory=InputBoundsSchema)
    initial_temperature: float = 300.0
    material: MaterialSchema = Field(default_factory=MaterialSchema)
    solver: SolverSchema = Field(default_factory=lambda: SolverSchema(num_steps=200))
    candidate_count: int = 4_096
    validation_count: int = 16
    max_final_oxide_fraction: float = 0.75
    seed: int = 2_025

    def to_domain(self) -> OptimizationConfig:
        values = self.model_dump()
        values["sampling_bounds"] = self.sampling_bounds.to_domain()
        values["material"] = self.material.to_domain()
        values["solver"] = self.solver.to_domain()
        return OptimizationConfig(**values)


class SimulationRunRequest(Schema):
    reactor: ReactorSchema = Field(default_factory=ReactorSchema)
    particle: ParticleSchema = Field(default_factory=ParticleSchema)
    material: MaterialSchema = Field(default_factory=MaterialSchema)
    solver: SolverSchema = Field(default_factory=SolverSchema)

    def to_domain(self) -> SimulationConfig:
        return SimulationConfig(
            reactor=self.reactor.to_domain(),
            particle=self.particle.to_domain(),
            material=self.material.to_domain(),
            solver=self.solver.to_domain(),
        )


class TrainingRunRequest(Schema):
    sampling_config: SamplingConfigSchema = Field(default_factory=SamplingConfigSchema)
    training_config: TrainingConfigSchema = Field(default_factory=TrainingConfigSchema)

    def sampling_domain(self) -> SamplingConfig:
        return self.sampling_config.to_domain()

    def training_domain(self) -> TrainingConfig:
        return self.training_config.to_domain()


class OptimizationRunRequest(Schema):
    model_id: UUID
    optimization_config: OptimizationConfigSchema = Field(
        default_factory=OptimizationConfigSchema
    )

    def to_domain(self) -> OptimizationConfig:
        return self.optimization_config.to_domain()


class ArtifactResponse(BaseModel):
    id: UUID
    name: str
    media_type: str
    size_bytes: int
    download_url: str


class RunResponse(BaseModel):
    id: UUID
    type: RunType
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    error: str | None
    model_id: UUID | None
    config: dict[str, Any]
    metrics: dict[str, Any] | None
    artifacts: list[ArtifactResponse]


class ModelResponse(BaseModel):
    id: UUID
    training_run_id: UUID
    created_at: datetime
    config: dict[str, Any]
    metrics: dict[str, Any]


class SimulationSummary(BaseModel):
    schema_version: int
    synthetic: bool
    final_oxide_fraction: float
    maximum_temperature: float
    reached_melting: bool


class TrajectoryProjection(BaseModel):
    time: list[float]
    temperature: list[float]
    oxide_fraction: list[float]
    gas_temperature: list[float]
    reached_melting: bool


class SimulationResults(BaseModel):
    type: Literal["simulation"]
    summary: SimulationSummary
    trajectory: TrajectoryProjection


class TrainingSplitProjection(BaseModel):
    simulated: list[list[float]]
    predicted: list[list[float]]


class TrainingResults(BaseModel):
    type: Literal["training"]
    metrics: dict[str, dict[str, dict[str, float | None]]]
    history: dict[str, list[float]]
    test: TrainingSplitProjection
    ood: TrainingSplitProjection


class OptimizationResults(BaseModel):
    type: Literal["optimization"]
    report: dict[str, Any]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
