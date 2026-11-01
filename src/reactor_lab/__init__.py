"""Differentiable Reactor Lab public API."""

from .config import (
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
from .data import (
    DatasetSplit,
    SyntheticDataset,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from .evaluation import evaluate_surrogate, regression_metrics
from .optimization import (
    OptimizationResult,
    ValidatedCandidate,
    optimize,
    save_optimization_result,
)
from .simulator import Trajectory, simulate, simulate_batch
from .surrogate import (
    Surrogate,
    TrainingResult,
    load_surrogate,
    predict,
    save_surrogate,
    train_surrogate,
)

__all__ = [
    "Bounds",
    "DatasetSplit",
    "InputBounds",
    "MaterialConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "ParticleConfig",
    "ReactorConfig",
    "SamplingConfig",
    "SimulationConfig",
    "SolverConfig",
    "Surrogate",
    "SyntheticDataset",
    "TrainingConfig",
    "TrainingResult",
    "Trajectory",
    "ValidatedCandidate",
    "evaluate_surrogate",
    "generate_dataset",
    "load_dataset",
    "load_surrogate",
    "optimize",
    "predict",
    "regression_metrics",
    "save_dataset",
    "save_optimization_result",
    "save_surrogate",
    "simulate",
    "simulate_batch",
    "train_surrogate",
]
