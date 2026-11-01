"""Deterministic surrogate screening followed by real simulator validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import (
    OptimizationConfig,
    ParticleConfig,
    ReactorConfig,
    SimulationConfig,
)
from .data import FEATURE_NAMES, TARGET_NAMES
from .simulator import simulate_batch
from .surrogate import Surrogate, predict


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    """Surrogate estimates and measured simulator outputs for one candidate."""

    features: tuple[float, ...]
    surrogate_final_oxide_fraction: float
    surrogate_maximum_temperature: float
    simulator_final_oxide_fraction: float
    simulator_maximum_temperature: float
    simulator_reached_melting: bool
    simulator_feasible: bool


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Complete surrogate-screening and simulator-validation outcome."""

    config: OptimizationConfig
    surrogate_feasible_count: int
    validated_candidates: tuple[ValidatedCandidate, ...]
    recommendation: ValidatedCandidate | None


def optimize(model: Surrogate, config: OptimizationConfig) -> OptimizationResult:
    """Rank seeded surrogate-feasible samples, then validate the leading batch."""
    if not isinstance(config, OptimizationConfig):
        raise TypeError("config must be an OptimizationConfig")
    if getattr(model, "feature_names", None) != FEATURE_NAMES:
        raise ValueError(f"model feature_names must be {FEATURE_NAMES}")
    if getattr(model, "target_names", None) != TARGET_NAMES:
        raise ValueError(f"model target_names must be {TARGET_NAMES}")

    rng = np.random.default_rng(config.seed)
    lower = np.asarray(
        [getattr(config.sampling_bounds, name).lower for name in FEATURE_NAMES]
    )
    upper = np.asarray(
        [getattr(config.sampling_bounds, name).upper for name in FEATURE_NAMES]
    )
    features = rng.uniform(
        lower, upper, size=(config.candidate_count, len(FEATURE_NAMES))
    )
    predictions = np.asarray(predict(model, features))
    expected_shape = (config.candidate_count, len(TARGET_NAMES))
    if predictions.shape != expected_shape:
        raise ValueError(f"model predictions must have shape {expected_shape}")

    finite = np.isfinite(predictions).all(axis=1)
    surrogate_feasible = (
        finite
        & (predictions[:, 0] <= config.max_final_oxide_fraction)
        & (predictions[:, 1] < config.material.melting_temperature)
    )
    feasible_indices = np.flatnonzero(surrogate_feasible)
    surrogate_feasible_count = int(feasible_indices.size)
    if surrogate_feasible_count == 0:
        return OptimizationResult(config, 0, (), None)

    # Stable sorting uses seeded sample order to resolve equal peak temperatures.
    ranked = feasible_indices[
        np.argsort(features[feasible_indices, 0], kind="stable")
    ]
    selected = ranked[: config.validation_count]
    simulations = tuple(
        SimulationConfig(
            reactor=ReactorConfig(
                peak_temperature=features[index, 0],
                residence_time=features[index, 1],
                hydrogen_concentration=features[index, 2],
            ),
            particle=ParticleConfig(
                diameter=features[index, 3],
                initial_temperature=config.initial_temperature,
                initial_oxide_fraction=features[index, 4],
            ),
            material=config.material,
            solver=config.solver,
        )
        for index in selected
    )
    trajectory = simulate_batch(simulations)
    simulator_oxide = np.asarray(trajectory.final_oxide_fraction)
    simulator_temperature = np.asarray(trajectory.maximum_temperature)
    simulator_melting = np.asarray(trajectory.reached_melting, dtype=bool)

    validated = []
    for batch_index, candidate_index in enumerate(selected):
        final_oxide = float(simulator_oxide[batch_index])
        maximum_temperature = float(simulator_temperature[batch_index])
        reached_melting = bool(simulator_melting[batch_index])
        finite_outputs = bool(np.isfinite((final_oxide, maximum_temperature)).all())
        simulator_feasible = (
            finite_outputs
            and final_oxide <= config.max_final_oxide_fraction
            and not reached_melting
        )
        validated.append(
            ValidatedCandidate(
                features=tuple(float(value) for value in features[candidate_index]),
                surrogate_final_oxide_fraction=float(predictions[candidate_index, 0]),
                surrogate_maximum_temperature=float(predictions[candidate_index, 1]),
                simulator_final_oxide_fraction=final_oxide,
                simulator_maximum_temperature=maximum_temperature,
                simulator_reached_melting=reached_melting,
                simulator_feasible=simulator_feasible,
            )
        )

    validated_candidates = tuple(validated)
    recommendation = next(
        (candidate for candidate in validated_candidates if candidate.simulator_feasible),
        None,
    )
    return OptimizationResult(
        config=config,
        surrogate_feasible_count=surrogate_feasible_count,
        validated_candidates=validated_candidates,
        recommendation=recommendation,
    )


def _candidate_data(candidate: ValidatedCandidate) -> dict[str, object]:
    return {
        "features": [float(value) for value in candidate.features],
        "surrogate_final_oxide_fraction": float(
            candidate.surrogate_final_oxide_fraction
        ),
        "surrogate_maximum_temperature": float(candidate.surrogate_maximum_temperature),
        "simulator_final_oxide_fraction": float(candidate.simulator_final_oxide_fraction),
        "simulator_maximum_temperature": float(candidate.simulator_maximum_temperature),
        "simulator_reached_melting": bool(candidate.simulator_reached_melting),
        "simulator_feasible": bool(candidate.simulator_feasible),
    }


def save_optimization_result(
    result: OptimizationResult,
    output_path: str | Path,
    model_checkpoint: str | Path | None = None,
) -> Path:
    """Write a self-contained, explicit JSON optimization report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "synthetic": True,
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "config": asdict(result.config),
        "surrogate_feasible_count": int(result.surrogate_feasible_count),
        "validated_candidate_count": len(result.validated_candidates),
        "validated_candidates": [
            _candidate_data(candidate) for candidate in result.validated_candidates
        ],
        "recommendation": (
            _candidate_data(result.recommendation)
            if result.recommendation is not None
            else None
        ),
        "model_checkpoint": (
            str(model_checkpoint) if model_checkpoint is not None else None
        ),
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path
