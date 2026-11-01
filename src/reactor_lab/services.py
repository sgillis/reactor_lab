"""Reusable scientific workflows shared by the CLI and asynchronous workers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import OptimizationConfig, SamplingConfig, SimulationConfig, TrainingConfig
from .data import generate_dataset, save_dataset
from .evaluation import evaluate_surrogate
from .optimization import OptimizationResult, optimize, save_optimization_result
from .plotting import (
    plot_surrogate_predictions,
    plot_surrogate_residuals,
    plot_trajectories,
    plot_training_history,
)
from .simulator import Trajectory, simulate
from .surrogate import (
    Surrogate,
    TrainingResult,
    load_surrogate,
    predict,
    save_surrogate,
    train_surrogate,
)


def json_value(value: Any) -> Any:
    """Convert scientific configuration and results to JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(output_path: str | Path, value: Any) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class SimulationOutcome:
    config: SimulationConfig
    trajectory: Trajectory
    summary: dict[str, bool | float | int]


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    model: Surrogate
    result: TrainingResult
    metrics: dict[str, dict[str, dict[str, float | None]]]
    simulated: dict[str, np.ndarray]
    predicted: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class OptimizationOutcome:
    result: OptimizationResult
    report: dict[str, Any]


def run_simulation(
    config: SimulationConfig,
    output_directory: str | Path,
    *,
    plot: bool = True,
) -> SimulationOutcome:
    """Simulate one particle and save the established trajectory artifacts."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    output_directory = Path(output_directory)
    trajectory = simulate(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "trajectory.npz",
        time=np.asarray(trajectory.time),
        temperature=np.asarray(trajectory.temperature),
        oxide_fraction=np.asarray(trajectory.oxide_fraction),
        gas_temperature=np.asarray(trajectory.gas_temperature),
        reached_melting=np.asarray(trajectory.reached_melting),
    )
    summary: dict[str, bool | float | int] = {
        "schema_version": 1,
        "synthetic": True,
        "final_oxide_fraction": float(trajectory.final_oxide_fraction),
        "maximum_temperature": float(trajectory.maximum_temperature),
        "reached_melting": bool(trajectory.reached_melting),
    }
    write_json(output_directory / "config.json", config)
    write_json(output_directory / "result.json", summary)
    if plot:
        plot_trajectories(
            {"Simulation": trajectory}, output_directory / "trajectories.png"
        )
    return SimulationOutcome(config, trajectory, summary)


def run_training(
    sampling_config: SamplingConfig,
    training_config: TrainingConfig,
    output_directory: str | Path,
) -> TrainingOutcome:
    """Generate data, train, reload the checkpoint, and evaluate its predictions."""
    if not isinstance(sampling_config, SamplingConfig):
        raise TypeError("sampling_config must be a SamplingConfig")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(sampling_config)
    save_dataset(dataset, output_directory / "dataset.npz")
    print(f"Generated {dataset.features.shape[0]} synthetic simulations")

    result = train_surrogate(dataset, training_config)
    save_surrogate(result.model, output_directory)
    write_json(output_directory / "history.json", result.history)

    # Evaluate the serialized checkpoint, preserving checkpoint compatibility.
    model = load_surrogate(output_directory)
    metrics = evaluate_surrogate(model, dataset)
    write_json(output_directory / "metrics.json", metrics)

    simulated: dict[str, np.ndarray] = {}
    predicted: dict[str, np.ndarray] = {}
    for split_name in ("test", "ood"):
        split = dataset.get_split(split_name)
        simulated[split_name] = split.targets
        predicted[split_name] = np.asarray(predict(model, split.features))
    plot_surrogate_predictions(simulated, predicted, output_directory / "predictions.png")
    plot_surrogate_residuals(simulated, predicted, output_directory / "residuals.png")
    plot_training_history(result.history, output_directory / "training-history.png")
    return TrainingOutcome(model, result, metrics, simulated, predicted)


def run_optimization(
    model: Surrogate,
    config: OptimizationConfig,
    output_directory: str | Path,
    *,
    model_checkpoint: str | Path | None = None,
) -> OptimizationOutcome:
    """Screen and validate candidates, saving the established JSON report."""
    if not isinstance(model, Surrogate):
        raise TypeError("model must be a Surrogate")
    if not isinstance(config, OptimizationConfig):
        raise TypeError("config must be an OptimizationConfig")
    output_directory = Path(output_directory)
    result = optimize(model, config)
    report_path = save_optimization_result(
        result,
        output_directory / "optimization.json",
        model_checkpoint=model_checkpoint,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return OptimizationOutcome(result, report)
