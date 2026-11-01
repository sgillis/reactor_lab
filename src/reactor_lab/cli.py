"""Command-line workflows for simulation, training, optimization, and demos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    OptimizationConfig,
    ParticleConfig,
    ReactorConfig,
    SamplingConfig,
    SimulationConfig,
    SolverConfig,
    TrainingConfig,
)
from .data import generate_dataset, save_dataset
from .evaluation import evaluate_surrogate
from .optimization import optimize, save_optimization_result
from .plotting import (
    plot_surrogate_predictions,
    plot_trajectories,
    plot_surrogate_residuals,
    plot_training_history,
)
from .simulator import simulate
from .surrogate import (
    Surrogate,
    TrainingResult,
    load_surrogate,
    predict,
    save_surrogate,
    train_surrogate,
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(output_path: Path, value: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _derived_split_size(train_samples: int) -> int:
    return max(4, train_samples // 4)


def _sampling_config(args: argparse.Namespace) -> SamplingConfig:
    other_samples = _derived_split_size(args.train_samples)
    return SamplingConfig(
        train_samples=args.train_samples,
        validation_samples=other_samples,
        test_samples=other_samples,
        ood_samples=other_samples,
        solver=SolverConfig(num_steps=args.solver_steps),
        seed=args.seed,
    )


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(epochs=args.epochs, seed=args.seed)


def _optimization_config(args: argparse.Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        candidate_count=args.candidate_count,
        validation_count=args.validation_count,
        max_final_oxide_fraction=args.max_final_oxide_fraction,
        solver=SolverConfig(num_steps=args.solver_steps),
        seed=args.seed,
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--solver-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2_025)


def _add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-count", type=int, default=4_096)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--max-final-oxide-fraction", type=float, default=0.75)
    parser.add_argument("--solver-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2_025)


def _print_metrics(metrics: dict[str, dict[str, dict[str, float | None]]]) -> None:
    for split_name, split_metrics in metrics.items():
        regime = "in-distribution" if split_name == "test" else "separated regime"
        print(f"{split_name} ({regime}):")
        for target_name, target_metrics in split_metrics.items():
            r2 = target_metrics["r2"]
            r2_text = "undefined" if r2 is None else f"{r2:.4f}"
            print(
                f"  {target_name}: MAE={target_metrics['mae']:.6g}, "
                f"RMSE={target_metrics['rmse']:.6g}, R²={r2_text}"
            )


def _run_simulation(args: argparse.Namespace) -> int:
    config = SimulationConfig(
        reactor=ReactorConfig(
            peak_temperature=args.peak_temperature,
            residence_time=args.residence_time,
            hydrogen_concentration=args.hydrogen_concentration,
        ),
        particle=ParticleConfig(
            diameter=args.diameter,
            initial_temperature=args.initial_temperature,
            initial_oxide_fraction=args.initial_oxide_fraction,
        ),
        solver=SolverConfig(num_steps=args.solver_steps),
    )
    trajectory = simulate(config)
    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "trajectory.npz",
        time=np.asarray(trajectory.time),
        temperature=np.asarray(trajectory.temperature),
        oxide_fraction=np.asarray(trajectory.oxide_fraction),
        gas_temperature=np.asarray(trajectory.gas_temperature),
        reached_melting=np.asarray(trajectory.reached_melting),
    )
    summary = {
        "schema_version": 1,
        "synthetic": True,
        "final_oxide_fraction": float(trajectory.final_oxide_fraction),
        "maximum_temperature": float(trajectory.maximum_temperature),
        "reached_melting": bool(trajectory.reached_melting),
    }
    _write_json(output_directory / "config.json", config)
    _write_json(output_directory / "result.json", summary)
    if args.plot:
        plot_trajectories(
            {"Simulation": trajectory}, output_directory / "trajectories.png"
        )
    print(
        f"Final oxide fraction: {summary['final_oxide_fraction']:.6f}\n"
        f"Maximum particle temperature: {summary['maximum_temperature']:.2f} K\n"
        f"Reached melting: {summary['reached_melting']}\n"
        f"Saved simulation to {output_directory}"
    )
    return 0


def _train_and_save(
    sampling_config: SamplingConfig,
    training_config: TrainingConfig,
    output_directory: Path,
) -> tuple[Surrogate, TrainingResult, dict[str, dict[str, dict[str, float | None]]]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(sampling_config)
    save_dataset(dataset, output_directory / "dataset.npz")
    print(f"Generated {dataset.features.shape[0]} synthetic simulations")

    result = train_surrogate(dataset, training_config)
    save_surrogate(result.model, output_directory)
    _write_json(output_directory / "history.json", result.history)

    # Evaluate the serialized checkpoint, not just the in-memory model.
    model = load_surrogate(output_directory)
    metrics = evaluate_surrogate(model, dataset)
    _write_json(output_directory / "metrics.json", metrics)

    simulated = {}
    predicted = {}
    for split_name in ("test", "ood"):
        split = dataset.get_split(split_name)
        simulated[split_name] = split.targets
        predicted[split_name] = np.asarray(predict(model, split.features))
    plot_surrogate_predictions(simulated, predicted, output_directory / "predictions.png")
    plot_surrogate_residuals(simulated, predicted, output_directory / "residuals.png")
    plot_training_history(result.history, output_directory / "training-history.png")
    return model, result, metrics


def _run_training(args: argparse.Namespace) -> int:
    _, _, metrics = _train_and_save(
        _sampling_config(args), _training_config(args), args.output_directory
    )
    _print_metrics(metrics)
    print(f"Saved synthetic dataset, model, metrics, and plots to {args.output_directory}")
    return 0


def _optimize_and_save(
    model: Surrogate,
    config: OptimizationConfig,
    output_directory: Path,
    model_checkpoint: Path,
) -> int:
    result = optimize(model, config)
    report_path = save_optimization_result(
        result,
        output_directory / "optimization.json",
        model_checkpoint=model_checkpoint,
    )
    recommendation = result.recommendation
    if recommendation is None:
        print(
            "No simulator-validated recommendation found; "
            f"surrogate-feasible candidates: {result.surrogate_feasible_count}"
        )
    else:
        print(
            f"Recommended peak-temperature energy proxy: {recommendation.features[0]:.2f} K\n"
            "Surrogate prediction: "
            f"oxide={recommendation.surrogate_final_oxide_fraction:.6f}, "
            f"maximum temperature={recommendation.surrogate_maximum_temperature:.2f} K\n"
            "Simulator validation: "
            f"oxide={recommendation.simulator_final_oxide_fraction:.6f}, "
            f"maximum temperature={recommendation.simulator_maximum_temperature:.2f} K, "
            f"reached melting={recommendation.simulator_reached_melting}"
        )
    print(f"Saved optimization report to {report_path}")
    return 0


def _run_optimization(args: argparse.Namespace) -> int:
    model = load_surrogate(args.model_directory)
    return _optimize_and_save(
        model,
        _optimization_config(args),
        args.output_directory,
        args.model_directory,
    )


def _run_demo(args: argparse.Namespace) -> int:
    sampling_config = _sampling_config(args)
    training_config = _training_config(args)
    optimization_config = _optimization_config(args)
    output_directory = args.output_directory
    _write_json(
        output_directory / "run-config.json",
        {
            "schema_version": 1,
            "synthetic": True,
            "sampling_config": sampling_config,
            "training_config": training_config,
            "optimization_config": optimization_config,
        },
    )
    model, _, metrics = _train_and_save(
        sampling_config, training_config, output_directory
    )
    _print_metrics(metrics)
    return _optimize_and_save(
        model, optimization_config, output_directory, output_directory
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reactor-lab",
        description="Synthetic reduced-order reactor simulation and optimization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="simulate one particle")
    simulate_parser.add_argument("--peak-temperature", type=float, default=2_200.0)
    simulate_parser.add_argument("--residence-time", type=float, default=0.5)
    simulate_parser.add_argument("--hydrogen-concentration", type=float, default=50.0)
    simulate_parser.add_argument("--diameter", type=float, default=50e-6)
    simulate_parser.add_argument("--initial-temperature", type=float, default=300.0)
    simulate_parser.add_argument("--initial-oxide-fraction", type=float, default=0.9)
    simulate_parser.add_argument("--solver-steps", type=int, default=1_000)
    simulate_parser.add_argument(
        "--plot", action="store_true", help="save temperature and oxide-fraction plots"
    )
    simulate_parser.add_argument(
        "--output-directory", type=Path, default=Path("artifacts/simulation")
    )
    simulate_parser.set_defaults(handler=_run_simulation)

    train_parser = subparsers.add_parser("train", help="generate data and train a surrogate")
    _add_training_arguments(train_parser)
    train_parser.add_argument(
        "--output-directory", type=Path, default=Path("artifacts/surrogate-training")
    )
    train_parser.set_defaults(handler=_run_training)

    optimize_parser = subparsers.add_parser(
        "optimize", help="recommend and validate operating conditions"
    )
    optimize_parser.add_argument("--model-directory", type=Path, required=True)
    _add_optimization_arguments(optimize_parser)
    optimize_parser.add_argument(
        "--output-directory", type=Path, default=Path("artifacts/optimization")
    )
    optimize_parser.set_defaults(handler=_run_optimization)

    demo_parser = subparsers.add_parser("run-demo", help="run the complete core workflow")
    _add_training_arguments(demo_parser)
    demo_parser.add_argument("--candidate-count", type=int, default=4_096)
    demo_parser.add_argument("--validation-count", type=int, default=16)
    demo_parser.add_argument("--max-final-oxide-fraction", type=float, default=0.75)
    demo_parser.add_argument(
        "--output-directory", type=Path, default=Path("artifacts/demo")
    )
    demo_parser.set_defaults(handler=_run_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a CLI command and return its process exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.exit(2, f"reactor-lab: error: {error}\n")
    return 2
