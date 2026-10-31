"""Plotting helpers for inspecting simulated trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .simulator import Trajectory


def plot_trajectories(
    trajectories: Mapping[str, Trajectory], output_path: str | Path
) -> Path:
    """Save temperature and oxide-fraction plots for labelled trajectories."""
    if not trajectories:
        raise ValueError("trajectories must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (temperature_axis, oxide_axis) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True
    )

    for label, trajectory in trajectories.items():
        time = np.asarray(trajectory.time)
        particle_line = temperature_axis.plot(time, trajectory.temperature, label=label)[0]
        temperature_axis.plot(
            time,
            trajectory.gas_temperature,
            linestyle="--",
            color=particle_line.get_color(),
            alpha=0.55,
        )
        oxide_axis.plot(time, trajectory.oxide_fraction, color=particle_line.get_color())

    temperature_axis.set_ylabel("Temperature (K)")
    temperature_axis.legend()
    temperature_axis.grid(alpha=0.25)
    oxide_axis.set_xlabel("Residence time (s)")
    oxide_axis.set_ylabel("Oxide fraction (-)")
    oxide_axis.set_ylim(bottom=0.0)
    oxide_axis.grid(alpha=0.25)
    figure.suptitle("Synthetic reduced-order particle trajectories")
    figure.text(0.5, 0.01, "Solid: particle temperature; dashed: gas temperature", ha="center")
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _prepare_plot(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _target_axis_labels(index: int) -> tuple[str, str]:
    if index == 0:
        return "Oxide fraction (-)", "Residual (-)"
    return "Maximum temperature (K)", "Residual (K)"


def plot_surrogate_predictions(
    simulated: Mapping[str, np.ndarray],
    predicted: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Save per-target predicted-versus-simulated plots for named test regimes."""
    if not simulated or simulated.keys() != predicted.keys():
        raise ValueError("simulated and predicted must have the same non-empty regimes")
    output_path = _prepare_plot(output_path)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for index, axis in enumerate(axes):
        values = []
        for regime, actual_regime in simulated.items():
            actual = np.asarray(actual_regime)[:, index]
            prediction = np.asarray(predicted[regime])[:, index]
            axis.scatter(actual, prediction, s=16, alpha=0.65, label=regime)
            values.extend((actual, prediction))
        combined = np.concatenate([np.ravel(value) for value in values])
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        padding = max((upper - lower) * 0.05, 1e-8)
        axis.plot((lower - padding, upper + padding), (lower - padding, upper + padding), "k--")
        label, _ = _target_axis_labels(index)
        axis.set_xlabel(f"Simulated {label.lower()}")
        axis.set_ylabel(f"Predicted {label.lower()}")
        axis.grid(alpha=0.25)
        axis.legend(title="Regime")
    figure.suptitle("Synthetic surrogate: predicted versus simulated")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_surrogate_residuals(
    simulated: Mapping[str, np.ndarray],
    predicted: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Save per-target residual trends for named test regimes."""
    if not simulated or simulated.keys() != predicted.keys():
        raise ValueError("simulated and predicted must have the same non-empty regimes")
    output_path = _prepare_plot(output_path)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for index, axis in enumerate(axes):
        for regime, actual_regime in simulated.items():
            actual = np.asarray(actual_regime)[:, index]
            prediction = np.asarray(predicted[regime])[:, index]
            axis.scatter(actual, prediction - actual, s=16, alpha=0.65, label=regime)
        label, residual_label = _target_axis_labels(index)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel(f"Simulated {label.lower()}")
        axis.set_ylabel(residual_label)
        axis.grid(alpha=0.25)
        axis.legend(title="Regime")
    figure.suptitle("Synthetic surrogate residuals (predicted - simulated)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_training_history(
    history: Mapping[str, np.ndarray | list[float]], output_path: str | Path
) -> Path:
    """Save training and validation normalized-MSE curves."""
    output_path = _prepare_plot(output_path)
    figure, axis = plt.subplots(figsize=(7, 4.8))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axis.semilogy(epochs, history["train_loss"], label="Training")
    axis.semilogy(epochs, history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized mean squared error (-)")
    axis.set_title("Synthetic surrogate training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
