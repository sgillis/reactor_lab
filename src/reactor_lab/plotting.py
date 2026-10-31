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
