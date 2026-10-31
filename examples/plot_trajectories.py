"""Generate representative Stage 1 trajectories and a diagnostic plot."""

from dataclasses import replace

from reactor_lab import (
    ParticleConfig,
    ReactorConfig,
    SimulationConfig,
    SolverConfig,
    Trajectory,
    simulate_batch,
)
from reactor_lab.plotting import plot_trajectories


base = SimulationConfig(
    reactor=ReactorConfig(
        peak_temperature=2_200.0,
        residence_time=0.5,
        hydrogen_concentration=50.0,
    ),
    particle=ParticleConfig(
        diameter=50e-6,
        initial_temperature=300.0,
        initial_oxide_fraction=0.9,
    ),
    solver=SolverConfig(num_steps=1_000),
)
labelled_configs = {
    "2000 K peak": base,
    "2500 K peak": replace(base, reactor=replace(base.reactor, peak_temperature=2_500.0)),
    "3000 K peak": replace(base, reactor=replace(base.reactor, peak_temperature=3_000.0)),
}
batch = simulate_batch(tuple(labelled_configs.values()))
trajectories = {
    label: Trajectory(*(values[index] for values in batch))
    for index, label in enumerate(labelled_configs)
}

for label, trajectory in trajectories.items():
    print(
        f"{label}: final oxide={float(trajectory.final_oxide_fraction):.4f}, "
        f"maximum particle temperature={float(trajectory.maximum_temperature):.1f} K, "
        f"reached melting={bool(trajectory.reached_melting)}"
    )

output = plot_trajectories(trajectories, "artifacts/stage1-trajectories.png")
print(f"Saved synthetic trajectory plot to {output}")
