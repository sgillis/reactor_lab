# Differentiable Reactor Lab

Differentiable Reactor Lab is an independent JAX portfolio project at the intersection of physics, machine learning, and production software engineering. It is inspired by the simulation-and-learning workflows used in industrial high-temperature processing.

The project will use a deliberately simplified model of metal-oxide particles passing through a hot hydrogen-gas reactor. The scientific goal is not to reproduce plasma-reactor physics quantitatively. The engineering goal is to turn a reduced-order physical model into a tested, reproducible, usable simulation-and-optimization system.

## Intended workflow

The MVP should demonstrate this complete loop:

1. Configure a reactor and particle population.
2. Simulate particle heating and oxide reduction with JAX.
3. Generate a synthetic dataset from the simulator.
4. Train a fast surrogate for selected simulation outputs.
5. Search for operating conditions that satisfy quality constraints.
6. Validate the recommended conditions with the physics simulator.

All data will be synthetic. Peak gas-temperature setpoint may be used as an explicitly labelled energy proxy; it must not be presented as a physical energy-consumption calculation.

## MVP deliverables

- A reduced-order particle simulator with documented assumptions and units.
- Batched simulation using JAX transformations.
- A small surrogate model with reproducible training and held-out evaluation.
- Constrained process-condition recommendations validated against the simulator.
- A command-line interface for simulation, training, optimization, and an end-to-end demo.
- Automated tests, trajectory and surrogate diagnostic plots, and a short performance benchmark.
- Setup instructions and one reproducible demo command.

A small FastAPI service, surrogate ensembles, and uncertainty-based experiment selection are stretch goals rather than requirements for the initial MVP.

## Non-goals for the MVP

- Quantitatively accurate plasma, reactor, or materials predictions.
- Plasma CFD, detailed plasma chemistry, atomization, or particle-size modelling.
- A frontend, cloud deployment, Kubernetes, or a large MLOps platform.
- Claims based on real reactor data.

## Stage 1: physics foundation

The package provides typed simulation configuration, jitted single-particle simulation, vmapped batches, trajectory summaries, and diagnostic plots. See [the physics documentation](docs/physics.md) for equations, units, validation rules, numerical details, and limitations. A [printable model summary](docs/model-summary.pdf) is also available.

Use [uv](https://docs.astral.sh/uv/) for environment setup and all Python execution:

```bash
uv sync
uv run pytest
uv run python examples/plot_trajectories.py
```

Use `make test`, `make example`, or `make` for both; these targets wrap the uv commands above. Run `make pdf` to rebuild the printable model summary with Pandoc and Chromium.

The example writes `artifacts/stage1-trajectories.png` and prints final oxide fraction, maximum particle temperature, and melting-threshold results for three synthetic cases.

```python
from reactor_lab import ParticleConfig, ReactorConfig, SimulationConfig, simulate

config = SimulationConfig(
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
)
trajectory = simulate(config)
print(trajectory.final_oxide_fraction)
```

## Current status

Stage 1 physics foundation is complete. Data generation, surrogate training, optimization, and the CLI have not started.

See [the MVP plan](docs/mvp-plan.md) for the detailed model scope, validation strategy, and definition of done.
