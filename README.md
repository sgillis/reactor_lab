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

## Current status

The scope and implementation plan are documented. Implementation has not started.

See [the MVP plan](docs/mvp-plan.md) for the detailed model scope, schedule, validation strategy, and definition of done.
