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
uv run basedpyright .
uv run python examples/plot_trajectories.py
```

Use `make test`, `make typecheck`, `make example`, or `make` for all three; these targets wrap the uv commands above. Run `make pdf` to rebuild the printable model summary with Pandoc and Chromium.

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

## Stage 2: synthetic data and surrogate

Run the complete data-and-surrogate workflow with:

```bash
uv run python examples/train_surrogate.py
# or
make surrogate
```

The example generates 896 synthetic simulations using a fixed seed. It samples peak temperature, residence time, hydrogen concentration, particle diameter, and initial oxide fraction. Initial particle temperature is fixed at 300 K. In-distribution peak temperatures span 1,800–2,800 K; the separated operating regime spans 2,800–3,400 K and is excluded from training and validation by construction. All sampling bounds, fixed physics settings, split assignments, and feature ordering are saved with the dataset.

The two-output Equinox MLP predicts final oxide fraction and maximum particle temperature. Optax Adam trains on targets normalized using training-set statistics only. The `model.eqx` checkpoint contains both network parameters and normalization arrays; `config.json` records the architecture, seed, and exact feature and target ordering. See [ADR 0001](docs/adr/0001-use-equinox-and-optax.md) for the framework decision.

Outputs are written to `artifacts/stage2/`:

- `dataset.npz` and `dataset.json`: synthetic arrays, split labels, sampling bounds, and run settings.
- `model.eqx` and `config.json`: reloadable surrogate and training configuration.
- `history.json` and `metrics.json`: normalized-MSE history and per-target test metrics.
- `predictions.png`, `residuals.png`, and `training-history.png`: diagnostic plots.

The example evaluates the reloaded checkpoint. With the checked-in defaults, the synthetic results are:

| Regime | Target | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| In-distribution test | Final oxide fraction | 0.01156 | 0.01414 | 0.9817 |
| In-distribution test | Maximum temperature | 37.99 K | 52.59 K | 0.7510 |
| Separated regime | Final oxide fraction | 0.01784 | 0.02247 | 0.9541 |
| Separated regime | Maximum temperature | 53.90 K | 73.77 K | -0.0483 |

The **in-distribution test** contains unseen samples drawn from the same input ranges used for training. The **separated regime** contains samples from higher peak temperatures—2,800–3,400 K versus 1,800–2,800 K for training—and therefore tests extrapolation.

In the formulas below, $y_i$ is the simulator output, $\hat y_i$ is the surrogate prediction, $\bar y$ is the mean simulator output for the evaluated dataset, and $n$ is the number of samples.

#### MAE: mean absolute error

$$
\mathrm{MAE}=\frac{1}{n}\sum_i \left|\hat y_i-y_i\right|
$$

MAE is the average absolute prediction error, expressed in the target's units. An oxide-fraction MAE of **0.01156** means an average absolute error of about **0.012 oxide-fraction units**, or **1.16 percentage points**. A temperature MAE of **37.99 K** means an average absolute error of about **38 K**. MAE is directly interpretable but does not strongly penalize occasional large errors.

#### RMSE: root mean squared error

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(\hat y_i-y_i)^2}
$$

RMSE also uses the target's units, but squaring the errors gives greater weight to large errors. RMSE is always greater than or equal to MAE for a given set of errors; a substantially larger RMSE suggests that some predictions have much larger errors than typical. For example, in-distribution temperature MAE is **37.99 K**, while RMSE is **52.59 K**, showing that less frequent, larger errors contribute significantly to overall error.

#### R²: coefficient of determination

$$
R^2=1-\frac{\sum_i(\hat y_i-y_i)^2}{\sum_i(\bar y-y_i)^2}
$$

R² compares the surrogate's squared error with a baseline that always predicts the evaluated dataset's mean:

- **R² = 1:** perfect predictions.
- **R² = 0:** the same total squared error as predicting the mean.
- **R² < 0:** worse than predicting the mean.
- **R² = 0.9817:** squared error is reduced by about **98.17%** relative to the mean-prediction baseline. It does not mean that 98.17% of predictions are correct.

R² values from different datasets are not perfectly like-for-like because each dataset has its own mean and variance. MAE and RMSE provide complementary error magnitudes in physical units, while the residual plots show how errors vary across the operating regimes.

Oxide-fraction predictions are relatively accurate in both regimes, although MAE increases from **0.01156** in-distribution to **0.01784** in the separated regime. Maximum-temperature predictions have moderate in-distribution performance ($R^2=0.7510$). In the separated regime, maximum-temperature $R^2=-0.0483$ means the surrogate performs slightly worse than predicting that regime's mean temperature, so it has not generalized reliably to higher peak temperatures.

These metrics measure agreement with the synthetic toy simulator, not accuracy against experimental reactor data. Interpolation accuracy does not establish extrapolation reliability.

## Current status

Stages 1 and 2 are complete. Optimization and the CLI have not started.

See [the MVP plan](docs/mvp-plan.md) for the detailed model scope, validation strategy, and definition of done.
