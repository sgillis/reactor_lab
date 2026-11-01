# Reactor Lab

Reactor Lab is an independent JAX portfolio project at the intersection of physics, machine learning, and production software engineering. It is inspired by the simulation-and-learning workflows used in industrial high-temperature processing.

The project uses a deliberately simplified model of metal-oxide particles passing through a hot hydrogen-gas reactor. The scientific goal is not to reproduce plasma-reactor physics quantitatively. The engineering goal is to turn a reduced-order physical model into a tested, reproducible, usable simulation-and-optimization system.

## Workflow

The project demonstrates this complete loop:

1. Configure a reactor and particle population.
2. Simulate particle heating and oxide reduction with JAX.
3. Generate a synthetic dataset from the simulator.
4. Train a fast surrogate for selected simulation outputs.
5. Search for operating conditions that satisfy quality constraints.
6. Validate the recommended conditions with the physics simulator.

All data are synthetic. Peak gas-temperature setpoint is used as an explicitly labelled energy proxy; it is not a physical energy-consumption calculation.

## Quickstart

Install [uv](https://docs.astral.sh/uv/), then run these commands from a fresh checkout:

```bash
uv sync --all-groups
uv run reactor-lab run-demo
uv run pytest
uv run basedpyright .
```

`uv sync --all-groups` installs the locked runtime and development dependencies. Use `uv sync` instead if only the CLI is needed. The demo writes the complete synthetic workflow to `artifacts/demo/`; the tests and type checker do not require a GPU or external data.

## MVP deliverables

- A reduced-order particle simulator with documented assumptions and units.
- Batched simulation using JAX transformations.
- A small surrogate model with reproducible training and held-out evaluation.
- Constrained process-condition recommendations validated against the simulator.
- A command-line interface for simulation, training, optimization, and an end-to-end demo.
- Automated tests and trajectory and surrogate diagnostic plots.
- Setup instructions and one reproducible demo command.

A small FastAPI service, surrogate ensembles, and uncertainty-based experiment selection are stretch goals rather than requirements for the initial MVP.

## Non-goals for the MVP

- Quantitatively accurate plasma, reactor, or materials predictions.
- Plasma CFD, detailed plasma chemistry, atomization, or particle-size modelling.
- A frontend, cloud deployment, Kubernetes, or a large MLOps platform.
- Claims based on real reactor data.

## Stage 1: physics foundation

The package provides typed simulation configuration, jitted single-particle simulation, vmapped batches, trajectory summaries, and diagnostic plots. See [the physics documentation](docs/physics.md) for equations, units, validation rules, numerical details, and limitations. A [printable model summary](docs/model-summary.pdf) is also available.

Run one simulation with a diagnostic plot:

```bash
uv run reactor-lab simulate --plot
# or
make example
```

Use `make test`, `make typecheck`, or `make demo` for the corresponding uv workflows; `make` runs tests, type checking, and the simulation example. Run `make pdf` to rebuild the printable model summary with Pandoc and Chromium.

The command writes `artifacts/simulation/trajectories.png`, trajectory arrays, and JSON settings and results for one synthetic case. It also prints final oxide fraction, maximum particle temperature, and the melting-threshold result.

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
uv run reactor-lab train --train-samples 512 --epochs 300
# or
make surrogate
```

The command generates 896 synthetic simulations using a fixed seed. It samples peak temperature, residence time, hydrogen concentration, particle diameter, and initial oxide fraction. Initial particle temperature is fixed at 300 K. In-distribution peak temperatures span 1,800–2,800 K; the separated operating regime spans 2,800–3,400 K and is excluded from training and validation by construction. All sampling bounds, fixed physics settings, split assignments, and feature ordering are saved with the dataset.

The two-output Equinox MLP predicts final oxide fraction and maximum particle temperature. Optax Adam trains on targets normalized using training-set statistics only. The `model.eqx` checkpoint contains both network parameters and normalization arrays; `config.json` records the architecture, seed, and exact feature and target ordering. See [ADR 0001](docs/adr/0001-use-equinox-and-optax.md) for the framework decision.

Outputs are written to `artifacts/surrogate-training/`:

- `dataset.npz` and `dataset.json`: synthetic arrays, split labels, sampling bounds, and run settings.
- `model.eqx` and `config.json`: reloadable surrogate and training configuration.
- `history.json` and `metrics.json`: normalized-MSE history and per-target test metrics.
- `predictions.png`, `residuals.png`, and `training-history.png`: diagnostic plots.

The command evaluates the reloaded checkpoint. With the command settings above, the synthetic results are:

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

## Stage 3: optimization and CLI

Install the local command with `uv sync`, then run the core workflows:

```bash
uv run reactor-lab simulate --plot
uv run reactor-lab train
uv run reactor-lab optimize --model-directory artifacts/surrogate-training
uv run reactor-lab run-demo
# or run the complete demo with:
make demo
```

`simulate` writes its typed configuration, scalar summary, and trajectory arrays to `artifacts/simulation/`; `--plot` also writes `trajectories.png` with particle and gas temperatures plus the oxide-fraction trajectory. `train` generates synthetic data, trains and reloads a checkpoint, evaluates both test regimes, and writes the model, metrics, history, and diagnostic plots to `artifacts/surrogate-training/`. Command options and smaller sample, epoch, and solver settings are available through `uv run reactor-lab COMMAND --help`.

`optimize` samples candidates reproducibly and predicts their final oxide fraction and maximum particle temperature with the surrogate. It rejects predictions outside the configured constraints, ranks feasible candidates by peak gas-temperature setpoint as an explicitly labelled **energy proxy**, and validates the leading candidates with the physics simulator. A recommendation must pass both surrogate screening and simulator validation. If no candidate passes, the command reports that outcome without relaxing the constraints.

The optimization report records the complete configuration and seed, model checkpoint location, surrogate-feasible candidate count, and surrogate predictions alongside simulator validation for every evaluated leading candidate. Simulator agreement still describes only the synthetic toy model, not experimental reactor accuracy.

`run-demo` executes data generation, training, checkpoint reload, evaluation, optimization, and simulator validation in `artifacts/demo/`. It saves `run-config.json` with the fixed sampling, training, and optimization settings alongside all resulting artifacts. `python -m reactor_lab` provides the same interface as `reactor-lab`.

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Immutable typed configuration, units, sampling bounds, seeds, and input validation |
| `simulator.py` | Fixed-step particle dynamics and jitted/vmapped execution |
| `data.py` | Seeded synthetic sampling, simulator-generated targets, split metadata, and dataset persistence |
| `surrogate.py` | Equinox MLP training, normalization, prediction, and self-contained checkpoints |
| `evaluation.py`, `plotting.py` | Held-out and separated-regime metrics plus diagnostic plots |
| `optimization.py` | Surrogate screening, energy-proxy ranking, and physics-simulator validation |
| `cli.py` | Console workflows, artifact writing, and saved-demo replay |

## Stage 4: reproducibility, CI, and results

Replay a demo directly from its saved typed configuration:

```bash
uv run reactor-lab run-demo \
  --config artifacts/demo/run-config.json \
  --output-directory artifacts/replayed-demo
```

With `--config`, the file supplies all sampling, training, and optimization settings, including model architecture, physical parameters, solver settings, and seeds. Corresponding command-line options are not used; `--output-directory` still selects the destination. The replay writes a fresh `run-config.json`, so subsequent runs can be replayed in the same way.

Fixed seeds and `uv.lock` make the workflow reproducible on the same dependency and hardware/software stack. The automated checks compare repeated datasets, training history, metrics, reloaded-model predictions, and optimization results. Bitwise-identical floating-point results across different JAX versions, operating systems, or hardware are not guaranteed.

In `optimization.json`, `recommendation` is the lowest-temperature candidate that passed both surrogate screening and simulator validation among the validated leading candidates; it is `null` when none passed. Each entry in `validated_candidates` records its input `features`, both surrogate predictions, simulator outputs, and `simulator_feasible`. Compare the surrogate and simulator values rather than treating the surrogate prediction as validation. `surrogate_feasible_count` reports screening outcomes before simulator validation. All results remain synthetic toy-model outputs.

GitHub Actions runs the locked test suite, static type checking, and a small installed-CLI demo on Python 3.11 and 3.12 for pull requests and pushes to `main`.

## Stage 5: local web and compute platform

The web platform wraps the same scientific workflows in a FastAPI control plane and asynchronous Dramatiq workers. PostgreSQL stores run and model metadata, Redis delivers jobs, and RustFS provides local S3-compatible artifact storage. The React frontend provides run submission, status monitoring, model lineage, artifact downloads, and interactive Plotly visualizations. See [ADR 0002](docs/adr/0002-adopt-production-platform-architecture.md) for the platform decisions and [ADR 0003](docs/adr/0003-use-rustfs-for-local-object-storage.md) for the local object-store selection.

Start the complete local stack with Docker and Docker Compose:

```bash
make platform
# or: docker compose up --build
```

Open `http://localhost:8080` for the application and `http://localhost:8000/docs` for the interactive OpenAPI documentation. The RustFS console is available at `http://localhost:9001`; the Compose development credentials are `minioadmin` / `minioadmin`. Host ports are bound to localhost. Do not expose this unauthenticated development configuration as a multi-user production deployment.

The API container applies Alembic migrations during startup, and the application creates the configured artifact bucket. PostgreSQL, Redis, and RustFS use named volumes. Stop the stack without deleting its data using:

```bash
make platform-down
```

The frontend uses TypeScript types generated from `openapi.json`. After changing the API, regenerate the schema and client types, then check the frontend build:

```bash
npm --prefix frontend ci
make frontend-generate
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

FastAPI submissions return `202 Accepted` and a run ID. Simulation, training, and optimization execute in the worker rather than in web-request processes. The frontend polls only active runs; results and artifacts remain available after completion.

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Immutable typed configuration, units, sampling bounds, seeds, and input validation |
| `simulator.py` | Fixed-step particle dynamics and jitted/vmapped execution |
| `data.py` | Seeded synthetic sampling, simulator-generated targets, split metadata, and dataset persistence |
| `surrogate.py` | Equinox MLP training, normalization, prediction, and self-contained checkpoints |
| `evaluation.py`, `plotting.py` | Held-out and separated-regime metrics plus diagnostic plots |
| `optimization.py` | Surrogate screening, energy-proxy ranking, and physics-simulator validation |
| `services.py` | Scientific workflow orchestration shared by CLI commands and asynchronous workers |
| `cli.py` | Console adapter for simulation, training, optimization, and saved-demo replay |
| `platform/` | FastAPI schemas and endpoints, SQLAlchemy records, Redis jobs, S3 storage, and worker dispatch |
| `alembic/` | Versioned PostgreSQL schema migrations |
| `frontend/` | React run/model interface, generated API types, and interactive numerical charts |
| `compose.yaml` | Local API, worker, frontend, PostgreSQL, Redis, and RustFS topology |

## Current status

The four scientific MVP stages and the initial local web platform are complete. The platform establishes production-shaped service boundaries; authentication, managed deployment, and cloud orchestration remain deployment concerns rather than local-demo features.

See [the MVP plan](docs/mvp-plan.md) for the detailed model scope, validation strategy, and definition of done.
