# MVP plan

## Outcome

Build a compact, end-to-end system that demonstrates how a reduced-order physical model can become reliable scientific software:

> configure reactor -> simulate particles -> train surrogate -> recommend operating conditions -> validate with simulator

The MVP should prioritize a working, understandable workflow over physical complexity. It will run on CPU with JAX and use only synthetic data.

## Scope and constraints

- Use a documented toy model, with no claim of quantitative plasma-reactor accuracy.
- Keep the primary workflow in packaged Python code rather than notebooks.
- Provide a CLI before considering an API or graphical interface.
- Make runs reproducible through typed configuration, fixed random seeds, and saved run settings.
- Treat FastAPI, model ensembles, and active-learning candidate selection as stretch goals.
- Defer cloud infrastructure and frontend work.

## Physical model

The simulator will follow a particle through a prescribed reactor temperature profile.

### Inputs

- Peak gas temperature
- Residence time
- Hydrogen concentration
- Particle diameter
- Initial oxide fraction
- Material and numerical parameters needed by the reduced-order model

### State and outputs

The state will contain particle temperature and remaining oxide fraction. The simulator will return:

- Temperature and oxide-fraction trajectories
- Final oxide fraction
- Maximum particle temperature
- Whether the particle reaches a specified melting temperature

### Included physics

- Convective heat transfer between gas and particle
- Radiative heat exchange with the surroundings
- Simplified Arrhenius-style oxide reduction
- A prescribed gas-temperature profile

The equations, parameter sources or illustrative nature, units, valid ranges, and limitations must be documented alongside the implementation. The model will not initially include detailed plasma chemistry, particle drag, atomization, or a coupled reactor flow field.

## Implementation workstreams

### 1. Physics foundation

- Establish typed configuration and input validation.
- Implement the particle simulator and trajectory output.
- Add batched execution with `jit` and `vmap`.
- Inspect representative trajectories for physical plausibility.
- Add correctness tests for important limiting cases.
- Benchmark individual and batched execution.

Diffrax is the preferred integration option if it remains simple. A small fixed-step solver is an acceptable MVP alternative; solver sophistication is not a project objective.

### 2. Synthetic data and surrogate

- Sample reactor configurations reproducibly.
- Generate and save a synthetic dataset with the simulator.
- Train a small MLP using Equinox or Flax and Optax.
- Predict final oxide fraction and maximum particle temperature.
- Save the trained model together with its configuration.
- Evaluate on held-out data and on an explicitly separated operating regime.
- Produce predicted-versus-simulated and training diagnostic plots.

An ensemble of small models may be added if the core workflow is complete and reliable.

### 3. Process-condition optimization

- Generate a batch of candidate operating conditions.
- Evaluate candidates with the surrogate.
- Minimize peak gas-temperature setpoint as a labelled energy proxy.
- Require the final oxide target and melting-temperature condition to be met.
- Validate selected candidates using the physics simulator.
- Report both surrogate predictions and simulator validation results.

If an ensemble is available, optionally rank informative candidates using predictive uncertainty as a first active-learning demonstration.

### 4. Interface and reproducibility

Provide CLI commands equivalent to:

```text
simulate
train
optimize
run-demo
```

The `run-demo` command should execute the complete core workflow with a small configuration and save its outputs. Notebooks may help visualize results but must not be required to use the system.

If time remains, add a minimal FastAPI service with health, simulation, and optimization endpoints.

### 5. Documentation and presentation

Document:

- Setup and demo commands
- Model equations, assumptions, units, and limitations
- Architecture and data flow
- Surrogate evaluation, including out-of-distribution behaviour
- Optimization results and simulator validation
- Benchmark methodology and results
- Reproducibility controls and remaining limitations

## Validation strategy

At minimum, test that:

- Zero reaction rate leaves the oxide fraction unchanged.
- Thermal equilibrium gives zero initial temperature change.
- Increasing the reaction rate increases reduction under otherwise identical conditions.
- Batched and individual simulations produce matching results.
- Typical valid inputs produce finite trajectories and outputs.
- Invalid configurations are rejected clearly.
- Surrogate recommendations are checked against the simulator rather than accepted solely from surrogate predictions.

Evaluation must distinguish interpolation within the training distribution from extrapolation to unseen operating conditions. Synthetic results must always be identified as synthetic.

## Implementation stages

### Stage 1: physics foundation

Implement configuration, the simulator, trajectory output, core tests, and initial plots. Add batched execution and inspect representative results.

### Stage 2: data and surrogate

Generate the dataset, train and save the surrogate, evaluate held-out performance, and create diagnostic plots.

### Stage 3: optimization and CLI

Implement candidate search, constrained selection, simulator validation, and the core CLI commands. Add ensemble uncertainty or active-learning selection only after the core workflow is complete.

### Stage 4: hardening and documentation

Finish tests and reproducibility checks, add an optional minimal API, record benchmarks, and ensure a fresh setup can run the demo and understand the results.

## Definition of done

The MVP is complete when a reviewer can:

1. Install and run the project locally.
2. Run one command that demonstrates the complete core workflow.
3. Inspect physically plausible particle trajectories.
4. Review honest surrogate accuracy and out-of-distribution limitations.
5. Obtain a constrained operating-condition recommendation and its simulator validation.
6. Run the automated tests successfully.
7. Reproduce reported results from saved configuration and fixed seeds.
8. Understand the architecture, assumptions, and limitations without reading every source file.

## Deferred until after the MVP

Potential follow-up work includes:

- Uncertainty quantification and a fuller active-learning loop
- Gradient-based or multi-objective process optimization
- Bayesian parameter calibration with NumPyro
- Richer heat-transfer, reaction, or particle-population physics
- A run-comparison interface
- Expanded API, deployment, and model-monitoring capabilities

## Progress

- [x] Define project objective, scope, and non-goals
- [x] Document the MVP plan
- [x] Implement and validate the physics simulator
- [x] Generate synthetic data and train the surrogate
- [ ] Implement optimization and simulator validation
- [ ] Add the CLI and reproducible end-to-end demo
- [ ] Complete tests, benchmarks, CI, and usage documentation
