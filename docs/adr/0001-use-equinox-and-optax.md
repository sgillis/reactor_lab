# ADR 0001: Use Equinox and Optax for the surrogate

- Status: Accepted
- Date: 2026-10-31

## Context

Stage 2 requires a small MLP that predicts final oxide fraction and maximum particle temperature from synthetic simulator outputs. Training and evaluation must be reproducible, the model must be saved with its configuration, and the implementation should remain compact enough for a reviewer to understand.

The simulator already uses JAX. The surrogate therefore needs a neural-network and optimization approach that works with JAX arrays and transformations without introducing unnecessary training infrastructure. The initial workload is a small, stateless, feed-forward model running locally on CPU; distributed training, complex model state, and integration with pretrained models are not current requirements.

## Decision

Use **Equinox** to define and serialize the neural network and **Optax** to construct the optimizer.

The model, parameters, and normalization statistics will be represented as JAX PyTrees. Training will use an explicit loop with JAX transformations and an Optax optimizer. Model configuration and feature ordering will be stored alongside the serialized model so that preprocessing and loaded predictions remain aligned.

## Rationale

### Equinox

- Represents models and parameters as PyTrees, which fits the existing JAX design and keeps model data visible rather than hiding it behind a framework-owned state container.
- Provides the required module and MLP abstractions without a separate training framework.
- Supports straightforward leaf serialization for the project's simple checkpoint requirements.
- Keeps the training loop explicit, making initialization, normalization, optimization, and evaluation easier to inspect and test.

### Optax

- Provides standard, composable gradient transformations, including Adam, so the project does not need to implement and validate optimizer mathematics itself.
- Keeps optimizer state in explicit values that fit naturally into a JAX training step.
- Does not impose a model abstraction, data pipeline, or checkpointing system.
- Can be replaced or extended independently of the Equinox model definition.

Together, the libraries cover the required model and optimizer functionality while leaving project-specific orchestration in packaged Python code.

## Alternatives considered

### Flax with Optax

Flax is the strongest alternative and provides a richer neural-module ecosystem. It would be attractive if the project expected to reuse Flax models, manage substantial mutable model state, or adopt its surrounding checkpointing and training tooling.

For the current stateless MLP, its module and parameter-state abstractions add concepts without enabling a required capability. Choosing Equinox keeps the initial implementation smaller. This is a project-fit decision, not a claim that Flax is less capable.

### Raw JAX with a hand-written MLP and optimizer

This would minimize dependencies and provide complete control. It would also require the project to implement parameter initialization, optimizer updates, and serialization conventions that established libraries already provide. The additional code and validation burden outweigh the dependency savings for this MVP.

Optax could still be used with a hand-written JAX model, but Equinox provides a useful module and serialization boundary at limited additional complexity.

### PyTorch

PyTorch offers a broad model and training ecosystem, but adopting it would introduce a second array, automatic-differentiation, and device runtime alongside JAX. Data would cross framework boundaries, and the project would need to maintain two sets of numerical and reproducibility conventions. That cost is not justified for a small surrogate in an otherwise JAX-based workflow.

### Scikit-learn MLP

Scikit-learn could train a simple regressor with little neural-network-specific code. However, it would not preserve a JAX-native model, would offer less control over the training and prediction path, and would make later JAX-based experimentation harder. It is a reasonable fallback if JAX ceases to be a project requirement.

## Tradeoffs

### Accepted costs

- Equinox and Optax become additional runtime dependencies whose compatibility with the selected JAX version must be maintained.
- The project owns the training loop, metric collection, checkpoint orchestration, and reproducibility controls rather than receiving them from a high-level training framework.
- Equinox serialization is a lightweight model-artifact mechanism, not a complete training-checkpoint manager. Resume-at-step, optimizer-state retention, rotation, and distributed checkpoints would require additional work.
- Equinox's filtered transformations and PyTree-based model style may be unfamiliar to contributors who primarily use other neural-network frameworks.
- The approach does not provide the broader pretrained-model and framework-specific tooling available in larger ecosystems.

### Avoided costs

- No second numerical framework or tensor-conversion boundary.
- No hand-written optimizer or neural-network initialization machinery.
- No high-level training framework for a workflow that currently needs only a small MLP and an explicit loop.
- No distributed-training or advanced checkpointing infrastructure before there is a requirement for it.

The decision optimizes for local simplicity and an inspectable JAX workflow, not for the largest feature set or ecosystem.

## Assumptions and reversal triggers

Revisit this decision if any of the following assumptions stops holding:

- **The surrogate remains small and stateless.** Substantially more complex architectures or mutable model state may justify Flax or another higher-level framework.
- **Training remains local and single-device.** Distributed training, sophisticated checkpoint recovery, or large-scale experiment orchestration may justify a broader training stack.
- **JAX remains the project's numerical backend.** A move to another backend would remove a major reason for selecting Equinox and Optax.
- **Simple artifact persistence is sufficient.** Requirements for portable inference artifacts, long-term cross-version compatibility, or complete resumable training state may require a different serialization or checkpointing solution, even if Equinox remains the training library.
- **The dependencies remain maintained and compatible.** Breaking compatibility, unresolved defects, or weak maintenance affecting the pinned JAX version would justify reassessment.
- **Explicit training code remains an advantage.** If training variants and orchestration grow enough that maintaining the loop becomes the dominant cost, a higher-level framework may reduce rather than add complexity.

Any future reassessment should compare implementations against the same requirements: reproducibility, checkpoint round trips, prediction parity, testability, dependency compatibility, and the complexity of the resulting training code.
