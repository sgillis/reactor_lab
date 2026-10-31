"""Equinox surrogate training, prediction, and self-contained checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import TrainingConfig
from .data import SyntheticDataset


class Surrogate(eqx.Module):
    """Two-output MLP with preprocessing stored in the same PyTree."""

    network: eqx.nn.MLP
    feature_mean: jax.Array
    feature_scale: jax.Array
    target_mean: jax.Array
    target_scale: jax.Array
    config: TrainingConfig = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)
    target_names: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, features: jax.Array) -> jax.Array:
        """Predict physical-unit targets for one feature vector."""
        normalized_features = (features - self.feature_mean) / self.feature_scale
        normalized_targets = self.network(normalized_features)
        return normalized_targets * self.target_scale + self.target_mean


class TrainingResult(NamedTuple):
    """Trained model and normalized-MSE history for both training splits."""

    model: Surrogate
    history: dict[str, list[float]]


def _new_model(
    features: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> Surrogate:
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    target_mean = np.mean(targets, axis=0)
    target_scale = np.std(targets, axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    target_scale[target_scale < 1e-8] = 1.0
    return Surrogate(
        network=eqx.nn.MLP(
            in_size=features.shape[1],
            out_size=targets.shape[1],
            width_size=config.hidden_width,
            depth=config.hidden_depth,
            key=jax.random.PRNGKey(config.seed),
        ),
        feature_mean=jnp.asarray(feature_mean),
        feature_scale=jnp.asarray(feature_scale),
        target_mean=jnp.asarray(target_mean),
        target_scale=jnp.asarray(target_scale),
        config=config,
        feature_names=feature_names,
        target_names=target_names,
    )


def _normalized_loss(
    model: Surrogate, features: jax.Array, targets: jax.Array
) -> jax.Array:
    normalized_features = (features - model.feature_mean) / model.feature_scale
    normalized_targets = (targets - model.target_mean) / model.target_scale
    predictions = jax.vmap(model.network)(normalized_features)
    return jnp.mean(jnp.square(predictions - normalized_targets))


@eqx.filter_jit
def predict(model: Surrogate, features: np.ndarray | jax.Array) -> jax.Array:
    """Predict a batch of physical-unit targets."""
    return jax.vmap(model)(jnp.asarray(features))


def train_surrogate(dataset: SyntheticDataset, config: TrainingConfig) -> TrainingResult:
    """Train on the train split and record validation loss after every epoch."""
    if not isinstance(dataset, SyntheticDataset):
        raise TypeError("dataset must be a SyntheticDataset")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")

    train = dataset.get_split("train")
    validation = dataset.get_split("validation")
    model = _new_model(
        train.features,
        train.targets,
        config,
        tuple(dataset.metadata["feature_names"]),
        tuple(dataset.metadata["target_names"]),
    )
    train_features = jnp.asarray(train.features)
    train_targets = jnp.asarray(train.targets)
    validation_features = jnp.asarray(validation.features)
    validation_targets = jnp.asarray(validation.targets)

    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, optimizer_state, features, targets):
        loss, gradients = eqx.filter_value_and_grad(_normalized_loss)(
            model, features, targets
        )
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
        model = eqx.apply_updates(model, updates)
        return model, optimizer_state, loss

    @eqx.filter_jit
    def dataset_loss(model, features, targets):
        return _normalized_loss(model, features, targets)

    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    sample_count = train_features.shape[0]
    for _ in range(config.epochs):
        order = rng.permutation(sample_count)
        for start in range(0, sample_count, config.batch_size):
            indices = order[start : start + config.batch_size]
            model, optimizer_state, _ = step(
                model,
                optimizer_state,
                train_features[indices],
                train_targets[indices],
            )
        history["train_loss"].append(
            float(dataset_loss(model, train_features, train_targets))
        )
        history["validation_loss"].append(
            float(dataset_loss(model, validation_features, validation_targets))
        )

    return TrainingResult(model, history)


def save_surrogate(model: Surrogate, output_directory: str | Path) -> Path:
    """Serialize model arrays and normalization together, with JSON configuration."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "model.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    metadata = {
        "schema_version": 1,
        "feature_names": list(model.feature_names),
        "target_names": list(model.target_names),
        "training_config": asdict(model.config),
    }
    (output_directory / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path


def load_surrogate(input_directory: str | Path) -> Surrogate:
    """Reconstruct a model skeleton, then restore parameters and normalizers."""
    input_directory = Path(input_directory)
    metadata = json.loads((input_directory / "config.json").read_text(encoding="utf-8"))
    config = TrainingConfig(**metadata["training_config"])
    feature_names = tuple(metadata["feature_names"])
    target_names = tuple(metadata["target_names"])
    like_model = Surrogate(
        network=eqx.nn.MLP(
            in_size=len(feature_names),
            out_size=len(target_names),
            width_size=config.hidden_width,
            depth=config.hidden_depth,
            key=jax.random.PRNGKey(config.seed),
        ),
        feature_mean=jnp.zeros(len(feature_names)),
        feature_scale=jnp.ones(len(feature_names)),
        target_mean=jnp.zeros(len(target_names)),
        target_scale=jnp.ones(len(target_names)),
        config=config,
        feature_names=feature_names,
        target_names=target_names,
    )
    return eqx.tree_deserialise_leaves(input_directory / "model.eqx", like_model)
