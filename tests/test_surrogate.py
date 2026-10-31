import numpy as np
import pytest

from reactor_lab import (
    SyntheticDataset,
    TrainingConfig,
    load_surrogate,
    predict,
    regression_metrics,
    save_surrogate,
    train_surrogate,
)
from reactor_lab.data import FEATURE_NAMES, TARGET_NAMES


def linear_dataset() -> SyntheticDataset:
    rng = np.random.default_rng(73)
    feature_splits = []
    target_splits = []
    labels = []
    for name, count in (("train", 96), ("validation", 32), ("test", 24), ("ood", 24)):
        features = rng.normal(size=(count, len(FEATURE_NAMES)))
        targets = np.column_stack(
            (
                0.5 + 0.05 * features[:, 0] - 0.03 * features[:, 1],
                1_500.0 + 100.0 * features[:, 0] - 50.0 * features[:, 1],
            )
        )
        feature_splits.append(features)
        target_splits.append(targets)
        labels.append(np.full(count, name))
    return SyntheticDataset(
        features=np.concatenate(feature_splits),
        targets=np.concatenate(target_splits),
        split=np.concatenate(labels),
        metadata={
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
        },
    )


@pytest.fixture(scope="module")
def trained_result():
    return train_surrogate(
        linear_dataset(),
        TrainingConfig(
            hidden_width=8,
            hidden_depth=1,
            learning_rate=0.01,
            batch_size=16,
            epochs=40,
            seed=7,
        ),
    )


def test_training_reduces_loss_and_produces_finite_predictions(trained_result) -> None:
    history = trained_result.history
    assert len(history["train_loss"]) == len(history["validation_loss"]) == 40
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["validation_loss"][-1] < history["validation_loss"][0]

    predictions = predict(trained_result.model, linear_dataset().get_split("test").features)
    assert predictions.shape == (24, len(TARGET_NAMES))
    assert np.isfinite(np.asarray(predictions)).all()


def test_checkpoint_round_trip_preserves_preprocessing_and_predictions(
    trained_result, tmp_path
) -> None:
    features = linear_dataset().get_split("test").features
    expected = predict(trained_result.model, features)
    save_surrogate(trained_result.model, tmp_path)
    loaded = load_surrogate(tmp_path)

    np.testing.assert_allclose(predict(loaded, features), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(loaded.feature_mean, trained_result.model.feature_mean)
    np.testing.assert_array_equal(loaded.target_scale, trained_result.model.target_scale)
    assert loaded.feature_names == FEATURE_NAMES
    assert loaded.target_names == TARGET_NAMES


def test_training_is_reproducible_for_a_fixed_seed() -> None:
    dataset = linear_dataset()
    config = TrainingConfig(hidden_width=4, hidden_depth=1, epochs=5, batch_size=32, seed=9)
    first = train_surrogate(dataset, config)
    second = train_surrogate(dataset, config)

    assert first.history == second.history
    features = dataset.get_split("test").features
    np.testing.assert_array_equal(predict(first.model, features), predict(second.model, features))


def test_regression_metrics_match_known_errors() -> None:
    metrics = regression_metrics(
        np.asarray([0.0, 2.0]), np.asarray([1.0, 0.0])
    )

    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["r2"] == pytest.approx(-1.5)
    assert regression_metrics(np.ones(2), np.ones(2))["r2"] is None
