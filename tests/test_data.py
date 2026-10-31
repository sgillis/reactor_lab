import numpy as np
import pytest

from reactor_lab import (
    Bounds,
    InputBounds,
    SamplingConfig,
    SolverConfig,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from reactor_lab.data import FEATURE_NAMES, SPLIT_NAMES, TARGET_NAMES


@pytest.fixture(scope="module")
def sampling_config() -> SamplingConfig:
    return SamplingConfig(
        train_samples=12,
        validation_samples=5,
        test_samples=6,
        ood_samples=7,
        solver=SolverConfig(num_steps=50),
        seed=41,
    )


@pytest.fixture(scope="module")
def dataset(sampling_config):
    return generate_dataset(sampling_config)


def test_seeded_generation_reproduces_arrays_and_metadata(dataset, sampling_config) -> None:
    repeated = generate_dataset(sampling_config)

    np.testing.assert_array_equal(repeated.features, dataset.features)
    np.testing.assert_array_equal(repeated.targets, dataset.targets)
    np.testing.assert_array_equal(repeated.split, dataset.split)
    assert repeated.metadata == dataset.metadata


def test_splits_have_finite_targets_and_respect_regime_bounds(dataset, sampling_config) -> None:
    assert dataset.features.shape == (30, len(FEATURE_NAMES))
    assert dataset.targets.shape == (30, len(TARGET_NAMES))
    assert set(dataset.split) == set(SPLIT_NAMES)
    assert np.isfinite(dataset.features).all()
    assert np.isfinite(dataset.targets).all()
    assert np.all((dataset.targets[:, 0] >= 0) & (dataset.targets[:, 0] <= 1))
    assert np.all(dataset.targets[:, 1] >= sampling_config.initial_temperature)

    for name in SPLIT_NAMES:
        split = dataset.get_split(name)
        bounds = sampling_config.ood_bounds if name == "ood" else sampling_config.sampling_bounds
        for index, feature_name in enumerate(FEATURE_NAMES):
            feature_bounds = getattr(bounds, feature_name)
            assert np.all(split.features[:, index] >= feature_bounds.lower)
            assert np.all(split.features[:, index] < feature_bounds.upper)

    assert np.all(dataset.get_split("ood").features[:, 0] >= 2_800.0)
    for name in ("train", "validation", "test"):
        assert np.all(dataset.get_split(name).features[:, 0] < 2_800.0)


def test_dataset_round_trip_preserves_arrays_and_metadata(dataset, tmp_path) -> None:
    path = save_dataset(dataset, tmp_path / "dataset.npz")
    loaded = load_dataset(path)

    np.testing.assert_array_equal(loaded.features, dataset.features)
    np.testing.assert_array_equal(loaded.targets, dataset.targets)
    np.testing.assert_array_equal(loaded.split, dataset.split)
    assert loaded.metadata == dataset.metadata


def test_sampling_rejects_overlapping_ood_regime() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        SamplingConfig(ood_bounds=InputBounds())


def test_training_config_rejects_invalid_epoch_count() -> None:
    from reactor_lab import TrainingConfig

    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0)


def test_bounds_reject_reversed_interval() -> None:
    with pytest.raises(ValueError, match="less than"):
        Bounds(2.0, 1.0)
