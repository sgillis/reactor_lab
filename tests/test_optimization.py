import json
from dataclasses import asdict, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactor_lab import Surrogate, TrainingConfig
from reactor_lab.config import (
    Bounds,
    InputBounds,
    MaterialConfig,
    OptimizationConfig,
    SolverConfig,
)
from reactor_lab.data import FEATURE_NAMES, TARGET_NAMES
from reactor_lab.optimization import optimize, save_optimization_result


def controlled_surrogate(
    final_oxide_fraction: float,
    maximum_temperature: float,
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    target_names: tuple[str, ...] = TARGET_NAMES,
) -> Surrogate:
    network = eqx.nn.MLP(
        in_size=len(FEATURE_NAMES),
        out_size=len(TARGET_NAMES),
        width_size=2,
        depth=0,
        key=jax.random.PRNGKey(0),
    )
    network = eqx.tree_at(
        lambda current: current.layers[-1].weight,
        network,
        jnp.zeros((len(TARGET_NAMES), len(FEATURE_NAMES))),
    )
    network = eqx.tree_at(
        lambda current: current.layers[-1].bias,
        network,
        jnp.asarray([final_oxide_fraction, maximum_temperature]),
    )
    return Surrogate(
        network=network,
        feature_mean=jnp.zeros(len(FEATURE_NAMES)),
        feature_scale=jnp.ones(len(FEATURE_NAMES)),
        target_mean=jnp.zeros(len(TARGET_NAMES)),
        target_scale=jnp.ones(len(TARGET_NAMES)),
        config=TrainingConfig(hidden_width=2, hidden_depth=1),
        feature_names=feature_names,
        target_names=target_names,
    )


def base_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(
        sampling_bounds=InputBounds(
            peak_temperature=Bounds(300.0, 600.0),
            residence_time=Bounds(0.1, 0.2),
            hydrogen_concentration=Bounds(20.0, 100.0),
            diameter=Bounds(30e-6, 50e-6),
            initial_oxide_fraction=Bounds(0.2, 0.3),
        ),
        material=MaterialConfig(
            melting_temperature=1_000.0,
            reaction_pre_exponential=10_000.0,
        ),
        solver=SolverConfig(num_steps=20),
        candidate_count=4,
        validation_count=2,
        max_final_oxide_fraction=0.125,
        seed=41,
    )


@pytest.mark.parametrize("initial_lower", [0.1, 0.125])
def test_initial_oxide_bounds_must_exceed_final_limit(initial_lower: float) -> None:
    base = base_optimization_config()
    sampling_bounds = replace(
        base.sampling_bounds,
        initial_oxide_fraction=Bounds(initial_lower, 0.6),
    )

    with pytest.raises(ValueError, match="max_final_oxide_fraction must be less"):
        replace(base, sampling_bounds=sampling_bounds)


def test_seeded_sampling_is_reproducible_and_within_bounds() -> None:
    config = replace(base_optimization_config(), candidate_count=6, validation_count=6)
    model = controlled_surrogate(0.0, 500.0)

    first = optimize(model, config)
    second = optimize(model, config)

    assert first == second
    assert len(first.validated_candidates) == config.candidate_count
    for candidate in first.validated_candidates:
        assert len(candidate.features) == len(FEATURE_NAMES)
        for name, value in zip(FEATURE_NAMES, candidate.features, strict=True):
            bounds = getattr(config.sampling_bounds, name)
            assert bounds.lower <= value < bounds.upper


def test_recommendation_has_lowest_peak_among_simulator_feasible_candidates() -> None:
    config = replace(base_optimization_config(), candidate_count=6, validation_count=4)

    result = optimize(controlled_surrogate(0.0, 500.0), config)

    assert result.surrogate_feasible_count == config.candidate_count
    assert len(result.validated_candidates) == config.validation_count
    peaks = [candidate.features[0] for candidate in result.validated_candidates]
    assert peaks == sorted(peaks)
    assert result.recommendation is result.validated_candidates[0]
    assert all(candidate.simulator_feasible for candidate in result.validated_candidates)


def test_oxide_limit_is_inclusive_and_surrogate_melting_limit_is_exclusive() -> None:
    config = base_optimization_config()

    at_oxide_limit = optimize(controlled_surrogate(0.125, 500.0), config)
    above_oxide_limit = optimize(controlled_surrogate(0.126, 500.0), config)
    at_melting_limit = optimize(controlled_surrogate(0.0, 1_000.0), config)

    assert at_oxide_limit.surrogate_feasible_count == config.candidate_count
    assert len(at_oxide_limit.validated_candidates) == config.validation_count
    assert above_oxide_limit.surrogate_feasible_count == 0
    assert above_oxide_limit.recommendation is None
    assert at_melting_limit.surrogate_feasible_count == 0
    assert at_melting_limit.validated_candidates == ()
    assert at_melting_limit.recommendation is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_surrogate_predictions_are_rejected(value: float) -> None:
    result = optimize(controlled_surrogate(value, 500.0), base_optimization_config())

    assert result.surrogate_feasible_count == 0
    assert result.validated_candidates == ()
    assert result.recommendation is None


def test_no_surrogate_feasible_candidates_returns_empty_result() -> None:
    result = optimize(controlled_surrogate(0.6, 500.0), base_optimization_config())

    assert result.surrogate_feasible_count == 0
    assert result.validated_candidates == ()
    assert result.recommendation is None


def test_real_simulator_failures_are_preserved_without_recommendation() -> None:
    base = base_optimization_config()
    config = replace(
        base,
        sampling_bounds=replace(
            base.sampling_bounds,
            initial_oxide_fraction=Bounds(0.9, 0.95),
        ),
        material=replace(base.material, reaction_pre_exponential=0.0),
        candidate_count=3,
        validation_count=2,
    )

    result = optimize(controlled_surrogate(0.0, 500.0), config)

    assert result.surrogate_feasible_count == config.candidate_count
    assert len(result.validated_candidates) == config.validation_count
    assert all(
        0.9 <= candidate.simulator_final_oxide_fraction < 0.95
        for candidate in result.validated_candidates
    )
    assert all(not candidate.simulator_feasible for candidate in result.validated_candidates)
    assert result.recommendation is None


@pytest.mark.parametrize(
    ("feature_names", "target_names", "match"),
    [
        (tuple(reversed(FEATURE_NAMES)), TARGET_NAMES, "feature_names"),
        (FEATURE_NAMES, tuple(reversed(TARGET_NAMES)), "target_names"),
    ],
)
def test_model_schema_must_match_exactly(
    feature_names: tuple[str, ...], target_names: tuple[str, ...], match: str
) -> None:
    model = controlled_surrogate(
        0.0,
        500.0,
        feature_names=feature_names,
        target_names=target_names,
    )

    with pytest.raises(ValueError, match=match):
        optimize(model, base_optimization_config())


def test_json_report_contains_complete_explicit_result(tmp_path) -> None:
    config = base_optimization_config()
    result = optimize(controlled_surrogate(0.0, 500.0), config)
    output_path = tmp_path / "reports" / "optimization.json"
    checkpoint = tmp_path / "checkpoint" / "model.eqx"

    returned_path = save_optimization_result(result, output_path, checkpoint)

    assert returned_path == output_path
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["synthetic"] is True
    assert report["feature_names"] == list(FEATURE_NAMES)
    assert report["target_names"] == list(TARGET_NAMES)
    assert report["config"] == asdict(config)
    assert report["surrogate_feasible_count"] == config.candidate_count
    assert report["validated_candidate_count"] == config.validation_count
    assert len(report["validated_candidates"]) == config.validation_count
    assert report["recommendation"] == report["validated_candidates"][0]
    assert report["model_checkpoint"] == str(checkpoint)
    assert set(report["validated_candidates"][0]) == {
        "features",
        "surrogate_final_oxide_fraction",
        "surrogate_maximum_temperature",
        "simulator_final_oxide_fraction",
        "simulator_maximum_temperature",
        "simulator_reached_melting",
        "simulator_feasible",
    }
