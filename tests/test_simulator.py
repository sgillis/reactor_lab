from dataclasses import replace

import numpy as np
import pytest

from reactor_lab import (
    MaterialConfig,
    ParticleConfig,
    ReactorConfig,
    SimulationConfig,
    SolverConfig,
    simulate,
    simulate_batch,
)


def base_config() -> SimulationConfig:
    return SimulationConfig(
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
        solver=SolverConfig(num_steps=200),
    )


def test_zero_reaction_rate_leaves_oxide_unchanged() -> None:
    config = base_config()
    config = replace(
        config, material=replace(config.material, reaction_pre_exponential=0.0)
    )

    trajectory = simulate(config)

    np.testing.assert_array_equal(
        trajectory.oxide_fraction,
        np.full(
            config.solver.num_steps + 1,
            config.particle.initial_oxide_fraction,
            dtype=trajectory.oxide_fraction.dtype,
        ),
    )


def test_thermal_equilibrium_gives_zero_initial_temperature_change() -> None:
    config = base_config()
    config = replace(
        config,
        particle=replace(
            config.particle,
            initial_temperature=config.material.surroundings_temperature,
        ),
    )

    trajectory = simulate(config)

    assert trajectory.temperature[1] == config.particle.initial_temperature


def test_increasing_reaction_rate_increases_reduction() -> None:
    config = base_config()
    slow = simulate(
        replace(config, material=replace(config.material, reaction_pre_exponential=0.01))
    )
    fast = simulate(
        replace(config, material=replace(config.material, reaction_pre_exponential=0.1))
    )

    assert fast.final_oxide_fraction < slow.final_oxide_fraction


def test_batched_and_individual_simulations_match() -> None:
    base = base_config()
    configs = (
        base,
        replace(base, reactor=replace(base.reactor, peak_temperature=2_800.0)),
        replace(
            base,
            reactor=replace(
                base.reactor, residence_time=0.8, hydrogen_concentration=100.0
            ),
            particle=replace(base.particle, diameter=80e-6, initial_oxide_fraction=0.7),
        ),
    )

    batch = simulate_batch(configs)
    individuals = [simulate(config) for config in configs]

    for index, individual in enumerate(individuals):
        for field in ("time", "temperature", "oxide_fraction", "gas_temperature"):
            np.testing.assert_allclose(
                getattr(batch, field)[index], getattr(individual, field), rtol=1e-6, atol=1e-6
            )
        assert batch.reached_melting[index] == individual.reached_melting
        np.testing.assert_allclose(
            batch.final_oxide_fraction[index], individual.final_oxide_fraction
        )
        np.testing.assert_allclose(
            batch.maximum_temperature[index], individual.maximum_temperature
        )


def test_typical_inputs_produce_finite_bounded_trajectories() -> None:
    config = base_config()
    trajectory = simulate(config)

    for field in ("time", "temperature", "oxide_fraction", "gas_temperature"):
        values = np.asarray(getattr(trajectory, field))
        assert values.shape == (config.solver.num_steps + 1,)
        assert np.isfinite(values).all()
    assert np.all((trajectory.oxide_fraction >= 0) & (trajectory.oxide_fraction <= 1))
    assert np.all(np.diff(trajectory.oxide_fraction) <= 0)
    assert np.isfinite(trajectory.final_oxide_fraction)
    assert np.isfinite(trajectory.maximum_temperature)
    assert isinstance(bool(trajectory.reached_melting), bool)


def test_melting_output_includes_initial_and_later_temperatures() -> None:
    config = base_config()
    at_melting = replace(
        config,
        material=replace(
            config.material, melting_temperature=config.particle.initial_temperature
        ),
    )
    never_melting = replace(
        config, material=replace(config.material, melting_temperature=4_000.0)
    )

    assert bool(simulate(at_melting).reached_melting)
    assert not bool(simulate(never_melting).reached_melting)


@pytest.mark.parametrize(
    ("make_config", "match"),
    [
        (lambda: ReactorConfig(2_000.0, 0.0, 50.0), "residence_time"),
        (lambda: ReactorConfig(2_000.0, 0.5, -1.0), "hydrogen_concentration"),
        (lambda: ReactorConfig(float("nan"), 0.5, 50.0), "finite"),
        (lambda: ParticleConfig(0.0, 300.0, 0.9), "diameter"),
        (lambda: ParticleConfig(50e-6, 300.0, 1.1), "initial_oxide_fraction"),
        (lambda: MaterialConfig(emissivity=-0.1), "emissivity"),
        (lambda: MaterialConfig(activation_energy=-1.0), "activation_energy"),
        (lambda: SolverConfig(num_steps=1.5), "integer"),  # pyright: ignore[reportArgumentType]
        (lambda: SolverConfig(num_steps=0), "at least one"),
    ],
)
def test_invalid_parameters_are_rejected(make_config, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        make_config()


def test_peak_temperature_must_not_be_below_surroundings() -> None:
    config = base_config()
    with pytest.raises(ValueError, match="peak_temperature"):
        replace(
            config,
            reactor=replace(config.reactor, peak_temperature=200.0),
        )


def test_batch_rejects_empty_and_mixed_static_settings() -> None:
    config = base_config()
    other_material = replace(config, material=MaterialConfig(density=6_000.0))

    with pytest.raises(ValueError, match="must not be empty"):
        simulate_batch([])
    with pytest.raises(ValueError, match="share material and solver"):
        simulate_batch([config, other_material])
