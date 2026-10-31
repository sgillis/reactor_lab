"""Fixed-step JAX implementation of the reduced-order particle model."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import MaterialConfig, SimulationConfig, SolverConfig

_GAS_CONSTANT = 8.314462618153239  # J / (mol K)
_STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 K^4)


class Trajectory(NamedTuple):
    """Trajectories and derived outputs, with time in seconds."""

    time: jax.Array
    temperature: jax.Array
    oxide_fraction: jax.Array
    gas_temperature: jax.Array
    reached_melting: jax.Array

    @property
    def final_oxide_fraction(self) -> jax.Array:
        return self.oxide_fraction[..., -1]

    @property
    def maximum_temperature(self) -> jax.Array:
        return jnp.max(self.temperature, axis=-1)


def _simulate_scalars(
    peak_temperature: float | jax.Array,
    residence_time: float | jax.Array,
    hydrogen_concentration: float | jax.Array,
    diameter: float | jax.Array,
    initial_temperature: float | jax.Array,
    initial_oxide_fraction: float | jax.Array,
    material: MaterialConfig,
    solver: SolverConfig,
) -> Trajectory:
    times = jnp.linspace(0.0, residence_time, solver.num_steps + 1)
    step_size = residence_time / solver.num_steps
    initial_temperature = jnp.asarray(initial_temperature)
    initial_oxide_fraction = jnp.asarray(initial_oxide_fraction)
    initial_state = (initial_temperature, initial_oxide_fraction)

    def gas_temperature(time: jax.Array) -> jax.Array:
        profile = jnp.sin(jnp.pi * time / residence_time)
        return material.surroundings_temperature + (
            peak_temperature - material.surroundings_temperature
        ) * profile

    def step(state: tuple[jax.Array, jax.Array], time: jax.Array):
        temperature, oxide_fraction = state
        gas = gas_temperature(time)
        heat_transfer = material.heat_transfer_coefficient * (gas - temperature)
        radiation = material.emissivity * _STEFAN_BOLTZMANN * (
            material.surroundings_temperature**4 - temperature**4
        )
        temperature_rate = 6.0 / (
            material.density * material.heat_capacity * diameter
        ) * (heat_transfer + radiation)
        next_temperature = temperature + step_size * temperature_rate

        reaction_rate = (
            material.reaction_pre_exponential
            * hydrogen_concentration
            * jnp.exp(-material.activation_energy / (_GAS_CONSTANT * temperature))
        )
        next_oxide_fraction = oxide_fraction * jnp.exp(-reaction_rate * step_size)
        return (next_temperature, next_oxide_fraction), (
            next_temperature,
            next_oxide_fraction,
        )

    _, (later_temperatures, later_oxide_fractions) = jax.lax.scan(
        step, initial_state, times[:-1]
    )
    temperatures = jnp.concatenate((initial_temperature[None], later_temperatures))
    oxide_fractions = jnp.concatenate(
        (initial_oxide_fraction[None], later_oxide_fractions)
    )
    reached_melting = jnp.any(
        temperatures >= material.melting_temperature, axis=-1
    )
    return Trajectory(
        times, temperatures, oxide_fractions, gas_temperature(times), reached_melting
    )


@partial(jax.jit, static_argnames=("config",))
def simulate(config: SimulationConfig) -> Trajectory:
    """Simulate one particle using a jitted fixed-step solver."""
    return _simulate_scalars(
        config.reactor.peak_temperature,
        config.reactor.residence_time,
        config.reactor.hydrogen_concentration,
        config.particle.diameter,
        config.particle.initial_temperature,
        config.particle.initial_oxide_fraction,
        config.material,
        config.solver,
    )


@partial(jax.jit, static_argnames=("material", "solver"))
def _simulate_batch(
    peak_temperatures: jax.Array,
    residence_times: jax.Array,
    hydrogen_concentrations: jax.Array,
    diameters: jax.Array,
    initial_temperatures: jax.Array,
    initial_oxide_fractions: jax.Array,
    material: MaterialConfig,
    solver: SolverConfig,
) -> Trajectory:
    return jax.vmap(_simulate_scalars, in_axes=(0, 0, 0, 0, 0, 0, None, None))(
        peak_temperatures,
        residence_times,
        hydrogen_concentrations,
        diameters,
        initial_temperatures,
        initial_oxide_fractions,
        material,
        solver,
    )


def simulate_batch(configs: Sequence[SimulationConfig]) -> Trajectory:
    """Simulate a non-empty batch with shared material and solver settings."""
    configs = tuple(configs)
    if not configs:
        raise ValueError("configs must not be empty")
    if not all(isinstance(config, SimulationConfig) for config in configs):
        raise TypeError("every batch entry must be a SimulationConfig")

    material = configs[0].material
    solver = configs[0].solver
    if any(config.material != material or config.solver != solver for config in configs):
        raise ValueError("batch entries must share material and solver settings")

    def values(attribute_path: str) -> jax.Array:
        group, name = attribute_path.split(".")
        return jnp.asarray([getattr(getattr(config, group), name) for config in configs])

    return _simulate_batch(
        values("reactor.peak_temperature"),
        values("reactor.residence_time"),
        values("reactor.hydrogen_concentration"),
        values("particle.diameter"),
        values("particle.initial_temperature"),
        values("particle.initial_oxide_fraction"),
        material,
        solver,
    )
