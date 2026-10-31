"""Validated, static configuration for particle simulations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(name: str, value: Real) -> float:
    value = _real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative(name: str, value: Real) -> float:
    value = _real(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _fraction(name: str, value: Real) -> float:
    value = _real(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


@dataclass(frozen=True, slots=True)
class ReactorConfig:
    """Reactor inputs; temperatures are in kelvin and time in seconds."""

    peak_temperature: Real
    residence_time: Real
    hydrogen_concentration: Real

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "peak_temperature", _positive("peak_temperature", self.peak_temperature)
        )
        object.__setattr__(
            self, "residence_time", _positive("residence_time", self.residence_time)
        )
        object.__setattr__(
            self,
            "hydrogen_concentration",
            _nonnegative("hydrogen_concentration", self.hydrogen_concentration),
        )


@dataclass(frozen=True, slots=True)
class ParticleConfig:
    """Particle inputs; diameter is in metres and temperature in kelvin."""

    diameter: Real
    initial_temperature: Real
    initial_oxide_fraction: Real

    def __post_init__(self) -> None:
        object.__setattr__(self, "diameter", _positive("diameter", self.diameter))
        object.__setattr__(
            self,
            "initial_temperature",
            _positive("initial_temperature", self.initial_temperature),
        )
        object.__setattr__(
            self,
            "initial_oxide_fraction",
            _fraction("initial_oxide_fraction", self.initial_oxide_fraction),
        )


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """Illustrative material and model parameters in SI units."""

    density: Real = 5_000.0
    heat_capacity: Real = 500.0
    heat_transfer_coefficient: Real = 100.0
    emissivity: Real = 0.8
    surroundings_temperature: Real = 300.0
    melting_temperature: Real = 1_800.0
    reaction_pre_exponential: Real = 0.05
    activation_energy: Real = 30_000.0

    def __post_init__(self) -> None:
        for name in (
            "density",
            "heat_capacity",
            "heat_transfer_coefficient",
            "surroundings_temperature",
            "melting_temperature",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        object.__setattr__(self, "emissivity", _fraction("emissivity", self.emissivity))
        for name in ("reaction_pre_exponential", "activation_energy"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Fixed-step numerical settings."""

    num_steps: Integral = 1_000

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, Integral):
            raise TypeError("num_steps must be an integer")
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least one")
        object.__setattr__(self, "num_steps", int(self.num_steps))


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Complete configuration for one simulation."""

    reactor: ReactorConfig
    particle: ParticleConfig
    material: MaterialConfig = MaterialConfig()
    solver: SolverConfig = SolverConfig()

    def __post_init__(self) -> None:
        expected_types = (
            ("reactor", self.reactor, ReactorConfig),
            ("particle", self.particle, ParticleConfig),
            ("material", self.material, MaterialConfig),
            ("solver", self.solver, SolverConfig),
        )
        for name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        if self.reactor.peak_temperature < self.material.surroundings_temperature:
            raise ValueError("peak_temperature must be at least surroundings_temperature")
