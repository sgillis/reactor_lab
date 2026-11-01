"""Validated, static configuration for particle simulations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _positive(name: str, value: float) -> float:
    value = _real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _real(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _fraction(name: str, value: float) -> float:
    value = _real(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


@dataclass(frozen=True, slots=True)
class ReactorConfig:
    """Reactor inputs; temperatures are in kelvin and time in seconds."""

    peak_temperature: float
    residence_time: float
    hydrogen_concentration: float

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

    diameter: float
    initial_temperature: float
    initial_oxide_fraction: float

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

    density: float = 5_000.0
    heat_capacity: float = 500.0
    heat_transfer_coefficient: float = 100.0
    emissivity: float = 0.8
    surroundings_temperature: float = 300.0
    melting_temperature: float = 1_800.0
    reaction_pre_exponential: float = 0.05
    activation_energy: float = 30_000.0

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

    num_steps: int = 1_000

    def __post_init__(self) -> None:
        num_steps = _integer("num_steps", self.num_steps)
        if num_steps < 1:
            raise ValueError("num_steps must be at least one")
        object.__setattr__(self, "num_steps", num_steps)


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


@dataclass(frozen=True, slots=True)
class Bounds:
    """Inclusive lower and exclusive upper bounds for uniform sampling."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _real("lower", self.lower))
        object.__setattr__(self, "upper", _real("upper", self.upper))
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")


@dataclass(frozen=True, slots=True)
class InputBounds:
    """Sampling bounds for the five variable simulation inputs."""

    peak_temperature: Bounds = Bounds(1_800.0, 2_800.0)
    residence_time: Bounds = Bounds(0.2, 0.8)
    hydrogen_concentration: Bounds = Bounds(20.0, 100.0)
    diameter: Bounds = Bounds(30e-6, 80e-6)
    initial_oxide_fraction: Bounds = Bounds(0.6, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "peak_temperature",
            "residence_time",
            "hydrogen_concentration",
            "diameter",
            "initial_oxide_fraction",
        ):
            if not isinstance(getattr(self, name), Bounds):
                raise TypeError(f"{name} must be Bounds")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Reproducible in-distribution and separated-regime sampling settings."""

    sampling_bounds: InputBounds = InputBounds()
    ood_bounds: InputBounds = InputBounds(
        peak_temperature=Bounds(2_800.0, 3_400.0)
    )
    initial_temperature: float = 300.0
    material: MaterialConfig = MaterialConfig()
    solver: SolverConfig = SolverConfig()
    train_samples: int = 512
    validation_samples: int = 128
    test_samples: int = 128
    ood_samples: int = 128
    seed: int = 2_025

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_temperature", _positive("initial_temperature", self.initial_temperature)
        )
        for name, value, expected_type in (
            ("sampling_bounds", self.sampling_bounds, InputBounds),
            ("ood_bounds", self.ood_bounds, InputBounds),
            ("material", self.material, MaterialConfig),
            ("solver", self.solver, SolverConfig),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        for name in (
            "train_samples",
            "validation_samples",
            "test_samples",
            "ood_samples",
        ):
            value = _integer(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be at least one")
            object.__setattr__(self, name, value)
        seed = _integer("seed", self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "seed", seed)

        for bounds in (self.sampling_bounds, self.ood_bounds):
            if bounds.peak_temperature.lower < self.material.surroundings_temperature:
                raise ValueError("peak_temperature bounds must be at least surroundings_temperature")
            if bounds.residence_time.lower <= 0:
                raise ValueError("residence_time bounds must be greater than zero")
            if bounds.hydrogen_concentration.lower < 0:
                raise ValueError("hydrogen_concentration bounds must be non-negative")
            if bounds.diameter.lower <= 0:
                raise ValueError("diameter bounds must be greater than zero")
            if not 0 <= bounds.initial_oxide_fraction.lower <= 1:
                raise ValueError("initial_oxide_fraction bounds must be between zero and one")
            if not 0 <= bounds.initial_oxide_fraction.upper <= 1:
                raise ValueError("initial_oxide_fraction bounds must be between zero and one")

        if not any(
            getattr(self.ood_bounds, name).upper <= getattr(self.sampling_bounds, name).lower
            or getattr(self.sampling_bounds, name).upper
            <= getattr(self.ood_bounds, name).lower
            for name in (
                "peak_temperature",
                "residence_time",
                "hydrogen_concentration",
                "diameter",
                "initial_oxide_fraction",
            )
        ):
            raise ValueError("OOD bounds must be disjoint in at least one input")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Deterministic surrogate screening and simulator-validation settings."""

    sampling_bounds: InputBounds = InputBounds(
        initial_oxide_fraction=Bounds(0.8, 1.0)
    )
    initial_temperature: float = 300.0
    material: MaterialConfig = MaterialConfig()
    solver: SolverConfig = SolverConfig()
    candidate_count: int = 4_096
    validation_count: int = 16
    max_final_oxide_fraction: float = 0.75
    seed: int = 2_025

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_temperature", _positive("initial_temperature", self.initial_temperature)
        )
        for name, value, expected_type in (
            ("sampling_bounds", self.sampling_bounds, InputBounds),
            ("material", self.material, MaterialConfig),
            ("solver", self.solver, SolverConfig),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        for name in ("candidate_count", "validation_count"):
            value = _integer(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be at least one")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "max_final_oxide_fraction",
            _fraction("max_final_oxide_fraction", self.max_final_oxide_fraction),
        )
        seed = _integer("seed", self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "seed", seed)

        bounds = self.sampling_bounds
        if bounds.peak_temperature.lower < self.material.surroundings_temperature:
            raise ValueError("peak_temperature bounds must be at least surroundings_temperature")
        if bounds.residence_time.lower <= 0:
            raise ValueError("residence_time bounds must be greater than zero")
        if bounds.hydrogen_concentration.lower < 0:
            raise ValueError("hydrogen_concentration bounds must be non-negative")
        if bounds.diameter.lower <= 0:
            raise ValueError("diameter bounds must be greater than zero")
        if not 0 <= bounds.initial_oxide_fraction.lower <= 1:
            raise ValueError("initial_oxide_fraction bounds must be between zero and one")
        if not 0 <= bounds.initial_oxide_fraction.upper <= 1:
            raise ValueError("initial_oxide_fraction bounds must be between zero and one")
        if bounds.initial_oxide_fraction.lower <= self.max_final_oxide_fraction:
            raise ValueError(
                "max_final_oxide_fraction must be less than the lower "
                "initial_oxide_fraction bound"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small-MLP architecture and deterministic Optax training settings."""

    hidden_width: int = 32
    hidden_depth: int = 2
    learning_rate: float = 0.003
    batch_size: int = 64
    epochs: int = 300
    seed: int = 2_025

    def __post_init__(self) -> None:
        for name in ("hidden_width", "hidden_depth", "batch_size", "epochs"):
            value = _integer(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be at least one")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "learning_rate", _positive("learning_rate", self.learning_rate)
        )
        seed = _integer("seed", self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "seed", seed)
