"""Scenario configuration: noise, perturbation, drift and faults."""

from __future__ import annotations

from dataclasses import dataclass, field

from sindy_fno.plant.channels import DEFAULT_SIGMA, SENSORS

# Position of each strain / transfer parameter in PenSimEnv.param_list.
PARAM_INDEX = {"mu_P": 0, "mu_X": 1, "alpha_kla": 2, "N_conc_paa": 3, "PAA_c": 4}

# X0 fields that may be scaled at reset.
INITIAL_CONDITIONS = ["S", "DO2", "X", "V", "Wt", "pH", "T", "PAA", "NH3"]


@dataclass(frozen=True)
class DriftRamp:
    """Parameter multiplier ramping linearly from 1 at ``start_h`` to ``factor`` at ``end_h``.

    ``end_h == start_h`` is a step: the factor applies from ``start_h`` on.
    """

    start_h: float
    end_h: float
    factor: float

    def __post_init__(self) -> None:
        if self.end_h < self.start_h:
            raise ValueError("end_h must not precede start_h")

    def multiplier(self, t_h: float) -> float:
        if t_h < self.start_h:
            return 1.0
        if t_h >= self.end_h:
            return self.factor
        frac = (t_h - self.start_h) / (self.end_h - self.start_h)
        return 1.0 + (self.factor - 1.0) * frac


@dataclass(frozen=True)
class BiasRamp:
    """Additive sensor offset ramping linearly from 0 at ``start_h`` to ``amount`` at ``end_h`` (S6: a fault, not a dynamics change)."""

    start_h: float
    end_h: float
    amount: float

    def __post_init__(self) -> None:
        if self.end_h < self.start_h:
            raise ValueError("end_h must not precede start_h")

    def value(self, t_h: float) -> float:
        if t_h < self.start_h:
            return 0.0
        if t_h >= self.end_h:
            return self.amount
        return self.amount * (t_h - self.start_h) / (self.end_h - self.start_h)


@dataclass
class ScenarioConfig:
    noise_scale: float = 1.0
    sensor_sigma: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGMA))
    assay_rel_sigma: float = 0.02
    param_scale: dict[str, float] = field(default_factory=dict)
    drift: dict[str, DriftRamp] = field(default_factory=dict)
    initial_condition_scale: dict[str, float] = field(default_factory=dict)
    faults: int = 0
    sensor_bias: dict[str, BiasRamp] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (*self.param_scale, *self.drift):
            if name not in PARAM_INDEX:
                raise ValueError(f"unknown parameter {name!r}; choose from {sorted(PARAM_INDEX)}")
        for name in self.initial_condition_scale:
            if name not in INITIAL_CONDITIONS:
                raise ValueError(f"unknown initial condition {name!r}; choose from {INITIAL_CONDITIONS}")
        for name in self.sensor_sigma:
            if name not in DEFAULT_SIGMA:
                raise ValueError(f"unknown sensor {name!r}; choose from {sorted(DEFAULT_SIGMA)}")
        for name in self.sensor_bias:
            if name not in SENSORS:
                raise ValueError(f"unknown sensor {name!r}; choose from {SENSORS}")
        if self.noise_scale < 0 or self.assay_rel_sigma < 0:
            raise ValueError("noise scales must be non-negative")
