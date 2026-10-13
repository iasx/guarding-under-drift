"""Corpus subsets, each a per-episode scenario sampler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sindy_fno.plant import DriftRamp, ScenarioConfig
from sindy_fno.seeds import SUBSET_SEED_OFFSET


@dataclass(frozen=True)
class Subset:
    name: str
    seed_offset: int
    sample: Callable[[np.random.Generator], ScenarioConfig]
    description: str


def _nominal(rng: np.random.Generator) -> ScenarioConfig:
    return ScenarioConfig()


def _noise(rng: np.random.Generator) -> ScenarioConfig:
    return ScenarioConfig(noise_scale=2.5)


def _dist(rng: np.random.Generator) -> ScenarioConfig:
    # Perturbed strain / transfer parameters plus one oxygen-transfer step
    # (foaming, antifoam dosing) somewhere in the production phase.
    scale = {name: float(rng.uniform(0.8, 1.2)) for name in ("mu_P", "mu_X", "alpha_kla")}
    at = float(rng.uniform(80.0, 180.0))
    step = DriftRamp(start_h=at, end_h=at, factor=float(rng.uniform(0.6, 0.85)))
    return ScenarioConfig(param_scale=scale, drift={"alpha_kla": step})


def _init(rng: np.random.Generator) -> ScenarioConfig:
    scale = {
        "X": float(rng.uniform(0.7, 1.3)),
        "S": float(rng.uniform(0.8, 1.2)),
        "V": float(rng.uniform(0.95, 1.05)),
        "PAA": float(rng.uniform(0.9, 1.1)),
        "NH3": float(rng.uniform(0.9, 1.1)),
    }
    return ScenarioConfig(initial_condition_scale=scale)


SUBSETS: dict[str, Subset] = {
    "nom": Subset("nom", SUBSET_SEED_OFFSET["nom"], _nominal, "nominal operation"),
    "noise": Subset("noise", SUBSET_SEED_OFFSET["noise"], _noise, "measurement noise x2.5"),
    "dist": Subset(
        "dist",
        SUBSET_SEED_OFFSET["dist"],
        _dist,
        "mu_P, mu_X, alpha_kla scaled U(0.8, 1.2); alpha_kla step U(0.6, 0.85) at U(80, 180) h",
    ),
    "init": Subset(
        "init",
        SUBSET_SEED_OFFSET["init"],
        _init,
        "initial X U(0.7, 1.3), S U(0.8, 1.2), V U(0.95, 1.05), PAA and NH3 U(0.9, 1.1)",
    ),
}
