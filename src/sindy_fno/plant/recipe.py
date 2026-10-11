"""Default IndPenSim recipe and the supervisory action space."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pensimpy.data.profiles import (
    DISCHARGE,
    DISCHARGE_DEFAULT_PROFILE,
    FG,
    FG_DEFAULT_PROFILE,
    FOIL,
    FOIL_DEFAULT_PROFILE,
    FS,
    FS_DEFAULT_PROFILE,
    PAA,
    PAA_DEFAULT_PROFILE,
    PRES,
    PRESS_DEFAULT_PROFILE,
    WATER,
    WATER_DEFAULT_PROFILE,
)
from pensimpy.examples.recipe import Recipe, RecipeCombo

# Agent-manipulated feeds, in the order the action vector uses. Fpaa is left
# to the recipe because the plant's PAA controller overrides it after 10 h.
ACTION_NAMES = ["Fs", "Foil", "Fg", "pressure", "discharge", "Fw"]

# Physical actuator bounds.
ACTION_MIN = np.array([0.0, 7.0, 21.0, 0.5, 0.0, 0.0])
ACTION_MAX = np.array([151.0, 36.0, 76.0, 1.2, 4100.0, 510.0])

# Argument order of PenSimEnv.step after (k, x).
ENV_STEP_ORDER = ["Fs", "Foil", "Fg", "pressure", "discharge", "Fw", "Fpaa"]


def default_recipe() -> RecipeCombo:
    return RecipeCombo(
        recipe_dict={
            FS: Recipe(FS_DEFAULT_PROFILE, FS),
            FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
            FG: Recipe(FG_DEFAULT_PROFILE, FG),
            PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
            DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
            WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
            PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
        }
    )


@lru_cache(maxsize=1)
def _shared_recipe() -> RecipeCombo:
    return default_recipe()


def recipe_values(t_hours: float) -> dict[str, float]:
    """Recipe setpoints at ``t_hours`` (each holds until its listed time)."""
    return _shared_recipe().get_values_dict_at(t_hours)


def action_to_physical(a: np.ndarray, t_h: float, deviation_max: float = 0.5) -> np.ndarray:
    """Plant action in [-1, 1]^6 to the physical feeds applied at ``t_h`` (same map as ``PenSimPlant``)."""
    r = recipe_values(t_h)
    nominal = np.array([r[n] for n in ACTION_NAMES])
    return np.clip(nominal * (1.0 + deviation_max * np.asarray(a, dtype=float)), ACTION_MIN, ACTION_MAX)


def physical_to_action(u: np.ndarray, t_h: float, deviation_max: float = 0.5) -> np.ndarray:
    """Inverse of ``action_to_physical``; zero wherever the recipe value is zero."""
    r = recipe_values(t_h)
    nominal = np.array([r[n] for n in ACTION_NAMES])
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(nominal > 0, (np.asarray(u, dtype=float) / nominal - 1.0) / deviation_max, 0.0)
    return np.clip(a, -1.0, 1.0)
