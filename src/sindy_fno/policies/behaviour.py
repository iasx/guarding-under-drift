"""Behaviour policy that generates the offline corpus."""

from __future__ import annotations

import math

import numpy as np

from sindy_fno.plant.recipe import ACTION_NAMES


class RecipePerturbationPolicy:
    """The default recipe with held random deviations and a per-episode sugar-feed scale.

    Every ``hold_h`` hours a deviation ``U(-amplitude, amplitude)`` per action
    is drawn and held; the Fs component is shifted so that the recipe's sugar
    feed is scaled by ``fs_scale`` (drawn once per episode). Deviations are
    in the plant's action units, so ``amplitude = 0.4`` with the plant's
    default ``deviation_max = 0.5`` means ±20 % of the recipe.
    """

    def __init__(
        self,
        amplitude: float = 0.4,
        hold_h: float = 5.0,
        fs_scale_range: tuple[float, float] = (0.7, 1.3),
        deviation_max: float = 0.5,
    ) -> None:
        if amplitude < 0 or hold_h <= 0 or deviation_max <= 0:
            raise ValueError("amplitude must be >= 0; hold_h and deviation_max must be > 0")
        self.amplitude = float(amplitude)
        self.hold_h = float(hold_h)
        self.fs_scale_range = (float(fs_scale_range[0]), float(fs_scale_range[1]))
        self.deviation_max = float(deviation_max)
        self.fs_scale = 1.0
        self._rng = np.random.default_rng(0)
        self._window = -1
        self._held = np.zeros(len(ACTION_NAMES))

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng([int(seed), 0xB7])
        self.fs_scale = float(self._rng.uniform(*self.fs_scale_range))
        self._window = -1
        self._held = np.zeros(len(ACTION_NAMES))

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        window = math.floor(float(info["t_h"]) / self.hold_h + 1e-9)
        if window != self._window:
            self._window = window
            dev = self._rng.uniform(-self.amplitude, self.amplitude, size=len(ACTION_NAMES))
            dev[0] += (self.fs_scale - 1.0) / self.deviation_max
            self._held = np.clip(dev, -1.0, 1.0)
        return self._held.copy()
