from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Anything that can drive ``PenSimPlant``: reset with a seed, then map (obs, info) to an action."""

    def reset(self, seed: int) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...
