"""Dissolved-CO2 soft sensor: a linear map from routine measurements, fitted by ``experiments/hybrid/fit_co2_sensor.py``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = ["1", "CO2outgas", "CO2outgas*pressure", "pressure", "T", "CER", "Fg", "CER/Fg"]


def features(obs: np.ndarray, obs_names: list[str]) -> np.ndarray:
    """Feature rows for observations of shape (n, channels) or (channels,)."""
    obs = np.atleast_2d(np.asarray(obs, float))
    co2g, p, temp, cer, fg = (obs[:, obs_names.index(k)] for k in ("CO2outgas", "pressure", "T", "CER", "Fg"))
    return np.column_stack([np.ones_like(p), co2g, co2g * p, p, temp, cer, fg, cer / np.maximum(fg, 1e-6)])


class CO2Sensor:
    def __init__(self, path: str | Path, obs_names: list[str]) -> None:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if spec["features"] != FEATURES:
            raise ValueError("sensor was fitted on a different feature set")
        self.coef = np.asarray(spec["coef"], float)
        self.obs_names = list(obs_names)

    def estimate(self, obs: np.ndarray) -> float:
        return float(features(obs, self.obs_names)[0] @ self.coef)
