"""Per-channel standardisation fitted on training episodes only."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

STD_FLOOR = 1e-6


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    names: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, x: np.ndarray, names: list[str] | None = None) -> Standardizer:
        """Fit on any array whose last axis is the channel; NaN entries (padding) are ignored."""
        flat = np.asarray(x, dtype=float).reshape(-1, x.shape[-1])
        mean = np.nanmean(flat, axis=0)
        std = np.maximum(np.nanstd(flat, axis=0), STD_FLOOR)
        return cls(mean=mean, std=std, names=list(names or []))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.std + self.mean

    def to_json(self) -> str:
        return json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist(), "names": self.names}, indent=1)

    @classmethod
    def from_json(cls, data: dict | str) -> Standardizer:
        if isinstance(data, str):
            data = json.loads(data)
        return cls(mean=np.asarray(data["mean"], dtype=float), std=np.asarray(data["std"], dtype=float), names=list(data.get("names", [])))
