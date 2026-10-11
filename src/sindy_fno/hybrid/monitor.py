"""Mismatch monitor: normalised residuals, EWMA, branch disagreement, regime FSM."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sindy_fno.plant.channels import RESP

NORMAL, WARNING, CRITICAL = "NORMAL", "WARNING", "CRITICAL"
ACTION_DIAMETER = 2.0 * math.sqrt(6.0)   # ‖a_max − a_min‖ over the plant's [-1, 1]^6 cube


@dataclass
class Thresholds:
    delta_star: float = 1.0
    delta_lo: float = 1.2
    delta_hi: float = 2.0
    eps: float = 0.2
    k: float = 5.0
    tau_dwell: int = 10
    tau_recover: int = 30
    beta: float = 0.95
    # Hindsight horizon of the mismatch (steps): the surrogate is rolled from
    # the window H steps ago with the executed actions and compared with the
    # observation that arrived. One-step residuals are blind to slow drift.
    horizon: int = 30
    # Relative channel weights (unlisted channels weigh 1); the method allows
    # upweighting safety-critical channels.
    weights: dict = field(default_factory=lambda: {"DO2": 6.0})

    def channel_weights(self) -> np.ndarray:
        w = np.array([float(self.weights.get(ch, 1.0)) for ch in RESP])
        return w / w.sum()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, data: str | dict) -> Thresholds:
        d = json.loads(data) if isinstance(data, str) else data
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def load(cls, path: str | Path) -> Thresholds:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def guarded(th: Thresholds, eps_scale: float = 0.5) -> Thresholds:
    """Guarded-deployment settings: blend rarely, fall back sooner, go CRITICAL sooner.

    δ_lo must come down with ε: in the gated law a disagreeing step below
    δ_lo goes to the *adaptive* branch, so shrinking ε alone hands it more
    authority, not less (measured: 46 % of steps pure adaptive and the only
    oxygen violations of any hybrid variant).
    """
    return Thresholds(**{**asdict(th), "eps": th.eps * eps_scale, "delta_lo": th.delta_star, "delta_hi": th.delta_lo})


class RegimeFSM:
    """NORMAL / WARNING / CRITICAL with dwell, latch and recovery."""

    def __init__(self, th: Thresholds) -> None:
        self.th = th
        self.reset()

    def reset(self) -> None:
        self.regime = NORMAL
        self._above_hi = 0
        self._below_lo = 0

    def step(self, delta_bar: float) -> str:
        th = self.th
        self._above_hi = self._above_hi + 1 if delta_bar >= th.delta_hi else 0
        self._below_lo = self._below_lo + 1 if delta_bar < th.delta_lo else 0
        if self.regime == CRITICAL:
            if self._below_lo >= th.tau_recover:
                self.regime = NORMAL
        elif self._above_hi >= th.tau_dwell:
            self.regime = CRITICAL
        elif delta_bar >= th.delta_lo:
            self.regime = WARNING
        else:
            self.regime = NORMAL
        return self.regime


def disagreement(a_f: np.ndarray, a_s: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a_f, float) - np.asarray(a_s, float)) / ACTION_DIAMETER)


class MismatchMonitor:
    """Per-step model-reality mismatch over the response channels, smoothed, plus the regime.

    ``sigma`` are the per-channel scales of the response channels (corpus
    standard deviations); ``weights`` sum to one (uniform by default).
    """

    def __init__(self, sigma: np.ndarray, th: Thresholds | None = None, weights: np.ndarray | None = None, norm: str = "l1") -> None:
        self.sigma = np.maximum(np.asarray(sigma, float), 1e-9)
        assert len(self.sigma) == len(RESP)
        self.th = th or Thresholds()
        w = self.th.channel_weights() if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()
        if norm not in ("l1", "l2"):
            raise ValueError("norm must be 'l1' or 'l2'")
        self.norm = norm
        self.fsm = RegimeFSM(self.th)
        self.reset()

    def reset(self) -> None:
        self.delta_bar = 0.0
        self.fsm.reset()
        self.k = 0

    def aggregate(self, residuals: np.ndarray) -> float:
        if self.norm == "l1":
            return float((self.weights * np.abs(residuals)).sum())
        return float(math.sqrt((self.weights * residuals**2).sum()))

    def update(self, o_resp: np.ndarray, o_hat_resp: np.ndarray | None, a_f: np.ndarray, a_s: np.ndarray) -> dict:
        """One monitor step; ``o_hat_resp`` is the prediction made at the previous step for the executed action."""
        if o_hat_resp is None:
            residuals = np.zeros(len(RESP))
        else:
            residuals = (np.asarray(o_hat_resp, float) - np.asarray(o_resp, float)) / self.sigma
        delta = self.aggregate(residuals)
        b = self.th.beta
        self.delta_bar = delta if self.k == 0 else b * self.delta_bar + (1 - b) * delta
        regime = self.fsm.step(self.delta_bar)
        self.k += 1
        return {"delta": delta, "delta_bar": self.delta_bar, "delta_a": disagreement(a_f, a_s), "regime": regime, "residuals": residuals}


def ewma(x: np.ndarray, beta: float) -> np.ndarray:
    """Offline EWMA of a sequence (used by calibration), matching the monitor's recursion."""
    out = np.empty_like(np.asarray(x, float))
    acc = 0.0
    for i, v in enumerate(x):
        acc = v if i == 0 else beta * acc + (1 - beta) * v
        out[i] = acc
    return out
