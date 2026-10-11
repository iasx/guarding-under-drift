"""Watchdog on the adaptive branch and the monitor ("monitor the monitor").

A silently dead monitor or surrogate would turn the hybrid into an
unguarded adaptive controller; the watchdog turns those failures into an
explicit fallback to the safe action for the affected step.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WatchdogConfig:
    # Wall time allowed to the adaptive branch per step. None disables the check: in offline evaluation a
    # host stall would otherwise trip the fallback and alter the recorded episode (it did, in the ε sweep).
    latency_budget_s: float | None = None
    stale_steps: int = 20             # identical δ̄ for this many steps after the horizon means a dead monitor
    fallback_regime: str = "WARNING"  # regime reported while the monitor is not trusted


@dataclass
class Watchdog:
    cfg: WatchdogConfig = field(default_factory=WatchdogConfig)
    trips: int = 0
    last_reason: str = ""
    _last_delta_bar: float | None = None
    _same: int = 0

    def reset(self) -> None:
        self.trips, self.last_reason, self._last_delta_bar, self._same = 0, "", None, 0

    def guard(self, fn: Callable[[], object]) -> tuple[object | None, float, str]:
        """Run ``fn``; return (result or None, wall seconds, failure reason or '')."""
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - the branch must never take the loop down with it
            return None, time.perf_counter() - t0, f"adaptive branch raised {type(exc).__name__}: {exc}"
        wall = time.perf_counter() - t0
        if self.cfg.latency_budget_s is not None and wall > self.cfg.latency_budget_s:
            return None, wall, f"adaptive branch took {wall:.1f} s > {self.cfg.latency_budget_s:.1f} s"
        return out, wall, ""

    def check_action(self, a: np.ndarray | None) -> str:
        if a is None or np.asarray(a).shape != (6,) or not np.isfinite(a).all():
            return "adaptive action missing or not finite"
        return ""

    def check_monitor(self, delta_bar: float, active: bool) -> str:
        """``active`` is whether the monitor has had predictions to compare (past the hindsight horizon)."""
        if not np.isfinite(delta_bar):
            return "mismatch not finite"
        if active and self._last_delta_bar is not None and delta_bar == self._last_delta_bar:
            self._same += 1
        else:
            self._same = 0
        self._last_delta_bar = delta_bar
        if active and self._same >= self.cfg.stale_steps:
            return f"mismatch unchanged for {self._same} steps"
        return ""

    def trip(self, reason: str) -> None:
        self.trips += 1
        self.last_reason = reason
