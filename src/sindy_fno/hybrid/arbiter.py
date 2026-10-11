"""Arbitration laws between the adaptive and safe actions."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from sindy_fno.hybrid.monitor import CRITICAL, Thresholds

LAWS = ("gated", "convex", "switch", "value", "fixed", "do2")


class Arbiter:
    """``combine(a_F, a_S, δ̄, Δa, regime) -> (a, diag)`` under one of four laws.

    ``gated`` (recommended) blends only inside the agreement region and
    switches otherwise; ``convex`` always blends; ``switch`` picks by δ̄;
    ``value`` scores candidates with the adaptive critic and the safe
    branch's cost through callables set by the controller. CRITICAL forces
    the safe action in every law except ``fixed``, which ignores the monitor
    entirely and blends with a constant weight: the control that asks
    whether the mismatch signal earns its place at all.
    """

    def __init__(self, th: Thresholds, law: str = "gated", eps_hysteresis: float = 1.2, fixed_weight: float = 0.5, do2_law: dict | None = None) -> None:
        if law not in LAWS:
            raise ValueError(f"law must be one of {LAWS}")
        self.th = th
        self.law = law
        self.eps_hysteresis = eps_hysteresis
        self.fixed_weight = fixed_weight
        # The ``do2`` law replaces the surrogate mismatch by the safe branch's own DO2 offset (|measured - modelled|):
        # w = 1 / (1 + exp(k (|offset| - o*))), with o* and k from the same quantile rules as delta_star and k.
        self.do2_law = do2_law or {}
        self.value_fn: Callable[[np.ndarray], float] | None = None
        self.cost_fn: Callable[[np.ndarray], float] | None = None
        self.reset()

    def reset(self) -> None:
        self._agree = True

    def weight(self, delta_bar: float) -> float:
        z = self.th.k * (delta_bar - self.th.delta_star)
        return 1.0 / (1.0 + math.exp(min(max(z, -60.0), 60.0)))

    def _agreement(self, delta_a: float) -> bool:
        # Hysteresis: leave agreement only above eps_hysteresis * eps, re-enter below eps.
        if self._agree:
            self._agree = delta_a <= self.eps_hysteresis * self.th.eps
        else:
            self._agree = delta_a <= self.th.eps
        return self._agree

    def combine(self, a_f: np.ndarray, a_s: np.ndarray, delta_bar: float, delta_a: float, regime: str, aux: float | None = None) -> tuple[np.ndarray, dict]:
        a_f, a_s = np.asarray(a_f, float), np.asarray(a_s, float)
        if self.law == "fixed":
            w = self.fixed_weight
            return w * a_f + (1 - w) * a_s, {"w": w, "branch": "blend", "agree": True}
        if self.law == "do2":
            if aux is None or not self.do2_law:
                raise RuntimeError("do2 law needs the DO2 offset and its calibrated o_star / k")
            w = 1.0 / (1.0 + math.exp(self.do2_law["k"] * (float(aux) - self.do2_law["o_star"])))
            return w * a_f + (1 - w) * a_s, {"w": w, "branch": "blend", "agree": True}
        w = self.weight(delta_bar)
        agree = self._agreement(delta_a)
        if regime == CRITICAL:
            return a_s.copy(), {"w": 0.0, "branch": "critical", "agree": agree}
        if self.law == "gated":
            if agree:
                return w * a_f + (1 - w) * a_s, {"w": w, "branch": "blend", "agree": True}
            if delta_bar < self.th.delta_lo:
                return a_f.copy(), {"w": 1.0, "branch": "adaptive", "agree": False}
            return a_s.copy(), {"w": 0.0, "branch": "safe", "agree": False}
        if self.law == "convex":
            return w * a_f + (1 - w) * a_s, {"w": w, "branch": "blend", "agree": agree}
        if self.law == "switch":
            if delta_bar < self.th.delta_star:
                return a_f.copy(), {"w": 1.0, "branch": "adaptive", "agree": agree}
            return a_s.copy(), {"w": 0.0, "branch": "safe", "agree": agree}
        # value-space blend
        if self.value_fn is None or self.cost_fn is None:
            raise RuntimeError("value law needs value_fn and cost_fn")
        best, best_score, best_w = None, -math.inf, None
        for wc in (1.0, 0.75, 0.5, 0.25, 0.0):
            cand = wc * a_f + (1 - wc) * a_s
            score = w * self.value_fn(cand) - (1 - w) * self.cost_fn(cand)
            if score > best_score:
                best, best_score, best_w = cand, score, wc
        return best, {"w": w, "branch": f"value:{best_w}", "agree": agree, "score": best_score}
