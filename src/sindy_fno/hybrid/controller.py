"""SINDy-FNO: both branches, the mismatch monitor, the arbiter and the unconditional safety projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.adaptive.policy import AdaptivePolicy
from sindy_fno.hybrid.arbiter import Arbiter
from sindy_fno.hybrid.monitor import CRITICAL, MismatchMonitor, Thresholds, guarded
from sindy_fno.hybrid.watchdog import Watchdog, WatchdogConfig
from sindy_fno.plant.channels import ECHO, LAB, RESP
from sindy_fno.plant.recipe import action_to_physical
from sindy_fno.safe.branch import SafeBranch
from sindy_fno.safe.mpc import MPCConfig

OBS_NAMES = RESP + ECHO + LAB
N_RESP = len(RESP)
GUARDED_RATE_LIMIT = 0.1   # per step, in action units; the plant's own limit is 0.2


class SindyFnoController:
    """The hybrid controller: ``reset(obs)``, then ``act(obs, info) -> (action, diagnostics)`` per step.

    ``guarded`` applies the guarded deployment settings: halved agreement
    threshold, CRITICAL at the WARNING level, and a controller-side rate
    limit on the executed action. ``rate_limit`` overrides the latter.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        thresholds: Thresholds | None = None,
        law: str = "gated",
        mpc_cfg: MPCConfig | None = None,
        resolve_every: int = 5,
        deviation_max: float = 0.5,
        device: str = "cpu",
        adaptive_enabled: bool = True,
        shadow: bool = False,
        guarded_mode: bool = False,
        fixed_weight: float = 0.5,
        rate_limit: float | None = None,
        watchdog: WatchdogConfig | None = None,
        agent_path: str | Path | None = None,
        members: list[int] | None = None,
        projection_enabled: bool = True,
        surrogate_dir: str | Path | None = None,
        regime_enabled: bool = True,
        do2_law: dict | None = None,
    ) -> None:
        models_dir = Path(models_dir)
        self.safe = SafeBranch(models_dir / "sindy", mpc_cfg, resolve_every, deviation_max)
        self.adaptive = AdaptivePolicy(surrogate_dir or models_dir / "fno", agent_path or models_dir / "agent" / "agent.pt", device=device, auto_commit=False,
                                       members=members)
        self.projection_enabled = projection_enabled
        th_path = models_dir / "hybrid" / "thresholds.json"
        th = thresholds or (Thresholds.load(th_path) if th_path.exists() else Thresholds())
        self.th = guarded(th) if guarded_mode else th
        self.monitor = MismatchMonitor(self.adaptive.norm.obs_std[:N_RESP], self.th)
        self.arbiter = Arbiter(self.th, law, fixed_weight=fixed_weight, do2_law=do2_law)
        # The regime machine is always computed for diagnostics; only when enabled does CRITICAL force the safe action.
        self.regime_enabled = regime_enabled
        self.arbiter.value_fn = self.adaptive.value_after
        self.arbiter.cost_fn = self._safe_cost
        self.watchdog = Watchdog(watchdog or WatchdogConfig())
        self.deviation_max = deviation_max
        self.adaptive_enabled = adaptive_enabled
        self.shadow = shadow
        self.rate_limit = rate_limit if rate_limit is not None else (GUARDED_RATE_LIMIT if guarded_mode else None)
        self.idx = {n: i for i, n in enumerate(OBS_NAMES)}
        self._a_s = np.zeros(6)
        self._a_prev = np.zeros(6)
        self._x_hat = None
        self._exo = None
        self._t_next = 0.0
        self._offset = 0.0

    def reset(self, obs: np.ndarray, seed: int = 0) -> None:
        self.safe.reset(obs)
        self.adaptive.reset(seed)
        self.monitor.reset()
        self.arbiter.reset()
        self.watchdog.reset()
        self._a_prev = np.zeros(6)

    def _safe_cost(self, action: np.ndarray) -> float:
        """Constraint deficit of a candidate one hour ahead plus its squared distance to the safe action (value law)."""
        u = action_to_physical(action, self._t_next, self.deviation_max)
        head = self.safe.projection.admissible(self._x_hat, u, self._exo, self._offset)
        deficit = float(np.sum(np.minimum(head, 0.0) ** 2))
        return deficit + float(np.sum((np.asarray(action) - self._a_s) ** 2))

    def _adaptive_step(self, obs: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray | None]:
        a_f = self.adaptive.act(obs, info)
        if self.arbiter.law in ("fixed", "do2"):
            return a_f, None   # no hindsight rollout: these laws do not use the surrogate's mismatch
        return a_f, self.adaptive.hindsight(self.th.horizon)

    def act(self, obs: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
        obs = np.asarray(obs, dtype=float)
        a_s, x_hat, diag_s = self.safe.act(obs)
        t_h = float(obs[self.idx["t_h"]])
        self._t_next, self._x_hat, self._exo, self._a_s = t_h + STEP_IN_HOURS, x_hat, self.safe.exogenous(obs), a_s
        self._offset = diag_s.get("do2_offset", 0.0)

        reason, wall_f, o_hat = "", 0.0, None
        if self.adaptive_enabled:
            out, wall_f, reason = self.watchdog.guard(lambda: self._adaptive_step(obs, info))
            a_f, o_hat = out if out is not None else (None, None)
            reason = reason or self.watchdog.check_action(a_f)
        else:
            a_f = a_s.copy()
        if reason:
            a_f = a_s.copy()
        mon = self.monitor.update(obs[:N_RESP], o_hat, a_f, a_s)
        reason = reason or self.watchdog.check_monitor(mon["delta_bar"], o_hat is not None)
        if reason:
            self.watchdog.trip(reason)
            a, diag_a = a_s.copy(), {"w": 0.0, "branch": "watchdog", "agree": False}
        else:
            regime = mon["regime"] if self.regime_enabled else "NORMAL"
            a, diag_a = self.arbiter.combine(a_f, a_s, mon["delta_bar"], mon["delta_a"], regime, aux=abs(self._offset))
        if self.rate_limit is not None:
            a = np.clip(a, self._a_prev - self.rate_limit, self._a_prev + self.rate_limit)
        if self.projection_enabled:
            a_exec, diag_p = self.safe.projection.project(a, x_hat, self._t_next, self._exo, self._offset)
        else:
            a_exec, diag_p = np.asarray(a, dtype=float), {"changed": False, "status": "disabled"}
        if self.shadow:
            a_exec = np.zeros(6)   # the incumbent (recipe) acts; everything else is computed and logged
        self._a_prev = a_exec.copy()
        u = float("nan")
        if self.adaptive_enabled and self.adaptive.window is not None:
            try:
                _, u = self.adaptive.commit(a_exec)
            except Exception as exc:   # noqa: BLE001 - keep the loop alive; the next step reports the branch as dead
                self.watchdog.trip(f"commit raised {type(exc).__name__}: {exc}")
        diag = {
            "a_f": a_f, "a_s": a_s, "a_arb": a, "a_exec": a_exec, "x_hat": x_hat,
            "delta": mon["delta"], "delta_bar": mon["delta_bar"], "delta_a": mon["delta_a"], "regime": mon["regime"],
            "w": diag_a["w"], "branch": diag_a["branch"], "agree": diag_a["agree"],
            "projected": diag_p["changed"], "u": u, "critical": mon["regime"] == CRITICAL,
            "mpc_status": diag_s["mpc_status"], "mpc_wall_s": diag_s["mpc_wall_s"], "adaptive_wall_s": wall_f,
            "watchdog": reason, "watchdog_trips": self.watchdog.trips, "feed_clip_steps": self.adaptive.feed_clip_steps,
            "do2_offset": self._offset,
        }
        return a_exec, diag
