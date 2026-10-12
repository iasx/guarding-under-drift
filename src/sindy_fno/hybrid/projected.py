"""Adaptive actor under the safety projection alone: the control that asks whether arbitration earns its complexity.

No mismatch monitor, no arbiter, no model-predictive solve. The safe
model appears only through the state estimator and the projection, which
is the part of the safe branch that is unconditional in the method.
If this holds the constraints under drift, the monitor and the
arbiter are yield mechanisms rather than safety mechanisms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.adaptive.policy import AdaptivePolicy
from sindy_fno.plant.channels import ECHO, LAB, RESP
from sindy_fno.safe.branch import SafeBranch
from sindy_fno.safe.mpc import MPCConfig

OBS_NAMES = RESP + ECHO + LAB


class ProjectedActor:
    """``reset(obs)``, then ``act(obs, info) -> (action, diagnostics)``, same interface as the hybrid controller."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        mpc_cfg: MPCConfig | None = None,
        deviation_max: float = 0.5,
        device: str = "cpu",
        agent_path: str | Path | None = None,
        surrogate_dir: str | Path | None = None,
    ) -> None:
        models_dir = Path(models_dir)
        self.safe = SafeBranch(models_dir / "sindy", mpc_cfg, deviation_max=deviation_max)
        self.adaptive = AdaptivePolicy(surrogate_dir or models_dir / "fno", agent_path or models_dir / "agent" / "agent.pt", device=device, auto_commit=False)
        self.idx = {n: i for i, n in enumerate(OBS_NAMES)}

    def reset(self, obs: np.ndarray, seed: int = 0) -> None:
        self.safe.reset(obs)
        self.adaptive.reset(seed)

    def act(self, obs: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
        obs = np.asarray(obs, dtype=float)
        x_hat, exo, offset = self.safe.estimate(obs)
        self.safe.k += 1
        a_f = self.adaptive.act(obs, info)
        t_next = float(obs[self.idx["t_h"]]) + STEP_IN_HOURS
        a_exec, diag_p = self.safe.projection.project(a_f, x_hat, t_next, exo, offset)
        _, u = self.adaptive.commit(a_exec)
        return a_exec, {
            "a_f": a_f, "a_exec": a_exec, "x_hat": x_hat, "u": u, "branch": "adaptive",
            "projected": diag_p["changed"], "projection_status": diag_p["status"],
            "projection_delta": float(np.abs(a_exec - a_f).max()), "do2_offset": offset,
        }


class FilteredActor(ProjectedActor):
    """The actor under a receding-horizon safety filter with the safe branch's full 24 h horizon and constraints,
    solved every step, instead of the one-hour projection; the control that separates filter horizon from blending."""

    def __init__(self, models_dir: str | Path = "models", mpc_cfg: MPCConfig | None = None, deviation_max: float = 0.5, device: str = "cpu",
                 agent_path: str | Path | None = None, surrogate_dir: str | Path | None = None, variant: str = "plain",
                 margin: float = 3.0, guard_on: float | None = None, sensor_bias: float = 0.0, fallback: str = "mpc") -> None:
        super().__init__(models_dir, mpc_cfg, deviation_max, device, agent_path, surrogate_dir)
        from sindy_fno.safe.filter import SafetyFilterMPC, TightenedSafetyFilter

        if variant == "plain":
            self.filter = SafetyFilterMPC(self.safe.model, self.safe.meas, self.safe.mpc.cfg)
        elif variant in ("tightened", "tightened_end"):
            self.filter = TightenedSafetyFilter(self.safe.model, self.safe.meas, self.safe.mpc.cfg, to_batch_end=variant == "tightened_end")
        elif variant in ("tightened_margin", "tightened_margin_co2_oracle", "tightened_margin_co2_sensor", "tightened_margin_co2_simplex"):
            self.filter = TightenedSafetyFilter(self.safe.model, self.safe.meas, self.safe.mpc.cfg, do2_extra=margin)
        elif variant == "tightened_error":
            self.filter = TightenedSafetyFilter(self.safe.model, self.safe.meas, self.safe.mpc.cfg, err_gain=1.0)
        elif variant == "tightened_capacity":
            self.filter = TightenedSafetyFilter(self.safe.model, self.safe.meas, self.safe.mpc.cfg, cap_scale=margin)
        elif variant in ("tightened_substrate_oracle", "tightened_co2_oracle"):
            self.filter = TightenedSafetyFilter(self.safe.model, self.safe.meas, self.safe.mpc.cfg)
        else:
            raise ValueError(f"unknown filter variant {variant!r}")
        self.deviation_max = deviation_max
        self.substrate_oracle = variant == "tightened_substrate_oracle"
        self.co2_oracle = variant in ("tightened_co2_oracle", "tightened_margin_co2_oracle")
        self.co2_sensor = None
        self.simplex = variant == "tightened_margin_co2_simplex"
        self._mpc_plan, self._switched = None, 0
        if variant in ("tightened_margin_co2_sensor", "tightened_margin_co2_simplex"):
            from sindy_fno.safe.co2_sensor import CO2Sensor

            self.co2_sensor = CO2Sensor(Path(models_dir) / "hybrid" / "co2_sensor.json", OBS_NAMES)
        # The guard switches on above guard_on and off 0.1 below it; the defaults are 7.45 for the oracle and 7.35 for
        # the soft sensor, whose 95th-percentile error near the switch is about 0.2.
        self.guard_on = guard_on
        self.sensor_bias = sensor_bias
        self.fallback = fallback
        self._guard = False
        self._u_prev = None

    def reset(self, obs: np.ndarray, seed: int = 0) -> None:
        super().reset(obs, seed)
        self.filter.reset()
        self._guard = False
        self._mpc_plan, self._switched = None, 0
        self._u_prev = None

    def act(self, obs: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
        from sindy_fno.plant.recipe import action_to_physical, physical_to_action

        obs = np.asarray(obs, dtype=float)
        x_hat, exo, offset = self.safe.estimate(obs)
        self.safe.k += 1
        a_f = self.adaptive.act(obs, info)
        t_next = float(obs[self.idx["t_h"]]) + STEP_IN_HOURS
        u_req = action_to_physical(a_f, t_next, self.deviation_max)
        u_prev = np.asarray(info["action_applied"], float)[self.filter.dec_idx] if self._u_prev is None else self._u_prev
        out = self.filter.filter(u_req, x_hat, t_next, exo, u_prev, offset)
        u_exec = u_req.copy()
        u_exec[self.filter.dec_idx] = out["u_next"][self.filter.dec_idx]
        if self.substrate_oracle:
            u_exec = self._substrate_guard(u_exec, info, t_next)
        co2_est = float("nan")
        if self.co2_oracle:
            from sindy_fno.plant.channels import HIDDEN

            on = 7.45 if self.guard_on is None else self.guard_on
            u_exec = self._co2_guard(u_exec, float(np.asarray(info["hidden"], float)[HIDDEN.index("CO2_d")]), t_next, on=on, off=on - 0.1)
        branch = "adaptive"
        if self.co2_sensor is not None:
            co2_est = self.co2_sensor.estimate(obs) + self.sensor_bias
            on = 7.35 if self.guard_on is None else self.guard_on
            if self.simplex:
                self._guard = co2_est > on or (self._guard and co2_est > on - 0.1)
            else:
                u_exec = self._co2_guard(u_exec, co2_est, t_next, on=on, off=on - 0.1)
        a_exec = physical_to_action(u_exec, t_next, self.deviation_max)
        if self.simplex and self._guard:
            a_exec = np.zeros_like(a_exec) if self.fallback == "recipe" else self._safe_action(obs, x_hat, exo, offset, t_next)
            u_exec = action_to_physical(a_exec, t_next, self.deviation_max)
            branch = "safe"
        elif self.simplex:
            self._mpc_plan, self._switched = None, 0
        self._u_prev = u_exec[self.filter.dec_idx]
        _, u = self.adaptive.commit(a_exec)
        return a_exec, {
            "a_f": a_f, "a_exec": a_exec, "x_hat": x_hat, "u": u, "branch": branch,
            "projected": out["changed"], "projection_status": out["status"], "mpc_wall_s": out["wall_s"],
            "projection_delta": float(np.abs(a_exec - a_f).max()), "do2_offset": offset, "co2_est": co2_est,
        }

    def _safe_action(self, obs: np.ndarray, x_hat: np.ndarray, exo: np.ndarray, offset: float, t_next: float) -> np.ndarray:
        """The safe branch's projected MPC action, for the Simplex-style baseline that hands control to it while the CO2
        estimate is high; the MPC is re-solved on the safe branch's cadence from the moment of the switch."""
        from sindy_fno.plant.recipe import physical_to_action

        if self._mpc_plan is None or self._switched % self.safe.resolve_every == 0:
            u_applied = obs[self.safe.echo_idx]
            self._mpc_plan = self.safe.mpc.solve(x_hat, float(obs[self.idx["t_h"]]), exo, u_applied[self.safe.dec_idx], offset)
        self._switched += 1
        action = physical_to_action(self._mpc_plan["u_next"].copy(), t_next, self.deviation_max)
        action, _ = self.safe.projection.project(action, x_hat, t_next, exo, offset)
        return action

    def _substrate_guard(self, u_exec: np.ndarray, info: dict, t_next: float) -> np.ndarray:
        """Oracle test of the substrate mechanism: with the true substrate known, hold both feeds at 80 % of the recipe
        while it is accumulating (on above 0.01 g/L, off below 0.004 g/L; normal operation sits at 0.001–0.002)."""
        from sindy_fno.plant.channels import HIDDEN
        from sindy_fno.plant.recipe import ACTION_NAMES, action_to_physical

        s = float(np.asarray(info["hidden"], float)[HIDDEN.index("S")])
        self._guard = s > 0.01 or (self._guard and s > 0.004)
        if self._guard:
            recipe = action_to_physical(np.zeros(len(ACTION_NAMES)), t_next, self.deviation_max)
            for name in ("Fs", "Foil"):
                i = ACTION_NAMES.index(name)
                u_exec[i] = min(u_exec[i], 0.8 * recipe[i])
        return u_exec

    def _co2_guard(self, u_exec: np.ndarray, co2: float, t_next: float, on: float, off: float) -> np.ndarray:
        """Keep every lever on the recipe's side in the direction that raises dissolved CO2 (feeds not above, aeration
        not below, head pressure not above the recipe) while CO2 is near the growth-inhibition switch at 7.57, which
        the identified model does not contain; ``co2`` is the true value (oracle) or the soft sensor's estimate."""
        from sindy_fno.plant.recipe import ACTION_NAMES, action_to_physical

        self._guard = co2 > on or (self._guard and co2 > off)
        if self._guard:
            recipe = action_to_physical(np.zeros(len(ACTION_NAMES)), t_next, self.deviation_max)
            for name, keep in (("Fs", min), ("Foil", min), ("Fg", max), ("pressure", min)):
                i = ACTION_NAMES.index(name)
                u_exec[i] = keep(u_exec[i], recipe[i])
        return u_exec
