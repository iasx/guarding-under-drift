"""PenSimPy as a partially observed control problem."""

from __future__ import annotations

import math
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pensimpy.constants import NUM_STEPS, STEP_IN_HOURS
from pensimpy.peni_env_setup import PenSimEnv

from sindy_fno.plant.channels import (
    ECHO,
    HIDDEN,
    LAB,
    LAB_SOURCE,
    PID_FLOWS,
    RESP,
    SENSORS,
    as_ph,
    read,
)
from sindy_fno.plant.recipe import ACTION_MAX, ACTION_MIN, ACTION_NAMES, default_recipe
from sindy_fno.plant.scenario import PARAM_INDEX, ScenarioConfig

INTEGRATOR_ERRORS = (ValueError, OverflowError, ZeroDivisionError, FloatingPointError)


class PenSimPlant(gym.Env):
    """Gymnasium view of a PenSimPy batch.

    Actions are deviations in [-1, 1] from the default recipe, scaled by
    ``deviation_max`` and rate-limited per step; observations are physical
    units in the response / echo / lab layout of ``channels.py``.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        deviation_max: float = 0.5,
        rate_limit: float = 0.2,
    ) -> None:
        super().__init__()
        self.scenario = scenario or ScenarioConfig()
        self.deviation_max = float(deviation_max)
        self.rate_limit = float(rate_limit)
        self.recipe = default_recipe()
        self.env = PenSimEnv(recipe_combo=self.recipe)
        self.env.ctrl_flags.Faults = self.scenario.faults

        self.obs_names = RESP + ECHO + LAB
        self.resp_idx = np.arange(0, len(RESP))
        self.echo_idx = np.arange(len(RESP), len(RESP) + len(ECHO))
        self.lab_idx = np.arange(len(RESP) + len(ECHO), len(self.obs_names))
        self.hidden_names = list(HIDDEN)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.obs_names),), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTION_NAMES),), dtype=np.float64)

        self._sigma = np.array([self.scenario.sensor_sigma.get(s, 0.0) for s in SENSORS]) * self.scenario.noise_scale
        self._seed = 0
        self._rng = np.random.default_rng(0)
        self._x = None
        self._active = False
        self.k = 0
        self._prev_dev = np.zeros(len(ACTION_NAMES))
        self._lab = np.zeros(len(LAB) - 1)
        self._lab_sample_h = 0.0
        self._param_baseline: list[float] = []
        self._last_obs = np.zeros(len(self.obs_names))

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        self.env.random_seed_ref = self._seed
        self._rng = np.random.default_rng([self._seed, 0xA5])
        _, self._x = self.env.reset()
        import pensimpy.ode.indpensim_ode_py as ode

        ode.CO2_INHIBITION_HALF_WIDTH = self.scenario.co2_inhibition_half_width
        ode.CO2_INHIBITION_OFF = self.scenario.co2_inhibition_off
        self._apply_initial_condition_scale()
        for name, factor in self.scenario.param_scale.items():
            self.env.param_list[PARAM_INDEX[name]] *= factor
        self._param_baseline = list(self.env.param_list)

        self.k = 0
        self._active = True
        self._prev_dev = np.zeros(len(ACTION_NAMES))
        x0 = self.env.x0
        self._lab = np.array([0.0, x0.X, x0.PAA, x0.NH3, 4.0])
        self._lab_sample_h = 0.0

        sensors = np.array([as_ph(x0.pH), x0.T, x0.DO2, x0.O2, x0.CO2outgas, x0.Wt, 0.0, 0.0])
        recipe = self.recipe.get_values_dict_at(STEP_IN_HOURS)
        applied = np.array([recipe[n] for n in ACTION_NAMES])
        obs = self._assemble(sensors, np.zeros(len(PID_FLOWS)), applied, 0.0)
        hidden = np.array([self._hidden_from_x0(name) for name in self.hidden_names])
        return obs, self._info(0.0, applied, recipe, hidden, assay_event=False)

    def _apply_initial_condition_scale(self) -> None:
        x0 = self.env.x0
        for name, factor in self.scenario.initial_condition_scale.items():
            if name == "pH":
                # Stored as H+ concentration; scale the pH value itself.
                x0.pH = 10 ** (-(-math.log10(x0.pH) * factor))
            else:
                setattr(x0, name, getattr(x0, name) * factor)
                if name == "V":
                    # The simulator draws the initial weight independently of the volume; a scaled volume with the
                    # nominal weight would bias the weight-based volume soft sensor by the whole perturbation.
                    x0.Wt = x0.Wt * factor

    def _hidden_from_x0(self, name: str) -> float:
        x0 = self.env.x0
        if name == "pH":
            return as_ph(x0.pH)
        if name == "Viscosity":
            return 4.0
        return float(getattr(x0, name, 0.0))

    # ------------------------------------------------------------------- step

    def step(self, action):
        if not self._active:
            raise RuntimeError("call reset() before step(); the episode has ended")
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (len(ACTION_NAMES),):
            raise ValueError(f"action must have shape ({len(ACTION_NAMES)},), got {a.shape}")
        if not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1.0 + 1e-9):
            raise ValueError("action components must lie in [-1, 1]")

        dev = np.clip(a, self._prev_dev - self.rate_limit, self._prev_dev + self.rate_limit)
        self._prev_dev = dev
        k = self.k + 1
        t_h = k * STEP_IN_HOURS
        self._apply_drift(t_h)
        recipe = self.recipe.get_values_dict_at(t_h)
        nominal = np.array([recipe[n] for n in ACTION_NAMES])
        applied = np.clip(nominal * (1.0 + self.deviation_max * dev), ACTION_MIN, ACTION_MAX)
        args = dict(zip(ACTION_NAMES, applied))

        try:
            _, x, reward, done = self.env.step(
                k, self._x, args["Fs"], args["Foil"], args["Fg"], args["pressure"],
                args["discharge"], args["Fw"], recipe["Fpaa"],
            )
        except INTEGRATOR_ERRORS as err:
            return self._catastrophic(str(err), t_h, applied, recipe)
        if not np.isfinite(reward):
            return self._catastrophic("non-finite reward", t_h, applied, recipe)

        self._x = x
        self.k = k
        assay_event = self._update_lab(k, t_h)
        sensors = np.array([read(x, s, k) for s in SENSORS])
        flows = np.array([read(x, f, k) for f in PID_FLOWS])
        obs = self._assemble(sensors, flows, applied, t_h)
        hidden = np.array([read(x, name, k) for name in self.hidden_names])
        terminated = bool(done)
        if terminated:
            self._active = False
        info = self._info(t_h, applied, recipe, hidden, assay_event)
        return obs, float(reward), terminated, False, info

    def _apply_drift(self, t_h: float) -> None:
        for name, ramp in self.scenario.drift.items():
            i = PARAM_INDEX[name]
            self.env.param_list[i] = self._param_baseline[i] * ramp.multiplier(t_h)

    def _update_lab(self, k: int, t_h: float) -> bool:
        x = self._x
        if math.isnan(x.P_offline.y[k - 1]):
            return False
        values = np.array([x.__dict__[f"{src}_offline"].y[k - 1] for src in LAB_SOURCE.values()])
        rel_sigma = self.scenario.assay_rel_sigma * self.scenario.noise_scale
        noise = 1.0 + rel_sigma * self._rng.standard_normal(values.shape)
        self._lab = values * noise
        self._lab_sample_h = float(x.P_offline.t[k - 1])
        return True

    def _assemble(self, sensors, flows, applied, t_h: float) -> np.ndarray:
        bias = np.array([self.scenario.sensor_bias[s].value(t_h) if s in self.scenario.sensor_bias else 0.0 for s in SENSORS])
        noisy = sensors + bias + self._sigma * self._rng.standard_normal(sensors.shape)
        lab = np.concatenate([self._lab, [t_h - self._lab_sample_h]])
        obs = np.concatenate([noisy, flows, applied, [t_h], lab])
        self._last_obs = obs
        return obs

    def _catastrophic(self, message: str, t_h: float, applied, recipe):
        self._active = False
        hidden = np.full(len(self.hidden_names), np.nan)
        info = self._info(t_h, applied, recipe, hidden, assay_event=False)
        info["catastrophic"] = True
        info["integrator_failure"] = message
        return self._last_obs, 0.0, True, False, info

    def _info(self, t_h: float, applied, recipe, hidden, assay_event: bool) -> dict:
        return {
            "t_h": t_h,
            "k": self.k,
            "action_applied": np.asarray(applied, dtype=float),
            "recipe": dict(recipe),
            "hidden": hidden,
            "assay_event": bool(assay_event),
            "catastrophic": False,
            "integrator_failure": None,
        }

    @property
    def max_episode_steps(self) -> int:
        return NUM_STEPS
