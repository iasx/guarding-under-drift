"""The safe branch as one controller: estimator, MPC and projection over the identified model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.plant.channels import ECHO, LAB, RESP
from sindy_fno.plant.recipe import ACTION_NAMES, physical_to_action
from sindy_fno.safe.estimator import SafeStateEstimator
from sindy_fno.safe.measurements import MeasurementModels
from sindy_fno.safe.mpc import DECIDED, EXOGENOUS, MPCConfig, SafeMPC, assemble_input
from sindy_fno.safe.projection import SafetyProjection
from sindy_fno.safe.sparse_model import STATE_FLOOR, SparseModel, StaticModel

OBS_NAMES = RESP + ECHO + LAB


class SafeBranch:
    """Comparator 1 on its own; the safe side of the arbiter in the full method."""

    def __init__(self, models_dir: str | Path = "models/sindy", cfg: MPCConfig | None = None, resolve_every: int = 5, deviation_max: float = 0.5) -> None:
        models_dir = Path(models_dir)
        self.model = SparseModel.from_json((models_dir / "model.json").read_text(encoding="utf-8"))
        do2 = StaticModel.from_json((models_dir / "do2_model.json").read_text(encoding="utf-8"))
        self.meas = MeasurementModels.load(models_dir)
        self.meas.do2 = do2
        self.cfg = cfg or MPCConfig()
        self.estimator = SafeStateEstimator(self.model, self.meas)
        self.mpc = SafeMPC(self.model, self.meas, self.cfg)
        self.projection = SafetyProjection(self.model, self.meas, self.cfg, deviation_max)
        self.resolve_every = resolve_every
        self.deviation_max = deviation_max
        self.idx = {n: i for i, n in enumerate(OBS_NAMES)}
        self.exo_idx = [self.idx[n] for n in EXOGENOUS]
        self.echo_idx = [self.idx[n] for n in ACTION_NAMES]
        self.dec_idx = [ACTION_NAMES.index(n) for n in DECIDED]
        self.k = 0
        self._last = None
        self.do2_offset = 0.0
        self.offset_beta = 0.8   # EWMA on the measured-minus-modelled DO2

    def set_models(self, model: SparseModel, do2: StaticModel) -> None:
        """Install refitted models everywhere they are baked in (estimator, MPC and projection CasADi graphs)."""
        self.model = model
        self.meas.do2 = do2
        self.estimator.model = model
        self.mpc.model = model
        names = [t.name for t in model.terms]
        self.mpc.production_rate = float(max(model.coef[1, names.index("X")], 0.0))
        self.mpc._build()
        self.projection.model = model
        self.projection._build()

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.k = 0
        self.mpc.reset()
        self._last = None
        self.do2_offset = 0.0
        return self.estimator.reset(obs)

    def _update_do2_offset(self, obs: np.ndarray, x_hat: np.ndarray) -> float:
        """Offset-free correction: the level comes from the sensor, the input sensitivity from the model."""
        u_full = assemble_input(obs[self.echo_idx][self.dec_idx], obs[self.idx["discharge"]], self.exogenous(obs))
        modelled = float(self.meas.do2.predict(np.maximum(x_hat, 1e-3), u_full))
        residual = float(obs[self.idx["DO2"]]) - modelled
        self.do2_offset = self.offset_beta * self.do2_offset + (1 - self.offset_beta) * residual
        return self.do2_offset

    def exogenous(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=float)[self.exo_idx]

    def estimate(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Advance the estimator and the offset-free DO2 correction; returns (x_hat, exogenous inputs, offset).

        Everything the projection needs, without solving the MPC: the
        projection-only controller uses this and leaves the safe action
        uncomputed.
        """
        obs = np.asarray(obs, dtype=float)
        x_hat = self.estimator.update(obs) if self.k > 0 else self.estimator.x.copy()
        offset = self._update_do2_offset(obs, x_hat) if self.k > 0 else 0.0
        return x_hat, self.exogenous(obs), offset

    def act(self, obs: np.ndarray, x_true: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Action for the next plant step from the current observation.

        Returns (plant-unit action, state estimate, diagnostics). The estimate
        is updated with every observation; the MPC is re-solved every
        ``resolve_every`` steps and its plan held in between. ``x_true``
        replaces the estimator (oracle diagnostic): the reduced state is
        taken as given and only the DO2 offset is updated.
        """
        obs = np.asarray(obs, dtype=float)
        if x_true is None:
            x_hat, exo, offset = self.estimate(obs)
        else:
            x_hat = np.maximum(np.asarray(x_true, dtype=float), STATE_FLOOR)
            self.estimator.x = x_hat.copy()
            exo = self.exogenous(obs)
            offset = self._update_do2_offset(obs, x_hat) if self.k > 0 else 0.0
        t_h = float(obs[self.idx["t_h"]])
        u_applied = obs[self.echo_idx]
        if self._last is None or self.k % self.resolve_every == 0:
            self._last = self.mpc.solve(x_hat, t_h, exo, u_applied[self.dec_idx], offset)
        u_next = self._last["u_next"].copy()
        t_next = t_h + STEP_IN_HOURS
        action = physical_to_action(u_next, t_next, self.deviation_max)
        action, proj = self.projection.project(action, x_hat, t_next, exo, offset)
        self.k += 1
        diag = {"mpc_status": self._last["status"], "mpc_wall_s": self._last["wall_s"], "mpc_slack": self._last["slack"],
                "projected": proj["changed"], "projection_status": proj["status"], "u_next": u_next, "do2_offset": offset}
        return action, x_hat, diag
