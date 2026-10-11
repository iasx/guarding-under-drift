"""Safety projection Π_safe onto the one-hour-ahead admissible action set."""

from __future__ import annotations

import casadi as ca
import numpy as np

from sindy_fno.plant.recipe import (
    ACTION_MAX,
    ACTION_MIN,
    ACTION_NAMES,
    action_to_physical,
    physical_to_action,
)
from sindy_fno.safe.measurements import MeasurementModels
from sindy_fno.safe.mpc import DECIDED, MPCConfig, assemble_input, constraint_values, rk4
from sindy_fno.safe.sparse_model import SparseModel


class SafetyProjection:
    """Closest admissible input to a requested one, judged on the sparse model one hour ahead."""

    def __init__(self, model: SparseModel, meas: MeasurementModels, cfg: MPCConfig | None = None, deviation_max: float = 0.5) -> None:
        self.model = model
        self.meas = meas
        self.cfg = cfg or MPCConfig()
        self.deviation_max = deviation_max
        self.dec_idx = [ACTION_NAMES.index(n) for n in DECIDED]
        self.u_lo = ACTION_MIN[self.dec_idx]
        self.u_hi = ACTION_MAX[self.dec_idx]
        self._build()

    def _build(self) -> None:
        opti = ca.Opti()
        u = opti.variable(5)
        x0 = opti.parameter(3)
        u_req = opti.parameter(5)
        disch = opti.parameter(1)
        exo = opti.parameter(3)
        lo = opti.parameter(5)
        hi = opti.parameter(5)
        do2_offset = opti.parameter(1)
        u_full = assemble_input([u[i] for i in range(5)], disch, [exo[i] for i in range(3)])
        f = lambda x, uu: ca.vertcat(*self.model.f(x, uu))
        x1 = rk4(f, x0, u_full, self.cfg.step_h, self.cfg.substeps)
        _gv_now, go, gc, ge = constraint_values(self.model, self.meas, self.cfg, x0, u_full, do2_offset)
        gv_next = self.cfg.v_max - self.cfg.v_margin - x1[2]
        opti.subject_to(go >= 0)
        opti.subject_to(gc >= 0)
        opti.subject_to(ge >= 0)
        opti.subject_to(gv_next >= 0)
        opti.subject_to(opti.bounded(lo, u, hi))
        opti.minimize(ca.sumsqr((u - u_req) / ca.DM(self.u_hi)))
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 200, "sb": "yes"})
        self.opti, self.u, self.x0, self.u_req, self.disch, self.exo = opti, u, x0, u_req, disch, exo
        self.lo, self.hi = lo, hi
        self.do2_offset_p = do2_offset

    def admissible(self, x_hat: np.ndarray, u_phys: np.ndarray, exogenous: np.ndarray, do2_offset: float = 0.0) -> np.ndarray:
        """Constraint headroom (V one hour ahead, DO2, capacity, envelope) for a physical 6-vector at ``x_hat``; all >= 0 means admissible."""
        u_full = assemble_input(u_phys[self.dec_idx], u_phys[ACTION_NAMES.index("discharge")], exogenous)
        x0 = np.maximum(np.asarray(x_hat, float), 1e-3)
        _gv_now, go, gc, ge = constraint_values(self.model, self.meas, self.cfg, x0, u_full, do2_offset)
        # The volume constraint is judged where the input takes the model in one hour, as the QP does; the current-state
        # headroom accepted requests that crossed the margin within the hour and left the QP infeasible when it was finally asked.
        x1 = self.model.simulate(x0, np.asarray(u_full, float)[None, :], self.cfg.step_h, self.cfg.substeps)[-1]
        gv_next = self.cfg.v_max - self.cfg.v_margin - float(x1[2])
        return np.array([gv_next, go, gc, ge], dtype=float)

    def project(self, action: np.ndarray, x_hat: np.ndarray, t_h: float, exogenous: np.ndarray, do2_offset: float = 0.0) -> tuple[np.ndarray, dict]:
        """Project a plant-unit action for the step starting at ``t_h``; returns (action, diagnostics)."""
        u_req = action_to_physical(action, t_h, self.deviation_max)
        if (self.admissible(x_hat, u_req, exogenous, do2_offset) >= -1e-9).all():
            return np.asarray(action, dtype=float), {"changed": False, "status": "admissible"}
        opti = self.opti
        opti.set_value(self.do2_offset_p, float(do2_offset))
        opti.set_value(self.x0, np.maximum(x_hat, 1e-3))
        opti.set_value(self.u_req, u_req[self.dec_idx])
        opti.set_value(self.disch, u_req[ACTION_NAMES.index("discharge")])
        opti.set_value(self.exo, exogenous)
        # Stay inside what the plant's action space can express around the recipe.
        recipe = action_to_physical(np.zeros(len(ACTION_NAMES)), t_h, self.deviation_max)[self.dec_idx]
        lo = np.maximum(recipe * (1 - self.deviation_max), self.u_lo)
        hi = np.maximum(np.minimum(recipe * (1 + self.deviation_max), self.u_hi), lo)
        opti.set_value(self.lo, lo)
        opti.set_value(self.hi, hi)
        opti.set_initial(self.u, np.clip(u_req[self.dec_idx], lo, hi))
        u_out = u_req.copy()
        try:
            sol = opti.solve()
            u_out[self.dec_idx] = np.asarray(sol.value(self.u)).ravel()
            status = "projected"
        except RuntimeError:
            # No admissible representable input: the most conservative representable feeds (feeds to their lower bound),
            # while aeration and head pressure go to their upper bound, which is the conservative side for the oxygen constraint.
            fallback = lo.copy()
            for name in ("Fg", "pressure"):
                fallback[DECIDED.index(name)] = hi[DECIDED.index(name)]
            u_out[self.dec_idx] = fallback
            status = "fallback"
        return physical_to_action(u_out, t_h, self.deviation_max), {"changed": True, "status": status}
