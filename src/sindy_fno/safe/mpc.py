"""Constrained MPC on the identified sparse model, CasADi + IPOPT."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.plant.recipe import ACTION_MAX, ACTION_MIN, ACTION_NAMES, recipe_values
from sindy_fno.safe.library import INPUT_NAMES
from sindy_fno.safe.measurements import MeasurementModels
from sindy_fno.safe.sparse_model import SparseModel

# Inputs the MPC decides; discharge follows the operator schedule, and
# Fpaa, Fa, Fb are the plant's own PID flows (exogenous, held).
DECIDED = ["Fs", "Foil", "Fg", "pressure", "Fw"]
EXOGENOUS = ["Fpaa", "Fa", "Fb"]
FEED_CONCENTRATION = {"Fs": 600.0, "Foil": 1000.0}
X_SCALE = np.array([10.0, 10.0, 1e5])


@dataclass
class MPCConfig:
    horizon_h: float = 24.0
    step_h: float = 1.0
    substeps: int = 4
    v_max: float = 100_000.0
    v_margin: float = 2_000.0
    do2_min: float = 4.0
    do2_margin: float = 1.4
    q_cap: float = 0.306
    # Below this biomass the culture is still consuming its initial substrate
    # charge (batch phase); the uptake-capacity rule is for the fed-batch phase.
    cap_x_min: float = 3.0
    batch_end_h: float = 230.0
    # A terminal value for biomass (production to go) drove the controller into
    # the oxygen-limited collapse in closed loop: it rewards growth, the sparse
    # models extrapolate linearly, and the real plant does not. Off by default.
    terminal_biomass_value: bool = False
    feed_max: float | None = None   # g/(L h) input envelope; None = from the measurement models
    deviation_max: float = 0.5      # the plant's action envelope around the recipe
    lambda_ref: float = 10.0
    lambda_move: float = 10.0
    slack_weight: float = 1e3
    ipopt: dict = field(default_factory=lambda: {"print_level": 0, "max_iter": 300, "tol": 1e-6, "sb": "yes"})


def rk4(f, x, u, h: float, substeps: int):
    hs = h / substeps
    for _ in range(substeps):
        k1 = f(x, u)
        k2 = f(x + 0.5 * hs * k1, u)
        k3 = f(x + 0.5 * hs * k2, u)
        k4 = f(x + hs * k3, u)
        x = x + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def assemble_input(decided, discharge, exogenous):
    """Full input vector in INPUT_NAMES order from the decided, scheduled and exogenous parts (CasADi or numpy)."""
    parts = {n: decided[i] for i, n in enumerate(DECIDED)}
    parts["discharge"] = discharge
    parts.update({n: exogenous[i] for i, n in enumerate(EXOGENOUS)})
    vals = [parts[n] for n in INPUT_NAMES]
    return ca.vertcat(*vals) if any(isinstance(v, ca.MX | ca.SX) for v in vals) else np.array(vals, dtype=float)


def feed_per_volume(u_full, x):
    return (FEED_CONCENTRATION["Fs"] * u_full[0] + FEED_CONCENTRATION["Foil"] * u_full[1]) / x[2]


def _fmax(a, b):
    return np.maximum(a, b) if isinstance(a, np.ndarray | float | int) else ca.fmax(a, b)


def constraint_values(model: SparseModel, meas: MeasurementModels, cfg: MPCConfig, x, u_full, do2_offset=0.0):
    """(V headroom, DO2 headroom, capacity headroom, envelope headroom); non-negative means satisfied.

    ``do2_offset`` is the measured-minus-modelled DO2 at the current operating
    point, so the constraint is offset-free: it trusts the model for the
    change with the input, and the sensor for the level.
    """
    v = x[2]
    do2 = meas.do2.predict(x, u_full) + do2_offset
    feed = feed_per_volume(u_full, x)
    feed_max = cfg.feed_max if cfg.feed_max is not None else meas.feed_max
    return (cfg.v_max - cfg.v_margin - v, do2 - cfg.do2_min - cfg.do2_margin, cfg.q_cap * _fmax(x[0], cfg.cap_x_min) - feed, feed_max - feed)


class SafeMPC:
    """Receding-horizon controller maximising penicillin mass inside the model's validity region."""

    def __init__(self, model: SparseModel, meas: MeasurementModels, cfg: MPCConfig | None = None) -> None:
        self.model = model
        self.meas = meas
        self.cfg = cfg or MPCConfig()
        self.N = round(self.cfg.horizon_h / self.cfg.step_h)
        self.dec_idx = [ACTION_NAMES.index(n) for n in DECIDED]
        self.u_lo = ACTION_MIN[self.dec_idx]
        self.u_hi = ACTION_MAX[self.dec_idx]
        self.u_scale = self.u_hi
        # Coefficient of X in dP/dt: g/L penicillin per g/L biomass per hour.
        names = [t.name for t in model.terms]
        self.production_rate = float(max(model.coef[1, names.index("X")], 0.0))
        self._plan = None
        self._last_status = ""
        self._build()

    # ---------------------------------------------------------------- build

    def _f(self, x, u):
        return ca.vertcat(*self.model.f(x, u))

    def _build(self) -> None:
        N, cfg = self.N, self.cfg
        opti = ca.Opti()
        Z = opti.variable(3, N + 1)            # scaled states
        U = opti.variable(5, N)
        s_v = opti.variable(1, N + 1)
        s_o = opti.variable(1, N)
        s_c = opti.variable(1, N)
        z0 = opti.parameter(3)
        u_ref = opti.parameter(5, N)
        disch = opti.parameter(1, N)
        exo = opti.parameter(3)
        u_prev = opti.parameter(5)
        u_lo = opti.parameter(5, N)
        u_hi = opti.parameter(5, N)
        do2_offset = opti.parameter(1)
        xs = ca.DM(X_SCALE)

        opti.subject_to(Z[:, 0] == z0)
        J = 0
        for k in range(N):
            x = Z[:, k] * xs
            u_full = assemble_input([U[i, k] for i in range(5)], disch[k], [exo[i] for i in range(3)])
            x_next = rk4(self._f, x, u_full, cfg.step_h, cfg.substeps)
            opti.subject_to(Z[:, k + 1] == x_next / xs)
            gv, go, gc, ge = constraint_values(self.model, self.meas, cfg, x, u_full, do2_offset)
            opti.subject_to(go + s_o[k] >= 0)
            opti.subject_to(gc + s_c[k] >= 0)
            opti.subject_to(gv + s_v[k] >= 0)
            opti.subject_to(ge >= 0)          # input envelope: hard, inputs can always satisfy it
            # Moves are penalised, not bounded: a hard rate bound relative to the
            # recipe is infeasible whenever the recipe itself jumps, and the plant
            # applies its own per-step rate limit anyway.
            prev = u_prev if k == 0 else U[:, k - 1]
            du = (U[:, k] - prev) / ca.DM(self.u_scale)
            dr = (U[:, k] - u_ref[:, k]) / ca.DM(self.u_scale)
            J += cfg.lambda_ref * ca.sumsqr(dr) + cfg.lambda_move * ca.sumsqr(du)
            J += cfg.slack_weight * (s_o[k] + s_c[k] + s_v[k] / 1000.0)
        gv_end = cfg.v_max - cfg.v_margin - Z[2, N] * X_SCALE[2]
        opti.subject_to(gv_end + s_v[N] >= 0)
        J += cfg.slack_weight * s_v[N] / 1000.0
        J -= Z[1, N] * X_SCALE[1] * Z[2, N] * X_SCALE[2] / 1000.0   # penicillin mass at the horizon end [kg]
        # Value to go of biomass: the identified production rate per unit biomass
        # times the batch time left after the horizon, so a 24 h objective does
        # not starve growth whose product arrives later.
        hours_left = opti.parameter(1)
        if cfg.terminal_biomass_value:
            J -= self.production_rate * Z[0, N] * X_SCALE[0] * hours_left * Z[2, N] * X_SCALE[2] / 1000.0
        # Inputs must stay inside what the plant's action space can express
        # (recipe +- deviation_max, within actuator bounds). CasADi reads a matrix
        # inequality as a definiteness constraint, so bound column by column.
        for k in range(N):
            opti.subject_to(opti.bounded(u_lo[:, k], U[:, k], u_hi[:, k]))
        for k in range(N + 1):
            opti.subject_to(Z[:, k] >= 1e-3 / xs)
        opti.subject_to(s_v >= 0)
        opti.subject_to(s_o >= 0)
        opti.subject_to(s_c >= 0)
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False}, dict(self.cfg.ipopt))
        self.opti, self.Z, self.U, self.z0, self.u_ref, self.disch, self.exo, self.u_prev = opti, Z, U, z0, u_ref, disch, exo, u_prev
        self.u_lo_p, self.u_hi_p = u_lo, u_hi
        self.hours_left = hours_left
        self.do2_offset_p = do2_offset
        self.slacks = (s_v, s_o, s_c)

    # ---------------------------------------------------------------- solve

    def references(self, t_h: float):
        """Recipe values of the decided inputs and the discharge schedule per MPC step from ``t_h``."""
        ts = t_h + self.cfg.step_h * (np.arange(self.N) + 0.5)
        u_ref = np.array([[recipe_values(t)[n] for t in ts] for n in DECIDED])
        disch = np.array([recipe_values(t)["discharge"] for t in ts])
        return u_ref, disch

    def representable(self, u_ref: np.ndarray):
        """Lower and upper bounds the plant's action space can express around the recipe."""
        d = self.cfg.deviation_max
        lo = np.maximum(u_ref * (1 - d), self.u_lo[:, None])
        hi = np.minimum(u_ref * (1 + d), self.u_hi[:, None])
        return lo, np.maximum(hi, lo)

    def solve(self, x_hat: np.ndarray, t_h: float, exogenous: np.ndarray, u_prev: np.ndarray, do2_offset: float = 0.0) -> dict:
        """Plan from ``x_hat`` at ``t_h``; returns the physical 6-vector for the next plant step and diagnostics.

        ``exogenous`` = [Fpaa, Fa, Fb] held over the horizon; ``u_prev`` = the
        decided inputs applied last (physical). On solver failure the previous
        plan is shifted by one step.
        """
        u_ref, disch = self.references(t_h)
        opti = self.opti
        opti.set_value(self.z0, np.maximum(x_hat, 1e-3) / X_SCALE)
        opti.set_value(self.u_ref, u_ref)
        opti.set_value(self.disch, disch)
        opti.set_value(self.exo, exogenous)
        lo, hi = self.representable(u_ref)
        opti.set_value(self.u_lo_p, lo)
        opti.set_value(self.u_hi_p, hi)
        opti.set_value(self.u_prev, np.clip(u_prev, self.u_lo, self.u_hi))
        opti.set_value(self.hours_left, max(self.cfg.batch_end_h - (t_h + self.cfg.horizon_h), 0.0))
        opti.set_value(self.do2_offset_p, float(do2_offset))
        if self._plan is not None:
            opti.set_initial(self.U, np.column_stack([self._plan[:, 1:], self._plan[:, -1:]]))
        else:
            opti.set_initial(self.U, u_ref)
        opti.set_initial(self.Z, np.tile((np.maximum(x_hat, 1e-3) / X_SCALE)[:, None], (1, self.N + 1)))
        for s in self.slacks:
            opti.set_initial(s, 0.0)
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            U = np.atleast_2d(sol.value(self.U))
            status = "ok"
            slack = float(sum(np.sum(sol.value(s)) for s in self.slacks))
        except RuntimeError:
            status = "failed:" + str(opti.debug.stats().get("return_status", "unknown"))
            slack = float("nan")
            U = np.column_stack([self._plan[:, 1:], self._plan[:, -1:]]) if self._plan is not None else u_ref
        wall = time.perf_counter() - t0
        self._plan = U
        u_next = np.zeros(len(ACTION_NAMES))
        u_next[self.dec_idx] = U[:, 0]
        u_next[ACTION_NAMES.index("discharge")] = recipe_values(t_h + STEP_IN_HOURS)["discharge"]
        return {"u_next": u_next, "plan": U, "status": status, "wall_s": wall, "slack": slack, "u_ref": u_ref}

    def reset(self) -> None:
        self._plan = None
