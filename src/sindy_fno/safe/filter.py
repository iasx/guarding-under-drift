"""A receding-horizon predictive safety filter on the identified model: the closest input to a request that keeps the
model inside its validity region over the MPC's full horizon (Wabersich and Zeilinger's construct, on the safe branch's
own constraints), against the one-hour projection of ``projection.py``."""

from __future__ import annotations

import casadi as ca
import numpy as np

from sindy_fno.plant.recipe import ACTION_NAMES
from sindy_fno.safe.mpc import DECIDED, X_SCALE, SafeMPC, assemble_input, constraint_values, rk4


class SafetyFilterMPC(SafeMPC):
    """Same dynamics, constraints, slacks and envelope as ``SafeMPC``; the objective is distance to the requested first
    input instead of penicillin mass, with the recipe anchor kept only on the tail of the plan."""

    def _build(self) -> None:
        N, cfg = self.N, self.cfg
        opti = ca.Opti()
        Z = opti.variable(3, N + 1)
        U = opti.variable(5, N)
        s_v = opti.variable(1, N + 1)
        s_o = opti.variable(1, N)
        s_c = opti.variable(1, N)
        z0 = opti.parameter(3)
        u_ref = opti.parameter(5, N)
        u_req = opti.parameter(5)
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
            opti.subject_to(ge >= 0)
            prev = u_prev if k == 0 else U[:, k - 1]
            du = (U[:, k] - prev) / ca.DM(self.u_scale)
            if k == 0:
                J += 100.0 * ca.sumsqr((U[:, 0] - u_req) / ca.DM(self.u_scale))   # the filter's purpose: stay close to the request
            else:
                dr = (U[:, k] - u_ref[:, k]) / ca.DM(self.u_scale)
                J += cfg.lambda_ref * ca.sumsqr(dr) + cfg.lambda_move * ca.sumsqr(du)   # a sane tail keeps the horizon feasible
            J += cfg.slack_weight * (s_o[k] + s_c[k] + s_v[k] / 1000.0)
        gv_end = cfg.v_max - cfg.v_margin - Z[2, N] * X_SCALE[2]
        opti.subject_to(gv_end + s_v[N] >= 0)
        J += cfg.slack_weight * s_v[N] / 1000.0
        hours_left = opti.parameter(1)
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
        self.u_req_p = u_req

    def filter(self, u_req_phys: np.ndarray, x_hat: np.ndarray, t_h: float, exogenous: np.ndarray, u_prev: np.ndarray, do2_offset: float = 0.0) -> dict:
        """Closest admissible-over-the-horizon input to the requested physical 6-vector; falls back to the MPC's own shift rule."""
        req = np.asarray(u_req_phys, float)[self.dec_idx]
        self.opti.set_value(self.u_req_p, np.clip(req, self.u_lo, self.u_hi))
        out = self.solve(x_hat, t_h, exogenous, u_prev, do2_offset)
        out["changed"] = bool(np.max(np.abs(out["u_next"][self.dec_idx] - req) / self.u_scale) > 1e-4)
        return out


class TightenedSafetyFilter(SafeMPC):
    """The predictive safety filter with the standard remedies for a minimal-intervention filter.

    - **Constraint tightening against model error.** The oxygen margin grows with prediction lead from the plain
      filter's value to ``do2_margin + do2_tighten`` at 24 h, in proportion to the square root of the lead (the
      oxygen model's residual is 1.5 mg/L). The volume margin grows linearly by ``v_tighten`` over the first day.
    - **Effectively hard constraints.** The slack weight is 100 times the plain filter's, so a violation is never
      traded for closeness to the request.
    - **Horizon to batch end** (``to_batch_end``). Hourly stages for the first day, then ``coarse_step_h`` stages
      to the end of the batch, with the tail anchored to the recipe as the backup policy. For a finite batch this
      takes the place of a terminal safe set: the filter accepts a request only if a recipe-anchored continuation
      exists that keeps the model inside its constraints until the batch ends.
    - **Margin matched to the blend** (``do2_extra``, ``err_gain``). An oxygen margin added at every stage, constant
      or proportional to the magnitude of the running oxygen offset. Under gradual drift the blends keep the plant
      about 3 mg/L further from the floor than these filters do; this asks whether that margin alone, given to a
      filter, holds the constraint.
    - **Margin on the feed bound** (``cap_scale``). The uptake-capacity bound stands in for substrate, which the model
      does not contain; scaling it below 1 tightens the constraint through which an overfed, unmodelled substrate
      would enter, instead of the oxygen constraint.
    """

    def __init__(self, model, meas, cfg=None, to_batch_end: bool = False, coarse_step_h: float = 6.0,
                 do2_tighten: float = 3.0, v_tighten: float = 2_000.0, slack_scale: float = 100.0,
                 do2_extra: float = 0.0, err_gain: float = 0.0, cap_scale: float = 1.0) -> None:
        self.to_batch_end = to_batch_end
        self.coarse_step_h = coarse_step_h
        self.do2_tighten = do2_tighten
        self.v_tighten = v_tighten
        self.slack_scale = slack_scale
        self.do2_extra = do2_extra
        self.err_gain = err_gain
        self.cap_scale = cap_scale
        super().__init__(model, meas, cfg)

    def stage_grid(self, t0: float) -> np.ndarray:
        cfg = self.cfg
        end = cfg.batch_end_h
        fine_end = min(t0 + cfg.horizon_h, end)
        ts = list(np.arange(t0, fine_end - 1e-9, cfg.step_h))
        if self.to_batch_end:
            t = fine_end
            while t < end - 1e-9:
                ts.append(t)
                t += self.coarse_step_h
        ts.append(end if self.to_batch_end else fine_end)
        return np.array(ts)

    def _max_stages(self) -> int:
        cfg = self.cfg
        n = round(cfg.horizon_h / cfg.step_h)
        if self.to_batch_end:
            n += int(np.ceil((cfg.batch_end_h - cfg.horizon_h) / self.coarse_step_h)) + 1
        return n

    def _build(self) -> None:
        cfg = self.cfg
        N = self._max_stages()
        self.N = N
        opti = ca.Opti()
        Z = opti.variable(3, N + 1)
        U = opti.variable(5, N)
        s_v = opti.variable(1, N + 1)
        s_o = opti.variable(1, N)
        s_c = opti.variable(1, N)
        z0 = opti.parameter(3)
        h = opti.parameter(1, N)
        u_ref = opti.parameter(5, N)
        u_req = opti.parameter(5)
        disch = opti.parameter(1, N)
        exo = opti.parameter(3)
        u_prev = opti.parameter(5)
        u_lo = opti.parameter(5, N)
        u_hi = opti.parameter(5, N)
        tight_o = opti.parameter(1, N)
        tight_v = opti.parameter(1, N + 1)
        do2_offset = opti.parameter(1)
        xs = ca.DM(X_SCALE)
        w_slack = cfg.slack_weight * self.slack_scale
        opti.subject_to(Z[:, 0] == z0)
        J = 0
        for k in range(N):
            x = Z[:, k] * xs
            u_full = assemble_input([U[i, k] for i in range(5)], disch[k], [exo[i] for i in range(3)])
            x_next = rk4(self._f, x, u_full, h[k], cfg.substeps)
            opti.subject_to(Z[:, k + 1] == x_next / xs)
            gv, go, gc, ge = constraint_values(self.model, self.meas, cfg, x, u_full, do2_offset)
            gc = gc - (1.0 - self.cap_scale) * cfg.q_cap * ca.fmax(x[0], cfg.cap_x_min)
            opti.subject_to(go - tight_o[k] + s_o[k] >= 0)
            opti.subject_to(gc + s_c[k] >= 0)
            opti.subject_to(gv - tight_v[k] + s_v[k] >= 0)
            opti.subject_to(ge >= 0)
            prev = u_prev if k == 0 else U[:, k - 1]
            du = (U[:, k] - prev) / ca.DM(self.u_scale)
            if k == 0:
                J += 100.0 * ca.sumsqr((U[:, 0] - u_req) / ca.DM(self.u_scale))
            else:
                dr = (U[:, k] - u_ref[:, k]) / ca.DM(self.u_scale)
                J += h[k] * (cfg.lambda_ref * ca.sumsqr(dr) + cfg.lambda_move * ca.sumsqr(du))
            J += w_slack * (s_o[k] + s_c[k] + s_v[k] / 1000.0)
        gv_end = cfg.v_max - cfg.v_margin - Z[2, N] * X_SCALE[2]
        opti.subject_to(gv_end - tight_v[N] + s_v[N] >= 0)
        J += w_slack * s_v[N] / 1000.0
        for k in range(N):
            opti.subject_to(opti.bounded(u_lo[:, k], U[:, k], u_hi[:, k]))
        for k in range(N + 1):
            opti.subject_to(Z[:, k] >= 1e-3 / xs)
        opti.subject_to(s_v >= 0)
        opti.subject_to(s_o >= 0)
        opti.subject_to(s_c >= 0)
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False}, dict(cfg.ipopt))
        self.opti, self.Z, self.U, self.z0, self.u_ref, self.disch, self.exo, self.u_prev = opti, Z, U, z0, u_ref, disch, exo, u_prev
        self.u_lo_p, self.u_hi_p = u_lo, u_hi
        self.h_p, self.tight_o_p, self.tight_v_p = h, tight_o, tight_v
        self.do2_offset_p = do2_offset
        self.slacks = (s_v, s_o, s_c)
        self.u_req_p = u_req
        self.hours_left = opti.parameter(1)

    def filter(self, u_req_phys: np.ndarray, x_hat: np.ndarray, t_h: float, exogenous: np.ndarray, u_prev: np.ndarray, do2_offset: float = 0.0) -> dict:
        """Closest input to the request that keeps the tightened constraints feasible over the horizon."""
        import time as _time
        from itertools import pairwise

        from pensimpy.constants import STEP_IN_HOURS

        from sindy_fno.plant.recipe import recipe_values

        cfg, N = self.cfg, self.N
        grid = self.stage_grid(t_h)
        n = len(grid) - 1
        h = np.zeros(N)
        h[:n] = np.diff(grid)
        starts = np.full(N, grid[-1])
        starts[:n] = grid[:-1]
        mids = starts + h / 2
        u_ref = np.array([[recipe_values(t)[nm] for t in mids] for nm in DECIDED])
        disch = np.zeros(N)
        for k, (a, b) in enumerate(pairwise(grid)):
            pts = np.arange(a, b, STEP_IN_HOURS) + STEP_IN_HOURS / 2
            disch[k] = float(np.mean([recipe_values(t)["discharge"] for t in pts])) if len(pts) else 0.0
        lo, hi = self.representable(u_ref)
        lead = starts - t_h + h
        tight_o = self.do2_tighten * np.sqrt(np.clip(lead / cfg.horizon_h, 0.0, 1.0)) + self.do2_extra + self.err_gain * abs(float(do2_offset))
        tight_o[n:] = 0.0
        ends = np.append(starts[0], starts + h)
        tight_v = self.v_tighten * np.clip((ends - t_h) / cfg.horizon_h, 0.0, 1.0)
        req = np.asarray(u_req_phys, float)[self.dec_idx]
        o = self.opti
        o.set_value(self.u_req_p, np.clip(req, self.u_lo, self.u_hi))
        o.set_value(self.z0, np.maximum(x_hat, 1e-3) / X_SCALE)
        o.set_value(self.h_p, h)
        o.set_value(self.u_ref, u_ref)
        o.set_value(self.disch, disch)
        o.set_value(self.exo, exogenous)
        o.set_value(self.u_lo_p, lo)
        o.set_value(self.u_hi_p, hi)
        o.set_value(self.u_prev, np.clip(u_prev, self.u_lo, self.u_hi))
        o.set_value(self.tight_o_p, tight_o)
        o.set_value(self.tight_v_p, tight_v)
        o.set_value(self.do2_offset_p, float(do2_offset))
        o.set_value(self.hours_left, 0.0)
        if self._plan is not None and self._plan.shape[1] == N:
            o.set_initial(self.U, np.clip(np.column_stack([self._plan[:, 1:], self._plan[:, -1:]]), lo, hi))
        else:
            o.set_initial(self.U, np.clip(u_ref, lo, hi))
        o.set_initial(self.Z, np.tile((np.maximum(x_hat, 1e-3) / X_SCALE)[:, None], (1, N + 1)))
        for s in self.slacks:
            o.set_initial(s, 0.0)
        t0 = _time.perf_counter()
        try:
            sol = o.solve()
            Uv = np.atleast_2d(sol.value(self.U))
            status = "ok"
            slack = float(sum(np.sum(sol.value(s)) for s in self.slacks))
            slack_o = np.atleast_1d(sol.value(self.slacks[1]))[:n]
        except RuntimeError:
            status = "failed:" + str(o.debug.stats().get("return_status", "unknown"))
            slack = float("nan")
            slack_o = np.full(n, np.nan)
            Uv = np.column_stack([self._plan[:, 1:], self._plan[:, -1:]]) if self._plan is not None else np.clip(u_ref, lo, hi)
        self._plan = Uv
        u_next = np.zeros(len(ACTION_NAMES))
        u_next[self.dec_idx] = Uv[:, 0]
        u_next[ACTION_NAMES.index("discharge")] = recipe_values(t_h + STEP_IN_HOURS)["discharge"]
        # Per-stage oxygen slack and stage lengths, for diagnosing where the plan predicts violations.
        self.last = {"slack_o": slack_o, "h": h[:n], "req": req, "u0": Uv[:, 0].copy()}
        return {"u_next": u_next, "status": status, "wall_s": _time.perf_counter() - t0, "slack": slack,
                "changed": bool(np.max(np.abs(Uv[:, 0] - req) / self.u_scale) > 1e-4)}
