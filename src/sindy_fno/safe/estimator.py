"""Unscented Kalman filter state estimator for the safe branch."""

from __future__ import annotations

import numpy as np
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.plant.channels import ECHO, LAB, RESP
from sindy_fno.safe.library import INPUT_NAMES, STATE_NAMES
from sindy_fno.safe.measurements import MeasurementModels
from sindy_fno.safe.sparse_model import STATE_FLOOR, SparseModel

OBS_NAMES = RESP + ECHO + LAB


def sigma_points(x: np.ndarray, P: np.ndarray, alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0):
    n = len(x)
    lam = alpha**2 * (n + kappa) - n
    P = 0.5 * (P + P.T) + 1e-12 * np.eye(n)
    try:
        L = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh((n + lam) * P)
        L = V @ np.diag(np.sqrt(np.maximum(w, 1e-12)))
    pts = np.vstack([x, x + L.T, x - L.T])
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    return pts, wm, wc


def ukf_predict(x, P, f, Q):
    pts, wm, wc = sigma_points(x, P)
    prop = np.array([f(p) for p in pts])
    xm = wm @ prop
    d = prop - xm
    return xm, (d.T * wc) @ d + Q


def ukf_update(x, P, h, y, R):
    pts, wm, wc = sigma_points(x, P)
    ys = np.array([h(p) for p in pts])
    ym = wm @ ys
    dy = ys - ym
    dx = pts - x
    Pyy = (dy.T * wc) @ dy + R
    Pxy = (dx.T * wc) @ dy
    K = Pxy @ np.linalg.inv(Pyy)
    return x + K @ (np.asarray(y) - ym), P - K @ Pyy @ K.T


class SafeStateEstimator:
    """UKF on the reduced state driven by the plant's observation vector.

    Continuous measurements: vessel weight, carbon evolution rate, dissolved
    oxygen through the fitted measurement models. Sparse measurements: the
    biomass and penicillin assays, applied whenever a new report arrives
    (the age channel drops). ``q_scale`` inflates the one-step process noise
    because model error between assays is bias, not white noise; without it
    the filter under-weights the assays.
    """

    def __init__(
        self,
        model: SparseModel,
        meas: MeasurementModels,
        dt: float = STEP_IN_HOURS,
        q_scale: float = 3.0,
        assay_rel_sigma: float = 0.02,
        assay_lag_rel: float = 0.02,
    ) -> None:
        self.model = model
        self.meas = meas
        self.dt = dt
        self.idx = {n: i for i, n in enumerate(OBS_NAMES)}
        self.input_idx = [self.idx[n] for n in INPUT_NAMES]
        self.Q = np.diag((q_scale * np.asarray(meas.process_std)) ** 2)
        self.R = np.diag([meas.noise_std[k] ** 2 for k in ("Wt", "CER", "DO2")])
        self.assay_rel = float(np.hypot(assay_rel_sigma, assay_lag_rel))
        self.x = np.zeros(len(STATE_NAMES))
        self.P = np.eye(len(STATE_NAMES))
        self._last_lab_age = np.inf

    def inputs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=float)[self.input_idx]

    def reset(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        x_lab, p_lab, wt = obs[self.idx["X_lab"]], obs[self.idx["P_lab"]], obs[self.idx["Wt"]]
        wt_coef = dict(zip([t.name for t in self.meas.wt.terms], self.meas.wt.coef))
        v0 = (wt - wt_coef.get("1", 0.0)) / wt_coef["V"]
        self.x = np.maximum(np.array([x_lab, p_lab, v0]), STATE_FLOOR)
        self.P = np.diag([(0.5 * self.x[0] + 0.1) ** 2, (0.5 * self.x[1] + 0.1) ** 2, (0.05 * self.x[2]) ** 2])
        self._last_lab_age = obs[self.idx["lab_age_h"]]
        return self.x.copy()

    def update(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        u = self.inputs(obs)
        self.x, self.P = ukf_predict(
            self.x, self.P, lambda p: self.model.simulate(np.maximum(p, STATE_FLOOR), u[None, :], self.dt)[-1], self.Q
        )
        y = obs[[self.idx["Wt"], self.idx["CER"], self.idx["DO2"]]]
        self.x, self.P = ukf_update(self.x, self.P, lambda p: self.meas.predict(np.maximum(p, STATE_FLOOR), u), y, self.R)
        lab_age = obs[self.idx["lab_age_h"]]
        if lab_age < self._last_lab_age:
            y_lab = obs[[self.idx["X_lab"], self.idx["P_lab"]]]
            R_lab = np.diag((self.assay_rel * np.maximum(y_lab, 0.05)) ** 2)
            self.x, self.P = ukf_update(self.x, self.P, lambda p: p[:2], y_lab, R_lab)
        self._last_lab_age = lab_age
        self.x = np.maximum(self.x, STATE_FLOOR)
        return self.x.copy()
