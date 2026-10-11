"""Sparse dynamic and algebraic models fitted with pysindy's STLSQ on a mechanistic library."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pysindy as ps

from sindy_fno.safe.library import (
    DO2_TERMS,
    DYNAMIC_TERMS,
    INPUT_NAMES,
    OUTPUT_NAME,
    STATE_NAMES,
    TERM_REGISTRY,
    Term,
    evaluate,
    variables,
)

STATE_FLOOR = 1e-3  # PenSimPy clamps states at this value for numerical stability


def _stlsq_row(theta: np.ndarray, target: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    """STLSQ for one target on unit-RMS features and target.

    In these units a coefficient is the fraction of the target's RMS that one
    RMS of the feature contributes, so ``threshold`` is a scale-free
    importance cut, and ``alpha`` is a per-sample ridge weight (pysindy's raw
    penalty is scaled by the row count) that damps the coefficient splits
    between near-collinear terms such as Fs and Fs/V. pysindy's own column
    normalisation uses unit norm, which grows with the sample count and makes
    the threshold meaningless.
    """
    f_scale = np.sqrt((theta**2).mean(axis=0))
    f_scale[f_scale == 0] = 1.0
    t_scale = float(np.sqrt((target**2).mean())) or 1.0
    opt = ps.STLSQ(threshold=float(threshold), alpha=alpha * theta.shape[0], normalize_columns=False)
    opt.fit(theta / f_scale, (target / t_scale)[:, None])
    c = np.atleast_2d(opt.coef_)[0].copy()
    c[np.abs(c) < threshold] = 0.0
    return c * t_scale / f_scale


def _stlsq(theta: np.ndarray, targets: np.ndarray, threshold, alpha: float) -> np.ndarray:
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), (targets.shape[1],))
    return np.vstack([_stlsq_row(theta, targets[:, j], thr[j], alpha) for j in range(targets.shape[1])])


def derivatives(X: np.ndarray, dt: float) -> np.ndarray:
    return ps.FiniteDifference(order=2)._differentiate(X, dt)


def _format(coef_row: np.ndarray, terms: list[Term]) -> str:
    parts = []
    for c, t in zip(coef_row, terms):
        if c != 0:
            sign = "-" if c < 0 else ("+" if parts else "")
            parts.append(f"{sign} {abs(c):.4g} {t.name}".strip())
    return " ".join(parts) if parts else "0"


@dataclass
class SparseModel:
    """ẋ = Θ(x, u)·Ξᵀ over the reduced state, fitted per equation."""

    terms: list[Term]
    coef: np.ndarray  # (n_states, n_terms)
    threshold: list[float]  # one per equation
    alpha: float
    state_names: list[str] = field(default_factory=lambda: list(STATE_NAMES))
    input_names: list[str] = field(default_factory=lambda: list(INPUT_NAMES))

    @classmethod
    def fit(
        cls,
        X_list: list[np.ndarray],
        U_list: list[np.ndarray],
        dt: float,
        terms: list[Term] | None = None,
        threshold: float | list[float] = 0.05,
        alpha: float = 1e-2,
        equation_terms: dict[str, list[str]] | None = None,
        sample_filters: dict[str, Callable] | None = None,
    ) -> SparseModel:
        """Fit one STLSQ problem per state equation.

        ``equation_terms`` restricts each equation to a subset of term names;
        ``sample_filters`` maps a state name to a predicate on the variable
        dictionary selecting the rows used for that equation.
        """
        terms = list(terms or DYNAMIC_TERMS)
        names = [t.name for t in terms]
        vs = [variables(X, U) for X, U in zip(X_list, U_list)]
        theta = np.vstack([evaluate(terms, v) for v in vs])
        xdot = np.vstack([derivatives(X, dt) for X in X_list])
        thr = np.broadcast_to(np.asarray(threshold, dtype=float), (xdot.shape[1],)).tolist()
        coef = np.zeros((xdot.shape[1], len(terms)))
        for i, state in enumerate(STATE_NAMES[: xdot.shape[1]]):
            cols = np.arange(len(terms))
            if equation_terms and state in equation_terms:
                cols = np.array([names.index(n) for n in equation_terms[state] if n in names])
            rows = np.ones(theta.shape[0], dtype=bool)
            if sample_filters and state in sample_filters:
                rows = np.concatenate([np.asarray(sample_filters[state](v), dtype=bool) for v in vs])
            coef[i, cols] = _stlsq_row(theta[rows][:, cols], xdot[rows, i], thr[i], alpha)
        return cls(terms=terms, coef=coef, threshold=thr, alpha=alpha)

    def theta(self, x, u):
        return evaluate(self.terms, variables(x, u))

    def f(self, x, u):
        """State derivative; numpy (n, 4) in → (n, 4) out, 1-D in → 1-D out, CasADi in → list out."""
        th = self.theta(x, u)
        if isinstance(th, np.ndarray):
            return th @ self.coef.T
        return [sum(c * t for c, t in zip(row, th) if c != 0) for row in self.coef]

    def simulate(self, x0: np.ndarray, U: np.ndarray, dt: float, substeps: int = 4) -> np.ndarray:
        """RK4 rollout; ``U[k]`` is the input held during the step from ``X[k]`` to ``X[k+1]``.

        Returns ``X`` of shape (len(U) + 1, n_states) starting at ``x0``.
        """
        x = np.maximum(np.asarray(x0, dtype=float).reshape(1, -1), STATE_FLOOR)
        out = [x[0].copy()]
        h = dt / substeps
        for u in U:
            u2 = np.asarray(u, dtype=float).reshape(1, -1)
            for _ in range(substeps):
                k1 = self.f(x, u2)
                k2 = self.f(x + 0.5 * h * k1, u2)
                k3 = self.f(x + 0.5 * h * k2, u2)
                k4 = self.f(x + h * k3, u2)
                x = np.maximum(x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), STATE_FLOOR)
            out.append(x[0].copy())
        return np.array(out)

    def equations(self) -> list[str]:
        return [f"d{n}/dt = {_format(row, self.terms)}" for n, row in zip(self.state_names, self.coef)]

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.coef))

    def to_json(self) -> str:
        return json.dumps({
            "terms": [t.name for t in self.terms], "coef": self.coef.tolist(),
            "threshold": list(self.threshold), "alpha": self.alpha,
            "state_names": self.state_names, "input_names": self.input_names,
        }, indent=1)

    @classmethod
    def from_json(cls, data: str | dict) -> SparseModel:
        d = json.loads(data) if isinstance(data, str) else data
        return cls(
            terms=[TERM_REGISTRY[n] for n in d["terms"]], coef=np.asarray(d["coef"], dtype=float),
            threshold=list(np.atleast_1d(d["threshold"]).tolist()), alpha=d["alpha"],
            state_names=list(d["state_names"]), input_names=list(d["input_names"]),
        )


@dataclass
class StaticModel:
    """Algebraic output y = Θ(x, u)·ξ (the quasi-steady DO2 balance)."""

    terms: list[Term]
    coef: np.ndarray  # (n_terms,)
    threshold: float
    alpha: float
    output_name: str = OUTPUT_NAME

    @classmethod
    def fit(
        cls,
        X_list: list[np.ndarray],
        U_list: list[np.ndarray],
        Y_list: list[np.ndarray],
        terms: list[Term] | None = None,
        threshold: float = 0.05,
        alpha: float = 1e-2,
    ) -> StaticModel:
        terms = list(terms or DO2_TERMS)
        theta = np.vstack([evaluate(terms, variables(X, U)) for X, U in zip(X_list, U_list)])
        y = np.concatenate([np.asarray(Y).reshape(-1) for Y in Y_list])[:, None]
        coef = _stlsq(theta, y, threshold, alpha)[0]
        return cls(terms=terms, coef=coef, threshold=threshold, alpha=alpha)

    def predict(self, x, u):
        th = evaluate(self.terms, variables(x, u))
        if isinstance(th, np.ndarray):
            return th @ self.coef
        return sum(c * t for c, t in zip(self.coef, th) if c != 0)

    def equation(self) -> str:
        return f"{self.output_name} = {_format(self.coef, self.terms)}"

    def to_json(self) -> str:
        return json.dumps({
            "terms": [t.name for t in self.terms], "coef": self.coef.tolist(),
            "threshold": self.threshold, "alpha": self.alpha, "output_name": self.output_name,
        }, indent=1)

    @classmethod
    def from_json(cls, data: str | dict) -> StaticModel:
        d = json.loads(data) if isinstance(data, str) else data
        return cls(
            terms=[TERM_REGISTRY[n] for n in d["terms"]], coef=np.asarray(d["coef"], dtype=float),
            threshold=d["threshold"], alpha=d["alpha"], output_name=d["output_name"],
        )


def bootstrap_survival(X_list, U_list, dt, n_boot=20, seed=0, **fit_kwargs) -> np.ndarray:
    """Fraction of episode-resampled refits in which each coefficient is non-zero."""
    rng = np.random.default_rng(seed)
    n = len(X_list)
    counts = None
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = SparseModel.fit([X_list[i] for i in idx], [U_list[i] for i in idx], dt, **fit_kwargs)
        active = (m.coef != 0).astype(float)
        counts = active if counts is None else counts + active
    return counts / n_boot
