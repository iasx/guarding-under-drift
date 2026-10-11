"""Mechanistic candidate library for the safe branch.

Terms are plain-arithmetic functions of a variable dictionary so they
evaluate identically on numpy arrays and on CasADi symbols.

The differential reduced state is biomass, penicillin and volume. Substrate
is not a state: inside the declared validity region (dissolved oxygen above
its floor, feed below the biomass uptake capacity) it is quasi-steady at
zero, and outside that region the plant enters an oxygen-limited cascade
that a sparse model cannot follow. The safe controller keeps the plant
inside the region by constraint instead of modelling the cascade.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

STATE_NAMES = ["X", "P", "V"]
INPUT_NAMES = ["Fs", "Foil", "Fg", "pressure", "discharge", "Fw", "Fpaa", "Fa", "Fb"]
OUTPUT_NAME = "DO2"


@dataclass(frozen=True)
class Term:
    name: str
    fn: Callable[[dict[str, Any]], Any]


def _col(a, i: int):
    if isinstance(a, np.ndarray) and a.ndim == 2:
        return a[:, i]
    return a[i]


def variables(x, u) -> dict[str, Any]:
    """Named columns of state ``x`` and input ``u`` plus the total liquid inflow."""
    v = {name: _col(x, i) for i, name in enumerate(STATE_NAMES)}
    v.update({name: _col(u, i) for i, name in enumerate(INPUT_NAMES)})
    v["F_in"] = v["Fs"] + v["Foil"] + v["Fw"] + v["Fpaa"] + v["Fa"] + v["Fb"]
    return v


def _one(v):
    x = v["X"]
    return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0 + 0.0 * x


DYNAMIC_TERMS: list[Term] = [
    Term("X", lambda v: v["X"]),
    Term("P", lambda v: v["P"]),
    Term("V", lambda v: v["V"]),
    Term("X^2", lambda v: v["X"] * v["X"]),
    Term("X*P", lambda v: v["X"] * v["P"]),
    Term("X*F_in/V", lambda v: v["X"] * v["F_in"] / v["V"]),
    Term("P*F_in/V", lambda v: v["P"] * v["F_in"] / v["V"]),
    Term("Fs/V", lambda v: v["Fs"] / v["V"]),
    Term("Foil/V", lambda v: v["Foil"] / v["V"]),
    Term("Fs", lambda v: v["Fs"]),
    Term("Foil", lambda v: v["Foil"]),
    Term("Fw", lambda v: v["Fw"]),
    Term("Fpaa", lambda v: v["Fpaa"]),
    Term("Fa", lambda v: v["Fa"]),
    Term("Fb", lambda v: v["Fb"]),
    Term("discharge", lambda v: v["discharge"]),
]

# Quasi-steady oxygen balance: DO2 ~ DO*(pressure) - OUR / kLa, where the
# uptake rate follows biomass maintenance and feed-driven growth and kLa
# follows aeration and vessel loading.
DO2_TERMS: list[Term] = [
    Term("1", _one),
    Term("pressure", lambda v: v["pressure"]),
    Term("X", lambda v: v["X"]),
    Term("Fg", lambda v: v["Fg"]),
    Term("Fg*pressure", lambda v: v["Fg"] * v["pressure"]),
    Term("X/Fg", lambda v: v["X"] / v["Fg"]),
    Term("X^2/Fg", lambda v: v["X"] * v["X"] / v["Fg"]),
    Term("V", lambda v: v["V"]),
    Term("X*V/Fg", lambda v: v["X"] * v["V"] / v["Fg"]),
    Term("Fs/V", lambda v: v["Fs"] / v["V"]),
    Term("Foil/V", lambda v: v["Foil"] / v["V"]),
    Term("Fs/(V*Fg)", lambda v: v["Fs"] / (v["V"] * v["Fg"])),
    Term("Foil/(V*Fg)", lambda v: v["Foil"] / (v["V"] * v["Fg"])),
]

# Soft-sensor libraries for the estimator: vessel weight follows volume;
# carbon evolution follows biomass respiration and feed-driven growth.
WT_TERMS: list[Term] = [Term("1", _one), Term("V", lambda v: v["V"])]
CER_TERMS: list[Term] = [
    Term("1", _one),
    Term("X", lambda v: v["X"]),
    Term("X^2", lambda v: v["X"] * v["X"]),
    Term("Fs/V", lambda v: v["Fs"] / v["V"]),
    Term("Foil/V", lambda v: v["Foil"] / v["V"]),
]

TERM_REGISTRY: dict[str, Term] = {t.name: t for t in [*DYNAMIC_TERMS, *DO2_TERMS, *WT_TERMS, *CER_TERMS]}

# Terms each balance may draw on. Biological rates see concentrations and
# feed per volume only; the volume balance is bookkeeping over flows.
EQUATION_TERMS: dict[str, list[str]] = {
    "X": ["X", "X^2", "X*F_in/V", "Fs/V", "Foil/V"],
    "P": ["P", "X", "X*P", "X^2", "P*F_in/V"],
    "V": ["V", "Fs", "Foil", "Fw", "Fpaa", "Fa", "Fb", "discharge"],
}

# Row predicates per equation (none needed for the three-state model; kept
# so the fitting interface stays general).
SAMPLE_FILTERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def evaluate(terms: list[Term], v: dict[str, Any]):
    """Feature matrix Θ (numpy, n × L) or a list of CasADi expressions."""
    values = [t.fn(v) for t in terms]
    if isinstance(values[0], np.ndarray):
        return np.stack(values, axis=-1)
    return values
