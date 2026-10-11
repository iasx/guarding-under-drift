"""Observation layout of the plant."""

from __future__ import annotations

import math

# Plant response: online sensors, then the flows commanded by the plant's own
# pH / temperature / PAA controllers. These respond to the state and belong in
# the mismatch signal.
SENSORS = ["pH", "T", "DO2", "O2", "CO2outgas", "Wt", "OUR", "CER"]
PID_FLOWS = ["Fa", "Fb", "Fc", "Fh", "Fpaa"]
RESP = SENSORS + PID_FLOWS

# Echo: the agent's own applied actions and batch time. Excluded from the
# mismatch signal because any model predicts them trivially.
ECHO = ["Fs", "Foil", "Fg", "pressure", "discharge", "Fw", "t_h"]

# Sparse, delayed laboratory assays, forward-held, plus their age.
LAB = ["P_lab", "X_lab", "PAA_lab", "NH3_lab", "Visc_lab", "lab_age_h"]
LAB_SOURCE = {"P_lab": "P", "X_lab": "X", "PAA_lab": "PAA", "NH3_lab": "NH3", "Visc_lab": "Viscosity"}

# Ground truth exposed through ``info`` for the oracle controller and metrics.
HIDDEN = [
    "S", "X", "P", "V", "PAA", "NH3", "Viscosity", "DO2", "pH", "T", "O2", "CO2outgas",
    "CO2_d", "Wt", "a0", "a1", "a3", "a4", "Culture_age", "mu_P_calc", "mu_X_calc",
]

# Additive sensor noise, physical units, at noise_scale = 1, roughly 1-2 % of
# each channel's nominal range: pH 6-7, T 298 K, DO2 9-16 mg/L, O2 fraction
# 0.18-0.22, CO2 0.1-2 %, Wt 62-100 t, OUR and CER of order 1.
DEFAULT_SIGMA = {
    "pH": 0.02,
    "T": 0.1,
    "DO2": 0.2,
    "O2": 0.001,
    "CO2outgas": 0.02,
    "Wt": 100.0,
    "OUR": 0.02,
    "CER": 0.02,
}


def as_ph(value: float) -> float:
    """PenSimPy stores H+ concentration until the batch ends; pH itself is > 1."""
    if value <= 0:
        return 0.0
    return -math.log10(value) if value < 1 else value


def read(x, name: str, k: int) -> float:
    """Channel ``name`` after step ``k`` (the batch object stores step k at index k-1)."""
    value = getattr(x, name).y[k - 1]
    return as_ph(value) if name == "pH" else float(value)
