"""Fit the dissolved-CO2 soft sensor the guarded safety filter uses, from measurements a plant already has.

    uv run python experiments/fit_co2_sensor.py

Least squares on the corpus training batches: dissolved CO2 against off-gas CO2 (alone and times head pressure, as
Henry's law suggests), head pressure, temperature, the CO2 evolution rate and aeration, all as observed with sensor
noise. Reports the error on the corpus test batches, overall and near the growth-inhibition switch, and writes the
coefficients to ``models/hybrid/co2_sensor.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sindy_fno.safe.co2_sensor import FEATURES, features

CORPUS = Path("data/corpus")
OUT = Path("models/hybrid/co2_sensor.json")
SWITCH = 7.57


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    index = pd.read_csv(CORPUS / "index.csv")
    X, y = [], []
    for subset, rows in index[index.split == split].groupby("subset"):
        with h5py.File(CORPUS / f"{subset}.h5", "r") as f:
            obs_names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["obs_names"]]
            hidden_names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["hidden_names"]]
            for ep, length in zip(rows.episode, rows.length, strict=True):
                obs = f["obs"][ep, :length]
                X.append(features(obs, obs_names))
                y.append(f["hidden"][ep, :length, hidden_names.index("CO2_d")])
    return np.vstack(X), np.concatenate(y)


def main() -> None:
    X, y = load("train")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt, yt = load("test")
    err = Xt @ coef - yt
    near = yt > 7.0
    report = {"rmse": float(np.sqrt(np.mean(err ** 2))), "p95_abs": float(np.percentile(np.abs(err), 95)),
              "rmse_near_switch": float(np.sqrt(np.mean(err[near] ** 2))), "p95_abs_near_switch": float(np.percentile(np.abs(err[near]), 95)),
              "bias_near_switch": float(err[near].mean())}
    OUT.write_text(json.dumps({"features": FEATURES, "coef": coef.tolist(), "test": report, "switch": SWITCH}, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
