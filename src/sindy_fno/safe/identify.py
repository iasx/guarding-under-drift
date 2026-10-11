"""Identification run: fit, select, evaluate and report the sparse model and its acceptance check.

    uv run python -m sindy_fno.safe.identify --corpus data/corpus --out models/sindy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.data.corpus import Corpus, load_subset
from sindy_fno.data.subsets import SUBSETS
from sindy_fno.safe.library import (
    EQUATION_TERMS,
    INPUT_NAMES,
    OUTPUT_NAME,
    SAMPLE_FILTERS,
    STATE_NAMES,
)
from sindy_fno.safe.sparse_model import SparseModel, StaticModel, bootstrap_survival, derivatives

# The model is accepted on rollout error at the MPC horizon (24 h) and at 72 h;
# the safe branch re-solves with state feedback every step, so open-loop
# full-batch accuracy is reported but not gated.
G1_NRMSE_24H = {"X": 0.15, "P": 0.15, "V": 0.15}
G1_NRMSE_72H = {"X": 0.30, "P": 0.30, "V": 0.20}
G1_SURVIVAL = 0.8

# Validity region of the reduced model (from the identification report): dissolved oxygen above its inhibition floor and
# feed below the biomass uptake capacity, so substrate stays quasi-steady.
DO2_MIN = 4.0
FEED_CONCENTRATION = {"Fs": 600.0, "Foil": 1000.0}  # g/L in the feeds (IndPenSim)

FIT_KWARGS = {"equation_terms": EQUATION_TERMS, "sample_filters": SAMPLE_FILTERS}


@dataclass
class ReducedEpisode:
    """Reduced-state view of one corpus episode.

    ``U[k]`` is the input applied during the step from ``X[k]`` to ``X[k+1]``;
    ``Y[k]`` (DO2) and ``S[k]`` (substrate) are measured at ``X[k]`` under
    ``U_at[k]``, the input of the step that ended there.
    """

    X: np.ndarray
    U: np.ndarray
    U_at: np.ndarray
    Y: np.ndarray
    S: np.ndarray
    discharge: np.ndarray
    yield_kg: float
    seed: int


def extract(corpus: Corpus) -> list[ReducedEpisode]:
    hidx = [corpus.hidden_names.index(n) for n in [*STATE_NAMES, "S", OUTPUT_NAME]]
    action_idx = [corpus.action_names.index(n) for n in INPUT_NAMES if n in corpus.action_names]
    obs_idx = [corpus.obs_names.index(n) for n in INPUT_NAMES if n not in corpus.action_names]
    n = len(STATE_NAMES)
    episodes = []
    for i in range(corpus.n_episodes):
        L = int(corpus.length[i])
        if L < 3 or corpus.catastrophic[i]:
            continue
        H = corpus.hidden[i, :L][:, hidx]
        inputs = np.concatenate([corpus.action[i, :L][:, action_idx], corpus.obs[i, :L][:, obs_idx]], axis=1)
        episodes.append(ReducedEpisode(
            X=H[:, :n], U=np.vstack([inputs[1:], inputs[-1:]]), U_at=inputs, Y=H[:, n + 1], S=H[:, n],
            discharge=inputs[:, INPUT_NAMES.index("discharge")], yield_kg=float(corpus.yield_kg[i]),
            seed=int(corpus.seed[i]),
        ))
    return episodes


def fit_dynamic(eps: list[ReducedEpisode], threshold: float | list[float], alpha: float) -> SparseModel:
    return SparseModel.fit(
        [e.X[:-1] for e in eps], [e.U[:-1] for e in eps], STEP_IN_HOURS,
        threshold=threshold, alpha=alpha, **FIT_KWARGS,
    )


def fit_static(eps: list[ReducedEpisode], threshold: float, alpha: float) -> StaticModel:
    return StaticModel.fit([e.X for e in eps], [e.U_at for e in eps], [e.Y for e in eps], threshold=threshold, alpha=alpha)


def state_std(eps: list[ReducedEpisode]) -> np.ndarray:
    return np.vstack([e.X for e in eps]).std(axis=0)


def rollout_nrmse(model: SparseModel, eps: list[ReducedEpisode], std: np.ndarray, horizon: int, stride: int) -> np.ndarray:
    """RMSE per state over ``horizon``-step rollouts started every ``stride`` steps, divided by ``std``."""
    sq, n = np.zeros(len(STATE_NAMES)), 0
    for e in eps:
        for s in range(0, len(e.X) - horizon, stride):
            sim = model.simulate(e.X[s], e.U[s : s + horizon], STEP_IN_HOURS)[1:]
            sq += ((sim - e.X[s + 1 : s + horizon + 1]) ** 2).sum(axis=0)
            n += horizon
    return np.sqrt(sq / max(n, 1)) / std


def yield_errors(model: SparseModel, eps: list[ReducedEpisode]) -> np.ndarray:
    """Relative error (%) of the full-batch yield implied by a rollout from the true initial state."""
    ip, iv = STATE_NAMES.index("P"), STATE_NAMES.index("V")
    errs = []
    for e in eps:
        sim = model.simulate(e.X[0], e.U[:-1], STEP_IN_HOURS)
        P, V = sim[:, ip], sim[:, iv]
        pred = P[-1] * V[-1] / 1000 + (e.discharge[1:] * P[1:] * STEP_IN_HOURS / 1000).sum()
        errs.append(100 * (pred - e.yield_kg) / e.yield_kg)
    return np.array(errs)


def r2_derivative(model: SparseModel, eps: list[ReducedEpisode]) -> np.ndarray:
    pred = np.vstack([model.f(e.X[:-1], e.U[:-1]) for e in eps])
    true = np.vstack([derivatives(e.X, STEP_IN_HOURS)[:-1] for e in eps])
    return 1 - ((pred - true) ** 2).sum(axis=0) / ((true - true.mean(axis=0)) ** 2).sum(axis=0)


def select_threshold(train, val, thresholds, alpha, std, horizon=360, stride=120):
    """Per-equation thresholds by coordinate descent on coupled validation rollouts.

    Start from the best global threshold, then for each equation in turn keep
    the sparsest threshold whose coupled ``horizon``-step rollout error (mean
    nRMSE over states) stays within 10 % of the best found for that equation.
    """

    def score(thr):
        m = fit_dynamic(train, thr, alpha)
        return float(rollout_nrmse(m, val, std, horizon, stride).mean()), m.n_active

    rows = []
    global_scores = {}
    for th in thresholds:
        sc, n = score([th] * len(STATE_NAMES))
        global_scores[th] = sc
        rows.append({"stage": "global", "state": "all", "threshold": th, "n_active": n, "val_nrmse": sc})
    chosen = [min(global_scores, key=global_scores.get)] * len(STATE_NAMES)
    for i, name in enumerate(STATE_NAMES):
        trials = []
        for th in thresholds:
            cand = list(chosen)
            cand[i] = th
            sc, n = score(cand)
            trials.append((th, sc, n))
            rows.append({"stage": "per-equation", "state": name, "threshold": th, "n_active": n, "val_nrmse": sc})
        best = min(sc for _, sc, _ in trials)
        ok = [t for t in trials if t[1] <= 1.1 * best]
        chosen[i] = max(ok, key=lambda t: t[0])[0]  # sparsest acceptable = largest threshold
    return chosen, pd.DataFrame(rows)


def validity_region(eps: list[ReducedEpisode]) -> dict:
    """Uptake capacity and feed usage statistics that define the model's validity region.

    Specific substrate uptake ``q = (feed - dS/dt) / X`` is measured where
    substrate is visible and oxygen is healthy; ``q_cap`` is a conservative
    fraction of its median, to be enforced by the MPC as feed/V <= q_cap * X.
    """
    ix = STATE_NAMES.index("X")
    iv = STATE_NAMES.index("V")
    q_all, ratio_all, do2_all = [], [], []
    for e in eps:
        X, V, S, Y = e.X[:, ix], e.X[:, iv], e.S, e.Y
        feed = (FEED_CONCENTRATION["Fs"] * e.U_at[:, 0] + FEED_CONCENTRATION["Foil"] * e.U_at[:, 1]) / V
        dS = np.gradient(S, STEP_IN_HOURS)
        m = (S > 0.1) & (Y >= 8.0)
        q_all.append((feed[m] - dS[m]) / X[m])
        ok = Y >= 8.0
        ratio_all.append(feed[ok] / np.maximum(X[ok], 0.5))
        do2_all.append(Y)
    q = np.concatenate(q_all)
    ratio = np.concatenate(ratio_all)
    do2 = np.concatenate(do2_all)
    q_median = float(np.median(q)) if q.size else float("nan")
    return {
        "do2_min": DO2_MIN,
        "q_median": q_median,
        "q_cap": round(0.75 * q_median, 3),
        "q_quantiles_5_25_75_95": np.percentile(q, [5, 25, 75, 95]).round(3).tolist() if q.size else [],
        "feed_over_X_p50_p90_p99": np.percentile(ratio, [50, 90, 99]).round(3).tolist(),
        "fraction_steps_do2_below_min": float((do2 < DO2_MIN).mean()),
    }


def load_split(corpus_dir: Path) -> dict[str, list[ReducedEpisode]]:
    index = pd.read_csv(corpus_dir / "index.csv")
    out = {"train": [], "val": [], "test": []}
    for name in SUBSETS:
        path = corpus_dir / f"{name}.h5"
        if not path.exists():
            continue
        eps = extract(load_subset(path))
        by_seed = {e.seed: e for e in eps}
        for _, row in index[index.subset == name].iterrows():
            if int(row.seed) in by_seed:
                out[row.split].append(by_seed[int(row.seed)])
    return out


def run(corpus_dir: Path, out_dir: Path, thresholds, alpha: float, n_boot: int, do2_threshold: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    split = load_split(corpus_dir)
    train, val, test = split["train"], split["val"], split["test"]
    std = state_std(train)

    threshold, sweep = select_threshold(train, val, thresholds, alpha, std)
    model = fit_dynamic(train, threshold, alpha)
    do2 = fit_static(train, do2_threshold, alpha)
    survival = bootstrap_survival(
        [e.X[:-1] for e in train], [e.U[:-1] for e in train], STEP_IN_HOURS,
        n_boot=n_boot, threshold=threshold, alpha=alpha, **FIT_KWARGS,
    )

    nrmse_24h = rollout_nrmse(model, test, std, 120, 120)
    nrmse_72h = rollout_nrmse(model, test, std, 360, 120)
    nrmse_full = rollout_nrmse(model, test, std, min(len(e.X) for e in test) - 1, 10**9)
    yerr = yield_errors(model, test)
    r2 = r2_derivative(model, test)
    do2_pred = np.concatenate([do2.predict(e.X, e.U_at) for e in test])
    do2_true = np.concatenate([e.Y for e in test])
    do2_r2 = 1 - ((do2_pred - do2_true) ** 2).sum() / ((do2_true - do2_true.mean()) ** 2).sum()
    do2_rmse = float(np.sqrt(((do2_pred - do2_true) ** 2).mean()))
    region = validity_region(train)

    active_survival = survival[model.coef != 0]
    g1 = {
        "nrmse_24h_ok": bool(all(nrmse_24h[i] <= G1_NRMSE_24H[n] for i, n in enumerate(STATE_NAMES))),
        "nrmse_72h_ok": bool(all(nrmse_72h[i] <= G1_NRMSE_72H[n] for i, n in enumerate(STATE_NAMES))),
        "survival_ok": bool(active_survival.min() >= G1_SURVIVAL) if active_survival.size else False,
    }
    g1["passed"] = all(g1.values())

    metrics = {
        "threshold": threshold, "alpha": alpha, "n_active": model.n_active,
        "episodes": {k: len(v) for k, v in split.items()},
        "state_std": dict(zip(STATE_NAMES, std.tolist())),
        "r2_derivative": dict(zip(STATE_NAMES, r2.tolist())),
        "nrmse_24h": dict(zip(STATE_NAMES, nrmse_24h.tolist())),
        "nrmse_72h": dict(zip(STATE_NAMES, nrmse_72h.tolist())),
        "nrmse_full_batch": dict(zip(STATE_NAMES, nrmse_full.tolist())),
        "yield_error_pct": {"median_abs": float(np.median(np.abs(yerr))), "p90_abs": float(np.percentile(np.abs(yerr), 90)), "mean": float(yerr.mean())},
        "do2": {"threshold": do2_threshold, "r2": float(do2_r2), "rmse": do2_rmse, "n_active": int(np.count_nonzero(do2.coef))},
        "validity_region": region,
        "sweep": sweep.to_dict(orient="records"),
        "g1": g1,
    }
    (out_dir / "model.json").write_text(model.to_json(), encoding="utf-8")
    (out_dir / "do2_model.json").write_text(do2.to_json(), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text(report(model, do2, survival, metrics), encoding="utf-8")
    return metrics


def report(model: SparseModel, do2: StaticModel, survival: np.ndarray, m: dict) -> str:
    lines = ["# Sparse identification report", ""]
    lines += [f"Thresholds per equation {m['threshold']} (alpha {m['alpha']}), {m['n_active']} active terms; episodes {m['episodes']}.", ""]
    lines += ["## Equations", ""] + [f"    {eq}" for eq in model.equations()] + ["", f"    {do2.equation()}", ""]
    lines += ["## Active terms and bootstrap survival", "", "| Equation | Term | Coefficient | Survival |", "|---|---|---|---|"]
    for i, n in enumerate(STATE_NAMES):
        for j, t in enumerate(model.terms):
            if model.coef[i, j] != 0:
                lines.append(f"| d{n}/dt | {t.name} | {model.coef[i, j]:.4g} | {survival[i, j]:.2f} |")
    lines += ["", "## Test metrics", "", "| State | R² (dx/dt) | nRMSE 24 h | nRMSE 72 h | nRMSE full batch |", "|---|---|---|---|---|"]
    for n in STATE_NAMES:
        lines.append(f"| {n} | {m['r2_derivative'][n]:.3f} | {m['nrmse_24h'][n]:.3f} | {m['nrmse_72h'][n]:.3f} | {m['nrmse_full_batch'][n]:.3f} |")
    y = m["yield_error_pct"]
    lines += ["", f"Full-batch open-loop yield error (information only): median |err| {y['median_abs']:.1f} %, p90 {y['p90_abs']:.1f} %, mean {y['mean']:+.1f} %.",
              f"DO2 static model (threshold {m['do2']['threshold']}): R² {m['do2']['r2']:.3f}, RMSE {m['do2']['rmse']:.2f} mg/L, {m['do2']['n_active']} terms.", ""]
    r = m["validity_region"]
    lines += ["## Validity region", "",
              f"- dissolved oxygen ≥ {r['do2_min']} mg/L (fraction of training steps below: {r['fraction_steps_do2_below_min']:.3f})",
              f"- feed/V ≤ q_cap · X with q_cap = {r['q_cap']} g/(g·h) (median specific uptake {r['q_median']:.3f}; quantiles 5/25/75/95 {r['q_quantiles_5_25_75_95']})",
              f"- feed/X actually used in healthy operation: p50/p90/p99 = {r['feed_over_X_p50_p90_p99']}", ""]
    lines += ["## Threshold sweep (validation, coupled 72 h rollouts, mean nRMSE)", "", "| stage | equation varied | threshold | active | nRMSE |", "|---|---|---|---|---|"]
    lines += [f"| {r_['stage']} | {r_['state']} | {r_['threshold']} | {r_['n_active']} | {r_['val_nrmse']:.3f} |" for r_ in m["sweep"]]
    g = m["g1"]
    lines += ["", "## Acceptance", "", f"- 24 h nRMSE within limits {G1_NRMSE_24H}: {g['nrmse_24h_ok']}",
              f"- 72 h nRMSE within limits {G1_NRMSE_72H}: {g['nrmse_72h_ok']}",
              f"- all active terms survive ≥ {G1_SURVIVAL} of bootstrap refits: {g['survival_ok']}",
              f"- **Acceptance {'PASSED' if g['passed'] else 'FAILED'}**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--out", type=Path, default=Path("models/sindy"))
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--do2-threshold", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=20)
    args = p.parse_args(argv)
    metrics = run(args.corpus, args.out, args.thresholds, args.alpha, args.n_boot, args.do2_threshold)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the report uses ≤ and ²; Windows consoles default to cp1252
    print((args.out / "report.md").read_text(encoding="utf-8"))
    print(json.dumps(metrics["g1"]))


if __name__ == "__main__":
    main()
