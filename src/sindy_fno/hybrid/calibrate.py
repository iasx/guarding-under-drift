"""Calibration of the monitor and arbiter thresholds, and a check of drift detection.

    uv run python -m sindy_fno.hybrid.calibrate --corpus data/corpus --models models --out models/hybrid

1. Nominal δ̄ distribution from teacher-forced one-step surrogate predictions
   on the validation episodes → δ*, δ_lo, δ_hi, k, nominal false-alarm rates.
2. Shadow runs of the full controller (recipe acting) on nominal and
   drifted evaluation seeds → ε from the nominal Δa distribution, and the detection check:
   separability of δ̄ under drift, detection delay, false alarms.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sindy_fno.adaptive.ensemble import SurrogateEnsemble
from sindy_fno.adaptive.train import load_corpus
from sindy_fno.adaptive.windows import N_RESP, WindowArrays
from sindy_fno.hybrid.monitor import NORMAL, Thresholds, ewma

warnings.filterwarnings("ignore")

DRIFT = {"alpha_kla": (100.0, 200.0, 0.4), "mu_P": (100.0, 200.0, 0.7)}
DRIFT_ONSET_H = 100.0
DRIFT_SETTLED_H = 130.0
WARMUP_H = 8.0          # the runtime window is left-padded before it fills; excluded from the statistics
G5_AUC_MIN = 0.8
G5_FALSE_ALARMS_MAX = 1.0   # WARNING entries per nominal episode
# Nominal-δ̄ quantiles defining δ*, δ_lo, δ_hi. δ_lo at q95 would put 5 % of
# nominal steps in WARNING by construction (two entries per episode observed);
# q99 / q99.9 keep the false-alarm rate within the detection check at the cost of ~20 h of delay.
QUANTILES = (90.0, 99.0, 99.9)


def nominal_delta(ens: SurrogateEnsemble, arrays: WindowArrays, th: Thresholds, stride: int = 5) -> list[dict]:
    """δ and δ̄ series per validation episode from hindsight rollouts (offline prior, behaviour-policy actions)."""
    H, w = th.horizon, torch.as_tensor(th.channel_weights(), dtype=torch.float32, device=ens.device)
    series = []
    with torch.no_grad():
        for e, L in enumerate(arrays.lengths):
            ts = list(range(ens.window - 1 + H, int(L) - 1, stride))
            deltas = []
            for i in range(0, len(ts), 128):
                chunk = ts[i : i + 128]
                x = torch.stack([torch.from_numpy(arrays.window_at(e, t - H)) for t in chunk]).to(ens.device)
                F = torch.stack([torch.from_numpy(np.ascontiguousarray(arrays.feeds[e, t - H + 1 : t + 1])) for t in chunk]).to(ens.device)
                out = ens.rollout(x, F)
                truth = torch.from_numpy(np.stack([arrays.obs[e, t, :N_RESP] for t in chunk])).to(ens.device)
                deltas.append(((out["resp"][:, -1] - truth).abs() * w).sum(dim=1).cpu().numpy())
            d = np.concatenate(deltas)
            series.append({"t_h": np.array(ts) * 0.2, "delta": d, "delta_bar": ewma(d, th.beta**stride)})
    return series


def quantile_thresholds(delta_bar_pooled: np.ndarray, th: Thresholds, quantiles: tuple = QUANTILES) -> Thresholds:
    delta_star, delta_lo, delta_hi = (float(np.percentile(delta_bar_pooled, q)) for q in quantiles)
    k = math.log(9.0) / max(delta_hi - delta_star, 1e-6)      # w(δ_hi) = 0.1
    return Thresholds(**{**asdict(th), "delta_star": delta_star, "delta_lo": delta_lo, "delta_hi": delta_hi, "k": k})


def prior_stats(series: list[dict], th: Thresholds, quantiles: tuple = QUANTILES) -> dict:
    pooled = np.concatenate([s["delta_bar"][s["t_h"] >= WARMUP_H] for s in series])
    prior = quantile_thresholds(pooled, th, quantiles)
    return {"quantiles": {p: float(np.percentile(pooled, p)) for p in (50, 90, 95, 99, 99.5)}, "max": float(pooled.max()),
            "episodes": len(series), "hours": float(sum((s["t_h"] >= WARMUP_H).sum() for s in series) * 0.2 * 5),
            "thresholds": {"delta_star": prior.delta_star, "delta_lo": prior.delta_lo, "delta_hi": prior.delta_hi}}


def shadow_run(args: tuple) -> dict:
    """One calibration episode; ``acting`` is "recipe" (shadow mode) or "hybrid" (the controller acts with ``th_json``)."""
    scenario, seed, models_dir, th_json, acting = args
    from sindy_fno.hybrid.controller import SindyFnoController
    from sindy_fno.plant import DriftRamp, PenSimPlant, ScenarioConfig

    drift = {k: DriftRamp(*v) for k, v in DRIFT.items()} if scenario == "S5" else {}
    plant = PenSimPlant(scenario=ScenarioConfig(drift=drift))
    obs, info = plant.reset(seed=seed)
    ctrl = SindyFnoController(models_dir, Thresholds.from_json(th_json), law="gated", shadow=acting == "recipe")
    ctrl.reset(obs, seed=seed)
    rows = []
    for _ in range(plant.max_episode_steps):
        a, d = ctrl.act(obs, info)
        rows.append((info["t_h"], d["delta"], d["delta_bar"], d["delta_a"], d["regime"]))
        obs, _, terminated, _, info = plant.step(a)
        if terminated:
            break
    t, delta, delta_bar, delta_a, regime = (np.array(c) for c in zip(*rows))
    return {"scenario": scenario, "seed": seed, "t_h": t.tolist(), "delta": delta.tolist(), "delta_bar": delta_bar.tolist(),
            "delta_a": delta_a.tolist(), "regime": regime.tolist()}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a drift δ̄ exceeds a nominal δ̄ (Mann–Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    order = np.argsort(allv)
    sorted_v = allv[order]
    r = np.empty_like(ranks)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def regimes_offline(delta_bar: np.ndarray, th: Thresholds) -> np.ndarray:
    """Regime sequence a monitor with ``th`` would have produced on a recorded δ̄ series."""
    from sindy_fno.hybrid.monitor import RegimeFSM

    fsm = RegimeFSM(th)
    return np.array([fsm.step(float(v)) for v in delta_bar])


def analyse_shadow(runs: list[dict], th: Thresholds, quantiles: tuple = QUANTILES) -> tuple[Thresholds, dict]:
    """Thresholds from the nominal shadow runs (where they will be used), then the detection check on the drifted ones.

    δ̄ is recomputed from the recorded δ with ``th.beta`` (shadow actions do
    not depend on the monitor, so β can be chosen after the runs).
    """
    for r in runs:
        r["delta_bar"] = ewma(np.array(r["delta"]), th.beta).tolist()
    nominal = [r for r in runs if r["scenario"] == "S1"]
    drift = [r for r in runs if r["scenario"] == "S5"]
    da = np.concatenate([np.array(r["delta_a"])[np.array(r["t_h"]) >= WARMUP_H] for r in nominal])
    nom_db = np.concatenate([np.array(r["delta_bar"])[np.array(r["t_h"]) >= WARMUP_H] for r in nominal])
    th = quantile_thresholds(nom_db, Thresholds(**{**asdict(th), "eps": float(np.percentile(da, 90))}), quantiles)
    drift_db = np.concatenate([np.array(r["delta_bar"])[np.array(r["t_h"]) >= DRIFT_SETTLED_H] for r in drift])
    # Regimes are re-derived offline with the final thresholds.
    for r in runs:
        r["regime"] = regimes_offline(np.array(r["delta_bar"]), th).tolist()
    false_alarms = []
    for r in nominal:
        t, reg = np.array(r["t_h"]), np.array(r["regime"])
        m = t >= WARMUP_H
        warn = reg[m] != NORMAL
        false_alarms.append(int(np.sum(warn[1:] & ~warn[:-1]) + warn[0]))
    delays, missed = [], 0
    for r in drift:
        t, reg = np.array(r["t_h"]), np.array(r["regime"])
        after = t >= DRIFT_ONSET_H
        idx = np.flatnonzero(after & (reg != NORMAL))
        # first WARNING sustained for 5 steps after onset
        found = None
        for i in idx:
            if i + 5 <= len(reg) and (reg[i : i + 5] != NORMAL).all():
                found = t[i]
                break
        if found is None:
            missed += 1
        else:
            delays.append(found - DRIFT_ONSET_H)
    fallback = [float(np.mean(np.array(r["regime"])[np.array(r["t_h"]) >= DRIFT_SETTLED_H] != NORMAL)) for r in drift]
    stats = {
        "eps": th.eps, "delta_a_quantiles": {p: float(np.percentile(da, p)) for p in (50, 90, 99)},
        "nominal_delta_bar_quantiles": {p: float(np.percentile(nom_db, p)) for p in (50, 95, 99)},
        "drift_delta_bar_quantiles": {p: float(np.percentile(drift_db, p)) for p in (50, 95, 99)},
        "auc": auc(drift_db, nom_db),
        # Matched windows: drift steps after the ramp against nominal steps of the same batch hours, so the batch-phase
        # component of delta_bar (late-batch nominal drift of the surrogate) does not count as detection.
        "auc_matched": auc(drift_db, np.concatenate([np.array(r["delta_bar"])[np.array(r["t_h"]) >= DRIFT_SETTLED_H] for r in nominal])),
        "auc_placebo": auc(np.concatenate([np.array(r["delta_bar"])[np.array(r["t_h"]) >= DRIFT_SETTLED_H] for r in nominal]),
                           np.concatenate([np.array(r["delta_bar"])[(np.array(r["t_h"]) >= WARMUP_H) & (np.array(r["t_h"]) < DRIFT_ONSET_H)] for r in nominal])),
        "false_alarms_per_nominal_episode": float(np.mean(false_alarms)),
        "detection_delay_h": {"values": delays, "median": float(np.median(delays)) if delays else float("nan")},
        "missed_detections": missed, "fraction_non_normal_after_settled": fallback,
        "critical_reached_in_drift": [bool((np.array(r["regime"]) == "CRITICAL").any()) for r in drift],
    }
    stats["g5"] = {"auc_ok": bool(stats["auc_matched"] >= G5_AUC_MIN), "false_alarm_ok": bool(stats["false_alarms_per_nominal_episode"] <= G5_FALSE_ALARMS_MAX)}
    stats["g5"]["passed"] = all(stats["g5"].values())
    return th, stats


def report(th: Thresholds, prior: dict, shadow: dict, quantiles: tuple = QUANTILES, acting: str = "recipe") -> str:
    q = prior["quantiles"]
    qs, ql, qh = quantiles
    lines = ["# Monitor and arbiter calibration", "", f"Calibration runs with the **{acting}** acting.", "",
             (f"Hindsight horizon {th.horizon} steps ({th.horizon * 0.2:.1f} h); channel weights {th.weights} (others 1); β {th.beta} "
              f"(half-life {math.log(2) / -math.log(th.beta):.1f} steps); τ_dwell / τ_recover {th.tau_dwell} / {th.tau_recover} steps."), "",
             "## Offline prior (validation episodes, behaviour-policy actions)", "",
             (f"δ̄ over {prior['episodes']} episodes ({prior['hours']:.0f} h, stride 5, warm-up excluded): q50 {q[50]:.3f}, q90 {q[90]:.3f}, "
              f"q95 {q[95]:.3f}, q99 {q[99]:.3f}, max {prior['max']:.3f}; implied thresholds δ* {prior['thresholds']['delta_star']:.3f}, "
              f"δ_lo {prior['thresholds']['delta_lo']:.3f}, δ_hi {prior['thresholds']['delta_hi']:.3f}."), "",
             f"## Deployment thresholds (nominal runs, {acting} acting)", "",
             "| Threshold | Value | Rule |", "|---|---|---|",
             f"| δ* | {th.delta_star:.3f} | nominal q{qs:g} (w = 0.5) |", f"| δ_lo | {th.delta_lo:.3f} | nominal q{ql:g} (WARNING) |",
             f"| δ_hi | {th.delta_hi:.3f} | nominal q{qh:g} (CRITICAL after τ_dwell) |", f"| k | {th.k:.2f} | w(δ_hi) = 0.1 |",
             f"| ε | {th.eps:.3f} | q90 of nominal Δa |", "",
             f"Nominal Δa quantiles q50/q90/q99: {shadow['delta_a_quantiles'][50]:.3f} / {shadow['delta_a_quantiles'][90]:.3f} / {shadow['delta_a_quantiles'][99]:.3f}.",
             (f"Nominal δ̄ q50/q95/q99: {shadow['nominal_delta_bar_quantiles'][50]:.3f} / {shadow['nominal_delta_bar_quantiles'][95]:.3f} / {shadow['nominal_delta_bar_quantiles'][99]:.3f}; "
              f"drift δ̄ (after {DRIFT_SETTLED_H:.0f} h) q50/q95/q99: {shadow['drift_delta_bar_quantiles'][50]:.3f} / {shadow['drift_delta_bar_quantiles'][95]:.3f} / {shadow['drift_delta_bar_quantiles'][99]:.3f}."), "",
             "## Drift detection (drift shadow runs)", "",
             f"- AUC(δ̄ drift vs nominal) = {shadow['auc']:.3f}",
             f"- WARNING entries per nominal episode = {shadow['false_alarms_per_nominal_episode']:.2f}",
             f"- detection delay from onset (first WARNING sustained 5 steps): {[round(v, 1) for v in shadow['detection_delay_h']['values']]} h, median {shadow['detection_delay_h']['median']:.1f} h; missed {shadow['missed_detections']}",
             f"- fraction of non-NORMAL steps after settling, per drift run: {[round(v, 2) for v in shadow['fraction_non_normal_after_settled']]}",
             f"- CRITICAL reached in drift runs: {shadow['critical_reached_in_drift']}", "",
             (f"**Detection** AUC ≥ {G5_AUC_MIN}: {shadow['g5']['auc_ok']}; false alarms ≤ {G5_FALSE_ALARMS_MAX} per episode: {shadow['g5']['false_alarm_ok']} → "
              f"**{'PASSED' if shadow['g5']['passed'] else 'FAILED'}**"), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--out", type=Path, default=Path("models/hybrid"))
    p.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    p.add_argument("--beta", type=float, default=0.98)
    p.add_argument("--quantiles", type=float, nargs=3, default=list(QUANTILES), metavar=("Q_STAR", "Q_LO", "Q_HI"))
    p.add_argument("--reanalyse", action="store_true", help="reuse the shadow runs and prior in --out; only redo the thresholds and the detection check")
    p.add_argument("--acting", choices=("recipe", "hybrid"), default="recipe",
                   help="who acts in the calibration runs: the recipe (shadow mode) or the hybrid with --thresholds")
    p.add_argument("--thresholds", type=Path, default=None, help="thresholds the hybrid uses when --acting hybrid (default models/hybrid/thresholds.json)")
    p.add_argument("--tau-dwell", type=int, default=10)
    p.add_argument("--tau-recover", type=int, default=30)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--do2-weight", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Greek letters in the report; Windows consoles default to cp1252

    quantiles = tuple(args.quantiles)
    th0 = Thresholds(beta=args.beta, tau_dwell=args.tau_dwell, tau_recover=args.tau_recover, horizon=args.horizon, weights={"DO2": args.do2_weight})
    if args.acting == "hybrid":
        acting_th = Thresholds.load(args.thresholds or args.models / "hybrid" / "thresholds.json")
        th0 = Thresholds(**{**asdict(acting_th), "beta": th0.beta, "tau_dwell": th0.tau_dwell, "tau_recover": th0.tau_recover})
    if args.reanalyse:
        prior = json.loads((args.out / "calibration.json").read_text(encoding="utf-8"))["prior"]
        runs = json.loads((args.out / "shadow_runs.json").read_text(encoding="utf-8"))
    else:
        ens = SurrogateEnsemble.load(args.models / "fno", args.device)
        corpus, splits = load_corpus(args.corpus)
        val = WindowArrays(corpus, ens.norm, splits["val"], ens.window, horizon=1)
        prior = prior_stats(nominal_delta(ens, val, th0), th0, quantiles)
        print(f"offline prior thresholds: {prior['thresholds']}")
        jobs = [(s, seed, args.models, th0.to_json(), args.acting) for s in ("S1", "S5") for seed in args.seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            runs = list(ex.map(shadow_run, jobs))
        (args.out / "shadow_runs.json").write_text(json.dumps(runs), encoding="utf-8")
    th, shadow = analyse_shadow(runs, th0, quantiles)
    (args.out / "thresholds.json").write_text(th.to_json(), encoding="utf-8")
    (args.out / "calibration.json").write_text(json.dumps({"thresholds": asdict(th), "prior": prior, "shadow": shadow, "quantiles": quantiles, "acting": args.acting,
                                                           "acting_thresholds": asdict(th0) if args.acting == "hybrid" else None}, indent=1), encoding="utf-8")
    text = report(th, prior, shadow, quantiles, args.acting)
    (args.out / "calibration.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
