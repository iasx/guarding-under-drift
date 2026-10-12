"""Closed-loop batches of the controllers the article compares, on the penicillin simulator.

    uv run python experiments/run_hybrid.py --tag co2_sensor --scenarios S1 S5 S5fast \
        --seeds $(seq 2011 2034) --controllers projected24ms --workers 12

Each batch writes its trajectory to ``results/traj_<controller>_<scenario>_<seed>.csv.gz``; the run writes one summary
row per batch to ``results/summary_<tag>.csv``.

Controllers: ``recipe``; ``safe`` (the MPC on the identified model); ``hybrid:convex:nofsm`` (the monitored blend);
``hybrid:fixed:<w>`` (the constant blend); ``policy=<dir>`` (a learned actor on its own, e.g. ``policy=sac_s0``);
and the safety filters below, where ``projected24m:<mg/L>`` and ``projected24ms:<mg/L>`` set the added oxygen margin
(3 mg/L when omitted) and ``projected24c:<scale>`` the feed-bound scale. Guarded filters take further options:
``:on<value>`` for the guard's switching point, ``:bias<value>`` for a soft-sensor offset, and ``:recipe`` for a Simplex
switch that falls back to the recipe (e.g. ``projected24ms:3:on7.0``, ``projected24mx:3:recipe``).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = Path(__file__).parents[1] / "results"
SCENARIOS = {
    "S1": {},
    "S5": {"drift": {"alpha_kla": (100.0, 200.0, 0.4), "mu_P": (100.0, 200.0, 0.7)}},
    "S5fast": {"drift": {"alpha_kla": (100.0, 130.0, 0.4), "mu_P": (100.0, 130.0, 0.7)}},
    # Dose-response: oxygen transfer falling to 0.6, 0.5 or 0.3 of nominal over the gradual ramp.
    **{f"S5k{int(10 * k):02d}": {"drift": {"alpha_kla": (100.0, 200.0, k), "mu_P": (100.0, 200.0, 0.7)}} for k in (0.6, 0.5, 0.3)},
    # Graded CO2 inhibition of half-width 0.1 or 0.5 instead of the simulator's step, nominal and under gradual drift.
    **{f"S1g{int(10 * w):02d}": {"co2_half_width": w} for w in (0.1, 0.5)},
    **{f"S5g{int(10 * w):02d}": {"co2_half_width": w, "drift": {"alpha_kla": (100.0, 200.0, 0.4), "mu_P": (100.0, 200.0, 0.7)}}
       for w in (0.1, 0.5)},
    # The counterfactual plant without the CO2 growth inhibition, nominal and under both drifts.
    "S1noco2": {"co2_off": True},
    "S5noco2": {"co2_off": True, "drift": {"alpha_kla": (100.0, 200.0, 0.4), "mu_P": (100.0, 200.0, 0.7)}},
    "S5fastnoco2": {"co2_off": True, "drift": {"alpha_kla": (100.0, 130.0, 0.4), "mu_P": (100.0, 130.0, 0.7)}},
}
BATCH_END_H = 230.0
FILTER_VARIANTS = {"projected24t": "tightened", "projected24m": "tightened_margin", "projected24c": "tightened_capacity",
                   "projected24s": "tightened_substrate_oracle", "projected24k": "tightened_co2_oracle",
                   "projected24mk": "tightened_margin_co2_oracle", "projected24ms": "tightened_margin_co2_sensor",
                   "projected24mx": "tightened_margin_co2_simplex"}
DEFAULT_PARAMETER = {"projected24c": 0.7}


def trajectory_path(controller: str, scenario_name: str, seed: int) -> Path:
    return OUT / f"traj_{controller.replace(':', '-')}_{scenario_name}_{seed}.csv.gz"


def first_sustained_alarm(after: pd.DataFrame, sustain: int = 5) -> float:
    """Hour of the first non-NORMAL regime of the blend's monitor held for ``sustain`` consecutive steps."""
    alarm = (after["regime"] != "NORMAL").to_numpy()
    t = after["t_h"].to_numpy()
    for i in range(len(alarm) - sustain + 1):
        if alarm[i : i + sustain].all():
            return float(t[i])
    return float("nan")


def onset_hour(spec: dict) -> float | None:
    """Start of the injected drift, or None for a scenario without one."""
    starts = [v[0] for v in spec.get("drift", {}).values()]
    return min(starts) if starts else None


def make_plant(scenario_name: str, seed: int):
    """The plant under a named scenario, reset on ``seed``; returns (plant, obs, info)."""
    from sindy_fno.plant import DriftRamp, PenSimPlant, ScenarioConfig

    spec = SCENARIOS[scenario_name]
    drift = {k: DriftRamp(*v) for k, v in spec.get("drift", {}).items()}
    plant = PenSimPlant(scenario=ScenarioConfig(drift=drift, co2_inhibition_half_width=spec.get("co2_half_width"),
                                                co2_inhibition_off=spec.get("co2_off", False)))
    obs, info = plant.reset(seed=seed)
    return plant, obs, info


def build_controller(controller: str, obs, seed: int, models_dir):
    """``act(obs, info) -> (action, diag)`` for a controller name."""
    from sindy_fno.adaptive.policy import AdaptivePolicy
    from sindy_fno.hybrid.controller import SindyFnoController
    from sindy_fno.hybrid.projected import FilteredActor
    from sindy_fno.safe.branch import SafeBranch

    parts = controller.split(":")
    kind, law = (parts + [None])[:2]
    if kind == "hybrid":
        flags = parts[2:]
        weights = [float(x) for x in flags if x.replace(".", "", 1).isdigit()]
        # "nofsm": the monitor's CRITICAL regime never forces the safe action.
        ctrl = SindyFnoController(models_dir, law=law, fixed_weight=weights[0] if weights else 0.5,
                                  regime_enabled="nofsm" not in flags)
        ctrl.reset(obs, seed=seed)
        return lambda o, i: ctrl.act(o, i)
    if kind.startswith("policy="):
        d = Path(models_dir) / kind[7:]
        sur = d if (d / "ensemble.json").exists() else Path(models_dir) / "fno"
        pol = AdaptivePolicy(sur, d / "agent.pt")
        pol.reset(seed)
        return lambda o, i: (pol.act(o, i), {})
    if kind in FILTER_VARIANTS:
        default = DEFAULT_PARAMETER.get(kind, 3.0)
        # Further options: on<value> moves the CO2 guard's switching point, bias<value> offsets the soft sensor, and
        # recipe makes the Simplex switch fall back to the recipe instead of the MPC.
        opts = parts[2:]
        guard_on = next((float(x[2:]) for x in opts if x.startswith("on")), None)
        sensor_bias = next((float(x[4:]) for x in opts if x.startswith("bias")), 0.0)
        fa = FilteredActor(models_dir, variant=FILTER_VARIANTS[kind], margin=float(law) if law else default, guard_on=guard_on,
                           sensor_bias=sensor_bias, fallback="recipe" if "recipe" in opts else "mpc")
        fa.reset(obs, seed=seed)
        return lambda o, i: fa.act(o, i)
    if kind == "safe":
        safe = SafeBranch(Path(models_dir) / "sindy")
        safe.reset(obs)
        return lambda o, i: (safe.act(o)[0], {"do2_offset": safe.do2_offset})
    if kind == "recipe":
        return lambda o, i: (np.zeros(6), {})
    raise ValueError(f"unknown controller {controller!r}")


def run_one(args: tuple) -> dict:
    controller, scenario_name, seed, models_dir = args
    plant, obs, info = make_plant(scenario_name, seed)
    act = build_controller(controller, obs, seed, models_dir)
    kind = controller.split(":")[0]
    hn = plant.hidden_names
    rows, total, t0 = [], 0.0, time.perf_counter()
    d: dict = {}
    for k in range(plant.max_episode_steps):
        a, d = act(obs, info)
        obs, r, terminated, _, info = plant.step(a)
        total += r
        h = dict(zip(hn, info["hidden"]))
        rows.append({"k": k + 1, "t_h": info["t_h"], "V": h["V"], "DO2": h["DO2"], "S": h["S"], "pH": h["pH"], "reward": r,
                     "X": h["X"], "Viscosity": h["Viscosity"], "CO2_d": h["CO2_d"], "mu_X": h["mu_X_calc"],
                     "regime": d.get("regime", ""), "branch": d.get("branch", ""), "delta_bar": d.get("delta_bar", np.nan),
                     "delta_a": d.get("delta_a", np.nan), "w": d.get("w", np.nan), "projected": d.get("projected", False),
                     "projection_delta": d.get("projection_delta", np.nan), "watchdog": d.get("watchdog", ""),
                     "do2_offset": d.get("do2_offset", np.nan), "co2_est": d.get("co2_est", np.nan)})
        if terminated:
            break
    df = pd.DataFrame(rows)
    df.to_csv(trajectory_path(controller, scenario_name, seed), index=False)
    out = {"controller": controller, "scenario": scenario_name, "seed": seed, "yield_kg": total,
           "max_V": float(df["V"].max()), "steps_V_above_100k": int((df["V"] > 1e5).sum()), "min_DO2": float(df["DO2"].min()),
           "steps_DO2_below_4": int((df["DO2"] < 4).sum()), "max_S": float(df["S"].max()), "min_pH": float(df["pH"].min()),
           "catastrophic": bool(info["catastrophic"]), "wall_s": time.perf_counter() - t0}
    if kind in FILTER_VARIANTS:
        out.update({"projected_steps": int(df["projected"].sum()), "projection_delta_p50": float(df.loc[df["projected"], "projection_delta"].median()),
                    "projection_delta_max": float(df["projection_delta"].max())})
    if kind == "hybrid":
        onset = onset_hour(SCENARIOS[scenario_name])
        after = df[df.t_h >= onset] if onset is not None else df
        out.update({
            "frac_safe": float(df["branch"].isin(["safe", "critical"]).mean()), "frac_blend": float(df["branch"].eq("blend").mean()),
            "frac_adaptive": float(df["branch"].eq("adaptive").mean()), "frac_warning": float(df["regime"].eq("WARNING").mean()),
            "frac_critical": float(df["regime"].eq("CRITICAL").mean()), "projected_steps": int(df["projected"].sum()),
            "watchdog_trips": int((df["watchdog"] != "").sum()), "delta_a_p50": float(df["delta_a"].median()),
            "delta_bar_p50": float(df["delta_bar"].median()), "delta_bar_max": float(df["delta_bar"].max()),
            "feed_clip_steps": int(d.get("feed_clip_steps", 0)), "first_non_normal_after_onset_h": first_sustained_alarm(after),
        })
    return out


def summary_from_trajectory(controller: str, scenario_name: str, seed: int) -> dict:
    """The summary row of a batch already run, rebuilt from its trajectory so an interrupted run can resume.

    A batch that ended in integrator failure stops before the batch end, which stands in for ``catastrophic``; the
    wall time is not recorded in the trajectory."""
    df = pd.read_csv(trajectory_path(controller, scenario_name, seed))
    out = {"controller": controller, "scenario": scenario_name, "seed": seed, "yield_kg": float(df["reward"].sum()),
           "max_V": float(df["V"].max()), "steps_V_above_100k": int((df["V"] > 1e5).sum()), "min_DO2": float(df["DO2"].min()),
           "steps_DO2_below_4": int((df["DO2"] < 4).sum()), "max_S": float(df["S"].max()), "min_pH": float(df["pH"].min()),
           "catastrophic": bool(df["t_h"].iloc[-1] < BATCH_END_H - 0.1), "wall_s": float("nan")}
    if controller.split(":")[0] in FILTER_VARIANTS:
        out.update({"projected_steps": int(df["projected"].sum()), "projection_delta_p50": float(df.loc[df["projected"], "projection_delta"].median()),
                    "projection_delta_max": float(df["projection_delta"].max())})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--scenarios", nargs="+", default=["S1", "S5", "S5fast"], choices=list(SCENARIOS))
    p.add_argument("--controllers", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tag", required=True, help="writes results/summary_<tag>.csv")
    p.add_argument("--resume", action="store_true", help="skip batches whose trajectory exists; summarise them from it")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s, seed, args.models) for c in args.controllers for s in args.scenarios for seed in args.seeds]
    done = [j for j in jobs if args.resume and trajectory_path(*j[:3]).exists()]
    todo = [j for j in jobs if j not in done]
    print(f"{len(todo)} batches to run, {len(done)} summarised from existing trajectories", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = [summary_from_trajectory(*j[:3]) for j in done] + list(ex.map(run_one, todo))
    table = pd.DataFrame(results)
    table.to_csv(OUT / f"summary_{args.tag}.csv", index=False)
    (OUT / f"summary_{args.tag}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(table.groupby(["controller", "scenario"])[["yield_kg", "min_DO2", "steps_DO2_below_4"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
