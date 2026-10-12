"""The article's descriptive numbers, from the results: Tables 1 and 2, the CO2 analysis, the collapse pathway, the
soft sensor's closed-loop error, the Simplex switch's hand-overs, fast-drift severity and Supplementary Table S2.

    uv run python analysis/article_numbers.py

Yields relative to the recipe are medians over seeds of the ratio to the recipe's yield on the same seed.
"""

from __future__ import annotations

import glob
import math

import numpy as np
import pandas as pd
from article_stats import DEEP, RES, SWITCH, UNGUARDED, load
from scipy import stats

FIRST, SECOND = (2011, 2022), (2023, 2034)
TABLE1 = [("Safe branch alone", "safe"), ("Blend, monitored weight", "hybrid:convex:nofsm"),
          ("Blend, constant weight", "hybrid:fixed:0.5"), ("Tightened filter", "projected24t"),
          ("Filter, +3 mg/L margin", "projected24m"), ("Guarded filter", "projected24ms"), ("Simplex switch", "projected24mx")]
TABLE2 = [("Tightened filter", "projected24t"), ("Feed bound x 0.8", "projected24c:0.8"), ("Feed bound x 0.6", "projected24c:0.6"),
          ("Substrate oracle", "projected24s"), ("CO2 oracle", "projected24k"), ("+3 mg/L oxygen margin", "projected24m"),
          ("Margin and CO2 oracle", "projected24mk"), ("Margin and CO2 soft sensor", "projected24ms")]


def cell(d: pd.DataFrame, rec: pd.Series, c: str, sc: str, seeds: tuple[int, int]) -> str:
    g = d[(d.controller == c) & (d.scenario == sc) & d.seed.between(*seeds)]
    if g.empty:
        return "-"
    ratio = np.median([y / rec[(sc, s)] for y, s in zip(g.yield_kg, g.seed)]) - 1
    return f"{100 * ratio:+.0f} ({int(g.violating.sum())})"


def tables(d: pd.DataFrame) -> None:
    rec = d[d.controller == "recipe"].set_index(["scenario", "seed"]).yield_kg
    for seeds in (FIRST, SECOND):
        r = d[(d.controller == "recipe") & d.seed.between(*seeds)]
        rows = [["Open-loop recipe (kg)"] + [f"{r[r.scenario == sc].yield_kg.median():,.0f} ({int(r[r.scenario == sc].violating.sum())})"
                                             for sc in ("S1", "S5", "S5fast")]]
        rows += [[name] + [cell(d, rec, c, sc, seeds) for sc in ("S1", "S5", "S5fast")] for name, c in TABLE1]
        print(f"\nTable 1, seeds {seeds[0]}-{seeds[1]}: nominal, gradual drift, fast drift")
        print(pd.DataFrame(rows, columns=["controller", "S1", "S5", "S5fast"]).to_string(index=False))
    rows = []
    for name, c in TABLE2:
        g = d[(d.controller == c) & (d.scenario == "S5") & d.seed.between(*FIRST)]
        rows.append([name, cell(d, rec, c, "S1", FIRST), cell(d, rec, c, "S5", FIRST), int(g.deep.sum()), round(g.min_DO2.min(), 2),
                     cell(d, rec, c, "S5fast", FIRST)])
    print("\nTable 2, seeds 2011-2022")
    print(pd.DataFrame(rows, columns=["filter", "nominal", "gradual", "deep", "lowest DO", "fast"]).to_string(index=False))


def spread(d: pd.DataFrame) -> None:
    """Supplementary Table S4: median and interquartile range over seeds of the yield relative to the recipe."""
    rec = d[d.controller == "recipe"].set_index(["scenario", "seed"]).yield_kg
    print("\nSupplementary Table S4: median [25th, 75th percentile] of the yield relative to the recipe, %")
    for seeds in (FIRST, SECOND):
        for name, c in TABLE1:
            cells = []
            for sc in ("S1", "S5", "S5fast"):
                g = d[(d.controller == c) & (d.scenario == sc) & d.seed.between(*seeds)]
                r = 100 * (np.array([y / rec[(sc, s)] for y, s in zip(g.yield_kg, g.seed)]) - 1)
                q1, med, q3 = np.percentile(r, [25, 50, 75])
                cells.append(f"{med:+.0f} [{q1:+.0f}, {q3:+.0f}]")
            print(f"| {seeds[0]}-{seeds[1]} | {name} | " + " | ".join(cells) + " |")


def trajectories(controllers, patterns=("S5", "S5fast")) -> dict[str, pd.DataFrame]:
    out = {}
    if isinstance(patterns, str):
        patterns = (patterns,)
    for c in controllers:
        for f in [g for p in patterns for g in glob.glob(str(RES / f"traj_{c}_{p}_20*.csv.gz"))]:
            t = pd.read_csv(f)
            if "CO2_d" in t:
                out[f.split("traj_")[-1].removesuffix(".csv.gz")] = t
    return out


def co2_and_collapses() -> None:
    rows = []
    for name, t in trajectories(UNGUARDED).items():
        late = t[t.t_h >= 100]
        cross = late.t_h[late.CO2_d >= SWITCH]
        deep = t.index[t.DO2 <= DEEP]
        r = {"batch": name, "crossed": len(cross) > 0, "deep": len(deep) > 0, "peak": late.CO2_d.max()}
        if len(deep):
            i = deep[0]
            r["lead_h"] = t.t_h[i] - cross.iloc[0] if len(cross) else np.nan
            w = t.loc[max(0, i - 60):i]                       # the 12 h before the collapse
            off = (w.mu_X < 0.02).astype(int)
            r["stalled"] = bool(off.any())
            r["substrate_up"] = bool(w.S.max() > 0.01)
            r["S_at_collapse"] = t.S.iloc[i - 1]
        rows.append(r)
    b = pd.DataFrame(rows)
    deep = b[b.deep]
    print(f"\nFilters without a CO2 guard: {len(deep)} of {len(b)} drift batches collapsed (oxygen <= {DEEP} mg/L)")
    print(f"  crossed before the collapse: {int(deep.crossed.sum())} of {len(deep)}, median lead {deep.lead_h.median():.1f} h")
    nc = b[~b.crossed]
    print(f"  of {len(nc)} batches that did not cross, {int(nc.deep.sum())} collapsed, peaks {sorted(nc[nc.deep].peak.round(2))}")
    path = deep.stalled & deep.substrate_up
    print(f"  growth stopped and substrate above 0.01 g/L in the 12 h before: {int(path.sum())} of {len(deep)};"
          f" median substrate when oxygen fell {deep.S_at_collapse.median():.1f} g/L; exceptions {list(deep.batch[~path])}")
    for label, ctrls in (("blends", ["hybrid-convex-nofsm", "hybrid-fixed-0.5"]), ("guarded filter", ["projected24ms"])):
        tr = trajectories(ctrls)
        peaks = {k: t.loc[t.t_h >= 100, "CO2_d"].max() for k, t in tr.items()}
        crossed = [k for k, p in peaks.items() if p >= SWITCH]
        collapsed = [k for k in crossed if (tr[k].DO2 <= DEEP).any()]
        deep_any = [k for k, t in tr.items() if (t.DO2 <= DEEP).any()]
        print(f"  {label}: {len(tr)} drift batches, highest peak {max(peaks.values()):.2f}, crossed {len(crossed)},"
              f" crossed and collapsed {len(collapsed)}, deep collapses {len(deep_any)}")


def sensor_error() -> None:
    print("\nSoft sensor in closed loop (guarded filter, true CO2 > 7 after 100 h)")
    for sc in ("S1", "S5", "S5fast"):
        e = []
        for t in trajectories(["projected24ms"], sc).values():
            w = t[(t.t_h >= 100) & (t.CO2_d > 7) & t.co2_est.notna()]
            e.append((w.co2_est - w.CO2_d).to_numpy())
        e = np.concatenate(e)
        print(f"  {sc}: RMSE {np.sqrt(np.mean(e ** 2)):.2f}, 95th percentile {np.percentile(np.abs(e), 95):.2f}, bias {e.mean():+.2f}")


def simplex_handover() -> None:
    print("\nSimplex switch: hand-overs to the safe branch")
    for sc in ("S1", "S5", "S5fast"):
        share = [(t.branch == "safe").mean() for t in trajectories(["projected24mx"], sc).values()]
        print(f"  {sc}: {sum(s > 0 for s in share)} of {len(share)} batches, {100 * np.mean(share):.1f} % of steps on average")


def severity(d: pd.DataFrame) -> None:
    print("\nFast drift, seeds 2011-2034: violating batches with minutes below the floor and lowest oxygen")
    f = d[(d.scenario == "S5fast") & d.seed.between(2011, 2034) & (d.violating == 1)]
    for c in ("projected24ms", "hybrid:convex:nofsm", "hybrid:fixed:0.5", "safe"):
        g = f[f.controller == c].sort_values("seed")
        print(f"  {c}: " + ", ".join(f"{s}: {int(m * 12)} min, {x:.2f}" for s, m, x in zip(g.seed, g.steps_DO2_below_4, g.min_DO2)))
    print(f"  exact 95 % upper bound for 0 violating batches of 24: {stats.binomtest(0, 24).proportion_ci(method='exact').high:.2f}")


def co2_units() -> None:
    """The simulator's Henry's law takes off-gas CO2 in per cent without dividing by 100, so its dissolved CO2 is in
    units of 10 mg/L."""
    henry = math.exp(11.25 - 395.9 / (298.15 - 175.9)) / 4400          # bar L/g at 25 C, as in the simulator
    per_bar = 100 / henry
    p = SWITCH / per_bar
    print(f"\nCO2 threshold {SWITCH}: partial pressure {p / 1.01325:.3f} atm, {10 * SWITCH:.0f} mg/L")
    for name, atm in (("Pirt & Mancini, synthesis halved", 0.08), ("Ho & Smith, growth inhibited", 0.126), ("Ho & Smith, 3 %", 0.03),
                      ("Ho & Smith, 5 %", 0.05)):
        print(f"  {name}: {atm} atm = {atm * 1.01325 * per_bar * 10:.0f} mg/L")


def table_s2() -> None:
    rows = []
    tr = trajectories(UNGUARDED)
    for name, t in tr.items():
        rows.append({"variant": name.rsplit("_S5", 1)[0], "crossed": bool((t.loc[t.t_h >= 100, "CO2_d"] >= SWITCH).any()), "min_do2": t.DO2.min()})
    b = pd.DataFrame(rows)
    print("\nSupplementary Table S2: crossed&deep, crossed&not, not&deep, not&not, Fisher P")

    def line(label, g, cut):
        tab = pd.crosstab(g.crossed, g.min_do2 <= cut).reindex(index=[True, False], columns=[True, False], fill_value=0)
        print(f"  {label}: {len(g)} batches, {tab.values.ravel().tolist()}, P = {stats.fisher_exact(tab.values)[1]:.2g}")
    for v, g in b.groupby("variant"):
        line(v, g, DEEP)
    for cut in (1.0, 1.5, 2.0, 2.5, 3.0):
        line(f"all, deep at <= {cut} mg/L", b, cut)


ROBUSTNESS = [  # (summary file, recipe for the yields: "base" or the scenario's own recipe, label)
    ("summary_co2_off.csv", "own", "plant without CO2 inhibition"),
    ("summary_oracle_sensor_points.csv", "base", "CO2 oracle at the sensor's switching points"),
    ("summary_simplex_recipe.csv", "base", "Simplex switch falling back to the recipe"),
    ("summary_margins_heldout.csv", "base", "4 and 5 mg/L margins, seeds 2023-2034"),
    ("summary_blend_sweep.csv", "base", "constant blends"),
    ("summary_graded_inhibition.csv", "own", "graded CO2 inhibition"),
    ("summary_drift_dose.csv", "own", "oxygen transfer falling to 0.6, 0.5 or 0.3"),
    ("summary_guard_misspecification.csv", "base", "guard switching point and sensor bias"),
]


def robustness(d: pd.DataFrame) -> None:
    """Supplementary Table S5: the validity experiments, per controller and scenario: violating batches, deep
    collapses, lowest oxygen and the median yield relative to the recipe on the same seed and plant."""
    base = d[d.controller == "recipe"].set_index(["scenario", "seed"]).yield_kg
    print("\nSupplementary Table S5: violating, deep, lowest DO (mg/L), yield vs recipe (%)")
    for f, ref, label in ROBUSTNESS:
        t = pd.read_csv(RES / f)
        t["violating"] = ((t.steps_DO2_below_4 > 0) | t.catastrophic.astype(bool)).astype(int)
        own = t[t.controller == "recipe"].set_index(["scenario", "seed"]).yield_kg
        print(f"  {label}")
        for (c, sc), g in t.groupby(["controller", "scenario"]):
            rec = own if ref == "own" or c == "recipe" else base
            ratio = np.median([y / rec[(sc, s)] for y, s in zip(g.yield_kg, g.seed)]) - 1
            print(f"| {c} | {sc} | {len(g)} | {int(g.violating.sum())} | {int((g.min_DO2 <= DEEP).sum())} | {g.min_DO2.min():.2f} | {100 * ratio:+.0f} |")


def main() -> None:
    d = load()
    tables(d)
    spread(d)
    co2_and_collapses()
    sensor_error()
    simplex_handover()
    severity(d)
    co2_units()
    table_s2()
    robustness(d)


if __name__ == "__main__":
    main()
