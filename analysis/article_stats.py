"""The comparisons the article reports, tested together.

    uv run python analysis/article_stats.py

Yield differences are paired by seed (two-sided Wilcoxon signed-rank, median with a 95 % percentile bootstrap
interval). Violating batches and deep collapses (oxygen at or below 1.5 mg/L) are compared with an exact McNemar test
on the discordant seeds. The association between dissolved CO2 crossing its switch and a deep collapse is tested by
permuting the collapse labels among seeds within each filter variant and scenario, since the same seeds recur across
variants. One Holm-corrected family; writes ``results/article_stats.csv``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from stats import holm, paired

RES = Path(__file__).parents[1] / "results"
GUARDED, MARGIN, TIGHT, BLEND, SIMPLEX = "projected24ms", "projected24m", "projected24t", "hybrid:convex:nofsm", "projected24mx"
GUARD_AT = {m: f"projected24ms:{m}" for m in (0, 1, 2)} | {3: GUARDED}
SWITCH, DEEP = 7.57, 1.5
PERMUTATIONS = 20_000
FILES = ("summary_main.csv", "summary_tightened_filters.csv", "summary_matched_margin.csv", "summary_margin_sweep.csv",
         "summary_mechanism.csv", "summary_co2_oracle.csv", "summary_margin_co2_oracle.csv", "summary_heldout.csv",
         "summary_co2_heldout.csv", "summary_co2_sensor.csv", "summary_simplex.csv", "summary_guard_margins.csv")
UNGUARDED = ("projected24c-0.8", "projected24c-0.6", "projected24s", MARGIN)


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(RES / f) for f in FILES], ignore_index=True)
    d = d.drop_duplicates(["controller", "scenario", "seed"], keep="last")
    d["violating"] = ((d.steps_DO2_below_4 > 0) | d.catastrophic.astype(bool)).astype(int)
    d["deep"] = (d.min_DO2 <= DEEP).astype(int)
    return d


def mcnemar(d: pd.DataFrame, a: str, b: str, sc: str, col: str, seeds: range) -> dict:
    v = d[(d.scenario == sc) & d.seed.isin(seeds)].pivot_table(index="seed", columns="controller", values=col)[[a, b]].dropna()
    only_a, only_b = int(((v[a] == 1) & (v[b] == 0)).sum()), int(((v[a] == 0) & (v[b] == 1)).sum())
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"kind": f"{col} batches", "a": a, "b": b, "scenario": sc, "seeds": f"{seeds.start}-{seeds.stop - 1}", "n": len(v),
            "a_count": int(v[a].sum()), "b_count": int(v[b].sum()), "p": float(p)}


def yield_test(d: pd.DataFrame, a: str, b: str, sc: str, seeds: range) -> dict:
    r = paired(d[d.seed.isin(seeds)], a, b, sc)
    return {"kind": "yield, kg", "a": a, "b": b, "scenario": sc, "seeds": f"{seeds.start}-{seeds.stop - 1}", "n": r["n"],
            "median_diff": r["median_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "seeds_a_greater": r["positive"],
            "p": r["wilcoxon_two_sided"]}


def crossing_batches() -> pd.DataFrame:
    """Drift batches of the filters without a CO2 guard: did CO2 cross the switch after the drift onset, did oxygen collapse."""
    rows = []
    for c in UNGUARDED:
        for f in glob.glob(str(RES / f"traj_{c}_S5*_20*.csv.gz")):
            t = pd.read_csv(f)
            if "CO2_d" in t:
                rows.append({"stratum": f"{c}/{'S5fast' if 'S5fast' in f else 'S5'}",
                             "crossed": bool((t.loc[t.t_h >= 100, "CO2_d"] >= SWITCH).any()), "deep": bool((t.DO2 <= DEEP).any())})
    return pd.DataFrame(rows)


def co2_permutation(rng: np.random.Generator) -> dict:
    b = crossing_batches()
    observed = int((b.crossed & b.deep).sum())
    strata = [g.index.to_numpy() for _, g in b.groupby("stratum")]
    deep, crossed = b.deep.to_numpy(), b.crossed.to_numpy()
    hits = 0
    for _ in range(PERMUTATIONS):
        perm = deep.copy()
        for idx in strata:
            perm[idx] = rng.permutation(deep[idx])
        hits += int((crossed & perm).sum()) >= observed
    return {"kind": "CO2 crossing vs deep collapse", "a": "crossed the switch", "b": "stayed below", "scenario": "S5, S5fast",
            "seeds": "2011-2022", "n": len(b), "a_count": observed, "b_count": int((~b.crossed & b.deep).sum()),
            "p": (hits + 1) / (PERMUTATIONS + 1)}


def main() -> None:
    d = load()
    rng = np.random.default_rng(0)
    first, second, both = range(2011, 2023), range(2023, 2035), range(2011, 2035)
    rows = [yield_test(d, GUARDED, BLEND, sc, seeds) for seeds in (first, second) for sc in ("S1", "S5", "S5fast")]
    rows += [mcnemar(d, TIGHT, GUARDED, "S5", "violating", both), mcnemar(d, MARGIN, GUARDED, "S5", "violating", both)]
    rows += [mcnemar(d, TIGHT, c, "S5", "deep", first) for c in ("projected24k", "projected24s", "projected24c:0.6")]
    rows.append(co2_permutation(rng))
    rows += [yield_test(d, GUARDED, SIMPLEX, sc, both) for sc in ("S1", "S5")]
    rows.append(mcnemar(d, GUARD_AT[0], GUARD_AT[3], "S5", "violating", both))
    rows += [yield_test(d, GUARD_AT[2], GUARD_AT[3], sc, both) for sc in ("S1", "S5")]
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.tolist())
    t.to_csv(RES / "article_stats.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(f"{len(t)} tests, Holm over all")
    print(t.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
