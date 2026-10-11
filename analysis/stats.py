"""Paired comparisons and multiple-testing correction shared by the article's analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BOOT = 10000


def boot_ci(values: np.ndarray, fn, n: int = BOOT, alpha: float = 0.05, rng=None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(len(values), size=(n, len(values)))
    samples = np.array([fn(values[i]) for i in idx])
    return float(np.nanpercentile(samples, 100 * alpha / 2)), float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))


def paired(t: pd.DataFrame, a: str, b: str, scenario: str, value: str = "yield_kg") -> dict:
    """Per-seed paired comparison of controller ``a`` minus ``b`` on one scenario: median difference with a 95 %
    percentile-bootstrap interval, seeds on which ``a`` is ahead, and a two-sided Wilcoxon signed-rank test."""
    p = t[t.scenario == scenario].pivot_table(index="seed", columns="controller", values=value)
    d = (p[a] - p[b]).dropna().values
    w_two = stats.wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    lo, hi = boot_ci(d, np.median)
    return {"n": len(d), "median_diff": float(np.median(d)), "ci_lo": lo, "ci_hi": hi, "positive": int((d > 0).sum()),
            "wilcoxon_two_sided": float(w_two)}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, m = np.empty(len(pvals)), len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
