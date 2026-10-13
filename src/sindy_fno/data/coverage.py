"""Operating-envelope coverage summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sindy_fno.data.corpus import Corpus


def coverage_table(x: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-channel min / percentiles / max over all non-NaN steps of ``x[..., channel]``."""
    flat = np.asarray(x, dtype=float).reshape(-1, x.shape[-1])
    rows = []
    for j, name in enumerate(names):
        col = flat[:, j]
        col = col[~np.isnan(col)]
        q = np.percentile(col, [5, 50, 95]) if col.size else [np.nan] * 3
        rows.append({
            "min": col.min() if col.size else np.nan,
            "p5": q[0], "p50": q[1], "p95": q[2],
            "max": col.max() if col.size else np.nan,
            "steps": int(col.size),
        })
    return pd.DataFrame(rows, index=names)


def to_markdown(table: pd.DataFrame, digits: int = 4) -> str:
    cols = list(table.columns)
    lines = ["| channel | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        cells = [f"{v:.{digits}g}" if isinstance(v, float) else str(v) for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return chr(10).join(lines)


def write_coverage(out_dir: str | Path, corpora: dict[str, Corpus]) -> Path:
    lines = ["# Corpus coverage", ""]
    total_eps = sum(c.n_episodes for c in corpora.values())
    total_steps = sum(int(c.length.sum()) for c in corpora.values())
    lines += [f"Episodes: {total_eps}; transitions: {total_steps}.", ""]
    lines += ["| Subset | Episodes | Transitions | Catastrophic | Yield median [kg] | Yield min–max |", "|---|---|---|---|---|---|"]
    for name, c in corpora.items():
        y = c.yield_kg
        lines.append(
            f"| {name} | {c.n_episodes} | {int(c.length.sum())} | {int(c.catastrophic.sum())} "
            f"| {np.median(y):.0f} | {y.min():.0f}–{y.max():.0f} |"
        )
    for name, c in corpora.items():
        lines += ["", f"## {name}: observations", "", to_markdown(coverage_table(c.obs, c.obs_names))]
        lines += ["", f"## {name}: actions", "", to_markdown(coverage_table(c.action, c.action_names))]
    path = Path(out_dir) / "coverage.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
