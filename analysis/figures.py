"""The article's figures, from the results.

    uv run python analysis/figures.py

Seaborn's default theme and palette, with bold lowercase panel letters; writes ``figures/fig*.png``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).parents[1]
RES, FIG_DIR = ROOT / "results", ROOT / "figures"
SWITCH, FLOOR = 7.57, 4.0
GUARDED, MARGIN, BLEND = "projected24ms", "projected24m", "hybrid-convex-nofsm"
CM = 1 / 2.54
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# Applied before seaborn's theme so that the font and the settings the theme leaves alone stay fixed.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 8.5, "axes.titleweight": "bold", "legend.frameon": False,
                     "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "figure.dpi": 100})
sns.set_theme(context="paper")
DEEP = sns.color_palette()
BLEND_C, FILTER_C, GUARD_C, ACTOR_C, SENSOR_C = DEEP[0], DEEP[1], DEEP[2], DEEP[4], DEEP[8]
LINE_C, TEXT_C = "0.2", "0.35"
NAMES = {MARGIN: "Filter, +3 mg/L margin", GUARDED: "Guarded filter", BLEND: "Monitored blend"}
GUARD_MARGINS = ((0, "projected24ms:0"), (1, "projected24ms:1"), (2, "projected24ms:2"), (3, "projected24ms"))
COLORS = {"Filter, +3 mg/L margin": FILTER_C, "Guarded filter": GUARD_C, "Monitored blend": BLEND_C}


def violating(d: pd.DataFrame) -> pd.Series:
    return (d.steps_DO2_below_4 > 0) | d.catastrophic.astype(bool)


def box(ax, xy, w, h, text, fc="white", ec=INK, lw=0.9, fs=7.5, ls="-"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def arrow(ax, p, q, ls="-", color=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color, ls=ls, shrinkA=1, shrinkB=1))


def co2_batches() -> pd.DataFrame:
    """Peak dissolved CO2 after the drift onset and minimum oxygen for every drift batch with CO2 logged."""
    groups = {"Filters without a CO$_2$ guard": ["projected24t", "projected24c-0.8", "projected24c-0.6", "projected24s", MARGIN],
              "Guarded filter": [GUARDED], "Blends": [BLEND, "hybrid-fixed-0.5"]}
    rows = []
    for group, ctrls in groups.items():
        for c in ctrls:
            for f in [*glob.glob(str(RES / f"traj_{c}_S5_20*.csv.gz")), *glob.glob(str(RES / f"traj_{c}_S5fast_20*.csv.gz"))]:
                t = pd.read_csv(f)
                if "CO2_d" not in t:
                    continue
                late = t[t.t_h >= 100]
                rows.append({"group": group, "file": Path(f).name, "co2_peak": late.CO2_d.max(), "min_do2": t.DO2.min()})
    return pd.DataFrame(rows)


def tint(color, amount: float = 0.82):
    return tuple(c + amount * (1 - c) for c in mcolors.to_rgb(color))


def panel_letters(axes, x: float, y: float) -> None:
    for ax, letter in zip(axes, "abcdefgh"):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, weight="bold", color="black", va="bottom", ha="right")


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_designs() -> None:
    """The three ways of executing the learned action."""
    fig, ax = plt.subplots(figsize=(17 * CM, 8.8 * CM))
    ax.set_xlim(0, 17); ax.set_ylim(0, 8.8); ax.axis("off")
    fs = 6.8
    filt = ("Filter: closest input the\nmodel admits for 24 h,\noxygen margin", FILTER_C)
    rows = (
        (6.9, "Safety filter", [("Actor", ACTOR_C), filt]),
        (4.1, "Blend", [("Actor and MPC", ACTOR_C), ("w a$^F$ + (1 − w) a$^S$,\nthen one-hour projection", BLEND_C)]),
        (1.3, "Guarded filter", [("Actor", ACTOR_C), ("Filter with\noxygen margin", FILTER_C),
                                 ("CO$_2$ guard: levers on the\nrecipe's side while the\nsensor reads CO$_2$ > 7.35", GUARD_C)]),
    )
    for (y, title, boxes), letter in zip(rows, "abc"):
        ax.text(0.3, y + 1.45, letter, fontsize=10, weight="bold", va="center")
        ax.text(0.75, y + 1.45, title, fontsize=8, weight="bold", va="center")
        x = 0.3
        for i, (text, color) in enumerate(boxes):
            w = 2.6 if i == 0 else 4.0
            box(ax, (x, y), w, 1.15, text, fc=tint(color), ec=color, fs=fs)
            if i:
                arrow(ax, (x - 0.6, y + 0.575), (x, y + 0.575))
            x += w + 0.6
        box(ax, (14.4, y), 2.3, 1.15, "Plant", fc=tint(SENSOR_C), ec=SENSOR_C, fs=fs)
        arrow(ax, (x - 0.6, y + 0.575), (14.4, y + 0.575))
    ax.text(8.5, 0.35, "Soft sensor: dissolved CO$_2$ from off-gas CO$_2$, head pressure, temperature, CO$_2$ evolution rate and aeration",
            fontsize=6.5, color=TEXT_C, ha="center")
    save(fig, "fig1_designs.png")


def fig_mechanism(scenario: str = "S5", seed: int = 2016) -> None:
    """One gradual-drift batch: oxygen, dissolved CO2, growth rate and substrate under three designs."""
    frames = []
    for c, name in NAMES.items():
        t = pd.read_csv(RES / f"traj_{c}_{scenario}_{seed}.csv.gz")
        t = t[t.t_h >= 90].assign(design=name, S=lambda d: np.maximum(d.S, 1e-4))
        frames.append(t[["t_h", "DO2", "CO2_d", "mu_X", "S", "design"]])
    d = pd.concat(frames, ignore_index=True)
    panels = (("DO2", "Oxygen (mg/L)"), ("CO2_d", "Dissolved CO$_2$\n(10 mg/L)"), ("mu_X", "Growth rate\n(1/h)"), ("S", "Substrate\n(g/L)"))
    fig, axes = plt.subplots(4, 1, figsize=(16 * CM, 14 * CM), sharex=True)
    for i, (ax, (col, label)) in enumerate(zip(axes, panels)):
        ax.axvspan(100, 200, color="0.75", alpha=0.25, lw=0)
        sns.lineplot(data=d, x="t_h", y=col, hue="design", style="design", palette=COLORS, dashes={k: (1, 0) if k != "monitored blend" else (4, 2) for k in COLORS},
                     estimator=None, lw=1.1, ax=ax, legend=i == 0)
        ax.set_ylabel(label)
    axes[0].axhline(FLOOR, color=LINE_C, lw=0.8, ls=(0, (2, 2)))
    axes[1].axhline(SWITCH, color=LINE_C, lw=0.8, ls=(0, (2, 2)))
    axes[1].text(92, SWITCH + 0.06, "growth-inhibition threshold", fontsize=6.5, color=TEXT_C)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Batch time (h); drift from 100 h to 200 h (shaded)")
    sns.move_legend(axes[0], "lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title=None, frameon=False)
    panel_letters(axes, x=-0.2, y=0.8)
    fig.tight_layout()
    save(fig, "fig2_mechanism.png")


def fig_co2() -> None:
    """Peak dissolved CO2 after the drift begins against the lowest oxygen of each drift batch."""
    d = co2_batches()
    counts = d.group.value_counts()
    d["group"] = d.group.map(lambda g: f"{g} ({counts[g]} batches)")
    groups = list(dict.fromkeys(d.group))
    palette = dict(zip(groups, (FILTER_C, GUARD_C, BLEND_C)))
    fig, ax = plt.subplots(figsize=(13 * CM, 9 * CM))
    sns.scatterplot(data=d, x="co2_peak", y="min_do2", hue="group", style="group", palette=palette, markers=["o", "^", "s"],
                    s=22, alpha=0.75, edgecolor="white", linewidth=0.3, ax=ax)
    ax.axvline(SWITCH, color=LINE_C, lw=0.8, ls=(0, (3, 2))); ax.axhline(FLOOR, color=LINE_C, lw=0.8, ls=(0, (2, 2)))
    ax.text(SWITCH + 0.03, 10.3, "CO$_2$ threshold", fontsize=6.5, color=TEXT_C)
    ax.text(8.6, FLOOR + 0.15, "oxygen floor", fontsize=6.5, color=TEXT_C)
    ax.set_xlabel("Peak dissolved CO$_2$ after the drift begins (10 mg/L)"); ax.set_ylabel("Lowest dissolved oxygen in the batch (mg/L)")
    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, title=None, frameon=False)
    save(fig, "fig3_co2.png")


def fig_margin() -> None:
    """Violating drift batches against the oxygen margin added to the tightened filter, with the guarded filter and the
    monitored blend."""
    runs = {0: ("summary_tightened_filters.csv", "projected24t"), 3: ("summary_matched_margin.csv", "projected24m"),
            **{m: ("summary_margin_sweep.csv", f"projected24m:{m}") for m in (1, 2, 4, 5)}}
    guarded = pd.concat([pd.read_csv(RES / f) for f in ("summary_co2_sensor.csv", "summary_guard_margins.csv")], ignore_index=True)
    held = pd.concat([pd.read_csv(RES / f) for f in ("summary_heldout.csv", "summary_margins_heldout.csv")], ignore_index=True)
    blend = pd.read_csv(RES / "summary_main.csv")
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 7 * CM), sharey=True)
    for ax, sc, title in ((axes[0], "S5", "Gradual drift"), (axes[1], "S5fast", "Fast drift")):
        sweep = pd.DataFrame([{"margin": m, "violating": int(violating(t[(t.controller == c) & (t.scenario == sc) & t.seed.between(2011, 2022)]).sum())}
                              for m, (f, c) in sorted(runs.items()) for t in [pd.read_csv(RES / f)]])
        sns.lineplot(data=sweep, x="margin", y="violating", marker="o", color=FILTER_C, ax=ax, label="Filter, design set", legend=False)
        for m, c in ((0, "projected24t"), (3, "projected24m"), (4, "projected24m:4"), (5, "projected24m:5")):
            ax.plot(m, int(violating(held[(held.controller == c) & (held.scenario == sc)]).sum()), marker="D", ms=5, mfc="white", mec=FILTER_C, mew=1.2, ls="none")
        for seeds, mk, ls in (((2011, 2022), "^", "-"), ((2023, 2034), "v", (0, (1, 1.5)))):
            g = guarded[guarded.seed.between(*seeds)]
            margins = [(m, g[g.controller == c]) for m, c in GUARD_MARGINS if not g[(g.controller == c) & (g.scenario == sc)].empty]
            ax.plot([m for m, _ in margins], [int(violating(t[t.scenario == sc]).sum()) for _, t in margins],
                    marker=mk, ms=6, color=GUARD_C, ls=ls if len(margins) > 1 else "none", lw=1.1)
        b = blend[(blend.controller == "hybrid:convex:nofsm") & (blend.scenario == sc) & blend.seed.between(2011, 2022)]
        ax.axhline(int(violating(b).sum()), color=BLEND_C, lw=1, ls=(0, (3, 2)))
        ax.set_title(title); ax.set_xlabel("Oxygen margin added (mg/L)"); ax.set_ylim(-0.5, 12.5)
    axes[0].set_ylabel("Violating batches of 12")
    axes[0].plot([], [], marker="o", color=FILTER_C, label="Filter, design set")
    axes[0].plot([], [], marker="D", ms=5, mfc="white", mec=FILTER_C, ls="none", label="Filter, held-out set")
    axes[0].plot([], [], marker="^", ms=6, color=GUARD_C, lw=1.1, label="Guarded filter, design set")
    axes[0].plot([], [], marker="v", ms=6, color=GUARD_C, lw=1.1, ls=(0, (1, 1.5)), label="Guarded filter, held-out set")
    axes[0].plot([], [], color=BLEND_C, lw=1, ls=(0, (3, 2)), label="Monitored blend, design set")
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[0].legend(keep.values(), keep.keys(), loc="upper right", fontsize=6.2)
    panel_letters(axes, x=-0.02, y=1.03)
    fig.tight_layout()
    save(fig, "fig4_margin.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_designs(); fig_mechanism(); fig_co2(); fig_margin()
    print("wrote", sorted(f.name for f in FIG_DIR.glob("fig*.png")))


if __name__ == "__main__":
    main()
