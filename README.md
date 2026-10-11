# guarding-under-drift

Code, trained models and results for *Guarding an unmodeled carbon-dioxide state keeps learned fermentation control
safe under gradual drift*: a learned controller for the IndPenSim penicillin fermentation executed through safety
filters, blends and a CO₂-guarded filter under process drift.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The simulator is a fork of PenSimPy
([iasx/PenSimPy](https://github.com/iasx/PenSimPy)) and is installed as a dependency.

```bash
uv sync
```

## Reproduce the article's numbers from the stored results

```bash
uv run python analysis/article_stats.py     # the 17 tests, Holm-corrected -> results/article_stats.csv
uv run python analysis/article_numbers.py   # Tables 1-2, CO2 analysis, sensor error, Supplementary Table S2
uv run python analysis/figures.py           # Figures 1-4 -> figures/
```

## Rerun the experiments

```bash
bash experiments/run_article.sh             # every closed-loop run -> results/summary_<tag>.csv and trajectories
```

A batch with a safety filter takes about 1,000 s on one core; set `WORKERS` for parallel batches (default 12). Runs
resume: batches whose trajectory is already in `results/` are summarised, not rerun, so delete `results/` to rerun
everything. Batches are deterministic per seed. A single run:

```bash
uv run python experiments/run_hybrid.py --tag mytest --seeds 2016 --scenarios S5 --controllers projected24ms
```

Scenarios: `S1` nominal, `S5` gradual drift, `S5fast` fast drift. Controllers:

| Name | Design |
|---|---|
| `recipe` | open-loop recipe |
| `safe` | MPC on the identified model |
| `hybrid:convex:nofsm`, `hybrid:fixed:0.5` | monitored and constant blend |
| `policy=sac_s0` | online soft actor–critic |
| `projected24t` | tightened safety filter |
| `projected24m[:<mg/L>]` | filter with an added oxygen margin (default 3) |
| `projected24c:<scale>`, `projected24s` | feed-bound and substrate interventions |
| `projected24k`, `projected24mk` | CO₂ oracle guard, without and with the margin |
| `projected24ms[:<mg/L>]` | guarded filter on the CO₂ soft sensor |
| `projected24mx` | Simplex switch to the MPC on the same CO₂ trigger |

## Retrain the models

```bash
bash experiments/train.sh                   # corpus, identified model, surrogate, actor, thresholds, SAC, CO2 sensor
```

This overwrites `models/`. GPU training is not bit-for-bit deterministic, so retrained models reproduce the results
statistically rather than exactly.

## Layout

- `src/sindy_fno/` — plant wrapper, identified model and MPC, safety filters, blend, actor, soft sensor
- `experiments/` — run and training scripts
- `analysis/` — statistics, numbers and figures of the article
- `models/` — the trained models used in the article
- `results/` — summary tables and per-batch trajectories (`traj_<controller>_<scenario>_<seed>.csv.gz`)
