# guarding-under-drift

[![DOI](https://zenodo.org/badge/1414611650.svg)](https://doi.org/10.5281/zenodo.23339975)

Code, trained models and results for *Guarding an unmodeled carbon dioxide state for safe learning-based fermentation control
under gradual process drift*: a learned controller for the IndPenSim penicillin fermentation executed through safety
filters, blends and a CO₂-guarded filter under process drift.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The modified PenSimPy simulator is included in `vendor/pensimpy`
(copied from [iasx/PenSimPy](https://github.com/iasx/PenSimPy); MIT licence), so the repository is self-contained.
PyTorch comes from the CUDA 13 index on Linux and Windows and from PyPI on macOS; no GPU is needed to run the
experiments, and the analysis scripts need neither PyTorch nor the simulator.

```bash
uv sync
```

## Reproduce the article's numbers from the stored results

```bash
uv run python analysis/article_stats.py     # the 20 tests, Holm-corrected -> results/article_stats.csv
uv run python analysis/article_numbers.py   # Tables 1-2, CO2 analysis, sensor error, Supplementary Tables S2, S4, S5
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

Scenarios: `S1` nominal, `S5` gradual drift, `S5fast` fast drift; for the validity tests, `S5k06`/`S5k05`/`S5k03` oxygen transfer
falling to 0.6/0.5/0.3, `S1g05`/`S5g01`/`S5g05` graded CO₂ inhibition of half-width 0.1/0.5, and `S5noco2`/`S5fastnoco2`
without CO₂ inhibition. Controllers:

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
| `projected24mx[:<mg/L>[:recipe]]` | Simplex switch to the MPC (or the recipe) on the same CO₂ trigger |

The CO₂-guarded filters also take `:on<value>` (switching-on point, off 0.1 below) and `:bias<value>` (added to the
soft sensor's estimate), e.g. `projected24ms:3:on7.5`; `hybrid:fixed:<w>` takes any constant weight.

## Retrain the models

```bash
bash experiments/train.sh                   # corpus, identified model, surrogate, actor, thresholds, SAC, CO2 sensor
```

This overwrites `models/`. GPU training is not bit-for-bit deterministic, so retrained models reproduce the results
statistically rather than exactly.

## Layout

- `src/sindy_fno/` — plant wrapper, identified model and MPC, safety filters, blend, actor, soft sensor
- `vendor/pensimpy/` — the IndPenSim simulator (PenSimPy) as modified for the study
- `experiments/` — run and training scripts
- `analysis/` — statistics, numbers and figures of the article
- `models/` — the trained models used in the article
- `results/` — summary tables and per-batch trajectories (`traj_<controller>_<scenario>_<seed>.csv.gz`)
