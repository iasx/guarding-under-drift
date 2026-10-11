#!/usr/bin/env bash
# Every closed-loop run the article reports; each writes results/summary_<tag>.csv and the batches' trajectories.
#
#     bash experiments/run_article.sh
#
# A filtered batch takes about 1,000 s of one core; --resume skips batches whose trajectory already exists.
# One IPOPT thread per process: many parallel solvers otherwise oversubscribe the cores and stall.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-12}
FIRST=$(seq 2011 2022)
SECOND=$(seq 2023 2034)

run() {
    tag=$1; seeds=$2; shift 2
    uv run python experiments/run_hybrid.py --tag "$tag" --resume --workers "$WORKERS" --seeds $seeds "$@"
}

# Recipe, safe branch, blends and online SAC; filters with and without an added oxygen margin.
run main "$FIRST" --controllers recipe safe hybrid:convex:nofsm hybrid:fixed:0.5 policy=sac_s0
run tightened_filters "$FIRST" --controllers projected24t
run matched_margin "$FIRST" --controllers projected24m
run margin_sweep "$FIRST" --controllers projected24m:1 projected24m:2 projected24m:4 projected24m:5
run heldout "$SECOND" --controllers recipe safe hybrid:convex:nofsm hybrid:fixed:0.5 projected24t projected24m

# Interventions on the failure mechanism: the feed bound that stands in for substrate, a substrate oracle, and a
# CO2 oracle, alone and with the oxygen margin.
run mechanism "$FIRST" --controllers projected24c:0.8 projected24c:0.6 projected24s
run co2_oracle "$FIRST" --controllers projected24k
run margin_co2_oracle "$FIRST" --controllers projected24mk
run co2_heldout "$SECOND" --controllers projected24k projected24mk

# The guarded filter on the soft sensor, the Simplex switch on the same trigger, and the guard's oxygen margin.
run co2_sensor "$FIRST $SECOND" --controllers projected24ms
run simplex "$FIRST $SECOND" --controllers projected24mx
run guard_margins "$FIRST $SECOND" --scenarios S1 S5 --controllers projected24ms:0 projected24ms:1 projected24ms:2
