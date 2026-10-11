#!/usr/bin/env bash
# The models the closed-loop runs load, rebuilt from scratch: the behavior-policy corpus, the identified model of the
# safe branch, the surrogate ensemble and the actor, the blend's monitor thresholds, online SAC and the CO2 soft sensor.
#
#     bash experiments/train.sh
#
# The trained models are in models/; this script overwrites them. Training on a GPU is not bit-for-bit deterministic,
# so retrained models reproduce the results statistically rather than to the kilogram.
set -euo pipefail
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

uv run python -m sindy_fno.data.generate --out data/corpus --nom 120 --noise 40 --dist 40 --init 40
uv run python -m sindy_fno.safe.identify --corpus data/corpus --out models/sindy
uv run python -m sindy_fno.safe.measurements --corpus data/corpus --models models/sindy
uv run python -m sindy_fno.adaptive.train --corpus data/corpus --out models/fno --window 32 --members 5 --epochs 8
uv run python -m sindy_fno.adaptive.pretrain --corpus data/corpus --surrogate models/fno --out models/agent --updates 20000
# The blend's monitor thresholds in two passes: first with the recipe acting, then with the blend acting under the
# first pass's thresholds, which is how the deployed thresholds were obtained.
uv run python -m sindy_fno.hybrid.calibrate --corpus data/corpus --models models --out models/hybrid/shadow
uv run python -m sindy_fno.hybrid.calibrate --corpus data/corpus --models models --out models/hybrid/acting \
    --acting hybrid --thresholds models/hybrid/shadow/thresholds.json
cp models/hybrid/acting/thresholds.json models/hybrid/thresholds.json
uv run python -m sindy_fno.baselines.sac_online --episodes 240 --seed 0 --out models/sac_s0
uv run python experiments/fit_co2_sensor.py
