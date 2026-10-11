"""Measurement models for the safe-branch estimator, fitted on the corpus.

    uv run python -m sindy_fno.safe.measurements --corpus data/corpus --models models/sindy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.data.corpus import Corpus
from sindy_fno.safe.library import CER_TERMS, WT_TERMS, evaluate, variables
from sindy_fno.safe.sparse_model import SparseModel, StaticModel

MEASURED = ["Wt", "CER", "DO2"]


@dataclass
class MeasurementModels:
    wt: StaticModel
    cer: StaticModel
    do2: StaticModel
    noise_std: dict[str, float]      # residual std per measured channel
    process_std: list[float]         # one-step model error std per state
    feed_max: float = float("inf")   # g/(L h): 99th percentile of feed per volume in healthy training steps

    def predict(self, x, u):
        """Measurement vector [Wt, CER, DO2] for state ``x`` and input ``u`` (numpy or CasADi)."""
        parts = [self.wt.predict(x, u), self.cer.predict(x, u), self.do2.predict(x, u)]
        if isinstance(parts[0], np.ndarray):
            return np.stack(parts, axis=-1)
        return parts

    def to_json(self) -> str:
        return json.dumps({
            "wt": json.loads(self.wt.to_json()), "cer": json.loads(self.cer.to_json()),
            "do2": json.loads(self.do2.to_json()), "noise_std": self.noise_std, "process_std": self.process_std,
            "feed_max": self.feed_max,
        }, indent=1)

    @classmethod
    def from_json(cls, data: str | dict) -> MeasurementModels:
        d = json.loads(data) if isinstance(data, str) else data
        return cls(
            wt=StaticModel.from_json(d["wt"]), cer=StaticModel.from_json(d["cer"]), do2=StaticModel.from_json(d["do2"]),
            noise_std={k: float(v) for k, v in d["noise_std"].items()}, process_std=[float(v) for v in d["process_std"]],
            feed_max=float(d.get("feed_max", float("inf"))),
        )

    @classmethod
    def load(cls, models_dir: str | Path) -> MeasurementModels:
        return cls.from_json(Path(models_dir, "measurements.json").read_text(encoding="utf-8"))


def measurement_episodes(corpus: Corpus, state_names: list[str], input_names: list[str]):
    """(X, U_at, Y[Wt, CER, DO2]) per episode, aligned at the same step."""
    hidx = [corpus.hidden_names.index(n) for n in state_names]
    aidx = [corpus.action_names.index(n) for n in input_names if n in corpus.action_names]
    oidx = [corpus.obs_names.index(n) for n in input_names if n not in corpus.action_names]
    yidx = [corpus.obs_names.index(n) for n in MEASURED]
    out = []
    for i in range(corpus.n_episodes):
        L = int(corpus.length[i])
        if L < 3 or corpus.catastrophic[i]:
            continue
        X = corpus.hidden[i, :L][:, hidx]
        U = np.concatenate([corpus.action[i, :L][:, aidx], corpus.obs[i, :L][:, oidx]], axis=1)
        Y = corpus.obs[i, :L][:, yidx]
        out.append((X, U, Y))
    return out


def fit_measurement_models(eps, model: SparseModel, do2: StaticModel, threshold: float = 0.05, alpha: float = 1e-2) -> MeasurementModels:
    X_list = [e[0] for e in eps]
    U_list = [e[1] for e in eps]
    # The weight offset is a few percent of the signal; keep it with a low threshold.
    wt = StaticModel.fit(X_list, U_list, [e[2][:, 0] for e in eps], terms=WT_TERMS, threshold=0.01, alpha=alpha)
    wt.output_name = "Wt"
    cer = StaticModel.fit(X_list, U_list, [e[2][:, 1] for e in eps], terms=CER_TERMS, threshold=threshold, alpha=alpha)
    cer.output_name = "CER"
    X = np.vstack(X_list)
    U = np.vstack(U_list)
    Y = np.vstack([e[2] for e in eps])
    noise = {
        "Wt": float((Y[:, 0] - wt.predict(X, U)).std()),
        "CER": float((Y[:, 1] - cer.predict(X, U)).std()),
        "DO2": float((Y[:, 2] - do2.predict(X, U)).std()),
    }
    # One-step prediction error of the dynamic model, per state.
    errs = []
    for Xe, Ue, _ in eps:
        pred = np.array([model.simulate(Xe[k], Ue[k + 1 : k + 2], STEP_IN_HOURS)[-1] for k in range(len(Xe) - 1)])
        errs.append(pred - Xe[1:])
    process_std = np.vstack(errs).std(axis=0).tolist()
    # Input envelope of the validity region: feed per volume seen in healthy
    # (oxygen-sufficient) operation. The sparse models extrapolate linearly
    # beyond it, and the plant does not.
    feed = (600.0 * U[:, 0] + 1000.0 * U[:, 1]) / X[:, 2]
    healthy = Y[:, 2] >= 8.0
    feed_max = float(np.percentile(feed[healthy], 99)) if healthy.any() else float("inf")
    return MeasurementModels(wt=wt, cer=cer, do2=do2, noise_std=noise, process_std=process_std, feed_max=feed_max)


def _check_terms_exist():
    # Guard: variables() must expose everything the measurement libraries use.
    x = np.ones((1, 3))
    u = np.ones((1, 9))
    evaluate(WT_TERMS + CER_TERMS, variables(x, u))


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    from sindy_fno.data.corpus import load_subset
    from sindy_fno.data.subsets import SUBSETS
    from sindy_fno.safe.library import INPUT_NAMES, STATE_NAMES

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--models", type=Path, default=Path("models/sindy"))
    args = p.parse_args(argv)
    _check_terms_exist()
    model = SparseModel.from_json((args.models / "model.json").read_text(encoding="utf-8"))
    do2 = StaticModel.from_json((args.models / "do2_model.json").read_text(encoding="utf-8"))
    index = pd.read_csv(args.corpus / "index.csv")
    train_seeds = set(index[index.split == "train"].seed)
    eps = []
    for name in SUBSETS:
        path = args.corpus / f"{name}.h5"
        if path.exists():
            c = load_subset(path)
            eps += [e for e, s in zip(measurement_episodes(c, STATE_NAMES, INPUT_NAMES), c.seed) if int(s) in train_seeds]
    mm = fit_measurement_models(eps, model, do2)
    (args.models / "measurements.json").write_text(mm.to_json(), encoding="utf-8")
    print(mm.wt.equation())
    print(mm.cer.equation())
    print("noise std:", {k: round(v, 4) for k, v in mm.noise_std.items()})
    print("process std per state:", [round(v, 4) for v in mm.process_std])
    print("feed_max [g/(L h)]:", round(mm.feed_max, 3))


if __name__ == "__main__":
    main()
