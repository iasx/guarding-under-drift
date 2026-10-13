"""Generate the offline corpus: HDF5 subsets, episode index, splits, standardiser, coverage.

    uv run python -m sindy_fno.data.generate --out data/corpus --nom 120 --noise 40 --dist 40 --init 40
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sindy_fno.data.corpus import generate_subset, load_subset
from sindy_fno.data.coverage import write_coverage
from sindy_fno.data.preprocess import Standardizer
from sindy_fno.data.splits import episode_split
from sindy_fno.data.subsets import SUBSETS


def build_corpus(out: Path, counts: dict[str, int], workers: int | None, scheme: str) -> pd.DataFrame:
    corpora = {}
    for name, n in counts.items():
        if n <= 0:
            continue
        t0 = time.perf_counter()
        path = generate_subset(name, n, out, workers=workers)
        corpora[name] = load_subset(path)
        print(f"{name}: {n} episodes in {time.perf_counter() - t0:.0f} s -> {path}")

    rows, train_obs = [], []
    for name, c in corpora.items():
        split = episode_split(c.n_episodes, scheme=scheme, seed=0)
        member = np.empty(c.n_episodes, dtype=object)
        for part, idx in split.items():
            member[idx] = part
        train_obs.append(c.obs[split["train"]])
        for i in range(c.n_episodes):
            rows.append({
                "subset": name, "episode": i, "seed": int(c.seed[i]), "split": member[i],
                "length": int(c.length[i]), "yield_kg": float(c.yield_kg[i]),
                "catastrophic": bool(c.catastrophic[i]), "fs_scale": float(c.fs_scale[i]),
            })
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)

    obs_names = next(iter(corpora.values())).obs_names
    std = Standardizer.fit(np.concatenate(train_obs, axis=0), names=obs_names)
    (out / "standardizer.json").write_text(std.to_json(), encoding="utf-8")
    write_coverage(out, corpora)
    return index


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/corpus"))
    for name in SUBSETS:
        p.add_argument(f"--{name}", type=int, default=0, help=f"episodes for subset {name}")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--scheme", choices=["temporal", "random"], default="temporal")
    args = p.parse_args(argv)
    counts = {name: getattr(args, name) for name in SUBSETS}
    if sum(counts.values()) == 0:
        p.error("request at least one episode")
    index = build_corpus(args.out, counts, args.workers, args.scheme)
    print(index.groupby(["subset", "split"]).size().unstack(fill_value=0).to_string())
    print(index.groupby("subset")["yield_kg"].describe()[["count", "mean", "min", "max"]].round(0).to_string())


if __name__ == "__main__":
    main()
