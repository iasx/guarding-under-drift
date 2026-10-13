"""Episode-level splits: never split by transition."""

from __future__ import annotations

import numpy as np


def episode_split(
    n_episodes: int,
    scheme: str = "temporal",
    fractions: tuple[float, float, float] = (0.75, 0.125, 0.125),
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Return train/val/test episode indices.

    ``temporal`` keeps episode order (train on earlier, test on later);
    ``random`` shuffles with ``seed``. Val and test sizes are floored, the
    remainder goes to train.
    """
    if len(fractions) != 3 or any(f < 0 for f in fractions) or sum(fractions) > 1 + 1e-9:
        raise ValueError("fractions must be three non-negative numbers summing to at most 1")
    if scheme not in ("temporal", "random"):
        raise ValueError("scheme must be 'temporal' or 'random'")
    n_val = int(np.floor(fractions[1] * n_episodes))
    n_test = int(np.floor(fractions[2] * n_episodes))
    n_train = n_episodes - n_val - n_test
    order = np.arange(n_episodes)
    if scheme == "random":
        order = np.random.default_rng(seed).permutation(n_episodes)
    return {
        "train": np.sort(order[:n_train]),
        "val": np.sort(order[n_train : n_train + n_val]),
        "test": np.sort(order[n_train + n_val :]),
    }
