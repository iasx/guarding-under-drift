"""Disjoint seed ranges so data generation, training and evaluation never share randomness."""

DATA_SEEDS = range(1000)
TRAIN_SEEDS = range(1000, 2000)
EVAL_SEEDS = range(2000, 3000)

# Offsets inside DATA_SEEDS for each corpus subset.
SUBSET_SEED_OFFSET = {"nom": 0, "noise": 200, "dist": 400, "init": 600}


def subset_seed(subset: str, episode: int) -> int:
    if episode < 0 or episode >= 200:
        raise ValueError("at most 200 episodes per subset fit in the seed range")
    return DATA_SEEDS.start + SUBSET_SEED_OFFSET[subset] + episode
