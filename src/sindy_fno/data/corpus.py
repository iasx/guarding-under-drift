"""Episode generation and HDF5 storage for the offline corpus."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from pensimpy.constants import NUM_STEPS

from sindy_fno.data.subsets import SUBSETS
from sindy_fno.plant import PenSimPlant, ScenarioConfig
from sindy_fno.plant.channels import ECHO, HIDDEN, LAB, RESP
from sindy_fno.plant.recipe import ACTION_NAMES
from sindy_fno.policies import RecipePerturbationPolicy
from sindy_fno.seeds import subset_seed

OBS_NAMES = RESP + ECHO + LAB


@dataclass
class Episode:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    hidden: np.ndarray
    assay_event: np.ndarray
    length: int
    yield_kg: float
    catastrophic: bool
    fs_scale: float
    seed: int
    scenario_json: str


@dataclass
class Corpus:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    hidden: np.ndarray
    assay_event: np.ndarray
    length: np.ndarray
    seed: np.ndarray
    yield_kg: np.ndarray
    fs_scale: np.ndarray
    catastrophic: np.ndarray
    scenario_json: list[str]
    obs_names: list[str]
    action_names: list[str]
    hidden_names: list[str]
    subset: str
    attrs: dict

    @property
    def n_episodes(self) -> int:
        return int(self.obs.shape[0])


def scenario_to_json(scenario: ScenarioConfig) -> str:
    return json.dumps(dataclasses.asdict(scenario), sort_keys=True)


def run_episode(
    scenario: ScenarioConfig,
    policy_kwargs: dict | None,
    seed: int,
    max_steps: int | None = None,
) -> Episode:
    """Roll out the behaviour policy on one batch. ``max_steps`` truncates (tests only)."""
    plant = PenSimPlant(scenario=scenario)
    policy = RecipePerturbationPolicy(**(policy_kwargs or {}))
    obs, info = plant.reset(seed=seed)
    policy.reset(seed)
    n_max = plant.max_episode_steps if max_steps is None else min(max_steps, plant.max_episode_steps)

    obs_l, act_l, rew_l, hid_l, assay_l = [], [], [], [], []
    catastrophic = False
    for _ in range(n_max):
        a = policy.act(obs, info)
        obs, r, terminated, _, info = plant.step(a)
        if info["catastrophic"]:
            catastrophic = True
            break
        obs_l.append(obs)
        act_l.append(info["action_applied"])
        rew_l.append(r)
        hid_l.append(info["hidden"])
        assay_l.append(info["assay_event"])
        if terminated:
            break

    reward = np.asarray(rew_l, dtype=float)
    return Episode(
        obs=np.asarray(obs_l, dtype=float),
        action=np.asarray(act_l, dtype=float),
        reward=reward,
        hidden=np.asarray(hid_l, dtype=float),
        assay_event=np.asarray(assay_l, dtype=bool),
        length=len(rew_l),
        yield_kg=float(reward.sum()),
        catastrophic=catastrophic,
        fs_scale=policy.fs_scale,
        seed=int(seed),
        scenario_json=scenario_to_json(scenario),
    )


def _episode_job(args: tuple) -> Episode:
    subset, episode, policy_kwargs, max_steps = args
    seed = subset_seed(subset, episode)
    scenario = SUBSETS[subset].sample(np.random.default_rng([seed, 0xC0]))
    return run_episode(scenario, policy_kwargs, seed, max_steps)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generate_subset(
    subset: str,
    n_episodes: int,
    out_dir: str | Path,
    workers: int | None = None,
    max_steps: int | None = None,
    policy_kwargs: dict | None = None,
) -> Path:
    """Generate ``n_episodes`` of one subset and write ``<out_dir>/<subset>.h5``."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; choose from {list(SUBSETS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(subset, i, policy_kwargs, max_steps) for i in range(n_episodes)]
    if workers == 1:
        episodes = [_episode_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            episodes = list(ex.map(_episode_job, jobs))

    n, T = len(episodes), NUM_STEPS
    obs = np.full((n, T, len(OBS_NAMES)), np.nan)
    action = np.full((n, T, len(ACTION_NAMES)), np.nan)
    reward = np.full((n, T), np.nan)
    hidden = np.full((n, T, len(HIDDEN)), np.nan)
    assay = np.zeros((n, T), dtype=bool)
    for i, ep in enumerate(episodes):
        L = ep.length
        obs[i, :L] = ep.obs
        action[i, :L] = ep.action
        reward[i, :L] = ep.reward
        hidden[i, :L] = ep.hidden
        assay[i, :L] = ep.assay_event

    path = out_dir / f"{subset}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("obs", data=obs, compression="gzip")
        f.create_dataset("action", data=action, compression="gzip")
        f.create_dataset("reward", data=reward, compression="gzip")
        f.create_dataset("hidden", data=hidden, compression="gzip")
        f.create_dataset("assay_event", data=assay)
        f.create_dataset("length", data=np.array([ep.length for ep in episodes], dtype=np.int64))
        f.create_dataset("seed", data=np.array([ep.seed for ep in episodes], dtype=np.int64))
        f.create_dataset("yield_kg", data=np.array([ep.yield_kg for ep in episodes]))
        f.create_dataset("fs_scale", data=np.array([ep.fs_scale for ep in episodes]))
        f.create_dataset("catastrophic", data=np.array([ep.catastrophic for ep in episodes], dtype=bool))
        f.create_dataset("scenario_json", data=[ep.scenario_json for ep in episodes], dtype=h5py.string_dtype())
        f.attrs["obs_names"] = OBS_NAMES
        f.attrs["action_names"] = ACTION_NAMES
        f.attrs["hidden_names"] = HIDDEN
        f.attrs["subset"] = subset
        f.attrs["scenario_sampler"] = SUBSETS[subset].description
        f.attrs["policy_kwargs"] = json.dumps(policy_kwargs or {}, sort_keys=True)
        f.attrs["git_hash"] = _git_hash()
        f.attrs["created"] = datetime.now(UTC).isoformat(timespec="seconds")
    return path


def load_subset(path: str | Path) -> Corpus:
    with h5py.File(path, "r") as f:
        names = {k: [str(s) for s in f.attrs[k]] for k in ("obs_names", "action_names", "hidden_names")}
        return Corpus(
            obs=f["obs"][()],
            action=f["action"][()],
            reward=f["reward"][()],
            hidden=f["hidden"][()],
            assay_event=f["assay_event"][()],
            length=f["length"][()],
            seed=f["seed"][()],
            yield_kg=f["yield_kg"][()],
            fs_scale=f["fs_scale"][()],
            catastrophic=f["catastrophic"][()],
            scenario_json=[s.decode() if isinstance(s, bytes) else str(s) for s in f["scenario_json"][()]],
            obs_names=names["obs_names"],
            action_names=names["action_names"],
            hidden_names=names["hidden_names"],
            subset=str(f.attrs["subset"]),
            attrs={k: (str(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in f.attrs.items()},
        )
