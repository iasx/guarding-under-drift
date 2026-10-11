"""Window belief b_t = H_t^N: construction, standardisation, differentiable advance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.data.corpus import Corpus
from sindy_fno.plant.channels import ECHO, LAB, RESP
from sindy_fno.plant.recipe import ACTION_MAX, ACTION_MIN, ACTION_NAMES, recipe_values

OBS_NAMES = RESP + ECHO + LAB
N_OBS = len(OBS_NAMES)
N_ACT = len(ACTION_NAMES)
IN_CH = N_OBS + N_ACT          # 32
N_RESP = len(RESP)             # 13
OUT_CH = N_RESP + 1            # response changes + reward
RESP_IDX = list(range(N_RESP))
ECHO_FEED_IDX = [OBS_NAMES.index(n) for n in ACTION_NAMES]   # echo channels that mirror the feeds
T_IDX = OBS_NAMES.index("t_h")
LAB_IDX = [OBS_NAMES.index(n) for n in LAB]
AGE_IDX = OBS_NAMES.index("lab_age_h")


@dataclass
class Normalizer:
    """Standardisation of observations (corpus statistics) and feeds (actuator bounds)."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    reward_std: float

    @classmethod
    def from_corpus_dir(cls, corpus_dir: str | Path, reward_std: float) -> Normalizer:
        d = json.loads(Path(corpus_dir, "standardizer.json").read_text(encoding="utf-8"))
        assert d["names"] == OBS_NAMES, "standardizer channel order differs from the plant's"
        return cls(np.asarray(d["mean"], float), np.maximum(np.asarray(d["std"], float), 1e-6), float(reward_std))

    def obs(self, o: np.ndarray) -> np.ndarray:
        return (o - self.obs_mean) / self.obs_std

    def feeds(self, u: np.ndarray) -> np.ndarray:
        return (u - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) * 2 - 1

    def feeds_inverse(self, f):
        lo, hi = torch.as_tensor(ACTION_MIN, dtype=f.dtype, device=f.device), torch.as_tensor(ACTION_MAX, dtype=f.dtype, device=f.device)
        return (f + 1) / 2 * (hi - lo) + lo

    def to_json(self) -> str:
        return json.dumps({"obs_mean": self.obs_mean.tolist(), "obs_std": self.obs_std.tolist(), "reward_std": self.reward_std})

    @classmethod
    def from_json(cls, data: str | dict) -> Normalizer:
        d = json.loads(data) if isinstance(data, str) else data
        return cls(np.asarray(d["obs_mean"], float), np.asarray(d["obs_std"], float), float(d["reward_std"]))


_RECIPE_TABLE: dict[tuple, torch.Tensor] = {}


def recipe_table(dtype=torch.float32, device="cpu", n_steps: int = 1300) -> torch.Tensor:
    """Recipe values of the six feeds at every plant step time k·Δt, k = 0 … n_steps (cached per dtype/device)."""
    key = (str(dtype), str(device))
    if key not in _RECIPE_TABLE:
        rows = [[recipe_values(k * STEP_IN_HOURS)[n] for n in ACTION_NAMES] for k in range(n_steps + 1)]
        _RECIPE_TABLE[key] = torch.tensor(rows, dtype=dtype, device=device)
    return _RECIPE_TABLE[key]


def nominal_feeds(t_h, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Recipe feeds (physical) at times ``t_h`` (tensor of any shape) via the step table."""
    t = torch.as_tensor(t_h, dtype=torch.float64, device=device)
    k = torch.clamp(torch.round(t / STEP_IN_HOURS).long(), 0, recipe_table(dtype, device).shape[0] - 1)
    return recipe_table(dtype, device)[k]


def feeds_from_actions(actions, t_h, norm: Normalizer, deviation_max: float = 0.5):
    """Scaled feed slot for plant-unit actions (torch, differentiable) applied during the step starting at ``t_h``.

    ``t_h`` may be a scalar or a per-sample tensor of shape (B,).
    """
    nominal = nominal_feeds(t_h, actions.dtype, actions.device)
    if nominal.dim() == 1:
        nominal = nominal.unsqueeze(0)
    lo = torch.as_tensor(ACTION_MIN, dtype=actions.dtype, device=actions.device)
    hi = torch.as_tensor(ACTION_MAX, dtype=actions.dtype, device=actions.device)
    phys = torch.clamp(nominal * (1 + deviation_max * actions), lo, hi)
    return (phys - lo) / (hi - lo) * 2 - 1


def actions_from_feeds(feeds_scaled: torch.Tensor, t_h, norm: Normalizer, deviation_max: float = 0.5) -> torch.Tensor:
    """Inverse map: plant-unit actions for scaled feeds applied at ``t_h`` (zero where the recipe is zero)."""
    phys = norm.feeds_inverse(feeds_scaled)
    nominal = nominal_feeds(t_h, feeds_scaled.dtype, feeds_scaled.device)
    if nominal.dim() == 1:
        nominal = nominal.unsqueeze(0)
    a = torch.where(nominal > 0, (phys / nominal.clamp_min(1e-9) - 1) / deviation_max, torch.zeros_like(phys))
    return a.clamp(-1.0, 1.0)


class WindowArrays:
    """Standardised corpus arrays plus a valid-window index."""

    def __init__(self, corpus: Corpus, norm: Normalizer, episodes: np.ndarray, window: int, horizon: int = 1):
        self.window, self.horizon = window, horizon
        self.obs = norm.obs(np.nan_to_num(corpus.obs[episodes])).astype(np.float32)
        self.feeds = norm.feeds(np.nan_to_num(corpus.action[episodes])).astype(np.float32)
        self.rew = (np.nan_to_num(corpus.reward[episodes]) / norm.reward_std).astype(np.float32)
        self.lengths = corpus.length[episodes]
        idx = []
        for e, L in enumerate(self.lengths):
            for t in range(window - 1, int(L) - 1 - horizon):
                idx.append((e, t))
        self.index = np.array(idx, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def window_at(self, e: int, t: int) -> np.ndarray:
        """(32, N): observations at t-N+1..t and the feeds applied during the following steps."""
        N = self.window
        o = self.obs[e, t - N + 1 : t + 1]                 # (N, 26)
        f = self.feeds[e, t - N + 2 : t + 2]               # (N, 6): applied during j+1
        return np.concatenate([o, f], axis=1).T

    def item(self, i: int):
        e, t = self.index[i]
        m = self.horizon
        x = self.window_at(e, t)
        F = self.feeds[e, t + 2 : t + 2 + m]               # feeds for steps t+2 .. t+m+1
        Y = self.obs[e, t + 1 : t + 1 + m, : N_RESP]        # response at t+1 .. t+m
        R = self.rew[e, t + 1 : t + 1 + m]
        return x, F, Y, R


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, arrays: WindowArrays):
        self.a = arrays

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, i: int):
        x, F, Y, R = self.a.item(i)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(F)), torch.from_numpy(np.ascontiguousarray(Y)), torch.from_numpy(np.ascontiguousarray(R))


def advance_window(x: torch.Tensor, resp_next: torch.Tensor, next_feeds: torch.Tensor, norm: Normalizer) -> torch.Tensor:
    """Slide the window one step (differentiable).

    ``x`` (B, 32, N) is the window at t; ``resp_next`` (B, 13) the predicted
    standardised response at t+1; ``next_feeds`` (B, 6) the scaled feeds for
    the step t+1 → t+2. Echo channels of the new observation are the feeds
    that were in the last slot (applied during t → t+1), time advances by
    one step, lab channels are held with their age advanced.
    """
    B = x.shape[0]
    dev, dt = x.device, x.dtype
    mean = torch.as_tensor(norm.obs_mean, dtype=dt, device=dev)
    std = torch.as_tensor(norm.obs_std, dtype=dt, device=dev)
    last = x[:, :, -1]
    new_obs = torch.zeros(B, N_OBS, dtype=dt, device=dev)
    new_obs[:, :N_RESP] = resp_next
    applied_phys = norm.feeds_inverse(last[:, N_OBS:])
    new_obs[:, ECHO_FEED_IDX] = (applied_phys - mean[ECHO_FEED_IDX]) / std[ECHO_FEED_IDX]
    new_obs[:, T_IDX] = last[:, T_IDX] + STEP_IN_HOURS / std[T_IDX]
    new_obs[:, LAB_IDX] = last[:, LAB_IDX]
    new_obs[:, AGE_IDX] = last[:, AGE_IDX] + STEP_IN_HOURS / std[AGE_IDX]
    new_col = torch.cat([new_obs, next_feeds], dim=1).unsqueeze(-1)
    return torch.cat([x[:, :, 1:], new_col], dim=2)


def window_t_h(x: torch.Tensor, norm: Normalizer) -> torch.Tensor:
    """Batch time (hours) of the last observation in each window."""
    return x[:, T_IDX, -1] * float(norm.obs_std[T_IDX]) + float(norm.obs_mean[T_IDX])
