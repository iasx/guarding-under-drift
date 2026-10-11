"""Shared pieces for the learned comparators: a Q-critic over the window belief, a window replay, checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from sindy_fno.adaptive.agent import Actor, WindowEncoder
from sindy_fno.adaptive.ensemble import without_candidate
from sindy_fno.adaptive.windows import N_ACT, N_OBS, Normalizer


class QCritic(nn.Module):
    """Q(b, a) with the same window encoder as the actor."""

    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.enc = WindowEncoder(d)
        self.mlp = nn.Sequential(nn.Linear(d + N_ACT, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.enc(x), a], dim=1)).squeeze(-1)


def zero_residual_actor(d: int = 128) -> Actor:
    """An actor whose mean action is exactly zero at initialisation: the recipe, with exploration noise around it."""
    actor = Actor(d)
    last = actor.mlp[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        last.bias[N_ACT:] = -1.0   # raw log-std input: moderate exploration (log σ ≈ −2.3 after the squash)
    return actor


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for pt, ps in zip(target.parameters(), source.parameters()):
            pt.lerp_(ps, tau)


def save_agent(path: Path, actor: Actor, d: int, history: list, extra: dict | None = None) -> None:
    """Checkpoint loadable by ``AdaptivePolicy`` (actor weights and the encoder width)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()}, "config": {"d": d}, "history": history}
    payload.update(extra or {})
    torch.save(payload, path)


class WindowReplay:
    """Transitions (b_t, a_t, r_t, b_{t+1}, done) over recorded episodes, windows built on demand.

    Episodes are stored standardised (observations by the corpus statistics,
    feeds by the actuator bounds); the window layout is the surrogate's, and
    the belief handed to the networks has the candidate feed slot zeroed.
    """

    def __init__(self, norm: Normalizer, window: int, capacity_episodes: int = 400) -> None:
        self.norm, self.window, self.capacity = norm, window, capacity_episodes
        self.episodes: list[dict] = []
        self._index: np.ndarray | None = None

    def add(self, obs: np.ndarray, applied: np.ndarray, actions: np.ndarray, rewards: np.ndarray, costs: np.ndarray | None = None) -> None:
        """``obs[k]`` follows ``applied[k]`` (physical actuator values) chosen as plant-unit ``actions[k]`` with reward ``rewards[k]``
        and, for constrained learners, constraint cost ``costs[k]``."""
        self.episodes.append({
            "obs": self.norm.obs(np.asarray(obs, float)).astype(np.float32),
            "feeds": self.norm.feeds(np.asarray(applied, float)).astype(np.float32),
            "act": np.asarray(actions, np.float32), "rew": (np.asarray(rewards, np.float32) / self.norm.reward_std).astype(np.float32),
            "cost": np.zeros(len(rewards), np.float32) if costs is None else np.asarray(costs, np.float32),
        })
        del self.episodes[: max(0, len(self.episodes) - self.capacity)]
        self._index = None

    def __len__(self) -> int:
        return int(sum(max(0, len(e["obs"]) - self.window) for e in self.episodes))

    def _window(self, ep: dict, t: int) -> np.ndarray:
        N = self.window
        o = ep["obs"][t - N + 1 : t + 1]
        f = np.zeros((N, 6), np.float32)
        f[:-1] = ep["feeds"][t - N + 2 : t + 1]
        return np.concatenate([o, f], axis=1).T

    def sample(self, n: int, rng: np.random.Generator, device) -> tuple[torch.Tensor, ...]:
        if self._index is None:
            self._index = np.array([(e, t) for e, ep in enumerate(self.episodes) for t in range(self.window - 1, len(ep["obs"]) - 1)], dtype=np.int64)
        picks = self._index[rng.integers(len(self._index), size=n)]
        s, a, r, s2, d, c = [], [], [], [], [], []
        for e, t in picks:
            ep = self.episodes[e]
            s.append(self._window(ep, t))
            s2.append(self._window(ep, t + 1))
            a.append(ep["act"][t + 1])
            r.append(ep["rew"][t + 1])
            d.append(1.0 if t + 1 == len(ep["obs"]) - 1 else 0.0)
            c.append(ep["cost"][t + 1])
        self.last_costs = c
        to = lambda v: torch.as_tensor(np.stack(v), dtype=torch.float32, device=device)
        return to(s), to(a), to(r), to(s2), to(d)


class OnlineWindow:
    """Runtime belief builder matching ``AdaptivePolicy.observe`` (left-padded with the first observation)."""

    def __init__(self, norm: Normalizer, window: int, device) -> None:
        self.norm, self.N, self.device = norm, window, device
        self.x = None

    def reset(self) -> None:
        self.x = None

    def push(self, obs: np.ndarray, last_feeds_scaled: np.ndarray | None) -> torch.Tensor:
        o = torch.as_tensor(self.norm.obs(np.asarray(obs, float)), dtype=torch.float32, device=self.device)
        col = torch.cat([o, torch.zeros(6, device=self.device)]).unsqueeze(-1)
        if self.x is None:
            self.x = col.repeat(1, self.N).unsqueeze(0)
        else:
            if last_feeds_scaled is not None:
                self.x[0, N_OBS:, -1] = torch.as_tensor(last_feeds_scaled, dtype=torch.float32, device=self.device)
            self.x = torch.cat([self.x[:, :, 1:], col.unsqueeze(0)], dim=2)
        return without_candidate(self.x)
