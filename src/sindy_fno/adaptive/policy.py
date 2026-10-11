"""Runtime policy of the adaptive branch: maintains the window belief on the plant and acts from it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.adaptive.agent import Actor, Critic
from sindy_fno.adaptive.ensemble import SurrogateEnsemble, with_candidate, without_candidate
from sindy_fno.adaptive.windows import (
    ECHO_FEED_IDX,
    IN_CH,
    N_OBS,
    N_RESP,
    advance_window,
    feeds_from_actions,
)


class AdaptivePolicy:
    """Implements the ``Policy`` protocol with the pretrained actor over the surrogate ensemble's window.

    ``act`` proposes an action; ``commit`` records the action actually
    executed (which may differ after arbitration and projection) in the
    window's candidate slot and returns the ensemble's prediction for it.
    With ``auto_commit`` (the default, for the policy on its own) ``act``
    commits its own proposal.
    """

    def __init__(self, surrogate_dir: str | Path, agent_path: str | Path | None = None, device: str = "cpu", deterministic: bool = True, auto_commit: bool = True,
                 members: list[int] | None = None) -> None:
        self.ens = SurrogateEnsemble.load(surrogate_dir, device, members)
        self.norm = self.ens.norm
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.auto_commit = auto_commit
        self.actor = None
        self.critic = None
        if agent_path is not None:
            ckpt = torch.load(agent_path, map_location="cpu", weights_only=False)
            self.actor = Actor(ckpt["config"]["d"]).to(self.device).eval()
            self.actor.load_state_dict(ckpt["actor"])
            if "critic" in ckpt:
                self.critic = Critic(ckpt["config"]["d"]).to(self.device).eval()
                self.critic.load_state_dict(ckpt["critic"])
        self.window = None
        self._x = None
        self.t_h = 0.0
        self._hist_x: list[torch.Tensor] = []       # policy-view windows at past steps
        self._hist_feeds: list[torch.Tensor] = []   # feeds applied at those steps (actuator feedback)
        self.feed_clip_steps = 0                    # steps on which the plant applied something else than commanded

    def reset(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.window = None
        self._hist_x, self._hist_feeds = [], []
        self.feed_clip_steps = 0

    def observe(self, obs: np.ndarray) -> torch.Tensor:
        """Push a plant observation.

        The feed slot of the previous position is overwritten with the feed
        the plant reports having applied (the echo channels of ``obs``), not
        the one ``commit`` was given: the actuator rate limit and clipping can
        alter the command, and the surrogate was trained on applied feeds.
        The hindsight history is corrected the same way.
        """
        obs = np.asarray(obs, float)
        o = torch.as_tensor(self.norm.obs(obs), dtype=torch.float32, device=self.device)
        col = torch.cat([o, torch.zeros(IN_CH - N_OBS, device=self.device)]).unsqueeze(-1)
        if self.window is None:
            self.window = col.repeat(1, self.ens.window).unsqueeze(0)
        else:
            applied = torch.as_tensor(self.norm.feeds(obs[ECHO_FEED_IDX]), dtype=torch.float32, device=self.device)
            if self._hist_feeds:
                if not torch.allclose(self._hist_feeds[-1][0], applied, atol=1e-3):
                    self.feed_clip_steps += 1
                self._hist_feeds[-1] = applied.reshape(1, -1)
            self.window = self.window.clone()
            self.window[0, N_OBS:, -1] = applied
            self.window = torch.cat([self.window[:, :, 1:], col.unsqueeze(0)], dim=2)
        return self.window

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        x = self.observe(obs)
        self._x = without_candidate(x)
        self.t_h = float(info["t_h"])
        with torch.no_grad():
            if self.actor is None:
                a = torch.zeros(1, IN_CH - N_OBS, device=self.device)
            else:
                a = self.actor.mean_action(self._x) if self.deterministic else self.actor.sample(self._x)[0]
        a_np = a[0].cpu().numpy()
        if self.auto_commit:
            self.commit(a_np)
        return a_np

    def _feeds(self, action: np.ndarray) -> torch.Tensor:
        a = torch.as_tensor(np.asarray(action, float), dtype=torch.float32, device=self.device).reshape(1, -1)
        return feeds_from_actions(a, self.t_h + STEP_IN_HOURS, self.norm)

    def commit(self, action: np.ndarray, history: int = 130) -> tuple[np.ndarray, float]:
        """Record the executed action; return the predicted next response (physical units) and its uncertainty."""
        with torch.no_grad():
            feeds = self._feeds(action)
            self.window = with_candidate(self._x, feeds)
            out = self.ens.step(self.window)
        self._hist_x.append(self._x)
        self._hist_feeds.append(feeds)
        if len(self._hist_x) > history:
            del self._hist_x[0], self._hist_feeds[0]
        resp = out["resp"][0].cpu().numpy() * self.norm.obs_std[:N_RESP] + self.norm.obs_mean[:N_RESP]
        return resp, float(out["u"][0])

    def hindsight(self, horizon: int) -> np.ndarray | None:
        """Prediction of the current observation's response from the window ``horizon`` steps ago with the executed feeds.

        Call after ``act`` (so the history holds the previous steps) and
        before ``commit``. Returns physical units, or None while the history
        is shorter than the horizon.
        """
        if horizon < 1 or len(self._hist_feeds) < horizon:
            return None
        with torch.no_grad():
            x0 = self._hist_x[-horizon]
            F = torch.stack(self._hist_feeds[-horizon:], dim=1)      # (1, H, 6)
            out = self.ens.rollout(x0, F)
        return out["resp"][0, -1].cpu().numpy() * self.norm.obs_std[:N_RESP] + self.norm.obs_mean[:N_RESP]

    def predict_next(self) -> tuple[np.ndarray, float]:
        """The ensemble's prediction for the currently committed action."""
        with torch.no_grad():
            out = self.ens.step(self.window)
        resp = out["resp"][0].cpu().numpy() * self.norm.obs_std[:N_RESP] + self.norm.obs_mean[:N_RESP]
        return resp, float(out["u"][0])

    def value_after(self, action: np.ndarray) -> float:
        """Critic value (kg) of the window one imagined step after ``action``; for the value-space arbitration law."""
        if self.critic is None:
            raise RuntimeError("no critic loaded")
        with torch.no_grad():
            feeds = self._feeds(action)
            xc = with_candidate(self._x, feeds)
            out = self.ens.step(xc)
            nxt = without_candidate(advance_window(xc, out["resp"], torch.zeros_like(feeds), self.norm))
            return float(self.critic(nxt)[0]) * self.norm.reward_std
