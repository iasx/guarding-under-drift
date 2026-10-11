"""Surrogate ensemble: prediction with disagreement-based uncertainty and differentiable imagination rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sindy_fno.adaptive.fno import FNOConfig, FNOSurrogate, build_surrogate
from sindy_fno.adaptive.windows import N_OBS, N_RESP, Normalizer, advance_window


def with_candidate(x: torch.Tensor, feeds: torch.Tensor) -> torch.Tensor:
    """Window ``x`` (B, 32, N) with the last feed slot replaced by ``feeds`` (B, 6)."""
    last = torch.cat([x[:, :N_OBS, -1], feeds], dim=1).unsqueeze(-1)
    return torch.cat([x[:, :, :-1], last], dim=2)


def without_candidate(x: torch.Tensor) -> torch.Tensor:
    """Window with the last feed slot zeroed: what a policy sees before choosing."""
    return with_candidate(x, torch.zeros_like(x[:, N_OBS:, -1]))


class SurrogateEnsemble:
    def __init__(self, members: list[FNOSurrogate], norm: Normalizer, device: str | torch.device = "cpu") -> None:
        self.members = [m.to(device).eval() for m in members]
        for m in self.members:
            for p in m.parameters():
                p.requires_grad_(False)
        self.norm = norm
        self.device = torch.device(device)
        self.window = members[0].cfg.window

    @classmethod
    def load(cls, models_dir: str | Path, device: str | torch.device = "cpu", members: list[int] | None = None) -> SurrogateEnsemble:
        """``members`` selects a subset by index (a single model has zero spread, hence no pessimism)."""
        models_dir = Path(models_dir)
        meta = json.loads((models_dir / "ensemble.json").read_text(encoding="utf-8"))
        norm = Normalizer.from_json(meta["normalizer"])
        names = meta["members"] if members is None else [meta["members"][i] for i in members]
        members = []
        for name in names:
            ckpt = torch.load(models_dir / name, map_location="cpu", weights_only=False)
            m = build_surrogate(FNOConfig(**ckpt["config"]))
            m.load_state_dict({k: v for k, v in ckpt["state_dict"].items() if torch.is_tensor(v)})
            members.append(m)
        return cls(members, norm, device)

    @property
    def size(self) -> int:
        return len(self.members)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and member-std of the 14 standardised outputs for windows ``x`` (B, 32, N)."""
        outs = torch.stack([m(x) for m in self.members])
        return outs.mean(0), outs.std(0, unbiased=False)

    def step(self, x: torch.Tensor, member: int | None = None) -> dict[str, torch.Tensor]:
        """One imagined step from a window whose last feed slot holds the candidate feeds.

        Returns the standardised next response ``resp`` (B, 13), the reward in
        kg ``reward`` (B,), and the uncertainty ``u`` (B,) as the L1 mean over
        response channels of the member spread. ``member`` selects one
        member's prediction for the mean path while ``u`` still uses all.
        """
        outs = torch.stack([m(x) for m in self.members])
        mean, std = outs.mean(0), outs.std(0, unbiased=False)
        out = mean if member is None else outs[member]
        return {
            "resp": x[:, :N_RESP, -1] + out[:, :N_RESP],
            "reward": out[:, N_RESP] * self.norm.reward_std,
            "u": std[:, :N_RESP].abs().mean(dim=1),
        }

    def rollout(self, x: torch.Tensor, feeds: torch.Tensor, mode: str = "mean") -> dict[str, torch.Tensor]:
        """Imagine H steps from windows ``x`` (B, 32, N) under scaled feeds (B, H, 6).

        ``feeds[:, h]`` drives the step t+h → t+h+1 (``feeds[:, 0]`` replaces the
        candidate slot of ``x``). Gradients flow to ``feeds``.
        """
        H = feeds.shape[1]
        member = int(mode.split(":")[1]) if mode.startswith("member") else None
        xw = with_candidate(x, feeds[:, 0])
        resp_seq, rew_seq, u_seq = [], [], []
        for h in range(H):
            out = self.step(xw, member)
            resp_seq.append(out["resp"])
            rew_seq.append(out["reward"])
            u_seq.append(out["u"])
            if h < H - 1:
                xw = advance_window(xw, out["resp"], feeds[:, h + 1], self.norm)
        return {"resp": torch.stack(resp_seq, 1), "reward": torch.stack(rew_seq, 1), "u": torch.stack(u_seq, 1), "window": xw}
