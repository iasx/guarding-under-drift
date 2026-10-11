"""Actor and critic over the window belief."""

from __future__ import annotations

import math

import torch
from torch import nn

from sindy_fno.adaptive.windows import IN_CH, N_ACT

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


class WindowEncoder(nn.Module):
    def __init__(self, d: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(IN_CH, hidden, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.GELU(),
        )
        self.out = nn.Sequential(nn.Linear(2 * hidden, d), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.out(torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1))


class Actor(nn.Module):
    """tanh-squashed Gaussian policy in the plant's action cube."""

    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.enc = WindowEncoder(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * N_ACT))

    def dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = self.mlp(self.enc(x)).chunk(2, dim=1)
        # Smooth squash of the log-std into its range: a hard clamp has zero
        # gradient at the bound and let the std collapse to the floor.
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(raw) + 1)
        return mean, log_std

    @staticmethod
    def _log_prob(pre: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = log_std.exp()
        gauss = -0.5 * ((pre - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        squash = torch.log1p(-torch.tanh(pre) ** 2 + 1e-6)
        return (gauss - squash).sum(dim=1)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised action in [-1, 1]^6 and its log-probability."""
        a, logp, _ = self.sample_with_pre(x)
        return a, logp

    def sample_with_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """As ``sample`` but also returns the pre-tanh activation (for saturation penalties)."""
        mean, log_std = self.dist(x)
        pre = mean + log_std.exp() * torch.randn_like(mean)
        return torch.tanh(pre), self._log_prob(pre, mean, log_std), pre

    def mean_action(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dist(x)[0])

    def log_prob(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.dist(x)
        pre = torch.atanh(a.clamp(-0.999, 0.999))
        return self._log_prob(pre, mean, log_std)


class Critic(nn.Module):
    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.enc = WindowEncoder(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.enc(x)).squeeze(-1)
