"""Fourier neural operator surrogate over the window belief."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from neuralop.models import FNO
from torch import nn

from sindy_fno.adaptive.windows import IN_CH, OUT_CH


@dataclass
class FNOConfig:
    window: int = 32
    n_modes: int = 12
    hidden: int = 48
    n_layers: int = 3
    seed: int = 0
    arch: str = "fno"      # "fno" or "gru" (a non-spectral sequence model behind the same interface)

    def to_dict(self) -> dict:
        return asdict(self)


class FNOSurrogate(nn.Module):
    """FNO along the time window, read out at the last grid point.

    Input (B, 32, N): standardised observations and applied feeds per past
    step. Output (B, 14): standardised one-step change of the 13 response
    channels and the standardised reward of the next step.
    """

    def __init__(self, cfg: FNOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        modes = max(1, min(cfg.n_modes, cfg.window // 2))
        self.fno = FNO(n_modes=(modes,), in_channels=IN_CH, out_channels=cfg.hidden, hidden_channels=cfg.hidden, n_layers=cfg.n_layers)
        self.head = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, OUT_CH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fno(x)[:, :, -1]
        return self.head(h)


class GRUSurrogate(nn.Module):
    """Non-spectral counterpart of ``FNOSurrogate``: a GRU over the window, read out at the last step."""

    def __init__(self, cfg: FNOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(IN_CH, cfg.hidden)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, OUT_CH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.gelu(self.inp(x.transpose(1, 2)))
        # The agent differentiates through a frozen (eval-mode) surrogate; cuDNN's RNN backward refuses that,
        # the native kernel does not.
        if not self.training and torch.is_grad_enabled():
            with torch.backends.cudnn.flags(enabled=False):
                h, _ = self.gru(z)
        else:
            h, _ = self.gru(z)
        return self.head(h[:, -1])


def build_surrogate(cfg: FNOConfig) -> nn.Module:
    return GRUSurrogate(cfg) if cfg.arch == "gru" else FNOSurrogate(cfg)
