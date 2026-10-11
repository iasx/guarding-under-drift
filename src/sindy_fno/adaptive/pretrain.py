"""Offline actor-critic pretraining in imagination with analytic gradients.

    uv run python -m sindy_fno.adaptive.pretrain --corpus data/corpus --surrogate models/fno --out models/agent --updates 20000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from pensimpy.constants import STEP_IN_HOURS

from sindy_fno.adaptive.agent import Actor, Critic
from sindy_fno.adaptive.ensemble import SurrogateEnsemble, with_candidate, without_candidate
from sindy_fno.adaptive.train import load_corpus
from sindy_fno.adaptive.windows import (
    N_OBS,
    WindowArrays,
    actions_from_feeds,
    advance_window,
    feeds_from_actions,
    window_t_h,
)


@dataclass
class PretrainConfig:
    horizon: int = 20
    gamma: float = 0.99
    lam_gae: float = 0.95
    lam_pess: float = 1.0
    # u is the member spread of a standardised one-step *increment* (order 0.05) while r̂/σ_r is order 1; with
    # u_ref = 1 the penalty is a few percent of the reward and λ cannot matter. u_ref = median u over training
    # windows (``--u-scaled``) puts λ in units of σ_r per typical uncertainty.
    u_ref: float = 1.0
    alpha_entropy: float = 3e-3
    eta_bc: float = 1.0
    pre_penalty: float = 1e-3     # on the squared pre-tanh activation, against saturation
    tau_target: float = 0.01      # EMA rate of the target critic
    batch: int = 128
    lr: float = 1e-4
    actor_clip: float = 1.0
    critic_clip: float = 10.0
    updates: int = 20000
    eval_every: int = 500
    seed: int = 0
    d: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """TD(λ) targets: ``rewards`` (B, H), ``values`` (B, H+1) with the bootstrap at H → (B, H)."""
    H = rewards.shape[1]
    out = [None] * H
    nxt = values[:, H]
    for h in reversed(range(H)):
        nxt = rewards[:, h] + gamma * ((1 - lam) * values[:, h + 1] + lam * nxt)
        out[h] = nxt
    return torch.stack(out, dim=1)


def batch_feeds_from_actions(actions: torch.Tensor, t_h: torch.Tensor, norm) -> torch.Tensor:
    """Scaled feeds for a batch of actions applied at per-sample times (vectorised recipe lookup)."""
    return feeds_from_actions(actions, t_h, norm)


def imagine(ens: SurrogateEnsemble, actor: Actor, critic: Critic, b0: torch.Tensor, cfg: PretrainConfig, rng: np.random.Generator) -> dict:
    """Roll the actor through the ensemble from real start windows; returns per-step tensors with gradients.

    Rewards are in units of the corpus reward standard deviation:
    ``r̃ = r̂/σ_r − λ·u`` (pessimism, eq. 7 of the method, with ``u`` in
    standardised response units). ``critic`` provides the values used for
    the λ-returns (pass the target critic during training).
    """
    H = cfg.horizon
    t_obs = window_t_h(b0, ens.norm)
    xw = without_candidate(b0)
    member = int(rng.integers(ens.size)) if ens.size > 1 else None
    rewards, us, logps, values, windows, pres = [], [], [], [], [], []
    for h in range(H):
        a, logp, pre = actor.sample_with_pre(xw)
        feeds = batch_feeds_from_actions(a, t_obs + STEP_IN_HOURS * (h + 1), ens.norm)
        xc = with_candidate(xw, feeds)
        out = ens.step(xc, member)
        windows.append(xw)
        values.append(critic(xw))
        rewards.append(out["reward"] / ens.norm.reward_std - cfg.lam_pess * out["u"] / cfg.u_ref)
        us.append(out["u"])
        logps.append(logp)
        pres.append(pre)
        xw = without_candidate(advance_window(xc, out["resp"], torch.zeros_like(feeds), ens.norm))
    values.append(critic(xw))
    return {
        "rewards": torch.stack(rewards, 1), "u": torch.stack(us, 1), "logp": torch.stack(logps, 1), "pre": torch.stack(pres, 1),
        "values": torch.stack(values, 1), "windows": windows, "final_window": xw,
    }


def bc_targets(x: torch.Tensor, norm) -> torch.Tensor:
    """Corpus actions in plant units: invert the recipe map on the recorded feeds of the last slot."""
    return actions_from_feeds(x[:, N_OBS:, -1], window_t_h(x, norm) + STEP_IN_HOURS, norm)


def sample_windows(arrays: WindowArrays, n: int, rng: np.random.Generator, device) -> torch.Tensor:
    picks = arrays.index[rng.integers(len(arrays.index), size=n)]
    return torch.stack([torch.from_numpy(arrays.window_at(e, t)) for e, t in picks]).to(device)


def imagination_updates(ens: SurrogateEnsemble, actor: Actor, critic: Critic, critic_target: Critic, opt_a, opt_c, sample_starts, cfg: PretrainConfig,
                        n_updates: int, rng: np.random.Generator, log=print, eval_windows: torch.Tensor | None = None, checkpoint=None,
                        history: list | None = None, start_update: int = 0) -> list[dict]:
    """``n_updates`` actor/critic updates in imagination from windows drawn by ``sample_starts(n)``.

    Shared by offline pretraining and the online stage; ``checkpoint(history)``
    is called at every evaluation. Returns the (extended) history.
    """
    history = [] if history is None else history
    kg = ens.norm.reward_std   # imagined returns are logged in kg
    t0 = time.perf_counter()
    for i in range(n_updates):
        it = start_update + i
        b0 = sample_starts(cfg.batch)
        im = imagine(ens, actor, critic_target, b0, cfg, rng)
        returns = lambda_returns(im["rewards"], im["values"], cfg.gamma, cfg.lam_gae)
        entropy = -im["logp"].mean()
        loss_actor = -returns[:, 0].mean() - cfg.alpha_entropy * entropy + cfg.pre_penalty * im["pre"].pow(2).mean()
        loss_bc = -actor.log_prob(without_candidate(b0), bc_targets(b0, ens.norm)).mean()
        opt_a.zero_grad(set_to_none=True)
        (loss_actor + cfg.eta_bc * loss_bc).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.actor_clip))
        opt_a.step()
        # Critic regresses the imagined windows on λ-targets bootstrapped by the target critic.
        w = torch.cat([wi.detach() for wi in im["windows"]], dim=0)
        loss_critic = torch.nn.functional.mse_loss(critic(w), returns.detach().T.reshape(-1))
        opt_c.zero_grad(set_to_none=True)
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_clip)
        opt_c.step()
        with torch.no_grad():
            for pt, pc in zip(critic_target.parameters(), critic.parameters()):
                pt.lerp_(pc, cfg.tau_target)
        if it % cfg.eval_every == 0 or i == n_updates - 1:
            rec = {"update": it, "imagined_return_train": im["rewards"].detach().sum(1).mean().item() * kg,
                   "critic_loss": loss_critic.item(), "bc_loss": loss_bc.item(), "entropy": entropy.item(), "actor_grad_norm": gn,
                   "lam": cfg.lam_pess, "eta": cfg.eta_bc, "wall_s": time.perf_counter() - t0, "s_per_update": (time.perf_counter() - t0) / (i + 1)}
            if eval_windows is not None:
                with torch.no_grad():
                    imv = imagine(ens, actor, critic_target, eval_windows, cfg, rng)
                rec["imagined_return_val"] = float(imv["rewards"].sum(1).mean()) * kg
                rec["u_val"] = float(imv["u"].mean())
            history.append(rec)
            ret_val = rec.get("imagined_return_val", rec["imagined_return_train"])
            log(f"  {it:6d}: return val {ret_val:8.2f} train {rec['imagined_return_train']:8.2f} u {rec.get('u_val', float('nan')):.3f} "
                f"critic {rec['critic_loss']:.3f} bc {rec['bc_loss']:.3f} ent {rec['entropy']:.2f} gn {gn:.2f} ({rec['s_per_update']:.2f} s/update)")
            if not np.isfinite(ret_val):
                raise RuntimeError("imagined return is not finite")
            if checkpoint is not None:
                checkpoint(history)
    return history


def pretrain(corpus_dir: Path, surrogate_dir: Path, out_dir: Path, cfg: PretrainConfig, device, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ens = SurrogateEnsemble.load(surrogate_dir, device)
    corpus, splits = load_corpus(corpus_dir)
    train = WindowArrays(corpus, ens.norm, splits["train"], ens.window, horizon=1)
    val = WindowArrays(corpus, ens.norm, splits["val"], ens.window, horizon=1)
    actor, critic = Actor(cfg.d).to(device), Critic(cfg.d).to(device)
    critic_target = Critic(cfg.d).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p_ in critic_target.parameters():
        p_.requires_grad_(False)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=cfg.lr)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=cfg.lr)
    val_windows = sample_windows(val, 128, np.random.default_rng(1), device)
    if cfg.u_ref <= 0:   # sentinel from the CLI: measure the typical one-step spread on training windows
        with torch.no_grad():
            b = sample_windows(train, 512, np.random.default_rng(2), device)
            cfg.u_ref = float(ens.step(with_candidate(without_candidate(b), b[:, N_OBS:, -1]))["u"].median())
        log(f"  u_ref (median one-step ensemble spread on training windows) = {cfg.u_ref:.4f}")

    def checkpoint(history):
        # Checkpoint at every evaluation so progress is visible and resumable.
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "config": cfg.to_dict(), "history": history}, out_dir / "agent.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    history = imagination_updates(ens, actor, critic, critic_target, opt_a, opt_c, lambda n: sample_windows(train, n, rng, device), cfg, cfg.updates, rng,
                                  log=log, eval_windows=val_windows, checkpoint=checkpoint)
    return {"history": history, "config": cfg.to_dict()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--surrogate", type=Path, default=Path("models/fno"))
    p.add_argument("--out", type=Path, default=Path("models/agent"))
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--lam", type=float, default=1.0, help="pessimism weight λ (0 = negative control)")
    p.add_argument("--eta", type=float, default=1.0, help="behaviour-cloning anchor weight")
    p.add_argument("--u-scaled", action="store_true", help="scale the pessimism penalty by the median training-window spread")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    cfg = PretrainConfig(updates=args.updates, horizon=args.horizon, lam_pess=args.lam, eta_bc=args.eta, eval_every=args.eval_every, seed=args.seed,
                         u_ref=-1.0 if args.u_scaled else 1.0)
    pretrain(args.corpus, args.surrogate, args.out, cfg, args.device)


if __name__ == "__main__":
    main()
