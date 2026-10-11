"""Online model-free SAC with a window encoder, learning by unrestricted plant interaction.

    uv run python -m sindy_fno.baselines.sac_online --episodes 240 --seed 0 --out models/sac_s0

The upper bound on adaptivity and the lower bound on safety: it acts on
the real plant from the start, with no model, no projection and no
fallback. Interaction budget: ``episodes`` full batches on nominal seeds,
one gradient update per plant step. The learning curve (yield per
episode) is compared with the offline-pretrained agent.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from sindy_fno.adaptive.agent import Actor
from sindy_fno.adaptive.ensemble import SurrogateEnsemble
from sindy_fno.baselines.common import OnlineWindow, QCritic, WindowReplay, save_agent, soft_update
from sindy_fno.plant import PenSimPlant, ScenarioConfig


@dataclass
class SACConfig:
    episodes: int = 240
    warmup_episodes: int = 5
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch: int = 256
    updates_per_step: int = 1
    target_entropy: float = -6.0
    seed: int = 0
    d: int = 128
    first_plant_seed: int = 1000
    max_steps: int = 10**9
    # SAC-Lagrangian: a cost critic on the two constraint indicators (true DO2 below the floor, volume above the vessel
    # limit) and a multiplier lambda <- max(0, lambda + lambda_lr * ((1 - gamma) * Qc - cost_limit)), i.e. a plain dual
    # ascent on the expected cost per step against a small limit. A softplus parametrisation with Adam was tried first
    # and died: once the policy was clean the parameter ran negative and the multiplier could not recover when costs returned.
    lagrangian: bool = False
    cost_limit: float = 0.005
    lambda_lr: float = 0.05
    # Discount of the cost critic. The vessel limit is reached at the end of a batch, hundreds of steps after the
    # feeds that cause it, so a cost critic at gamma = 0.99 (100-step horizon) barely sees it; 0.999 spans the batch.
    cost_gamma: float = 0.99

    def to_dict(self) -> dict:
        return asdict(self)


def train_sac(surrogate_dir: Path, out_dir: Path, cfg: SACConfig, device, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    norm_src = SurrogateEnsemble.load(surrogate_dir, "cpu")           # corpus statistics and window length only
    norm, N = norm_src.norm, norm_src.window
    actor = Actor(cfg.d).to(device)
    q1, q2, q1_t, q2_t = (QCritic(cfg.d).to(device) for _ in range(4))
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros((), device=device, requires_grad=True)
    qc, qc_t = QCritic(cfg.d).to(device), QCritic(cfg.d).to(device)
    qc_t.load_state_dict(qc.state_dict())
    opt_qc = torch.optim.AdamW(qc.parameters(), lr=cfg.lr)
    lam_value = 0.0                                                   # the multiplier, updated by dual ascent below
    opt_a, opt_alpha = torch.optim.AdamW(actor.parameters(), lr=cfg.lr), torch.optim.Adam([log_alpha], lr=cfg.lr)
    opt_q = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr)
    replay, window = WindowReplay(norm, N), OnlineWindow(norm, N, device)
    history, t0, n_updates = [], time.perf_counter(), 0

    def update():
        nonlocal n_updates, lam_value
        s, a, r, s2, done = replay.sample(cfg.batch, rng, device)
        cost = torch.as_tensor(np.asarray(replay.last_costs), dtype=torch.float32, device=device)
        alpha = log_alpha.exp().detach()
        lam = torch.tensor(lam_value, device=device) if cfg.lagrangian else torch.zeros((), device=device)
        with torch.no_grad():
            a2, logp2 = actor.sample(s2)
            y = r + cfg.gamma * (1 - done) * (torch.min(q1_t(s2, a2), q2_t(s2, a2)) - alpha * logp2)
        loss_q = torch.nn.functional.mse_loss(q1(s, a), y) + torch.nn.functional.mse_loss(q2(s, a), y)
        opt_q.zero_grad(set_to_none=True); loss_q.backward(); opt_q.step()
        if cfg.lagrangian:
            with torch.no_grad():
                yc = cost + cfg.cost_gamma * (1 - done) * qc_t(s2, a2)
            loss_qc = torch.nn.functional.mse_loss(qc(s, a), yc)
            opt_qc.zero_grad(set_to_none=True); loss_qc.backward(); opt_qc.step()
        a_new, logp = actor.sample(s)
        loss_pi = (alpha * logp - torch.min(q1(s, a_new), q2(s, a_new)) + lam * qc(s, a_new)).mean()
        opt_a.zero_grad(set_to_none=True); loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()
        loss_alpha = -(log_alpha * (logp.detach() + cfg.target_entropy)).mean()
        opt_alpha.zero_grad(set_to_none=True); loss_alpha.backward(); opt_alpha.step()
        soft_update(q1_t, q1, cfg.tau); soft_update(q2_t, q2, cfg.tau)
        if cfg.lagrangian:
            with torch.no_grad():
                excess = float((1 - cfg.cost_gamma) * qc(s, a_new).mean()) - cfg.cost_limit   # expected cost per step above the limit
            lam_value = max(0.0, lam_value + cfg.lambda_lr * excess)
            soft_update(qc_t, qc, cfg.tau)
        n_updates += 1
        return float(loss_q), float(loss_pi), float(alpha), lam_value

    for ep in range(cfg.episodes):
        plant = PenSimPlant(scenario=ScenarioConfig())
        obs, info = plant.reset(seed=cfg.first_plant_seed + ep)
        window.reset()
        obs_l, app_l, act_l, rew_l, cost_l = [obs.copy()], [info["action_applied"].copy()], [np.zeros(6, np.float32)], [0.0], [0.0]
        last_feeds, total, losses = None, 0.0, []
        b = window.push(obs, None)
        for _ in range(min(plant.max_episode_steps, cfg.max_steps)):
            if ep < cfg.warmup_episodes:
                a = np.clip(rng.normal(0.0, 0.3, size=6), -1, 1)      # recipe with exploration noise fills the replay
            else:
                with torch.no_grad():
                    a = actor.sample(b)[0][0].cpu().numpy()
            obs, r, terminated, _, info = plant.step(a)
            total += r
            last_feeds = norm.feeds(info["action_applied"])
            obs_l.append(obs.copy()); app_l.append(info["action_applied"].copy()); act_l.append(a.astype(np.float32)); rew_l.append(r)
            hid = dict(zip(plant.hidden_names, info["hidden"]))
            cost_l.append(float(hid["DO2"] < 4.0) + float(hid["V"] > 1e5))
            b = window.push(obs, last_feeds)
            if ep >= cfg.warmup_episodes and len(replay) >= cfg.batch:
                for _ in range(cfg.updates_per_step):
                    losses.append(update())
            if terminated:
                break
        replay.add(np.array(obs_l), np.array(app_l), np.array(act_l), np.array(rew_l), np.array(cost_l))
        hn = plant.hidden_names
        rec = {"episode": ep, "plant_seed": cfg.first_plant_seed + ep, "yield_kg": total, "catastrophic": bool(info["catastrophic"]),
               "steps": len(rew_l) - 1, "updates": n_updates, "wall_s": time.perf_counter() - t0,
               "min_DO2": float(min(h[hn.index("DO2")] for h in [info["hidden"]])) if not info["catastrophic"] else float("nan"),
               "cost_steps": float(np.sum(cost_l))}
        if losses:
            rec.update({"q_loss": float(np.mean([v[0] for v in losses])), "pi_loss": float(np.mean([v[1] for v in losses])), "alpha": losses[-1][2], "lambda": losses[-1][3]})
        history.append(rec)
        log(f"  episode {ep:4d}: yield {total:8.1f} kg  catastrophic {rec['catastrophic']}  updates {n_updates}  ({rec['wall_s'] / 60:.1f} min)")
        if ep % 10 == 0 or ep == cfg.episodes - 1:
            save_agent(out_dir / "agent.pt", actor, cfg.d, history, {"learner": "sac-lagrangian" if cfg.lagrangian else "sac-online", "config_sac": cfg.to_dict()})
            (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return {"history": history, "config": cfg.to_dict()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surrogate", type=Path, default=Path("models/fno"), help="source of the corpus normaliser and window length")
    p.add_argument("--out", type=Path, default=Path("models/sac_s0"))
    p.add_argument("--episodes", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lagrangian", action="store_true", help="SAC-Lagrangian on the two constraint indicators")
    p.add_argument("--cost-limit", type=float, default=0.005)
    p.add_argument("--cost-gamma", type=float, default=0.99)
    args = p.parse_args(argv)
    train_sac(args.surrogate, args.out, SACConfig(episodes=args.episodes, seed=args.seed, first_plant_seed=1000 + 300 * args.seed,
                                                  lagrangian=args.lagrangian, cost_limit=args.cost_limit, cost_gamma=args.cost_gamma), args.device)


if __name__ == "__main__":
    main()
