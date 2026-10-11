"""Train the FNO surrogate ensemble on the corpus.

    uv run python -m sindy_fno.adaptive.train --corpus data/corpus --out models/fno --window 32 --members 5 --epochs 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sindy_fno.adaptive.fno import FNOConfig, FNOSurrogate, build_surrogate
from sindy_fno.adaptive.windows import (
    N_RESP,
    Normalizer,
    WindowArrays,
    WindowDataset,
    advance_window,
)
from sindy_fno.data.corpus import Corpus, load_subset
from sindy_fno.data.subsets import SUBSETS

MODE_CYCLE = [8, 12, 16]


def load_corpus(corpus_dir: Path) -> tuple[Corpus, dict[str, np.ndarray]]:
    """Concatenate all subsets into one Corpus and return episode indices per split."""
    index = pd.read_csv(corpus_dir / "index.csv")
    parts, seeds = [], []
    for name in SUBSETS:
        path = corpus_dir / f"{name}.h5"
        if path.exists():
            c = load_subset(path)
            parts.append(c)
            seeds.append(c.seed)
    first = parts[0]
    corpus = Corpus(
        obs=np.concatenate([c.obs for c in parts]), action=np.concatenate([c.action for c in parts]),
        reward=np.concatenate([c.reward for c in parts]), hidden=np.concatenate([c.hidden for c in parts]),
        assay_event=np.concatenate([c.assay_event for c in parts]), length=np.concatenate([c.length for c in parts]),
        seed=np.concatenate(seeds), yield_kg=np.concatenate([c.yield_kg for c in parts]),
        fs_scale=np.concatenate([c.fs_scale for c in parts]), catastrophic=np.concatenate([c.catastrophic for c in parts]),
        scenario_json=list(itertools.chain.from_iterable(c.scenario_json for c in parts)), obs_names=first.obs_names, action_names=first.action_names,
        hidden_names=first.hidden_names, subset="all", attrs={},
    )
    split_of = dict(zip(index.seed, index.split))
    splits = {k: np.array([i for i, s in enumerate(corpus.seed) if split_of.get(int(s)) == k]) for k in ("train", "val", "test")}
    return corpus, splits


def tensor_state(model: torch.nn.Module) -> dict:
    """State dict restricted to tensors: neuralop adds a ``_metadata`` item that load_state_dict rejects."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}


def multistep_loss(model, x, F, Y, R, norm, weight: float = 0.5):
    """One-step loss plus a weighted autoregressive loss over the horizon in the batch."""
    m = Y.shape[1]
    out = model(x)
    loss = torch.nn.functional.mse_loss(out[:, :N_RESP], Y[:, 0] - x[:, :N_RESP, -1]) + torch.nn.functional.mse_loss(out[:, N_RESP], R[:, 0])
    if m > 1 and weight > 0:
        xw, total = x, 0.0
        for h in range(m):
            o = model(xw) if h > 0 else out
            resp_next = xw[:, :N_RESP, -1] + o[:, :N_RESP]
            total = total + torch.nn.functional.mse_loss(resp_next, Y[:, h]) + torch.nn.functional.mse_loss(o[:, N_RESP], R[:, h])
            if h < m - 1:
                xw = advance_window(xw, resp_next, F[:, h], norm)
        loss = loss + weight * total / m
    return loss


def train_member(cfg: FNOConfig, train_arrays: WindowArrays, val_arrays: WindowArrays, norm: Normalizer, epochs: int, device, batch_size: int = 256, lr: float = 1e-3, log=print) -> tuple[FNOSurrogate, dict]:
    torch.manual_seed(cfg.seed)
    model = build_surrogate(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    train_loader = DataLoader(WindowDataset(train_arrays), batch_size=batch_size, shuffle=True, drop_last=len(train_arrays) > batch_size)
    val_loader = DataLoader(WindowDataset(val_arrays), batch_size=1024, shuffle=False)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        t0, run = time.perf_counter(), 0.0
        for x, F, Y, R in train_loader:
            x, F, Y, R = (t.to(device) for t in (x, F, Y, R))
            loss = multistep_loss(model, x, F, Y, R, norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for x, F, Y, R in val_loader:
                x, Y, R = x.to(device), Y.to(device), R.to(device)
                out = model(x)
                vl += float(torch.nn.functional.mse_loss(out[:, :N_RESP], Y[:, 0] - x[:, :N_RESP, -1]) + torch.nn.functional.mse_loss(out[:, N_RESP], R[:, 0])) * len(x)
                n += len(x)
        vl /= max(n, 1)
        history.append({"epoch": epoch, "train": run / max(len(train_loader), 1), "val": vl, "wall_s": time.perf_counter() - t0})
        log(f"  epoch {epoch}: train {history[-1]['train']:.4f} val {vl:.4f} ({history[-1]['wall_s']:.0f} s)")
        if vl < best:
            best, best_state = vl, tensor_state(model)
    model.load_state_dict(best_state)
    return model.cpu(), {"best_val": best, "history": history}


def train_ensemble(corpus_dir: Path, out_dir: Path, window: int, members: int, epochs: int, hidden: int, n_layers: int, horizon: int, device, seed0: int = 1000, batch_size: int = 256,
                   arch: str = "fno") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, splits = load_corpus(corpus_dir)
    train_eps, val_eps = splits["train"], splits["val"]
    reward_std = float(np.nanstd(corpus.reward[train_eps]))
    norm = Normalizer.from_corpus_dir(corpus_dir, reward_std)
    val_arrays = WindowArrays(corpus, norm, val_eps, window, horizon=1)
    names, records = [], []
    for k in range(members):
        rng = np.random.default_rng(seed0 + k)
        boot = rng.choice(train_eps, size=len(train_eps), replace=True) if members > 1 else train_eps
        cfg = FNOConfig(window=window, n_modes=MODE_CYCLE[k % len(MODE_CYCLE)], hidden=hidden, n_layers=n_layers, seed=seed0 + k, arch=arch)
        print(f"member {k}: modes {cfg.n_modes}, {len(np.unique(boot))} distinct training episodes")
        train_arrays = WindowArrays(corpus, norm, boot, window, horizon=horizon)
        model, rec = train_member(cfg, train_arrays, val_arrays, norm, epochs, device, batch_size=batch_size)
        name = f"member_{k}.pt"
        torch.save({"config": cfg.to_dict(), "state_dict": tensor_state(model), "record": rec}, out_dir / name)
        names.append(name)
        records.append({"member": k, "n_modes": cfg.n_modes, "best_val": rec["best_val"]})
    meta = {"members": names, "normalizer": json.loads(norm.to_json()), "window": window, "horizon": horizon, "epochs": epochs, "records": records}
    (out_dir / "ensemble.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--out", type=Path, default=Path("models/fno"))
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5, help="multi-step training horizon")
    p.add_argument("--sweep", type=int, nargs="*", default=None, help="window lengths to train one member each")
    p.add_argument("--arch", choices=("fno", "gru"), default="fno")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.sweep:
        for w in args.sweep:
            print(f"== window {w}")
            train_ensemble(args.corpus, args.out / f"sweep_w{w}", w, 1, args.epochs, args.hidden, args.layers, args.horizon, args.device)
    else:
        meta = train_ensemble(args.corpus, args.out, args.window, args.members, args.epochs, args.hidden, args.layers, args.horizon, args.device, arch=args.arch)
        print(json.dumps(meta["records"], indent=1))


if __name__ == "__main__":
    main()
