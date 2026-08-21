"""Generate K-member ensembles for a set of held-out target steps.

Runs full-frame inference — the domain divides by the UNet's total downsample
factor (checked at dataset construction), so no tiling is needed. Output is a
single ``.npz`` with denormalised generated ensembles and truth per target
variable, consumed by ``evaluate.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import torch

from .config import Config
from .data import ZarrWindowDataset
from .edm import edm_heun_sampler
from .train import build_net


def load_precond(ckpt_path: str, device):
    sd = torch.load(ckpt_path, map_location=device)
    cfg = config_from_ckpt(sd["config"])
    net = build_net(cfg, "diffusion").to(device)
    net.load_state_dict(sd["ema"])  # EMA weights for sampling
    net.eval()
    return net, cfg


def config_from_ckpt(saved: dict) -> Config:
    """Rebuild a ``Config`` from a checkpoint, tolerating schema drift.

    Keys the current ``Config`` no longer has are dropped rather than raising, so
    a checkpoint written before a field was renamed or removed still loads; new
    fields simply take their defaults.
    """
    valid = {f.name for f in dataclasses.fields(Config)}
    dropped = sorted(set(saved) - valid)
    if dropped:
        print(f"[warn] ignoring unknown config keys in checkpoint: {dropped}")
    return Config(**{k: v for k, v in saved.items() if k in valid})


@torch.no_grad()
def sample_days(net, cfg: Config, ds: ZarrWindowDataset, days, k_members: int,
                device, member_batch: int = 4, seed: int = 0):
    Ct = cfg.target_channels()
    gen = {v: [] for v in cfg.target}
    truth = {v: [] for v in cfg.target}
    used_days = []
    g = torch.Generator(device=device)

    for item in ds.full_frames(days):
        cond = item["cond"].to(device)[None]           # (1, Cc, H, W)
        members = []
        remaining = k_members
        seed_off = 0
        while remaining > 0:
            m = min(member_batch, remaining)
            g.manual_seed(seed + 1000 * int(item["day"]) + seed_off)
            cond_b = cond.expand(m, -1, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x = edm_heun_sampler(
                    net, cond_b, Ct, num_steps=cfg.sampler_steps,
                    sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max, rho=cfg.rho,
                    s_churn=cfg.s_churn, s_noise=cfg.s_noise,
                    s_tmin=cfg.s_tmin, s_tmax=cfg.s_tmax,
                    generator=g,
                )
            members.append(x.float().cpu().numpy())    # (m, Ct, H, W)
            remaining -= m
            seed_off += 1
        members = np.concatenate(members, 0)           # (K, Ct, H, W)
        tgt = item["target"].numpy()                   # (Ct, H, W)
        day = int(item["day"])
        for ci, v in enumerate(cfg.target):
            gen[v].append(np.stack([ds.denormalize(v, members[j, ci], day)
                                    for j in range(k_members)], 0))
            truth[v].append(ds.denormalize(v, tgt[ci], day))
        used_days.append(day)
        print(f"  sampled day {int(item['day'])}  ({k_members} members)")

    out = {"days": np.array(used_days), "mask": ds.ocean,
           "target_vars": json.dumps(cfg.target),
           "config": json.dumps(cfg.as_dict())}
    for v in cfg.target:
        out[f"gen_{v}"] = np.stack(gen[v], 0)          # (n_days, K, H, W)
        out[f"truth_{v}"] = np.stack(truth[v], 0)      # (n_days, H, W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir or ckpt.pt path")
    ap.add_argument("--out", default=None, help="output .npz (default <ckpt>/ensembles.npz)")
    ap.add_argument("--k-members", type=int, default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="number of held-out days (0 = the whole split)")
    ap.add_argument("--split", default="val", help="split to draw target days from")
    ap.add_argument("--seed", type=int, default=0)
    # Store relocation. A checkpoint records the store path as it resolved at
    # training time, so sampling from a different site needs an override.
    ap.add_argument("--zarr-path", default=None,
                    help="override the checkpoint's store path (local or s3://)")
    ap.add_argument("--dataset", default=None,
                    help="override the checkpoint's dataset descriptor")
    # sampler overrides (the checkpoint's config wins otherwise)
    ap.add_argument("--sampler-steps", type=int, default=None)
    ap.add_argument("--s-churn", type=float, default=None)
    ap.add_argument("--s-noise", type=float, default=None)
    ap.add_argument("--s-tmin", type=float, default=None,
                    help="churn injected only for sigma >= s_tmin (default cfg 0.0)")
    ap.add_argument("--s-tmax", type=float, default=None,
                    help="churn injected only for sigma <= s_tmax (default cfg inf)")
    args = ap.parse_args()

    ckpt = args.ckpt if args.ckpt.endswith(".pt") else os.path.join(args.ckpt, "ckpt.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg = load_precond(ckpt, device)
    if args.zarr_path:
        cfg.zarr_path = args.zarr_path
    if args.dataset:
        cfg.dataset = args.dataset
    if args.k_members:
        cfg.k_members = args.k_members
    if args.sampler_steps is not None:
        cfg.sampler_steps = args.sampler_steps
    if args.s_churn is not None:
        cfg.s_churn = args.s_churn
    if args.s_noise is not None:
        cfg.s_noise = args.s_noise
    if args.s_tmin is not None:
        cfg.s_tmin = args.s_tmin
    if args.s_tmax is not None:
        cfg.s_tmax = args.s_tmax
    n_days = cfg.n_eval_days if args.days is None else args.days

    ds = ZarrWindowDataset(cfg, split=args.split)
    days = ds.default_eval_days(n_days)
    print(f"[sample] {len(days)} days x {cfg.k_members} members, targets={cfg.target}, "
          f"steps={cfg.sampler_steps}, s_churn={cfg.s_churn}, "
          f"s_tmin={cfg.s_tmin}, s_tmax={cfg.s_tmax}")

    out = sample_days(net, cfg, ds, days, cfg.k_members, device, seed=args.seed)
    out_path = args.out or os.path.join(os.path.dirname(ckpt), "ensembles.npz")
    np.savez_compressed(out_path, **out)
    print(f"[sample] wrote {out_path}")


if __name__ == "__main__":
    main()
