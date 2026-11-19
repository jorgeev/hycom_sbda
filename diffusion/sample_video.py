"""Prior ensembles of multi-frame windows from one network or two experts.

  python -m diffusion.sample_video --small <run> --large <run>      # two experts
  python -m diffusion.sample_video --single <run>                   # one network

``<run>`` is a run directory (its ``ckpt.pt``) or a ``.pt`` file. With two
experts the large-sigma network denoises above ``sigma_divide`` and the
small-sigma one below (:class:`diffusion.edm_video.ExpertSwitch`).

For each chosen window start the script takes that window's calendar inputs
(and labels, when the model has any) from the store, generates ``K`` members,
and writes generated and true fields in physical units to one ``.npz``.
``--labels zero`` zeroes the store-variable labels (``label_vars``) only: tidal
phase labels come from the window's timestamps and are always given.

    gen_<var>    (n_windows, K, T, H, W)
    truth_<var>  (n_windows, T, H, W)
    starts (n_windows,)   time_unix (n_windows, T)   pos (n_windows, 2)
    mask (n_windows, H, W)   target_vars, config_small, config_large, meta (JSON)

The network runs convolutionally on whatever size is asked for (``--size``,
default the training patch). ``--size full`` uses the largest top-left region
that divides by the network's downsample factor -- 576x928 of the 576x936
gulfstream grid for a 5-level network; there is no tiling.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import Config
from .data_video import ZarrVideoDataset
from .edm_video import ExpertSwitch, heun_sampler_video
from .sample import config_from_ckpt
from .train_video import build_video_precond

# Keys the two experts must agree on: they define the state, the inputs and the
# normalisation, i.e. everything that makes their outputs interchangeable.
_PAIR_KEYS = ("dataset", "target", "cond_vars", "label_vars", "tidal_labels",
              "tidal_epoch", "use_ocean_mask",
              "use_calendar", "calendar_harmonics", "n_frames", "frame_stride",
              "norm_mode", "clim_vars", "clim_mode", "clim_full_record")


def resolve_ckpt(path: str) -> str:
    return path if path.endswith(".pt") else os.path.join(path, "ckpt.pt")


def load_expert(path: str, device, weights: str = "ema"):
    """``(precond, cfg, step)`` from a checkpoint, EMA weights by default."""
    sd = torch.load(resolve_ckpt(path), map_location=device, weights_only=False)
    cfg = config_from_ckpt(sd["config"])
    net = build_video_precond(cfg).to(device)
    net.load_state_dict(sd[weights])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, cfg, int(sd["step"]) + 1


def check_pair(small: Config, large: Config, sigma_divide: float,
               sigma_min: float, sigma_max: float) -> list[str]:
    """Hard errors (returned) and soft warnings (printed) for an expert pair."""
    a, b = small.as_dict(), large.as_dict()
    errors = [f"{k}: small has {a[k]!r}, large has {b[k]!r}"
              for k in _PAIR_KEYS if a[k] != b[k]]
    for name, cfg in (("small", small), ("large", large)):
        if cfg.sigma_sampling != "loguniform":
            print(f"[warn] the {name} expert was trained with {cfg.sigma_sampling} "
                  "sigma, not a fixed log-uniform range; range checks skipped")
            return errors
    if not (large.train_sigma_min <= sigma_divide <= small.train_sigma_max):
        print(f"[warn] sigma_divide {sigma_divide} is outside the experts' overlap "
              f"[{large.train_sigma_min}, {small.train_sigma_max}]: one expert "
              "will be queried at sigmas it never trained on")
    if sigma_max > large.train_sigma_max * 1.001:
        print(f"[warn] sampler sigma_max {sigma_max} exceeds the large expert's "
              f"training range (max {large.train_sigma_max})")
    if sigma_min < small.train_sigma_min * 0.999:
        print(f"[warn] sampler sigma_min {sigma_min} is below the small expert's "
              f"training range (min {small.train_sigma_min})")
    return errors


def pick_starts(ds: ZarrVideoDataset, n: int) -> list[int]:
    """``n`` window starts spread evenly over the split (``n <= 0`` = all)."""
    v = ds.valid_days
    if n <= 0 or n >= v.size:
        return v.tolist()
    return v[np.linspace(0, v.size - 1, n).round().astype(int)].tolist()


def window_slices(ds: ZarrVideoDataset, size, start: int, seed: int):
    """Crop for one window: ``"full"`` or an ``N`` x ``N`` box at a seeded corner."""
    if size == "full":
        return ds.full_slices()
    n = int(size)
    rng = np.random.default_rng([seed, int(start)])
    if n == ds.cfg.patch and ds.positions is not None:
        y0, x0 = ds.positions[int(rng.integers(len(ds.positions)))]
    else:
        y0 = int(rng.integers(ds.NY - n + 1))
        x0 = int(rng.integers(ds.NX - n + 1))
    return slice(y0, y0 + n), slice(x0, x0 + n)


@torch.no_grad()
def sample_windows(switch: ExpertSwitch, cfg: Config, ds: ZarrVideoDataset, starts,
                   k_members: int, device, *, size="patch", sampler: str = "heun",
                   sampler_kw: dict | None = None, labels: str = "truth",
                   member_batch: int = 2, seed: int = 0) -> dict:
    size = cfg.patch if size == "patch" else size
    sampler_kw = sampler_kw or {}
    ct, T = len(cfg.target), cfg.n_frames
    gen = {v: [] for v in cfg.target}
    truth = {v: [] for v in cfg.target}
    masks, pos, times = [], [], []
    g = torch.Generator(device=device)

    for start in starts:
        ysl, xsl = window_slices(ds, size, start, seed)
        item = ds.window(start, ysl, xsl)
        h, w = item["mask"].shape[-2:]
        lab = item["labels"]
        if labels == "zero":
            # only the store-variable labels: the tidal phases that follow them
            # are known for any date, and an all-zero phase is not a phase
            lab = lab.clone()
            lab[:, : len(cfg.label_vars)] = 0.0
        base = {"cond": item["cond"], "labels": lab, "sod": item["sod"],
                "doy": item["doy"], "lon": item["lon"]}
        members, remaining, off = [], k_members, 0
        while remaining > 0:
            m = min(member_batch, remaining)
            ctx = {k: v.to(device)[None].expand(m, *v.shape).contiguous()
                   for k, v in base.items()}
            switch.bind(ctx)
            g.manual_seed(seed + 1000 * int(start) + off)
            shape = (m, T, ct, h, w)
            if sampler == "heun":
                x = heun_sampler_video(switch, shape, generator=g, device=device,
                                       **sampler_kw)
            elif sampler == "vp":
                from sda.vpsde import vp_sample
                x = vp_sample(switch, shape, generator=g, device=device, **sampler_kw)
            else:
                raise ValueError(f"Unknown sampler {sampler!r}")
            members.append(x.float().cpu().numpy())
            remaining -= m
            off += 1
        members = np.concatenate(members, 0)                 # (K, T, Ct, H, W)
        tgt = item["target"].numpy()                         # (T, Ct, H, W)
        for ci, v in enumerate(cfg.target):
            gen[v].append(ds.denormalize_window(v, members[:, :, ci], start, ysl, xsl)
                          .astype(np.float32))
            truth[v].append(ds.denormalize_window(v, tgt[:, ci], start, ysl, xsl)
                            .astype(np.float32))
        masks.append(item["mask"][0].numpy())
        pos.append((ysl.start, xsl.start))
        times.append(ds.t_unix[ds.frames_of(start)])
        print(f"  window start {int(start)} at ({ysl.start}, {xsl.start}), "
              f"{h}x{w}: {k_members} members", flush=True)

    out = {"starts": np.array([int(s) for s in starts]), "time_unix": np.stack(times),
           "pos": np.array(pos), "mask": np.stack(masks),
           "target_vars": json.dumps(cfg.target)}
    for v in cfg.target:
        out[f"gen_{v}"] = np.stack(gen[v], 0)                # (n, K, T, H, W)
        out[f"truth_{v}"] = np.stack(truth[v], 0)            # (n, T, H, W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=None, help="small-sigma expert (run dir or .pt)")
    ap.add_argument("--large", default=None, help="large-sigma expert (run dir or .pt)")
    ap.add_argument("--single", default=None, help="one network for all sigma")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--k-members", type=int, default=None)
    ap.add_argument("--windows", type=int, default=None,
                    help="number of window starts, spread over the split (0 = all)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--size", default="patch",
                    help="'patch' (training patch), 'full', or an integer N")
    ap.add_argument("--labels", default="truth", choices=["truth", "zero"],
                    help="label_vars scalars from the store window, or zeros "
                         "(tidal phase labels are always given)")
    ap.add_argument("--member-batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-days", type=int, default=None,
                    help="truncate the time axis (quick tests; use --split train)")
    ap.add_argument("--dataset", default=None, help="override the dataset descriptor")
    ap.add_argument("--zarr-path", default=None, help="override the store path")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR"),
                    help="look for the climatology / norm caches here instead of "
                         "the path recorded at training time (default $CACHE_DIR)")
    ap.add_argument("--sigma-divide", type=float, default=None)
    ap.add_argument("--sampler", default="heun", choices=["heun", "vp"])
    # heun (EDM) settings; the checkpoint config wins otherwise
    ap.add_argument("--sampler-steps", type=int, default=None)
    ap.add_argument("--sigma-min", type=float, default=None)
    ap.add_argument("--sigma-max", type=float, default=None)
    ap.add_argument("--s-churn", type=float, default=None)
    ap.add_argument("--s-noise", type=float, default=None)
    # vp (sda.vpsde.vp_sample) settings
    ap.add_argument("--vp-steps", type=int, default=256)
    ap.add_argument("--corrections", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()

    two = bool(args.small and args.large)
    if not two and not args.single:
        raise SystemExit("pass --small and --large, or --single")
    if two and args.single:
        raise SystemExit("--single cannot be combined with --small/--large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if two:
        net_s, cfg_s, step_s = load_expert(args.small, device, args.weights)
        net_l, cfg_l, step_l = load_expert(args.large, device, args.weights)
        cfg = cfg_l                      # sampler bounds follow the large expert
    else:
        net_s, cfg_s, step_s = load_expert(args.single, device, args.weights)
        net_l, cfg_l, step_l = None, None, None
        cfg = cfg_s
    for k, v in (("dataset", args.dataset), ("zarr_path", args.zarr_path),
                 ("max_days", args.max_days), ("k_members", args.k_members),
                 ("sampler_steps", args.sampler_steps), ("sigma_min", args.sigma_min),
                 ("sigma_max", args.sigma_max), ("s_churn", args.s_churn),
                 ("s_noise", args.s_noise), ("sigma_divide", args.sigma_divide)):
        if v is not None:
            setattr(cfg, k, v)
            if two and k in ("dataset", "zarr_path"):
                setattr(cfg_s, k, v)

    # A checkpoint records the cache paths as they resolved on the training
    # host (e.g. /mnt/runs/cache on AWS). Elsewhere that directory may not even
    # be creatable, so re-root both caches when told where they live here.
    if args.cache_dir:
        for k in ("clim_cache", "norm_cache"):
            setattr(cfg, k, os.path.join(args.cache_dir, os.path.basename(getattr(cfg, k))))

    if two:
        errors = check_pair(cfg_s, cfg_l, cfg.sigma_divide, cfg.sigma_min, cfg.sigma_max)
        if errors:
            raise SystemExit("[sample] the two experts are not interchangeable:\n  - "
                             + "\n  - ".join(errors))
    switch = ExpertSwitch(net_s, net_l, sigma_divide=cfg.sigma_divide)

    if args.size not in ("patch", "full"):
        down = 2 ** (len(cfg.channel_mult) - 1)
        if int(args.size) % down:
            raise SystemExit(f"--size {args.size} is not divisible by {down}")
    size = args.size if args.size in ("patch", "full") else int(args.size)

    ds = ZarrVideoDataset(cfg, split=args.split)
    n_windows = cfg.n_eval_days if args.windows is None else args.windows
    starts = pick_starts(ds, n_windows)
    if args.sampler == "heun":
        sampler_kw = dict(num_steps=cfg.sampler_steps, sigma_min=cfg.sigma_min,
                          sigma_max=cfg.sigma_max, rho=cfg.rho, s_churn=cfg.s_churn,
                          s_noise=cfg.s_noise, s_tmin=cfg.s_tmin, s_tmax=cfg.s_tmax)
    else:
        sampler_kw = dict(steps=args.vp_steps, corrections=args.corrections, tau=args.tau)
    print(f"[sample] {'two experts, switch at ' + str(cfg.sigma_divide) if two else 'single network'}; "
          f"{len(starts)} windows x {cfg.k_members} members x {cfg.n_frames} frames, "
          f"size={size}, targets={cfg.target}")
    print(f"[sample] sampler={args.sampler} {sampler_kw}")

    out = sample_windows(switch, cfg, ds, starts, cfg.k_members, device, size=size,
                         sampler=args.sampler, sampler_kw=sampler_kw,
                         labels=args.labels, member_batch=args.member_batch,
                         seed=args.seed)
    out["config_small"] = json.dumps(cfg_s.as_dict())
    out["config_large"] = json.dumps(cfg_l.as_dict() if two else None)
    out["meta"] = json.dumps({
        "two_experts": two, "sigma_divide": cfg.sigma_divide, "weights": args.weights,
        "step_small": step_s, "step_large": step_l, "sampler": args.sampler,
        "sampler_kw": {k: (v if np.isfinite(v) else str(v)) for k, v in sampler_kw.items()},
        "size": size, "labels": args.labels, "split": args.split, "seed": args.seed})
    base = args.large if two else args.single
    out_path = args.out or os.path.join(os.path.dirname(resolve_ckpt(base)),
                                        "ensembles_video.npz")
    np.savez_compressed(out_path, **out)
    print(f"[sample] wrote {out_path}")


if __name__ == "__main__":
    main()
