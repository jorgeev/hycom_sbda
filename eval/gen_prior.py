"""Draw UNCONSTRAINED samples from the trained prior, plus matched real fields.

Nothing conditions these samples. ``prior_gulfstream.yaml`` sets ``cond_vars:
[]``, ``use_ocean_mask: false`` and ``use_doy: false``, so ``cond_channels()``
is 0 and the sampler gets a zero-width conditioning tensor: pure ``p(x)`` draws,
tied to no day, no observation and no location.

Two geometries, because they answer different questions:

  --size full   576x936, the whole domain. This is how the prior will be used in
                assimilation, and how GenDA runs its own inference. The UNet is
                fully convolutional so it works, but the single attention block
                was built for 16x16 tokens (patch 128 >> 3 levels) and here runs
                on 72x117 -- an extrapolation.
  --size 128    exactly the training patch distribution, so nothing is
                extrapolated and the statistics are the cleanest read of what
                was actually learned.

ON UNITS. The model outputs ANOMALIES. ``sst`` and ``sss`` carry a monthly
climatology (``clim_vars: [sst, sss]``) that dwarfs the mesoscale -- 20.2 degC in
March to 26.4 degC in July -- and which the network was deliberately never asked
to model. Real val frames span 2017-05-26 to 2017-07-01, so denormalising them by
their own day would put that seasonal spread into the comparison and the sst PDF
would look wrong for a reason that has nothing to do with the model. So BOTH the
generated and the real fields are denormalised with the SAME ``--ref-day``
climatology. Everything downstream is then apples-to-apples in physical units.
The raw anomalies are saved too, and are what the spectra and EKE actually use.

COST. ``ZarrWindowDataset`` preloads every channel of the whole 4367-hour axis:
7 x 4367 x 576 x 936 x 4 B = ~66 GB of host RAM. Measured on the COAPS node:
**~56 min cold, ~2.5 min once the OS page cache holds the store** (the box has
1 TB, so the second run of the day is cheap; a fresh boot or a busy node is
not). Decompression is single-threaded, which is what makes the cold case slow.
That is the price of using the loader as the single source of truth for
normalisation rather than reimplementing it here -- worth paying once, which is
why ``--size`` takes a comma-separated list and every geometry is served from
one load. Generation itself is minutes: ~4.3 s per full frame, ~0.12 s per patch.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m eval.gen_prior \
        --ckpt /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt \
        --out-dir <run>/eval_step1140000

writes ``samples_full.npz`` and ``samples_128.npz`` into that directory.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from diffusion.dataset_spec import resolve_spec
from diffusion.data import ZarrWindowDataset
from diffusion.edm import edm_heun_sampler
from diffusion.sample import config_from_ckpt
from diffusion.train import build_net


def load_ckpt(path: str, device):
    """``(net, cfg, meta)`` from a checkpoint, using the EMA weights.

    Mirrors ``diffusion.sample.load_precond`` but also hands back the training
    metadata, so every figure can be stamped with the step it came from instead
    of leaving provenance to the filename.
    """
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_ckpt(sd["config"])
    net = build_net(cfg, "diffusion").to(device)
    net.load_state_dict(sd["ema"])          # EMA weights, as sample.py does
    net.eval()
    meta = {"ckpt": os.path.abspath(path),
            "step": int(sd.get("step", -1)) + 1,
            "val_loss": float(sd["val_loss"]) if "val_loss" in sd else None,
            "best_val": float(sd["best_val"]) if "best_val" in sd else None}
    return net, cfg, meta


def pick_real_days(valid_days: np.ndarray, n: int, skip_tail: int,
                   allow_repeat: bool = False) -> np.ndarray:
    """``n`` indices spread evenly across the split.

    NOT the last n. The base cadence here is hourly, so the final n steps are a
    single weather state and would badly understate how much real fields vary --
    which is exactly the quantity every diagnostic below compares against.

    ``skip_tail`` drops the end of the record: it finishes 2017-07-01 23:00, so
    July's monthly climatology is constrained by 24 hours alone and the ridge in
    ``fit_climatology`` shrinks it back toward the time-mean. Those steps are not
    clean anomalies (see the config comment).
    """
    pool = valid_days[:-skip_tail] if skip_tail > 0 else valid_days
    if pool.size < n and not allow_repeat:
        raise ValueError(f"split has {pool.size} usable steps, asked for {n}")
    take = min(n, pool.size)
    days = pool[np.linspace(0, pool.size - 1, take).round().astype(int)]
    if n > take:
        # Patch mode: more crops than there are steps. Cycling the list draws
        # several crops from one frame, which is exactly what training does
        # (crops_per_day: 8) -- and they are different crops, so they are not
        # duplicate samples.
        days = np.resize(days, n)
    return days


@torch.no_grad()
def generate(net, cfg, n: int, hw: tuple[int, int], batch: int, device,
             seed: int, sampler: dict) -> np.ndarray:
    """``n`` unconditional draws of shape (n, Ct, H, W), in anomaly units."""
    ct = cfg.target_channels()
    h, w = hw
    out = []
    g = torch.Generator(device=device)
    done = 0
    t0 = time.time()
    while done < n:
        b = min(batch, n - done)
        g.manual_seed(seed + done)
        # cond_channels() == 0: a zero-width tensor. EDMPrecond concatenates it
        # to the noised target, and edm_heun_sampler reads B/H/W off its shape.
        cond = torch.zeros(b, cfg.cond_channels(), h, w, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            x = edm_heun_sampler(net, cond, ct, generator=g, **sampler)
        out.append(x.float().cpu().numpy())
        done += b
        el = time.time() - t0
        print(f"  [gen] {done}/{n}  {el:6.1f}s elapsed  "
              f"{el / done:5.2f}s/sample", flush=True)
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="run dir or ckpt .pt path")
    ap.add_argument("--out-dir", required=True,
                    help="directory for samples_<size>.npz")
    # Comma-separated so one dataset load serves every geometry. The load is by
    # far the most expensive part of this script (see the module docstring), and
    # running it once per geometry would double an hour of it for nothing.
    ap.add_argument("--size", default="full,128",
                    help="comma-separated geometries: 'full' and/or a patch edge")
    ap.add_argument("--n", default=None,
                    help="samples per geometry, comma-separated "
                         "(default 48 full / 512 patch)")
    ap.add_argument("--batch", default=None,
                    help="sampler batch per geometry, comma-separated "
                         "(default 8 full / 64 patch)")
    ap.add_argument("--split", default="val", help="split the real fields come from")
    # The real reference fields do not depend on the checkpoint, so a sweep over
    # several checkpoints should pay the ~1 h dataset load once. Point this at an
    # earlier run's output directory and the store is never opened.
    ap.add_argument("--real-from", default=None,
                    help="reuse real fields + denorm reference from an earlier "
                         "--out-dir (skips the dataset load entirely)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-tail", type=int, default=24,
                    help="trailing base-cadence steps to exclude (default 24 h)")
    ap.add_argument("--ref-day", type=int, default=None,
                    help="climatology day for BOTH gen and real "
                         "(default: median of the selected real steps)")
    # Sampler overrides; the checkpoint's config wins otherwise.
    # --s-churn 0 gives the deterministic Heun path (EDM Algorithm 1), which is
    # the cleaner probe of the learned score; the config's 10.0 is what the
    # ensemble machinery uses to decorrelate members.
    ap.add_argument("--sampler-steps", type=int, default=None)
    ap.add_argument("--s-churn", type=float, default=None)
    ap.add_argument("--s-noise", type=float, default=None)
    args = ap.parse_args()

    ckpt = args.ckpt if args.ckpt.endswith(".pt") else os.path.join(args.ckpt, "ckpt.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg, meta = load_ckpt(ckpt, device)
    print(f"[ckpt] {ckpt}\n[ckpt] step={meta['step']} val_loss={meta['val_loss']} "
          f"best_val={meta['best_val']}")

    # This script is for UNCONDITIONAL priors only. On a conditional checkpoint
    # a zero cond tensor is not "no conditioning" -- in normalised units zero is
    # the mean, so it would quietly produce samples conditioned on an average
    # observation and every figure downstream would look reasonable and mean
    # nothing. Use diffusion.sample for conditional checkpoints.
    if cfg.cond_channels() != 0:
        raise SystemExit(
            f"{ckpt} is a CONDITIONAL model (cond_channels="
            f"{cfg.cond_channels()}, cond_vars={cfg.cond_vars}, "
            f"use_ocean_mask={cfg.use_ocean_mask}, use_doy={cfg.use_doy}).\n"
            "eval.gen_prior draws unconstrained p(x) samples and has no "
            "observations to condition on. Use `python -m diffusion.sample`.")

    sizes = [s.strip() for s in args.size.split(",") if s.strip()]
    # The domain must divide by the UNet's total downsample factor or full-frame
    # inference dies inside a skip connection. dataset_spec.validate() checks this
    # for the store's own shape; --size can name anything, so check it here.
    factor = 2 ** (len(cfg.channel_mult) - 1)
    for sz in sizes:
        if sz != "full" and int(sz) % factor:
            raise SystemExit(f"--size {sz} is not divisible by the UNet "
                             f"downsample factor {factor}")
    defaults = {"full": (48, 8)}
    ns = _per_size(args.n, sizes, [defaults.get(s, (512, 64))[0] for s in sizes], "n")
    bs = _per_size(args.batch, sizes, [defaults.get(s, (512, 64))[1] for s in sizes],
                   "batch")

    sampler = dict(num_steps=cfg.sampler_steps, sigma_min=cfg.sigma_min,
                   sigma_max=cfg.sigma_max, rho=cfg.rho, s_churn=cfg.s_churn,
                   s_noise=cfg.s_noise, s_tmin=cfg.s_tmin, s_tmax=cfg.s_tmax)
    if args.sampler_steps is not None:
        sampler["num_steps"] = args.sampler_steps
    if args.s_churn is not None:
        sampler["s_churn"] = args.s_churn
    if args.s_noise is not None:
        sampler["s_noise"] = args.s_noise

    # The store: NOT via cfg.zarr_path. That field still holds the gom_nemo
    # default, because resolve_spec only back-fills it for single-store families
    # and gulfstream has seven. Reading dx/lat from it would silently use the
    # wrong grid.
    spec = resolve_spec(cfg)
    dx_m = spec.dx_m()
    lat_full = np.asarray(spec.family.coord("coords/lat"), dtype=np.float32)

    if args.real_from:
        ds = None
        print(f"[data] reusing real fields from {args.real_from} "
              f"-- the 66 GB record is not read", flush=True)
    else:
        print(f"[data] loading {cfg.target} over the {spec.n_steps()}-step axis "
              f"(~66 GB, ~1 h -- zarr decompression is single-threaded)", flush=True)
        t0 = time.time()
        ds = ZarrWindowDataset(cfg, split=args.split)
        print(f"[data] ready in {time.time() - t0:.0f}s; "
              f"{ds.valid_days.size} valid steps in split {args.split!r}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for size, n, batch in zip(sizes, ns, bs):
        out = os.path.join(args.out_dir, f"samples_{size}.npz")
        print(f"\n=== {size}: {n} samples, batch {batch} -> {out} ===", flush=True)
        run_size(net, cfg, ds, lat_full, dx_m, size, n, batch, device,
                 args, dict(meta), sampler, out)


def _per_size(spec_str, sizes, defaults, what):
    """Parse a comma-separated per-geometry option, or fall back to defaults."""
    if spec_str is None:
        return defaults
    vals = [int(x) for x in str(spec_str).split(",")]
    if len(vals) == 1:
        return vals * len(sizes)
    if len(vals) != len(sizes):
        raise SystemExit(f"--{what} has {len(vals)} values but --size has {len(sizes)}")
    return vals


def _denorm_reference(ds, cfg, lat_full, full, hw, ref_day):
    """``(hw, mask, lat, clim_ref, std, lat_note)`` for one geometry.

    Both generated and real fields are denormalised with this same reference-day
    climatology; see the module docstring for why.
    """
    std = np.array([ds.denorm_params(v, ref_day)[1] for v in cfg.target],
                   dtype=np.float64)
    clim = np.stack([np.broadcast_to(ds.denorm_params(v, ref_day)[0],
                                     (ds.NY, ds.NX)) for v in cfg.target], 0)
    if full:
        return (hw, (ds.ocean > 0.5).astype(np.float32), lat_full,
                clim.astype(np.float32), std, "per-pixel coords/lat")
    # A generated 128x128 patch is not located anywhere -- the model has no
    # positional conditioning -- so there is no per-pixel latitude or climatology
    # to give it. Use the domain mean for BOTH sides: the comparison stays fair,
    # which is what matters, at the cost of the real patches not sitting at their
    # own latitude.
    #
    # The all-ones ``mask`` this returns is right for the GENERATED side only.
    # A REAL crop can contain land, and land pixels hold anomaly == 0.0 exactly,
    # so anything that differentiates a real patch must know where its land is.
    # ``run_size`` therefore stores the full-domain ocean mask as ``mask_full``
    # next to ``real_pos``; diagnostics rebuilds a per-sample mask from the two
    # (see the land-edge EKE gotcha in eval/README.md).
    ocean_b = ds.ocean > 0.5
    clim_ref = np.stack([np.full(hw, float(c[ocean_b].mean()), dtype=np.float32)
                         for c in clim], 0)
    return (hw, np.ones(hw, dtype=np.float32),
            np.full(hw, float(lat_full.mean()), dtype=np.float32),
            clim_ref, std, f"constant domain-mean latitude {lat_full.mean():.2f} N")


def _load_real_cache(path: str, size: str, n: int, target) -> dict:
    """Real fields + denormalisation reference from an earlier run's npz.

    The real half of a samples file depends only on the dataset, split and seed,
    never on the checkpoint -- so a sweep over checkpoints can reuse it and skip
    the ~1 h store load. Everything that could make the reuse wrong is checked:
    the variable list, and that the cache has at least as many samples as asked
    for.
    """
    src = path if path.endswith(".npz") else os.path.join(path, f"samples_{size}.npz")
    d = np.load(src, allow_pickle=False)
    cached = json.loads(str(d["vars"]))
    if cached != list(target):
        raise SystemExit(f"--real-from {src}: variables {cached} != {list(target)}")
    real = d["real_norm"]
    if real.shape[0] < n:
        raise SystemExit(f"--real-from {src}: has {real.shape[0]} real fields, "
                         f"asked for {n}")
    meta = json.loads(str(d["meta"]))
    print(f"[real] reused {n} of {real.shape[0]} fields from {src} "
          f"(split={meta['split']}, ref_day={int(d['ref_day'])})")
    pos = d["real_pos"]
    # Caches written before the land-edge EKE fix have no mask_full. A
    # zero-size array keeps the payload savez-able and lets diagnostics detect
    # the gap and warn, instead of this loader refusing an otherwise-good cache.
    if "mask_full" in d.files:
        mask_full = d["mask_full"]
    else:
        mask_full = np.zeros((0, 0), dtype=np.float32)
        if pos.size:
            print(f"[real] WARNING: {src} predates mask_full: real patches "
                  "cannot be land-masked downstream (see the land-edge EKE "
                  "gotcha in eval/README.md, including the backfill recipe)")
    return dict(real=real[:n], pos=pos[:n].tolist() if pos.size else [],
                days=d["real_days"][:n], ref_day=int(d["ref_day"]),
                hw=real.shape[2:], mask=d["mask"], mask_full=mask_full,
                lat=d["lat"], clim_ref=d["clim_ref"], std=d["std"],
                lat_note=meta.get("lat_note", "from --real-from cache"))


def run_size(net, cfg, ds, lat_full, dx_m, size, n, batch, device, args,
             meta, sampler, out):
    """Draw ``n`` samples at one geometry and write its npz."""
    full = size == "full"
    patch = None if full else int(size)

    if ds is None:
        ref = _load_real_cache(args.real_from, size, n, cfg.target)
        real, pos, days, ref_day = ref["real"], ref["pos"], ref["days"], ref["ref_day"]
        hw, mask, lat, clim_ref = ref["hw"], ref["mask"], ref["lat"], ref["clim_ref"]
        std, lat_note, mask_full = ref["std"], ref["lat_note"], ref["mask_full"]
    else:
        days = pick_real_days(ds.valid_days, n, args.skip_tail, allow_repeat=not full)
        ref_day = (int(args.ref_day) if args.ref_day is not None
                   else int(np.median(days)))
        print(f"[real] {n} steps spanning {days[0]}..{days[-1]} "
              f"(stride ~{int(np.diff(days).mean())} h); ref_day={ref_day}", flush=True)

        # -- real reference fields, in anomaly units -----------------------
        real, pos = [], []
        if full:
            hw = (ds.NY, ds.NX)
            for item in ds.full_frames(days):
                real.append(item["target"].numpy())
        else:
            hw = (patch, patch)
            ds.set_epoch_rng(args.seed)      # reuse the loader's own crop lattice
            for item in ds.full_frames(days):
                y0, x0 = ds.sample_position()
                real.append(item["target"].numpy()[:, y0:y0 + patch, x0:x0 + patch])
                pos.append((y0, x0))
        real = np.stack(real, 0).astype(np.float32)
        hw, mask, lat, clim_ref, std, lat_note = _denorm_reference(
            ds, cfg, lat_full, full, hw, ref_day)
        # Stored in BOTH geometries (identical to ``mask`` in full mode) so a
        # patch npz is self-contained: real_pos + mask_full is what lets
        # diagnostics mask the land inside each real crop before differentiating.
        mask_full = (ds.ocean > 0.5).astype(np.float32)
    print(f"[real] {real.shape}")

    # -- generated samples --------------------------------------------------
    print(f"[gen ] {n} unconditional draws at {hw[0]}x{hw[1]}, batch {batch}, "
          f"steps={sampler['num_steps']}, s_churn={sampler['s_churn']}", flush=True)
    gen = generate(net, cfg, n, hw, batch, device, args.seed, sampler).astype(np.float32)

    meta.update(size=size, split=args.split, seed=args.seed, n=n,
                ref_day=ref_day, skip_tail=args.skip_tail, dx_m=dx_m,
                lat_note=lat_note, sampler=sampler, target=list(cfg.target))

    payload = dict(
        vars=json.dumps(list(cfg.target)), gen_norm=gen, real_norm=real,
        clim_ref=clim_ref, std=std, mask=mask, mask_full=mask_full, lat=lat,
        dx_m=np.float64(dx_m), real_days=days.astype(np.int64),
        real_pos=np.asarray(pos, dtype=np.int64) if pos else np.zeros((0, 2), np.int64),
        ref_day=np.int64(ref_day), size=size,
        config=json.dumps(cfg.as_dict()), meta=json.dumps(meta),
    )
    # Plain savez: ~1 GB of float noise barely compresses and savez_compressed
    # spends minutes finding that out.
    np.savez(out, **payload)
    print(f"[out ] wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
