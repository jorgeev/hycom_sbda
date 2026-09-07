"""Draw K-member ensembles from a CONDITIONAL checkpoint for D held-out target
steps, and store them beside the truth, the conditioning, and the baselines
the diagnostics need.

The conditional counterpart of ``eval.gen_prior``. That script draws
unconstrained p(x) samples and refuses any checkpoint with
``cfg.cond_channels() != 0``; this one refuses the opposite case. Nothing in
``eval/`` is imported -- the two layers are deliberately parallel, not nested.

WHAT "CONDITIONAL" MEANS HERE. Conditioning is channel concatenation
(``EDMPrecond.forward``, diffusion/edm.py:33): a k-step window of observation
channels (``cfg.cond_vars``) plus optional static ocean-mask and day-of-year
planes. ``ZarrWindowDataset.full_frames`` builds the exact tensor the network
saw in training, so it is used unchanged; a patch is CUT from that tensor at
the loader's own crop lattice, so unlike gen_prior a generated patch IS located
-- it has a real land mask, a real latitude and a real truth.

TWO FAMILIES ARE SERVED. (a) mask + day-of-year only (``cond_vars: []``,
``use_ocean_mask``/``use_doy`` on, e.g. runs/prior_joint): Cobs = 0, so there is
no observation to score against and the conditioning tests reduce to seasonal
consistency and land handling. (b) observation-conditioned (gom_nemo ssh_128,
or a gulfstream config conditioned on daily/weekly obs): the newest step of each
observation channel is also stored, denormalised with ITS OWN statistics, as a
deterministic baseline, and its exact-0.0 pixels are recorded as unobserved.

ON 0.0. The loader writes an exact 0.0 into a normalised channel wherever the
raw value was non-finite (an observation gap) or the pixel is land. In
normalised units 0.0 is the mean, and the network cannot tell a gap from a mean
observation -- so ``obs_avail = (cond != 0.0)`` is exactly the model's view of
what it was told, which is what the fidelity diagnostics need. Land is
therefore also "unobserved".

UNITS. Everything generated is stored in the model's own anomaly / sigma units
(``gen_norm``, ``truth_norm``, float16 for the ensemble by default: the sampler
runs under bf16 autocast, so fp16 storage is below its noise floor). The
per-day denormalisation mean ``clim_day`` and scalar ``std`` are stored so the
diagnostics recover physical units with ``norm * std + clim_day`` (then
``10 **`` for log10 variables). Unlike gen_prior there is no shared reference
day: every sample is paired with a truth of the same day, so each day's own
climatology is the right one.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m eval.gen_cond \
        --ckpt runs/prior_joint/ckpt.pt --out-dir runs/prior_joint/eval_cond_step0200000 \
        --size full,256

writes ``samples_cond_full.npz`` and ``samples_cond_256.npz`` there. Then
``python -m eval.diagnostics_cond --samples <npz> --out <dir>/figs_cond_<size>``.
``--selftest`` runs the whole path on a fake in-memory dataset and a zero
network, no store and no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from diffusion.config import TRUTH_TO_DEGRADED
from diffusion.data import ZarrWindowDataset, ocean_patch_positions
from diffusion.dataset_spec import resolve_spec
from diffusion.edm import edm_heun_sampler
from diffusion.sample import config_from_ckpt
from diffusion.train import build_net

EXTRA_NAMES = ("ocean_mask", "doy_sin", "doy_cos")


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def load_ckpt(path: str, device, weights: str = "ema"):
    """``(net, cfg, meta)`` from a checkpoint (mirrors eval.gen_prior.load_ckpt)."""
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_ckpt(sd["config"])
    net = build_net(cfg, "diffusion").to(device)
    net.load_state_dict(sd[weights])
    net.eval()
    meta = {"ckpt": os.path.abspath(path),
            "step": int(sd.get("step", -1)) + 1,
            "weights": weights,
            "val_loss": float(sd["val_loss"]) if "val_loss" in sd else None,
            "best_val": float(sd["best_val"]) if "best_val" in sd else None,
            "arch": cfg.arch, "norm_mode": cfg.norm_mode,
            "dataset": cfg.dataset or "gom_nemo(legacy)",
            "cond_channels": int(cfg.cond_channels())}
    return net, cfg, meta


# ---------------------------------------------------------------------------
# which steps, which crops
# ---------------------------------------------------------------------------
def pick_days(valid_days: np.ndarray, n: int, skip_tail: int,
              allow_repeat: bool = False) -> np.ndarray:
    """``n`` target steps spread evenly across the split (not the last n).

    The last n steps of an hourly record are one weather state; of gom_nemo's
    375-day val split they are one season, which would defeat the seasonal
    diagnostic a mask+doy model is judged on. ``skip_tail`` drops the end of the
    record where a monthly climatology is poorly constrained (24 h on the
    hourly gulfstream family; 0 on a daily store).
    """
    pool = valid_days[:-skip_tail] if skip_tail > 0 else valid_days
    if pool.size < n and not allow_repeat:
        raise ValueError(f"split has {pool.size} usable steps, asked for {n}")
    take = min(n, pool.size)
    days = pool[np.linspace(0, pool.size - 1, take).round().astype(int)]
    if n > take:
        days = np.resize(days, n)
    return days


def crop_lattice(ds, cfg, patch: int, seed: int):
    """A callable returning ``(y0, x0)`` crop corners, and a note saying which
    lattice it draws from.

    When the eval patch equals the training patch the loader's own lattice is
    used, exactly as training did. Otherwise a lattice is rebuilt for THIS
    patch size so the ocean-fraction filter applies to the window actually cut
    -- a 128 sub-window of a 256 lattice corner could be mostly land.
    """
    if patch == cfg.patch and getattr(ds, "positions", None) is not None:
        ds.set_epoch_rng(seed)
        return ds.sample_position, f"loader crop lattice (patch == cfg.patch == {patch})"
    pos = ocean_patch_positions(ds.ocean, patch, max(patch // 4, 1), cfg.ocean_frac)
    rng = np.random.default_rng(seed)
    return (lambda: tuple(pos[rng.integers(len(pos))])), (
        f"rebuilt stride-{max(patch // 4, 1)} lattice for patch {patch} "
        f"(model trained at {cfg.patch})")


def select_samples(ds, cfg, size: str, n: int, seed: int, skip_tail: int):
    """``(samples, lattice_note)``: one dict(day, y0, x0, rep) per sample."""
    full = size == "full"
    days = pick_days(ds.valid_days, n, skip_tail, allow_repeat=not full)
    if full:
        return [dict(day=int(d), y0=0, x0=0, rep=0) for d in days], "full frame"
    draw, note = crop_lattice(ds, cfg, int(size), seed)
    seen: dict[int, int] = {}
    out = []
    for d in days:
        d = int(d)
        y0, x0 = draw()
        out.append(dict(day=d, y0=int(y0), x0=int(x0), rep=seen.get(d, 0)))
        seen[d] = seen.get(d, 0) + 1
    return out, note


# ---------------------------------------------------------------------------
# conditioning layout and baselines
# ---------------------------------------------------------------------------
def cond_layout(cfg):
    """``(names_full, idx_last, extras)`` for the channel order of
    ``ZarrWindowDataset.build_condition``: the k-step window oldest -> newest,
    variables minor, then ``ocean_mask``, ``doy_sin``, ``doy_cos``."""
    k, cobs = cfg.k_days, len(cfg.cond_vars)
    extras = ([] if not cfg.use_ocean_mask else ["ocean_mask"]) + \
             ([] if not cfg.use_doy else ["doy_sin", "doy_cos"])
    names = [f"{v}@t-{k - 1 - t}" for t in range(k) for v in cfg.cond_vars] + extras
    idx_last = [(k - 1) * cobs + j for j in range(cobs)] + \
               [k * cobs + e for e in range(len(extras))]
    return names, idx_last, extras


def default_baseline_map(cfg) -> dict:
    """``{target: cond_var | None}``: the observation channel that degrades each
    target. ``config.TRUTH_TO_DEGRADED`` first, then the unique cond var whose
    lowercase name starts with the target's (ssh -> ssha_aviso, sst ->
    sst_radiometer, sss -> sss_sat)."""
    out = {}
    for v in cfg.target:
        c = TRUTH_TO_DEGRADED.get(v)
        if c in cfg.cond_vars:
            out[v] = c
            continue
        cands = [c for c in cfg.cond_vars if c.lower().startswith(v.lower())]
        out[v] = cands[0] if len(cands) == 1 else None
    return out


def parse_baseline_map(spec_str: str | None, cfg) -> dict | None:
    """``"ssh=ssha_aviso,sst=sst_radiometer,sss=none"`` -> dict, validated."""
    if not spec_str:
        return None
    out = {}
    for item in spec_str.split(","):
        if not item.strip():
            continue
        t, _, c = item.partition("=")
        t, c = t.strip(), c.strip()
        if t not in cfg.target:
            raise SystemExit(f"--baseline-map: {t!r} is not a target {cfg.target}")
        if c.lower() in ("", "none", "null"):
            out[t] = None
        elif c not in cfg.cond_vars:
            raise SystemExit(f"--baseline-map: {c!r} is not a cond var {cfg.cond_vars}")
        else:
            out[t] = c
    return out


def baseline_kinds(bmap: dict, anom_targets: set[str]) -> dict:
    """``absolute`` unless the obs channel is an anomaly product (name contains
    ``ssha`` / ``anom``) or the target was listed in ``--baseline-anom``."""
    out = {}
    for t, c in bmap.items():
        anom = (c is not None and ("ssha" in c.lower() or "anom" in c.lower())) \
            or t in anom_targets
        out[t] = "anomaly" if anom else "absolute"
    return out


# ---------------------------------------------------------------------------
# reference extraction (truth, cond, baseline, per-day climatology)
# ---------------------------------------------------------------------------
def _times_for(ds, spec, var: str):
    """Timestamps of ``var``'s own cadence, or None when it is the base cadence."""
    if ds._vmap.get(var) is None:
        return None
    return np.asarray(spec.family.times(spec.cadence_of(var)), dtype=np.float64)


def extract_reference(ds, cfg, spec, samples: list[dict], size: str,
                      store_cond: str, bmap: dict, lat_full: np.ndarray) -> dict:
    """Everything except the ensemble, in one pass over ``ds.full_frames``.

    Per sample: the exact conditioning tensor (kept whole in ``cond_full`` for
    the sampler), the truth, the per-sample ocean mask and latitude, the
    observation-availability pattern of the newest window step, the degraded
    obs channel as a baseline in its own physical units, and the per-day
    denormalisation mean of each target.
    """
    full = size == "full"
    patch = None if full else int(size)
    names_full, idx_last, extras = cond_layout(cfg)
    cobs, k = len(cfg.cond_vars), cfg.k_days
    D = len(samples)
    hw = (ds.NY, ds.NX) if full else (patch, patch)
    ct = cfg.target_channels()
    cc = cfg.cond_channels()

    t_base = np.asarray(spec.family.times(spec.base), dtype=np.float64)
    t_var = {c: _times_for(ds, spec, c) for c in cfg.cond_vars}

    cond_full = np.zeros((D, cc) + hw, dtype=np.float32)
    truth = np.zeros((D, ct) + hw, dtype=np.float32)
    obs_avail = np.zeros((D, cobs) + hw, dtype=bool)
    obs_age = np.zeros((D, cobs), dtype=np.float64)
    baseline = np.full((D, ct) + hw, np.nan, dtype=np.float32)
    mask = np.zeros((D,) + hw, dtype=bool)
    lat = np.zeros((D,) + hw, dtype=np.float32)
    anomaly = cfg.norm_mode == "anomaly"
    clim = np.zeros((D, ct) + (hw if anomaly else (1, 1)), dtype=np.float32)
    days = np.array([s["day"] for s in samples], dtype=np.int64)

    frames = ds.full_frames(days.tolist())
    t0 = time.time()
    for d, s in enumerate(samples):
        item = next(frames)
        day = s["day"]
        ysl = slice(None) if full else slice(s["y0"], s["y0"] + patch)
        xsl = slice(None) if full else slice(s["x0"], s["x0"] + patch)
        c_all = item["cond"].numpy()[:, ysl, xsl]
        cond_full[d] = c_all
        truth[d] = item["target"].numpy()[:, ysl, xsl]
        mask[d] = ds.ocean[ysl, xsl] > 0.5
        lat[d] = lat_full[ysl, xsl]
        last = c_all[idx_last]
        obs_avail[d] = last[:cobs] != 0.0
        for j, c in enumerate(cfg.cond_vars):
            tv = t_var[c]
            obs_age[d, j] = 0.0 if tv is None else t_base[day] - tv[ds._idx(c, day)]
        for i, v in enumerate(cfg.target):
            m, _s, _ = ds.denorm_params(v, day)
            if anomaly:
                clim[d, i] = np.broadcast_to(m, (ds.NY, ds.NX))[ysl, xsl]
            else:
                clim[d, i] = float(m)
            c = bmap.get(v)
            if c is None:
                continue
            j = cfg.cond_vars.index(c)
            mc, sc, is_log_c = ds.denorm_params(c, day)
            mc = np.broadcast_to(mc, (ds.NY, ds.NX))[ysl, xsl] if np.ndim(mc) else float(mc)
            phys = last[j].astype(np.float64) * sc + mc
            if is_log_c:
                phys = np.power(10.0, phys)
            baseline[d, i] = np.where(obs_avail[d, j], phys, np.nan)
        if (d + 1) % 8 == 0 or d + 1 == D:
            print(f"  [ref] {d + 1}/{D} steps  {time.time() - t0:6.1f}s", flush=True)

    std = np.array([ds.stats[v][1] for v in cfg.target], dtype=np.float64)
    is_log = np.array([bool(ds.stats[v][2]) for v in cfg.target])

    # baseline bias in physical units, over observed ocean pixels
    bias = np.full(ct, np.nan)
    for i, v in enumerate(cfg.target):
        c = bmap.get(v)
        if c is None:
            continue
        tp = truth[:, i].astype(np.float64) * std[i] + clim[:, i]
        if is_log[i]:
            tp = np.power(10.0, tp)
        sel = mask & np.isfinite(baseline[:, i])
        if sel.any():
            bias[i] = float(np.mean((baseline[:, i] - tp)[sel]))

    if store_cond == "full":
        cond_norm, cond_names = cond_full, names_full
    elif store_cond == "last":
        cond_norm = cond_full[:, idx_last]
        cond_names = [names_full[i] for i in idx_last]
    else:
        cond_norm = np.zeros((D, 0) + hw, dtype=np.float32)
        cond_names = []

    step_s = float(np.median(np.diff(t_base))) if t_base.size > 1 else float("nan")
    return dict(
        cond_full=cond_full, cond_norm=cond_norm, cond_channel_names=cond_names,
        truth_norm=truth, obs_avail=obs_avail, obs_age_s=obs_age,
        baseline_phys=baseline, baseline_bias=bias, clim_day=clim, std=std,
        is_log=is_log, mask=mask, lat=lat, days=days,
        time_unix=t_base[days], doy=(ds.doy_ang[days] / (2 * np.pi) * 365.2425).astype(np.float32),
        pos=(np.zeros((0, 2), np.int64) if full
             else np.array([(s["y0"], s["x0"]) for s in samples], dtype=np.int64)),
        mask_full=(ds.ocean > 0.5).astype(np.float32), lat_full=lat_full.astype(np.float32),
        extras=extras, k_days=k, step_seconds=step_s,
        base_cadence=getattr(spec, "base_cadence", "unknown"), hw=hw,
    )


# ---------------------------------------------------------------------------
# cache reuse
# ---------------------------------------------------------------------------
def _load_reference_cache(path: str, size: str, n: int, cfg, spec, split: str,
                          bmap: dict) -> dict:
    """The reference half of an earlier ``samples_cond_<size>.npz``.

    Truth AND conditioning depend on the dataset, split, seed and the
    config's conditioning fields -- not on the weights -- so a checkpoint
    ladder can reuse them. Everything that could make the reuse wrong is
    checked, and a cache written with ``--store-cond last`` (k > 1) is
    refused: the network needs the whole window.
    """
    src = path if path.endswith(".npz") else os.path.join(path, f"samples_cond_{size}.npz")
    d = np.load(src, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    _, _, extras = cond_layout(cfg)

    def need(cond, msg):
        if not cond:
            raise SystemExit(f"--real-from {src}: {msg}")

    need(json.loads(str(d["vars"])) == list(cfg.target),
         f"targets {json.loads(str(d['vars']))} != {list(cfg.target)}")
    need(json.loads(str(d["cond_vars"])) == list(cfg.cond_vars),
         f"cond_vars {json.loads(str(d['cond_vars']))} != {list(cfg.cond_vars)}")
    need(int(d["k_days"]) == cfg.k_days, f"k_days {int(d['k_days'])} != {cfg.k_days}")
    need(json.loads(str(d["extras"])) == extras,
         f"extras {json.loads(str(d['extras']))} != {extras}")
    need(meta.get("norm_mode") == cfg.norm_mode,
         f"norm_mode {meta.get('norm_mode')} != {cfg.norm_mode}")
    need(meta.get("split") == split, f"split {meta.get('split')} != {split}")
    need(str(d["size"]) == size, f"size {str(d['size'])} != {size}")
    need(meta.get("store_cond") == "full",
         f"was written with --store-cond {meta.get('store_cond')}; the network "
         "needs the whole conditioning window, regenerate the cache with 'full'")
    need(meta.get("baseline_map") == bmap,
         f"baseline_map {meta.get('baseline_map')} != {bmap} (pass the same --baseline-map)")
    D = int(d["truth_norm"].shape[0])
    need(D >= n, f"has {D} samples, asked for {n}")
    ds_name = getattr(spec, "name", None)
    if meta.get("dataset_name") not in (None, ds_name):
        print(f"[real] WARNING: cache dataset {meta.get('dataset_name')!r} != {ds_name!r}")

    sl = slice(0, n)
    out = dict(
        cond_full=d["cond_norm"][sl], cond_norm=d["cond_norm"][sl],
        cond_channel_names=json.loads(str(d["cond_channel_names"])),
        truth_norm=d["truth_norm"][sl], obs_avail=d["obs_avail"][sl],
        obs_age_s=d["obs_age_s"][sl], baseline_phys=d["baseline_phys"][sl],
        baseline_bias=d["baseline_bias"], clim_day=d["clim_day"][sl], std=d["std"],
        is_log=d["is_log"], mask=d["mask"][sl], lat=d["lat"][sl], days=d["days"][sl],
        time_unix=d["time_unix"][sl], doy=d["doy"][sl],
        pos=d["pos"][sl] if d["pos"].size else d["pos"], mask_full=d["mask_full"],
        lat_full=d["lat_full"], extras=extras, k_days=cfg.k_days,
        step_seconds=meta.get("step_seconds"), base_cadence=meta.get("base_cadence"),
        hw=tuple(d["truth_norm"].shape[2:]),
        lattice_note=meta.get("lattice_note", "from --real-from cache"),
    )
    print(f"[real] reused {n} of {D} reference samples from {src} "
          f"(split={meta.get('split')}, steps {out['days'][0]}..{out['days'][-1]})")
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_ensembles(net, cfg, cond_full: np.ndarray, samples: list[dict], K: int,
                       member_batch: int, device, seed: int, sampler: dict,
                       out_dtype) -> np.ndarray:
    """``(D, K, Ct, H, W)`` members. Seeds ``seed + 1000*day + 100*rep + off``
    (``rep`` = repeat index of that day in patch mode, ``off`` = member batch),
    so at full geometry (rep = 0) the members are bit-reproducible against
    ``diffusion.sample.sample_days`` for the same sampler settings."""
    D = len(samples)
    ct = cfg.target_channels()
    hw = cond_full.shape[2:]
    gen = np.zeros((D, K, ct) + hw, dtype=out_dtype)
    g = torch.Generator(device=device)
    t0 = time.time()
    for d, s in enumerate(samples):
        cond = torch.from_numpy(cond_full[d]).to(device)[None]
        remaining, off, j = K, 0, 0
        while remaining > 0:
            m = min(member_batch, remaining)
            g.manual_seed(seed + 1000 * int(s["day"]) + 100 * int(s["rep"]) + off)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x = edm_heun_sampler(net, cond.expand(m, -1, -1, -1), ct,
                                     generator=g, **sampler)
            gen[d, j:j + m] = x.float().cpu().numpy().astype(out_dtype)
            remaining -= m
            off += 1
            j += m
        el = time.time() - t0
        print(f"  [gen] {d + 1}/{D}  {el:6.1f}s  {el / (d + 1):5.2f}s/sample  "
              f"{el / ((d + 1) * K):5.3f}s/member", flush=True)
    return gen


# ---------------------------------------------------------------------------
# one geometry
# ---------------------------------------------------------------------------
def run_size(net, cfg, spec, ds, lat_full, dx_m, size, n, K, member_batch, device,
             args, meta, sampler, bmap, bkind, out):
    t_all = time.time()
    if ds is None:
        ref = _load_reference_cache(args.real_from, size, n, cfg, spec, args.split, bmap)
        samples = [dict(day=int(d), y0=int(p[0]) if ref["pos"].size else 0,
                        x0=int(p[1]) if ref["pos"].size else 0, rep=0)
                   for d, p in zip(ref["days"], (ref["pos"] if ref["pos"].size
                                                 else np.zeros((n, 2), int)))]
        # rep bookkeeping so repeated days get distinct seeds, as in a fresh run
        seen: dict[int, int] = {}
        for s in samples:
            s["rep"] = seen.get(s["day"], 0)
            seen[s["day"]] = s["rep"] + 1
        lattice_note = ref["lattice_note"]
        t_ref = 0.0
    else:
        samples, lattice_note = select_samples(ds, cfg, size, n, args.seed, args.skip_tail)
        days = [s["day"] for s in samples]
        print(f"[real] {n} steps spanning {min(days)}..{max(days)}; {lattice_note}",
              flush=True)
        t0 = time.time()
        ref = extract_reference(ds, cfg, spec, samples, size, args.store_cond, bmap, lat_full)
        t_ref = time.time() - t0
    hw = ref["hw"]
    print(f"[real] truth {ref['truth_norm'].shape}, cond {ref['cond_full'].shape}, "
          f"obs_avail {ref['obs_avail'].shape}", flush=True)

    out_dtype = np.float16 if args.gen_dtype == "float16" else np.float32
    print(f"[gen ] {n} x {K} members at {hw[0]}x{hw[1]}, member batch {member_batch}, "
          f"steps={sampler['num_steps']}, s_churn={sampler['s_churn']}", flush=True)
    t0 = time.time()
    gen = generate_ensembles(net, cfg, ref["cond_full"], samples, K, member_batch,
                             device, args.seed, sampler, out_dtype)
    t_gen = time.time() - t0

    meta = dict(meta)
    meta.update(size=size, split=args.split, seed=args.seed, D=n, K=K,
                member_batch=member_batch, sampler=sampler, store_cond=args.store_cond,
                gen_dtype=args.gen_dtype, skip_tail=args.skip_tail, dx_m=dx_m,
                lat_note="per-pixel coords/lat, cropped at pos", lattice_note=lattice_note,
                pos_mode="full" if size == "full" else "lattice", target=list(cfg.target),
                cond_vars=list(cfg.cond_vars), k_days=cfg.k_days, extras=ref["extras"],
                baseline_map=bmap, baseline_kind=bkind, norm_mode=cfg.norm_mode,
                dataset_name=getattr(spec, "name", None), base_cadence=ref["base_cadence"],
                step_seconds=ref["step_seconds"],
                timings=dict(reference_s=t_ref, generate_s=t_gen))

    payload = dict(
        vars=json.dumps(list(cfg.target)), cond_vars=json.dumps(list(cfg.cond_vars)),
        k_days=np.int64(cfg.k_days), extras=json.dumps(ref["extras"]),
        gen_norm=gen, truth_norm=ref["truth_norm"], cond_norm=ref["cond_norm"],
        cond_channel_names=json.dumps(ref["cond_channel_names"]),
        obs_avail=ref["obs_avail"], obs_age_s=ref["obs_age_s"],
        baseline_phys=ref["baseline_phys"], baseline_map=json.dumps(bmap),
        baseline_kind=json.dumps(bkind), baseline_bias=ref["baseline_bias"],
        clim_day=ref["clim_day"], std=ref["std"], is_log=ref["is_log"],
        mask=ref["mask"], mask_full=ref["mask_full"], lat=ref["lat"],
        lat_full=ref["lat_full"], dx_m=np.float64(dx_m), days=ref["days"],
        time_unix=np.asarray(ref["time_unix"], dtype=np.float64), doy=ref["doy"],
        pos=ref["pos"], size=size, config=json.dumps(cfg.as_dict()),
    )
    t0 = time.time()
    payload["meta"] = json.dumps(dict(meta, timings=dict(meta["timings"], total_s=time.time() - t_all)))
    np.savez(out, **payload)                     # plain savez: fp16 noise does not compress
    print(f"[out ] wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return payload


def _per_size(spec_str, sizes, defaults, what):
    if spec_str is None:
        return defaults
    vals = [int(x) for x in str(spec_str).split(",")]
    if len(vals) == 1:
        return vals * len(sizes)
    if len(vals) != len(sizes):
        raise SystemExit(f"--{what} has {len(vals)} values but --size has {len(sizes)}")
    return vals


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="run dir or ckpt.pt path")
    ap.add_argument("--out-dir", help="directory for samples_cond_<size>.npz")
    ap.add_argument("--size", default="full,128",
                    help="comma-separated geometries: 'full' and/or patch sizes")
    ap.add_argument("--n", default=None, help="D target steps per geometry (48 full / 256 patch)")
    ap.add_argument("--k", type=int, default=16, help="K ensemble members per step")
    ap.add_argument("--member-batch", default=None, help="members per forward (4 full / 16 patch)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--real-from", default=None,
                    help="reuse truth/cond/baselines from an earlier --out-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-tail", type=int, default=None,
                    help="drop this many trailing steps (default 24 if hourly, else 0)")
    ap.add_argument("--store-cond", default="full", choices=("full", "last", "none"))
    ap.add_argument("--gen-dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--baseline-map", default=None,
                    help="target=cond_var,... (default from TRUTH_TO_DEGRADED + name prefix)")
    ap.add_argument("--baseline-anom", default="",
                    help="comma list of targets whose baseline is an anomaly product")
    ap.add_argument("--sampler-steps", type=int, default=None)
    ap.add_argument("--s-churn", type=float, default=None)
    ap.add_argument("--s-noise", type=float, default=None)
    ap.add_argument("--s-tmin", type=float, default=None)
    ap.add_argument("--s-tmax", type=float, default=None)
    ap.add_argument("--weights", default="ema", choices=("ema", "model"))
    ap.add_argument("--selftest", action="store_true",
                    help="run the whole path on a fake dataset and zero net (no store, no GPU)")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.ckpt or not args.out_dir:
        ap.error("--ckpt and --out-dir are required")

    ckpt = args.ckpt if args.ckpt.endswith(".pt") else os.path.join(args.ckpt, "ckpt.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg, meta = load_ckpt(ckpt, device, weights=args.weights)
    if cfg.cond_channels() == 0:
        raise SystemExit(
            f"{ckpt} is an UNCONDITIONAL model (cond_vars={cfg.cond_vars}, "
            f"use_ocean_mask={cfg.use_ocean_mask}, use_doy={cfg.use_doy}). "
            "eval.gen_cond pairs every sample with the observations it was conditioned "
            "on; use `python -m eval.gen_prior` for p(x) draws.")
    print(f"[ckpt] {ckpt}  step {meta['step']:,}  arch={cfg.arch}  "
          f"Cc={cfg.cond_channels()} (cond_vars={cfg.cond_vars}, k={cfg.k_days}, "
          f"mask={cfg.use_ocean_mask}, doy={cfg.use_doy})  Ct={cfg.target_channels()} "
          f"{cfg.target}  norm={cfg.norm_mode}", flush=True)

    sizes = [s.strip() for s in args.size.split(",") if s.strip()]
    ns = _per_size(args.n, sizes, [48 if s == "full" else 256 for s in sizes], "n")
    bs = _per_size(args.member_batch, sizes, [4 if s == "full" else 16 for s in sizes],
                   "member-batch")
    ds_factor = 2 ** (len(cfg.channel_mult) - 1)
    for s in sizes:
        if s != "full" and int(s) % ds_factor:
            raise SystemExit(f"--size {s} is not divisible by the UNet's {ds_factor}x downsample")

    sampler = dict(num_steps=cfg.sampler_steps, sigma_min=cfg.sigma_min,
                   sigma_max=cfg.sigma_max, rho=cfg.rho, s_churn=cfg.s_churn,
                   s_noise=cfg.s_noise, s_tmin=cfg.s_tmin, s_tmax=cfg.s_tmax)
    for key, val in (("num_steps", args.sampler_steps), ("s_churn", args.s_churn),
                     ("s_noise", args.s_noise), ("s_tmin", args.s_tmin),
                     ("s_tmax", args.s_tmax)):
        if val is not None:
            sampler[key] = val

    # The store: via resolve_spec, never cfg.zarr_path (see eval/gen_prior.py:226).
    spec = resolve_spec(cfg)
    dx_m = spec.dx_m()
    lat_full = np.asarray(spec.family.coord("coords/lat"), dtype=np.float32)
    if args.skip_tail is None:
        args.skip_tail = 24 if spec.base_cadence == "hourly" else 0
    bmap = parse_baseline_map(args.baseline_map, cfg) or default_baseline_map(cfg)
    bkind = baseline_kinds(bmap, {t.strip() for t in args.baseline_anom.split(",") if t.strip()})
    print(f"[base] baseline map {bmap}  kinds {bkind}")

    if args.real_from:
        ds = None
        print(f"[data] reusing reference fields from {args.real_from} -- the store is not read",
              flush=True)
    else:
        needed = list(dict.fromkeys(cfg.cond_vars + cfg.target))
        print(f"[data] loading {needed} over the {spec.n_steps()}-step axis "
              f"({spec.name}); this can take minutes to an hour depending on the store",
              flush=True)
        t0 = time.time()
        ds = ZarrWindowDataset(cfg, split=args.split, spec=spec)
        print(f"[data] ready in {time.time() - t0:.0f}s; {ds.valid_days.size} valid "
              f"steps in split {args.split!r}, grid {ds.NY}x{ds.NX}", flush=True)
        if ds.NY % ds_factor or ds.NX % ds_factor:
            print(f"[data] WARNING: {ds.NY}x{ds.NX} is not divisible by {ds_factor}; "
                  "full-frame generation will fail inside the UNet")

    os.makedirs(args.out_dir, exist_ok=True)
    for size, n, mb in zip(sizes, ns, bs):
        out = os.path.join(args.out_dir, f"samples_cond_{size}.npz")
        print(f"\n=== {size}: D={n} x K={args.k}, member batch {mb} -> {out} ===", flush=True)
        run_size(net, cfg, spec, ds, lat_full, dx_m, size, n, args.k, mb, device,
                 args, meta, sampler, bmap, bkind, out)
    return 0


# ---------------------------------------------------------------------------
# selftest: fake dataset + zero network, no store, no GPU
# ---------------------------------------------------------------------------
class _ZeroNet(torch.nn.Module):
    """A denoiser that returns 0: the Heun path then ends at exactly x = 0."""

    def forward(self, x, sigma, cond):            # noqa: ARG002
        assert cond.shape[1] == self.cc, f"cond has {cond.shape[1]} channels, expected {self.cc}"
        return torch.zeros_like(x)


class _FakeSpec:
    def __init__(self, t_unix):
        self.name = "fake"
        self.base_cadence = "daily"
        self._t = t_unix

        class _Cad:
            name = "daily"
        self.base = _Cad()

        class _Fam:
            def __init__(fam, t):
                fam._t = t

            def times(fam, cad):
                return fam._t
        self.family = _Fam(t_unix)

    def cadence_of(self, name):
        return self.base

    def dx_m(self):
        return 1818.33


def _fake_dataset(cfg, ny=48, nx=64, T=80, seed=0):
    """A ``ZarrWindowDataset`` with its attributes planted, so the REAL
    ``full_frames`` / ``build_condition`` / ``denorm_params`` / crop-lattice
    methods run over synthetic arrays. Returns ``(ds, gap_mask)``."""
    rng = np.random.default_rng(seed)
    ds = ZarrWindowDataset.__new__(ZarrWindowDataset)
    ds.cfg, ds.k, ds.split = cfg, cfg.k_days, "val"
    ds.ocean = np.ones((ny, nx), dtype=np.float32)
    ds.ocean[:5, :] = 0.0
    ds.ocean[20:28, 30:38] = 0.0
    ds.NY, ds.NX = ny, nx
    ds.t_end = T
    t_unix = 1.5e9 + np.arange(T) * 86400.0
    year = 365.2425 * 86400.0
    ds.doy_ang = 2 * np.pi * (np.mod(t_unix, year) / year)
    ds.doy_sin = np.sin(ds.doy_ang).astype(np.float32)
    ds.doy_cos = np.cos(ds.doy_ang).astype(np.float32)
    needed = list(dict.fromkeys(cfg.cond_vars + cfg.target))
    ds._vmap = {v: None for v in needed}
    ds.stats, ds.clim, ds.data = {}, {}, {}
    ocean_b = ds.ocean > 0.5
    gap = np.zeros((ny, nx), dtype=bool)
    gap[10:18, 5:20] = True
    gap[30:40, 45:60] = True
    for i, v in enumerate(needed):
        is_log = v.upper().startswith("CHL")
        ds.stats[v] = (0.5 * i, 1.0 + 0.1 * i, is_log)
        arr = rng.standard_normal((T, ny, nx)).astype(np.float32) + 0.3
        arr[:, ~ocean_b] = 0.0
        if v in cfg.cond_vars:
            arr[:, gap] = 0.0
        ds.data[v] = arr
    ds.valid_days = np.arange(cfg.k_days - 1 + 10, T, dtype=np.int64)
    ds.positions = ocean_patch_positions(ds.ocean, cfg.patch, max(cfg.patch // 4, 1),
                                         cfg.ocean_frac)
    ds.full_ocean = None
    ds._rng = np.random.default_rng(cfg.seed)
    ds.spec = _FakeSpec(t_unix)
    return ds, gap & ocean_b


def _selftest() -> int:
    import tempfile

    from diffusion.config import Config

    print("[gen_cond] selftest")
    cfg = Config(target=["SSH", "CHL"], cond_vars=["SSH_aviso"], k_days=2, patch=32,
                 channel_mult=[1, 2], use_ocean_mask=True, use_doy=True, norm_mode="zscore",
                 ocean_frac=0.5, sampler_steps=4, s_churn=5.0)
    ds, gap = _fake_dataset(cfg)
    spec = ds.spec
    lat_full = (25.0 + np.arange(ds.NY)[:, None] * 0.02 + 0 * np.arange(ds.NX)[None, :]
                ).astype(np.float32)
    device = torch.device("cpu")
    net = _ZeroNet()
    net.cc = cfg.cond_channels()
    meta = {"ckpt": "/fake/run/checkpoints/ckpt_step0000010.pt", "step": 10,
            "weights": "ema", "val_loss": None, "best_val": None}
    sampler = dict(num_steps=4, sigma_min=0.002, sigma_max=80.0, rho=7.0, s_churn=5.0,
                   s_noise=1.0, s_tmin=0.0, s_tmax=float("inf"))
    bmap = default_baseline_map(cfg)
    bkind = baseline_kinds(bmap, set())
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    check(bmap == {"SSH": "SSH_aviso", "CHL": None}, f"default baseline map {bmap}")
    names, idx_last, extras = cond_layout(cfg)
    check(names == ["SSH_aviso@t-1", "SSH_aviso@t-0", "ocean_mask", "doy_sin", "doy_cos"]
          and idx_last == [1, 2, 3, 4], f"cond layout {names} idx_last {idx_last}")

    with tempfile.TemporaryDirectory() as tmp:
        ns = argparse.Namespace(real_from=None, seed=0, skip_tail=0, store_cond="full",
                                gen_dtype="float16", split="val")
        # -- full geometry ---------------------------------------------------
        out_full = os.path.join(tmp, "samples_cond_full.npz")
        p = run_size(net, cfg, spec, ds, lat_full, spec.dx_m(), "full", 3, 2, 2, device,
                     ns, meta, sampler, bmap, bkind, out_full)
        d = np.load(out_full, allow_pickle=False)
        check(d["gen_norm"].shape == (3, 2, 2, ds.NY, ds.NX) and d["gen_norm"].dtype == np.float16,
              f"gen_norm {d['gen_norm'].shape} {d['gen_norm'].dtype}")
        check(np.all(d["gen_norm"] == 0), "zero net -> gen_norm == 0 exactly")
        days = d["days"]
        for i, day in enumerate(days):
            check(np.array_equal(d["truth_norm"][i, 0], ds.data["SSH"][day]),
                  f"truth_norm[{i}] is data[SSH][{day}]")
            check(np.array_equal(d["cond_norm"][i, idx_last[0]], ds.data["SSH_aviso"][day]),
                  "idx_last picks the NEWEST window step")
            check(np.array_equal(d["cond_norm"][i, 0], ds.data["SSH_aviso"][day - 1]),
                  "channel 0 is the oldest window step")
        check(np.array_equal(d["obs_avail"][:, 0], ~gap[None] & (ds.ocean > 0.5)[None]
                             & np.ones((3, 1, 1), bool)) or
              all(np.array_equal(d["obs_avail"][i, 0], (~gap) & (ds.ocean > 0.5))
                  for i in range(3)), "obs_avail == planted gap pattern & ocean")
        b = d["baseline_phys"]
        m, s, _ = ds.stats["SSH_aviso"]
        i0 = 0
        expect = ds.data["SSH_aviso"][days[i0]] * s + m
        av = d["obs_avail"][i0, 0]
        check(np.all(np.isnan(b[i0, 0][~av])) and
              np.allclose(b[i0, 0][av], expect[av], rtol=1e-5, atol=1e-5),
              "baseline_phys: NaN where unobserved, obs*std+mean elsewhere")
        check(np.all(np.isnan(b[:, 1])), "baseline_phys NaN for the unmapped target (CHL)")
        check(d["clim_day"].shape == (3, 2, 1, 1), f"clim_day shape {d['clim_day'].shape} (zscore)")
        check(list(d["is_log"]) == [False, True], f"is_log {list(d['is_log'])}")
        check(np.array_equal(d["mask"][0], ds.ocean > 0.5) and d["pos"].shape == (0, 2),
              "full: mask == ocean, pos empty")
        check(np.allclose(d["cond_norm"][:, 2], d["mask"].astype(np.float32)) and
              np.allclose(d["cond_norm"][0, 3], np.sin(ds.doy_ang[days[0]])),
              "extras: ocean_mask plane == mask, doy_sin == sin(doy angle)")
        mj = json.loads(str(d["meta"]))
        check(mj["pos_mode"] == "full" and mj["base_cadence"] == "daily"
              and abs(mj["step_seconds"] - 86400.0) < 1e-6, "meta pos_mode/base_cadence/step_seconds")

        # -- cache round trip --------------------------------------------------
        ns2 = argparse.Namespace(real_from=out_full, seed=0, skip_tail=0, store_cond="full",
                                 gen_dtype="float16", split="val")
        out2 = os.path.join(tmp, "again", "samples_cond_full.npz")
        os.makedirs(os.path.dirname(out2))
        run_size(net, cfg, spec, None, lat_full, spec.dx_m(), "full", 2, 2, 2, device,
                 ns2, meta, sampler, bmap, bkind, out2)
        d2 = np.load(out2, allow_pickle=False)
        check(np.array_equal(d2["truth_norm"], d["truth_norm"][:2]) and
              np.array_equal(d2["cond_norm"], d["cond_norm"][:2]) and
              np.array_equal(d2["days"], d["days"][:2]), "--real-from: truth/cond/days identical")
        bad_cfg = Config(**dict(cfg.as_dict(), k_days=3))
        try:
            _load_reference_cache(out_full, "full", 2, bad_cfg, spec, "val", bmap)
            check(False, "cache with the wrong k_days must be refused")
        except SystemExit as e:
            check("k_days" in str(e), f"cache with wrong k_days refused: {e}")

        # -- patch geometry (training patch, loader lattice) -------------------
        out_p = os.path.join(tmp, "samples_cond_32.npz")
        run_size(net, cfg, spec, ds, lat_full, spec.dx_m(), "32", 6, 2, 3, device,
                 ns, meta, sampler, bmap, bkind, out_p)
        dp = np.load(out_p, allow_pickle=False)
        pos = dp["pos"]
        check(pos.shape == (6, 2), f"patch pos {pos.shape}")
        check(all(np.array_equal(dp["mask"][i], (ds.ocean > 0.5)[y0:y0 + 32, x0:x0 + 32])
                  for i, (y0, x0) in enumerate(pos)), "patch: mask[d] == ocean crop at pos")
        check(all(tuple(p) in {tuple(q) for q in ds.positions} for p in pos),
              "patch: every pos is on the loader lattice")
        check(all(np.array_equal(dp["truth_norm"][i, 0],
                                 ds.data["SSH"][dp["days"][i], y0:y0 + 32, x0:x0 + 32])
                  for i, (y0, x0) in enumerate(pos)), "patch: truth is the same crop")
        check(np.all(dp["lat"][0] == lat_full[pos[0][0]:pos[0][0] + 32, pos[0][1]:pos[0][1] + 32]),
              "patch: lat is cropped at pos")
        # -- patch geometry at a size != training patch (rebuilt lattice) ------
        out_q = os.path.join(tmp, "samples_cond_16.npz")
        run_size(net, cfg, spec, ds, lat_full, spec.dx_m(), "16", 4, 2, 4, device,
                 ns, meta, sampler, bmap, bkind, out_q)
        dq = np.load(out_q, allow_pickle=False)
        check("rebuilt" in json.loads(str(dq["meta"]))["lattice_note"]
              and dq["gen_norm"].shape[-1] == 16, "patch != cfg.patch uses a rebuilt lattice")
        # -- store-cond last ---------------------------------------------------
        ns3 = argparse.Namespace(real_from=None, seed=0, skip_tail=0, store_cond="last",
                                 gen_dtype="float32", split="val")
        out_l = os.path.join(tmp, "last", "samples_cond_full.npz")
        os.makedirs(os.path.dirname(out_l))
        run_size(net, cfg, spec, ds, lat_full, spec.dx_m(), "full", 2, 2, 2, device,
                 ns3, meta, sampler, bmap, bkind, out_l)
        dl = np.load(out_l, allow_pickle=False)
        check(dl["cond_norm"].shape[1] == 4 and dl["gen_norm"].dtype == np.float32
              and json.loads(str(dl["cond_channel_names"])) == names[1:],
              "--store-cond last keeps newest step + extras; --gen-dtype float32 honoured")
        try:
            _load_reference_cache(out_l, "full", 2, cfg, spec, "val", bmap)
            check(False, "a 'last' cache with k>1 must be refused as --real-from")
        except SystemExit as e:
            check("store-cond" in str(e), "a 'last' cache with k>1 is refused as --real-from")

    print(f"[gen_cond] {'all checks passed' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
