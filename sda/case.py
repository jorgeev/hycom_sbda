"""Build an OSSE case: truth, climatology, masks and observation vectors per day.

    python -m sda.case --ckpt runs/prior_gulfstream/best.pt \
        --obs sda/configs/gs_store_obs.yaml --size 256 --n 24 --out-dir <dir>
    -> <dir>/case_256.npz

This is the only SDA step that opens the store, and it does so once per
geometry: ``ZarrWindowDataset`` preloads every target channel of the whole
record (~66 GB, ~1 h cold on gulfstream -- eval/gen_prior.py:29-37), so every
gamma / steps / sampler sweep in ``sda.assimilate`` reads the case file instead.

Per picked base-cadence step (an hour on gulfstream) and geometry the case holds
the normalised truth (``ds.full_frames``, cropped in patch mode), the day's
denormalisation climatology and std (``ds.denorm_params``), the ocean mask, the
latitude, and for every observation term of the spec its pixel mask and its
observation values on the grid (NaN where unobserved). Store-sourced terms read
the raw product frame at the matched cadence index (``time_match: nearest`` by
default -- ``spec.map_days`` would hand back the LAST product at or before the
hour, up to 23 h stale for a daily field); everything else is derived from the
truth frame by ``sda.obs.build_terms``.

Day picking spreads ``n`` steps evenly over the split (never the tail: on an
hourly axis the last n steps are one weather state) and drops ``--skip-tail``
steps whose July climatology is not a clean anomaly, as ``eval/gen_prior.py:87``.
``--require-coverage VAR`` restricts the pool to steps where the store variable
has any finite pixel (SWOT swaths exist in 6.6% of hours).

Units of the file: everything the network sees is normalised (anomaly / std);
``clim_day`` and ``std`` turn it back into physical units.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from diffusion.config import Config
from diffusion.data import ZarrWindowDataset, ocean_patch_positions
from diffusion.dataset_spec import resolve_spec
from diffusion.sample import config_from_ckpt
from sda import obs as O

CASE_KEYS = ("vars", "truth_norm", "clim_day", "std", "is_log", "mask", "mask_full", "lat",
             "lat_full", "dx_m", "days", "time_unix", "doy", "pos", "size", "config",
             "obs_spec", "terms", "y_grid", "obs_mask", "meta")


# ---------------------------------------------------------------------------
# checkpoint / config
# ---------------------------------------------------------------------------
def resolve_ckpt(path: str) -> str:
    """``--ckpt`` accepts a run dir (-> ``best.pt`` if present, else ``ckpt.pt``) or a .pt."""
    if os.path.isdir(path):
        for name in ("best.pt", "ckpt.pt"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"no best.pt / ckpt.pt in {path}")
    return path


def load_cfg(ckpt: str, dataset: str | None = None) -> tuple[Config, dict]:
    sd = torch.load(resolve_ckpt(ckpt), map_location="cpu", weights_only=False)
    cfg = config_from_ckpt(sd["config"])
    if dataset:
        cfg.dataset = dataset
    meta = {"ckpt": os.path.abspath(resolve_ckpt(ckpt)), "step": int(sd.get("step", -1)) + 1}
    return cfg, meta


def load_ckpt(path: str, device, weights: str = "ema"):
    """``(net, cfg, meta)`` -- copy of ``eval/gen_prior.py:63-84`` (EMA weights by default)."""
    from diffusion.train import build_net
    path = resolve_ckpt(path)
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_ckpt(sd["config"])
    net = build_net(cfg, "diffusion").to(device)
    net.load_state_dict(sd[weights])
    net.eval()
    meta = {"ckpt": os.path.abspath(path), "step": int(sd.get("step", -1)) + 1,
            "weights": weights,
            "val_loss": float(sd["val_loss"]) if "val_loss" in sd else None,
            "best_val": float(sd["best_val"]) if "best_val" in sd else None}
    return net, cfg, meta


# ---------------------------------------------------------------------------
# day / position selection
# ---------------------------------------------------------------------------
def pick_days(valid_days: np.ndarray, n: int, skip_tail: int) -> np.ndarray:
    """``n`` steps spread evenly, tail dropped (eval/gen_prior.py:87-111)."""
    pool = valid_days[:-skip_tail] if skip_tail > 0 else valid_days
    if pool.size == 0:
        raise ValueError("no usable steps after skip_tail")
    take = min(n, pool.size)
    days = pool[np.linspace(0, pool.size - 1, take).round().astype(int)]
    return np.resize(days, n) if n > take else days


def pick_contiguous(valid_days: np.ndarray, n: int, skip_tail: int) -> np.ndarray:
    """The LAST ``n`` consecutive steps of the split (after ``skip_tail``): a time
    series for animation rather than a spread of independent weather states.
    Refuses a gap in the base axis inside the window."""
    pool = valid_days[:-skip_tail] if skip_tail > 0 else valid_days
    if pool.size < n:
        raise ValueError(f"split has {pool.size} usable steps, asked for {n} contiguous")
    days = pool[-n:]
    if np.any(np.diff(days) != 1):
        raise ValueError("the last steps of the split are not contiguous on the base axis")
    return days


def steps_with_coverage(spec, var: str, candidates: np.ndarray, chunk: int = 48) -> np.ndarray:
    """Subset of base steps whose matched frame of ``var`` has any finite pixel."""
    cad = spec.cadence_of(var)
    idx = spec.map_days(var, candidates)
    keep = np.zeros(candidates.size, dtype=bool)
    order = np.argsort(idx)
    uniq = np.unique(idx[idx >= 0])
    has = {}
    for lo in range(0, uniq.size, chunk):
        sel = uniq[lo:lo + chunk]
        fr = spec.family.take(cad, var, sel)
        fin = np.isfinite(fr).reshape(fr.shape[0], -1).any(axis=1)
        has.update(zip(sel.tolist(), fin.tolist()))
    for j in order:
        keep[j] = idx[j] >= 0 and has.get(int(idx[j]), False)
    return candidates[keep]


def store_index(spec, var: str, day: int, time_match: str = "nearest") -> int:
    """Index of ``var`` on its own cadence for base step ``day``; -1 if none."""
    cad = spec.cadence_of(var)
    if cad is spec.base or getattr(cad, "name", None) == getattr(spec.base, "name", None):
        return int(day)
    if time_match == "last":
        return int(spec.map_days(var, [day])[0])
    t_base = float(spec.family.times(spec.base)[day])
    t_var = np.asarray(spec.family.times(cad), dtype=np.float64)
    return int(np.argmin(np.abs(t_var - t_base)))


def crop_positions(ds, cfg, patch: int, n: int, seed: int, fixed: bool = False,
                   pos: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """One crop corner per sample from the ocean-fraction lattice at ``patch``
    (rebuilt when patch != cfg.patch, as eval/gen_cond.py:116 does).
    ``fixed`` draws one corner and repeats it (a time series at one place);
    ``pos`` pins it explicitly."""
    if pos is not None:
        y0, x0 = int(pos[0]), int(pos[1])
        if not (0 <= y0 <= ds.NY - patch and 0 <= x0 <= ds.NX - patch):
            raise ValueError(f"--pos {pos} does not fit a {patch} patch in {ds.NY}x{ds.NX}")
        return [(y0, x0)] * n
    if fixed:
        return crop_positions(ds, cfg, patch, 1, seed) * n
    if patch == cfg.patch:
        ds.set_epoch_rng(seed)
        return [tuple(int(v) for v in ds.sample_position()) for _ in range(n)]
    lat = ocean_patch_positions(ds.ocean, patch, max(patch // 4, 1), cfg.ocean_frac)
    if not lat:
        raise ValueError(f"no {patch}x{patch} crop reaches ocean_frac {cfg.ocean_frac}")
    rng = np.random.default_rng(seed)
    return [tuple(int(v) for v in lat[rng.integers(len(lat))]) for _ in range(n)]


def coverage_positions(ocean: np.ndarray, patch: int, min_ocean_frac: float, frames: np.ndarray,
                       seed: int, min_cov: float = 0.10, stride: int | None = None,
                       log=print) -> tuple[np.ndarray, np.ndarray]:
    """One crop corner per step placed UNDER a store product's footprint (SWOT
    swaths): among the ocean lattice at ``patch``, draw uniformly from the corners
    where >= ``min_cov`` of the patch has finite pixels of ``frames[d]``; if no
    corner reaches ``min_cov``, take the best one. Returns ``(pos (D,2), cov (D,))``.
    ``--require-coverage`` alone only guarantees a swath SOMEWHERE in the domain;
    a random 256 crop of a 576x936 frame misses a 120 km swath most of the time."""
    stride = stride or max(patch // 8, 1)
    lat = ocean_patch_positions(ocean, patch, stride, min_ocean_frac)
    lat = np.asarray(lat, dtype=np.int64)
    rng = np.random.default_rng([seed, 7])
    D = frames.shape[0]
    pos = np.zeros((D, 2), np.int64)
    cov = np.zeros(D, np.float64)
    for d in range(D):
        fin = np.isfinite(np.asarray(frames[d])).astype(np.int64)
        c = np.zeros((fin.shape[0] + 1, fin.shape[1] + 1), np.int64)
        c[1:, 1:] = fin.cumsum(0).cumsum(1)
        y0, x0 = lat[:, 0], lat[:, 1]
        n = c[y0 + patch, x0 + patch] - c[y0, x0 + patch] - c[y0 + patch, x0] + c[y0, x0]
        frac = n / float(patch * patch)
        ok = np.flatnonzero(frac >= min_cov)
        j = int(rng.choice(ok)) if ok.size else int(np.argmax(frac))
        pos[d] = lat[j]
        cov[d] = frac[j]
    log(f"[case] placed {D} crops on coverage: {int((cov >= min_cov).sum())}/{D} reach "
        f"{min_cov:.0%}, footprint fraction min/median/max {cov.min():.2f}/{np.median(cov):.2f}/{cov.max():.2f}")
    return pos, cov


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------
def build_case(cfg, ds, spec, obs_spec: dict, days: np.ndarray, size: str, seed: int,
               term_names: list[str] | None = None, include_optional: bool = False,
               time_match: str | None = None, log=print, fixed_pos: bool = False,
               pos: tuple[int, int] | None = None, contiguous: bool = False,
               place_on: str | None = None, min_cov: float = 0.10) -> dict:
    full = size == "full"
    patch = None if full else int(size)
    hw = (ds.NY, ds.NX) if full else (patch, patch)
    C = len(cfg.target)
    D = len(days)
    anomaly = cfg.norm_mode == "anomaly"
    tcfgs = O.select_terms(obs_spec, term_names)
    if term_names is None and include_optional:
        tcfgs = tcfgs + list(obs_spec["optional_terms"])
    defaults = dict(obs_spec.get("defaults", {}))
    time_match = time_match or defaults.get("time_match", "nearest")
    store_vars = O.store_vars_needed({"terms": tcfgs, "optional_terms": []})
    T = len(tcfgs)
    dx_km = spec.dx_m() / 1000.0
    lat_full = np.asarray(spec.family.coord("coords/lat"), dtype=np.float32)
    ocean_full = ds.ocean > 0.5

    if full:
        pos = np.zeros((0, 2), np.int64)
    elif place_on and pos is None and not fixed_pos:
        # per-step placement under the footprint of a store product (SWOT)
        k = np.array([store_index(spec, place_on, int(day), time_match) for day in days])
        if np.any(k < 0):
            raise ValueError(f"{place_on} has no frame for steps {days[k < 0].tolist()}")
        fr = np.stack([np.asarray(spec.family.take(spec.cadence_of(place_on), place_on, np.array([kk]))[0])
                       for kk in k])
        pos, _cov = coverage_positions(ds.ocean, patch, cfg.ocean_frac, fr, seed, min_cov, log=log)
    else:
        pos = np.asarray(crop_positions(ds, cfg, patch, D, seed, fixed_pos, pos), dtype=np.int64)
    truth = np.zeros((D, C) + hw, np.float32)
    clim = np.zeros((D, C) + (hw if anomaly else (1, 1)), np.float32)
    mask = np.zeros((D,) + hw, bool)
    lat = np.zeros((D,) + hw, np.float32)
    y_grid = np.full((D, T) + hw, np.nan, np.float32)
    obs_mask = np.zeros((D, T) + hw, bool)
    std = np.array([ds.stats[v][1] for v in cfg.target], np.float64)
    is_log = np.array([bool(ds.stats[v][2]) for v in cfg.target])
    store_idx = np.full((D, len(store_vars)), -1, np.int64)
    t_base = np.asarray(spec.family.times(spec.base), dtype=np.float64)
    term_static = None

    frames = ds.full_frames([int(d) for d in days])
    t0 = time.time()
    for d, day in enumerate(days):
        day = int(day)
        item = next(frames)
        if full:
            ysl = xsl = slice(None)
        else:
            y0, x0 = pos[d]
            ysl, xsl = slice(y0, y0 + patch), slice(x0, x0 + patch)
        truth[d] = item["target"].numpy()[:, ysl, xsl]
        mask[d] = ocean_full[ysl, xsl]
        lat[d] = lat_full[ysl, xsl]
        for i, v in enumerate(cfg.target):
            m, _s, _ = ds.denorm_params(v, day)
            clim[d, i] = (np.broadcast_to(m, (ds.NY, ds.NX))[ysl, xsl] if anomaly else float(m))
        frames_d = {}
        for j, sv in enumerate(store_vars):
            k = store_index(spec, sv, day, time_match)
            store_idx[d, j] = k
            if k < 0:
                frames_d[sv] = None
                continue
            fr = spec.family.take(spec.cadence_of(sv), sv, np.array([k]))[0]
            frames_d[sv] = np.asarray(fr, dtype=np.float32)[ysl, xsl]
        ctx = O.ObsContext(ocean=torch.from_numpy(mask[d]), clim=torch.from_numpy(clim[d]),
                           std=torch.from_numpy(std.astype(np.float32)), is_log=list(is_log),
                           dx_km=dx_km, target=list(cfg.target))
        rng = np.random.default_rng([seed, day, d])
        terms = O.build_terms(tcfgs, defaults, ctx, torch.from_numpy(truth[d]), frames_d, rng)
        for ti, t in enumerate(terms):
            y_grid[d, ti] = t.y_grid
            obs_mask[d, ti] = t.mask.numpy()
        if term_static is None:
            term_static = [dict(name=t.name, var=t.var, channel=t.channel, kind=t.kind,
                                std_norm=t.std_norm, gamma=t.gamma, sigma_px=t.sigma_px,
                                demean=t.demean, anomaly=t.anomaly, y_source=t.y_source,
                                cfg=tcfgs[ti_])
                           for ti_, t in enumerate(terms)]
            log("[case] terms:\n" + O.describe(terms, hw))
        if (d + 1) % 4 == 0 or d + 1 == D:
            log(f"  [case] {d + 1}/{D} steps  {time.time() - t0:6.1f}s")

    step_s = float(np.median(np.diff(t_base))) if t_base.size > 1 else float("nan")
    meta = dict(size=size, seed=seed, D=D, split=ds.split, time_match=time_match,
                store_vars=store_vars, store_index=store_idx.tolist(),
                dataset_name=getattr(spec, "name", None),
                base_cadence=getattr(spec, "base_cadence", "unknown"), step_seconds=step_s,
                norm_mode=cfg.norm_mode, target=list(cfg.target), obs_name=obs_spec["name"],
                coverage=[float(obs_mask[:, ti].mean()) for ti in range(T)],
                contiguous=bool(contiguous), fixed_pos=bool(fixed_pos or pos is not None))
    return dict(
        vars=json.dumps(list(cfg.target)), truth_norm=truth, clim_day=clim, std=std,
        is_log=is_log, mask=mask, mask_full=ocean_full.astype(np.float32), lat=lat,
        lat_full=lat_full, dx_m=np.float64(spec.dx_m()), days=np.asarray(days, np.int64),
        time_unix=t_base[np.asarray(days, np.int64)],
        doy=(ds.doy_ang[np.asarray(days, np.int64)] / (2 * np.pi) * 365.2425).astype(np.float32),
        pos=pos, size=size, config=json.dumps(cfg.as_dict()), obs_spec=json.dumps(obs_spec),
        terms=json.dumps(term_static), y_grid=y_grid, obs_mask=obs_mask, meta=json.dumps(meta),
    )


def load_case(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    for k in ("vars", "config", "obs_spec", "terms", "meta"):
        out[k] = json.loads(str(d[k]))
    out["size"] = str(d["size"])
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="run dir or .pt (only its config is read)")
    ap.add_argument("--obs", help="observing-system YAML (sda/configs/*.yaml)")
    ap.add_argument("--out-dir", help="directory for case_<size>.npz")
    ap.add_argument("--size", default="256", help="comma-separated: 'full' and/or patch sizes")
    ap.add_argument("--n", type=int, default=24, help="target steps per geometry")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-tail", type=int, default=None,
                    help="drop trailing steps (default 24 if hourly, else 0)")
    ap.add_argument("--contiguous", type=int, default=0,
                    help="instead of --n spread steps, take the LAST N consecutive steps of the "
                         "split (a time series); patch crops then share one position")
    ap.add_argument("--pos", default=None, help="y0,x0 crop corner to pin (patch sizes only)")
    ap.add_argument("--require-coverage", default=None,
                    help="store var that must have finite pixels at every picked step")
    ap.add_argument("--place-on-coverage", default=None,
                    help="store var whose footprint each patch crop is placed under (e.g. ssha_swot); "
                         "implies --require-coverage of the same var")
    ap.add_argument("--min-coverage", type=float, default=0.10,
                    help="min fraction of the patch under the --place-on-coverage footprint (0.10)")
    ap.add_argument("--terms", default=None, help="comma-separated term names (default: required terms)")
    ap.add_argument("--include-optional", action="store_true", help="also build optional_terms")
    ap.add_argument("--time-match", default=None, choices=(None, "nearest", "last"))
    ap.add_argument("--dataset", default=None, help="override the checkpoint's dataset descriptor")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        raise SystemExit(_selftest())
    for k in ("ckpt", "obs", "out_dir"):
        if getattr(args, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required")

    cfg, cmeta = load_cfg(args.ckpt, args.dataset)
    if cfg.cond_channels() != 0:
        raise SystemExit(f"sda expects an unconditional prior; cfg has cond_channels="
                         f"{cfg.cond_channels()} (cond_vars={cfg.cond_vars}, "
                         f"use_ocean_mask={cfg.use_ocean_mask}, use_doy={cfg.use_doy})")
    obs_spec = O.load_obs_spec(args.obs)
    spec = resolve_spec(cfg)
    sizes = [s.strip() for s in args.size.split(",")]
    ds_factor = 2 ** (len(cfg.channel_mult) - 1)
    for s in sizes:
        if s != "full" and int(s) % ds_factor:
            raise SystemExit(f"--size {s} must divide {ds_factor}")
    print(f"[case] ckpt={cmeta['ckpt']} step={cmeta['step']} target={cfg.target} "
          f"norm_mode={cfg.norm_mode} obs={obs_spec['name']}", flush=True)
    t0 = time.time()
    ds = ZarrWindowDataset(cfg, split=args.split, spec=spec)
    print(f"[case] dataset ready in {time.time() - t0:.0f}s  ({ds.NY}x{ds.NX}, "
          f"{ds.valid_days.size} valid steps)", flush=True)
    skip = args.skip_tail
    if skip is None:
        skip = 24 if getattr(spec, "base_cadence", "") == "hourly" else 0
    pool = ds.valid_days
    if args.contiguous:
        days = pick_contiguous(pool, args.contiguous, skip)
        t = np.asarray(spec.family.times(spec.base), dtype=np.float64)[days]
        print(f"[case] {args.contiguous} contiguous steps {days.min()}..{days.max()} "
              f"({time.strftime('%Y-%m-%d %H:%M', time.gmtime(t[0]))} .. "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(t[-1]))} UTC), one crop position",
              flush=True)
    else:
        req = args.require_coverage or args.place_on_coverage
        if req:
            pool = steps_with_coverage(spec, req, pool)
            print(f"[case] {pool.size} steps carry {req}", flush=True)
        days = pick_days(pool, args.n, skip)
        print(f"[case] {args.n} steps spanning {days.min()}..{days.max()}", flush=True)
    pos = tuple(int(v) for v in args.pos.split(",")) if args.pos else None
    term_names = args.terms.split(",") if args.terms else None
    os.makedirs(args.out_dir, exist_ok=True)
    for size in sizes:
        t0 = time.time()
        case = build_case(cfg, ds, spec, obs_spec, days, size, args.seed, term_names,
                          args.include_optional, args.time_match, fixed_pos=bool(args.contiguous),
                          pos=pos, contiguous=bool(args.contiguous),
                          place_on=args.place_on_coverage, min_cov=args.min_coverage)
        out = os.path.join(args.out_dir, f"case_{size}.npz")
        np.savez(out, **case)
        print(f"[case] wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t0:.0f}s)",
              flush=True)


# ---------------------------------------------------------------------------
def fake_obs_spec() -> dict:
    return dict(name="fake", defaults={"gamma": 0.1, "border_px": 1}, terms=[
        {"name": "p_store", "var": "sst", "kind": "pointwise", "y_source": "store:sst_sat",
         "mask": {"source": "store_finite", "var": "sst_sat"}, "noise_std": 0.1},
        {"name": "b_truth", "var": "ssh", "kind": "blur", "sigma_km": 4.0, "demean": True,
         "y_source": "truth", "mask": {"source": "full"}, "noise_std": 0.05},
        {"name": "b_store", "var": "sss", "kind": "blur", "sigma_km": 6.0,
         "y_source": "store:sss_sat", "mask": {"source": "full"}, "noise_std": 0.05},
        {"name": "full_t", "var": "tau", "kind": "pointwise", "y_source": "truth",
         "mask": {"source": "full"}, "noise_std_norm": 0.01},
    ], optional_terms=[
        {"name": "tracks", "var": "ssh", "kind": "pointwise", "y_source": "truth",
         "mask": {"source": "tracks", "spacing_km": 30}, "noise_std": 0.02}])


def _selftest() -> int:
    import tempfile
    from sda._testing import _fake_dataset
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    print("[sda.case] selftest")
    cfg = Config(target=["ssh", "sst", "sss", "tau"], cond_vars=[], k_days=1, patch=16,
                 channel_mult=[1, 2], norm_mode="zscore", ocean_frac=0.5,
                 use_ocean_mask=False, use_doy=False)
    ds = _fake_dataset(cfg)
    spec = ds.spec
    obs_spec = fake_obs_spec()
    days = pick_days(ds.valid_days, 3, 0)
    check(days.size == 3 and days[0] == ds.valid_days[0] and days[-1] == ds.valid_days[-1],
          f"pick_days spreads over the split {days.tolist()}")
    cd = pick_contiguous(ds.valid_days, 5, 2)
    check(cd.tolist() == ds.valid_days[-7:-2].tolist(), f"pick_contiguous takes the last N before the tail {cd.tolist()}")
    cc = build_case(cfg, ds, spec, obs_spec, cd, "32", seed=0, fixed_pos=True, contiguous=True,
                    log=lambda *a: None)
    check(len({tuple(p) for p in cc["pos"].tolist()}) == 1 and json.loads(cc["meta"])["contiguous"],
          "contiguous case shares one crop position")
    cp = build_case(cfg, ds, spec, obs_spec, cd[:2], "32", seed=0, pos=(3, 7), log=lambda *a: None)
    check(cp["pos"].tolist() == [[3, 7], [3, 7]], "--pos pins the crop corner")
    cov = steps_with_coverage(spec, "sst_sat", ds.valid_days[:10])
    check(cov.size == 10, "steps_with_coverage keeps steps with finite pixels")
    check(store_index(spec, "sss_sat", 20, "nearest") in (2, 3) and
          store_index(spec, "sss_sat", 20, "last") == 2 and store_index(spec, "sst_sat", 20) == 20,
          "store_index: nearest vs last on a weekly cadence, identity on the base cadence")
    for size in ("32", "full"):
        case = build_case(cfg, ds, spec, obs_spec, days, size, seed=0, include_optional=True,
                          log=lambda *a: None)
        hw = (ds.NY, ds.NX) if size == "full" else (32, 32)
        check(case["truth_norm"].shape == (3, 4) + hw and case["y_grid"].shape == (3, 5) + hw
              and case["obs_mask"].shape == (3, 5) + hw and case["clim_day"].shape == (3, 4, 1, 1),
              f"[{size}] array shapes {case['truth_norm'].shape} {case['y_grid'].shape}")
        ym, om = case["y_grid"], case["obs_mask"]
        check(bool(np.isfinite(ym[om]).all()) and bool(np.isnan(ym[~om]).all()),
              f"[{size}] y_grid finite exactly on obs_mask")
        check(not om[:, :, ~case["mask"][0]].any() if size == "full" else True,
              f"[{size}] no observation on land")
        terms = json.loads(case["terms"])
        check([t["name"] for t in terms] == ["p_store", "b_truth", "b_store", "full_t", "tracks"]
              and terms[1]["demean"] and abs(terms[1]["std_norm"] - 0.05 / 1.0) < 1e-9
              and abs(terms[3]["std_norm"] - 0.01) < 1e-12,
              f"[{size}] term metadata (std_norm from physical / normalised)")
        # full_t observes truth with 0.01 noise: y ~= truth on every ocean pixel
        ti = 3
        err = np.abs(ym[:, ti][om[:, ti]] - case["truth_norm"][:, 3][om[:, ti]])
        check(err.max() < 0.06, f"[{size}] truth-sourced pointwise y within noise of truth ({err.max():.3f})")
        meta = json.loads(case["meta"])
        check(meta["store_vars"] == ["sst_sat", "sss_sat"] and len(meta["coverage"]) == 5
              and 0.3 < meta["coverage"][0] < 0.7, f"[{size}] meta store_vars/coverage {meta['coverage'][0]:.2f}")
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, f"case_{size}.npz")
            np.savez(p, **case)
            back = load_case(p)
            check(set(CASE_KEYS) <= set(back) and back["size"] == size
                  and back["terms"][0]["name"] == "p_store", f"[{size}] round-trip through npz")
    print(f"[sda.case] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
