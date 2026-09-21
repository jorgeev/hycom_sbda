"""Score-based data assimilation: case + prior checkpoint -> guided ensembles.

    python -m sda.assimilate --ckpt runs/prior_gulfstream/best.pt \
        --case <dir>/case_256.npz --out-dir <dir> [--k 24] [--guidance on|off] \
        [--sampler vp|edm] [--steps 256] [--corrections 0] [--tau 0.3] [--gamma-scale 1]
    -> <dir>/samples_sda_256.npz          (--out overrides the file name)

For every step in the case, draws ``K`` members of ``p(x | y)`` with GenDA's
algorithm (``--sampler vp``: Rozet & Louppe's VP predictor-corrector over the
EDM prior, 256 steps, Gaussian-likelihood guidance with Tweedie's estimate and
autograd through the network; ``sda/vpsde.py``, ``sda/guidance.py``) or the
EDM-native equivalent (``--sampler edm``: Karras sigmas, Euler/Heun, the same
guided denoiser). ``--guidance off`` is the unguided control with identical
seeds -- the null result any guided run has to beat.

Output: ``samples_sda_<size>.npz`` in the ``eval.diagnostics_cond.CondSamples``
schema, so ``python -m eval.diagnostics_cond --samples ... --out ...`` scores it
unchanged (pointwise terms appear as ``cond_vars`` / ``obs_avail`` and as the
per-variable baseline; a demeaned term is flagged ``baseline_kind: anomaly``).
The SDA-specific extras (every term's grid, mask and settings) travel alongside
for ``sda.diagnostics_sda``.

Cost: one forward AND one backward through the UNet per member per step.
Memory scales with member chunk x activations (~0.6 GB / member at 128^2,
~2.3 GB at 256^2, ~19 GB at 576x936 in bf16); ``--member-chunk auto`` starts
from a table and halves on OOM. ``--grad-ckpt`` trades ~1/3 of that memory for
one extra forward per block. The network runs in bf16 (GenDA: fp16), the
likelihood in fp32; ``--check-precision`` measures the bf16-vs-fp32 guidance
agreement on the first step before committing GPU hours.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from diffusion.edm import karras_sigmas
from sda import obs as O
from sda.case import load_case, load_ckpt
from sda.guidance import GaussianGuidance, per_term_gradients, wrap_checkpoint
from sda.vpsde import PriorDenoiser, edm_sample, vp_sample, vp_sigma_edm

# Keys eval.diagnostics_cond.CondSamples reads (eval/diagnostics_cond.py:135-176).
COND_KEYS = ("vars", "cond_vars", "k_days", "extras", "meta", "config", "gen_norm", "truth_norm",
             "std", "is_log", "clim_day", "cond_norm", "cond_channel_names", "obs_avail",
             "obs_age_s", "baseline_phys", "baseline_map", "baseline_kind", "baseline_bias",
             "mask", "mask_full", "lat", "lat_full", "dx_m", "size", "days", "time_unix", "doy",
             "pos")

# measured on an A100-80GB with the 7-channel genda prior, bf16: 256^2 x 24 members
# = 32 GB peak, 0.42 s per guided step; full frame (576x936) is 8.2x the pixels.
CHUNK_DEFAULT = [(128 * 128, 24), (256 * 256, 24), (400 * 400, 8), (10 ** 9, 4)]


def default_chunk(hw, grad_ckpt: bool) -> int:
    px = hw[0] * hw[1]
    for lim, m in CHUNK_DEFAULT:
        if px <= lim:
            return m * (2 if grad_ckpt and m < 24 else 1)
    return 1


# ---------------------------------------------------------------------------
# terms from a case
# ---------------------------------------------------------------------------
def terms_for_day(case: dict, d: int, names: list[str] | None, gamma_scale: float,
                  device) -> list[O.ObsTerm]:
    out = []
    for ti, t in enumerate(case["terms"]):
        if names is not None and t["name"] not in names:
            continue
        m = np.asarray(case["obs_mask"][d, ti], dtype=bool)
        yg = np.asarray(case["y_grid"][d, ti], dtype=np.float32)
        term = O.ObsTerm(name=t["name"], var=t["var"], channel=int(t["channel"]), kind=t["kind"],
                         mask=torch.from_numpy(m).to(device), std_norm=float(t["std_norm"]),
                         gamma=float(t["gamma"]) * gamma_scale, sigma_px=float(t["sigma_px"]),
                         demean=bool(t["demean"]), anomaly=bool(t.get("anomaly", False)), y_source=t["y_source"])
        term.y_norm = torch.from_numpy(yg[m]).to(device)
        term.y_grid = yg
        out.append(term)
    return out


def context_for_day(case: dict, d: int, device) -> O.ObsContext:
    std = np.asarray(case["std"], dtype=np.float32)
    return O.ObsContext(ocean=torch.from_numpy(np.asarray(case["mask"][d], bool)).to(device),
                        clim=torch.from_numpy(np.asarray(case["clim_day"][d], np.float32)).to(device),
                        std=torch.from_numpy(std).to(device), is_log=[bool(v) for v in case["is_log"]],
                        dx_km=float(case["dx_m"]) / 1000.0, target=list(case["vars"]))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_members(guid, shape, args, cfg, device, generator, land_mask):
    if args.sampler == "vp":
        return vp_sample(guid, shape, steps=args.steps, corrections=args.corrections, tau=args.tau,
                         eta=args.eta, generator=generator, device=device, land_mask=land_mask,
                         land_mode=args.land_mode)
    sigmas = karras_sigmas(args.steps, cfg.sigma_min, args.sigma_max, cfg.rho, device)
    return edm_sample(guid, shape, sigmas, heun=args.edm_heun, generator=generator, device=device,
                     land_mask=land_mask, land_mode=args.land_mode)


def run_case(net, cfg, case: dict, args, device, log=print) -> dict:
    """The whole loop: returns the payload written to ``samples_sda_<size>.npz``."""
    D, C = case["truth_norm"].shape[:2]
    hw = tuple(case["truth_norm"].shape[2:])
    K = args.k
    names = args.terms.split(",") if args.terms else None
    if names is not None:
        known = [t["name"] for t in case["terms"]]
        bad = [n for n in names if n not in known]
        if bad:
            raise SystemExit(f"--terms {bad} not in case terms {known}")
    shard_i, shard_n = (0, 1) if not args.day_shard else tuple(int(v) for v in args.day_shard.split("/"))
    my_days = [d for d in range(D) if d % shard_n == shard_i]

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else None
    dfn = PriorDenoiser(net, cfg.cond_channels(), autocast_dtype=autocast_dtype)
    if args.grad_ckpt:
        log(f"[sda] gradient checkpointing on {wrap_checkpoint(net)} blocks")
    chunk = default_chunk(hw, args.grad_ckpt) if args.member_chunk == "auto" else int(args.member_chunk)
    chunk = max(1, min(chunk, K))
    out_dtype = np.float16 if args.gen_dtype == "float16" else np.float32
    gen = np.zeros((D, K, C) + hw, dtype=out_dtype)
    log_p = np.full((D,), np.nan)
    g = torch.Generator(device=device)
    guidance_on = args.guidance == "on"
    t_all = time.time()
    first = True
    peak = 0.0
    for d in my_days:
        day = int(case["days"][d])
        ctx = context_for_day(case, d, device)
        terms = terms_for_day(case, d, names, args.gamma_scale, device)
        A = O.build_operator(terms, ctx)
        y, std, gam = O.concat_y(terms, 1, device)
        n_obs = int(y.shape[1])
        guid = GaussianGuidance(dfn, A, y[0], std, gam, detach=args.detach,
                                enabled=guidance_on and n_obs > 0)
        land = None if bool(ctx.ocean.all()) else ~ctx.ocean
        if first:
            log(f"[sda] {D} steps x {K} members at {hw[0]}x{hw[1]}, {n_obs} obs, guidance "
                f"{'on' if guid.enabled else 'OFF'}, sampler={args.sampler} steps={args.steps} "
                f"corrections={args.corrections} chunk={chunk} precision={args.precision}")
            log("[sda] terms:\n" + O.describe(terms, hw))
        rem, off, j = K, 0, 0
        t0 = time.time()
        while rem > 0:
            m = min(chunk, rem)
            g.manual_seed(args.seed + 1000 * day + off)
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                x = sample_members(guid, (m, C) + hw, args, cfg, device, g, land)
            except torch.cuda.OutOfMemoryError:
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                log(f"[sda] OOM at member chunk {m}; retrying with {chunk}")
                torch.cuda.empty_cache()
                continue
            if first and args.verbose and guid.enabled:
                sig_probe = [float(vp_sigma_edm(torch.tensor(t), args.eta)) for t in (0.9, 0.5, 0.2)]
                for s in sig_probe:
                    xs = torch.randn((min(m, 2), C) + hw, device=device) * s
                    norms = per_term_gradients(guid, terms, ctx, xs, s)
                    log(f"[sda]   |sigma^2 grad log p_i| at sigma={s:8.3f}: " +
                        ", ".join(f"{k}={v:.3g}" for k, v in norms.items()))
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device) / 2 ** 30)
            gen[d, j:j + m] = x.float().cpu().numpy().astype(out_dtype)
            rem -= m
            off += 1
            j += m
            if first:
                log(f"[sda] first chunk of {m}: {time.time() - t0:.1f}s"
                    + (f", peak {peak:.1f} GB" if device.type == "cuda" else ""))
                first = False
        if guid.last_log_p is not None:
            log_p[d] = guid.last_log_p
        el = time.time() - t_all
        done = my_days.index(d) + 1
        log(f"  [sda] {done}/{len(my_days)} step {day}  {el:7.1f}s  {el / done:6.1f}s/step"
            + (f"  log_p/obs={guid.last_log_p / max(n_obs, 1) / K:.3f}" if guid.last_log_p is not None else ""))

    return assemble_payload(case, gen, args, cfg, names, log_p, dict(
        peak_gb=peak, chunk=chunk, seconds=time.time() - t_all, shard=args.day_shard,
        days_done=my_days))


# ---------------------------------------------------------------------------
# output in the CondSamples schema
# ---------------------------------------------------------------------------
def assemble_payload(case: dict, gen: np.ndarray, args, cfg, names, log_p, timing: dict) -> dict:
    D, K, C = gen.shape[:3]
    hw = gen.shape[3:]
    terms = [t for t in case["terms"] if names is None or t["name"] in names]
    tidx = [i for i, t in enumerate(case["terms"]) if names is None or t["name"] in names]
    vars_ = list(case["vars"])
    std = np.asarray(case["std"], np.float64)
    clim = np.asarray(case["clim_day"], np.float32)
    point = [(i, t) for i, t in zip(tidx, terms) if t["kind"] == "pointwise"]
    cond_names = [t["name"] for _, t in point]
    cond_norm = np.zeros((D, len(point)) + hw, np.float32)
    obs_avail = np.zeros((D, len(point)) + hw, bool)
    for j, (i, _t) in enumerate(point):
        yg = np.asarray(case["y_grid"][:, i], np.float32)
        obs_avail[:, j] = np.asarray(case["obs_mask"][:, i], bool)
        cond_norm[:, j] = np.nan_to_num(yg, nan=0.0)
    # baseline: the first pointwise term observing each variable, in physical units
    bmap, bkind = {}, {}
    baseline = np.full((D, C) + hw, np.nan, np.float32)
    for ci, v in enumerate(vars_):
        hit = [(j, t) for j, (_i, t) in enumerate(point) if t["var"] == v]
        if not hit:
            bmap[v] = None
            bkind[v] = "absolute"
            continue
        j, t = hit[0]
        bmap[v] = t["name"]
        bkind[v] = "anomaly" if t["demean"] else "absolute"
        phys = cond_norm[:, j] * std[ci] + np.broadcast_to(clim[:, ci], (D,) + hw)
        if bool(case["is_log"][ci]):
            phys = np.power(10.0, phys)
        baseline[:, ci] = np.where(obs_avail[:, j], phys, np.nan)
    sda_meta = dict(guidance=args.guidance, sampler=args.sampler, steps=args.steps,
                    corrections=args.corrections, tau=args.tau, eta=args.eta,
                    gamma_scale=args.gamma_scale, precision=args.precision,
                    edm_heun=args.edm_heun, sigma_max=args.sigma_max, detach=args.detach,
                    land_mode=args.land_mode, grad_ckpt=args.grad_ckpt, terms=[t["name"] for t in terms],
                    obs_name=case["meta"].get("obs_name"), case_meta=case["meta"],
                    log_p_last=[None if not np.isfinite(v) else float(v) for v in log_p], **timing)
    meta = dict(ckpt=args._ckpt_meta.get("ckpt"), step=args._ckpt_meta.get("step"),
                weights=args._ckpt_meta.get("weights"), size=case["size"], split=case["meta"].get("split"),
                seed=args.seed, D=D, K=K, member_batch=timing.get("chunk"),
                sampler=dict(num_steps=args.steps, s_churn=0.0, s_noise=1.0, sigma_min=cfg.sigma_min,
                             sigma_max=args.sigma_max if args.sampler == "edm" else None, rho=cfg.rho,
                             kind=args.sampler, corrections=args.corrections, tau=args.tau),
                store_cond="full", gen_dtype=args.gen_dtype, dx_m=float(case["dx_m"]),
                pos_mode="full" if case["size"] == "full" else "lattice", target=vars_,
                cond_vars=cond_names, k_days=1, extras=[], baseline_map=bmap, baseline_kind=bkind,
                norm_mode=case["meta"].get("norm_mode"), dataset_name=case["meta"].get("dataset_name"),
                base_cadence=case["meta"].get("base_cadence"), step_seconds=case["meta"].get("step_seconds"),
                sda=sda_meta, timings=dict(generate_s=timing.get("seconds")))
    return dict(
        vars=json.dumps(vars_), cond_vars=json.dumps(cond_names), k_days=np.int64(1),
        extras=json.dumps([]), gen_norm=gen, truth_norm=np.asarray(case["truth_norm"], np.float32),
        cond_norm=cond_norm, cond_channel_names=json.dumps(cond_names), obs_avail=obs_avail,
        obs_age_s=np.zeros((D, len(point)), np.float64), baseline_phys=baseline,
        baseline_map=json.dumps(bmap), baseline_kind=json.dumps(bkind),
        baseline_bias=np.zeros(C, np.float64), clim_day=clim, std=std,
        is_log=np.asarray(case["is_log"], bool), mask=np.asarray(case["mask"], bool),
        mask_full=np.asarray(case["mask_full"], np.float32), lat=np.asarray(case["lat"], np.float32),
        lat_full=np.asarray(case["lat_full"], np.float32), dx_m=np.float64(case["dx_m"]),
        days=np.asarray(case["days"], np.int64), time_unix=np.asarray(case["time_unix"], np.float64),
        doy=np.asarray(case["doy"], np.float32), pos=np.asarray(case["pos"], np.int64),
        size=case["size"], config=json.dumps(case["config"]),
        # SDA extras
        y_grid=np.asarray(case["y_grid"][:, tidx], np.float32),
        obs_mask=np.asarray(case["obs_mask"][:, tidx], bool), terms=json.dumps(terms),
        obs_spec=json.dumps(case["obs_spec"]), meta=json.dumps(meta),
    )


def merge_shards(paths: list[str], out: str):
    """Concatenate day-sharded outputs (zeros where a shard did not run) into one file."""
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    base = parts[0]
    gen = np.zeros_like(base["gen_norm"])
    for p in parts:
        done = json.loads(str(p["meta"]))["sda"]["days_done"]
        gen[done] = p["gen_norm"][done]
    base["gen_norm"] = gen
    meta = json.loads(str(base["meta"]))
    meta["sda"]["shard"] = None
    meta["sda"]["days_done"] = sorted(set(sum((json.loads(str(p["meta"]))["sda"]["days_done"] for p in parts), [])))
    base["meta"] = json.dumps(meta)
    np.savez(out, **base)


# ---------------------------------------------------------------------------
# precision check
# ---------------------------------------------------------------------------
def check_precision(net, cfg, case, args, device, log=print) -> bool:
    """Cosine similarity and norm ratio of the guidance term sigma^2 grad log p
    between bf16 and fp32 network passes, at three noise levels."""
    d = 0
    ctx = context_for_day(case, d, device)
    terms = terms_for_day(case, d, args.terms.split(",") if args.terms else None, args.gamma_scale, device)
    A = O.build_operator(terms, ctx)
    y, std, gam = O.concat_y(terms, 1, device)
    C = len(case["vars"])
    hw = tuple(case["truth_norm"].shape[2:])
    ok = True
    for t in (0.9, 0.5, 0.1):
        s = float(vp_sigma_edm(torch.tensor(t), args.eta))
        x = torch.randn((2, C) + hw, device=device, generator=torch.Generator(device=device).manual_seed(0)) * s
        outs = {}
        for name, dt in (("bf16", torch.bfloat16), ("fp32", None)):
            dfn = PriorDenoiser(net, cfg.cond_channels(), autocast_dtype=dt)
            guid = GaussianGuidance(dfn, A, y[0], std, gam)
            dpr, dpo = guid.prior_and_guided(x, s)
            outs[name] = (dpo - dpr).flatten()
        a, b = outs["bf16"], outs["fp32"]
        cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
        ratio = float(a.norm() / (b.norm() + 1e-12))
        good = cos > 0.99 and 0.95 < ratio < 1.05
        ok &= good
        log(f"[sda] precision t={t} sigma={s:8.3f}: cos={cos:.4f} norm ratio={ratio:.3f} "
            f"{'ok' if good else 'MISMATCH'}")
    return ok


# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="run dir or .pt")
    ap.add_argument("--case", help="case_<size>.npz from sda.case")
    ap.add_argument("--out-dir", help="directory for samples_sda_<size>.npz")
    ap.add_argument("--out", default=None, help="explicit output file name")
    ap.add_argument("--k", type=int, default=24, help="ensemble members")
    ap.add_argument("--member-chunk", default="auto", help="members per pass (auto = table, halves on OOM)")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--corrections", type=int, default=0, help="Langevin corrections per step (vp)")
    ap.add_argument("--tau", type=float, default=0.3, help="Langevin amplitude (vp)")
    ap.add_argument("--eta", type=float, default=1e-3, help="VP schedule eta (sigma/mu max = 1/eta)")
    ap.add_argument("--gamma-scale", type=float, default=1.0, help="multiply every term's gamma")
    ap.add_argument("--guidance", default="on", choices=("on", "off"))
    ap.add_argument("--sampler", default="vp", choices=("vp", "edm"))
    ap.add_argument("--edm-heun", action="store_true", help="2nd-order correction (edm sampler)")
    ap.add_argument("--sigma-max", type=float, default=1000.0, help="edm sampler sigma_max")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--detach", action="store_true", help="drop the network Jacobian in the guidance")
    ap.add_argument("--land-mode", default="free", choices=("free", "replace"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--day-shard", default=None, help="i/n: run days i, i+n, ... only")
    ap.add_argument("--terms", default=None, help="comma-separated subset of the case's terms")
    ap.add_argument("--gen-dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--weights", default="ema", choices=("ema", "model"))
    ap.add_argument("--verbose", action="store_true", help="per-term gradient norms on the first chunk")
    ap.add_argument("--check-precision", action="store_true", help="bf16 vs fp32 guidance check, then exit")
    ap.add_argument("--merge", nargs="+", default=None, help="merge day-sharded outputs into --out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--with-diagnostics", action="store_true", help="selftest: also run eval.diagnostics_cond")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.selftest:
        raise SystemExit(_selftest(args.with_diagnostics))
    if args.merge:
        if not args.out:
            ap.error("--merge needs --out")
        merge_shards(args.merge, args.out)
        print(f"[sda] merged {len(args.merge)} shards -> {args.out}")
        return
    for k in ("ckpt", "case", "out_dir"):
        if getattr(args, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        # a pinned GPU outside the job's device cgroup (e.g. a 2-GPU SLURM gres with
        # a hand-picked physical id) shows up as "no CUDA": on the CPU one full-frame
        # step takes 17 h, so fail instead of silently falling back
        raise SystemExit(f"[sda] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} but no CUDA "
                         f"device is usable (outside this job's cgroup? request every GPU with --gres)")
    case = load_case(args.case)
    net, cfg, cmeta = load_ckpt(args.ckpt, device, args.weights)
    args._ckpt_meta = cmeta
    if cfg.cond_channels() != 0:
        raise SystemExit(f"sda expects an unconditional prior (cond_channels()==0); got "
                         f"cond_vars={cfg.cond_vars} use_ocean_mask={cfg.use_ocean_mask} use_doy={cfg.use_doy}")
    if list(cfg.target) != list(case["vars"]):
        raise SystemExit(f"case targets {case['vars']} != checkpoint targets {list(cfg.target)}")
    print(f"[sda] ckpt={cmeta['ckpt']} step={cmeta['step']} weights={args.weights} device={device}",
          flush=True)
    if args.check_precision:
        ok = check_precision(net, cfg, case, args, device)
        raise SystemExit(0 if ok else 2)
    payload = run_case(net, cfg, case, args, device)
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out or (f"samples_sda_{case['size']}" +
                        (f"_shard{args.day_shard.replace('/', 'of')}" if args.day_shard else "") + ".npz")
    out = os.path.join(args.out_dir, name)
    t0 = time.time()
    np.savez(out, **payload)                    # plain savez: fp16 noise does not compress
    print(f"[sda] wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB, {time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
def _selftest(with_diagnostics: bool = False) -> int:
    import subprocess
    import tempfile

    from diffusion.config import Config
    from sda._testing import _GaussPriorNet, _ZeroNet, _fake_dataset
    from sda.case import build_case, fake_obs_spec

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    print("[sda.assimilate] selftest")
    device = torch.device("cpu")
    cfg = Config(target=["ssh", "sst", "sss", "tau"], cond_vars=[], k_days=1, patch=16,
                 channel_mult=[1, 2], norm_mode="zscore", ocean_frac=0.5, sigma_data=1.0,
                 use_ocean_mask=False, use_doy=False)
    ds = _fake_dataset(cfg)
    obs_spec = fake_obs_spec()
    days = ds.valid_days[[0, 5]]
    case = build_case(cfg, ds, ds.spec, obs_spec, days, "32", seed=0, log=lambda *a: None)
    for k in ("vars", "config", "obs_spec", "terms", "meta"):   # as load_case would
        case[k] = json.loads(case[k])

    def run(net, **over):
        argv = ["--ckpt", "x", "--case", "x", "--out-dir", "x", "--k", "8", "--steps", "32",
                "--member-chunk", "4", "--precision", "fp32"]
        for k, v in over.items():
            argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        a = build_parser().parse_args(argv)
        a._ckpt_meta = {"ckpt": "/fake/run/checkpoints/ckpt_step0000010.pt", "step": 10, "weights": "ema"}
        return run_case(net, cfg, case, a, device, log=lambda *a_: None), a

    # 1. zero net, guidance off: VP sample is eta * x_1 -> |gen| < 1e-2
    p, _ = run(_ZeroNet(0), guidance="off")
    check(p["gen_norm"].shape == (2, 8, 4, 32, 32) and p["gen_norm"].dtype == np.float16
          and float(np.abs(p["gen_norm"].astype(np.float32)).max()) < 1e-2,
          f"zero net, guidance off: gen_norm {p['gen_norm'].shape}, max |x| = {np.abs(p['gen_norm']).max():.2e}")
    check(set(COND_KEYS) <= set(p), f"payload has every CondSamples key (missing {set(COND_KEYS) - set(p)})")
    check(json.loads(p["cond_vars"]) == ["p_store", "full_t"] and p["cond_norm"].shape == (2, 2, 32, 32)
          and p["obs_avail"].shape == (2, 2, 32, 32) and p["baseline_phys"].shape == (2, 4, 32, 32),
          f"pointwise terms become cond_vars {json.loads(p['cond_vars'])}, baseline per target")
    bm = json.loads(p["baseline_map"])
    check(bm == {"ssh": None, "sst": "p_store", "sss": None, "tau": "full_t"}, f"baseline_map {bm}")

    # 2. Gaussian prior, guidance on: observed pixels of 'tau' (std 0.01, fully observed)
    #    are pulled onto y; the unobserved 'sss' pixels stay ~N(0,1) except via blur term.
    for sampler in ("vp", "edm"):
        # the edm path needs a finer schedule than 32 steps when a term has std 0.01
        p, a = run(_GaussPriorNet(0), guidance="on", sampler=sampler, gamma_scale=1.0, k=16,
                   **({"steps": 128} if sampler == "edm" else {}))
        gen = p["gen_norm"].astype(np.float32)
        ti = [t["name"] for t in json.loads(p["terms"])].index("full_t")
        yg = p["y_grid"][:, ti]
        m = p["obs_mask"][:, ti]
        err = np.abs(gen[:, :, 3].mean(1) - yg)[m]
        free = np.abs(gen[:, :, 3].mean(1))[m]
        check(err.mean() < 0.35 * free.mean(),
              f"[{sampler}] guided ensemble mean tracks the fully observed channel: "
              f"|mean-y|={err.mean():.3f} vs |mean|={free.mean():.3f}")
        check(np.isfinite(gen).all() and 0.2 < gen[:, :, 2].std() < 1.5,
              f"[{sampler}] finite; blur-observed channel keeps sub-blur spread (std {gen[:, :, 2].std():.2f})")
    p, a = run(_GaussPriorNet(0), guidance="on", sampler="edm", edm_heun=True, k=8)
    check(np.isfinite(p["gen_norm"]).all(), "edm heun path runs")
    p, a = run(_GaussPriorNet(0), guidance="on", corrections=1, k=8)
    check(np.isfinite(p["gen_norm"]).all(), "vp with Langevin corrections runs")
    p, a = run(_GaussPriorNet(0), guidance="on", terms="full_t,b_truth", k=8)
    check(json.loads(p["cond_vars"]) == ["full_t"] and len(json.loads(p["terms"])) == 2,
          "--terms subsets the observing system")
    p1, _ = run(_GaussPriorNet(0), guidance="on", day_shard="0/2", k=8)
    p2, _ = run(_GaussPriorNet(0), guidance="on", day_shard="1/2", k=8)
    check(np.abs(p1["gen_norm"][1]).max() == 0 and np.abs(p2["gen_norm"][0]).max() == 0
          and np.abs(p1["gen_norm"][0]).max() > 0, "day shards fill disjoint days")
    with tempfile.TemporaryDirectory() as tmp:
        f1, f2, fm = (os.path.join(tmp, n) for n in ("s1.npz", "s2.npz", "m.npz"))
        np.savez(f1, **p1)
        np.savez(f2, **p2)
        merge_shards([f1, f2], fm)
        mm = np.load(fm, allow_pickle=False)
        check(np.array_equal(mm["gen_norm"][0], p1["gen_norm"][0]) and np.array_equal(mm["gen_norm"][1], p2["gen_norm"][1])
              and json.loads(str(mm["meta"]))["sda"]["days_done"] == [0, 1], "merge_shards stitches the day axis")
        if with_diagnostics:
            out = os.path.join(tmp, "samples_sda_32.npz")
            np.savez(out, **p)
            r = subprocess.run([sys.executable, "-m", "eval.diagnostics_cond", "--samples", out,
                                "--out", os.path.join(tmp, "figs"), "--no-eke"],
                               capture_output=True, text=True)
            check(r.returncode == 0 and os.path.exists(os.path.join(tmp, "figs", "summary.csv")),
                  "eval.diagnostics_cond runs on the payload" + ("" if r.returncode == 0 else f"\n{r.stderr[-2000:]}"))
    print(f"[sda.assimilate] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
