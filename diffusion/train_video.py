"""DDP training for the multi-frame EDM prior (``arch: genllc_video``).

The video counterpart of :mod:`diffusion.train`, sharing its checkpoint layout,
learning-rate schedule, EMA rule and atomic saves. What differs:

  * the sample is a window of frames and the network gets calendar inputs and
    optional scalar labels (:mod:`diffusion.data_video`, :mod:`diffusion.edm_video`);
  * sigma can be drawn log-uniform on ``[train_sigma_min, train_sigma_max]`` --
    one such range per expert, each expert being an ordinary run of this script
    with its own config and output directory;
  * ``accum`` micro-batches are accumulated per optimiser step, so a target
    global batch can be reached on few GPUs;
  * validation uses the same windows, the same noise and an evenly spaced grid
    of sigma quantiles every round, so its curve is comparable across rounds.

Launch:
  torchrun --standalone --nproc_per_node=N -m diffusion.train_video \
      --config diffusion/configs/video_gulfstream_small.yaml

Fine-tune from another run (e.g. at a larger patch):
  ... -m diffusion.train_video --config <new>.yaml --init-from <run>/ckpt.pt
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import Config, load_config
from .data import worker_init_fn
from .data_video import ZarrVideoDataset, batch_ctx
from .dataset_spec import resolve_spec
from .edm_video import VideoEDMLoss, VideoPrecond
from .ema import EMA
from .networks_video import attention_levels, build_video_unet, video_in_channels
from .train import lr_at, save_ckpt
from .utils import CSVLogger, cleanup_ddp, is_main, seed_everything, setup_ddp


def build_video_precond(cfg: Config) -> VideoPrecond:
    if cfg.arch != "genllc_video":
        raise ValueError(
            f"train_video needs arch: genllc_video, got {cfg.arch!r} "
            "(single-frame models train with diffusion.train)")
    return VideoPrecond(
        build_video_unet(cfg), sigma_data=cfg.sigma_data,
        calendar_harmonics=cfg.calendar_harmonics if cfg.use_calendar else 0)


def build_loss(cfg: Config) -> VideoEDMLoss:
    return VideoEDMLoss(cfg.sigma_sampling, cfg.p_mean, cfg.p_std,
                        cfg.train_sigma_min, cfg.train_sigma_max, cfg.sigma_data,
                        cfg.loss_reduction)


# ---------------------------------------------------------------------------
# --init-from: is a checkpoint weight-compatible with this config?
# ---------------------------------------------------------------------------
_SAME_KEYS = ("target", "cond_vars", "label_vars", "use_ocean_mask", "use_calendar",
              "calendar_harmonics", "n_frames", "model_channels", "channel_mult",
              "num_blocks", "channels_per_head")


def check_init_compat(cfg: Config, src: dict) -> list[str]:
    """Problems that stop ``src`` (a checkpoint's config dict) seeding ``cfg``.

    Weights are stored by level, so the patch may differ; what must match is
    the channel layout, the widths, the frame count and WHICH LEVELS attend.
    Returns human-readable problems (empty = compatible); an attention mismatch
    names the resolution lists that would keep the checkpoint's levels.
    """
    new = cfg.as_dict()
    defaults = Config().as_dict()
    get = lambda k: src.get(k, defaults[k])
    problems = [f"{k}: checkpoint has {get(k)!r}, config has {new[k]!r}"
                for k in _SAME_KEYS if get(k) != new[k]]
    n_levels = len(get("channel_mult"))
    lv_src = attention_levels(get("patch"), n_levels, get("attn_resolutions"),
                              get("temporal_attn_resolutions"))
    lv_new = attention_levels(cfg.patch, len(cfg.channel_mult), cfg.attn_resolutions,
                              cfg.temporal_attn_resolutions)
    if lv_src != lv_new and len(cfg.channel_mult) == n_levels:
        want_sp = [cfg.patch >> lv for lv in lv_src["spatial"]]
        want_tp = [cfg.patch >> lv for lv in lv_src["temporal"]]
        problems.append(
            f"attention levels differ: checkpoint (patch {get('patch')}) attends at "
            f"levels {lv_src}, this config (patch {cfg.patch}) at {lv_new}. To keep "
            f"the checkpoint's levels at patch {cfg.patch} set "
            f"attn_resolutions: {want_sp} and temporal_attn_resolutions: {want_tp}")
    return problems


def _parse_set(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override any config field (YAML value); repeatable")
    ap.add_argument("--init-from", default=None,
                    help="seed raw and EMA weights from this checkpoint's EMA "
                         "(ignored when <out>/ckpt.pt exists: resume wins)")
    ap.add_argument("--keep-opt", action="store_true",
                    help="with --init-from: also keep the optimiser state and "
                         "the step counter")
    args = ap.parse_args()

    overrides = dict(batch=args.batch, accum=args.accum, steps=args.steps,
                     max_days=args.max_days, num_workers=args.num_workers, out=args.out)
    overrides.update(_parse_set(args.set))
    cfg = load_config(args.config, **overrides)
    if cfg.mode != "diffusion":
        raise SystemExit("train_video only trains mode: diffusion")
    if cfg.accum < 1:
        raise SystemExit("accum must be >= 1")

    # Decide resume-vs-seed, and reject an incompatible seed, BEFORE the store
    # preload: that takes minutes and a config mismatch needs none of it.
    ckpt_path = os.path.join(cfg.out, "ckpt.pt")
    resuming = os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0
    init_sd = None
    if args.init_from and not resuming:
        init_sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
        problems = check_init_compat(cfg, init_sd["config"])
        if problems:
            raise SystemExit("[init] checkpoint is not weight-compatible with this "
                             "config:\n  - " + "\n  - ".join(problems))

    rank, world, local, device = setup_ddp()
    seed_everything(cfg.seed + rank)
    spec = resolve_spec(cfg)
    net = build_video_precond(cfg).to(device)
    global_batch = cfg.batch * world * cfg.accum
    if is_main():
        os.makedirs(cfg.out, exist_ok=True)
        os.makedirs(os.path.join(cfg.out, "checkpoints"), exist_ok=True)
        n_par = sum(p.numel() for p in net.parameters())
        print(f"[cfg] arch={cfg.arch} expert={cfg.expert or '-'} T={cfg.n_frames} "
              f"stride={cfg.frame_stride} Ct={len(cfg.target)} "
              f"in_ch={video_in_channels(cfg)} labels={len(cfg.label_vars)} "
              f"params={n_par:,}")
        print(f"[cfg] attention levels {net.model.attn_levels} at patch {cfg.patch}; "
              f"sigma {cfg.sigma_sampling}"
              + (f" [{cfg.train_sigma_min}, {cfg.train_sigma_max}]"
                 if cfg.sigma_sampling == "loguniform"
                 else f" (p_mean {cfg.p_mean}, p_std {cfg.p_std})"))
        print(f"[cfg] batch {cfg.batch}/GPU x {world} rank(s) x accum {cfg.accum} "
              f"= global batch {global_batch}; {cfg.steps} optimiser steps "
              f"= {cfg.steps * global_batch:,} samples")
        print(f"[data] dataset={spec.name} stores={len(spec.stores)} "
              f"base_cadence={spec.base_cadence} steps={spec.n_steps()}")

    train_ds = ZarrVideoDataset(cfg, split="train", spec=spec)
    if world > 1:
        dist.barrier()          # rank 0 has finished writing the climatology cache
    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
               if world > 1 else None)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )

    val_ds, val_loader = None, None
    if cfg.val_every > 0 and cfg.val_batches > 0:
        try:
            val_ds = ZarrVideoDataset(cfg, split="val", share_from=train_ds, spec=spec)
        except ValueError as e:
            if is_main():
                print(f"[warn] in-loop validation disabled: {e}")
        else:
            val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                                    num_workers=0, pin_memory=True, drop_last=True)

    ddp_net = DDP(net, device_ids=[local]) if world > 1 else net
    ema = EMA(net, decay=cfg.ema_decay, halflife_kimg=cfg.ema_halflife_kimg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                            weight_decay=0.0)
    loss_fn = build_loss(cfg)

    # resume / seed
    start_step = 0
    ckpt_dir = os.path.join(cfg.out, "checkpoints")
    best_path = os.path.join(cfg.out, "best.pt")
    best_val = float("inf")
    if resuming:
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(sd["model"])
        ema.load_state_dict(sd["ema"])
        opt.load_state_dict(sd["opt"])
        start_step = sd["step"] + 1
        best_val = sd.get("best_val", float("inf"))
        if is_main():
            print(f"[resume] from step {start_step}"
                  + (" (--init-from ignored)" if args.init_from else ""))
    elif init_sd is not None:
        sd = init_sd
        net.load_state_dict(sd["ema"])
        ema.load_state_dict(sd["ema"])
        if args.keep_opt:
            opt.load_state_dict(sd["opt"])
            start_step = sd["step"] + 1
        if is_main():
            print(f"[init] raw and EMA weights from the EMA of {args.init_from} "
                  f"(its step {sd['step'] + 1}, patch {sd['config'].get('patch')}); "
                  f"starting at step {start_step}")
    elif os.path.exists(ckpt_path) and is_main():
        print(f"[warn] {ckpt_path} is empty; starting from step 0")

    logger = CSVLogger(os.path.join(cfg.out, "log.csv")) if is_main() else None

    def batches():
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for b in loader:
                yield b
            epoch += 1

    def compute_loss(module, batch, sigma=None, generator=None):
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ctx = batch_ctx(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return loss_fn(module, target, ctx, mask, sigma=sigma, generator=generator)

    @torch.no_grad()
    def validate() -> float:
        """Mean val loss over ``cfg.val_batches``, made repeatable across rounds.

        The windows (dataset RNG), the noise (generator) and the sigmas (an
        evenly spaced grid of quantiles of the training distribution) are the
        same every round, so the curve moves only because the weights did.
        """
        net.eval()
        val_ds.set_epoch_rng(cfg.seed + 7919)
        gen = torch.Generator(device=device).manual_seed(cfg.seed + 104729)
        sig = loss_fn.stratified_sigma(cfg.batch, device)
        total, n = 0.0, 0
        for i, b in enumerate(val_loader):
            if i >= cfg.val_batches:
                break
            total += compute_loss(net, b, sigma=sig, generator=gen).item()
            n += 1
        net.train()
        return total / max(n, 1)

    def log_row(step: int, lv: float, val: float | None = None) -> None:
        row = {"step": step, "loss": lv, "lr": lr_at(step, cfg)}
        if val_loader is not None:
            row["val_loss"] = "" if val is None else val
        logger.log(row)

    def ckpt_state(step: int, val: float | None = None) -> dict:
        st = {"model": net.state_dict(), "ema": ema.state_dict(),
              "opt": opt.state_dict(), "step": step, "config": cfg.as_dict(),
              "best_val": best_val}
        if val is not None:
            st["val_loss"] = val
        return st

    net.train()
    it = batches()
    t_log, s_log = time.time(), start_step
    for step in range(start_step, cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        opt.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(cfg.accum):
            loss = compute_loss(ddp_net, next(it)) / cfg.accum
            loss.backward()
            loss_sum += loss.detach()
        if cfg.grad_nan_to_num:
            for p in net.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(net, cur_nimg=(step + 1) * global_batch, global_batch=global_batch)

        if is_main() and step % cfg.log_every == 0:
            lv = loss_sum.item()             # also syncs the GPU, so the timing is real
            now = time.time()
            rate = (now - t_log) / max(step + 1 - s_log, 1)
            t_log, s_log = now, step + 1
            mem = (f"  {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GB"
                   if device.type == "cuda" else "")
            print(f"step {step:>7d}  loss {lv:.4f}  lr {lr_at(step, cfg):.2e}"
                  f"  {rate:.2f} s/step{mem}")
            log_row(step, lv)

        if val_loader is not None and (step + 1) % cfg.val_every == 0:
            vl = validate()
            if is_main():
                print(f"step {step:>7d}  val_loss {vl:.4f}")
                log_row(step, loss_sum.item(), vl)
                improved = vl < best_val
                if improved:
                    best_val = vl
                state = ckpt_state(step, val=vl)
                save_ckpt(state, os.path.join(ckpt_dir, f"ckpt_step{step + 1:07d}.pt"))
                if improved:
                    save_ckpt(state, best_path)

        if is_main() and (step + 1) % cfg.ckpt_every == 0:
            save_ckpt(ckpt_state(step), ckpt_path)

    if is_main():
        save_ckpt(ckpt_state(cfg.steps - 1), ckpt_path)
        print(f"[done] saved {ckpt_path}")
    cleanup_ddp()


def _selftest() -> None:
    base = dict(arch="genllc_video", target=["a", "b"], n_frames=5, patch=256,
                channel_mult=[1, 2, 2, 3, 4], attn_resolutions=[32, 16],
                temporal_attn_resolutions=[64, 32, 16])
    src = Config(**base).as_dict()
    assert check_init_compat(Config(**base), src) == []
    # a 512 fine-tune with the lists left alone is refused, and told what to set
    p = check_init_compat(Config(**{**base, "patch": 512}), src)
    assert len(p) == 1 and "[64, 32]" in p[0] and "[128, 64, 32]" in p[0], p
    ok512 = Config(**{**base, "patch": 512, "attn_resolutions": [64, 32],
                      "temporal_attn_resolutions": [128, 64, 32]})
    assert check_init_compat(ok512, src) == []
    # frame count and channel changes are named
    p = check_init_compat(Config(**{**base, "n_frames": 3, "target": ["a"]}), src)
    assert len(p) == 2 and any("n_frames" in s for s in p) and any("target" in s for s in p)
    assert _parse_set(["expert=large", "train_sigma_min=0.1", "target=[x, y]"]) == {
        "expert": "large", "train_sigma_min": 0.1, "target": ["x", "y"]}
    print("[ok] --init-from compatibility check and --set parsing")
    print("train_video selftest passed")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
