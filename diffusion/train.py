"""DDP training for the conditional EDM diffusion model (and MSE baseline).

Modes:
  - ``diffusion``: EDM-preconditioned UNet trained with the Karras loss.
  - ``regression``: the same UNet backbone trained with masked MSE (no noise) —
    the deterministic MSE-UNet baseline. Shares all data/DDP plumbing.

Launch with torchrun for multi-GPU:
  torchrun --nproc_per_node=3 -m diffusion.train --config diffusion/configs/ssh.yaml
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import load_config
from .data import ZarrWindowDataset, worker_init_fn
from .dataset_spec import resolve_spec
from .edm import EDMLoss, EDMPrecond
from .ema import EMA
from .networks import SongUNet
from .networks_genda import GenDASongUNet
from .utils import CSVLogger, cleanup_ddp, is_main, seed_everything, setup_ddp


def save_ckpt(state, path):
    """Atomic checkpoint save: write to <path>.tmp then os.replace onto <path>.

    A crash mid-write (e.g. a transient full disk) then leaves the previous good
    checkpoint intact instead of truncating it to 0 bytes. os.replace is atomic
    within a filesystem, so a reader never sees a half-written ckpt.pt.
    """
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_net(cfg, mode: str):
    in_ch = cfg.cond_channels() + (cfg.target_channels() if mode == "diffusion" else 0)
    if cfg.arch == "genda":
        model = GenDASongUNet(
            in_channels=in_ch,
            out_channels=cfg.target_channels(),
            img_resolution=cfg.patch,
            model_channels=cfg.model_channels,
            channel_mult=tuple(cfg.channel_mult),
            num_blocks=cfg.num_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
            dropout=cfg.dropout,
        )
    elif cfg.arch == "nemo":
        model = SongUNet(
            in_channels=in_ch,
            out_channels=cfg.target_channels(),
            model_channels=cfg.model_channels,
            channel_mult=tuple(cfg.channel_mult),
            num_blocks=cfg.num_blocks,
            attn_levels=tuple(cfg.attn_levels),
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown arch {cfg.arch!r} (expected 'nemo' or 'genda')")
    if mode == "diffusion":
        return EDMPrecond(model, sigma_data=cfg.sigma_data)
    return model


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if cfg.lr_schedule == "constant":   # GenDA: linear warm-up, then flat
        return cfg.lr
    prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def masked_mse(pred, target, mask):
    se = (pred - target) ** 2
    return (se * mask).sum() / (mask.sum() * target.shape[1] + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    # No default: the old one named diffusion/configs/ssh.yaml, which this
    # standalone copy does not ship -- a bare `python -m diffusion.train`
    # died on a missing file rather than saying what was missing.
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default=None, choices=["diffusion", "regression"])
    ap.add_argument("--k-days", type=int, default=None)
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(
        args.config, mode=args.mode, k_days=args.k_days, patch=args.patch,
        batch=args.batch, steps=args.steps, max_days=args.max_days,
        num_workers=args.num_workers, out=args.out,
    )

    rank, world, local, device = setup_ddp()
    seed_everything(cfg.seed + rank)
    # One spec per process, shared by the train and val datasets, so the member
    # stores of a multi-store family are opened exactly once.
    spec = resolve_spec(cfg)
    if is_main():
        os.makedirs(cfg.out, exist_ok=True)
        print(f"[cfg] mode={cfg.mode} k={cfg.k_days} Cc={cfg.cond_channels()} "
              f"Ct={cfg.target_channels()} in_ch={cfg.in_channels()} world={world}")
        print(f"[data] dataset={spec.name} stores={len(spec.stores)} "
              f"base_cadence={spec.base_cadence} steps={spec.n_steps()}")

    train_ds = ZarrWindowDataset(cfg, split="train", spec=spec)
    if world > 1:
        # Only rank 0 writes the climatology cache (data.py). Sync here so the
        # file is fully in place before anything downstream reads it.
        dist.barrier()
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) \
        if world > 1 else None
    loader = DataLoader(
        train_ds, batch_size=cfg.batch, sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )

    # In-loop validation (GenDA has one; nemo's val_every was a dead field until
    # now). Built only when enabled so the val split is never touched otherwise.
    val_loader = None
    if cfg.val_every > 0 and cfg.val_batches > 0:
        # share_from reuses the train dataset's preloaded arrays -- they are
        # split-independent, so a second copy would just double the RAM.
        try:
            val_ds = ZarrWindowDataset(cfg, split="val", share_from=train_ds,
                                       spec=spec)
        except ValueError as e:
            # e.g. --max-days truncates the time axis before the val split
            # starts. Validation is a diagnostic; never let it abort training.
            if is_main():
                print(f"[warn] in-loop validation disabled: {e}")
        else:
            val_loader = DataLoader(
                val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0,
                pin_memory=True, drop_last=True,
            )

    net = build_net(cfg, cfg.mode).to(device)
    ddp_net = DDP(net, device_ids=[local]) if world > 1 else net
    ema = EMA(net, decay=cfg.ema_decay, halflife_kimg=cfg.ema_halflife_kimg)
    # AdamW with weight_decay=0 is Adam; betas/eps already match GenDA's.
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                            weight_decay=0.0)
    loss_fn = (EDMLoss(cfg.p_mean, cfg.p_std, cfg.sigma_data, cfg.loss_reduction)
               if cfg.mode == "diffusion" else None)
    global_batch = cfg.batch * world

    # resume
    start_step = 0
    ckpt_path = os.path.join(cfg.out, "ckpt.pt")
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        sd = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(sd["model"])
        ema.load_state_dict(sd["ema"])
        opt.load_state_dict(sd["opt"])
        start_step = sd["step"] + 1
        if is_main():
            print(f"[resume] from step {start_step}")
    elif os.path.exists(ckpt_path) and is_main():
        # 0-byte ckpt = a prior save was interrupted; start fresh rather than crash.
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

    def compute_loss(module, batch):
        cond = batch["cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if cfg.mode == "diffusion":
                return loss_fn(module, target, cond, mask)
            pred = module(cond, torch.zeros(cond.shape[0], device=device))
            return masked_mse(pred, target, mask)

    @torch.no_grad()
    def validate() -> float:
        """Mean val loss over ``cfg.val_batches``: eval mode, same autocast.

        GenDA validates in train mode (dropout active), in fp32 while training in
        fp16, and re-runs it on every step of a validating tick; done cleanly
        here instead. Runs on the unwrapped ``net`` -- no gradients, so there is
        nothing for DDP to reduce; each rank reports its own value and only
        rank 0's is logged.
        """
        net.eval()
        total, n = 0.0, 0
        for i, b in enumerate(val_loader):
            if i >= cfg.val_batches:
                break
            total += compute_loss(net, b).item()
            n += 1
        net.train()
        return total / max(n, 1)

    def log_row(step: int, lv: float, val: float | None = None) -> None:
        row = {"step": step, "loss": lv, "lr": lr_at(step, cfg)}
        if val_loader is not None:   # keep the column in the header from row 0
            row["val_loss"] = "" if val is None else val
        logger.log(row)

    net.train()
    it = batches()
    for step in range(start_step, cfg.steps):
        batch = next(it)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        loss = compute_loss(ddp_net, batch)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_nan_to_num:   # GenDA's guard in place of clipping
            for p in net.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5,
                                     out=p.grad)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(net, cur_nimg=(step + 1) * global_batch,
                   global_batch=global_batch)

        if is_main() and step % cfg.log_every == 0:
            lv = loss.item()
            print(f"step {step:>7d}  loss {lv:.4f}  lr {lr_at(step, cfg):.2e}")
            log_row(step, lv)

        if val_loader is not None and (step + 1) % cfg.val_every == 0:
            vl = validate()   # all ranks run it; only rank 0 logs
            if is_main():
                print(f"step {step:>7d}  val_loss {vl:.4f}")
                log_row(step, loss.item(), vl)

        if is_main() and (step + 1) % cfg.ckpt_every == 0:
            save_ckpt({"model": net.state_dict(), "ema": ema.state_dict(),
                       "opt": opt.state_dict(), "step": step, "config": cfg.as_dict()},
                      ckpt_path)

    if is_main():
        save_ckpt({"model": net.state_dict(), "ema": ema.state_dict(),
                   "opt": opt.state_dict(), "step": cfg.steps - 1,
                   "config": cfg.as_dict()}, ckpt_path)
        print(f"[done] saved {ckpt_path}")
    cleanup_ddp()


if __name__ == "__main__":
    main()
