"""Seed a fine-tune at a new patch size from a checkpoint trained at another.

``GenDASongUNet`` names its blocks by the resolution of their level
(``enc.128x128_block0``, ``dec.32x32_in0``...), computed from ``img_resolution``
= the training patch. The weights themselves are patch-independent (every
block is convolutional or attention over whatever grid it is given), but the
*names* are not, and ``train.py`` loads strictly, so a checkpoint from a
256-patch run cannot be resumed under a 512-patch config as it is. This tool
rewrites the names by the patch ratio and writes a resume-ready ``ckpt.pt``
into the new config's ``out`` directory. It touches no trainer code.

What the seed contains:
  model, ema  : BOTH the source EMA weights (renamed). The raw weights of a
                constant-lr run are one noisy point of the oscillation the EMA
                averages over; the fine-tune should start from the average.
  opt         : the source Adam state, unchanged (parameter order is the same
                at every patch size, and the moments are a better start than
                zeros at a converged point).
  step        : the source step, unchanged, so EMA's half-life rule
                (``cur_nimg * rampup_ratio``) does not restart from a one-image
                half-life and overwrite the loaded EMA with the raw weights.
                The new config's ``warmup``/``steps`` must therefore be set
                relative to this step (see the ft512 configs).
  config      : the NEW config, so ``eval.gen_prior`` builds the right net.
  best_val    : dropped, so best.pt tracking restarts under the new patch
                (val losses at a different patch are not comparable).

The strict ``load_state_dict`` at the end is the proof that the new config's
``attn_resolutions`` selects exactly the attention blocks the source has
(a 128-patch model's 16x16 attention becomes 64x64 at 512, so that config
must say ``attn_resolutions: [64]``).

Usage:
  python -m diffusion.remap_patch_ckpt --src <ckpt.pt> --config <new.yaml>
      [--log <log.csv to copy>] [--force]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

import torch

from .config import load_config
from .train import build_net

_RES = re.compile(r"(\d+)x(\d+)")


def remap_key(key: str, factor: int) -> str:
    def sub(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if a != b:
            raise ValueError(f"non-square level name in {key!r}")
        return f"{a * factor}x{b * factor}"
    return _RES.sub(sub, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source checkpoint (any patch)")
    ap.add_argument("--config", required=True, help="new config (sets patch and out)")
    ap.add_argument("--log", default=None, help="log.csv to copy beside the seed")
    ap.add_argument("--force", action="store_true", help="overwrite an existing seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sd = torch.load(args.src, map_location="cpu", weights_only=False)
    old_patch = int(sd["config"]["patch"])
    new_patch = int(cfg.patch)
    if new_patch % old_patch:
        raise SystemExit(f"new patch {new_patch} is not a multiple of source patch {old_patch}")
    factor = new_patch // old_patch
    print(f"[remap] {args.src}: step {sd['step']}, patch {old_patch} -> {new_patch} (x{factor})")

    ema = {remap_key(k, factor): v.clone() for k, v in sd["ema"].items()}
    changed = sum(1 for k in sd["ema"] if remap_key(k, factor) != k)
    print(f"[remap] {changed}/{len(ema)} tensor names rewritten")
    attn = sorted({k.rsplit(".qkv", 1)[0] for k in ema if ".qkv." in k})
    print(f"[remap] attention blocks in the seed: {attn}")

    net = build_net(cfg, "diffusion")
    net.load_state_dict(ema, strict=True)          # raises on any mismatch
    print(f"[remap] strict load into build_net({os.path.basename(args.config)}) OK")

    out_dir = cfg.out
    dst = os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(dst) and not args.force:
        raise SystemExit(f"{dst} exists; pass --force to overwrite")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    seed = {
        "model": {k: v.clone() for k, v in ema.items()},
        "ema": ema,
        "opt": sd["opt"],
        "step": int(sd["step"]),
        "config": cfg.as_dict(),
    }
    torch.save(seed, dst)
    print(f"[remap] wrote {dst} (resume step {seed['step'] + 1}; "
          f"schedule: warmup {cfg.warmup}, steps {cfg.steps}, lr {cfg.lr}, {cfg.lr_schedule})")
    if seed["step"] + 1 >= cfg.steps:
        raise SystemExit("config.steps is not beyond the resume step: nothing would train")

    log = args.log
    if log is None:
        for cand in (os.path.join(os.path.dirname(args.src), "log.csv"),
                     os.path.join(os.path.dirname(os.path.dirname(args.src)), "log.csv")):
            if os.path.exists(cand):
                log = cand
                break
    if log and os.path.exists(log):
        shutil.copyfile(log, os.path.join(out_dir, "log.csv"))
        print(f"[remap] copied {log}")


if __name__ == "__main__":
    main()
