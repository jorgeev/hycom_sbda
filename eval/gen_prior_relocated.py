"""``eval.gen_prior`` for a checkpoint trained at another site.

A checkpoint records its ``Config`` with every ``${VAR:-default}`` already
expanded, so one trained inside the AWS image carries
``clim_cache=/mnt/runs/cache/_clim_cache.npz`` (and ``norm_cache`` beside it).
On-prem that directory does not exist and ``/mnt`` is not writable: the
climatology fit would run to completion and then die in ``_save_clim_cache``
on ``os.makedirs``, after the full store load. Everything else in the config
is site-independent (``dataset:`` names a descriptor, not a path, and the
descriptor expands ``GULFSTREAM_MANIFEST`` itself).

This wrapper swaps those two cache paths for ``${CACHE_DIR:-diffusion}`` --
the same default the on-prem config would have resolved to -- and otherwise
runs ``eval.gen_prior.main`` untouched, with the same CLI. The cache keys are
per dataset and variable (``gulfstream::ssh|mean|...``), never per config, so
the on-prem cache is valid for any gulfstream checkpoint; a variable it lacks
(``ug``/``vg`` for prior_gulfstream_geo) is fit and appended.

Usage: exactly ``python -m eval.gen_prior ...`` with ``gen_prior_relocated``
in its place. Set ``CACHE_DIR`` to choose the cache location.
"""
from __future__ import annotations

import dataclasses
import os

import eval.gen_prior as gen_prior
from diffusion.sample import config_from_ckpt as _config_from_ckpt


def config_from_ckpt_relocated(saved: dict):
    cfg = _config_from_ckpt(saved)
    cache_dir = os.environ.get("CACHE_DIR", "diffusion")
    new = dict(clim_cache=os.path.join(cache_dir, "_clim_cache.npz"),
               norm_cache=os.path.join(cache_dir, "_norm_cache.json"))
    for k, v in new.items():
        old = getattr(cfg, k)
        if old != v:
            print(f"[relocate] {k}: {old} -> {v}", flush=True)
    return dataclasses.replace(cfg, **new)


gen_prior.config_from_ckpt = config_from_ckpt_relocated

if __name__ == "__main__":
    gen_prior.main()
