"""Experiment configuration: a dataclass with a YAML loader and CLI overrides.

Input (``cond_vars``) and output (``target``) variable lists are both plain
config fields, so activating/deactivating any channel — or sweeping the number
of assimilated days ``k_days`` — is a one-line edit.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

# Variables whose native distribution is lognormal and must be log10-transformed
# before normalisation. Their stored linear meta stats are therefore unusable;
# log-space mean/std are computed and cached at dataset-build time (see data.py).
LOG10_VARS = frozenset({
    "CHL", "CHL_D", "CHL_N", "CHL_olci", "CHL_D_olci", "CHL_N_olci",
    # gapped variant from add_cloud_gaps.py -- still lognormal, so it must be
    # log10-transformed before normalising like every other CHL channel.
    # (log10_grad_CHL_gap is already in log space and stays out, as its
    # gap-free counterpart log10_grad_CHL does.)
    "CHL_olci_gap",
})

# Map a truth target variable to the degraded channel that conditions it (used
# by the bicubic baseline and cycle-consistency re-degradation).
TRUTH_TO_DEGRADED = {
    "SSH": "SSH_aviso",
    "SST": "SST_odyssea",
    "CHL": "CHL_olci",
}


@dataclass
class Config:
    # --- data ---------------------------------------------------------------
    # Dataset descriptor (see diffusion/datasets/*.yaml). ``None`` resolves to
    # ``dataset_spec.legacy_spec(zarr_path)`` -- the original single-store,
    # single-cadence layout -- so every pre-existing config and checkpoint keeps
    # behaving exactly as it did. Set this to point at another store family.
    dataset: str | None = None
    # Used when ``dataset`` is unset, and back-filled from a single-store
    # descriptor so the evaluation modules (which open the store directly) keep
    # working without changes.
    zarr_path: str = ("${HYCOM_STORE:-/unity/f1/ozavala/DATA/GOFFISH/GOES/datasets/gom_nemo_diffusive.zarr}")
    # Forwarded verbatim to fsspec when a store path is a URL (``s3://`` etc.);
    # ignored for plain POSIX paths. e.g. ``{region_name: us-east-1}``, or
    # ``{anon: true}`` for a public bucket. With an EC2 instance role, unset.
    storage_options: dict[str, Any] | None = None
    target: list[str] = field(default_factory=lambda: ["SSH"])
    cond_vars: list[str] = field(default_factory=lambda: [
        "SSH_aviso", "SST_odyssea", "CHL_olci",
        "log10_grad_SST", "log10_grad_CHL_D", "log10_grad_CHL_N", "log10_grad_CHL",
    ])
    use_ocean_mask: bool = True          # append static ocean_mask as a cond channel
    use_doy: bool = True                 # append day-of-year sin/cos of the target day
    # Length of the conditioning window, in steps of the dataset's *base cadence*
    # -- days for gom_nemo, hours for a store family whose base cadence is
    # hourly. The name is kept (rather than the more accurate ``k_steps``)
    # because every saved checkpoint carries this key in its config dict.
    k_days: int = 14                     # steps of observational history to assimilate
    patch: int = 128                     # training crop size (must divide by 2^(L-1))
    ocean_frac: float = 0.5              # min ocean fraction to keep a training patch
    crops_per_day: int = 8               # nominal crops per valid target day per epoch
    max_days: int | None = None          # subset the time axis (quick tests); None=all
    # Written at run time, so they must NOT default into the source tree: a
    # container image is read-only, and caches beside code are wrong even
    # where it is writable. ${CACHE_DIR:-diffusion} keeps the on-prem
    # location (and the existing cache files) when the var is unset.
    norm_cache: str = "${CACHE_DIR:-diffusion}/_norm_cache.json"
    # Crop sampling. "grid" = the precomputed ocean_frac-filtered patch//4 lattice
    # (default, what every existing checkpoint trained on); "random" = GenDA's
    # uniform continuous top-left corner. ``reject_land`` (random mode only)
    # redraws until the window is entirely ocean, GenDA's land handling.
    crop_mode: str = "grid"              # {"grid", "random"}
    reject_land: bool = False
    # Normalisation. "zscore" = scalar (mean, std) per variable, the default and
    # what every existing checkpoint used. "anomaly" = GenDA-style: subtract a
    # per-pixel time-mean field (or a per-pixel seasonal climatology for vars
    # listed in ``clim_vars``) fit on the train split, then divide by a scalar
    # per-variable anomaly std.
    norm_mode: str = "zscore"            # {"zscore", "anomaly"}
    clim_vars: list[str] = field(default_factory=list)
    clim_cache: str = "${CACHE_DIR:-diffusion}/_clim_cache.npz"
    val_gap_days: int = 0                # drop this many steps off the end of train
    # Reject conditioning windows that straddle a gap in the time axis. Off by
    # default, and the cost of turning it on is not small: the gom_nemo daily
    # axis has 15 real gaps (NoLeap calendar artifacts), and each one
    # invalidates the k-1 windows spanning it -- measured, 195 of 1859 windows
    # at k=14, 30 of 1870 at k=3, none at k=1. That reshapes the training
    # distribution, so existing runs would stop reproducing.
    require_contiguous_window: bool = False

    # --- optimisation -------------------------------------------------------
    mode: str = "diffusion"              # {"diffusion", "regression"}
    batch: int = 16                      # per-GPU batch size
    lr: float = 1e-4
    steps: int = 200_000
    warmup: int = 1_000
    lr_schedule: str = "cosine"          # {"cosine", "constant"} after warmup
    grad_clip: float = 1.0               # 0 = no clipping (GenDA)
    grad_nan_to_num: bool = False        # sanitise grads instead of clipping (GenDA)
    ema_decay: float = 0.9999
    # >0 switches EMA to GenDA's halflife-in-samples rule with a 0.05 ramp-up;
    # 0 keeps the constant ``ema_decay`` every existing checkpoint was trained with.
    ema_halflife_kimg: float = 0.0
    loss_reduction: str = "masked_mean"  # {"masked_mean", "sum"} (sum = GenDA)
    val_batches: int = 10                # batches averaged per in-loop validation
    num_workers: int = 4
    seed: int = 0

    # --- model (SongUNet) ---------------------------------------------------
    arch: str = "nemo"                   # {"nemo", "genda"} backbone
    model_channels: int = 128
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 2])
    num_blocks: int = 2
    attn_levels: list[int] = field(default_factory=lambda: [3])  # 0-indexed levels
    # arch="genda" only: attention is gated on the *absolute* resolution
    # (img_resolution >> level) rather than the level index, so patch size and
    # attention placement stay coupled the way GenDA trained them.
    attn_resolutions: list[int] = field(default_factory=lambda: [16])
    dropout: float = 0.0

    # --- EDM ----------------------------------------------------------------
    sigma_data: float = 1.0              # data normalised to ~unit variance
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sampler_steps: int = 18
    s_churn: float = 0.0                 # >0 = stochastic sampler (EDM Alg. 2)
    s_noise: float = 1.0
    s_tmin: float = 0.0                  # churn only injected for s_tmin <= sigma <= s_tmax
    s_tmax: float = float("inf")         # (EDM Alg. 2 S_tmin/S_tmax gate; defaults = all sigma)

    # --- sampling / eval ----------------------------------------------------
    k_members: int = 16                  # ensemble members per target day
    n_eval_days: int = 40                # held-out target days used for evaluation

    # --- assimilation (GenDA-style guided posterior sampling) ---------------
    assim_config: str | None = None      # path to an assim_targets YAML spec
    corrector_steps: int = 0             # SDA-style Langevin MC steps per reverse step (0 = off)
    corrector_step_size: float = 0.01    # LMC step size (delta in SDA eq. 17)

    # --- io / logging -------------------------------------------------------
    out: str = "runs/ssh"
    log_every: int = 100
    ckpt_every: int = 5_000
    sample_every: int = 5_000
    val_every: int = 0                   # 0 = no in-loop validation (default)

    # ---------------------------------------------------------------------
    def n_extra(self) -> int:
        return (1 if self.use_ocean_mask else 0) + (2 if self.use_doy else 0)

    def cond_channels(self) -> int:
        """Cc = Cobs * k + n_extra (channel-stacked window + static/time extras)."""
        return len(self.cond_vars) * self.k_days + self.n_extra()

    def target_channels(self) -> int:
        return len(self.target)

    def in_channels(self) -> int:
        """UNet input channels: cond (+ noisy target for diffusion)."""
        c = self.cond_channels()
        if self.mode == "diffusion":
            c += self.target_channels()
        return c

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_ALIASES = {"k_days": "k_days", "k-days": "k_days"}


# ---------------------------------------------------------------------------
# Environment-variable expansion
# ---------------------------------------------------------------------------
# Every path in a shipped config is written as ``${VAR:-<the on-prem path>}``,
# so one YAML runs unchanged on the COAPS cluster (no env vars set -> the
# fallback, i.e. exactly the path that used to be hardcoded) and inside a
# container on AWS (export the var -> an s3:// URL or a bind-mounted path).
# Ported from the sibling nemo_anfo project, which uses the same convention.
_VAR_DEFAULT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _expand_str(s: str) -> str:
    # Resolve the ``${VAR:-default}`` form first, then plain ``$VAR`` /
    # ``${VAR}``. ``os.environ.get(...) or default`` (rather than a plain get)
    # means an env var set to the empty string also falls back -- an empty
    # HYCOM_STORE is a mistake, never a request to read the current directory.
    s = _VAR_DEFAULT.sub(lambda m: os.environ.get(m.group(1)) or m.group(2), s)
    return os.path.expandvars(s)


def expandvars(obj):
    """Recursively expand env vars in every string of a loaded config tree.

    Supports ``$VAR``, ``${VAR}`` and ``${VAR:-default}``. Used on both
    experiment configs (here) and dataset descriptors (``dataset_spec``).
    """
    if isinstance(obj, str):
        return _expand_str(obj)
    if isinstance(obj, list):
        return [expandvars(v) for v in obj]
    if isinstance(obj, dict):
        return {k: expandvars(v) for k, v in obj.items()}
    return obj


def load_config(path: str | None = None, **overrides) -> Config:
    """Build a Config from an optional YAML file plus keyword overrides.

    Only keys present in ``Config`` are accepted; ``None`` overrides are ignored
    so CLI flags that default to ``None`` don't clobber YAML values. Every
    string value is env-var-expanded (see :func:`expandvars`), which is what
    keeps one config portable between the cluster and a container.
    """
    data: dict[str, Any] = {}
    if path:
        with open(path) as f:
            data.update(yaml.safe_load(f) or {})
    for k, v in overrides.items():
        if v is None:
            continue
        data[_ALIASES.get(k, k)] = v

    # After the overrides, so a `--out '${OUT_DIR:-runs}/x'` on the command line
    # expands the same way a YAML value does.
    data = expandvars(data)

    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = Config(**data)

    # Dataclass defaults never went through the loop above, so expand the ones
    # the config did not override. Without this a ${VAR:-default} default (e.g.
    # norm_cache) would reach the filesystem verbatim, as a literal directory
    # named "${CACHE_DIR:-diffusion}".
    for f in dataclasses.fields(cfg):
        if f.name in data:
            continue
        v = getattr(cfg, f.name)
        if isinstance(v, str):
            setattr(cfg, f.name, _expand_str(v))
    return cfg
