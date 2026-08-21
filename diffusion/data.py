"""Windowed, config-driven dataset for the assimilation-style diffusion model.

For a target step ``t`` the model conditions on the previous ``k`` steps
(``t-k+1 … t``) of the degraded satellite-like channels and reconstructs the
truth field(s) at step ``t``. Conditioning is channel-stacked (time-ordered
oldest→newest) via :func:`build_condition`, which is deliberately isolated so a
temporal encoder can replace the flatten step later.

Nothing here knows a store's layout or its variable names: a
:class:`~diffusion.dataset_spec.DatasetSpec` supplies both. ``t`` indexes the
dataset's *base cadence* (days for gom_nemo, hours for gulfstream), and a
variable living on a coarser cadence is looked up through the spec's
timestamp-based index map, so one sample can mix hourly truth with daily and
weekly observations.

Normalisation uses the store's own ``mean``/``std`` for non-log fields; lognormal
fields are ``log10``-transformed first and their log-space stats are computed
once and cached (the stored linear stats are wrong for log space). Land pixels
are set to 0 after normalisation and the loss is masked to ocean pixels.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .dataset_spec import DatasetSpec, resolve_spec


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
def _log10_positive(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(np.where(a > 0, a, np.nan))


def compute_norm_stats(
    spec: DatasetSpec,
    variables: list[str],
    ocean: np.ndarray,
    cache_path: str,
    *,
    n_sample: int = 1200,
    seed: int = 42,
) -> dict[str, tuple[float, float, bool]]:
    """Return ``{var: (mean, std, is_log)}``.

    Non-log vars take the stats the store (or its manifest) already carries. Log
    vars compute ocean-masked log10-space mean/std from an ``n_sample``-step
    subset, cached to ``cache_path`` under a dataset-namespaced key so two
    datasets that happen to share a channel name cannot poison each other.
    """
    cache: dict[str, list] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    rng = np.random.default_rng(seed)
    fam = spec.family
    n_total = fam.n_steps(spec.cadence_of(variables[0]))
    # NOTE(val-leakage): sampling spans the whole time axis, including the
    # chronological validation split, so the cached log-space stats carry mild
    # train+val leakage (METRICS_REVIEW #3). Deliberately left unchanged: the
    # trained checkpoints normalise with exactly these stats, so restricting to
    # splits.json["train"] would invalidate them. Fix only alongside a retrain.
    idx = np.sort(rng.choice(n_total, size=min(n_sample, n_total), replace=False))
    ocean_b = ocean.astype(bool)

    stats: dict[str, tuple[float, float, bool]] = {}
    dirty = False
    for v in variables:
        is_log = spec.is_log(v)
        key = spec.cache_key(v)
        if not is_log:
            stored = spec.stored_stats(v)
            if stored is None:
                raise KeyError(
                    f"{spec.name}: no stored mean/std for {v!r} "
                    f"(stats_source={spec.stats_source!r}). Add it to the store's "
                    "meta group / manifest, or mark the variable log10."
                )
            m, sd = stored
            stats[v] = (float(m), max(float(sd), 1e-8), False)
            continue
        if key in cache:
            m, sd = cache[key]
            stats[v] = (float(m), float(sd), True)
            continue
        cad = spec.cadence_of(v)
        if cad.name == spec.base_cadence:
            vidx = idx
        else:
            mapped = spec.index_map(cad.name)[np.clip(idx, 0, spec.n_steps() - 1)]
            vidx = np.unique(mapped[mapped >= 0])
        arr = np.asarray(fam.take(cad, v, vidx), dtype=np.float64)
        arr = _log10_positive(arr)
        arr = np.where(ocean_b[None], arr, np.nan)
        if np.isfinite(arr).any():
            m = float(np.nanmean(arr))
            sd = max(float(np.nanstd(arr)), 1e-8)
        else:
            m, sd = 0.0, 1.0
        stats[v] = (m, sd, True)
        cache[key] = [m, sd]
        dirty = True

    if dirty and cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, cache_path)
    return stats


# ---------------------------------------------------------------------------
# Anomaly normalisation (GenDA-style)
# ---------------------------------------------------------------------------
# GenDA removes seasonality from the *data* rather than handing it to the network
# as day-of-year channels: thermodynamic variables lose a per-pixel monthly
# climatology, everything else a per-pixel time-mean, and each is then divided by
# a single scalar per-variable anomaly std. We fit an annual + semiannual
# harmonic instead of raw monthly means because this dataset has only ~4 training
# years -- 12 monthly means from 4 samples each would alias mesoscale eddies into
# the "climatology" and get subtracted out of the very signal we are modelling.
_HARMONICS = 2   # annual + semiannual
_YEAR_S = 365.2425 * 86400.0


def _doy_angle(t_unix: np.ndarray) -> np.ndarray:
    """Day-of-year phase in radians from unix seconds."""
    return 2.0 * np.pi * (np.mod(t_unix, _YEAR_S) / _YEAR_S)


def _harmonic_design(ang: np.ndarray) -> np.ndarray:
    """(T, 1 + 2*_HARMONICS) design matrix: constant + sin/cos of each harmonic."""
    cols = [np.ones_like(ang)]
    for h in range(1, _HARMONICS + 1):
        cols.append(np.sin(h * ang))
        cols.append(np.cos(h * ang))
    return np.stack(cols, axis=1)


def fit_climatology(
    arr: np.ndarray, ang: np.ndarray, ocean: np.ndarray, seasonal: bool,
    *, days: np.ndarray | None = None, chunk: int = 8192,
) -> np.ndarray:
    """Per-pixel climatology coefficients from ``arr`` (T, NY, NX).

    ``days`` restricts the fit to those time indices (use the train split).
    Returns ``(1 + 2*_HARMONICS, NY, NX)`` when ``seasonal`` (least-squares
    harmonic fit), else ``(1, NY, NX)`` holding just the per-pixel time-mean.
    Land pixels and non-finite samples are excluded; land coefficients are 0.

    Pixels are independent and share the design matrix, so this is a batch of
    tiny 5x5 normal-equation solves -- but forming the masked design tensor for
    all ~245k pixels at once would need tens of GB, hence the pixel chunking.
    """
    _, ny, nx = arr.shape
    ocean_b = np.asarray(ocean).astype(bool)
    flat = arr.reshape(arr.shape[0], -1)
    if days is not None:
        flat = flat[days]
        ang = ang[days]
    npix = flat.shape[1]

    d = _harmonic_design(ang) if seasonal else np.ones((flat.shape[0], 1))
    p = d.shape[1]
    dd = (d[:, :, None] * d[:, None, :]).reshape(-1, p * p)   # (T, P*P)
    eye = 1e-8 * np.eye(p)                                    # keeps land solvable
    coef = np.zeros((p, npix))

    for lo in range(0, npix, chunk):
        hi = min(lo + chunk, npix)
        y = flat[:, lo:hi].astype(np.float64)                 # (T, C)
        g = np.isfinite(y)
        np.copyto(y, 0.0, where=~g)
        gram = (g.astype(np.float64).T @ dd).reshape(-1, p, p) + eye
        rhs = y.T @ d                                         # (C, P)
        coef[:, lo:hi] = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0].T

    coef = coef.reshape(p, ny, nx)
    coef[:, ~ocean_b] = 0.0
    return coef.astype(np.float32)


def evaluate_climatology(coef: np.ndarray, ang: float | np.ndarray) -> np.ndarray:
    """Climatology field(s) at phase ``ang`` from :func:`fit_climatology` coefs.

    A scalar ``ang`` returns one ``(NY, NX)`` field; an array returns
    ``(len(ang), NY, NX)``. ``coef`` with a leading dim of 1 is a static
    time-mean, so the phase is ignored.
    """
    scalar = np.ndim(ang) == 0
    ang = np.atleast_1d(np.asarray(ang, dtype=np.float64))
    d = _harmonic_design(ang)[:, : coef.shape[0]]      # (T, P), P=1 for time-mean
    out = np.tensordot(d, coef, axes=(1, 0)).astype(np.float32)   # (T, NY, NX)
    return out[0] if scalar else out


def subtract_climatology_(arr: np.ndarray, coef: np.ndarray, ang: np.ndarray,
                          *, chunk: int = 128) -> None:
    """In-place ``arr -= climatology``, chunked over time.

    Materialising the full (T, NY, NX) climatology would cost a couple of GB per
    variable for no reason -- the coefficients are tiny, so re-evaluating them a
    chunk of days at a time is both cheaper and simpler.
    """
    for lo in range(0, arr.shape[0], chunk):
        hi = min(lo + chunk, arr.shape[0])
        arr[lo:hi] -= evaluate_climatology(coef, ang[lo:hi])


def _clim_key(var: str, seasonal: bool, t_end: int, n_train: int) -> str:
    """Cache key: the coefficients depend on the variable and the fit window."""
    return f"{var}|{'seas' if seasonal else 'mean'}|{t_end}|{n_train}"


def _load_clim_cache(path: str) -> dict[str, tuple[np.ndarray, float]]:
    if not path or not os.path.exists(path):
        return {}
    with np.load(path) as z:   # keys are plain unicode so allow_pickle stays off
        keys = [str(k) for k in z["__keys__"]]
        return {k: (z[f"coef::{i}"], float(z[f"std::{i}"])) for i, k in enumerate(keys)}


def _save_clim_cache(path: str, cache: dict[str, tuple[np.ndarray, float]]) -> None:
    """Write the coefficient cache atomically.

    A plain ``np.savez`` here was a latent race: every DDP rank runs the
    climatology fit, so several processes could write this file at once and a
    reader could see a truncated archive. Writing to a temp file and renaming
    makes the swap atomic, mirroring ``train.save_ckpt``; ``train.py``
    additionally confines the write to rank 0.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(cache)
    payload = {"__keys__": np.array(keys, dtype=np.str_)}
    for i, k in enumerate(keys):
        payload[f"coef::{i}"] = cache[k][0]
        payload[f"std::{i}"] = np.float64(cache[k][1])
    tmp = path + ".tmp.npz"          # np.savez appends .npz unless already there
    np.savez(tmp, **payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Patch positions
# ---------------------------------------------------------------------------
def fully_ocean_mask(ocean: np.ndarray, patch: int) -> np.ndarray:
    """Boolean map of top-left corners whose ``patch x patch`` window is all ocean.

    GenDA rejects any crop containing land by redrawing; a summed-area table
    turns that into an O(1) lookup and, more importantly, makes "no valid window
    exists" a startup error instead of an infinite loop in a DataLoader worker.
    """
    m = np.asarray(ocean) > 0.5
    ny, nx = m.shape
    if patch > ny or patch > nx:
        return np.zeros((0, 0), dtype=bool)
    ii = np.pad(np.cumsum(np.cumsum(m.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    s = (ii[patch:, patch:] - ii[:-patch, patch:]
         - ii[patch:, :-patch] + ii[:-patch, :-patch])
    return s == patch * patch


def ocean_patch_positions(
    ocean: np.ndarray, patch: int, stride: int, min_frac: float
) -> list[tuple[int, int]]:
    """Top-left corners of ``patch x patch`` windows with >= ``min_frac`` ocean."""
    mask = np.asarray(ocean) > 0.5
    ny, nx = mask.shape
    thresh = min_frac * patch * patch
    pos: list[tuple[int, int]] = []
    for y0 in range(0, ny - patch + 1, stride):
        for x0 in range(0, nx - patch + 1, stride):
            if mask[y0:y0 + patch, x0:x0 + patch].sum() >= thresh:
                pos.append((y0, x0))
    if not pos:
        raise ValueError(
            f"No {patch}x{patch} patches with >={min_frac:.0%} ocean; "
            "reduce patch/ocean_frac."
        )
    return pos


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZarrWindowDataset(Dataset):
    """k-step windowed samples of (cond, target, mask) from any described store."""

    def __init__(self, cfg: Config, split: str = "train",
                 share_from: "ZarrWindowDataset | None" = None,
                 spec: DatasetSpec | None = None):
        self.cfg = cfg
        self.split = split
        self.k = cfg.k_days
        # One spec per process; ``train.py`` builds it once and passes it in so
        # the member stores are opened a single time.
        self.spec = spec if spec is not None else resolve_spec(cfg)
        spec = self.spec

        self.ocean = spec.ocean_mask()
        self.NY, self.NX = self.ocean.shape

        # union of channels to preload (dedup, target + cond)
        self.needed = list(dict.fromkeys(cfg.cond_vars + cfg.target))
        # Shape/presence checks up front: without them a mismatched channel
        # surfaces as a broadcast error mid-preload, and an indivisible domain as
        # a skip-connection shape error deep inside the UNet.
        spec.validate(
            self.needed, patch=cfg.patch,
            downsample=2 ** (len(cfg.channel_mult) - 1),
        )

        t_full = spec.n_steps()
        self.t_end = t_full if cfg.max_days is None else min(cfg.max_days, t_full)

        # Base-cadence day-of-year phase (unix seconds, UTC).
        t_unix = spec.family.times(spec.base)[: self.t_end].astype(np.float64)
        self.doy_ang = _doy_angle(t_unix)
        self.doy_sin = np.sin(self.doy_ang).astype(np.float32)
        self.doy_cos = np.cos(self.doy_ang).astype(np.float32)

        # Per-variable base-index -> own-cadence-index maps (None = base cadence,
        # i.e. identity, which is every variable of a single-cadence dataset).
        self._vmap = {v: spec.index_map(spec.var(v).cadence) for v in self.needed}

        # Preloaded normalised channels are split-independent, so a val dataset
        # built alongside a train one reuses them rather than doubling the RAM.
        if share_from is not None:
            self.stats = share_from.stats
            self.clim = share_from.clim
            self.data = share_from.data
        elif cfg.norm_mode == "anomaly":
            spec.warn_climatology(list(cfg.clim_vars))
            self.stats, self.clim, self.data = self._load_anomaly()
        elif cfg.norm_mode == "zscore":
            self.clim = {}
            self.stats = compute_norm_stats(
                spec, self.needed, self.ocean, cfg.norm_cache, seed=cfg.seed + 42
            )
            self.data = self._load_zscore()
        else:
            raise ValueError(f"Unknown norm_mode {cfg.norm_mode!r}")

        # valid target steps: whole window inside this contiguous split & loaded
        splits = spec.splits()
        if split not in splits:
            raise KeyError(
                f"{spec.name}: no split {split!r} (have {sorted(splits)})"
            )
        split_days = sorted(splits[split])
        if split == "train" and cfg.val_gap_days > 0:
            # GenDA holds a year out between train and val; trim the tail here.
            split_days = split_days[: max(len(split_days) - cfg.val_gap_days, 0)]
        if not split_days:
            raise ValueError(f"split={split!r} is empty after val_gap_days trim")
        start = split_days[0]
        # A coarser-cadence channel may have no sample yet at the very start of
        # the record (see DatasetSpec.index_map); those steps are not usable.
        first_ok = max((spec.first_mapped_step(v) for v in self.needed), default=0)
        valid = [d for d in split_days
                 if d - (self.k - 1) >= max(start, first_ok) and d < self.t_end]
        if cfg.require_contiguous_window and self.k > 1:
            # A window is only meaningful if its steps are actually consecutive
            # in time. Off by default -- see Config.require_contiguous_window.
            # The slack is half of ONE step, not a fraction of the whole span:
            # scaling the span itself would let the allowance grow with k until
            # a real gap fits inside it and nothing is ever rejected.
            step = float(np.median(np.diff(t_unix))) if t_unix.size > 1 else 0.0
            span = step * (self.k - 1) + step * 0.5
            valid = [d for d in valid
                     if t_unix[d] - t_unix[d - (self.k - 1)] <= span]
        self.valid_days = np.array(valid, dtype=np.int64)
        if self.valid_days.size == 0:
            raise ValueError(
                f"No valid target steps for split={split!r} "
                f"(k={self.k}, t_end={self.t_end})."
            )

        if cfg.crop_mode == "grid":
            stride = max(cfg.patch // 4, 1)
            self.positions = ocean_patch_positions(
                self.ocean, cfg.patch, stride, cfg.ocean_frac
            )
            self.full_ocean = None
        elif cfg.crop_mode == "random":
            # Uniform corners over the whole domain (GenDA). Under reject_land,
            # rejection sampling from a uniform proposal is *exactly* uniform
            # over the accepted set, so enumerate the fully-ocean corners once
            # instead of running an unbounded redraw loop in every worker.
            self.positions = None
            if cfg.reject_land:
                ok = fully_ocean_mask(self.ocean, cfg.patch)
                if not ok.any():
                    raise ValueError(
                        f"reject_land=true but no fully-ocean {cfg.patch}x"
                        f"{cfg.patch} window exists in this domain; reduce patch."
                    )
                self.positions = [tuple(map(int, p)) for p in np.argwhere(ok)]
        else:
            raise ValueError(f"Unknown crop_mode {cfg.crop_mode!r}")
        self._rng = np.random.default_rng(cfg.seed)

    # -- preloading --------------------------------------------------------
    def _cad_extent(self, var: str) -> int:
        """How many steps of ``var``'s own cadence the loaded window reaches.

        For a base-cadence variable that is just ``t_end``; for a coarser one it
        is one past the index the last loaded base step maps to, so nothing is
        read that no sample can address.
        """
        m = self._vmap[var]
        if m is None:
            return self.t_end
        return int(max(m[self.t_end - 1], 0)) + 1

    def _read(self, var: str) -> np.ndarray:
        """Raw frames of ``var`` over its own cadence, ``[0, _cad_extent)``."""
        cad = self.spec.cadence_of(var)
        return np.asarray(self.spec.family.read(cad, var, 0, self._cad_extent(var)),
                          dtype=np.float32)

    def _load_zscore(self) -> dict[str, np.ndarray]:
        """Preload channels normalised by scalar (mean, std). Land -> 0.

        Shared read-only with DataLoader workers via fork copy-on-write. Each
        variable is stored against its own cadence's index space; ``_window``
        does the mapping.
        """
        data: dict[str, np.ndarray] = {}
        ocean_b = self.ocean.astype(bool)
        for v in self.needed:
            raw = self._read(v)
            m, s, is_log = self.stats[v]
            arr = _log10_positive(raw) if is_log else raw.astype(np.float32)
            arr = (arr - m) / s
            arr[~np.isfinite(arr)] = 0.0
            arr[:, ~ocean_b] = 0.0
            data[v] = arr.astype(np.float32)
        return data

    def _cad_phase(self, var: str) -> np.ndarray:
        """Day-of-year phase for each loaded step of ``var``'s own cadence."""
        m = self._vmap[var]
        if m is None:
            return self.doy_ang
        cad = self.spec.cadence_of(var)
        t = self.spec.family.times(cad)[: self._cad_extent(var)].astype(np.float64)
        return _doy_angle(t)

    def _load_anomaly(self):
        """Preload channels as GenDA-style anomalies.

        Per variable: log10 first if lognormal, subtract a per-pixel climatology
        (harmonic in day-of-year for ``clim_vars``, plain time-mean otherwise)
        fit on the **train split only**, then divide by a scalar anomaly std.

        Unlike :func:`compute_norm_stats`, this path is fit strictly on the train
        split -- it is newer code with no checkpoints depending on it, so the
        leakage noted in that function is simply not reproduced.
        """
        cfg = self.cfg
        spec = self.spec
        base_train = np.array(sorted(spec.splits()["train"]), dtype=np.int64)
        base_train = base_train[base_train < self.t_end]
        if base_train.size == 0:
            raise ValueError("No train steps available to fit the climatology")

        cache = _load_clim_cache(cfg.clim_cache)
        ocean_b = self.ocean.astype(bool)
        clim_vars = set(cfg.clim_vars)
        stats: dict[str, tuple] = {}
        clim: dict[str, np.ndarray] = {}
        data: dict[str, np.ndarray] = {}
        dirty = False

        for v in self.needed:
            is_log = spec.is_log(v)
            seasonal = v in clim_vars
            arr = self._read(v)
            if is_log:
                arr = _log10_positive(arr).astype(np.float32)
            ang = self._cad_phase(v)
            # The fit must run in the variable's own index space, so map the
            # train steps across when it is not on the base cadence.
            m = self._vmap[v]
            if m is None:
                train_days = base_train
            else:
                mapped = m[base_train]
                train_days = np.unique(mapped[mapped >= 0])
                if train_days.size == 0:
                    raise ValueError(
                        f"{v!r} has no train-split samples on cadence "
                        f"{spec.var(v).cadence!r}"
                    )

            key = _clim_key(spec.cache_key(v), seasonal,
                            int(arr.shape[0]), int(train_days.size))
            coef = cache[key][0] if key in cache else fit_climatology(
                arr, ang, self.ocean, seasonal, days=train_days
            )
            subtract_climatology_(arr, coef, ang)

            if key in cache:
                s = float(cache[key][1])
            else:
                # Scalar per-variable anomaly std over ocean pixels of the train
                # split -- GenDA's var_stds JSON is one number per variable, not
                # a per-pixel field.
                tr = arr[train_days][:, ocean_b]
                s = float(np.nanstd(tr)) if np.isfinite(tr).any() else 1.0
                s = max(s, 1e-8)
                cache[key] = (coef, s)
                dirty = True
                del tr

            arr /= s
            arr[~np.isfinite(arr)] = 0.0
            arr[:, ~ocean_b] = 0.0
            stats[v] = (coef, s, is_log)
            clim[v] = coef
            data[v] = arr

        if dirty and cfg.clim_cache and self._may_write_cache():
            _save_clim_cache(cfg.clim_cache, cache)
        return stats, clim, data

    @staticmethod
    def _may_write_cache() -> bool:
        """Only rank 0 writes the shared climatology cache.

        Every DDP rank runs the fit, so without this guard several processes
        race on the same file. The write itself is atomic (see
        :func:`_save_clim_cache`); this keeps the redundant writes from
        happening at all.
        """
        return int(os.environ.get("RANK", 0)) == 0

    # -- length: nominal crops per valid day -------------------------------
    def __len__(self) -> int:
        return int(self.valid_days.size * self.cfg.crops_per_day)

    def set_epoch_rng(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    # -- assemble the channel-stacked conditioning tensor ------------------
    def build_condition(
        self, obs: np.ndarray, mask2d: np.ndarray, day: int
    ) -> np.ndarray:
        """``obs`` is (k, Cobs, H, W), oldest->newest. Returns (Cc, H, W)."""
        k, cobs, h, w = obs.shape
        chans = [obs.reshape(k * cobs, h, w)]
        if self.cfg.use_ocean_mask:
            chans.append(mask2d[None])
        if self.cfg.use_doy:
            chans.append(np.full((1, h, w), self.doy_sin[day], dtype=np.float32))
            chans.append(np.full((1, h, w), self.doy_cos[day], dtype=np.float32))
        return np.concatenate(chans, axis=0).astype(np.float32)

    def _idx(self, var: str, day: int) -> int:
        """Base-cadence step ``day`` -> the index to read for ``var``."""
        m = self._vmap[var]
        return day if m is None else int(m[day])

    def _window(self, day: int, ysl: slice, xsl: slice):
        days = range(day - self.k + 1, day + 1)
        cobs = len(self.cfg.cond_vars)
        if cobs == 0:
            # unconditional prior: no observation channels to window over.
            # np.stack needs >=1 array, so build the zero-width (k,0,H,W)
            # array directly instead of stacking an empty list.
            h, w = self.ocean[ysl, xsl].shape
            obs = np.zeros((self.k, 0, h, w), dtype=np.float32)
        else:
            # A coarser-cadence channel repeats across the base steps it covers,
            # which is the honest representation: that is the only observation
            # available at those times.
            obs = np.stack(
                [np.stack([self.data[v][self._idx(v, d), ysl, xsl]
                           for v in self.cfg.cond_vars], 0)
                 for d in days],
                axis=0,
            )  # (k, Cobs, H, W)
        target = np.stack([self.data[v][self._idx(v, day), ysl, xsl]
                           for v in self.cfg.target], 0)
        return obs, target

    def sample_position(self) -> tuple[int, int]:
        """Top-left corner of one training crop.

        ``self.positions`` is either the ocean_frac-filtered lattice (grid mode)
        or the enumerated fully-ocean corners (random + reject_land); ``None``
        means unconstrained uniform sampling over the domain.
        """
        if self.positions is not None:
            return self.positions[self._rng.integers(len(self.positions))]
        p = self.cfg.patch
        return (int(self._rng.integers(self.NY - p + 1)),
                int(self._rng.integers(self.NX - p + 1)))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        day = int(self._rng.choice(self.valid_days))
        y0, x0 = self.sample_position()
        p = self.cfg.patch
        ysl, xsl = slice(y0, y0 + p), slice(x0, x0 + p)

        obs, target = self._window(day, ysl, xsl)
        mask2d = (self.ocean[ysl, xsl] > 0.5).astype(np.float32)
        cond = self.build_condition(obs, mask2d, day)
        return {
            "cond": torch.from_numpy(cond),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask2d[None]),
            "day": torch.tensor(day, dtype=torch.long),
        }

    # -- full-frame iterator for sampling / evaluation ---------------------
    def full_frames(self, day_list):
        """Yield full-domain (cond, target, mask, day) per target step."""
        full = slice(None)
        mask2d = (self.ocean > 0.5).astype(np.float32)
        for day in day_list:
            day = int(day)
            obs, target = self._window(day, full, full)
            cond = self.build_condition(obs, mask2d, day)
            yield {
                "cond": torch.from_numpy(cond),
                "target": torch.from_numpy(target),
                "mask": torch.from_numpy(mask2d[None]),
                "day": day,
            }

    # -- inverse normalisation --------------------------------------------
    def denorm_params(self, var: str, day: int | None = None):
        """``(mean, std, is_log)`` for ``var``, resolved for ``day``.

        ``mean`` is a scalar under ``norm_mode: zscore``, a ``(NY, NX)`` field
        under ``anomaly`` (the day's climatology). Both broadcast against a
        full-frame array, so callers need no special-casing.
        """
        m, s, is_log = self.stats[var]
        if self.cfg.norm_mode == "anomaly":
            if day is None:
                raise ValueError(
                    f"norm_mode=anomaly needs a day to denormalize {var!r} "
                    "(the climatology is day-dependent)"
                )
            m = evaluate_climatology(m, float(self.doy_ang[int(day)]))
        return m, s, is_log

    def denormalize(self, var: str, arr: np.ndarray,
                    day: int | None = None) -> np.ndarray:
        m, s, is_log = self.denorm_params(var, day)
        out = arr * s + m
        return np.power(10.0, out) if is_log else out

    def default_eval_days(self, n: int) -> list[int]:
        """The last ``n`` valid target steps (``n <= 0`` = the whole split)."""
        if n <= 0:
            return self.valid_days.tolist()
        return self.valid_days[-n:].tolist()


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    base = ds.cfg.seed + 1000 * (int(os.environ.get("RANK", 0)) + 1)
    ds.set_epoch_rng(base + worker_id)
