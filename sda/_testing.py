"""Test doubles for the SDA layer: run every code path without a store or GPU.

Pattern copied from ``eval/gen_cond.py:627-698`` (``_ZeroNet``, ``_FakeSpec``,
``_fake_dataset``); not imported from there because the eval layers are kept
parallel, not nested. Extended with a Gaussian-prior denoiser whose posterior
is known in closed form, and a fake store family that serves NaN-gapped frames.
"""
from __future__ import annotations

import numpy as np
import torch

from diffusion.data import ZarrWindowDataset, ocean_patch_positions


class _ZeroNet(torch.nn.Module):
    """``D = 0``: the VP predictor then contracts x by sigma'/sigma each step and
    the sample ends at exactly ``eta * x_1``."""

    def __init__(self, cc: int = 0):
        super().__init__()
        self.cc = cc

    def forward(self, x, sigma, cond):  # noqa: ARG002
        assert cond.shape[1] == self.cc, f"cond has {cond.shape[1]} channels, expected {self.cc}"
        return torch.zeros_like(x)


class _GaussPriorNet(torch.nn.Module):
    """Exact EDM denoiser of the prior ``x0 ~ N(0, I)``: ``D(x, sigma) = x / (1 + sigma^2)``.

    With it the guided sampler targets a linear-Gaussian posterior we can write
    down, and ``var_fn = std^2 + sigma^2/(1+sigma^2)`` is the *exact* Tweedie
    covariance (``Cov[x0 | x] = sigma^2/(1+sigma^2) I``)."""

    def __init__(self, cc: int = 0):
        super().__init__()
        self.cc = cc

    def forward(self, x, sigma, cond):  # noqa: ARG002
        s = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        return x / (1.0 + s ** 2)


def gauss_prior_var_fn(std, sigma):
    return std ** 2 + sigma ** 2 / (1.0 + sigma ** 2)


class _FakeFamily:
    """Serves ``take(cad, var, idx)`` frames with NaN gaps, and times per cadence."""

    def __init__(self, t_by_cad: dict, ny: int, nx: int, seed: int = 0):
        self._t = t_by_cad
        self.ny, self.nx = ny, nx
        self.seed = seed

    def times(self, cad):
        return self._t[cad.name]

    def take(self, cad, var, idx):
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros((idx.size, self.ny, self.nx), dtype=np.float32)
        for j, i in enumerate(idx):
            rng = np.random.default_rng([self.seed, hash(var) % 2 ** 31, int(i)])
            # consistent with _fake_dataset's zscore stats: sst-like fields sit at 20
            mean = 20.0 if var.startswith("sst") else 0.0
            f = rng.standard_normal((self.ny, self.nx)).astype(np.float32) * 0.8 + mean
            gap = rng.random((self.ny, self.nx)) < 0.4
            f[gap] = np.nan
            out[j] = f
        return out

    def coord(self, path):
        if path == "coords/lat":
            return (25.0 + np.arange(self.ny)[:, None] * 0.02 + 0 * np.arange(self.nx)[None, :]
                    ).astype(np.float32)
        raise KeyError(path)


class _Cad:
    def __init__(self, name):
        self.name = name


class _FakeSpec:
    """Enough of ``DatasetSpec`` for ``sda.case``: base + one coarser cadence."""

    def __init__(self, t_unix, ny=48, nx=64, seed=0):
        self.name = "fake"
        self.base_cadence = "daily"
        self.base = _Cad("daily")
        self._cads = {"daily": self.base, "weekly": _Cad("weekly")}
        t_weekly = t_unix[::7] + 43200.0
        self.family = _FakeFamily({"daily": t_unix, "weekly": t_weekly}, ny, nx, seed)
        self._var_cad = {"sss_sat": "weekly"}
        self._ny, self._nx = ny, nx

    def cadence_of(self, name):
        return self._cads[self._var_cad.get(name, "daily")]

    def map_days(self, name, days):
        days = np.asarray(days, dtype=np.int64)
        cad = self._var_cad.get(name, "daily")
        if cad == "daily":
            return days
        t_base = self.family.times(self.base)[days]
        t_other = self.family.times(self._cads[cad])
        return np.minimum(np.searchsorted(t_other, t_base, side="right") - 1, len(t_other) - 1)

    def dx_m(self):
        return 1818.33

    def ocean_mask(self):
        m = np.ones((self._ny, self._nx), dtype=np.float32)
        m[:5, :] = 0.0
        m[20:28, 30:38] = 0.0
        return m


def _fake_dataset(cfg, ny=48, nx=64, T=80, seed=0):
    """A ``ZarrWindowDataset`` with its attributes planted so the REAL
    ``full_frames`` / ``denorm_params`` / crop methods run over synthetic arrays."""
    rng = np.random.default_rng(seed)
    ds = ZarrWindowDataset.__new__(ZarrWindowDataset)
    ds.cfg, ds.k, ds.split = cfg, cfg.k_days, "val"
    spec = _FakeSpec(1.5e9 + np.arange(T) * 86400.0, ny, nx, seed)
    ds.ocean = spec.ocean_mask()
    ds.NY, ds.NX = ny, nx
    ds.t_end = T
    t_unix = spec.family.times(spec.base)
    year = 365.2425 * 86400.0
    ds.doy_ang = 2 * np.pi * (np.mod(t_unix, year) / year)
    ds.doy_sin = np.sin(ds.doy_ang).astype(np.float32)
    ds.doy_cos = np.cos(ds.doy_ang).astype(np.float32)
    needed = list(dict.fromkeys(cfg.cond_vars + cfg.target))
    ds._vmap = {v: None for v in needed}
    ds.stats, ds.clim, ds.data = {}, {}, {}
    ocean_b = ds.ocean > 0.5
    for i, v in enumerate(needed):
        is_log = v.upper().startswith("CHL")
        ds.stats[v] = (20.0 * (i == 1), 1.0 + 0.1 * i, is_log)   # zscore: scalar mean/std
        arr = rng.standard_normal((T, ny, nx)).astype(np.float32)
        arr[:, ~ocean_b] = 0.0
        ds.data[v] = arr
    ds.valid_days = np.arange(cfg.k_days - 1 + 10, T, dtype=np.int64)
    ds.positions = ocean_patch_positions(ds.ocean, cfg.patch, max(cfg.patch // 4, 1),
                                         cfg.ocean_frac)
    ds.full_ocean = None
    ds._rng = np.random.default_rng(cfg.seed)
    ds.spec = spec
    return ds
