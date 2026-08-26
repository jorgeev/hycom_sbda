"""Shared plumbing for the tau spectral-spike investigation.

Temporary analysis code (see tau_interp_check/README.md). Everything numeric
that matters is imported from ``eval.kernels`` so the three spectra compared
here go through the byte-identical PSD path used by fig2_spectra.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SRC_ROOT = "/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3"
DATASETS_DIR = f"{SRC_ROOT}/datasets"
MANIFEST = f"{DATASETS_DIR}/gulfstream_manifest.json"
AUX_DIR = f"{SRC_ROOT}/auxiliar_datasets"
BICUBIC_STORE = f"{AUX_DIR}/gulfstream_tau_bicubic.zarr"
SAMPLES_NPZ = os.path.join(
    REPO_ROOT, "runs/prior_gulfstream/eval_step1140000/samples_128.npz")


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


class StoreCat:
    """The seven monthly zarr stores presented as one global hourly axis."""

    def __init__(self):
        import zarr
        man = load_manifest()
        self.stores, self.offsets, self.lengths = [], [], []
        for key in sorted(man["stores"]):
            s = man["stores"][key]
            self.stores.append(
                zarr.open(os.path.join(DATASETS_DIR, s["path"]), mode="r"))
            self.offsets.append(int(s["global_hour_offset"]))
            self.lengths.append(int(s["n_hours"]))
        self.n_steps = self.offsets[-1] + self.lengths[-1]
        self.grid = man["grid"]

    def locate(self, g: int) -> tuple[int, int]:
        si = int(np.searchsorted(self.offsets, g, side="right")) - 1
        return si, g - self.offsets[si]

    def take(self, var: str, global_idx: np.ndarray) -> np.ndarray:
        """(N, NY, NX) float32 frames of ``dynamic/<var>`` at global indices."""
        out = []
        for g in np.asarray(global_idx, dtype=np.int64):
            si, li = self.locate(int(g))
            out.append(self.stores[si][f"dynamic/{var}"][li])
        return np.stack(out, 0).astype(np.float32)

    def coord(self, name: str) -> np.ndarray:
        return np.asarray(self.stores[0][name])

    def time_axis(self) -> np.ndarray:
        return np.concatenate([np.asarray(s["coords/time"]) for s in self.stores])


def source_path(var_dir: str, t_unix: float) -> str:
    """Native HYCOM file for one hourly timestamp (var_dir: 'taux'|'tauy')."""
    t = datetime.datetime.fromtimestamp(round(float(t_unix)),
                                        tz=datetime.timezone.utc)
    doy, hh = t.timetuple().tm_yday, t.hour
    return f"{SRC_ROOT}/{var_dir}/{var_dir}_box56_0017_{doy:03d}_{hh:02d}.nc"


def native_axes() -> tuple[np.ndarray, np.ndarray]:
    """Separable 1-D (lat, lon) axes of the native grid, from one taux file."""
    from netCDF4 import Dataset
    with Dataset(source_path("taux", 1483232400)) as ds:   # 2017-01-01T01
        plat = np.asarray(ds["plat"][:], dtype=np.float64)
        plon = np.asarray(ds["plon"][:], dtype=np.float64)
    lat_dev = np.nanmax(np.abs(plat - plat[:, :1]))
    lon_dev = np.nanmax(np.abs(plon - plon[:1, :]))
    if lat_dev > 1e-3 or lon_dev > 1e-3:
        raise ValueError(f"native grid not separable ({lat_dev=}, {lon_dev=})")
    return plat[:, 0].copy(), plon[0, :].copy()


def read_native(var_dir: str, t_unix: float) -> np.ndarray:
    from netCDF4 import Dataset
    with Dataset(source_path(var_dir, t_unix)) as ds:
        return np.asarray(ds[var_dir][:], dtype=np.float64)


def nan_fill_nearest(field: np.ndarray) -> np.ndarray:
    """Replace NaNs by nearest finite value (no-op when already finite)."""
    bad = ~np.isfinite(field)
    if not bad.any():
        return field
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def _win1d(n: int, window: str) -> np.ndarray:
    if window == "hann":
        return np.hanning(n)
    if window == "tukey02":
        from scipy.signal.windows import tukey
        return tukey(n, alpha=0.2, sym=True)
    raise ValueError(f"unknown window {window!r}")


def radial_psd_win(field2d: np.ndarray, dx_km: float, window: str = "hann"
                   ) -> tuple[np.ndarray, np.ndarray]:
    """eval.kernels.radial_psd with a configurable taper.

    Byte-identical to the eval version for window='hann' (asserted by
    ``check_hann_parity``); only the 1-D taper is swappable.
    """
    arr = np.asarray(field2d, dtype=np.float64)
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"radial_psd expects a square tile, got {arr.shape}.")
    arr = arr - arr.mean()
    win1d = _win1d(n, window)
    win = np.outer(win1d, win1d)
    power = np.abs(np.fft.fft2(arr * win)) ** 2
    power /= np.sum(win ** 2)

    freq = np.fft.fftfreq(n, d=dx_km)
    kx, ky = np.meshgrid(freq, freq)
    kmag = np.sqrt(kx ** 2 + ky ** 2).ravel()
    p = power.ravel()

    kmax = float(freq.max())
    edges = np.linspace(0.0, kmax, n // 2 + 1)
    which = np.digitize(kmag, edges)
    psd = np.array([
        p[which == b].mean() if np.any(which == b) else np.nan
        for b in range(1, len(edges))
    ])
    centers = 0.5 * (edges[1:] + edges[:-1])
    keep = np.isfinite(psd) & (centers > 0)
    return centers[keep], psd[keep]


def check_hann_parity(dx_km: float, n: int = 128, seed: int = 0) -> None:
    """Assert radial_psd_win('hann') == eval.kernels.radial_psd exactly."""
    from eval.kernels import radial_psd
    x = np.random.default_rng(seed).standard_normal((n, n))
    ka, pa = radial_psd(x, dx_km)
    kb, pb = radial_psd_win(x, dx_km, "hann")
    if not (np.array_equal(ka, kb) and np.allclose(pa, pb, rtol=0, atol=0)):
        raise AssertionError("radial_psd_win('hann') != eval.kernels.radial_psd")


def psd_stats(patches: np.ndarray, dx_km: float, window: str = "hann"
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(k, median, q25, q75) across patches."""
    rows, k = [], None
    for p in patches:
        p = np.where(np.isfinite(p), p, 0.0)     # loader convention: land -> 0
        k, psd = radial_psd_win(p, dx_km, window)
        rows.append(psd)
    rows = np.asarray(rows)
    return (k, np.median(rows, 0),
            np.percentile(rows, 25, 0), np.percentile(rows, 75, 0))
