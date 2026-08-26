"""Regrid native taux/tauy onto the training AEQD grid with a C2 bicubic spline.

Same source files, same target grid and time axis as the training stores --
the ONLY difference from make_gulfstream_dataset.py is the interpolant:
scipy RectBivariateSpline(kx=3, ky=3) on the separable native lat/lon axes
instead of the hand-rolled bilinear weights. Output is a single zarr v2 store
so the spectra comparison (and any later retraining experiment) can read it
exactly like `dynamic/tau_{x,y}` in the monthly stores.

Run:  python tau_interp_check/make_bicubic_tau.py [--steps a:b] [--workers N]
Resumable: finished steps are recorded in the `done` array and skipped.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import (AUX_DIR, BICUBIC_STORE, StoreCat, nan_fill_nearest,
                    native_axes, read_native)

_G = {}   # per-worker globals (fork inherits, but keep it explicit/lazy)


def _init_worker(lat1d, lon1d, tlat_flat, tlon_flat, shape):
    _G.update(lat1d=lat1d, lon1d=lon1d, tlat=tlat_flat, tlon=tlon_flat,
              shape=shape)


def _regrid_one(arg):
    """(step, t_unix) -> (step, tau_x, tau_y) bicubic on the target grid."""
    from scipy.interpolate import RectBivariateSpline
    step, t_unix = arg
    out = []
    for var_dir in ("taux", "tauy"):
        field = nan_fill_nearest(read_native(var_dir, t_unix))
        spl = RectBivariateSpline(_G["lat1d"], _G["lon1d"], field, kx=3, ky=3, s=0)
        out.append(spl.ev(_G["tlat"], _G["tlon"])
                   .reshape(_G["shape"]).astype(np.float32))
    return step, out[0], out[1]


def open_or_create(cat: StoreCat, time_axis: np.ndarray):
    import os

    import zarr
    from numcodecs import Blosc
    assert zarr.__version__.startswith("2."), "needs zarr v2 (datorch env)"
    os.makedirs(AUX_DIR, exist_ok=True)
    g = zarr.open_group(BICUBIC_STORE, mode="a")
    if "tau_x" in g:
        return g
    ny, nx = cat.grid["NY"], cat.grid["NX"]
    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    for name in ("tau_x", "tau_y"):
        g.create_dataset(name, shape=(time_axis.size, ny, nx),
                         chunks=(1, ny, nx), dtype="f4", compressor=comp,
                         fill_value=np.nan)
    g.array("time", time_axis.astype("f8"))
    for c in ("x", "y", "lat", "lon"):
        g.array(c, cat.coord(f"coords/{c}"))
    g.zeros("done", shape=(time_axis.size,), chunks=(time_axis.size,), dtype="u1")
    g.attrs.update(
        method="RectBivariateSpline kx=3 ky=3 (C2 bicubic) on separable native "
               "lat/lon axes, evaluated at the target grid's 2-D lat/lon",
        source="/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/{taux,tauy}",
        target_grid=dict(cat.grid),
        note="same target grid/time axis as datasets/gulfstream_*.zarr dynamic "
             "group; built by hycom_sbda/tau_interp_check/make_bicubic_tau.py "
             "to test whether the ~4dx tau spectral spike is a bilinear-regrid "
             "artifact",
    )
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=None, help="a:b global step slice")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cat = StoreCat()
    time_axis = cat.time_axis()
    lat1d, lon1d = native_axes()
    tlat = cat.coord("coords/lat").astype(np.float64)
    tlon = cat.coord("coords/lon").astype(np.float64)
    g = open_or_create(cat, time_axis)

    lo, hi = 0, time_axis.size
    if args.steps:
        a, b = args.steps.split(":")
        lo, hi = int(a or 0), int(b or time_axis.size)
    done = np.asarray(g["done"][:])
    todo = [(s, float(time_axis[s])) for s in range(lo, hi) if not done[s]]
    print(f"[bicubic] {len(todo)} of {hi - lo} steps to do "
          f"(target {tlat.shape}, native {lat1d.size}x{lon1d.size})", flush=True)

    t0, nd = time.time(), 0
    with ProcessPoolExecutor(
            args.workers, initializer=_init_worker,
            initargs=(lat1d, lon1d, tlat.ravel(), tlon.ravel(), tlat.shape)) as ex:
        for step, tx, ty in ex.map(_regrid_one, todo, chunksize=4):
            g["tau_x"][step] = tx
            g["tau_y"][step] = ty
            g["done"][step] = 1
            nd += 1
            if nd % 100 == 0 or nd == len(todo):
                el = time.time() - t0
                print(f"[bicubic] {nd}/{len(todo)}  {el:6.0f}s  "
                      f"{el / nd:.2f}s/step", flush=True)
    print(f"[bicubic] wrote {BICUBIC_STORE}", flush=True)


if __name__ == "__main__":
    main()
