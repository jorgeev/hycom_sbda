"""Median patch spectra of tau_x/tau_y from three provenances, same PSD path.

Arms (128x128 patches, dx = 1.81833 km, identical radial-PSD numerics):
  eval    -- the real curve exactly as fig2_spectra sees it: real_norm * std
             straight out of samples_128.npz (bilinear store + loader anomaly).
  store   -- recomputed from datasets/gulfstream_*.zarr at the SAME
             (day, y0, x0) triples as the npz; anomaly = frame minus the
             per-pixel mean over the sampled frames. Should overlay `eval`
             (method-parity check).
  source  -- raw native-grid taux/tauy files, no regridding by us. Same days;
             each patch is the 128x128 native-cell window centred on the same
             geographic point as the store patch (native cells are also
             1818.33 m squares, so footprint and dx match).
  bicubic -- the RectBivariateSpline k=3 re-regridded store from
             make_bicubic_tau.py, same (day, y0, x0) and anomaly as `store`.

Every arm is evaluated under TWO tapers: the eval's 2-D Hann and a Tukey
alpha=0.2, written to figs/tau_spectra_compare.npz and
figs/tau_spectra_compare_tukey02.npz respectively (each self-describing via
its `window` entry; plot_compare.py renders one figure per file).

Run:  python tau_interp_check/compute_spectra.py [--n 512]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import (BICUBIC_STORE, SAMPLES_NPZ, StoreCat, check_hann_parity,
                    native_axes, psd_stats, read_native)

TAUS = ("tau_x", "tau_y")
WINDOWS = ("hann", "tukey02")
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
OUT = os.path.join(FIGS_DIR, "tau_spectra_compare.npz")


def crops(frames_by_day: dict[int, np.ndarray], days, pos, patch=128):
    out = []
    for d, (y0, x0) in zip(days, pos):
        out.append(frames_by_day[int(d)][y0:y0 + patch, x0:x0 + patch])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="use first n patches")
    ap.add_argument("--npz", default=SAMPLES_NPZ)
    ap.add_argument("--bicubic", default=BICUBIC_STORE)
    ap.add_argument("--out", default=OUT,
                    help="hann output; the tukey02 file gets a suffix")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    vars_ = json.loads(str(d["vars"]))
    days = d["real_days"]
    pos = d["real_pos"]
    std = d["std"]
    dx_km = float(d["dx_m"]) / 1000.0
    if args.n:
        days, pos = days[:args.n], pos[:args.n]
    udays = np.unique(days)
    patch = int(d["real_norm"].shape[-1])
    check_hann_parity(dx_km, patch)
    print(f"[cfg] {len(days)} patches, {udays.size} unique steps, "
          f"patch {patch}, dx {dx_km:.5f} km, windows {WINDOWS}", flush=True)

    cat = StoreCat()
    time_axis = cat.time_axis()

    # ---- gather the anomaly patch stacks once per (arm, var) --------------
    anoms: dict[str, dict[str, np.ndarray]] = {}

    anoms["eval"] = {
        v: d["real_norm"][:len(days), vars_.index(v)].astype(np.float64)
           * float(std[vars_.index(v)])
        for v in TAUS}
    print("[eval   ] patches ready", flush=True)

    import zarr
    bic = zarr.open(args.bicubic, mode="r")
    bic_done = np.asarray(bic["done"][:])
    if not bic_done[udays].all():
        raise SystemExit(f"[bicubic] store incomplete: "
                         f"{int((~bic_done[udays].astype(bool)).sum())} of the "
                         f"sampled steps missing -- run make_bicubic_tau.py")
    for arm, reader in (("store", lambda v, g: cat.take(v, [g])[0]),
                        ("bicubic", lambda v, g: np.asarray(bic[v][g]))):
        t0 = time.time()
        anoms[arm] = {}
        for v in TAUS:
            frames = {int(g): reader(v, int(g)).astype(np.float64)
                      for g in udays}
            pixmean = np.mean([f for f in frames.values()], axis=0)
            anoms[arm][v] = crops({g: f - pixmean for g, f in frames.items()},
                                  days, pos, patch)
            print(f"[{arm:7s}] {v} patches ready ({time.time() - t0:.0f}s)",
                  flush=True)

    # source arm: map each target patch centre to the nearest native cell and
    # take the 128x128 native window around it -- same geography, same cells.
    lat1d, lon1d = native_axes()
    tlat = cat.coord("coords/lat").astype(np.float64)
    tlon = cat.coord("coords/lon").astype(np.float64)
    half = patch // 2

    def native_corner(y0, x0):
        yc, xc = y0 + half, x0 + half
        jc = np.searchsorted(lat1d, tlat[yc, xc])       # nearest-below is fine
        ic = np.searchsorted(lon1d, tlon[yc, xc])
        j0 = int(np.clip(jc - half, 0, lat1d.size - patch))
        i0 = int(np.clip(ic - half, 0, lon1d.size - patch))
        return j0, i0

    npos = [native_corner(y0, x0) for (y0, x0) in pos]
    t0 = time.time()
    anoms["source"] = {}
    for var_dir, v in (("taux", "tau_x"), ("tauy", "tau_y")):
        frames = {int(g): read_native(var_dir, float(time_axis[g])).astype(np.float32)
                  for g in udays}
        pixmean = np.mean([f for f in frames.values()], axis=0, dtype=np.float64)
        anoms["source"][v] = crops({g: f - pixmean for g, f in frames.items()},
                                   days, npos, patch)
        print(f"[source ] {v} patches ready ({time.time() - t0:.0f}s)", flush=True)

    # ---- PSD under each taper --------------------------------------------
    os.makedirs(FIGS_DIR, exist_ok=True)
    for window in WINDOWS:
        res = {"dx_km": dx_km, "n": len(days), "days": days, "pos": pos,
               "window": window}
        for arm in ("eval", "store", "bicubic", "source"):
            for v in TAUS:
                k, med, q25, q75 = psd_stats(anoms[arm][v], dx_km, window)
                res["k"] = k
                res.update({f"{arm}_{v}_med": med, f"{arm}_{v}_q25": q25,
                            f"{arm}_{v}_q75": q75})
        out = (args.out if window == "hann"
               else args.out.replace(".npz", f"_{window}.npz"))
        np.savez(out, **res)
        print(f"[out] wrote {out}", flush=True)

        # parity check: store recompute vs the eval curve
        for v in TAUS:
            r = np.abs(np.log10(res[f"store_{v}_med"] / res[f"eval_{v}_med"]))
            print(f"[check] {window} {v}: median |log10(store/eval)| = "
                  f"{np.median(r):.3f} (max {r.max():.3f}) -- should be small",
                  flush=True)


if __name__ == "__main__":
    main()
