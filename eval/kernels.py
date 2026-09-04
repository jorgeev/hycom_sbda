"""Numerics shared by the evaluation scripts: spectra, geostrophy, EKE, moments.

Pure numpy/scipy -- no torch, no store access except in ``--selftest``. Every
function here is small enough to check by eye, which is the point: the figures
in ``diagnostics.py`` are only worth as much as these are.

``radial_psd`` is ported verbatim from ``nemo_confusion/run_diagnostics.py``
:630. ``tile_positions`` comes from ``_ocean_tile_positions`` :667 there, with a
``min_frac`` threshold added (see its docstring). ``geostrophic_uv`` is adapted
from ``nemo_confusion/diffusion/metrics.py`` :340, whose default
``lat_deg=22.43`` is Gulf-of-Mexico specific. They are copied rather than
imported because this repo is standalone by construction.

Run ``python -m eval.kernels --selftest`` before trusting any number that comes
out of them. The geostrophy check in particular is not decorative: it pins the
sign convention and the Coriolis factor against the store's own ``ug``/``vg``,
and a sign error there would silently halve every EKE number instead of
crashing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

OMEGA = 7.2921e-5      # earth rotation rate, s^-1
G = 9.81               # gravity, m s^-2


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def radial_psd(field2d: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic (azimuthally averaged) power spectral density of a 2D tile.

    The tile must be square, finite and NaN-free. The field mean is removed and
    a 2D Hann window applied before the FFT to suppress spectral leakage; power
    is normalised by the window energy so tiles are mutually comparable.

    Returns ``(k, psd)`` with wavenumber ``k`` in cycles/km (wavelength = 1/k).
    The DC (k=0) bin is dropped.
    """
    arr = np.asarray(field2d, dtype=np.float64)
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"radial_psd expects a square tile, got {arr.shape}.")
    arr = arr - arr.mean()
    win1d = np.hanning(n)
    win = np.outer(win1d, win1d)
    power = np.abs(np.fft.fft2(arr * win)) ** 2
    power /= np.sum(win ** 2)                    # window-energy normalisation

    freq = np.fft.fftfreq(n, d=dx_km)            # cycles/km
    kx, ky = np.meshgrid(freq, freq)
    kmag = np.sqrt(kx ** 2 + ky ** 2).ravel()
    p = power.ravel()

    kmax = float(freq.max())                     # Nyquist
    edges = np.linspace(0.0, kmax, n // 2 + 1)
    which = np.digitize(kmag, edges)
    psd = np.array([
        p[which == b].mean() if np.any(which == b) else np.nan
        for b in range(1, len(edges))
    ])
    centers = 0.5 * (edges[1:] + edges[:-1])
    keep = np.isfinite(psd) & (centers > 0)
    return centers[keep], psd[keep]


def tile_positions(mask: np.ndarray, tile: int, stride: int,
                   min_frac: float = 1.0) -> list[tuple[int, int]]:
    """Top-left corners of every ``tile x tile`` window that is ocean enough.

    ``min_frac = 1.0`` reproduces the parent repo's fully-ocean rule, which is
    what you want where land is a large connected region whose zeros would
    dominate the FFT. On gulfstream "land" is 17 isolated pixels in a 576x936
    domain, and the loader sets them to 0.0 -- which in anomaly units IS the
    mean, so they perturb a tile's spectrum by essentially nothing. Insisting on
    1.0 there just throws away tiles, and on a 128x128 patch it throws away the
    only one.
    """
    m = np.asarray(mask) > 0.5
    ny, nx = m.shape
    need = min_frac * tile * tile
    out: list[tuple[int, int]] = []
    for y0 in range(0, ny - tile + 1, stride):
        for x0 in range(0, nx - tile + 1, stride):
            if m[y0:y0 + tile, x0:x0 + tile].sum() >= need:
                out.append((y0, x0))
    return out


def psd_tile(mask: np.ndarray, ny: int, nx: int,
             candidates: tuple[int, ...] = (256, 192, 128, 96, 64),
             min_frac: float = 0.999) -> int:
    """Largest square tile that both fits the domain and finds an ocean window.

    A fixed 256 works on an open-ocean box like gulfstream and fails outright on
    the Gulf of Mexico: at 464x528 with 63.8 % ocean, land is one large connected
    region and NO 256x256 window -- nor 192 -- is even 95 % ocean, so
    ``tile_positions`` returns nothing and ``mean_radial_psd`` raises.

    The fix is to shrink the tile, not to relax ``min_frac``. Land enters the
    anomaly field as exact zeros, which is the mean, so a tile straddling
    Florida is a large hole rather than a slightly noisier estimate, and its
    spectrum is dominated by the edge. A smaller all-ocean tile measures less of
    the spectrum but measures it correctly.

    Returns ``min(ny, nx)`` if no candidate qualifies, which is the whole-frame
    behaviour the 128 px patch geometry wants.
    """
    for t in candidates:
        if t <= min(ny, nx) and tile_positions(mask, t, t // 2, min_frac):
            return t
    return min(ny, nx)


def mean_radial_psd(stack: np.ndarray, mask: np.ndarray, dx_km: float,
                    tile: int = 256, stride: int = 128, min_frac: float = 0.999
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample tile-averaged radial PSD of ``stack`` (N, H, W).

    Returns ``(k, psd)`` with ``psd`` of shape ``(N, nk)`` -- one spectrum per
    sample, so the caller can draw a spread band across samples rather than
    collapsing everything into a single mean and hiding the variability.

    If the domain is smaller than ``tile`` the whole frame is used instead,
    which is what the 128 px patch geometry wants.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n, ny, nx = stack.shape
    if ny < tile or nx < tile:
        tile = min(ny, nx)
        pos = [(0, 0)]
    else:
        pos = tile_positions(mask, tile, stride, min_frac)
        if not pos:
            raise ValueError(
                f"no {tile}x{tile} tile in a {ny}x{nx} domain reaches "
                f"{min_frac:.3f} ocean fraction")

    k = None
    rows = []
    for i in range(n):
        acc = None
        for (y0, x0) in pos:
            kk, pp = radial_psd(stack[i, y0:y0 + tile, x0:x0 + tile], dx_km)
            acc = pp if acc is None else acc + pp
            k = kk
        rows.append(acc / len(pos))
    return k, np.asarray(rows)


def band_ratio(k: np.ndarray, psd_gen: np.ndarray, psd_real: np.ndarray,
               lo_km: float, hi_km: float) -> float:
    """Mean generated/real PSD ratio over the wavelength band ``[lo_km, hi_km]``.

    1.0 means the prior carries the right amount of variance at those scales,
    below 1 means it is too smooth there, above 1 too noisy.
    """
    lam = 1.0 / k
    sel = (lam >= lo_km) & (lam <= hi_km)
    if not sel.any():
        return float("nan")
    return float(np.mean(psd_gen[sel] / psd_real[sel]))


# ---------------------------------------------------------------------------
# geostrophy and energetics
# ---------------------------------------------------------------------------
def coriolis(lat_deg) -> np.ndarray:
    """``f = 2 Omega sin(lat)``. This domain is 30-40 N, so f is never near 0."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))


def geostrophic_uv(eta: np.ndarray, dx_m: float, lat2d
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Geostrophic velocity from sea surface height.

    ``u = -(g/f) d(eta)/dy``, ``v = (g/f) d(eta)/dx``, central differences on the
    uniform grid. ``lat2d`` is a scalar or an ``(H, W)`` latitude field; a 2-D f
    also lets the (g/f) factor carry the beta term, which is intended.

    Feed this the *anomaly*, not the denormalised full field: the gradient of the
    record time-mean is the mean Gulf Stream jet, and including it turns eddy
    kinetic energy into total kinetic energy.
    """
    f = coriolis(lat2d)
    deta_dy, deta_dx = np.gradient(np.asarray(eta, dtype=np.float64), dx_m)
    return -(G / f) * deta_dy, (G / f) * deta_dx


def erode_mask(mask: np.ndarray, pix: int = 2) -> np.ndarray:
    """Shrink ``mask`` by ``pix`` pixels in each direction.

    Needed wherever a spatial derivative is taken. The loader maps the store's
    17 non-finite pixels to 0.0, and ``np.gradient`` turns each of those into a
    spurious gradient spike across its neighbours -- 17 pixels is nothing for a
    domain mean but plenty to set the colour scale of an EKE map. Eroding also
    drops the frame edge, where ``np.gradient`` is one-sided.

    The default ``pix=2`` removes the NUMERICAL smear only, which is all the
    selftest below needs. It does NOT remove the physical land-edge rim: on
    gom_nemo, real geostrophic EKE binned by distance from the coast is still
    3x the far field at 3-4 px and 2x at 4-6 px, reaching background near 6 px.
    Anything that pools EKE near a coastline should pass 6 -- see
    ``GRAD_ERODE_PX`` in ``diagnostics.py``, which is where that measurement is
    written down.

    ``pix`` is a EUCLIDEAN radius: a pixel survives only if the nearest land is
    strictly more than ``pix`` away. This used to be four-neighbour erosion
    iterated ``pix`` times, which is erosion by a DIAMOND -- it clears the
    cardinal directions to ``pix`` but the diagonals only to ``pix/sqrt(2)``.
    At ``pix=6`` that left a ring of pixels 4-6 px from land whose mean EKE was
    still 0.075 against 0.032 at 8-10 px, i.e. the rim this function exists to
    remove, surviving in the corners of the diamond and drawing exactly the
    coastline outlines that motivated widening it in the first place.
    """
    m = np.asarray(mask) > 0.5
    # distance_transform_edt measures distance to the nearest zero; with no land
    # in this window there is nothing to erode away from, only the frame to trim.
    out = (m & (ndimage.distance_transform_edt(m) > pix)) if (~m).any() else m.copy()
    out[:pix, :] = False
    out[-pix:, :] = False
    out[:, :pix] = False
    out[:, -pix:] = False
    return out


def eke(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``0.5 (u^2 + v^2)`` -- eddy kinetic energy per unit mass, m^2 s^-2.

    "Eddy" is carried by the inputs: both the geostrophic velocity derived from
    an SSH anomaly and the stored ``uag``/``vag`` anomalies are already
    departures from the record time-mean.
    """
    return 0.5 * (np.asarray(u) ** 2 + np.asarray(v) ** 2)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def to_physical(a_norm: np.ndarray, std: float, clim: np.ndarray) -> np.ndarray:
    """Undo the anomaly normalisation: ``x = a * std + climatology``.

    ``clim`` is the reference-day climatology field written by ``gen_prior.py``.
    Both generated and real fields go through the same ``clim`` on purpose --
    see the module docstring of ``gen_prior.py``.
    """
    return a_norm * std + clim


def moments(vals: np.ndarray) -> tuple[float, float, float, float]:
    """``(mean, std, skewness, excess kurtosis)`` of a 1-D sample."""
    v = np.asarray(vals, dtype=np.float64).ravel()
    return (float(v.mean()), float(v.std()),
            float(stats.skew(v)), float(stats.kurtosis(v)))


def ocean_values(stack: np.ndarray, mask: np.ndarray, n_max: int = 200_000,
                 seed: int = 0) -> np.ndarray:
    """Flatten ocean pixels of ``stack`` (N, H, W), subsampled to ``n_max``.

    Subsampling keeps ``gaussian_kde`` tractable -- 48 full frames is 26 M
    pixels, and the KDE is O(n) per evaluation point.

    ``mask`` is either ``(H, W)``, applied to every sample, or ``(N, H, W)``,
    one mask per sample -- which is what the real side of a patch comparison
    needs, since each real crop carries its own land.
    """
    m = np.asarray(mask) > 0.5
    stack = np.asarray(stack)
    v = stack[m] if m.ndim == 3 else stack[:, m].ravel()
    v = v[np.isfinite(v)]
    if v.size > n_max:
        rng = np.random.default_rng(seed)
        v = rng.choice(v, size=n_max, replace=False)
    return v


def corr_matrix(stack: np.ndarray, mask: np.ndarray, n_max: int = 200_000,
                seed: int = 0) -> np.ndarray:
    """``(C, C)`` pixelwise correlation between channels of ``stack`` (N, C, H, W).

    Pixels are pooled across samples and locations, so this measures the
    *local* co-variation of the channels -- whether a positive SSH anomaly comes
    with the SST and velocity anomalies it should.

    ``mask`` is ``(H, W)`` or, for a real side whose samples each carry their
    own land, ``(N, H, W)``. Land must not reach this: it is an exact 0.0 in
    every channel at once, which reads as perfect correlation everywhere it
    appears and quietly pulls the real matrix toward the identity.
    """
    m = np.asarray(mask) > 0.5
    n, c = stack.shape[:2]
    stack = np.asarray(stack)
    if m.ndim == 3:
        v = np.stack([stack[:, j][m] for j in range(c)], axis=0)
    else:
        v = stack[:, :, m].transpose(1, 0, 2).reshape(c, -1)
    if v.shape[1] > n_max:
        rng = np.random.default_rng(seed)
        v = v[:, rng.choice(v.shape[1], size=n_max, replace=False)]
    return np.corrcoef(v)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _synthetic_powerlaw(n: int, beta: float, seed: int = 0) -> np.ndarray:
    """A random field whose isotropic PSD follows ``k**-beta``."""
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(freq, freq)
    kmag = np.sqrt(kx ** 2 + ky ** 2)
    kmag[0, 0] = 1.0
    amp = kmag ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=(n, n))
    return np.real(np.fft.ifft2(amp * np.exp(1j * phase)))


def _test_psd_slope() -> bool:
    """radial_psd recovers a prescribed spectral slope."""
    n, dx_km, beta = 256, 1.81833, 3.0
    # Average a few realisations: a single random field's PSD is chi-squared
    # noisy bin by bin, which the fit would otherwise have to absorb.
    acc = None
    for s in range(8):
        k, p = radial_psd(_synthetic_powerlaw(n, beta, seed=s), dx_km)
        acc = p if acc is None else acc + p
    p = acc / 8

    # Fit away from both ends: the largest scales have too few Fourier modes per
    # bin and the last octave before Nyquist is where the Hann window's skirt
    # sits.
    lam = 1.0 / k
    sel = (lam > 8 * dx_km) & (lam < n * dx_km / 6)
    slope = np.polyfit(np.log(k[sel]), np.log(p[sel]), 1)[0]
    ok = abs(slope + beta) < 0.3
    print(f"  [psd ] fitted slope {slope:+.3f} vs expected {-beta:+.3f} "
          f"over {lam[sel].min():.1f}-{lam[sel].max():.1f} km  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def _test_geostrophy() -> bool:
    """geostrophic_uv reproduces the store's own ug/vg from its own ssh.

    Reads 4 hourly frames from one member store -- a few hundred MB, not the
    66 GB record. Uses the *full* ssh field (not an anomaly) because that is what
    the stored ug/vg are derived from.
    """
    from diffusion.config import load_config
    from diffusion.dataset_spec import resolve_spec

    cfg = load_config("diffusion/configs/prior_gulfstream.yaml")
    spec = resolve_spec(cfg)
    fam, cad = spec.family, spec.base
    dx_m = spec.dx_m()
    lat2d = np.asarray(fam.coord("coords/lat"), dtype=np.float64)

    idx = np.array([2200, 2300, 2400, 2500])          # mid-April, one store
    ssh = fam.take(cad, "ssh", idx)
    ug = fam.take(cad, "ug", idx)
    vg = fam.take(cad, "vg", idx)

    # Compare only where everything is finite. The store carries 17 NaN pixels
    # in ssh and 35-49 in ug/vg, and np.gradient smears ssh's across their
    # neighbours -- without this every correlation below comes out NaN.
    cu, cv = [], []
    for i in range(len(idx)):
        u, v = geostrophic_uv(ssh[i], dx_m, lat2d)
        good = (erode_mask(np.isfinite(ssh[i]), 2)
                & np.isfinite(ug[i]) & np.isfinite(vg[i])
                & np.isfinite(u) & np.isfinite(v))
        cu.append(np.corrcoef(u[good], ug[i][good])[0, 1])
        cv.append(np.corrcoef(v[good], vg[i][good])[0, 1])
    cu, cv = float(np.mean(cu)), float(np.mean(cv))
    ok = cu > 0.95 and cv > 0.95
    print(f"  [geo ] dx={dx_m:.2f} m  lat {lat2d.min():.2f}-{lat2d.max():.2f} N  "
          f"corr(u,ug)={cu:.4f}  corr(v,vg)={cv:.4f}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok and (cu < -0.95 or cv < -0.95):
        print("  [geo ] NOTE: correlation is strongly NEGATIVE -- sign convention "
              "is inverted, every EKE number would still look plausible.")
    return ok


def _test_velocity_decomposition() -> bool:
    """The store's ``u`` really is ``ug + uag`` (and likewise for v).

    fig4's "total EKE" adds the geostrophic velocity implied by the generated
    ssh to the generated uag/vag channels. That is only the total velocity if the
    store defines its own decomposition that way -- worth checking rather than
    assuming, since nothing would crash if it did not.
    """
    from diffusion.config import load_config
    from diffusion.dataset_spec import resolve_spec

    cfg = load_config("diffusion/configs/prior_gulfstream.yaml")
    spec = resolve_spec(cfg)
    fam, cad = spec.family, spec.base
    idx = np.array([1000, 2200, 3800])
    u, v, ug, vg, uag, vag = (fam.take(cad, nm, idx)
                              for nm in ("u", "v", "ug", "vg", "uag", "vag"))
    good = np.all([np.isfinite(a) for a in (u, v, ug, vg, uag, vag)], axis=0)
    ok = True
    for name, a, b, c in (("u", u, ug, uag), ("v", v, vg, vag)):
        rel = (a[good] - (b[good] + c[good])).std() / a[good].std()
        ok &= rel < 5e-3
        print(f"  [vel ] {name} = {name}g + {name}ag to {rel:.2e} of its own rms "
              f"-> {'PASS' if rel < 5e-3 else 'FAIL'}")
    return ok


def _selftest() -> int:
    print("[kernels] selftest")
    results = [_test_psd_slope(), _test_geostrophy(),
               _test_velocity_decomposition()]
    ok = all(results)
    print(f"[kernels] {'all checks passed' if ok else 'FAILURES -- do not trust the diagnostics'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not args.selftest:
        ap.error("nothing to do; pass --selftest")
    sys.exit(_selftest())
