"""Paired-ensemble numerics for the CONDITIONAL evaluation: CRPS, rank
histograms, spread-skill, cross-spectra/coherence, a leakage-quantified band-pass,
and a synthetic-npz writer for end-to-end tests without a store.

Pure numpy/scipy, no torch, no store. ``eval.kernels`` is imported for the
spectral primitives it already gets right (``radial_psd`` binning, tiling,
erosion, moments); nothing else in ``eval/`` is imported, by design -- these
scripts are a parallel layer beside the unconditional one, not an extension of
it.

Every function takes ONE target step: members ``(K, H, W)`` and truth ``(H, W)``.
The caller streams over days, so nothing here ever allocates ``(D, K, H, W)``.

Run ``python -m eval.kernels_cond --selftest`` before trusting any number that
comes out of ``eval.diagnostics_cond``. CRPS, rank histograms and coherence all
fail QUIETLY -- a wrong normalisation produces a plausible number, not a crash
-- so each is pinned here against a closed form.

``--make-synthetic <path>`` writes a samples_cond npz whose truth is exchangeable
with its members, so the paired metrics have known answers (spread-skill 1, rank
histogram uniform, fair CRPS = sigma/sqrt(pi) for Gaussian noise).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage, stats

from . import kernels as K


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------
def crps_ensemble(x: np.ndarray, y: np.ndarray, fair: bool = True) -> np.ndarray:
    """Pixelwise CRPS of an ensemble ``x`` (K, ...) against truth ``y`` (...).

    ``CRPS = E|X - y| - 1/2 E|X - X'|``. The second term is computed from the
    SORTED members as ``sum_i (2i - K - 1) x_(i)`` -- equal to
    ``sum_{i<j} (x_(j) - x_(i))`` -- which is O(K log K) and never forms a
    (K, K, ...) pairwise array.

    ``fair=True`` (Ferro 2014) divides the pairwise term by ``K (K - 1)`` so the
    estimator is unbiased for the CRPS of the distribution the members were
    drawn from; ``fair=False`` is the plain PWM/ensemble form (``K^2``), which
    is biased HIGH by the sampling of a finite ensemble. Use the fair form to
    compare ensembles of different size; needs ``K >= 2``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = x.shape[0]
    if fair and k < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    term1 = np.abs(x - y).mean(axis=0)
    xs = np.sort(x, axis=0)
    i = np.arange(1, k + 1, dtype=np.float64).reshape((-1,) + (1,) * (x.ndim - 1))
    gini = ((2.0 * i - k - 1.0) * xs).sum(axis=0)
    denom = k * (k - 1) if fair else k * k
    return term1 - gini / denom


def _crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of N(mu, sigma^2) against a scalar observation."""
    z = (y - mu) / sigma
    return float(sigma * (z * (2 * stats.norm.cdf(z) - 1)
                          + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)))


# ---------------------------------------------------------------------------
# rank histogram
# ---------------------------------------------------------------------------
def rank_of_truth(x: np.ndarray, y: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Rank of the truth among the members, ``0..K`` (K+1 possible bins).

    Ties are split uniformly at random: fp16 storage and exact zeros produce
    them, and counting a tie as "below" would pile the histogram into bin 0.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    lt = (x < y).sum(axis=0)
    eq = (x == y).sum(axis=0)
    return lt + rng.integers(0, eq + 1)


def rank_hist_stats(counts: np.ndarray) -> dict:
    """``tv`` (total variation from uniform), ``tails_ratio``, ``slope``.

    TV is what the old decision gate thresholded at 0.20, but it has no sign.
    ``tails_ratio`` = mass in the two outer bins / uniform expectation: above 1
    the truth falls outside the ensemble too often (under-dispersed), below 1
    the ensemble is too wide. ``slope`` is the linear trend of the histogram
    across the rank axis, in units of the uniform density per unit of
    normalised rank: positive means the truth sits above the members (a low
    bias in the ensemble), negative a high bias.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    nb = c.size
    if n <= 0 or nb < 2:
        return {"tv": np.nan, "tails_ratio": np.nan, "slope": np.nan}
    p = c / n
    u = 1.0 / nb
    tv = 0.5 * float(np.abs(p - u).sum())
    tails = float((p[0] + p[-1]) / (2 * u))
    xr = np.linspace(-0.5, 0.5, nb)
    slope = float(np.polyfit(xr, p / u, 1)[0])
    return {"tv": tv, "tails_ratio": tails, "slope": slope}


# ---------------------------------------------------------------------------
# spread / skill
# ---------------------------------------------------------------------------
def ens_mean_var(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble mean and unbiased (ddof=1) variance over the member axis."""
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0), x.var(axis=0, ddof=1)


def spread_skill_ratio(mean_var: float, mse_ensmean: float, k: int) -> float:
    """Fortin et al. (2014) calibration ratio; 1.0 is calibrated for ANY K.

    For an exchangeable ensemble ``E[var_ddof1] = sigma^2`` and the ensemble
    mean's MSE is ``sigma^2 (K + 1) / K``, so the naive spread/skill ratio sits
    below 1 by ``sqrt(K / (K + 1))`` even when nothing is wrong. The ``(K+1)/K``
    factor removes that.
    """
    if not (mse_ensmean > 0):
        return float("nan")
    return float(np.sqrt(mean_var * (k + 1) / k / mse_ensmean))


def member_over_ensmean_target(k: int) -> float:
    """Calibrated ratio ``RMSE(member) / RMSE(ensemble mean)`` = ``sqrt(2K/(K+1))``."""
    return float(np.sqrt(2.0 * k / (k + 1)))


# ---------------------------------------------------------------------------
# cross-spectra and coherence
# ---------------------------------------------------------------------------
def _radial_bins(n: int, dx_km: float):
    """The binning ``eval.kernels.radial_psd`` uses, exposed so a cross-spectrum
    lands on IDENTICAL wavenumber bins."""
    freq = np.fft.fftfreq(n, d=dx_km)
    kx, ky = np.meshgrid(freq, freq)
    kmag = np.sqrt(kx ** 2 + ky ** 2).ravel()
    kmax = float(freq.max())
    edges = np.linspace(0.0, kmax, n // 2 + 1)
    which = np.digitize(kmag, edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return which, centers, len(edges)


def _bin_mean(p_flat: np.ndarray, which: np.ndarray, nedges: int) -> np.ndarray:
    return np.array([p_flat[which == b].mean() if np.any(which == b) else np.nan
                     for b in range(1, nedges)])


def radial_cross_psd(a: np.ndarray, b: np.ndarray, dx_km: float):
    """``(k, Paa, Pbb, Pab)`` on the same bins as ``kernels.radial_psd``.

    Same mean removal, 2-D Hann window and window-energy normalisation, so
    ``Paa`` equals ``radial_psd(a)`` to rounding (the selftest asserts it) and
    ``Pab`` is the complex cross-spectrum that ``coherence`` needs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError(f"radial_cross_psd expects two equal square tiles, got "
                         f"{a.shape} and {b.shape}")
    win1d = np.hanning(n)
    win = np.outer(win1d, win1d)
    norm = np.sum(win ** 2)
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    paa = (np.abs(fa) ** 2 / norm).ravel()
    pbb = (np.abs(fb) ** 2 / norm).ravel()
    pab = (fa * np.conj(fb) / norm).ravel()
    which, centers, nedges = _radial_bins(n, dx_km)
    baa = _bin_mean(paa, which, nedges)
    bbb = _bin_mean(pbb, which, nedges)
    bab = _bin_mean(pab.real, which, nedges) + 1j * _bin_mean(pab.imag, which, nedges)
    keep = np.isfinite(baa) & (centers > 0)
    return centers[keep], baa[keep], bbb[keep], bab[keep]


def mean_radial_cross(stack_a: np.ndarray, stack_b: np.ndarray, mask: np.ndarray,
                      dx_km: float, tile: int = 256, stride: int = 128,
                      min_frac: float = 0.999):
    """Tile-averaged ``(k, Paa (N,nk), Pbb (N,nk), Pab (N,nk) complex)``.

    ``stack_a`` is ``(N, H, W)``; ``stack_b`` is ``(N, H, W)`` or a single
    ``(H, W)`` field paired with every sample (the truth against K members).
    Tiling follows ``kernels.mean_radial_psd`` exactly.
    """
    sa = np.asarray(stack_a, dtype=np.float64)
    sb = np.asarray(stack_b, dtype=np.float64)
    n, ny, nx = sa.shape
    if sb.ndim == 2:
        sb = np.broadcast_to(sb, sa.shape)
    if ny < tile or nx < tile:
        tile = min(ny, nx)
        pos = [(0, 0)]
    else:
        pos = K.tile_positions(mask, tile, stride, min_frac)
        if not pos:
            raise ValueError(f"no {tile}x{tile} tile in a {ny}x{nx} domain reaches "
                             f"{min_frac:.3f} ocean fraction")
    k = None
    raa, rbb, rab = [], [], []
    for i in range(n):
        acc = None
        for (y0, x0) in pos:
            kk, paa, pbb, pab = radial_cross_psd(sa[i, y0:y0 + tile, x0:x0 + tile],
                                                 sb[i, y0:y0 + tile, x0:x0 + tile], dx_km)
            acc = [paa, pbb, pab] if acc is None else [acc[0] + paa, acc[1] + pbb,
                                                        acc[2] + pab]
            k = kk
        raa.append(acc[0] / len(pos))
        rbb.append(acc[1] / len(pos))
        rab.append(acc[2] / len(pos))
    return k, np.asarray(raa), np.asarray(rbb), np.asarray(rab)


def coherence(sum_pab: np.ndarray, sum_paa: np.ndarray,
              sum_pbb: np.ndarray) -> np.ndarray:
    """Magnitude-squared coherence ``|<Pab>|^2 / (<Paa><Pbb>)`` per wavenumber.

    The sums (or means) must be over several tiles/samples: a single tile's
    coherence is identically 1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.abs(sum_pab) ** 2 / (np.real(sum_paa) * np.real(sum_pbb))
    return np.clip(np.where(np.isfinite(g), g, np.nan), 0.0, 1.0)


def coherence_half_wavelength(k: np.ndarray, gamma2: np.ndarray) -> float:
    """The finest scale (km) still coherent: the wavelength where coherence
    last crosses 0.5, scanning from the largest scales down.

    Defined from the FINEST bin with ``gamma2 >= 0.5`` (interpolated
    log-linearly to the next, incoherent bin) rather than from the first
    crossing, so a single noisy largest-scale bin -- which averages only a
    handful of Fourier modes -- cannot zero out the statistic. NaN if no bin is
    coherent; the Nyquist wavelength if every bin is."""
    lam = 1.0 / np.asarray(k)
    order = np.argsort(-lam)
    lam, g = lam[order], np.asarray(gamma2)[order]
    ok = np.isfinite(g)
    lam, g = lam[ok], g[ok]
    coh = np.flatnonzero(g >= 0.5)
    if g.size == 0 or coh.size == 0:
        return float("nan")
    j = int(coh[-1])
    if j == g.size - 1:
        return float(lam[-1])
    l0, l1, g0, g1 = np.log(lam[j]), np.log(lam[j + 1]), g[j], g[j + 1]
    t = (g0 - 0.5) / max(g0 - g1, 1e-12)
    return float(np.exp(l0 + t * (l1 - l0)))


def band_power(k: np.ndarray, psd: np.ndarray, lo_km: float, hi_km: float) -> float:
    """Variance carried by the wavelength band ``[lo_km, hi_km]``: the annulus
    integral ``sum psd(k) k dk`` over the bins in the band (Parseval, up to a
    constant shared by every field on the same grid). NaN if no bin falls in
    the band."""
    k = np.asarray(k, dtype=np.float64)
    lam = 1.0 / k
    sel = (lam >= lo_km) & (lam <= hi_km) & np.isfinite(psd)
    if not sel.any():
        return float("nan")
    dk = float(np.median(np.diff(k))) if k.size > 1 else 1.0
    return float(np.sum(np.asarray(psd)[sel] * k[sel] * dk))


def log_distance(k, pg, pr, lo_km, hi_km) -> float:
    """RMS of log10(pg/pr) over a wavelength band -- 0 is a perfect match.
    (Same definition as ``eval.step_trajectory.log_distance``.)"""
    lam = 1.0 / k
    sel = (lam >= lo_km) & (lam <= hi_km) & (pg > 0) & (pr > 0)
    if not sel.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.log10(pg[sel] / pr[sel]) ** 2)))


# ---------------------------------------------------------------------------
# difference-of-Gaussians band-pass (pointwise band metrics, opt-in)
# ---------------------------------------------------------------------------
def _gauss_lp(sigma_km: float, k: np.ndarray) -> np.ndarray:
    return np.exp(-2.0 * np.pi ** 2 * sigma_km ** 2 * k ** 2)


def dog_sigmas(lo_km: float, hi_km: float, iters: int = 40) -> tuple[float, float]:
    """``(sigma_small, sigma_large)`` in km such that the DoG transfer
    ``G(k) = LP(sigma_small) - LP(sigma_large)`` equals exactly 0.5 at BOTH band
    edges. Solved by fixed-point iteration from the single-Gaussian half-power
    widths; falls back to those if the band is too narrow to admit a solution."""
    k_hi, k_lo = 1.0 / lo_km, 1.0 / hi_km             # k_hi > k_lo
    c = np.sqrt(np.log(2.0)) / (np.sqrt(2.0) * np.pi)
    s_s, s_l = c / k_hi, c / k_lo
    for _ in range(iters):
        # LP_s(k_hi) = 0.5 + LP_l(k_hi)
        t = 0.5 + _gauss_lp(s_l, k_hi)
        if not (0 < t < 1):
            break
        s_s_new = np.sqrt(-np.log(t) / (2 * np.pi ** 2 * k_hi ** 2))
        # LP_l(k_lo) = LP_s(k_lo) - 0.5
        t2 = _gauss_lp(s_s_new, k_lo) - 0.5
        if not (0 < t2 < 1):
            break
        s_l_new = np.sqrt(-np.log(t2) / (2 * np.pi ** 2 * k_lo ** 2))
        if abs(s_s_new - s_s) < 1e-9 and abs(s_l_new - s_l) < 1e-9:
            s_s, s_l = s_s_new, s_l_new
            break
        s_s, s_l = s_s_new, s_l_new
    return float(s_s), float(s_l)


def dog_transfer(dx_km: float, lo_km: float, hi_km: float, n: int = 512):
    """``(k, G)``: the analytic amplitude transfer function of ``dog_bandpass``
    on a grid of ``n`` wavenumbers up to Nyquist. Quote this next to any band
    metric derived from the filtered fields -- it says exactly what leaks."""
    s_s, s_l = dog_sigmas(lo_km, hi_km)
    k = np.linspace(1e-6, 0.5 / dx_km, n)
    return k, _gauss_lp(s_s, k) - _gauss_lp(s_l, k)


def dog_bandpass(field: np.ndarray, mask: np.ndarray, dx_km: float,
                 lo_km: float, hi_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass ``field`` (..., H, W) to wavelengths ``[lo_km, hi_km]``.

    Difference of two Gaussians whose half-power points sit exactly on the band
    edges (see ``dog_sigmas``). Each Gaussian is applied as a NORMALISED
    convolution -- ``filter(f * m) / filter(m)`` -- so land pixels (mask 0)
    contribute nothing and no coastline step is ever smeared into the ocean.
    The frame edge is handled the same way (zero padding of both numerator and
    denominator).

    Returns ``(filtered, valid)`` where ``valid`` is the mask eroded by
    ``ceil(3 sigma_large / dx)``: outside it the larger Gaussian's support is
    partly land/frame and the normalisation is doing real work.
    """
    f = np.asarray(field, dtype=np.float64)
    m = (np.asarray(mask) > 0.5).astype(np.float64)
    s_s, s_l = dog_sigmas(lo_km, hi_km)
    ps, pl = s_s / dx_km, s_l / dx_km
    fm = f * m
    out = np.empty_like(f)
    for idx in np.ndindex(f.shape[:-2]):
        num_s = ndimage.gaussian_filter(fm[idx], ps, mode="constant", cval=0.0)
        num_l = ndimage.gaussian_filter(fm[idx], pl, mode="constant", cval=0.0)
        den_s = ndimage.gaussian_filter(m, ps, mode="constant", cval=0.0)
        den_l = ndimage.gaussian_filter(m, pl, mode="constant", cval=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp_s = np.where(den_s > 1e-6, num_s / den_s, 0.0)
            lp_l = np.where(den_l > 1e-6, num_l / den_l, 0.0)
        out[idx] = lp_s - lp_l
    valid = K.erode_mask(m, int(np.ceil(3.0 * pl)))
    return out, valid


# ---------------------------------------------------------------------------
# synthetic samples_cond npz
# ---------------------------------------------------------------------------
def _smooth_field(rng: np.random.Generator, h: int, w: int, beta: float) -> np.ndarray:
    """Zero-mean, unit-std power-law field of shape (h, w)."""
    n = max(h, w)
    a = K._synthetic_powerlaw(n, beta, seed=int(rng.integers(1 << 31)))[:h, :w]
    a = a - a.mean()
    return a / max(a.std(), 1e-12)


def synthetic_cond_npz(path: str, D: int = 6, K_: int = 8, H: int = 128, W: int = 128,
                       vars: tuple[str, ...] = ("ssh", "sst", "chl"),
                       noise: float = 1.0, step: int = 1_000_000,
                       sampler: dict | None = None, cobs: int = 1,
                       coverage: float = 0.4, seed: int = 0,
                       underdispersed: float | None = None,
                       extras: tuple[str, ...] = ("ocean_mask", "doy_sin", "doy_cos"),
                       doy_signal: float = 0.0, patch_lattice: bool = False,
                       dx_km: float = 1.81833, gen_dtype: str = "float16",
                       run_name: str = "synthetic_run") -> str:
    """Write a ``samples_cond_<size>.npz`` with KNOWN answers.

    Truth = predictable part + unpredictable part; each member = the same
    predictable part + its own draw of the unpredictable part with std
    ``noise`` (times ``underdispersed`` for the members only, if given). So with
    ``underdispersed=None`` the truth is exchangeable with the members: the
    rank histogram is uniform, the Fortin spread-skill ratio is 1, the fair
    CRPS is ``noise * std / sqrt(pi)`` for Gaussian noise (the noise here is a
    power-law field, so that holds approximately), and
    ``RMSE(member)/RMSE(mean) = sqrt(2K/(K+1))`` exactly in expectation.

    ``doy_signal`` adds ``A sin(2 pi doy / 365)`` to the domain mean of truth
    AND members, the signal a mask+doy model is supposed to carry.

    ``patch_lattice=True`` writes patch geometry: a (H+64, W+64) synthetic
    domain with land, and one stride-32 lattice crop per sample, so ``pos``
    varies and every per-sample mask is different.
    """
    rng = np.random.default_rng(seed)
    ct = len(vars)
    std = np.array([0.1, 1.5, 0.3, 0.05, 0.05, 0.02, 0.02][:ct], dtype=np.float64)
    clim0 = np.array([0.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0][:ct], dtype=np.float32)
    is_log = np.array([v.lower().startswith("chl") for v in vars])

    # -- domain, land, lattice ----------------------------------------------
    if patch_lattice:
        ny, nx = H + 64, W + 64
    else:
        ny, nx = H, W
    mask_full = np.ones((ny, nx), dtype=np.float32)
    mask_full[:6, :] = 0.0                                   # a coastal strip
    yy, xx = np.mgrid[:ny, :nx]
    # An island. In patch mode it sits in the far corner so that some lattice
    # crops are land-free (the spectra need whole rectangles) while others are
    # not (the per-sample masks need exercising).
    cy, cx = (0.9, 0.9) if patch_lattice else (0.62, 0.35)
    mask_full[(yy - ny * cy) ** 2 + (xx - nx * cx) ** 2 < (min(ny, nx) * 0.07) ** 2] = 0.0
    lat_full = (25.0 + 8.0 * yy / max(ny - 1, 1)).astype(np.float32)
    if patch_lattice:
        lattice = [(y0, x0) for y0 in range(0, ny - H + 1, 32)
                   for x0 in range(0, nx - W + 1, 32)]
        pos = np.array([lattice[i] for i in rng.integers(len(lattice), size=D)],
                       dtype=np.int64)
        size = str(H)
    else:
        pos = np.zeros((0, 2), dtype=np.int64)
        size = "full"

    # -- time axis -----------------------------------------------------------
    t0 = 1.5e9
    days = np.linspace(20, 20 + 365 * 1.5, D).round().astype(np.int64)
    time_unix = t0 + days * 86400.0
    doy = (np.mod(time_unix, 365.2425 * 86400.0) / 86400.0).astype(np.float32)
    ang = 2 * np.pi * doy / 365.2425

    gen = np.zeros((D, K_, ct, H, W), dtype=np.float32)
    truth = np.zeros((D, ct, H, W), dtype=np.float32)
    cond = np.zeros((D, cobs + len(extras), H, W), dtype=np.float32)
    obs_avail = np.zeros((D, cobs, H, W), dtype=bool)
    baseline = np.full((D, ct, H, W), np.nan, dtype=np.float32)
    mask = np.zeros((D, H, W), dtype=bool)
    lat = np.zeros((D, H, W), dtype=np.float32)
    mem_sigma = noise * (underdispersed if underdispersed else 1.0)

    for d in range(D):
        y0, x0 = (pos[d] if patch_lattice else (0, 0))
        m = mask_full[y0:y0 + H, x0:x0 + W] > 0.5
        mask[d] = m
        lat[d] = lat_full[y0:y0 + H, x0:x0 + W]
        offset = doy_signal * np.sin(ang[d])
        for ci in range(ct):
            base = _smooth_field(rng, H, W, 3.0)
            tr = base + noise * _smooth_field(rng, H, W, 2.0) + offset
            truth[d, ci] = np.where(m, tr, 0.0)
            for j in range(K_):
                mem = base + mem_sigma * _smooth_field(rng, H, W, 2.0) + offset
                # the model draws whatever it likes on land; a small residue
                mem = np.where(m, mem, 0.3 * mem_sigma * rng.standard_normal((H, W)))
                gen[d, j, ci] = mem
        # observation channels: truth of var j + smooth error, cloud gaps
        for j in range(cobs):
            ci = j % ct
            cloud = _smooth_field(rng, H, W, 3.0)
            avail = (cloud < np.quantile(cloud, coverage)) & m
            obs = truth[d, ci] + 0.5 * noise * _smooth_field(rng, H, W, 2.5)
            cond[d, j] = np.where(avail, obs, 0.0)
            obs_avail[d, j] = avail
            if j == 0:
                phys = obs * std[ci] + clim0[ci]
                baseline[d, ci] = np.where(avail, phys, np.nan)
        e = cobs
        for name in extras:
            if name == "ocean_mask":
                cond[d, e] = m.astype(np.float32)
            elif name == "doy_sin":
                cond[d, e] = np.sin(ang[d])
            elif name == "doy_cos":
                cond[d, e] = np.cos(ang[d])
            e += 1

    cond_vars = [f"obs_{vars[j % ct]}" for j in range(cobs)]
    bmap = {v: None for v in vars}
    bkind = {v: "absolute" for v in vars}
    if cobs > 0:
        bmap[vars[0]] = cond_vars[0]
    truth_phys0 = truth[:, 0] * std[0] + clim0[0]
    bias = np.full(ct, np.nan)
    if cobs > 0:
        bias[0] = float(np.nanmean(baseline[:, 0] - truth_phys0))
    sampler = sampler or dict(num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                              s_churn=10.0, s_noise=1.0, s_tmin=0.0, s_tmax=float("inf"))
    meta = dict(
        ckpt=f"/synthetic/runs/{run_name}/checkpoints/ckpt_step{step:07d}.pt",
        step=int(step), weights="ema", val_loss=None, best_val=None, split="val",
        seed=seed, D=D, K=K_, member_batch=4, sampler=sampler, store_cond="full",
        gen_dtype=gen_dtype, skip_tail=0, dx_m=dx_km * 1000.0,
        lat_note="synthetic linear latitude", lattice_note=("stride-32 synthetic lattice"
                                                              if patch_lattice else "full frame"),
        pos_mode="lattice" if patch_lattice else "full", target=list(vars),
        cond_vars=cond_vars, k_days=1, extras=list(extras), baseline_map=bmap,
        baseline_kind=bkind, norm_mode="zscore", dataset="synthetic",
        base_cadence="daily", step_seconds=86400.0, timings={},
        synthetic=dict(noise=noise, underdispersed=underdispersed, doy_signal=doy_signal),
    )
    payload = dict(
        vars=json.dumps(list(vars)), cond_vars=json.dumps(cond_vars),
        k_days=np.int64(1), extras=json.dumps(list(extras)),
        gen_norm=gen.astype(np.float16 if gen_dtype == "float16" else np.float32),
        truth_norm=truth, cond_norm=cond,
        cond_channel_names=json.dumps([f"{c}@t-0" for c in cond_vars] + list(extras)),
        obs_avail=obs_avail,
        obs_age_s=rng.integers(0, 24, size=(D, cobs)).astype(np.float64) * 3600.0,
        baseline_phys=baseline, baseline_map=json.dumps(bmap),
        baseline_kind=json.dumps(bkind), baseline_bias=bias,
        clim_day=np.broadcast_to(clim0[None, :, None, None], (D, ct, 1, 1)).astype(np.float32),
        std=std, is_log=is_log, mask=mask, mask_full=mask_full, lat=lat, lat_full=lat_full,
        dx_m=np.float64(dx_km * 1000.0), days=days, time_unix=time_unix, doy=doy,
        pos=pos, size=size, config=json.dumps({"synthetic": True, "target": list(vars)}),
        meta=json.dumps(meta),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _report(tag: str, ok: bool, msg: str) -> bool:
    print(f"  [{tag:<5}] {msg}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _test_crps_identity() -> bool:
    rng = np.random.default_rng(0)
    k, n = 5, 2000
    x = rng.standard_normal((k, n))
    y = rng.standard_normal(n)
    pair = np.abs(x[:, None, :] - x[None, :, :]).sum((0, 1))     # sum_ij |xi - xj|
    t1 = np.abs(x - y).mean(0)
    ref_fair = t1 - pair / (2 * k * (k - 1))
    ref_pwm = t1 - pair / (2 * k * k)
    e1 = np.abs(crps_ensemble(x, y, fair=True) - ref_fair).max()
    e2 = np.abs(crps_ensemble(x, y, fair=False) - ref_pwm).max()
    return _report("crps", e1 < 1e-10 and e2 < 1e-10,
                   f"sorted form vs brute force: max err fair {e1:.1e}, pwm {e2:.1e}")


def _test_crps_closed_form() -> bool:
    rng = np.random.default_rng(1)
    k, n, y = 8, 200_000, 0.7
    x = rng.standard_normal((k, n))
    got = float(crps_ensemble(x, np.full(n, y), fair=True).mean())
    ref = _crps_gaussian(0.0, 1.0, y)
    ok = abs(got / ref - 1) < 0.01
    r1 = _report("crps", ok, f"fixed obs: fair CRPS {got:.4f} vs closed form {ref:.4f}")
    sigma = 1.7
    xs = sigma * rng.standard_normal((k, n))
    ys = sigma * rng.standard_normal(n)
    fair = float(crps_ensemble(xs, ys, fair=True).mean())
    pwm = float(crps_ensemble(xs, ys, fair=False).mean())
    ref2 = sigma / np.sqrt(np.pi)
    ok2 = abs(fair / ref2 - 1) < 0.01 and pwm > fair
    r2 = _report("crps", ok2, f"exchangeable: fair {fair:.4f} vs sigma/sqrt(pi) {ref2:.4f}; "
                              f"pwm {pwm:.4f} (biased high)")
    return r1 and r2


def _test_rank_hist() -> bool:
    rng = np.random.default_rng(2)
    k, n = 8, 200_000
    x = rng.standard_normal((k, n))
    y = rng.standard_normal(n)
    r = rank_of_truth(x, y, rng)
    st = rank_hist_stats(np.bincount(r, minlength=k + 1))
    ok1 = st["tv"] < 0.02 and 0.9 < st["tails_ratio"] < 1.1
    a = _report("rank", ok1, f"exchangeable: TV {st['tv']:.4f}, tails {st['tails_ratio']:.3f}")
    r_under = rank_of_truth(0.5 * x, y, rng)
    st_u = rank_hist_stats(np.bincount(r_under, minlength=k + 1))
    r_over = rank_of_truth(2.0 * x, y, rng)
    st_o = rank_hist_stats(np.bincount(r_over, minlength=k + 1))
    ok2 = st_u["tails_ratio"] > 2 and st_o["tails_ratio"] < 0.5
    b = _report("rank", ok2, f"members sigma/2: tails {st_u['tails_ratio']:.2f} (>2); "
                             f"2 sigma: {st_o['tails_ratio']:.2f} (<0.5)")
    xt = np.broadcast_to(y, (k, n)).copy()
    r_tie = rank_of_truth(xt, y, rng)
    st_t = rank_hist_stats(np.bincount(r_tie, minlength=k + 1))
    c = _report("rank", st_t["tv"] < 0.02, f"all-tie members: TV {st_t['tv']:.4f} (uniform split)")
    return a and b and c


def _test_spread_skill() -> bool:
    rng = np.random.default_rng(3)
    k, n = 8, 200_000
    x = rng.standard_normal((k, n))
    y = rng.standard_normal(n)
    mean, var = ens_mean_var(x)
    mse = float(((mean - y) ** 2).mean())
    ratio = spread_skill_ratio(float(var.mean()), mse, k)
    ok1 = 0.98 < ratio < 1.02
    a = _report("sprd", ok1, f"exchangeable Fortin spread/skill {ratio:.4f}")
    rm = float(np.sqrt(((x - y) ** 2).mean()))
    got = rm / np.sqrt(mse)
    tgt = member_over_ensmean_target(k)
    ok2 = abs(got / tgt - 1) < 0.02
    b = _report("sprd", ok2, f"rmse member / rmse mean {got:.4f} vs sqrt(2K/(K+1)) {tgt:.4f}")
    return a and b


def _test_cross_psd() -> bool:
    n, dx = 128, 1.81833
    a = K._synthetic_powerlaw(n, 3.0, seed=5)
    b = K._synthetic_powerlaw(n, 3.0, seed=6)
    k0, p0 = K.radial_psd(a, dx)
    k1, paa, pbb, pab = radial_cross_psd(a, a, dx)
    ok1 = np.allclose(k0, k1) and np.allclose(p0, paa, rtol=1e-12, atol=0)
    r1 = _report("xpsd", ok1, f"Paa equals kernels.radial_psd: max rel err "
                              f"{np.max(np.abs(paa / p0 - 1)):.1e}")
    g_self = coherence(pab, paa, pbb)
    ok2 = np.allclose(g_self, 1.0, atol=1e-9)
    r2 = _report("xpsd", ok2, f"coherence(a, a) = 1 at every bin (min {g_self.min():.6f})")
    acc = None
    for s in range(32):
        aa = K._synthetic_powerlaw(n, 3.0, seed=100 + s)
        bb = K._synthetic_powerlaw(n, 3.0, seed=200 + s)
        _, paa, pbb, pab = radial_cross_psd(aa, bb, dx)
        acc = [paa, pbb, pab] if acc is None else [acc[0] + paa, acc[1] + pbb, acc[2] + pab]
    g_ind = coherence(acc[2], acc[0], acc[1])
    ok3 = float(np.nanmean(g_ind)) < 0.1
    r3 = _report("xpsd", ok3, f"coherence of independent fields over 32 tiles: mean "
                              f"{np.nanmean(g_ind):.3f} (<0.1)")
    return r1 and r2 and r3


def _test_band_power_parseval() -> bool:
    rng = np.random.default_rng(7)
    n, dx = 256, 1.0
    lo, hi = 10.0, 40.0
    fr, fa = 0.0, 0.0
    for _ in range(8):
        k, p = K.radial_psd(rng.standard_normal((n, n)), dx)
        fr += band_power(k, p, lo, hi)
        fa += band_power(k, p, 2 * dx, 1e9)
    got = fr / fa
    kmin, kmax = 1.0 / (n * dx), 0.5 / dx
    ref = ((1 / lo) ** 2 - (1 / hi) ** 2) / (kmax ** 2 - kmin ** 2)
    ok = abs(got / ref - 1) < 0.05
    return _report("band", ok, f"white-noise band fraction {got:.4f} vs annulus area {ref:.4f}")


def _test_dog() -> bool:
    dx, lo, hi = 1.81833, 10.0, 40.0
    k, g = dog_transfer(dx, lo, hi)
    g_edges = np.interp([1.0 / hi, 1.0 / lo], k, g)
    ok1 = np.all(np.abs(g_edges - 0.5) < 0.02)
    r1 = _report("dog", ok1, f"transfer at band edges {g_edges[0]:.3f}, {g_edges[1]:.3f} (0.5)")
    leak = np.interp([0.25 / hi, 4.0 / lo], k, g)
    ok2 = np.all(np.abs(leak) < 0.05)
    r2 = _report("dog", ok2, f"leakage a factor 4 outside the band: {leak[0]:.3f}, {leak[1]:.3f}")
    # spatial filter reproduces the analytic power transfer on white noise
    rng = np.random.default_rng(8)
    n = 256
    ratio_acc, kk = None, None
    ones = np.ones((n, n))
    for _ in range(6):
        w = rng.standard_normal((n, n))
        f, _valid = dog_bandpass(w, ones, dx, lo, hi)
        kk, p_in = K.radial_psd(w, dx)
        _, p_out = K.radial_psd(f, dx)
        r = p_out / p_in
        ratio_acc = r if ratio_acc is None else ratio_acc + r
    ratio = ratio_acc / 6
    got = np.interp([1.0 / hi, 1.0 / lo], kk, ratio)
    ok3 = np.all(np.abs(got - 0.25) < 0.04)
    r3 = _report("dog", ok3, f"numerical power transfer at edges {got[0]:.3f}, {got[1]:.3f} (0.25)")
    # masked constant: no land bleed
    m = np.ones((n, n))
    m[:, :60] = 0
    m[100:140, 100:140] = 0
    f, valid = dog_bandpass(np.full((n, n), 3.0) * m, m, dx, lo, hi)
    resid = np.abs(f[valid]).max() if valid.any() else np.inf
    ok4 = resid < 1e-6
    r4 = _report("dog", ok4, f"masked constant -> |band-pass| max {resid:.1e} on valid pixels")
    return r1 and r2 and r3 and r4


def _selftest() -> int:
    print("[kernels_cond] selftest")
    results = [_test_crps_identity(), _test_crps_closed_form(), _test_rank_hist(),
               _test_spread_skill(), _test_cross_psd(), _test_band_power_parseval(),
               _test_dog()]
    ok = all(results)
    print(f"[kernels_cond] {'all checks passed' if ok else 'FAILURES -- do not trust diagnostics_cond'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--make-synthetic", metavar="PATH",
                    help="write a synthetic samples_cond npz with known answers")
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--vars", default="ssh,sst,chl")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=1_000_000)
    ap.add_argument("--s-churn", type=float, default=10.0)
    ap.add_argument("--cobs", type=int, default=1)
    ap.add_argument("--coverage", type=float, default=0.4)
    ap.add_argument("--underdispersed", type=float, default=None)
    ap.add_argument("--extras", default="ocean_mask,doy_sin,doy_cos")
    ap.add_argument("--doy-signal", type=float, default=0.0)
    ap.add_argument("--patch-lattice", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="synthetic_run")
    args = ap.parse_args()
    if args.make_synthetic:
        sampler = dict(num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                       s_churn=args.s_churn, s_noise=1.0, s_tmin=0.0, s_tmax=float("inf"))
        p = synthetic_cond_npz(
            args.make_synthetic, D=args.D, K_=args.K, H=args.H, W=args.W,
            vars=tuple(v for v in args.vars.split(",") if v), noise=args.noise,
            step=args.step, sampler=sampler, cobs=args.cobs, coverage=args.coverage,
            seed=args.seed, underdispersed=args.underdispersed,
            extras=tuple(e for e in args.extras.split(",") if e),
            doy_signal=args.doy_signal, patch_lattice=args.patch_lattice,
            run_name=args.run_name)
        print(f"[kernels_cond] wrote {p}")
        sys.exit(0)
    if not args.selftest:
        ap.error("nothing to do; pass --selftest or --make-synthetic PATH")
    sys.exit(_selftest())
