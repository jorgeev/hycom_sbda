"""Turn one samples .npz into figures, a metric table, and a readable report.

    python -m eval.diagnostics --samples samples_full.npz --out figs_full/

Five figures, each answering one question about the unconditional prior:

  fig1_gallery       Does it look like the Gulf Stream?
  fig2_spectra       Does it carry the right variance at each spatial scale?
  fig3_pdf           Are the one-point distributions right, tails included?
  fig4_eke           Does it have the right eddy kinetic energy, and in the
                     right places?
  fig5_crosschannel  Did it learn the RELATIONSHIPS between the seven channels,
                     or just seven plausible margins? For a joint prior this is
                     the one that matters.

WHICH UNITS WHERE. Everything quantitative -- spectra, PDFs and moments, EKE,
channel correlations -- uses ANOMALIES ONLY. The reference climatology is a fixed
spatial field common to both sides, so including it adds identical power to both
spectra, a shared spatial pattern to both correlation matrices, and identical
variance to both moment estimates: agreement that says nothing about the model.
It is not a small effect here -- the climatology's spatial variance is 1.9x the
anomaly's for sst and 3.8x for sss.

Only fig1 is drawn in physical units, because degC and psu are what the eye
reads, and it is emitted a second time as fig1_gallery_anom.png so you can see
what the model actually drew rather than the climatology it was handed.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

from . import kernels as K

# (colormap, force symmetric about zero). ssh/sst/sss carry a real mean in
# physical units so a symmetric scale would waste half the colour range; the
# ageostrophic velocities and wind stresses are anomalies about a near-zero
# time-mean and read best centred.
STYLE = {
    "ssh":   ("Spectral_r", False), "sst": ("RdYlBu_r", False),
    "sss":   ("viridis",    False), "uag": ("RdBu_r",   True),
    "vag":   ("RdBu_r",     True),  "tau_x": ("PuOr_r", True),
    "tau_y": ("PuOr_r",     True),
}
UNITS = {"ssh": "m", "sst": "degC", "sss": "psu", "uag": "m/s", "vag": "m/s",
         "tau_x": "N/m2", "tau_y": "N/m2"}
BANDS = [("mesoscale 40-150 km", 40.0, 150.0), ("submeso 10-40 km", 10.0, 40.0)]

# These two tables cover prior_gulfstream's seven channels. A config with a
# different target set still plots -- it just gets a neutral diverging map and a
# blank unit rather than a KeyError, in keeping with the repo's rule that no
# module hardcodes a store's variable names.
def _style(v: str) -> tuple[str, bool]:
    return STYLE.get(v, ("RdBu_r", True))


def _unit(v: str) -> str:
    return UNITS.get(v, "")


class Samples:
    """The npz, with the unit conventions applied once and named clearly."""

    def __init__(self, path: str):
        d = np.load(path, allow_pickle=False)
        self.vars = json.loads(str(d["vars"]))
        self.meta = json.loads(str(d["meta"]))
        self.gen_a = d["gen_norm"]                  # (N, C, H, W) anomaly, sigma units
        self.real_a = d["real_norm"]
        self.std = d["std"]                         # (C,)
        self.clim = d["clim_ref"]                   # (C, H, W)
        self.mask = d["mask"]
        self.lat = d["lat"]
        self.dx_m = float(d["dx_m"])
        self.dx_km = self.dx_m / 1000.0
        self.size = str(d["size"])
        self.days = d["real_days"]
        self.ref_day = int(d["ref_day"])
        self.n, self.c, self.ny, self.nx = self.gen_a.shape
        # eroded mask for anything involving a spatial derivative
        self.grad_mask = K.erode_mask(self.mask, 2)
        # Per-sample equivalent for the REAL side. In patch mode ``mask`` is
        # all ones (a generated patch is nowhere, see gen_prior), but each REAL
        # patch was cropped somewhere: land pixels inside it hold anomaly ==
        # 0.0 exactly, and np.gradient across that ocean->land step yields
        # O(10) m/s spurious geostrophic velocity, O(100) m2/s2 EKE. Worse, the
        # stride-32 crop lattice puts the same island at the same few patch
        # offsets in every sample, so the sample-mean EKE map showed it as a
        # regular grid of bright blobs (see the land-edge EKE gotcha in
        # eval/README.md). Rebuild each real patch's eroded ocean mask from its
        # crop corner and the full-domain mask.
        pos = d["real_pos"]
        full = d["mask_full"] if "mask_full" in d.files else np.zeros((0, 0))
        if pos.size and full.size:
            rgm = np.empty((self.n, self.ny, self.nx), dtype=bool)
            for i, (y0, x0) in enumerate(np.asarray(pos, dtype=int)):
                rgm[i] = K.erode_mask(full[y0:y0 + self.ny, x0:x0 + self.nx], 2)
            self.real_grad_mask = rgm & self.grad_mask
        else:
            if pos.size:
                print("[diag] WARNING: patch npz predates mask_full -- real "
                      "patches cannot be land-masked and derivative-based "
                      "figures will show land-edge artifacts. Backfill recipe "
                      "in eval/README.md.")
            self.real_grad_mask = np.broadcast_to(
                self.grad_mask, (self.n, self.ny, self.nx))

    def anom(self, which: str, v: str) -> np.ndarray:
        """(N, H, W) anomaly in PHYSICAL units (sigma * std), no climatology."""
        i = self.vars.index(v)
        a = self.gen_a if which == "gen" else self.real_a
        return a[:, i] * self.std[i]

    def phys(self, which: str, v: str) -> np.ndarray:
        """(N, H, W) full field: anomaly + reference-day climatology."""
        i = self.vars.index(v)
        a = self.gen_a if which == "gen" else self.real_a
        return K.to_physical(a[:, i], self.std[i], self.clim[i])

    def title(self) -> str:
        m = self.meta
        return (f"prior_gulfstream  step {m['step']:,}  |  {self.size} "
                f"{self.ny}x{self.nx}  |  N={self.n}  |  "
                f"{m['sampler']['num_steps']} steps, s_churn={m['sampler']['s_churn']}")


def _masked(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    out[..., ~(mask > 0.5)] = np.nan
    return out


def _limits(field: np.ndarray, mask: np.ndarray, symmetric: bool):
    v = field[:, mask > 0.5].ravel()
    lo, hi = np.percentile(v, [2, 98])
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


# ---------------------------------------------------------------------------
# fig 1: faceted gallery
# ---------------------------------------------------------------------------
def fig_gallery(S: Samples, out: str, ncol: int = 4, units: str = "physical"):
    """Rows = variables, columns = 4 generated then 4 real, shared scales.

    The colour limits come from the REAL fields, so a generated field with the
    wrong dynamic range shows up as washed out or saturated rather than being
    silently rescaled into looking fine.

    ``units="physical"`` adds the reference-day climatology to both sides, which
    is what makes the panels readable in degC / m / psu. But that field is
    IDENTICAL in every panel, and for sst and sss its spatial standard deviation
    is 1.9x and 3.8x the anomaly's -- so a physical-units gallery of those two
    is mostly a picture of the climatology, and generated and real look alike for
    a reason that has nothing to do with the model. ``units="anomaly"`` drops it
    and shows only what the model actually drew. Read both.
    """
    ncol = min(ncol, S.n)
    anom = units == "anomaly"
    fig = plt.figure(figsize=(2.05 * 2 * ncol + 1.6, 1.55 * len(S.vars) + 1.0))
    gs = GridSpec(len(S.vars), 2 * ncol + 1, figure=fig,
                  width_ratios=[1] * ncol + [0.18] + [1] * ncol,
                  wspace=0.06, hspace=0.10, left=0.055, right=0.90,
                  top=0.915, bottom=0.02)
    for r, v in enumerate(S.vars):
        cmap, sym = _style(v)
        if anom:
            cmap, sym = "RdBu_r", True
        real = S.anom("real", v) if anom else S.phys("real", v)
        gen = S.anom("gen", v) if anom else S.phys("gen", v)
        lo, hi = _limits(real, S.mask, sym)
        for c in range(ncol):
            for half, arr in ((0, gen), (1, real)):
                ax = fig.add_subplot(gs[r, c + half * (ncol + 1)])
                im = ax.imshow(_masked(arr[c], S.mask), origin="lower", cmap=cmap,
                               vmin=lo, vmax=hi, interpolation="nearest",
                               aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(("generated" if half == 0 else "real")
                                 + f" #{c + 1}", fontsize=8, pad=3)
        ax.text(1.03, 0.5, f"{v}\n[{_unit(v)}]", transform=ax.transAxes,
                fontsize=9, va="center", ha="left", weight="bold")
        cax = fig.add_axes([0.945, gs[r, 0].get_position(fig).y0,
                            0.011, gs[r, 0].get_position(fig).height])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6)
    # the divider between the two halves
    xd = gs[0, ncol].get_position(fig).x0 + gs[0, ncol].get_position(fig).width / 2
    fig.add_artist(plt.Line2D([xd, xd], [0.02, 0.915], color="0.35", lw=1.4))
    kind = ("ANOMALIES only (what the model drew)" if anom
            else "physical units (anomaly + shared reference climatology)")
    fig.suptitle(f"Unconditional prior samples vs real fields, {kind}  --  "
                 + S.title(), fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gallery_single(S: Samples, which: str, out: str, ncol: int = 8):
    """A denser single-sided gallery, real-derived colour limits either way."""
    fig, axes = plt.subplots(len(S.vars), ncol,
                             figsize=(1.75 * ncol + 1.4, 1.5 * len(S.vars) + 0.8),
                             squeeze=False)
    for r, v in enumerate(S.vars):
        cmap, sym = _style(v)
        lo, hi = _limits(S.phys("real", v), S.mask, sym)
        arr = S.phys(which, v)
        for c in range(ncol):
            ax = axes[r][c]
            im = ax.imshow(_masked(arr[c % S.n], S.mask), origin="lower", cmap=cmap,
                           vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{v}\n[{_unit(v)}]", fontsize=8)
        fig.colorbar(im, ax=axes[r].tolist(), fraction=0.012, pad=0.006
                     ).ax.tick_params(labelsize=6)
    fig.suptitle(f"{which} samples  --  " + S.title(), fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 2: spectra
# ---------------------------------------------------------------------------
def fig_spectra(S: Samples, out: str) -> dict:
    """Radial PSD of the anomalies, gen vs real, plus the ratio.

    Two reference scales are marked. 2*dx is the Nyquist wavelength -- nothing
    below it is real. The 128 px training-patch span is the largest structure the
    model ever saw during training; at full frame it is being asked to place
    power beyond that, which it was never shown, so the ratio curve to the left
    of that line is the interesting part.
    """
    tile = 256 if min(S.ny, S.nx) >= 256 else min(S.ny, S.nx)
    stride = tile // 2
    patch_km = 128 * S.dx_km
    res = {}

    ncol = 4
    nrow = -(-len(S.vars) // ncol)              # ceil
    fig = plt.figure(figsize=(15, 5.25 * nrow + 4.2))
    gs = GridSpec(nrow + 1, ncol, figure=fig,
                  height_ratios=[1] * nrow + [1.15], hspace=0.42,
                  wspace=0.27, left=0.06, right=0.985, top=0.90, bottom=0.07)
    ratio_ax = fig.add_subplot(gs[nrow, :])

    for i, v in enumerate(S.vars):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        k, pg = K.mean_radial_psd(S.anom("gen", v), S.mask, S.dx_km, tile, stride)
        _, pr = K.mean_radial_psd(S.anom("real", v), S.mask, S.dx_km, tile, stride)
        lam = 1.0 / k
        for p, col, lab in ((pr, "k", "real"), (pg, "crimson", "generated")):
            med = np.median(p, 0)
            ax.fill_between(lam, np.percentile(p, 25, 0), np.percentile(p, 75, 0),
                            color=col, alpha=0.20, lw=0)
            ax.plot(lam, med, color=col, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.axvline(2 * S.dx_km, color="0.5", ls=":", lw=1)
        ax.axvline(patch_km, color="tab:blue", ls="--", lw=1)
        ax.set_title(f"{v}  [{_unit(v)}]", fontsize=10)
        ax.set_xlabel("wavelength [km]", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)

        r = np.median(pg, 0) / np.median(pr, 0)
        ratio_ax.plot(lam, r, lw=1.5, label=v)
        res[v] = {f"psd_ratio_{name}": K.band_ratio(k, np.median(pg, 0),
                                                    np.median(pr, 0), lo, hi)
                  for name, lo, hi in BANDS}

    ratio_ax.axhline(1.0, color="k", lw=1)
    ratio_ax.axvline(2 * S.dx_km, color="0.5", ls=":", lw=1)
    ratio_ax.axvline(patch_km, color="tab:blue", ls="--", lw=1)
    ratio_ax.text(patch_km, ratio_ax.get_ylim()[1], " 128 px training patch",
                  color="tab:blue", fontsize=8, va="top")
    ratio_ax.text(2 * S.dx_km, ratio_ax.get_ylim()[1], " 2dx", color="0.5",
                  fontsize=8, va="top")
    band_handles = []
    for (name, lo, hi), col in zip(BANDS, ("tab:green", "tab:purple")):
        ratio_ax.axvspan(lo, hi, color=col, alpha=0.09, lw=0)
        band_handles.append(Patch(facecolor=col, alpha=0.28, label=name))
    ratio_ax.set_xscale("log"); ratio_ax.set_yscale("log"); ratio_ax.invert_xaxis()
    ratio_ax.set_xlabel("wavelength [km]")
    ratio_ax.set_ylabel("PSD ratio  generated / real")
    ratio_ax.set_title("Above 1 = too much variance at that scale; below 1 = too smooth",
                       fontsize=9)
    ratio_ax.grid(alpha=0.25, which="both", lw=0.4)
    # Two legends on one axes: the first must be re-added as an artist BEFORE
    # the second is created, otherwise creating the second removes it.
    var_leg = ratio_ax.legend(fontsize=8, ncol=len(S.vars), frameon=False,
                              loc="lower left")
    ratio_ax.add_artist(var_leg)
    # The shaded bands are the two wavelength ranges summarised as scalars in
    # summary.csv and report.md.
    ratio_ax.legend(handles=band_handles, fontsize=8, frameon=False,
                    loc="upper right", title="summarised bands",
                    title_fontsize=8)

    free = nrow * ncol - len(S.vars)             # leftover panel slots, if any
    note = fig.add_subplot(gs[nrow - 1, ncol - 1]) if free else None
    if note is not None:
        note.axis("off")
        note.text(0.0, 0.5,
                  "Anomalies only -- the reference\nclimatology is a fixed spatial\n"
                  "field common to both sides, so\nincluding it would add identical\n"
                  "power to both curves.\n\n"
                  "Bands are the inter-sample IQR.\n\n"
                  "blue dashed: the 128 px training\npatch. At full frame the model\n"
                  "is asked for structure larger\nthan anything it was trained on.\n\n"
                  "dotted: 2dx Nyquist. Nothing to\nthe right of it is real.",
                  fontsize=8.5, va="center")
    fig.suptitle(f"Radial power spectra of anomalies ({tile}px tiles)  --  " + S.title(),
                 fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# fig 3: marginal PDFs
# ---------------------------------------------------------------------------
def fig_pdf(S: Samples, out: str) -> dict:
    """Per-variable marginal density of the ANOMALIES, log-y to show the tails.

    Anomalies, not full fields, and this is load-bearing. The reference
    climatology is a fixed spatial field added identically to both sides, and its
    spatial variance is 1.9x the anomaly's for sst and 3.8x for sss -- so a
    physical-units moment ratio is mostly measuring the climatology the model
    never produced. Concretely: a model whose sst anomalies are half the right
    amplitude reports a physical std ratio of 0.88, which reads as almost
    correct. Anomalies have the same units (degC, m, psu), so nothing is lost
    in readability; only the absolute level is dropped.
    """
    res = {}
    ncol = 4
    nrow = -(-(len(S.vars) + 1) // ncol)         # +1 reserves the note slot
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.6 * nrow), squeeze=False)
    for i, v in enumerate(S.vars):
        ax = axes.flat[i]
        vals = {}
        for which, col in (("real", "k"), ("gen", "crimson")):
            x = K.ocean_values(S.anom(which, v), S.mask, seed=1)
            vals[which] = x
            grid = np.linspace(*np.percentile(x, [0.05, 99.95]), 400)
            ax.plot(grid, stats.gaussian_kde(x)(grid), color=col, lw=1.6,
                    label=which if i == 0 else None)
        ax.set_yscale("log")
        ax.set_title(f"{v} anomaly  [{_unit(v)}]", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
        mg, mr = K.moments(vals["gen"]), K.moments(vals["real"])
        res[v] = dict(zip(
            ("mean_gen", "std_gen", "skew_gen", "kurt_gen"), mg))
        res[v].update(dict(zip(
            ("mean_real", "std_real", "skew_real", "kurt_real"), mr)))
        res[v]["std_ratio"] = mg[1] / mr[1]
        ax.text(0.02, 0.04, f"std gen/real = {mg[1] / mr[1]:.2f}",
                transform=ax.transAxes, fontsize=8)
    for ax in axes.flat[len(S.vars):]:
        ax.axis("off")
    axes.flat[len(S.vars)].text(0.0, 0.5,
                       "Physical units.\nSame reference-day climatology\n"
                       "added to generated and real,\nso the seasonal cycle is not\n"
                       "part of the comparison.\n\nLog y-axis: the tails are where\n"
                       "an under-trained prior fails first.",
                       fontsize=9, va="center")
    fig.suptitle("Marginal distributions of the anomalies over ocean pixels  --  "
                 + S.title(), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# fig 4: eddy kinetic energy
# ---------------------------------------------------------------------------
def _velocities(S: Samples, which: str):
    """Geostrophic (from the ssh anomaly), ageostrophic (channels), and total.

    Adding the two is meaningful because the store defines its own velocity that
    way: ``u = ug + uag`` holds to ~0.1 % of u's rms, which
    ``eval.kernels --selftest`` verifies against the store's `u`/`ug`/`uag`
    fields rather than taking it on faith.
    """
    ssh = S.anom(which, "ssh")
    uag, vag = S.anom(which, "uag"), S.anom(which, "vag")
    ug = np.empty_like(ssh); vg = np.empty_like(ssh)
    for i in range(ssh.shape[0]):
        ug[i], vg[i] = K.geostrophic_uv(ssh[i], S.dx_m, S.lat)
    return (ug, vg), (uag, vag), (ug + uag, vg + vag)


def fig_eke(S: Samples, out: str) -> dict:
    """EKE maps and budget, gen vs real, identical recipe on both sides.

    Geostrophic velocity is derived from the ssh ANOMALY, so its kinetic energy
    is already eddy kinetic energy -- using the denormalised field instead would
    fold in the mean Gulf Stream jet. The ageostrophic part comes straight from
    the generated uag/vag channels, which are anomalies by construction.

    The two sides are masked differently on purpose: the generated side uses the
    shared ``grad_mask`` (its land, if any, is wherever the model drew it), the
    real side uses ``real_grad_mask`` so each patch's own land edges never meet
    np.gradient. In full geometry the two masks coincide.
    """
    m = S.grad_mask
    rm = S.real_grad_mask                                 # (N, H, W)
    res = {}
    fields = {}
    for which in ("gen", "real"):
        geo, ageo, tot = _velocities(S, which)
        fields[which] = {"geostrophic": K.eke(*geo), "ageostrophic": K.eke(*ageo),
                         "total": K.eke(*tot)}
        for part, arr in fields[which].items():
            sel = arr[rm] if which == "real" else arr[:, m]
            res[f"eke_{part}_{which}"] = float(np.nanmean(sel))
    for part in ("geostrophic", "ageostrophic", "total"):
        res[f"eke_{part}_ratio"] = res[f"eke_{part}_gen"] / res[f"eke_{part}_real"]

    # Generated side: average over samples FIRST, then mask. Masking first
    # would leave every masked pixel as a column of all-NaN, and averaging that
    # is a mean of an empty slice -- same numbers, but a screenful of warnings.
    gmap = _masked(fields["gen"]["total"].mean(0), m)
    # Real side: each sample has its own mask, so a plain mean(0) is exactly
    # the bug this masking exists to fix. Sum/count keeps a pixel's mean over
    # the samples where it is valid ocean and divides all-masked pixels to NaN
    # without an empty-slice warning.
    cnt = rm.sum(0)
    rsum = np.where(rm, fields["real"]["total"], 0.0).sum(0)
    rmap = np.where(cnt > 0, rsum / np.maximum(cnt, 1), np.nan)
    vmax = float(np.nanpercentile(rmap, 99))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2))
    for ax, (arr, ttl) in zip(axes[0][:2],
                              ((gmap, "generated"), (rmap, "real"))):
        im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_title(f"mean total EKE, {ttl}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("m2/s2", fontsize=8)

    ax = axes[0][2]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(gmap / rmap)
    # Adaptive, not a fixed +-1: a mid-training prior can sit an order of
    # magnitude off, and a clipped panel would show a uniform block of colour
    # that hides where the error actually varies.
    lim = float(np.clip(np.nanpercentile(np.abs(lr), 99), 0.3, 2.0))
    im = ax.imshow(lr, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="auto")
    ax.set_title("log10(generated / real)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[1][0]
    lat1d = np.asarray(S.lat).mean(axis=1)
    varies = float(lat1d.max() - lat1d.min()) > 0.01
    yy = lat1d if varies else np.arange(S.ny)
    # The eroded mask leaves the first and last rows entirely masked, so a plain
    # nanmean over them is a mean of an empty slice. Drop those rows instead of
    # plotting NaN endpoints.
    rows = np.isfinite(rmap).any(axis=1) & np.isfinite(gmap).any(axis=1)
    ax.plot(np.nanmean(rmap[rows], 1), yy[rows], "k", lw=1.6, label="real")
    ax.plot(np.nanmean(gmap[rows], 1), yy[rows], "crimson", lw=1.6,
            label="generated")
    ax.set_xlabel("zonal-mean EKE [m2/s2]", fontsize=8)
    ax.set_ylabel("latitude [degN]" if varies else "grid row (south -> north)",
                  fontsize=8)
    ax.set_title("meridional structure", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=7)

    ax = axes[1][1]
    per_sample = {
        "gen": np.nanmean(fields["gen"]["total"][:, m], axis=1),
        "real": (np.where(rm, fields["real"]["total"], 0.0).sum((1, 2))
                 / np.maximum(rm.sum((1, 2)), 1)),
    }
    # Shared bin edges: per-series bins would put the two histograms on
    # different grids and make an eyeball comparison meaningless.
    allv = np.concatenate(list(per_sample.values()))
    bins = np.linspace(allv.min(), allv.max(), 21)
    for which, col in (("real", "k"), ("gen", "crimson")):
        ax.hist(per_sample[which], bins=bins, histtype="step", color=col,
                lw=1.6, label=which)
    ax.set_xlabel("domain-mean total EKE [m2/s2]", fontsize=8)
    ax.set_ylabel("samples", fontsize=8)
    ax.set_title("spread across samples", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=7)

    ax = axes[1][2]
    parts = ["geostrophic", "ageostrophic", "total"]
    xs = np.arange(3)
    ax.bar(xs - 0.19, [res[f"eke_{p}_real"] for p in parts], 0.36, color="0.25",
           label="real")
    ax.bar(xs + 0.19, [res[f"eke_{p}_gen"] for p in parts], 0.36, color="crimson",
           label="generated")
    ax.set_xticks(xs); ax.set_xticklabels(parts, fontsize=8)
    ax.set_ylabel("domain-mean EKE [m2/s2]", fontsize=8)
    ax.set_title("energy budget", fontsize=10)
    for i, p in enumerate(parts):
        ax.text(i, max(res[f"eke_{p}_real"], res[f"eke_{p}_gen"]),
                f"x{res[f'eke_{p}_ratio']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, axis="y", lw=0.4)
    ax.tick_params(labelsize=7)

    fig.suptitle("Eddy kinetic energy: geostrophy from the ssh anomaly + the "
                 "uag/vag channels  --  " + S.title(), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# fig 5: cross-channel structure
# ---------------------------------------------------------------------------
def fig_crosschannel(S: Samples, out: str) -> dict:
    """Do the seven channels co-vary the way the real ocean's do?

    Panels (a)-(c) are the pixelwise channel correlation matrices and their
    difference. (d)-(e) test one specific physical relationship the prior has
    every opportunity to learn: the geostrophic velocity implied by its own ssh
    against the ageostrophic velocity it drew alongside it.
    """
    cg = K.corr_matrix(S.gen_a, S.mask, seed=2)
    cr = K.corr_matrix(S.real_a, S.mask, seed=2)
    diff = cg - cr
    res = {"corr_frobenius_error": float(np.linalg.norm(diff)),
           "corr_max_abs_error": float(np.abs(diff).max())}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (mat, ttl, cmap, lim) in zip(
            axes[0],
            ((cr, "real", "RdBu_r", 1.0), (cg, "generated", "RdBu_r", 1.0),
             (diff, "generated - real", "PuOr_r", 0.5))):
        im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(S.c)); ax.set_xticklabels(S.vars, rotation=45, fontsize=8)
        ax.set_yticks(range(S.c)); ax.set_yticklabels(S.vars, fontsize=8)
        ax.set_title(f"channel correlation, {ttl}", fontsize=10)
        for a in range(S.c):
            for b in range(S.c):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(mat[a, b]) > 0.55 * lim else "black")
        fig.colorbar(im, ax=ax, fraction=0.045)

    m = S.grad_mask
    pts = {}
    for which in ("real", "gen"):
        geo, ageo, _ = _velocities(S, which)
        # Real patches carry their own land (see Samples.real_grad_mask); a
        # generated patch has none, so the shared eroded mask is right for it.
        if which == "real":
            x, y = geo[0][S.real_grad_mask], ageo[0][S.real_grad_mask]
        else:
            x, y = geo[0][:, m].ravel(), ageo[0][:, m].ravel()
        keep = np.isfinite(x) & np.isfinite(y)
        pts[which] = (x[keep], y[keep])
    # One range for both panels, or the two clouds cannot be compared by eye.
    xlim = max(float(np.percentile(np.abs(pts[w][0]), 99.5)) for w in pts)
    ylim = max(float(np.percentile(np.abs(pts[w][1]), 99.5)) for w in pts)
    for ax, which in zip(axes[1][:2], ("real", "gen")):
        x, y = pts[which]
        ax.hist2d(x, y, bins=120, range=[[-xlim, xlim], [-ylim, ylim]],
                  cmap="magma", norm=matplotlib.colors.LogNorm())
        ax.set_xlabel("geostrophic u from ssh [m/s]", fontsize=8)
        ax.set_ylabel("uag channel [m/s]", fontsize=8)
        ax.set_title(f"{which}: r = {np.corrcoef(x, y)[0, 1]:+.3f}", fontsize=10)
        ax.tick_params(labelsize=7)
        res[f"r_ug_uag_{which}"] = float(np.corrcoef(x, y)[0, 1])

    ax = axes[1][2]
    ax.axis("off")
    ax.text(0.0, 0.5,
            "A joint prior exists to model the\nDEPENDENCE between channels.\n\n"
            "Seven correct marginals with the\nwrong correlation structure is a\n"
            "failure mode the loss curve and\nthe per-variable spectra both\n"
            "report as success.\n\n"
            f"Frobenius |gen - real| = {res['corr_frobenius_error']:.3f}\n"
            f"largest single error   = {res['corr_max_abs_error']:.3f}",
            fontsize=9.5, va="center")

    fig.suptitle("Cross-channel structure  --  " + S.title(), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(S: Samples, spec: dict, pdf: dict, eke: dict, cross: dict,
                 out_dir: str):
    rows = []
    for v in S.vars:
        r = {"var": v, "units": _unit(v)}
        r.update({k: pdf[v][k] for k in pdf[v]})
        r.update(spec[v])
        rows.append(r)

    csv = os.path.join(out_dir, "summary.csv")
    cols = list(rows[0].keys())
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")

    m = S.meta
    L = [f"# Unconditional prior diagnostics -- step {m['step']:,}", "",
         f"- checkpoint: `{m['ckpt']}`",
         f"- val_loss at this checkpoint: {m['val_loss']}",
         f"- geometry: {S.size} ({S.ny}x{S.nx}), N={S.n} samples, "
         f"real fields from split `{m['split']}` steps {S.days[0]}..{S.days[-1]}",
         f"- sampler: {m['sampler']['num_steps']} Heun steps, "
         f"s_churn={m['sampler']['s_churn']}, s_noise={m['sampler']['s_noise']}",
         f"- climatology reference step (both sides): {S.ref_day}",
         f"- latitude for geostrophy: {m['lat_note']}",
         "",
         "All ratios below are generated / real. 1.00 is perfect.",
         "All statistics are computed on ANOMALIES -- adding the shared "
         "reference climatology would dilute every ratio (1.9x the anomaly "
         "variance for sst, 3.8x for sss).", "",
         "## Per-variable", "",
         "| var | std gen/real | skew gen / real | kurt gen / real | "
         "PSD ratio 40-150 km | PSD ratio 10-40 km |",
         "|---|---|---|---|---|---|"]
    for v in S.vars:
        p, s = pdf[v], spec[v]
        L.append(f"| {v} | {p['std_ratio']:.2f} | "
                 f"{p['skew_gen']:+.2f} / {p['skew_real']:+.2f} | "
                 f"{p['kurt_gen']:+.2f} / {p['kurt_real']:+.2f} | "
                 f"{s['psd_ratio_mesoscale 40-150 km']:.2f} | "
                 f"{s['psd_ratio_submeso 10-40 km']:.2f} |")

    L += ["", "## Energetics", "",
          "| component | real | generated | ratio |", "|---|---|---|---|"]
    for part in ("geostrophic", "ageostrophic", "total"):
        L.append(f"| {part} EKE [m2/s2] | {eke[f'eke_{part}_real']:.4f} | "
                 f"{eke[f'eke_{part}_gen']:.4f} | {eke[f'eke_{part}_ratio']:.2f} |")

    L += ["", "## Cross-channel", "",
          f"- Frobenius norm of the correlation-matrix error: "
          f"**{cross['corr_frobenius_error']:.3f}** (0 = perfect)",
          f"- largest single correlation error: {cross['corr_max_abs_error']:.3f}",
          f"- corr(geostrophic u from ssh, uag): real "
          f"{cross['r_ug_uag_real']:+.3f}, generated {cross['r_ug_uag_gen']:+.3f}",
          "", "## How to read this", "",
          "- **std ratio well below 1** means the prior is under-dispersed: it "
          "draws samples that are too smooth/too weak overall.",
          "- **PSD ratio falling with wavelength** means the small scales are "
          "missing (over-smoothed); rising means grid-scale noise.",
          "- **EKE ratio** is the most demanding scalar here: it depends on the "
          "ssh gradient, so it punishes both over-smoothing and noise.",
          "- **Frobenius correlation error** is the joint-prior check. Good "
          "marginals with a bad correlation matrix means the model has learned "
          "seven fields but not one ocean.", ""]

    rep = os.path.join(out_dir, "report.md")
    with open(rep, "w") as f:
        f.write("\n".join(L))
    return csv, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="npz from eval.gen_prior")
    ap.add_argument("--out", required=True, help="output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    S = Samples(args.samples)
    print(f"[diag] {S.n} samples, {S.c} vars, {S.ny}x{S.nx}, "
          f"dx={S.dx_km:.3f} km  ({S.size})")

    p = lambda name: os.path.join(args.out, name)
    print("[diag] fig1 gallery ...", flush=True)
    fig_gallery(S, p("fig1_gallery.png"))
    fig_gallery(S, p("fig1_gallery_anom.png"), units="anomaly")
    fig_gallery_single(S, "gen", p("fig1_gallery_gen.png"))
    fig_gallery_single(S, "real", p("fig1_gallery_real.png"))
    print("[diag] fig2 spectra ...", flush=True)
    spec = fig_spectra(S, p("fig2_spectra.png"))
    print("[diag] fig3 pdf ...", flush=True)
    pdf = fig_pdf(S, p("fig3_pdf.png"))
    print("[diag] fig4 eke ...", flush=True)
    eke = fig_eke(S, p("fig4_eke.png"))
    print("[diag] fig5 crosschannel ...", flush=True)
    cross = fig_crosschannel(S, p("fig5_crosschannel.png"))
    csv, rep = write_report(S, spec, pdf, eke, cross, args.out)
    print(f"[diag] wrote {csv} and {rep}")
    print("\n" + open(rep).read())


if __name__ == "__main__":
    main()
