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
from scipy import ndimage, stats

from . import kernels as K

# (colormap, force symmetric about zero). ssh/sst/sss carry a real mean in
# physical units so a symmetric scale would waste half the colour range; the
# ageostrophic velocities and wind stresses are anomalies about a near-zero
# time-mean and read best centred.
STYLE = {
    "ssh":   ("Spectral_r", False), "sst": ("RdYlBu_r", False),
    "sss":   ("viridis",    False), "uag": ("RdBu_r",   True),
    "vag":   ("RdBu_r",     True),  "tau_x": ("PuOr_r", True),
    "tau_y": ("PuOr_r",     True),  "chl": ("YlGn",     False),
    "mld":   ("cividis",    False),
}
UNITS = {"ssh": "m", "sst": "degC", "sss": "psu", "uag": "m/s", "vag": "m/s",
         "tau_x": "N/m2", "tau_y": "N/m2", "mld": "m",
         # gom_nemo takes log10 of CHL in the loader (descriptor `log10: true`),
         # so the model's channel, and everything here, is in log space.
         "chl": "log10(mg/m3)"}
BANDS = [("mesoscale 40-150 km", 40.0, 150.0), ("submeso 10-40 km", 10.0, 40.0)]

# How much of a frame to throw away before pooling pixels. Both numbers are
# MEASURED on prior_genda_masked, not chosen for looking safe; both are written
# up in the corresponding gotchas in eval/README.md.
#
# GRAD_ERODE_PX -- how far from a coastline a spatial derivative is still
# contaminated. Land enters as an exact 0.0 anomaly, so np.gradient across an
# ocean->land step manufactures velocity. The numerical smear is only 1 px wide,
# but the contaminated RIM is wider: real geostrophic EKE binned by distance
# from that patch's own coast runs 80x the far field at 1-2 px, 10x at 2-3 px,
# 3x at 3-4 px and 2x at 4-6 px, reaching background only near 6 px. The old
# 2 px erosion kept everything from 3 px outward -- a rim still 2-3x too bright.
#
# 6 px is where the answer stops moving, which is the real argument for it: the
# real patch-mean EKE is 0.0438 at 2 px, 0.0408 at 6 and 0.0409 at 10, then
# drifts UP to 0.0426 at 24 as the surviving pixels become a deep-basin
# subsample rather than a cleaner one. A thin tail of contaminated pixels does
# survive 6 px (the map's 99.9th percentile falls 0.163 -> 0.102 between 6 and
# 10 px) but it is 0.1 % of pixels and worth 0.2 % of the mean.
GRAD_ERODE_PX = 6

# PATCH_BORDER_PX -- the frame edge itself. A generated patch contains no land,
# but the UNet's zero padding leaves an energy halo at its border: generated
# geostrophic EKE is 0.0855 within 2-4 px of the frame against 0.0648 in the
# interior (+32 %), while the real side is flat over the same bins (0.0454 vs
# 0.0407). Split-half reproducibility of the generated mean EKE map falls from
# r = 0.78 with a 2 px cut to 0.45 at 16 px and 0.29 at 32 px -- 0.29 being the
# real side's own level -- so the reproducible structure in that map is the
# border, not geography (it correlates with land frequency at -0.06). 16 px is
# where the inflation drops into the sample noise.
#
# Applied in PATCH geometry only, and identically to both sides: at full frame
# the frame edge is a real domain boundary that the real fields share, and
# erode_mask already drops the one-sided-gradient row. The spectra are exempt
# because an FFT needs a whole rectangle, and radial_psd's 2-D Hann window
# already weights the outer 16 px of a 128 px tile below 0.15 in amplitude.
PATCH_BORDER_PX = 16

# These two tables cover prior_gulfstream's seven channels. A config with a
# different target set still plots -- it just gets a neutral diverging map and a
# blank unit rather than a KeyError, in keeping with the repo's rule that no
# module hardcodes a store's variable names.
def _style(v: str) -> tuple[str, bool]:
    return STYLE.get(v.lower(), ("RdBu_r", True))


def _unit(v: str) -> str:
    return UNITS.get(v.lower(), "")


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
        # Channel names are the store's, and stores disagree on case: gulfstream
        # calls it `ssh`, gom_nemo `SSH`. Physics-aware figures ask for the
        # channel by its conventional lowercase name and get whatever this store
        # actually calls it, or None.
        self._byname = {v.lower(): v for v in self.vars}
        pos = d["real_pos"]
        self.is_patch = bool(pos.size)
        # Spacing of the crop lattice the real patches came from, read off the
        # corners rather than assumed. It is the period at which a patch-relative
        # SAMPLE MEAN aliases the domain's own geography: patch pixel (i, j) and
        # (i + stride, j) average almost the same set of absolute positions, so a
        # mean map in patch geometry is a FOLD of the domain, not a map of it.
        # Measured on prior_genda_masked, the real mean-EKE map's shift
        # autocorrelation is 0.02 at 24 px, 0.67 at 32 px and 0.05 at 36 px --
        # a lattice spike, not a smooth field.
        self.crop_stride = 0
        if self.is_patch:
            p0 = np.asarray(pos, dtype=int)
            dif = np.concatenate([np.diff(np.unique(p0[:, 0])),
                                  np.diff(np.unique(p0[:, 1]))])
            self.crop_stride = int(dif.min()) if dif.size else 0
        # Interior mask: drops PATCH_BORDER_PX from every frame edge in patch
        # geometry, all-True at full frame. See PATCH_BORDER_PX for the
        # measurement, and note it is applied to BOTH sides -- the point is a
        # like-for-like recipe, not a flattering one.
        self.interior = np.ones((self.ny, self.nx), dtype=bool)
        if self.is_patch and 2 * PATCH_BORDER_PX < min(self.ny, self.nx):
            b = PATCH_BORDER_PX
            self.interior[:b, :] = False
            self.interior[-b:, :] = False
            self.interior[:, :b] = False
            self.interior[:, -b:] = False

        # Land as the DATA reports it, per sample. The loader writes an exact
        # 0.0 into EVERY channel at once on land, and no ocean pixel is exactly
        # 0.0 in all of them, so this is an exact test rather than a threshold.
        #
        # It is also the authority, not the store's ``ocean_mask``: on gom_nemo
        # that mask calls 143 pixels ocean whose data is zero in every frame of
        # the record -- 12 blobs, the largest 102 px (~1600 km2, the size and
        # place of Isla de la Juventud), the rest keys and islets. Because the
        # mask does not know they are land, ``erode_mask`` never touches them
        # and ``np.gradient`` runs straight across their coastline: they were
        # drawing bright closed rings in the real EKE map next to the properly
        # eroded islands, and they cost 4.8 % of the full-frame real EKE mean
        # (3.7 % at patch level) out of 0.06 % of the pixels.
        self.data_land = np.all(self.real_a == 0.0, axis=1)      # (N, H, W)
        # Domain-level correction: a pixel that is land in most samples is land,
        # whatever the stored mask says. In patch geometry ``mask`` is all ones
        # and a given patch offset is land in ~12 % of crops, so this leaves it
        # alone and the per-sample term below does the work.
        self.mask_ocean = (self.mask > 0.5) & (self.data_land.mean(0) <= 0.5)
        # eroded mask for anything involving a spatial derivative
        self.grad_mask = K.erode_mask(self.mask_ocean, GRAD_ERODE_PX) & self.interior
        # plain mask for statistics that pool pixels without differentiating
        # (PDFs and moments, the channel correlation matrices)
        self.stat_mask = self.mask_ocean & self.interior
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
        #
        # The SAME contamination reaches every other real-side statistic, not
        # just the derivative ones. On gom_nemo the grid crop lattice admits any
        # patch with ocean_frac >= 0.5, so real patches average 87.5 % ocean and
        # dip to 51 %, while a generated patch is all ocean by construction.
        # Land enters as an exact 0.0 anomaly, so pooling it into the real side
        # of a comparison:
        #   - drags the real std down and the real kurtosis up (fig_pdf),
        #   - correlates every channel perfectly wherever it appears
        #     (fig_crosschannel),
        #   - and, worst by far, puts a step edge in the real field that the
        #     FFT reads as broadband power. Measured on prior_genda_masked at
        #     step 1,130,000 that inflates the real 10-40 km PSD by 40x for SSH
        #     (whose true submesoscale power is minuscule) and ~4x for SST and
        #     CHL, which turned a genuine 5x EXCESS of SSH grid-scale power into
        #     an apparent 0.18 deficit -- the opposite diagnosis.
        # So keep three things: the eroded per-sample mask for derivatives, the
        # plain per-sample mask for pixel statistics, and a per-sample flag for
        # the figures that need a whole rectangle (spectra) and must select
        # land-free samples instead of masking pixels.
        full = d["mask_full"] if "mask_full" in d.files else np.zeros((0, 0))
        if pos.size and full.size:
            base = np.stack([full[y0:y0 + self.ny, x0:x0 + self.nx] > 0.5
                             for y0, x0 in np.asarray(pos, dtype=int)])
        else:
            if pos.size:
                print("[diag] NOTE: patch npz predates mask_full -- the real "
                      "patches' land is being taken from the data itself "
                      "(all-channel zeros), which is exact; only the stored "
                      "mask is missing.")
            base = np.broadcast_to(self.mask > 0.5,
                                   (self.n, self.ny, self.nx))
        rm = base & ~self.data_land
        rgm = np.stack([K.erode_mask(rm[i], GRAD_ERODE_PX)
                        for i in range(self.n)])
        self.real_grad_mask = rgm & self.grad_mask
        self.real_mask = rm & self.stat_mask
        # ``real_clean`` selects whole rectangles for the spectra. At full frame
        # the tiles come from the shared ocean mask and every sample is usable;
        # in patch geometry a sample is usable only if its crop holds no land.
        self.real_clean = (rm.reshape(self.n, -1).all(axis=1) if self.is_patch
                           else np.ones(self.n, dtype=bool))

    def name(self, v: str) -> str | None:
        """This store's spelling of channel ``v``, or None if it has no such channel."""
        return self._byname.get(v.lower())

    def has(self, *names: str) -> bool:
        return all(self.name(v) is not None for v in names)

    def anom(self, which: str, v: str) -> np.ndarray:
        """(N, H, W) anomaly in PHYSICAL units (sigma * std), no climatology."""
        i = self.vars.index(self.name(v) or v)
        a = self.gen_a if which == "gen" else self.real_a
        return a[:, i] * self.std[i]

    def phys(self, which: str, v: str) -> np.ndarray:
        """(N, H, W) full field: anomaly + reference-day climatology."""
        i = self.vars.index(self.name(v) or v)
        a = self.gen_a if which == "gen" else self.real_a
        return K.to_physical(a[:, i], self.std[i], self.clim[i])

    def run_name(self) -> str:
        """The run this npz came from, read off the checkpoint path.

        Not hardcoded: the same figures are produced for every config, and a
        gulfstream label on a gom_nemo figure is the kind of error that is only
        noticed after the figure has been in a talk.
        """
        ck = self.meta.get("ckpt", "")
        d = os.path.dirname(ck)
        if os.path.basename(d) == "checkpoints":
            d = os.path.dirname(d)
        return os.path.basename(d) or "run"

    def fold_note(self) -> str:
        """The patch-geometry caveat for any SAMPLE-MEAN map, or "" at full frame.

        See ``crop_stride``. The generated side inherits the same fold, because
        its training patches came off the same lattice: measured here, the
        generated and real 32 px folds correlate at +0.41 while their residuals
        correlate at +0.09.
        """
        if not (self.is_patch and self.crop_stride):
            return ""
        return (f"patch-relative mean: the {self.crop_stride} px crop lattice "
                f"folds the domain's own geography at {self.crop_stride} px, on "
                "BOTH sides -- read the level, not the pattern")

    def mask_note(self) -> str:
        """One line recording which pixels were thrown away, and how many.

        Every figure that pools pixels quotes this. A masking choice that is not
        written next to the number it produced is a masking choice that gets
        re-litigated six weeks later.
        """
        keep = 100.0 * float(self.interior.mean())
        border = (f"{PATCH_BORDER_PX} px frame border dropped both sides "
                  f"({keep:.0f} % of the patch kept)" if self.is_patch
                  else "no frame-border cut (full geometry: the frame edge is a "
                       "real boundary both sides share)")
        return (f"masking: {GRAD_ERODE_PX} px land erosion before any "
                f"derivative; {border}")

    def title(self) -> str:
        m = self.meta
        return (f"{self.run_name()}  step {m['step']:,}  |  {self.size} "
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
        lo, hi = _limits(real, S.mask_ocean, sym)
        for c in range(ncol):
            for half, arr in ((0, gen), (1, real)):
                ax = fig.add_subplot(gs[r, c + half * (ncol + 1)])
                im = ax.imshow(_masked(arr[c], S.mask_ocean), origin="lower", cmap=cmap,
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
        lo, hi = _limits(S.phys("real", v), S.mask_ocean, sym)
        arr = S.phys(which, v)
        for c in range(ncol):
            ax = axes[r][c]
            im = ax.imshow(_masked(arr[c % S.n], S.mask_ocean), origin="lower", cmap=cmap,
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
    tile = K.psd_tile(S.mask_ocean, S.ny, S.nx)
    stride = tile // 2
    nclean = int(S.real_clean.sum())
    if nclean < S.n:
        print(f"[diag] spectra: real side uses {nclean}/{S.n} land-free samples")
    if nclean < 16:
        print(f"[diag] WARNING: only {nclean} land-free real samples -- the "
              "real spectrum is a small-sample estimate. Raise --n in "
              "eval.gen_prior, or read the full geometry instead.")
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
        k, pg = K.mean_radial_psd(S.anom("gen", v), S.mask_ocean, S.dx_km, tile,
                                  stride)
        # An FFT needs a whole rectangle, so the real side cannot mask land
        # pixel-by-pixel the way fig_pdf does -- it has to drop the samples that
        # contain any. See Samples.real_clean for what land does to a spectrum.
        _, pr = K.mean_radial_psd(S.anom("real", v)[S.real_clean], S.mask_ocean,
                                  S.dx_km, tile, stride)
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
    clean_note = ("" if nclean == S.n
                  else f", real side from the {nclean}/{S.n} land-free samples")
    fig.suptitle(f"Radial power spectra of anomalies ({tile}px tiles"
                 f"{clean_note})  --  " + S.title(), fontsize=11)
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
            x = K.ocean_values(S.anom(which, v),
                               S.real_mask if which == "real" else S.stat_mask,
                               seed=1)
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
    ug = np.empty_like(ssh); vg = np.empty_like(ssh)
    for i in range(ssh.shape[0]):
        ug[i], vg[i] = K.geostrophic_uv(ssh[i], S.dx_m, S.lat)
    # prior_genda_masked is a three-channel prior (ssh/sst/chl) with no
    # ageostrophic velocity to add, so there is no "total" to report -- the
    # geostrophic part IS the whole of what this prior says about the flow.
    if not S.has("uag", "vag"):
        return (ug, vg), None, None
    uag, vag = S.anom(which, "uag"), S.anom(which, "vag")
    return (ug, vg), (uag, vag), (ug + uag, vg + vag)


# A pixel counts as sitting on a one-cell STEP when the two-cell difference
# across it is at least this multiple of half the four-cell difference. A
# resolved front scores ~1 (the four-cell difference is twice the two-cell one);
# a true discontinuity scores ~2, because both differences span the same jump.
STEP_SCORE = 1.9


def _step_rate(S: Samples, which: str, mask: np.ndarray) -> float:
    """Fraction of masked pixels on a one-cell step in EVERY channel at once.

    This is the only statistic here that separates a drawn coastline from a
    sharp ocean front. A front is resolved -- it spans two or three cells, and
    it is sharp in SST and CHL while SSH stays smooth across it. A coastline is
    a discontinuity, and it steps in all three channels at the same pixel
    because the loader wrote the same 0.0 into all three. So: score each channel
    for one-cell-ness where its jump is large, then take the MINIMUM over
    channels.

    It separates in patch geometry and does not at full frame -- see the "drawn
    coastlines" gotcha in eval/README.md, which also says why the full-frame
    real side is not a clean control.
    """
    sc = None
    for v in S.vars:
        a = S.anom(which, v)
        best = np.zeros_like(a)
        for ax in (1, 2):
            d2 = np.abs(np.roll(a, -1, ax) - np.roll(a, 1, ax))
            d4 = np.abs(np.roll(a, -2, ax) - np.roll(a, 2, ax))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(d4 > 0, 2.0 * d2 / d4, 0.0)
            # Only where the jump is big for this field: the ratio is unstable
            # and meaningless in a flat region, where d2 and d4 are both noise.
            big = d2 > np.quantile(d2[mask], 0.9)
            best = np.maximum(best, np.where(big, r, 0.0))
        sc = best if sc is None else np.minimum(sc, best)
    return float(np.mean(sc[mask] > STEP_SCORE))


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

    Both masks drop ``GRAD_ERODE_PX`` = 6 px around land, which is where the
    land-edge EKE rim actually reaches background, and in patch geometry both
    drop a ``PATCH_BORDER_PX`` = 16 px frame border, which is where the
    generated side's zero-padding halo dies out. The two corrections push the
    ratio in opposite directions and largely cancel; the point of making them is
    the MAPS, which were showing coastline outlines on the real side and a
    saturated frame on the generated one. Read the constants for the numbers.
    """
    m = S.grad_mask
    rm = S.real_grad_mask                                 # (N, H, W)
    res = {}
    fields = {}
    for which in ("gen", "real"):
        geo, ageo, tot = _velocities(S, which)
        fields[which] = {"geostrophic": K.eke(*geo)}
        if ageo is not None:
            fields[which]["ageostrophic"] = K.eke(*ageo)
            fields[which]["total"] = K.eke(*tot)
        for part, arr in fields[which].items():
            sel = arr[rm] if which == "real" else arr[:, m]
            res[f"eke_{part}_{which}"] = float(np.nanmean(sel))
    parts = list(fields["gen"])
    for part in parts:
        res[f"eke_{part}_ratio"] = res[f"eke_{part}_gen"] / res[f"eke_{part}_real"]
    gm3 = np.broadcast_to(m, (S.n, S.ny, S.nx))
    res["step_rate_gen"] = _step_rate(S, "gen", gm3)
    res["step_rate_real"] = _step_rate(S, "real", rm)
    res["step_rate_ratio"] = (res["step_rate_gen"]
                              / max(res["step_rate_real"], 1e-12))
    # The map and histogram panels show the most complete energy this prior has.
    main = "total" if "total" in parts else "geostrophic"

    # Generated side: average over samples FIRST, then mask. Masking first
    # would leave every masked pixel as a column of all-NaN, and averaging that
    # is a mean of an empty slice -- same numbers, but a screenful of warnings.
    gmap = _masked(fields["gen"][main].mean(0), m)
    # Real side: each sample has its own mask, so a plain mean(0) is exactly
    # the bug this masking exists to fix. Sum/count keeps a pixel's mean over
    # the samples where it is valid ocean and divides all-masked pixels to NaN
    # without an empty-slice warning.
    cnt = rm.sum(0)
    rsum = np.where(rm, fields["real"][main], 0.0).sum(0)
    rmap = np.where(cnt > 0, rsum / np.maximum(cnt, 1), np.nan)
    # Scale to BOTH maps, not just the real one. When the real side is clean its
    # 99th percentile can sit below the generated side's mean -- masking the
    # under-masked islands dropped it from 0.078 to 0.050 while the generated
    # mean is 0.064 -- and a real-only vmax then saturates both panels to a flat
    # block of colour.
    both = np.concatenate([rmap[np.isfinite(rmap)], gmap[np.isfinite(gmap)]])
    vmax = float(np.percentile(both, 99))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2))
    for ax, (arr, ttl) in zip(axes[0][:2],
                              ((gmap, "generated"), (rmap, "real"))):
        im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_title(f"mean {main} EKE, {ttl}", fontsize=10)
        side = "gen" if ttl == "generated" else "real"
        lab = []
        if S.fold_note():
            lab.append(f"folded at {S.crop_stride} px (crop lattice)")
        lab.append(f"one-cell steps in all channels: "
                   f"{res[f'step_rate_{side}'] * 1e4:.1f} per 10k px")
        ax.set_xlabel("  |  ".join(lab), fontsize=7, color="0.35")
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
    ax.set_xlabel(f"zonal-mean {main} EKE [m2/s2]", fontsize=8)
    ax.set_ylabel("latitude [degN]" if varies else "grid row (south -> north)",
                  fontsize=8)
    ax.set_title("meridional structure", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=7)

    ax = axes[1][1]
    per_sample = {
        "gen": np.nanmean(fields["gen"][main][:, m], axis=1),
        "real": (np.where(rm, fields["real"][main], 0.0).sum((1, 2))
                 / np.maximum(rm.sum((1, 2)), 1)),
    }
    # Shared bin edges: per-series bins would put the two histograms on
    # different grids and make an eyeball comparison meaningless.
    allv = np.concatenate(list(per_sample.values()))
    bins = np.linspace(allv.min(), allv.max(), 21)
    for which, col in (("real", "k"), ("gen", "crimson")):
        ax.hist(per_sample[which], bins=bins, histtype="step", color=col,
                lw=1.6, label=which)
    ax.set_xlabel(f"domain-mean {main} EKE [m2/s2]", fontsize=8)
    ax.set_ylabel("samples", fontsize=8)
    ax.set_title("spread across samples", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=7)

    ax = axes[1][2]
    xs = np.arange(len(parts))
    ax.bar(xs - 0.19, [res[f"eke_{p}_real"] for p in parts], 0.36, color="0.25",
           label="real")
    ax.bar(xs + 0.19, [res[f"eke_{p}_gen"] for p in parts], 0.36, color="crimson",
           label="generated")
    ax.set_xticks(xs); ax.set_xticklabels(parts, fontsize=8)
    ax.set_xlim(-0.6, len(parts) - 0.4)
    ax.set_ylabel("domain-mean EKE [m2/s2]", fontsize=8)
    ax.set_title("energy budget", fontsize=10)
    for i, p in enumerate(parts):
        ax.text(i, max(res[f"eke_{p}_real"], res[f"eke_{p}_gen"]),
                f"x{res[f'eke_{p}_ratio']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, axis="y", lw=0.4)
    ax.tick_params(labelsize=7)

    src = ("geostrophy from the ssh anomaly + the uag/vag channels"
           if "total" in parts else
           "geostrophy from the ssh anomaly (no ageostrophic channels in this prior)")
    fold = ("\n" + S.fold_note()) if S.fold_note() else ""
    fig.suptitle(f"Eddy kinetic energy: {src}  --  " + S.title()
                 + "\n" + S.mask_note() + fold, fontsize=11)
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
    cg = K.corr_matrix(S.gen_a, S.stat_mask, seed=2)
    cr = K.corr_matrix(S.real_a, S.real_mask, seed=2)
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
    # Panels (d)-(e) test ssh-implied geostrophy against the drawn ageostrophic
    # velocity. A prior without uag/vag channels has no such relationship to
    # test, so the correlation matrices above stand alone.
    if not S.has("ssh", "uag", "vag"):
        for ax in axes[1]:
            ax.axis("off")
        axes[1][0].text(
            0.0, 0.5,
            "No geostrophic/ageostrophic panel:\nthis prior has no uag/vag "
            "channels,\nso there is no drawn ageostrophic\nvelocity to test "
            "its own ssh against.\n\nThe correlation matrices above are the\n"
            "whole of the cross-channel check here.\n\n"
            f"Frobenius |gen - real| = {res['corr_frobenius_error']:.3f}\n"
            f"largest single error   = {res['corr_max_abs_error']:.3f}",
            fontsize=9.5, va="center")
        fig.suptitle("Cross-channel structure  --  " + S.title(), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return res

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
         f"- {S.mask_note()}",
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
        if f"eke_{part}_real" not in eke:
            continue                        # prior without uag/vag channels
        L.append(f"| {part} EKE [m2/s2] | {eke[f'eke_{part}_real']:.4f} | "
                 f"{eke[f'eke_{part}_gen']:.4f} | {eke[f'eke_{part}_ratio']:.2f} |")
    if "eke_total_real" not in eke:
        L += ["", "Geostrophic only: this prior carries no ageostrophic "
              "velocity channels, so there is no total to report."]

    L += ["", "## Cross-channel", "",
          f"- Frobenius norm of the correlation-matrix error: "
          f"**{cross['corr_frobenius_error']:.3f}** (0 = perfect)",
          f"- largest single correlation error: {cross['corr_max_abs_error']:.3f}",
          *([f"- corr(geostrophic u from ssh, uag): real "
             f"{cross['r_ug_uag_real']:+.3f}, generated "
             f"{cross['r_ug_uag_gen']:+.3f}"]
            if "r_ug_uag_real" in cross else []),
          "", "## How to read this", "",
          "- **std ratio well below 1** means the prior is under-dispersed: it "
          "draws samples that are too smooth/too weak overall.",
          "- **PSD ratio falling with wavelength** means the small scales are "
          "missing (over-smoothed); rising means grid-scale noise.",
          "- **EKE ratio** is the most demanding scalar here: it depends on the "
          "ssh gradient, so it punishes both over-smoothing and noise.",
          "- **Frobenius correlation error** is the joint-prior check. Good "
          "marginals with a bad correlation matrix means the model has learned "
          f"{S.c} fields but not one ocean.", "",
          "## What was masked, and why", "",
          f"- **{GRAD_ERODE_PX} px land erosion** before any spatial "
          "derivative, up from 2 px, and measured as a Euclidean distance "
          "rather than by iterated four-neighbour erosion (which is a diamond: "
          "at 6 it cleared the diagonals to only ~4 px, and the 4-6 px ring it "
          "left behind still had mean EKE 0.075 against 0.032 at 8-10 px). Real "
          "EKE binned by distance from that patch's own coast runs 80x the far "
          "field at 1-2 px, 10x at 2-3 px, 3x at 3-4 px and 2x at 4-6 px. 6 px "
          "is where the answer stops moving: the real patch mean is 0.0438 at "
          "2 px, 0.0408 at 6 and 0.0409 at 10, then drifts up to 0.0426 at 24 "
          "as the survivors become a deep-basin subsample rather than a cleaner "
          "one.",
          (f"- **{PATCH_BORDER_PX} px frame border excluded**, both sides. A "
           "generated patch has no land, but the UNet's zero padding inflates "
           "EKE by ~32 % within 2-4 px of the frame (real is flat there), and "
           "the generated mean-EKE map's split-half reproducibility falls from "
           "r = 0.78 at a 2 px cut to 0.29 at 32 px -- the real side's own "
           "level. The reproducible structure was the border, not geography."
           if S.is_patch else
           "- **No frame-border cut**: at full frame the edge is a real domain "
           "boundary that generated and real fields share."),
          "- **The spectra are exempt from the border cut.** An FFT needs a "
          "whole rectangle, and `radial_psd` already applies a 2-D Hann window "
          "that weights the outer 16 px of a 128 px tile below 0.15 in "
          "amplitude. The real side instead drops land-contaminated samples "
          "whole (`real_clean`); the figure title says how many survived.",
          *([f"- **The two EKE maps are folds, not maps.** Real patches come "
             f"off a {S.crop_stride} px crop lattice, so patch pixel (i, j) and "
             f"(i + {S.crop_stride}, j) average nearly the same absolute "
             "positions and the sample-mean map repeats the domain's geography "
             f"at {S.crop_stride} px. The generated side shows the same period "
             "-- it was trained on that lattice -- and the two folds correlate "
             "at +0.41 against +0.09 for what is left after removing them. The "
             "shift autocorrelation is 0.02 at 24 px, 0.67 at 32 px, 0.05 at "
             "36 px on the real side. Read the level of those panels and the "
             "log-ratio, not the pattern; the fold amplitude is +-13 % of the "
             "mean on the real side and +-24 % on the generated one."]
            if S.is_patch and S.crop_stride else []),
          "", "## Drawn coastlines (generated side)", "",
          "The bright closed rings in the generated EKE map are the model's "
          "own work, not a masking artifact: it draws coastline-shaped "
          "discontinuities in open ocean -- closed curves across which SSH, "
          "SST and CHL all step in a single cell. `prior_genda_masked` is the "
          "arm that KEEPS land in the training distribution "
          "(`reject_land: false`, `loss_reduction: masked_mean`), so the "
          "coastline is part of what it was asked to learn to draw.", "",
          f"- one-cell steps in every channel at once: **"
          f"{eke['step_rate_gen'] * 1e4:.1f} per 10k ocean px** generated, "
          f"{eke['step_rate_real'] * 1e4:.1f} real "
          f"(x{eke['step_rate_ratio']:.1f}).",
          "- This statistic separates the two sides in PATCH geometry and does "
          "not at full frame. Do not read a full-frame ratio near 1 as absence: "
          "the real full-frame field is not a clean control, because the store "
          "carries ~37 isolated locations of bad pixels that are neither land "
          "nor ocean signal and that imply up to 6.7 m/s of geostrophic "
          "velocity. See eval/README.md.",
          "- It does not drive the energy scalar. Dropping the top 1 % of "
          "joint-gradient pixels moves the full-frame EKE ratio the wrong way "
          "(1.65 -> 1.69): the excess is broadly distributed, and the drawn "
          "coastlines are a separate, smaller defect that dominates the MAP.", "",
          "- The two corrections move the EKE ratio in opposite directions "
          "(erosion lowers the real side, the border cut lowers the generated "
          "side) and largely cancel. They were made for the maps, which were "
          "misleading, not for the scalar, which was not.", ""]

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
