"""Movies of the ageostrophic current components (uag, vag) with an
OBSERVATION-DERIVED reference in the observation slots.

The ageostrophic components are never observed, so ``sda.diagnostics_sda``'s
per-variable movie shows empty observation frames for them. Here those two
frames hold the geostrophic velocity computed from the OBSERVED sea-surface
height anomaly of the same case (``eval.kernels.geostrophic_uv``):

  row 0, col 0 : geostrophic u (or v) from the AVISO-like SSHA with the SWOT
                 swath values inserted where a swath exists ("AVISO+SWOT");
  row 1, col 0 : the same from the SWOT swath alone (NaN elsewhere).

Both SSHA terms are demeaned anomalies in the case file; the swath is re-levelled
to the AVISO field under it before insertion so the two share one mean. The rest
of rows 0-1 is the stock layout (truth, guided mean, control mean; spread and
|mean - truth|), and ``--members N`` extra rows show N individual guided members
(four per row). Member/truth/mean panels are in the variable's own sigma units;
the observation-derived panels are in the sigma units of the corresponding
GEOSTROPHIC component (``ug``/``vg`` std from the manifest, ``--manifest``), so
that a feature of comparable energy looks comparable. The two are different
quantities: the point of the panel is to see whether the structure SWOT resolves
(fronts, eddy rims) is present in the members at all, not to score them.

``--reference residual`` (the default) replaces the two observation slots with the
SWOT-minus-AVISO SSHA residual: row 0 holds SWOT SSHA minus the AVISO-like field
under the swath (what SWOT observes that AVISO lacks: SSH content between SWOT's
resolution and the 65 km blur), row 1 holds the guided-mean SSH minus its own
65 km blur over the whole frame (the ensemble's SWOT-scale content, same
operator), both in ssh sigma units. ``--reference geo`` gives the geostrophic
velocity panels described above.

Whenever a SWOT swath exists at the shown hour, its outline is drawn on every
panel (thin black line, ``--no-outline`` to drop it) so the structure inside
the observed strip can be compared with what the swath resolved.

Usage:
  python -m sda.animate_currents --samples <arm>/samples_sda_full.npz \
      [--control <arm>/samples_sda_full_free.npz] --out <arm>/figs_sda_full \
      [--vars uag,vag] [--members 8] [--fps 6]
Writes anim_<var>_obsgeo.mp4 and fig_<var>_obsgeo.png (the frame with the most
swath pixels) into --out. Reads only; nothing in the samples is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_erosion, gaussian_filter  # noqa: E402

from eval import kernels as K  # noqa: E402
from sda.diagnostics_sda import SdaSamples, _stamp  # noqa: E402

DEFAULT_MANIFEST = "/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/datasets/gulfstream_manifest.json"
GEO_OF = {"uag": "ug", "vag": "vg", "u": "ug", "v": "vg"}


def manifest_std(path: str | None, var: str) -> float | None:
    path = path or os.environ.get("GULFSTREAM_MANIFEST", DEFAULT_MANIFEST)
    try:
        m = json.load(open(path))
        return float(m["normalization"][var]["std"])
    except Exception:
        return None


def observed_eta(S: SdaSamples, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical SSHA [m] from the case's observations at step d.

    Returns ``(eta_combined, eta_swot_only, swath_mask)``; NaN where nothing is
    observed. The AVISO term is a blurred, demeaned anomaly over the frame; the
    SWOT term a demeaned anomaly over the swath. Both are stored in ssh sigma
    units, so x std[ssh] gives metres."""
    ci = S.vars.index("ssh")
    sd = float(S.std[ci])
    blur = S.terms_of("ssh", "blur")
    point = S.terms_of("ssh", "pointwise")
    ny, nx = S.ny, S.nx
    eta = np.full((ny, nx), np.nan, np.float32)
    for i, _ in blur:
        sel = S.obs_mask[d, i]
        eta[sel] = S.y_grid[d, i][sel] * sd
    swot = np.full((ny, nx), np.nan, np.float32)
    swath = np.zeros((ny, nx), bool)
    for i, _ in point:
        sel = S.obs_mask[d, i]
        swot[sel] = S.y_grid[d, i][sel] * sd
        swath |= sel
    comb = eta.copy()
    if swath.any():
        # re-level the swath onto the AVISO field beneath it (both are demeaned,
        # over different supports) before inserting it
        under = eta[swath]
        if np.isfinite(under).any():
            swot_lev = swot[swath] - np.nanmean(swot[swath]) + np.nanmean(under)
        else:
            swot_lev = swot[swath]
        comb[swath] = swot_lev
    return comb, swot, swath


def geo_component(eta: np.ndarray, dx_m: float, lat2d: np.ndarray, comp: str,
                  valid: np.ndarray | None = None, smooth_px: float = 2.0) -> np.ndarray:
    """``ug`` or ``vg`` [m/s] from an SSHA field with NaN gaps: the gradient is
    taken on a lightly smoothed, gap-aware copy and masked back to pixels whose
    5x5 stencil is fully observed (so no edge of a gap leaks in)."""
    ok = np.isfinite(eta) if valid is None else (valid & np.isfinite(eta))
    # the store's own ug/vg are taken from a 9x9 Gaussian-smoothed SSH (manifest
    # notes); do the same (sigma 2 px, gap-aware normalised convolution) so the
    # SWOT swath's pixel noise does not dominate the gradient
    num = gaussian_filter(np.where(ok, eta, 0.0).astype(np.float64), smooth_px, truncate=2.0)
    den = gaussian_filter(ok.astype(np.float64), smooth_px, truncate=2.0)
    eta_s = np.where(den > 0.5, num / np.maximum(den, 1e-6), 0.0)
    ug, vg = K.geostrophic_uv(eta_s, dx_m, lat2d)
    g = ug if comp == "ug" else vg
    inner = binary_erosion(ok, structure=np.ones((5, 5), bool), border_value=0)
    return np.where(inner, g, np.nan).astype(np.float32)


def swot_minus_aviso(S: SdaSamples, d: int) -> tuple[np.ndarray, np.ndarray]:
    """SWOT SSHA minus the AVISO-like field beneath the swath, in ssh sigma
    units, NaN off-swath. Both terms are demeaned anomalies over different
    supports, so the swath is first re-levelled onto the AVISO field under it
    (then the difference has zero mean over the swath by construction)."""
    ci = S.vars.index("ssh")
    ny, nx = S.ny, S.nx
    aviso = np.full((ny, nx), np.nan, np.float32)
    for i, _ in S.terms_of("ssh", "blur"):
        sel = S.obs_mask[d, i]
        aviso[sel] = S.y_grid[d, i][sel]
    swot = np.full((ny, nx), np.nan, np.float32)
    swath = np.zeros((ny, nx), bool)
    for i, _ in S.terms_of("ssh", "pointwise"):
        sel = S.obs_mask[d, i]
        swot[sel] = S.y_grid[d, i][sel]
        swath |= sel
    res = np.full((ny, nx), np.nan, np.float32)
    both = swath & np.isfinite(aviso) & np.isfinite(swot)
    if both.any():
        diff = swot[both] - aviso[both]
        res[both] = diff - diff.mean()
    return res, swath


def aviso_sigma_px(S: SdaSamples) -> float:
    for _, t in S.terms_of("ssh", "blur"):
        if t.get("sigma_px"):
            return float(t["sigma_px"])
    return 65.0 / S.dx_km


def field_minus_blur(f: np.ndarray, ok: np.ndarray, sigma_px: float) -> np.ndarray:
    """f minus its gap-aware Gaussian blur (the AVISO operator applied to a
    model field): the field's SWOT-scale residual."""
    num = gaussian_filter(np.where(ok, f, 0.0).astype(np.float64), sigma_px, truncate=3.0)
    den = gaussian_filter(ok.astype(np.float64), sigma_px, truncate=3.0)
    blur = num / np.maximum(den, 1e-6)
    return np.where(ok, f - blur, np.nan).astype(np.float32)


def animate(S: SdaSamples, Sc: SdaSamples | None, var: str, out_dir: str, members: int,
            fps: int, manifest: str | None, still: bool = True, reference: str = "residual",
            outline: bool = True) -> str:
    import matplotlib.animation as manim

    ci = S.vars.index(var)
    comp = GEO_OF[var]
    gstd = manifest_std(manifest, comp)
    geo_units = f"{comp} sigma units" if gstd else f"{comp} m/s x10"
    gscale = (1.0 / gstd) if gstd else 10.0
    dx_m = S.dx_km * 1000.0
    ncol = 4
    mrows = int(np.ceil(members / ncol)) if members > 0 else 0
    nrow = 2 + mrows
    aspect = S.ny / S.nx
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol + 1, 3.2 * aspect * nrow + 1.0),
                            constrained_layout=True, squeeze=False)
    if reference == "residual":
        col0 = ["SWOT - AVISO SSHA (swath)", "guided-mean ssh - 65 km blur"]
        geo_units = "ssh sigma units"
    else:
        col0 = [f"{comp} from AVISO+SWOT SSHA", f"{comp} from SWOT swath"]
    titles = [[col0[0], f"{var} truth", f"{var} guided mean",
               f"{var} control mean" if Sc is not None else ""],
              [col0[1], f"{var} guided spread", f"|guided - truth|",
               "|control - truth|" if Sc is not None else ""]]
    ssh_ci = S.vars.index("ssh")
    sig_px = aviso_sigma_px(S)
    ims = {}
    blank = np.full((S.ny, S.nx), np.nan, np.float32)
    for i in range(nrow):
        for j in range(ncol):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i < 2:
                t = titles[i][j]
                if not t:
                    ax.axis("off")
                    continue
                ax.set_title(t, fontsize=9)
                if i == 1 and j >= 1:
                    ims[(i, j)] = ax.imshow(blank, origin="lower", cmap="magma", vmin=0, vmax=2)
                elif j == 0 and reference == "residual":
                    # the SWOT-scale residual is sub-sigma: its own, tighter scale
                    ims[(i, j)] = ax.imshow(blank, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
                else:
                    ims[(i, j)] = ax.imshow(blank, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
            else:
                k = (i - 2) * ncol + j
                if k >= members or k >= S.K:
                    ax.axis("off")
                    continue
                ax.set_title(f"{var} member {k + 1}", fontsize=9)
                ims[(i, j)] = ax.imshow(blank, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(ims[(0, 1)], ax=axs[0].tolist(), shrink=0.7, location="right").set_label(
        f"sigma units (col 0: {geo_units}" + (", range +-1)" if reference == "residual" else ")"))
    fig.colorbar(ims[(1, 1)], ax=axs[1].tolist(), shrink=0.7, location="right").set_label("sigma units")
    sup = fig.suptitle("", fontsize=10)
    outlines: list = []

    def draw_outline(swath):
        for c in outlines:
            try:
                c.remove()
            except Exception:
                pass
        outlines.clear()
        if not outline or not swath.any():
            return
        edge = swath.astype(np.float32)
        for (i, j) in ims.keys():
            outlines.append(axs[i, j].contour(edge, levels=[0.5], colors="k", linewidths=0.6, origin="lower"))

    def frame(d):
        land = ~S.mask_day[d]
        lat2d = S.lat[d] if getattr(S, "lat", None) is not None else 35.0
        if reference == "residual":
            g_comb, swath = swot_minus_aviso(S, d)
            mean_ssh = S.gen_var(ssh_ci, d).mean(0)
            g_swot = field_minus_blur(mean_ssh, S.mask_day[d], sig_px)
        else:
            comb, swot, swath = observed_eta(S, d)
            g_comb = geo_component(comb, dx_m, lat2d, comp) * gscale
            g_swot = geo_component(swot, dx_m, lat2d, comp, valid=swath) * gscale
        x = S.gen_var(ci, d)
        y = S.truth[d, ci]
        mean = x.mean(0)
        sd = x.std(0, ddof=1) if S.K > 1 else np.zeros_like(y)
        panels = {(0, 0): g_comb, (0, 1): y, (0, 2): mean,
                  (1, 0): g_swot, (1, 1): sd, (1, 2): np.abs(mean - y)}
        if Sc is not None:
            mc = Sc.gen_var(ci, d).mean(0)
            panels[(0, 3)] = mc
            panels[(1, 3)] = np.abs(mc - y)
        for k in range(min(members, S.K)):
            panels[(2 + k // ncol, k % ncol)] = x[k]
        for key, f in panels.items():
            if key in ims:
                ims[key].set_data(np.where(land, np.nan, f))
        draw_outline(swath)
        what = ("col 0 = SWOT - AVISO SSHA residual (top) and guided-mean ssh minus its 65 km blur (bottom), in ssh sigma units"
                if reference == "residual" else f"col 0 = geostrophic {comp} from observed SSHA")
        sup.set_text(f"{S.title()}\n{_stamp(S, d)} (sample {d}/{S.D}); swath px {int(swath.sum())}; {what}; other panels {var}")
        return list(ims.values())

    os.makedirs(out_dir, exist_ok=True)
    tag = "swotres" if reference == "residual" else "obsgeo"
    if still:
        cover = [int(observed_eta(S, d)[2].sum()) for d in range(S.D)]
        frame(int(np.argmax(cover)))
        fig.savefig(os.path.join(out_dir, f"fig_{var}_{tag}.png"), dpi=110)
    anim = manim.FuncAnimation(fig, frame, frames=S.D, blit=False)
    use_ffmpeg = "ffmpeg" in manim.writers.list()
    writer = manim.FFMpegWriter(fps=fps, bitrate=2800) if use_ffmpeg else manim.PillowWriter(fps=fps)
    out = os.path.join(out_dir, f"anim_{var}_{tag}." + ("mp4" if use_ffmpeg else "gif"))
    anim.save(out, writer=writer, dpi=110)
    plt.close(fig)
    return out


def animate_still(S, Sc, var, out_dir, members, manifest, reference, outline=True):
    """The PNG alone (same frame choice as animate)."""
    import matplotlib.animation as manim  # noqa: F401
    # reuse animate with a 1-frame movie suppressed: simplest is to call the
    # frame builder through animate's still path and then stop before saving
    _orig = manim.FuncAnimation.save
    manim.FuncAnimation.save = lambda self, *a, **k: None
    try:
        animate(S, Sc, var, out_dir, members, 1, manifest, still=True, reference=reference, outline=outline)
    finally:
        manim.FuncAnimation.save = _orig
    print(f"[animate_currents] wrote {os.path.join(out_dir, f'fig_{var}_' + ('swotres' if reference == 'residual' else 'obsgeo') + '.png')}")


# attach lat to SdaSamples instances (the stock class does not keep it)
def _load(path: str) -> SdaSamples:
    S = SdaSamples(path)
    d = np.load(path, allow_pickle=False)
    S.lat = np.asarray(d["lat"], np.float32) if "lat" in d.files else None
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vars", default="uag,vag")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--manifest", default=None, help="manifest json for the ug/vg std (default: gulfstream)")
    ap.add_argument("--reference", default="residual", choices=("residual", "geo"),
                    help="observation-slot content: SWOT-AVISO SSHA residual (default) or geostrophic velocity")
    ap.add_argument("--still-only", action="store_true", help="write the PNG, skip the movie")
    ap.add_argument("--no-outline", action="store_true", help="do not draw the swath outline on the panels")
    args = ap.parse_args(argv)
    S = _load(args.samples)
    Sc = _load(args.control) if args.control else None
    if not S.terms_of("ssh"):
        raise SystemExit("no ssh observation term in this case: nothing to derive the reference from")
    for v in args.vars.split(","):
        if v not in S.vars:
            raise SystemExit(f"{v} is not a target of this case ({S.vars})")
        if args.still_only:
            animate_still(S, Sc, v, args.out, args.members, args.manifest, args.reference, not args.no_outline)
            continue
        out = animate(S, Sc, v, args.out, args.members, args.fps, args.manifest, reference=args.reference,
                      outline=not args.no_outline)
        print(f"[animate_currents] wrote {out}")


if __name__ == "__main__":
    main()
