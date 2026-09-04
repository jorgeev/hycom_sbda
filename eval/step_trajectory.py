"""How the prior's spectra converge on the truth as training proceeds.

``diagnostics.py`` answers "is this checkpoint any good". This answers the
question a *partial* run actually poses -- **is it still getting better, or has
it stopped paying for the GPU hours** -- by laying every checkpoint's radial PSD
on one axes and reducing each to a scalar distance from the real spectrum.

It exists because the validation loss cannot answer that. On
``prior_genda_masked`` the val curve is flat and noisy from ~880k steps onward
(0.054-0.065, no trend), which is consistent both with a converged model and
with one whose EDM loss is dominated by high-sigma noise levels that carry no
mesoscale information. The spectra separate those two cases: a model still
learning keeps moving power into the 10-40 km band long after the scalar loss
has gone quiet.

INPUT is whatever ``eval.gen_prior`` already wrote -- this script draws no
samples and opens no store. Point it at a run directory and it finds every
``samples_<size>.npz`` beneath it, reads the training step out of each npz's
``meta``, and keeps only the points whose sampler settings and weights match the
newest one, so the single variable across the curve is the training step. A
sampler sweep living in the same run (``sweep/s64`` and friends) is therefore
skipped automatically rather than being silently plotted as if it were a
training-step effect.

OUTPUT, per geometry:
  fig_spectra_vs_step.png    PSD per variable, real in black, one coloured
                             curve per checkpoint
  fig_ratio_vs_step.png      the same as generated/real ratio, where "too
                             smooth" is unambiguous: below 1
  fig_bands_vs_step.png      the scalars -- band ratios and spectral distance
                             against training step. This is the plot that says
                             continue or stop.
  trajectory.csv            every number in those figures

The distance is RMS of log10(gen/real) over the resolved band (2dx to the
training-patch span), so a factor-2 deficit and a factor-2 excess count the
same, and it is 0 for a perfect match.

Usage:
    python -m eval.step_trajectory --run <run_dir>
    python -m eval.step_trajectory --run <run_dir> --size 128
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from . import kernels as K
from .diagnostics import BANDS, Samples, _unit


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def collect(run: str, size: str) -> list[tuple[int, str]]:
    """``(step, npz_path)`` for the training ladder under ``run``, sorted by step.

    Two filters, in this order, and the order matters. First keep only the
    points whose sampler settings and weights match the MODAL configuration:
    a run directory that also holds a sampler sweep has several npz files at
    the same training step differing only in ``s_churn``, and plotting those
    against step would read as a training effect and be wrong. The modal key is
    the right reference because a sweep visits each sampler setting once while
    the training ladder shares one setting across every checkpoint.

    Only then deduplicate by step -- doing it the other way round can seat a
    sweep variant as the reference purely by directory walk order.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(run):
        name = f"samples_{size}.npz"
        if name not in filenames:
            continue
        path = os.path.join(dirpath, name)
        try:
            with np.load(path, allow_pickle=False) as d:
                m = json.loads(str(d["meta"]))
            s = m.get("sampler", {})
            key = (m.get("weights", "ema"), s.get("num_steps"),
                   s.get("s_churn"), s.get("s_noise"))
            found.append((int(m["step"]), path, key))
        except Exception as exc:                       # noqa: BLE001
            print(f"[traj] skipping unreadable {path}: {exc}")
    if not found:
        return []

    counts = {}
    for _step, _path, key in found:
        counts[key] = counts.get(key, 0) + 1
    ref = max(counts, key=lambda k: (counts[k], max(
        st for st, _p, kk in found if kk == k)))
    dropped = [p for _s, p, k in found if k != ref]
    if dropped:
        print(f"[traj] reference sampler weights={ref[0]} steps={ref[1]} "
              f"s_churn={ref[2]} s_noise={ref[3]}; dropped {len(dropped)} "
              "point(s) that do not match it")

    keep: dict[int, str] = {}
    for step, path, key in sorted(found, key=lambda t: t[1]):
        if key == ref:
            keep.setdefault(step, path)
    return sorted(keep.items())


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def spectra_for(path: str) -> dict:
    """Median gen/real radial PSD per variable for one samples npz.

    Tiling matches ``diagnostics.fig_spectra`` exactly -- 256 px tiles at
    stride 128 where the domain allows, the whole frame otherwise -- so a curve
    here and the corresponding per-checkpoint figure are the same numbers.
    """
    S = Samples(path)
    tile = K.psd_tile(S.mask, S.ny, S.nx)
    out = {"step": int(S.meta["step"]), "vars": list(S.vars),
           "dx_km": S.dx_km, "run": S.run_name(), "n": S.n,
           "patch_km": 128 * S.dx_km, "psd": {}}
    out["n_clean"] = int(S.real_clean.sum())
    for v in S.vars:
        k, pg = K.mean_radial_psd(S.anom("gen", v), S.mask, S.dx_km, tile, tile // 2)
        # Land-free real samples only -- identical to diagnostics.fig_spectra,
        # and for the same reason (Samples.real_clean).
        _, pr = K.mean_radial_psd(S.anom("real", v)[S.real_clean], S.mask,
                                  S.dx_km, tile, tile // 2)
        out["k"] = k
        out["psd"][v] = (np.median(pg, 0), np.median(pr, 0))
    return out


def log_distance(k, pg, pr, lo_km, hi_km) -> float:
    """RMS of log10(gen/real) over a wavelength band -- 0 is a perfect match."""
    lam = 1.0 / k
    sel = (lam >= lo_km) & (lam <= hi_km) & (pg > 0) & (pr > 0)
    if not sel.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.log10(pg[sel] / pr[sel]) ** 2)))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _grid(nvar):
    ncol = min(3, nvar)
    nrow = -(-nvar // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.2 * nrow),
                             squeeze=False)
    return fig, axes.ravel()


def _step_colors(steps):
    norm = colors.Normalize(vmin=min(steps) / 1e6, vmax=max(steps) / 1e6)
    return cm.viridis, norm


def fig_spectra(runs: list[dict], out: str, title: str):
    steps = [r["step"] for r in runs]
    cmap, norm = _step_colors(steps)
    fig, axes = _grid(len(runs[-1]["vars"]))
    for ax, v in zip(axes, runs[-1]["vars"]):
        ref = runs[-1]
        lam = 1.0 / ref["k"]
        ax.plot(lam, ref["psd"][v][1], color="k", lw=2.4, label="real", zorder=5)
        for r in runs:
            ax.plot(1.0 / r["k"], r["psd"][v][0], lw=1.3,
                    color=cmap(norm(r["step"] / 1e6)))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.axvline(2 * ref["dx_km"], color="0.5", ls=":", lw=1)
        ax.axvline(ref["patch_km"], color="tab:blue", ls="--", lw=1)
        ax.set_title(f"{v}  [{_unit(v)}]", fontsize=10)
        ax.set_xlabel("wavelength [km]", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8, frameon=False)
    for ax in axes[len(runs[-1]["vars"]):]:
        ax.axis("off")
    _finish(fig, axes, cmap, norm, title, out)


def fig_ratio(runs: list[dict], out: str, title: str):
    steps = [r["step"] for r in runs]
    cmap, norm = _step_colors(steps)
    fig, axes = _grid(len(runs[-1]["vars"]))
    for ax, v in zip(axes, runs[-1]["vars"]):
        ref = runs[-1]
        for r in runs:
            pg, pr = r["psd"][v]
            ax.plot(1.0 / r["k"], pg / pr, lw=1.3,
                    color=cmap(norm(r["step"] / 1e6)))
        ax.axhline(1.0, color="k", lw=1.2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.axvline(2 * ref["dx_km"], color="0.5", ls=":", lw=1)
        ax.axvline(ref["patch_km"], color="tab:blue", ls="--", lw=1)
        for (name, lo, hi), col in zip(BANDS, ("tab:green", "tab:purple")):
            ax.axvspan(lo, hi, color=col, alpha=0.09, lw=0)
        ax.set_title(f"{v}", fontsize=10)
        ax.set_xlabel("wavelength [km]", fontsize=8)
        ax.set_ylabel("PSD ratio gen / real", fontsize=8)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=7)
    for ax in axes[len(runs[-1]["vars"]):]:
        ax.axis("off")
    _finish(fig, axes, cmap, norm, title, out)


def _finish(fig, axes, cmap, norm, title, out):
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 0.92, 0.95))
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=list(axes), fraction=0.02, pad=0.015)
    cb.set_label("training step [M]", fontsize=9, labelpad=2)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[traj] wrote {out}")


def fig_bands(rows: list[dict], variables: list[str], out: str, title: str):
    """The scalar convergence curves: band ratio and spectral distance vs step."""
    npanel = len(BANDS) + 1
    fig, axes = plt.subplots(1, npanel, figsize=(5.0 * npanel, 4.2), squeeze=False)
    axes = axes.ravel()
    steps = sorted({r["step"] for r in rows})
    for ax, (name, _lo, _hi) in zip(axes, BANDS):
        for v in variables:
            y = [_pick(rows, s, v, f"band_{name.split()[0]}") for s in steps]
            ax.plot([s / 1e6 for s in steps], y, "o-", lw=1.5, ms=4, label=v)
        ax.axhline(1.0, color="k", lw=1, ls=":")
        # Log y: a ratio is a log quantity, and on a linear axis a single
        # channel that is 250x off (gulfstream's tau, whose real 4dx spike the
        # prior cannot reproduce) squashes every other channel onto the 1.0
        # line and the figure says nothing.
        ax.set_yscale("log")
        ax.set(xlabel="training step [M]", ylabel="PSD ratio gen / real",
               title=name)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    ax = axes[-1]
    for v in variables:
        y = [_pick(rows, s, v, "logdist") for s in steps]
        ax.plot([s / 1e6 for s in steps], y, "o-", lw=1.5, ms=4, label=v)
    ax.set(xlabel="training step [M]",
           ylabel="RMS log10(gen/real) over resolved band",
           title="spectral distance (0 = perfect)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    # One legend for the figure, below the panels: inside the first axes it sat
    # on top of the curves it was labelling.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 8), frameon=False,
               fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[traj] wrote {out}")


def _pick(rows, step, var, key):
    for r in rows:
        if r["step"] == step and r["var"] == var:
            return r[key]
    return np.nan


# ---------------------------------------------------------------------------
def run_size(run: str, size: str, outdir: str) -> list[dict]:
    points = collect(run, size)
    if len(points) < 2:
        print(f"[traj] {size}: {len(points)} comparable eval(s) found -- need "
              "at least 2, skipping")
        return []
    print(f"[traj] {size}: {len(points)} checkpoints -- "
          + ", ".join(f"{s // 1000}k" for s, _p in points))

    runs = [spectra_for(p) for _s, p in points]
    variables = runs[-1]["vars"]
    ref = runs[-1]
    resolved = (2 * ref["dx_km"], ref["patch_km"])

    rows = []
    for r in runs:
        for v in variables:
            pg, pr = r["psd"][v]
            row = {"run": r["run"], "size": size, "step": r["step"], "var": v,
                   "n": r["n"], "n_real_clean": r["n_clean"],
                   "logdist": log_distance(r["k"], pg, pr, *resolved)}
            for name, lo, hi in BANDS:
                row[f"band_{name.split()[0]}"] = K.band_ratio(r["k"], pg, pr, lo, hi)
            rows.append(row)

    os.makedirs(outdir, exist_ok=True)
    head = (f"{ref['run']}  --  {size} geometry, N={ref['n']} per checkpoint, "
            f"{len(runs)} checkpoints")
    fig_spectra(runs, os.path.join(outdir, "fig_spectra_vs_step.png"),
                "Radial power spectra vs training step  --  " + head)
    fig_ratio(runs, os.path.join(outdir, "fig_ratio_vs_step.png"),
              "Spectral ratio vs training step  --  " + head)
    fig_bands(rows, variables, os.path.join(outdir, "fig_bands_vs_step.png"),
              "Spectral convergence vs training step  --  " + head)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory to scan")
    ap.add_argument("--size", default="full,128",
                    help="comma-separated geometries (default 'full,128')")
    ap.add_argument("--out", default=None,
                    help="output directory (default <run>/traj/figs_<size>)")
    args = ap.parse_args()

    all_rows = []
    for size in [s.strip() for s in args.size.split(",") if s.strip()]:
        outdir = (os.path.join(args.out, f"figs_{size}") if args.out
                  else os.path.join(args.run, "traj", f"figs_{size}"))
        all_rows += run_size(args.run, size, outdir)
    if not all_rows:
        raise SystemExit(f"no comparable eval directories under {args.run}")

    csv_path = os.path.join(args.out or os.path.join(args.run, "traj"),
                            "trajectory.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[traj] wrote {csv_path} ({len(all_rows)} rows)")

    for size in sorted({r["size"] for r in all_rows}):
        sub = [r for r in all_rows if r["size"] == size]
        print(f"\n== {size} ==")
        print(f"{'step':>9} {'var':>6} " +
              " ".join(f"{n.split()[0]:>10}" for n, _l, _h in BANDS) +
              f"{'logdist':>10}")
        for r in sorted(sub, key=lambda r: (r["step"], r["var"])):
            print(f"{r['step']:>9} {r['var']:>6} " +
                  " ".join(f"{r[f'band_{n.split()[0]}']:>10.3f}"
                           for n, _l, _h in BANDS) +
                  f"{r['logdist']:>10.3f}")


if __name__ == "__main__":
    main()
