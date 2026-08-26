"""Overlay figure for the tau spike provenance test.

One panel per component: PSD vs wavelength (eval fig2 style -- loglog,
inverted x) for the four arms of compute_spectra.py, with the 2dx / 4dx /
patch-scale markers. A bottom panel shows each arm's ratio to the raw source
spectrum, which is the actual verdict: a ratio of ~1 at 4dx means that arm
merely inherits the spike from the source; >>1 means the regridding created it.

Run:  python tau_interp_check/plot_compare.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = (  # key, label, color, linewidth
    ("source", "raw source (native HYCOM grid, no regrid)", "k", 2.0),
    ("eval", "training store, eval fig2 curve (bilinear)", "crimson", 1.6),
    ("store", "training store recomputed (parity check)", "orange", 1.0),
    ("bicubic", "bicubic RectBivariateSpline regrid", "tab:blue", 1.6),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=os.path.join(HERE, "figs",
                                                  "tau_spectra_compare.npz"))
    ap.add_argument("--out", default=None,
                    help="default: fig_<npz basename>.png next to the npz")
    args = ap.parse_args()
    if args.out is None:
        base = os.path.basename(args.npz).replace("tau_spectra_compare",
                                                  "fig_tau_spectra_compare")
        args.out = os.path.join(os.path.dirname(args.npz),
                                base.replace(".npz", ".png"))

    d = np.load(args.npz)
    window = str(d["window"]) if "window" in d.files else "hann"
    k, dx = d["k"], float(d["dx_km"])
    lam = 1.0 / k
    patch_km = 128 * dx

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True,
                             height_ratios=[2.2, 1], layout="constrained")
    for col, v in enumerate(("tau_x", "tau_y")):
        ax, rax = axes[0, col], axes[1, col]
        for key, lab, colr, lw in ARMS:
            med = d[f"{key}_{v}_med"]
            ax.plot(lam, med, color=colr, lw=lw, label=lab,
                    ls="--" if key == "store" else "-")
            ax.fill_between(lam, d[f"{key}_{v}_q25"], d[f"{key}_{v}_q75"],
                            color=colr, alpha=0.15, lw=0)
            if key != "source":
                rax.plot(lam, med / d[f"source_{v}_med"], color=colr,
                         lw=lw, ls="--" if key == "store" else "-", label=lab)
        rax.axhline(1.0, color="k", lw=0.8)
        for a in (ax, rax):
            a.set_xscale("log")
            a.set_yscale("log")
            for x, c, ls in ((2 * dx, "0.5", ":"), (4 * dx, "tab:green", ":"),
                             (patch_km, "tab:blue", "--")):
                a.axvline(x, color=c, ls=ls, lw=1)
            a.grid(alpha=0.25, which="both", lw=0.4)
        # sharex=True: all panels share ONE x-axis, so invert exactly once
        # (once per column would toggle it straight back).
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.set_title(f"{v}  [N m$^{{-2}}$]")
        ax.set_ylabel("PSD (anomaly)")
        rax.set_ylabel("PSD / raw source")
        rax.set_xlabel("wavelength [km]")
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="lower left")
        rax.text(4 * dx, rax.get_ylim()[1], " 4dx", color="tab:green",
                 fontsize=8, va="top")
        rax.text(2 * dx, rax.get_ylim()[1], " 2dx", color="0.5", fontsize=8,
                 va="top")

    wlabel = {"hann": "2-D Hann (eval fig2 default)",
              "tukey02": "2-D Tukey alpha=0.2"}.get(window, window)
    fig.suptitle(
        f"tau spectral spike provenance -- {int(d['n'])} patches of "
        f"{128}px at dx={dx:.4f} km, val split, window: {wlabel}\n"
        "if the spike is already in the raw source curve, it is not a "
        "regridding artifact of the dataset build", fontsize=11)
    fig.savefig(args.out, dpi=150)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
