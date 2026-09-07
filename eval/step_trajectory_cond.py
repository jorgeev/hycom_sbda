"""Paired skill and calibration of a CONDITIONAL model against training step.

The conditional counterpart of ``eval.step_trajectory``. That script lays the
prior's spectra over a checkpoint ladder; this one does the same with the
numbers that only exist once every sample has a paired truth -- ensemble-mean
RMSE and CRPS, spread/skill, rank-histogram TV, coherence scale -- and asks the
question a partial run poses: **is it still getting better?**

INPUT is whatever ``eval.gen_cond`` + ``eval.diagnostics_cond`` already wrote.
Point it at a run directory; it finds every ``samples_cond_<size>.npz`` beneath
it that has a ``figs_cond_<size>/summary.csv`` beside it, reads the training
step and the sampler settings out of the npz's ``meta``, and keeps only the
points whose sampler settings, weights, K, D, split, seed AND TARGET DAYS match
the modal configuration. The target-day hash is what makes this a paired
comparison: skill on different days is not a training-step effect. A sampler
sweep in the same run (``sweep_cond/churn0`` etc.) is dropped automatically.

Nothing here opens ``gen_norm``: the per-step numbers come from ``summary.csv``
and the curves from ``spectra.npz``, both written by ``diagnostics_cond``.

OUTPUT, per geometry, in ``<run>/traj_cond/figs_<size>/``:
  fig_skill_vs_step.png         RMSE and CRPS (sigma units) per variable, with
                                the climatology and obs baselines as reference
  fig_calibration_vs_step.png   spread/skill, rank TV, tails ratio, member/mean
  fig_spectral_vs_step.png      spectral distance, coherence scale, band NRMSE
                                and band spread/skill
  fig_spectra_vs_step.png       member and ensemble-mean PSD coloured by step
plus ``<run>/traj_cond/trajectory_cond.csv`` with every number plotted.

Usage:
    python -m eval.step_trajectory_cond --run <run_dir> [--size full,128]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

BAND_KEYS = ("mesoscale", "submeso")
SCALAR_KEYS = ["rmse_ens_sigma", "crps_sigma", "spread_sigma", "rmse_clim_sigma", "rmse_obs_sigma",
               "spread_skill_ratio", "rank_tv", "rank_tails_ratio", "rmse_member_over_ens",
               "rmse_member_over_ens_target", "ss_rmse_clim", "ss_rmse_obs", "crpss_climens",
               "frac_px_calibrated", "std_ratio", "coh50_km", "logdist_member", "logdist_ensmean",
               "eke_geostrophic_ratio", "eke_geostrophic_ensmean_ratio", "n_clean_days"]
SCALAR_KEYS += [f"{p}_{b}" for b in BAND_KEYS
                for p in ("band_nrmse_ens", "band_nrmse_member", "band_spread_skill",
                          "psd_ratio_member", "psd_ratio_ensmean")]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def collect(run: str, size: str) -> list[tuple[int, str, str]]:
    """``(step, npz_path, figs_dir)`` for the ladder under ``run``, sorted by step.

    Same two-stage filter as ``eval.step_trajectory.collect`` -- modal
    configuration first, then dedupe by step -- with the configuration key
    extended by K, D, split, seed and a hash of the target days.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(run):
        name = f"samples_cond_{size}.npz"
        if name not in filenames:
            continue
        path = os.path.join(dirpath, name)
        figs = os.path.join(dirpath, f"figs_cond_{size}")
        if not os.path.isfile(os.path.join(figs, "summary.csv")):
            print(f"[traj_cond] {path}: no figs_cond_{size}/summary.csv -- run diagnostics_cond first; skipped")
            continue
        try:
            with np.load(path, allow_pickle=False) as d:
                m = json.loads(str(d["meta"]))
                days = np.asarray(d["days"], dtype=np.int64)
            s = m.get("sampler", {})
            key = (m.get("weights", "ema"), s.get("num_steps"), s.get("s_churn"), s.get("s_noise"),
                   s.get("s_tmin"), s.get("s_tmax"), m.get("K"), m.get("D"), m.get("split"),
                   m.get("seed"), hashlib.md5(days.tobytes()).hexdigest()[:8])
            found.append((int(m["step"]), path, figs, key))
        except Exception as exc:                               # noqa: BLE001
            print(f"[traj_cond] skipping unreadable {path}: {exc}")
    if not found:
        return []
    counts: dict = {}
    for _s, _p, _f, key in found:
        counts[key] = counts.get(key, 0) + 1
    ref = max(counts, key=lambda k: (counts[k], max(st for st, _p, _f, kk in found if kk == k)))
    dropped = [p for _s, p, _f, k in found if k != ref]
    if dropped:
        print(f"[traj_cond] reference: weights={ref[0]} steps={ref[1]} s_churn={ref[2]} "
              f"s_noise={ref[3]} K={ref[6]} D={ref[7]} days#{ref[10]}; dropped {len(dropped)} "
              "point(s) with another sampler / member count / target-day set")
    keep: dict[int, tuple[str, str]] = {}
    for step, path, figs, key in sorted(found, key=lambda t: t[1]):
        if key == ref:
            keep.setdefault(step, (path, figs))
    return sorted((s, p, f) for s, (p, f) in keep.items())


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def rows_for(step: int, figs: str, run_name: str, size: str) -> list[dict]:
    rows = []
    with open(os.path.join(figs, "summary.csv")) as fh:
        for r in csv.DictReader(fh):
            row = {"run": run_name, "size": size, "step": step, "var": r["var"]}
            std = _float(r.get("rmse_ens")) / _float(r.get("rmse_ens_sigma")) \
                if _float(r.get("rmse_ens_sigma")) > 0 else np.nan
            r = dict(r)
            r["rmse_clim_sigma"] = _float(r.get("rmse_clim")) / std if std > 0 else np.nan
            r["rmse_obs_sigma"] = _float(r.get("rmse_obs")) / std if std > 0 else np.nan
            for k in SCALAR_KEYS:
                row[k] = _float(r.get(k))
            rows.append(row)
    return rows


def spectra_for(figs: str) -> dict | None:
    p = os.path.join(figs, "spectra.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as d:
        out = {"k": d["k"], "vars": json.loads(str(d["vars"])), "dx_km": float(d["dx_km"]), "psd": {}}
        for v in out["vars"]:
            if f"psd_truth_{v}" in d.files:
                out["psd"][v] = {kind: d[f"psd_{kind}_{v}"] for kind in ("truth", "member", "ensmean", "err_ens")}
                out["psd"][v]["gamma2"] = d[f"gamma2_{v}"]
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _pick(rows, step, var, key):
    for r in rows:
        if r["step"] == step and r["var"] == var:
            return r[key]
    return np.nan


def _series(rows, steps, var, key):
    return np.array([_pick(rows, s, var, key) for s in steps], dtype=np.float64)


def _panel(ax, rows, steps, variables, key, ylabel, title, hline=None, logy=False, ref_key=None):
    xs = [s / 1e6 for s in steps]
    for v in variables:
        y = _series(rows, steps, v, key)
        line, = ax.plot(xs, y, "o-", lw=1.5, ms=4, label=v)
        if ref_key:
            ref = _pick(rows, steps[-1], v, ref_key)
            if np.isfinite(ref):
                ax.axhline(ref, color=line.get_color(), lw=0.9, ls=":")
    if hline is not None:
        ax.axhline(hline, color="k", lw=1, ls=":")
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel="training step [M]", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.tick_params(labelsize=8)


def _save(fig, axes, out, title):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 8), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[traj_cond] wrote {out}")


def fig_skill(rows, steps, variables, out, title):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    _panel(axes[0], rows, steps, variables, "rmse_ens_sigma", "RMSE of ensemble mean [sigma]",
           "RMSE (dotted: climatology baseline at the newest point)", ref_key="rmse_clim_sigma")
    _panel(axes[1], rows, steps, variables, "crps_sigma", "CRPS [sigma]",
           "CRPS (dotted: obs-baseline RMSE where available)", ref_key="rmse_obs_sigma")
    _panel(axes[2], rows, steps, variables, "ss_rmse_clim", "skill score vs climatology",
           "1 - RMSE/RMSE_clim (0 = no better than the mean state)", hline=0.0)
    _save(fig, axes, out, title)


def fig_calibration(rows, steps, variables, out, title):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    _panel(axes[0], rows, steps, variables, "spread_skill_ratio", "spread / skill (Fortin)",
           "1 = calibrated", hline=1.0)
    _panel(axes[1], rows, steps, variables, "rank_tv", "rank-histogram TV", "0 = uniform (old gate 0.2)",
           hline=0.2)
    _panel(axes[2], rows, steps, variables, "rank_tails_ratio", "tails ratio", "> 1 under-dispersed",
           hline=1.0, logy=True)
    _panel(axes[3], rows, steps, variables, "rmse_member_over_ens", "RMSE member / RMSE mean",
           "dotted: sqrt(2K/(K+1))", ref_key="rmse_member_over_ens_target")
    _save(fig, axes, out, title)


def fig_spectral(rows, steps, variables, out, title):
    fig, axes = plt.subplots(1, 3 + len(BAND_KEYS), figsize=(5.0 * (3 + len(BAND_KEYS)), 4.4))
    _panel(axes[0], rows, steps, variables, "logdist_member", "RMS log10(member / truth)",
           "member spectral distance (0 = realistic)")
    _panel(axes[1], rows, steps, variables, "logdist_ensmean", "RMS log10(mean / truth)",
           "ensemble-mean spectral distance")
    _panel(axes[2], rows, steps, variables, "coh50_km", "wavelength [km]",
           "coherence half-scale (smaller = more scales placed right)", logy=True)
    for ax, b in zip(axes[3:], BAND_KEYS):
        xs = [s / 1e6 for s in steps]
        for v in variables:
            line, = ax.plot(xs, _series(rows, steps, v, f"band_nrmse_ens_{b}"), "o-", lw=1.5, ms=4, label=v)
            ax.plot(xs, _series(rows, steps, v, f"band_spread_skill_{b}"), "s--", lw=1.0, ms=3,
                    color=line.get_color())
        ax.axhline(1.0, color="k", lw=1, ls=":")
        ax.set(xlabel="training step [M]", ylabel="NRMSE (solid)  /  spread/skill (dashed)",
               title=f"{b} band")
        ax.grid(alpha=0.3, lw=0.4); ax.tick_params(labelsize=8)
    _save(fig, axes, out, title)


def fig_spectra(specs: list[tuple[int, dict]], out, title):
    steps = [s for s, _ in specs]
    norm = colors.Normalize(vmin=min(steps) / 1e6, vmax=max(steps) / 1e6)
    cmap = cm.viridis
    ref = specs[-1][1]
    variables = [v for v in ref["vars"] if v in ref["psd"]]
    ncol = min(3, max(len(variables), 1))
    nrow = -(-len(variables) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.3 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, v in zip(axes, variables):
        lam = 1.0 / ref["k"]
        ax.plot(lam, ref["psd"][v]["truth"], color="k", lw=2.4, label="truth", zorder=5)
        for s, sp in specs:
            if v not in sp["psd"]:
                continue
            c = cmap(norm(s / 1e6))
            ax.plot(1.0 / sp["k"], sp["psd"][v]["member"], lw=1.3, color=c)
            ax.plot(1.0 / sp["k"], sp["psd"][v]["ensmean"], lw=1.0, ls="--", color=c)
        ax.plot([], [], "k-", lw=1.3, label="members (solid)"); ax.plot([], [], "k--", lw=1.0, label="ens mean (dashed)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.axvline(2 * ref["dx_km"], color="0.5", ls=":", lw=1)
        ax.set_title(v, fontsize=10); ax.set_xlabel("wavelength [km]", fontsize=8); ax.set_ylabel("PSD", fontsize=8)
        ax.grid(alpha=0.25, which="both", lw=0.4); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes[len(variables):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 0.92, 0.94))
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), fraction=0.02, pad=0.015)
    cb.set_label("training step [M]", fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[traj_cond] wrote {out}")


# ---------------------------------------------------------------------------
def run_size(run: str, size: str, outdir: str) -> list[dict]:
    points = collect(run, size)
    if len(points) < 2:
        print(f"[traj_cond] {size}: {len(points)} comparable eval(s) -- need at least 2, skipping")
        return []
    print(f"[traj_cond] {size}: {len(points)} checkpoints -- " + ", ".join(f"{s // 1000}k" for s, _p, _f in points))
    run_name = os.path.basename(os.path.normpath(run))
    rows = []
    specs = []
    for step, _path, figs in points:
        rows += rows_for(step, figs, run_name, size)
        sp = spectra_for(figs)
        if sp is not None:
            specs.append((step, sp))
    steps = sorted({r["step"] for r in rows})
    variables = list(dict.fromkeys(r["var"] for r in rows if r["step"] == steps[-1]))
    os.makedirs(outdir, exist_ok=True)
    head = f"{run_name}  --  {size} geometry, {len(steps)} checkpoints"
    fig_skill(rows, steps, variables, os.path.join(outdir, "fig_skill_vs_step.png"),
              "Paired skill vs training step  --  " + head)
    fig_calibration(rows, steps, variables, os.path.join(outdir, "fig_calibration_vs_step.png"),
                    "Calibration vs training step  --  " + head)
    fig_spectral(rows, steps, variables, os.path.join(outdir, "fig_spectral_vs_step.png"),
                 "Scale-dependent skill vs training step  --  " + head)
    if len(specs) >= 2:
        fig_spectra(specs, os.path.join(outdir, "fig_spectra_vs_step.png"),
                    "Member / ensemble-mean spectra vs training step  --  " + head)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory to scan")
    ap.add_argument("--size", default="full,128", help="comma-separated geometries")
    ap.add_argument("--out", default=None, help="output directory (default <run>/traj_cond)")
    args = ap.parse_args()

    base = args.out or os.path.join(args.run, "traj_cond")
    all_rows = []
    for size in [s.strip() for s in args.size.split(",") if s.strip()]:
        all_rows += run_size(args.run, size, os.path.join(base, f"figs_{size}"))
    if not all_rows:
        raise SystemExit(f"no comparable conditional eval directories under {args.run}")
    os.makedirs(base, exist_ok=True)
    csv_path = os.path.join(base, "trajectory_cond.csv")
    keys = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[traj_cond] wrote {csv_path} ({len(all_rows)} rows)")
    for size in sorted({r["size"] for r in all_rows}):
        sub = [r for r in all_rows if r["size"] == size]
        print(f"\n== {size} ==")
        print(f"{'step':>9} {'var':>6} {'rmse[s]':>8} {'crps[s]':>8} {'spr/skl':>8} {'rankTV':>7} "
              f"{'SSclim':>7} {'coh50':>7}")
        for r in sorted(sub, key=lambda r: (r["step"], r["var"])):
            print(f"{r['step']:>9} {r['var']:>6} {r['rmse_ens_sigma']:>8.3f} {r['crps_sigma']:>8.3f} "
                  f"{r['spread_skill_ratio']:>8.2f} {r['rank_tv']:>7.3f} {r['ss_rmse_clim']:>7.2f} "
                  f"{r['coh50_km']:>7.0f}")


if __name__ == "__main__":
    main()
