"""One table over every sampler-sweep arm of a CONDITIONAL model, per geometry.

The conditional counterpart of ``eval.sweep_summary``. A sampler sweep asks
which knob (steps, churn, s_noise, EMA vs raw weights, an earlier checkpoint)
buys calibration without costing skill, so the columns are the paired numbers:
RMSE and CRPS of the ensemble mean in sigma units, Fortin spread/skill,
rank-histogram TV and tails ratio, member std ratio, submesoscale member PSD
ratio, skill scores vs the two baselines, and the EKE ratios.

Everything comes from ``figs_cond_<size>/summary.csv`` plus each npz's ``meta``
(sampler settings, weights, step, K, D). Nothing is scraped from ``report.md``:
``diagnostics_cond`` repeats the domain-level scalars on every csv row precisely
so no regex over prose is ever needed -- the regex in ``eval.sweep_summary``
silently drops its EKE column whenever the prose changes.

Usage:
    python -m eval.sweep_summary_cond --run <run_dir> [--baseline eval_cond_step0200000]

reads ``<run>/<baseline>`` (default: the newest ``<run>/eval_cond_step*``) and
every ``<run>/sweep_cond/*/`` that has figures, and writes
``<run>/sweep_cond/summary.md`` and ``summary.csv``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

PER_VAR = ["rmse_ens_sigma", "crps_sigma", "spread_skill_ratio", "rank_tv", "rank_tails_ratio",
           "std_ratio", "psd_ratio_member_submeso", "ss_rmse_obs", "ss_rmse_clim"]
DOMAIN = ["eke_geostrophic_ratio", "eke_total_ratio", "eke_geostrophic_ensmean_ratio"]
TARGET = {"rmse_ens_sigma": "min", "crps_sigma": "min", "spread_skill_ratio": 1.0, "rank_tv": 0.0,
          "rank_tails_ratio": 1.0, "std_ratio": 1.0, "psd_ratio_member_submeso": 1.0,
          "ss_rmse_obs": "> 0", "ss_rmse_clim": "> 0", "eke_geostrophic_ratio": 1.0,
          "eke_total_ratio": 1.0, "eke_geostrophic_ensmean_ratio": "<= 1"}


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_arm(d: str, size: str):
    """Everything the table needs from one eval dir at one geometry, or None."""
    npz = os.path.join(d, f"samples_cond_{size}.npz")
    csvf = os.path.join(d, f"figs_cond_{size}", "summary.csv")
    if not (os.path.isfile(npz) and os.path.isfile(csvf)):
        return None
    with np.load(npz, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
    s = meta.get("sampler", {})
    row = dict(config=os.path.basename(d.rstrip("/")), size=size, step=meta.get("step"),
               weights=meta.get("weights", "ema"), steps=s.get("num_steps"), s_churn=s.get("s_churn"),
               s_noise=s.get("s_noise"), s_tmin=s.get("s_tmin"), s_tmax=s.get("s_tmax"),
               K=meta.get("K"), D=meta.get("D"))
    variables = []
    with open(csvf) as fh:
        for r in csv.DictReader(fh):
            v = r["var"]
            variables.append(v)
            for k in PER_VAR:
                row[f"{k}_{v}"] = _float(r.get(k))
            for k in DOMAIN:
                row[k] = _float(r.get(k))
    row["_vars"] = variables
    return row


def fmt(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, float):
        return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def _delta(v, ref):
    if not (isinstance(v, float) and isinstance(ref, float) and np.isfinite(v) and np.isfinite(ref)) or ref == 0:
        return ""
    return f" ({100 * (v - ref) / abs(ref):+.0f}%)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--baseline", default=None,
                    help="baseline eval dir relative to --run (default: newest eval_cond_step*)")
    args = ap.parse_args()

    if args.baseline is None:
        cands = sorted(glob.glob(os.path.join(args.run, "eval_cond_step*")))
        if not cands:
            raise SystemExit(f"no eval_cond_step* under {args.run}; pass --baseline")
        baseline = cands[-1]
    else:
        baseline = os.path.join(args.run, args.baseline)
    sweep = os.path.join(args.run, "sweep_cond")
    dirs = [baseline]
    if os.path.isdir(sweep):
        dirs += sorted(os.path.join(sweep, d) for d in os.listdir(sweep)
                       if os.path.isdir(os.path.join(sweep, d)))
    sizes = sorted({os.path.basename(p)[len("samples_cond_"):-4]
                    for d in dirs for p in glob.glob(os.path.join(d, "samples_cond_*.npz"))},
                   key=lambda s: (s != "full", s))

    tables, all_rows = {}, []
    for size in sizes:
        rows = [r for d in dirs if (r := read_arm(d, size))]
        tables[size] = rows
        all_rows += rows
    if not all_rows:
        raise SystemExit(f"no completed conditional eval dirs under {args.run}")
    variables = list(dict.fromkeys(v for r in all_rows for v in r["_vars"]))
    os.makedirs(sweep, exist_ok=True)

    keys = ["config", "size", "step", "weights", "steps", "s_churn", "s_noise", "s_tmin", "s_tmax", "K", "D"]
    keys += [f"{k}_{v}" for v in variables for k in PER_VAR] + DOMAIN
    with open(os.path.join(sweep, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and not np.isfinite(r.get(k)) else r.get(k))
                        for k in keys})

    lines = ["# Conditional sampler sweep", "",
             f"Baseline arm: `{os.path.basename(baseline)}` (first row of each table; deltas are "
             "relative to it). Targets: spread/skill 1, rank TV 0, tails 1, std ratio 1, "
             "submeso PSD ratio 1, skill scores > 0, EKE ratio (members/truth) 1.", ""]
    show = [("rmse_ens_sigma", "RMSE[s]"), ("crps_sigma", "CRPS[s]"), ("spread_skill_ratio", "spr/skl"),
            ("rank_tv", "rankTV"), ("rank_tails_ratio", "tails"), ("std_ratio", "std"),
            ("psd_ratio_member_submeso", "PSDsub"), ("ss_rmse_clim", "SSclim")]
    for size, rows in tables.items():
        if not rows:
            continue
        ref = rows[0]
        for v in variables:
            lines += [f"## Geometry {size} -- {v}", ""]
            hdr = ["config", "step", "wts", "steps", "churn", "s_noise", "K"] + [lab for _k, lab in show] + ["EKE geo"]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "---|" * len(hdr))
            lines.append("| " + " | ".join(["target", "", "", "", "", "", ""] +
                                           [str(TARGET[k]) if not isinstance(TARGET[k], float) else fmt(TARGET[k])
                                            for k, _l in show] + ["1.00"]) + " |")
            for r in rows:
                cells = [str(r["config"]), fmt(r["step"]), r["weights"], fmt(r["steps"]), fmt(r["s_churn"]),
                         fmt(r["s_noise"], 3), fmt(r["K"])]
                for k, _l in show:
                    val = r.get(f"{k}_{v}")
                    cells.append(fmt(val) + (_delta(val, ref.get(f"{k}_{v}")) if r is not ref and
                                             k in ("rmse_ens_sigma", "crps_sigma") else ""))
                cells.append(fmt(r.get("eke_geostrophic_ratio")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    out = os.path.join(sweep, "summary.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[sweep_summary_cond] wrote {out} and summary.csv ({len(all_rows)} rows, {len(variables)} variables)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
