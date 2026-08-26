"""One table over every sweep configuration, per geometry.

The sweep answers a single question -- which cheap knob recovers dispersion --
so the numbers that matter are the ones the diagnosis was made from: normalised
per-channel std (gen and real, in units of the training sigma stored in the
npz) and the total-EKE ratio. ``diagnostics.py`` writes std in physical units
to ``summary.csv`` and the EKE scalars only into ``report.md`` prose, so this
script pulls from both, plus each npz's ``meta`` for the sampler settings --
the directory name never has to be trusted.

Usage:
    python -m eval.sweep_summary --run <run_dir>

reads ``<run>/eval_step1140000`` (the baseline) and every ``<run>/sweep/*/``
that has figures, and writes ``<run>/sweep/summary.md`` and ``summary.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np

EKE_ROW = re.compile(r"\|\s*(geostrophic|ageostrophic|total) EKE \[m2/s2\]\s*\|"
                     r"\s*([-\d.eE]+)\s*\|\s*([-\d.eE]+)\s*\|\s*([-\d.eE]+)\s*\|")


def read_config(d: str, size: str):
    """Everything the table needs from one eval dir at one geometry, or None."""
    npz = os.path.join(d, f"samples_{size}.npz")
    csvf = os.path.join(d, f"figs_{size}", "summary.csv")
    rep = os.path.join(d, f"figs_{size}", "report.md")
    if not (os.path.isfile(npz) and os.path.isfile(csvf)):
        return None
    with np.load(npz, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        std = np.asarray(f["std"], dtype=np.float64)
        variables = json.loads(str(f["vars"]))
    std_of = dict(zip(variables, std))
    sampler = meta.get("sampler", {})
    row = dict(
        config=os.path.basename(d.rstrip("/")),
        step=meta.get("step"),
        weights=meta.get("weights", "ema"),
        steps=sampler.get("num_steps"),
        s_churn=sampler.get("s_churn"),
        s_noise=sampler.get("s_noise"),
    )
    with open(csvf) as fh:
        for r in csv.DictReader(fh):
            row[f"nstd_gen_{r['var']}"] = float(r["std_gen"]) / std_of[r["var"]]
            row[f"nstd_real_{r['var']}"] = float(r["std_real"]) / std_of[r["var"]]
            row[f"std_ratio_{r['var']}"] = float(r["std_ratio"])
    if os.path.isfile(rep):
        for kind, real, gen, ratio in EKE_ROW.findall(open(rep).read()):
            row[f"eke_{kind}_ratio"] = float(ratio)
    return row


def fmt(v, nd=2):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--baseline", default="eval_step1140000",
                    help="baseline eval dir (relative to --run)")
    args = ap.parse_args()

    sweep = os.path.join(args.run, "sweep")
    dirs = [os.path.join(args.run, args.baseline)]
    if os.path.isdir(sweep):
        dirs += sorted(os.path.join(sweep, d) for d in os.listdir(sweep)
                       if os.path.isdir(os.path.join(sweep, d)))

    tables, all_rows = {}, []
    for size in ("128", "full"):
        rows = [r for d in dirs if (r := read_config(d, size))]
        for r in rows:
            r["size"] = size
        tables[size] = rows
        all_rows += rows
    if not all_rows:
        raise SystemExit(f"no completed eval dirs under {args.run}")

    variables = sorted({k[len("nstd_gen_"):] for r in all_rows
                        for k in r if k.startswith("nstd_gen_")})
    os.makedirs(sweep, exist_ok=True)

    keys = ["config", "size", "step", "weights", "steps", "s_churn", "s_noise"]
    keys += [f"{p}_{v}" for v in variables
             for p in ("nstd_gen", "nstd_real", "std_ratio")]
    keys += [f"eke_{k}_ratio" for k in ("geostrophic", "ageostrophic", "total")]
    with open(os.path.join(sweep, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    lines = ["# Dispersion sweep", "",
             "Normalised std is in training-sigma units; the target is the "
             "`real` row (~1.0-1.17 for the ocean-state channels). "
             "`EKE` is the total-EKE ratio gen/real (target 1).", ""]
    for size, rows in tables.items():
        if not rows:
            continue
        lines += [f"## Geometry: {size}", ""]
        hdr = ["config", "step", "wts", "steps", "churn", "s_noise"] + \
              variables + ["EKE"]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        ref = rows[0]
        lines.append("| " + " | ".join(
            ["real", "", "", "", "", ""] +
            [fmt(ref.get(f"nstd_real_{v}")) for v in variables] + ["1.00"]) + " |")
        for r in rows:
            lines.append("| " + " | ".join(
                [str(r["config"]), fmt(r["step"]), r["weights"],
                 fmt(r["steps"]), fmt(r["s_churn"]), fmt(r["s_noise"], 3)] +
                [fmt(r.get(f"nstd_gen_{v}")) for v in variables] +
                [fmt(r.get("eke_total_ratio"))]) + " |")
        lines.append("")
    out = os.path.join(sweep, "summary.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[sweep_summary] wrote {out} and summary.csv "
          f"({len(all_rows)} rows, {len(variables)} variables)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
