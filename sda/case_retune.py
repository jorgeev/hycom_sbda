"""Derive a cheaper or re-weighted case from an existing ``case_<size>.npz``
without touching the store (seconds instead of the 1-2.5 h store load).

Two independent edits, both optional:

* ``--std TERM=FACTOR`` or ``TERM=xFACTOR``: multiply the assumed error
  (``std_norm``) of a term; ``TERM==VALUE`` sets it in normalised units. The
  stored observation values are kept (they were drawn with the original noise),
  so a factor > 1 is a pure error INFLATION: the term is trusted less than the
  noise actually in it, which is the knob against over-fitting.
* ``--gamma TERM=FACTOR``: the same for a term's gamma (per-term guidance
  weight), if the guidance strength rather than the error model is the lever.
* ``--days a:b`` or ``--days 0,3,7``: keep a subset of the case's steps, e.g. a
  12 h sub-window of the 48 h SWOT case. The per-step arrays are sliced
  consistently; ``pos`` is kept as is (full frame: empty).

The result is written as a new case file; the terms json records the edits in
``retune`` so the report shows what changed. Usage:

  python -m sda.case_retune --case <dir>/case_full.npz --out <dir2>/case_full.npz \
      --days 0:12 --std sss_l4=x3
  # then: SIZES=full ... OUTDIR=<dir2> sbatch assimilate.slurm <ckpt> 2,3 <obs yaml>
  # (assimilate.slurm skips the build when case_full.npz exists)

A control ensemble is independent of the terms, so the same seeds give the same
control for every re-weighting: link <arm>/samples_sda_full_free.npz from a
finished arm into the new arm directory (same --days subset only) and the
driver reuses it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def parse_days(spec: str, D: int) -> np.ndarray:
    if ":" in spec:
        a, b = spec.split(":")
        return np.arange(int(a or 0), int(b or D))
    return np.array([int(x) for x in spec.split(",")], dtype=int)


def parse_edits(items: list[str]) -> dict[str, tuple[str, float]]:
    out = {}
    for it in items or []:
        name, val = it.split("=", 1)
        if val.startswith("="):
            out[name] = ("set", float(val[1:]))
        else:
            out[name] = ("mul", float(val.lstrip("x")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", default=None, help="a:b slice or comma list of step indices to keep")
    ap.add_argument("--std", nargs="*", default=[], help="TERM=x3 (multiply) or TERM==0.9 (set, normalised)")
    ap.add_argument("--gamma", nargs="*", default=[], help="TERM=x0.5 (multiply) or TERM==0.05 (set)")
    args = ap.parse_args(argv)

    d = np.load(args.case, allow_pickle=False)
    arrays = {k: d[k] for k in d.files}
    terms = json.loads(str(arrays["terms"]))
    D = int(arrays["truth_norm"].shape[0])
    names = [t["name"] for t in terms]
    edits = {"std": parse_edits(args.std), "gamma": parse_edits(args.gamma), "days": args.days}
    for field in ("std", "gamma"):
        for name, (mode, val) in edits[field].items():
            if name not in names:
                raise SystemExit(f"term {name!r} not in case terms {names}")
            t = terms[names.index(name)]
            key = "std_norm" if field == "std" else "gamma"
            old = float(t[key])
            t[key] = val if mode == "set" else old * val
            print(f"[retune] {name}.{key}: {old:.4g} -> {t[key]:.4g}")
    if args.days:
        keep = parse_days(args.days, D)
        if keep.min() < 0 or keep.max() >= D:
            raise SystemExit(f"--days out of range 0..{D - 1}")
        for k, v in list(arrays.items()):
            if v.ndim >= 1 and v.shape[0] == D and k not in ("std", "is_log", "mask_full", "lat_full"):
                arrays[k] = v[keep]
        print(f"[retune] kept {len(keep)} of {D} steps: {keep.tolist()}")
    meta = json.loads(str(arrays["meta"])) if "meta" in arrays else {}
    if args.days:
        meta["D"] = int(len(keep))
        si = meta.get("store_index")
        if isinstance(si, list) and len(si) == D:
            meta["store_index"] = [si[int(i)] for i in keep]
    meta["retune"] = {"from": os.path.abspath(args.case), **{k: v for k, v in edits.items() if v}}
    # the per-term coverage list in meta refers to the whole case; recompute
    if "coverage" in meta and "obs_mask" in arrays:
        meta["coverage"] = [float(arrays["obs_mask"][:, ti].mean()) for ti in range(arrays["obs_mask"].shape[1])]
    arrays["terms"] = np.array(json.dumps(terms))
    arrays["meta"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"[retune] wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
