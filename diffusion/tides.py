"""Tidal phase labels: where each frame sits in the cycle of each constituent.

The video prior is trained on hourly frames that contain the tide, and its
calendar channels carry SOLAR time only, so nothing tells it the phase of the
lunar constituents (M2, 12.42 h, is most of the signal). :func:`tidal_labels`
turns a frame's timestamp into that information, as scalar labels fed through
the network's label embedding (the route GenLLC4320 uses for its flux labels):

    labels[n] = [cos p_1, sin p_1, ..., cos p_K, sin p_K] / sqrt(K)
    p_k       = speed_k * (t_n - epoch) / 3600        (degrees; t, epoch in s)

Conventions, fixed here so every caller and every checkpoint agrees:

  * ORDER is ``[cos, sin]`` per constituent, constituents in the order given.
    With the default eight that is the term order of the coefficient files
    written by ``extract_hycom_tides.py`` (``K1_cos, K1_sin, O1_cos, ...``).
  * EPOCH defaults to that fit's phase origin, its first snapshot
    (2017-01-01T01:00:00Z). The fitted tide is therefore
    ``sum_j coefficient_j(x) * labels[n, j] * sqrt(K)``.
  * SCALE is ``1/sqrt(K)``: the label vector has unit norm on every frame. The
    label layer (``VideoUNet.map_label``) is initialised for a unit-norm label
    -- a one-hot class -- so its contribution to the embedding starts with unit
    variance per channel. Raw cos/sin of eight constituents would start it at
    eight. The scale is NOT a config key: it is not recorded in a checkpoint,
    so changing it would silently mis-condition every trained model.

These are pure harmonics at fixed speeds. There are no nodal corrections and no
astronomical arguments, which is what the fit used; they are exact relative to
that fit and drift from the real ocean's tide over years.
"""
from __future__ import annotations

import numpy as np

# Angular speeds in degrees per solar hour, as in extract_hycom_tides.py.
SPEED_DEG_PER_HOUR = {
    "K1": 15.0410686, "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609,
    "M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K2": 30.0821373,
}
DEFAULT_EPOCH = 1483232400      # 2017-01-01T01:00:00Z, unix seconds


def check_constituents(names) -> list[str]:
    """``names`` as a list of known, distinct constituents, or ``ValueError``."""
    known = ", ".join(SPEED_DEG_PER_HOUR)
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise ValueError(
            f"tidal_labels must be a list of constituent names such as [M2, S2], "
            f"got {names!r}. Known constituents: {known}")
    names = list(names)
    unknown = [n for n in names if n not in SPEED_DEG_PER_HOUR]
    if unknown:
        raise ValueError(f"unknown tidal constituents {unknown}. Known: {known}")
    if len(set(names)) != len(names):
        raise ValueError(f"tidal_labels lists a constituent twice: {names}")
    return names


def tidal_labels(t_unix, names, epoch: float = DEFAULT_EPOCH) -> np.ndarray:
    """``(N, 2K)`` float32 unit-norm phase labels for timestamps ``t_unix`` (s).

    ``(N, 0)`` when ``names`` is empty. The angle is formed in float64 and
    reduced modulo 360 before the trigonometry; only cos/sin are cast down.
    """
    names = check_constituents(names)
    t = np.atleast_1d(np.asarray(t_unix, dtype=np.float64))
    if not names:
        return np.zeros((t.size, 0), dtype=np.float32)
    hours = (t - float(epoch)) / 3600.0
    speed = np.array([SPEED_DEG_PER_HOUR[n] for n in names], dtype=np.float64)
    phase = np.deg2rad(np.mod(hours[:, None] * speed[None, :], 360.0))      # (N, K)
    out = np.stack([np.cos(phase), np.sin(phase)], axis=2).reshape(t.size, -1)
    return (out / np.sqrt(len(names))).astype(np.float32)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import os
    from decimal import Decimal, getcontext

    names = list(SPEED_DEG_PER_HOUR)
    K = len(names)
    hour = 3600.0
    t = DEFAULT_EPOCH + hour * np.arange(4367, dtype=np.float64)   # the gulfstream record
    lab = tidal_labels(t, names)
    assert lab.shape == (t.size, 2 * K) and lab.dtype == np.float32
    assert np.allclose((lab.astype(np.float64) ** 2).sum(axis=1), 1.0, atol=1e-6)
    # at the epoch every phase is zero: cos columns 1/sqrt(K), sin columns 0
    assert np.allclose(lab[0, 0::2], 1 / np.sqrt(K)) and np.allclose(lab[0, 1::2], 0.0)
    print(f"[ok] {K} constituents -> {2 * K} labels per frame, unit norm, "
          "[cos, sin] per constituent")

    # one constituent at a time: column pair, period, and subset = columns
    m2 = tidal_labels(t, ["M2"])
    assert np.allclose(m2, lab[:, 8:10] * np.sqrt(K), atol=1e-6)
    period = 360.0 / SPEED_DEG_PER_HOUR["M2"]
    assert abs(period - 12.4206) < 1e-4
    a = tidal_labels([DEFAULT_EPOCH + 5 * hour], ["M2"])
    b = tidal_labels([DEFAULT_EPOCH + (5 + 3 * period) * hour], ["M2"])
    assert np.allclose(a, b, atol=1e-6)
    sub = tidal_labels(t, ["S2", "K1"])
    assert np.allclose(sub[:, 0:2], lab[:, 10:12] * np.sqrt(K / 2), atol=1e-6)
    assert np.allclose(sub[:, 2:4], lab[:, 0:2] * np.sqrt(K / 2), atol=1e-6)
    assert tidal_labels(t, []).shape == (t.size, 0)
    print(f"[ok] M2 repeats every {period:.4f} h; a subset selects its own columns "
          "in the order listed")

    # S2 turns exactly twice per solar day, so with this epoch (01:00 UTC) it is
    # the second solar-time harmonic the calendar channels already carry.
    sod = np.mod(t, 86400.0)
    ref = 4.0 * np.pi * (sod - 3600.0) / 86400.0
    s2 = tidal_labels(t, ["S2"])
    assert np.allclose(s2[:, 0], np.cos(ref), atol=1e-6)
    assert np.allclose(s2[:, 1], np.sin(ref), atol=1e-6)
    print("[ok] the S2 label equals cos/sin of 4 pi (second_of_day - 3600) / 86400")

    # precision far from the epoch, against exact decimal arithmetic; forming the
    # angle in float32 instead is wrong by a visible amount.
    getcontext().prec = 40
    far = DEFAULT_EPOCH + hour * np.array([1.0, 4366.0, 24.0 * 365.25 * 30, 24.0 * 365.25 * 60])
    got = tidal_labels(far, ["M2"]).astype(np.float64) * 1.0
    worst, worst32 = 0.0, 0.0
    for i, tf in enumerate(far):
        h = (Decimal(repr(float(tf))) - Decimal(DEFAULT_EPOCH)) / Decimal(3600)
        deg = float((h * Decimal(repr(SPEED_DEG_PER_HOUR["M2"]))) % Decimal(360))
        worst = max(worst, abs(got[i, 0] - np.cos(np.deg2rad(deg))),
                    abs(got[i, 1] - np.sin(np.deg2rad(deg))))
        naive = np.float32((tf - DEFAULT_EPOCH) / 3600.0) * np.float32(
            np.deg2rad(SPEED_DEG_PER_HOUR["M2"]))
        worst32 = max(worst32, abs(float(np.cos(naive)) - np.cos(np.deg2rad(deg))))
    assert worst < 1e-6, worst
    assert worst32 > 1e-3, worst32
    print(f"[ok] phase is exact to {worst:.1e} sixty years from the epoch "
          f"(a float32 angle would be off by {worst32:.2f})")

    for bad, why in (("M2", "a bare string"), (["M2", "XX"], "an unknown name"),
                     (["M2", "S2", "M2"], "a duplicate"), (8, "a number")):
        try:
            tidal_labels(t[:2], bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} was accepted: {bad!r}")
    print("[ok] a bare string, an unknown name, a duplicate and a number are refused")

    # Against the harmonic fit itself, where its files are on this machine.
    fit = os.environ.get("TIDE_FIT_DIR", "/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/tides/ssh")
    coef_path = os.path.join(fit, "coefficients_ssh_box56_0017.nc")
    day, hh = 150, 15
    snap_path = os.path.join(fit, f"tide_ssh_box56_0017_{day:03d}_{hh:02d}.nc")
    try:
        import xarray as xr
        have = os.path.exists(coef_path) and os.path.exists(snap_path)
    except ImportError:
        have = False
    if not have:
        print(f"[skip] harmonic-fit files not found under {fit}; reconstruction not checked")
    else:
        with xr.open_dataset(coef_path) as ds:
            terms = [str(x) for x in ds["term"].values]
            coef = ds["coefficient"].values.astype(np.float64)              # (16, y, x)
        assert terms == [f"{c}_{trig}" for c in names for trig in ("cos", "sin")], terms
        with xr.open_dataset(snap_path, decode_times=False) as ds:
            snap = ds[[n for n, v in ds.data_vars.items() if v.ndim == 3][0]].values[0]
        jan1 = 1483228800                                                    # 2017-01-01T00:00Z
        ts = jan1 + ((day - 1) * 24 + hh) * 3600
        w = tidal_labels([ts], names)[0].astype(np.float64) * np.sqrt(K)
        rec = np.tensordot(w, coef, axes=(0, 0))
        err = float(np.nanmax(np.abs(rec - snap)))
        assert err < 1e-5, err
        print(f"[ok] coefficient maps x labels x sqrt(K) reproduce the fitted tide of "
              f"day {day} {hh:02d}:00 to {err:.1e} m (amplitude {np.nanmax(np.abs(snap)):.2f} m)")
    print("tides selftest passed")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
