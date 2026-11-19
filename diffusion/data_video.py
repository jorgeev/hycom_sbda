"""Multi-frame windows of (target, cond, labels, calendar inputs) from a store.

:class:`ZarrVideoDataset` is :class:`diffusion.data.ZarrWindowDataset` with the
sample changed from one step to ``n_frames`` consecutive steps at ONE crop
position. Everything about loading is inherited unchanged: the whole-record
preload, normalisation and climatology (and their caches), ``share_from``, the
crop sampler and ``worker_init_fn``. Host RAM therefore does not grow with the
number of frames -- frames are slices of the same preloaded arrays.

One sample (before collation):

    target (T, Ct, H, W)   the generated channels, ``cfg.target``
    cond   (T, Cc, H, W)   per-frame input channels: each ``cfg.cond_vars`` entry
                           at that frame's time, then the ocean mask if
                           ``use_ocean_mask`` (zero-width when there are none)
    labels (T, L)          ocean-mean of each ``cfg.label_vars`` field per frame,
                           then cos and sin of each ``cfg.tidal_labels``
                           constituent's phase at the frame's time
                           (:mod:`diffusion.tides`; the same for every crop)
    sod    (T,)            UTC second of day of each frame
    doy    (T,)            fraction of the year in [0, 1) of each frame
    lon    (H, W)          degrees east (zeros unless ``use_calendar``)
    mask   (1, H, W)       1 = ocean
    day    ()              base-cadence index of the first frame

Frames are ``d, d + s, ..., d + (T-1) s`` in base-cadence steps
(``s = cfg.frame_stride``); a window is only offered when all of them lie in
the split and are consecutive in real time.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .config import Config
from .data import ZarrWindowDataset, evaluate_climatology
from .dataset_spec import DatasetSpec, resolve_spec
from .tides import check_constituents, tidal_labels


def valid_window_starts(valid_steps: np.ndarray, t_unix: np.ndarray, n_frames: int,
                        stride: int) -> np.ndarray:
    """Steps ``d`` whose whole window is usable.

    ``valid_steps`` are the single steps the split offers (sorted). A start is
    kept when every frame ``d + i*stride`` is itself a valid step and sits
    exactly ``i*stride`` nominal steps after ``d`` in real time (half a step of
    slack), so a window never crosses a split boundary or a gap in the record.
    """
    valid_steps = np.asarray(valid_steps, dtype=np.int64)
    if n_frames <= 1 or valid_steps.size == 0:
        return valid_steps
    step = float(np.median(np.diff(t_unix))) if t_unix.size > 1 else 0.0
    ok = np.ones(valid_steps.size, dtype=bool)
    for i in range(1, n_frames):
        f = valid_steps + i * stride
        inside = np.isin(f, valid_steps)
        ok &= inside
        fi = np.where(inside, f, valid_steps)          # safe index where not inside
        ok &= np.abs((t_unix[fi] - t_unix[valid_steps]) - i * stride * step) <= 0.5 * step
    return valid_steps[ok]


class ZarrVideoDataset(ZarrWindowDataset):
    """``n_frames``-step windows; see the module docstring for the sample layout."""

    def __init__(self, cfg: Config, split: str = "train",
                 share_from: "ZarrVideoDataset | None" = None,
                 spec: DatasetSpec | None = None):
        if cfg.k_days != 1:
            raise ValueError(
                "video configs need k_days: 1 -- the window is n_frames/"
                "frame_stride, and cond_vars are taken at each frame's own time")
        if cfg.use_doy:
            raise ValueError(
                "video configs need use_doy: false -- use use_calendar for "
                "time-of-day and day-of-year inputs")
        both = sorted(set(cfg.target) & set(cfg.label_vars))
        if both:
            raise ValueError(
                f"{both} listed in both target and label_vars: a variable is "
                "either generated (target) or a scalar label (label_vars)")
        if cfg.n_frames < 1 or cfg.frame_stride < 1:
            raise ValueError("n_frames and frame_stride must be >= 1")
        check_constituents(cfg.tidal_labels)      # before the preload, not after
        down = 2 ** (len(cfg.channel_mult) - 1)
        if cfg.patch % down:
            raise ValueError(
                f"patch={cfg.patch} is not divisible by {down}, the network's "
                f"total downsample factor ({len(cfg.channel_mult)} levels)")

        spec = spec if spec is not None else resolve_spec(cfg)
        # The base class preloads ``cond_vars + target`` and checks that the FULL
        # domain divides by the UNet's downsample factor. Hand it a view of the
        # config that (a) also lists the label variables, so they are preloaded
        # and normalised like any other channel, and (b) has a one-level
        # ``channel_mult``, which turns the full-domain check off: a 5-level
        # network needs /16 and the 576x936 gulfstream grid is not (936 = 8*117),
        # but training only needs the PATCH to divide, checked above.
        base_cfg = dataclasses.replace(
            cfg, cond_vars=list(cfg.cond_vars) + list(cfg.label_vars),
            channel_mult=[1])
        super().__init__(base_cfg, split=split, share_from=share_from, spec=spec)
        self.cfg = cfg
        self.T = int(cfg.n_frames)
        self.stride = int(cfg.frame_stride)

        self.t_unix = spec.family.times(spec.base)[: self.t_end].astype(np.float64)
        self.year_frac = (self.doy_ang / (2.0 * np.pi)).astype(np.float32)
        self.sod = np.mod(self.t_unix, 86400.0).astype(np.float32)
        # (t_end, 2K) tidal phase labels, zero-width when there are none. A
        # function of the timestamp alone, so cheap enough to rebuild per dataset.
        self.tide_phase = tidal_labels(self.t_unix, cfg.tidal_labels, cfg.tidal_epoch)

        single = self.valid_days                      # k=1: every usable step
        self.valid_days = valid_window_starts(single, self.t_unix, self.T, self.stride)
        if self.valid_days.size == 0:
            raise ValueError(
                f"No valid {self.T}-frame windows (stride {self.stride}) for "
                f"split={split!r} (t_end={self.t_end}).")

        if cfg.use_calendar:
            if share_from is not None and getattr(share_from, "lon", None) is not None:
                self.lon = share_from.lon
            else:
                lon = np.asarray(spec.family.coord(cfg.lon_path), dtype=np.float32)
                if lon.shape != (self.NY, self.NX):
                    raise ValueError(
                        f"{cfg.lon_path!r} has shape {lon.shape}, expected the "
                        f"{self.NY}x{self.NX} grid (a 2-D per-pixel longitude)")
                self.lon = lon
        else:
            self.lon = None

    # -- one window --------------------------------------------------------
    def frames_of(self, start: int) -> np.ndarray:
        return int(start) + self.stride * np.arange(self.T, dtype=np.int64)

    def _stack(self, var: str, frames: np.ndarray, ysl: slice, xsl: slice) -> np.ndarray:
        """``(T, h, w)`` of ``var`` at the given base-cadence steps."""
        return np.stack([self.data[var][self._idx(var, int(f)), ysl, xsl]
                         for f in frames], axis=0)

    def window(self, start: int, ysl: slice, xsl: slice) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        frames = self.frames_of(start)
        mask2d = (self.ocean[ysl, xsl] > 0.5).astype(np.float32)
        h, w = mask2d.shape

        target = np.stack([self._stack(v, frames, ysl, xsl) for v in cfg.target], axis=1)

        chans = [self._stack(v, frames, ysl, xsl) for v in cfg.cond_vars]
        if cfg.use_ocean_mask:
            chans.append(np.broadcast_to(mask2d, (self.T, h, w)))
        cond = (np.stack(chans, axis=1).astype(np.float32) if chans
                else np.zeros((self.T, 0, h, w), dtype=np.float32))

        if cfg.label_vars:
            n_ocean = max(float(mask2d.sum()), 1.0)
            labels = np.stack(
                [(self._stack(v, frames, ysl, xsl) * mask2d).sum(axis=(1, 2)) / n_ocean
                 for v in cfg.label_vars], axis=1).astype(np.float32)
        else:
            labels = np.zeros((self.T, 0), dtype=np.float32)
        if self.tide_phase.shape[1]:
            labels = np.concatenate([labels, self.tide_phase[frames]], axis=1)

        lon = (self.lon[ysl, xsl] if self.lon is not None
               else np.zeros((h, w), dtype=np.float32))
        return {
            "target": torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
            "cond": torch.from_numpy(cond),
            "labels": torch.from_numpy(labels),
            "sod": torch.from_numpy(self.sod[frames]),
            "doy": torch.from_numpy(self.year_frac[frames]),
            "lon": torch.from_numpy(np.ascontiguousarray(lon, dtype=np.float32)),
            "mask": torch.from_numpy(mask2d[None]),
            "day": torch.tensor(int(start), dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self._rng.choice(self.valid_days))
        y0, x0 = self.sample_position()
        p = self.cfg.patch
        return self.window(start, slice(y0, y0 + p), slice(x0, x0 + p))

    # -- sampling / evaluation helpers ------------------------------------
    def full_slices(self) -> tuple[slice, slice]:
        """The largest top-left region the network can run on in one pass.

        The whole domain when it divides by the total downsample factor,
        otherwise cropped down to the nearest multiple (576x936 -> 576x928 for
        a 5-level network).
        """
        down = 2 ** (len(self.cfg.channel_mult) - 1)
        return (slice(0, self.NY - self.NY % down), slice(0, self.NX - self.NX % down))

    def full_frames(self, day_list):      # the single-frame API does not apply
        raise NotImplementedError("use ZarrVideoDataset.window(start, ysl, xsl)")

    def denormalize_window(self, var: str, arr: np.ndarray, start: int,
                           ysl: slice, xsl: slice) -> np.ndarray:
        """Physical units for ``arr (..., T, h, w)`` of ``var`` on a window.

        Under ``norm_mode: anomaly`` each frame gets its own climatology field
        (the phase advances frame to frame), cropped to the window.
        """
        m, s, is_log = self.stats[var]
        if self.cfg.norm_mode == "anomaly":
            frames = self.frames_of(start)
            m = evaluate_climatology(m, self.doy_ang[frames])[:, ysl, xsl]   # (T, h, w)
        out = arr * s + m
        return np.power(10.0, out) if is_log else out

    def denorm_params(self, var, day=None):   # scalar-day API of the base class
        raise NotImplementedError("use ZarrVideoDataset.denormalize_window")


def batch_ctx(batch: dict, device) -> dict:
    """The ``ctx`` dict :class:`edm_video.VideoPrecond` expects, on ``device``."""
    return {k: batch[k].to(device, non_blocking=True)
            for k in ("cond", "labels", "sod", "doy", "lon")}


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _selftest_windows() -> None:
    hour = 3600.0
    # 0..9 valid, 10..19 belongs to another split, plus a 5-hour gap after step 5
    t = np.arange(20, dtype=np.float64) * hour
    t[6:] += 5 * hour
    valid = np.arange(10)
    s = valid_window_starts(valid, t, n_frames=3, stride=1)
    # windows wholly before the gap (0..3) or after it (6, 7); none straddle
    # the gap, and none run past the split end at step 9
    assert s.tolist() == [0, 1, 2, 3, 6, 7], s
    s2 = valid_window_starts(valid, t, n_frames=2, stride=3)
    assert s2.tolist() == [0, 1, 2, 6], s2
    assert valid_window_starts(valid, t, 1, 1).tolist() == valid.tolist()
    print("[ok] window starts respect the split end, the stride and gaps in time")


def _selftest_store(config: str, max_days: int | None) -> None:
    from .config import load_config
    cfg = load_config(config, max_days=max_days)
    ds = ZarrVideoDataset(cfg, split="train")
    T, p = cfg.n_frames, cfg.patch
    ds.set_epoch_rng(0)
    it = ds[0]
    ct = len(cfg.target)
    cc = len(cfg.cond_vars) + (1 if cfg.use_ocean_mask else 0)
    assert it["target"].shape == (T, ct, p, p), it["target"].shape
    assert it["cond"].shape == (T, cc, p, p)
    assert it["labels"].shape == (T, len(cfg.label_vars) + 2 * len(cfg.tidal_labels))
    assert it["sod"].shape == (T,) and it["doy"].shape == (T,)
    assert it["lon"].shape == (p, p) and it["mask"].shape == (1, p, p)
    print(f"[ok] sample shapes: target {tuple(it['target'].shape)}, cond "
          f"{tuple(it['cond'].shape)}, labels {tuple(it['labels'].shape)}; "
          f"{ds.valid_days.size} window starts in the train split")

    # frames are the preloaded single-step fields at d, d+s, ...
    start = int(ds.valid_days[len(ds.valid_days) // 2])
    ysl, xsl = slice(64, 64 + p), slice(128, 128 + p)
    w = ds.window(start, ysl, xsl)
    frames = ds.frames_of(start)
    for ci, v in enumerate(cfg.target):
        for i, f in enumerate(frames):
            assert np.array_equal(w["target"][i, ci].numpy(), ds.data[v][int(f), ysl, xsl])
    assert frames[-1] <= ds.valid_days[-1] + (T - 1) * cfg.frame_stride
    assert np.all(np.diff(ds.t_unix[frames]) == np.diff(ds.t_unix[:2])[0] * cfg.frame_stride)
    print("[ok] each frame equals the single-step field at its own time; frames "
          "are evenly spaced in real time")

    # denormalisation inverts the preload against the RAW store
    v = cfg.target[0]
    cad = ds.spec.cadence_of(v)
    phys = ds.denormalize_window(v, w["target"][:, 0].numpy(), start, ysl, xsl)
    ocean = ds.ocean[ysl, xsl] > 0.5
    for i, f in enumerate(frames):
        raw = np.asarray(ds.spec.family.read(cad, v, int(f), int(f) + 1))[0][ysl, xsl]
        good = ocean & np.isfinite(raw)
        err = float(np.abs(phys[i][good] - raw[good]).max())
        scale = float(np.abs(raw[good]).max()) + 1e-6
        assert err < 1e-3 * scale + 1e-4, (v, i, err, scale)
    print(f"[ok] denormalize_window reproduces the raw store for {v!r} on every frame")

    # calendar inputs
    if cfg.use_calendar:
        ts = ds.t_unix[frames].astype("datetime64[s]")
        sod_ref = (ts - ts.astype("datetime64[D]")).astype(np.float64)
        assert np.allclose(w["sod"].numpy(), sod_ref)
        day_of_year = (ts.astype("datetime64[D]") - ts.astype("datetime64[Y]")).astype(int)
        # the loader's year phase counts from the 1970 epoch with a 365.2425-day
        # year; it tracks the calendar day of year to within about a day
        assert np.all(np.abs(w["doy"].numpy() * 365.2425 - day_of_year) < 2.0)
        lon = w["lon"].numpy()
        assert -180.0 <= lon.min() and lon.max() <= 360.0 and np.ptp(lon) > 0
        local_h = ((w["sod"][0].item() + lon * 240.0) % 86400.0) / 3600.0
        print(f"[ok] calendar: frame 0 is {ts[0]} UTC, lon {lon.min():.2f}..{lon.max():.2f} "
              f"-> local solar time {local_h.min():.2f}..{local_h.max():.2f} h")

    # the flux switch: move the last generated variable to label_vars. The
    # preloaded arrays are shared, so this costs no second load.
    if len(cfg.target) > 1:
        moved = cfg.target[-1]
        cfg_l = dataclasses.replace(cfg, target=cfg.target[:-1],
                                    label_vars=list(cfg.label_vars) + [moved])
        dl = ZarrVideoDataset(cfg_l, split="train", share_from=ds, spec=ds.spec)
        wl = dl.window(start, ysl, xsl)
        assert wl["target"].shape == (T, ct - 1, p, p)
        m = (ds.ocean[ysl, xsl] > 0.5).astype(np.float32)
        ref = (w["target"][:, -1].numpy() * m).sum(axis=(1, 2)) / m.sum()
        col = len(cfg_l.label_vars) - 1          # store labels come before tidal ones
        assert np.allclose(wl["labels"][:, col].numpy(), ref, atol=1e-6)
        print(f"[ok] moving {moved!r} from target to label_vars turns it into the "
              f"per-frame ocean mean: {np.round(ref, 3).tolist()}")

    # tidal phase labels: computed from the frame times, appended after any
    # store label, the same at every crop. Reference: datetime arithmetic.
    from .tides import SPEED_DEG_PER_HOUR
    names = list(SPEED_DEG_PER_HOUR)
    flux = [cfg.target[-1]] if len(cfg.target) > 1 else []
    cfg_t = dataclasses.replace(cfg, target=[v for v in cfg.target if v not in flux],
                                label_vars=flux, tidal_labels=names)
    dt = ZarrVideoDataset(cfg_t, split="train", share_from=ds, spec=ds.spec)
    wt = dt.window(start, ysl, xsl)
    lab = wt["labels"].numpy()
    assert lab.shape == (T, len(flux) + 2 * len(names)) and lab.dtype == np.float32
    origin = np.datetime64("1970-01-01T00:00:00") + np.timedelta64(int(cfg_t.tidal_epoch), "s")
    hours = (ds.t_unix[frames].astype("datetime64[s]") - origin) / np.timedelta64(1, "h")
    for k, n in enumerate(names):
        ang = np.deg2rad(SPEED_DEG_PER_HOUR[n] * hours)
        ref2 = np.stack([np.cos(ang), np.sin(ang)], axis=1) / np.sqrt(len(names))
        assert np.allclose(lab[:, len(flux) + 2 * k: len(flux) + 2 * k + 2], ref2, atol=1e-5), n
    assert np.allclose((lab[:, len(flux):] ** 2).sum(axis=1), 1.0, atol=1e-5)
    if flux:
        m = (ds.ocean[ysl, xsl] > 0.5).astype(np.float32)
        ref = (w["target"][:, -1].numpy() * m).sum(axis=(1, 2)) / m.sum()
        assert np.allclose(lab[:, 0], ref, atol=1e-6)
    other = dt.window(start, slice(0, p), slice(0, p))["labels"].numpy()
    assert np.array_equal(other[:, len(flux):], lab[:, len(flux):])
    m2 = lab[:, len(flux) + 8: len(flux) + 10] * np.sqrt(len(names))
    step_deg = np.rad2deg(np.unwrap(np.arctan2(m2[:, 1], m2[:, 0])))
    print(f"[ok] tidal labels: {lab.shape[1]} per frame ({len(flux)} store + "
          f"{2 * len(names)} phase), unit norm, crop-independent; M2 advances "
          f"{np.diff(step_deg).mean():.2f} deg per frame from "
          f"{ds.t_unix[frames][0].astype('datetime64[s]')}")

    # collation, as the DataLoader will see it
    from torch.utils.data import default_collate
    b = default_collate([ds[0], ds[1]])
    assert b["target"].shape == (2, T, ct, p, p) and b["cond"].shape == (2, T, cc, p, p)
    ys, xs = ds.full_slices()
    print(f"[ok] batches collate to {tuple(b['target'].shape)}; one-pass full region "
          f"is {ys.stop}x{xs.stop} of {ds.NY}x{ds.NX}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", nargs="?", const="", default=None, metavar="CONFIG",
                    help="window logic only, or pass a config to also test "
                         "against its store")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    if args.selftest is not None:
        _selftest_windows()
        if args.selftest:
            _selftest_store(args.selftest, args.max_days)
        print("data_video selftest passed")
