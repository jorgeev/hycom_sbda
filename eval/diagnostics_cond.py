"""Turn one ``samples_cond_<size>.npz`` into figures, a metric table and a report.

    python -m eval.diagnostics_cond --samples samples_cond_full.npz --out figs_cond_full/

The conditional counterpart of ``eval.diagnostics``. That script asks whether
unconstrained samples LOOK like the ocean; this one has, for every sample, the
truth of the same day and the observations the sample was conditioned on, so it
can also ask whether the ensemble is RIGHT, and whether it is honest about how
right it is. Nine figures:

  figc1_gallery      one day: truth, observation, ensemble mean, two members,
                     spread, error (physical and anomaly units); figc1_days
                     shows the best / median / worst day
  figc2_skill        RMSE and CRPS against the two baselines (the degraded
                     observation, and climatology = zero anomaly); spread vs
                     skill per day; skill through the record
  figc3_rankhist     where the truth falls among the members
  figc4_errmaps      bias, RMSE, spread and calibration maps
  figc5_spectra      power spectra of truth / members / ensemble mean / error,
                     and the coherence between ensemble mean and truth
  figc6_pdf          marginal PDFs; error vs member-deviation PDFs
  figc7_eke          eddy kinetic energy: member realism and ensemble-mean
                     predictability
  figc8_crosschannel channel correlations; correlation of the errors
  figc9_fidelity     obs-conditioned: fit to the observation where observed,
                     spread vs coverage and obs age. mask+doy-conditioned:
                     seasonal consistency and land handling

TWO FAMILIES. ``family() == "obs"`` when ``cond_vars`` is non-empty;
``"maskdoy"`` when the only conditioning is the ocean mask and/or day-of-year.
For the second, paired skill against climatology is expected to be ~0 -- the
question is whether it is a CALIBRATED climatological sampler for that day of
year, which is what figc3 / figc2(c) answer.

UNITS. Skill numbers are in physical units (m, degC, log10 mg/m3) of the
ANOMALY: ``norm * std``. Physical-field panels add the day's own climatology
(``clim_day``), which is the right thing here because every sample is paired
with the truth of the same day. Log10 variables stay in log space, as in
``eval.diagnostics``. Sigma-unit twins of the headline numbers are also
written so variables can be compared to each other.

MEMORY. ``gen_norm`` is (D, K, Ct, H, W) and can be several GB. Every pass here
streams over days and touches one ``(K, H, W)`` slice at a time; nothing
materialises a (D*K, ...) float64 copy. Only ``eval.kernels`` and
``eval.kernels_cond`` are imported.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

from . import kernels as K
from . import kernels_cond as KC

# Copied from eval/diagnostics.py:48-97 (importing it is out of bounds for this
# layer). The rationale for the two pixel-count constants is written there.
STYLE = {
    "ssh":   ("Spectral_r", False), "sst": ("RdYlBu_r", False),
    "sss":   ("viridis",    False), "uag": ("RdBu_r",   True),
    "vag":   ("RdBu_r",     True),  "tau_x": ("PuOr_r", True),
    "tau_y": ("PuOr_r",     True),  "chl": ("YlGn",     False),
    "mld":   ("cividis",    False),
}
UNITS = {"ssh": "m", "sst": "degC", "sss": "psu", "uag": "m/s", "vag": "m/s",
         "tau_x": "N/m2", "tau_y": "N/m2", "mld": "m", "chl": "log10(mg/m3)"}
BANDS = [("mesoscale 40-150 km", 40.0, 150.0), ("submeso 10-40 km", 10.0, 40.0)]
GRAD_ERODE_PX = 6
PATCH_BORDER_PX = 16
STEP_SCORE = 1.9


def _style(v: str) -> tuple[str, bool]:
    return STYLE.get(v.lower(), ("RdBu_r", True))


def _unit(v: str) -> str:
    return UNITS.get(v.lower(), "")


def _band_key(name: str) -> str:
    return name.split()[0]


def _masked(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    out[..., ~(np.asarray(mask) > 0.5)] = np.nan
    return out


def _limits(field2d: np.ndarray, mask: np.ndarray, symmetric: bool):
    v = np.asarray(field2d)[np.asarray(mask) > 0.5]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(v, [2, 98])
    if symmetric:
        m = max(abs(lo), abs(hi), 1e-12)
        return -m, m
    return (lo, hi) if hi > lo else (lo - 1e-6, hi + 1e-6)


def _rms(a, m) -> float:
    v = np.asarray(a, dtype=np.float64)[m]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def _nanmean(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


# ---------------------------------------------------------------------------
# the npz
# ---------------------------------------------------------------------------
class CondSamples:
    """The npz, with unit conventions applied once and per-day accessors.

    ``gen`` stays in its stored dtype (float16 by default) and is upcast one
    ``(K, H, W)`` slice at a time.
    """

    def __init__(self, path: str, members_subsample: int | None = None, seed: int = 0,
                 border_cut: bool = True):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.vars = json.loads(str(d["vars"]))
        self.cond_vars = json.loads(str(d["cond_vars"]))
        self.k_days = int(d["k_days"])
        self.extras = json.loads(str(d["extras"]))
        self.meta = json.loads(str(d["meta"]))
        self.config = json.loads(str(d["config"]))
        self.gen = d["gen_norm"]                                    # (D,K,Ct,H,W), stored dtype
        self.truth = np.asarray(d["truth_norm"], dtype=np.float32)  # (D,Ct,H,W)
        self.std = np.asarray(d["std"], dtype=np.float64)
        self.is_log = np.asarray(d["is_log"], dtype=bool)
        self.clim = np.asarray(d["clim_day"], dtype=np.float32)     # (D,Ct,H,W) or (D,Ct,1,1)
        self.store_cond = self.meta.get("store_cond", "full")
        cn = d["cond_norm"]
        self.cond = cn if cn.shape[1] else None
        self.cond_names = (json.loads(str(d["cond_channel_names"]))
                           if "cond_channel_names" in d.files else [])
        self.obs_avail = np.asarray(d["obs_avail"], dtype=bool)     # (D,Cobs,H,W)
        self.obs_age_s = np.asarray(d["obs_age_s"], dtype=np.float64) if "obs_age_s" in d.files else None
        self.baseline_phys = np.asarray(d["baseline_phys"], dtype=np.float32)
        self.baseline_map = json.loads(str(d["baseline_map"]))
        self.baseline_kind = (json.loads(str(d["baseline_kind"])) if "baseline_kind" in d.files
                              else {v: "absolute" for v in self.vars})
        self.baseline_bias = np.asarray(d["baseline_bias"], dtype=np.float64)
        self.D, self.K, self.Ct, self.ny, self.nx = self.gen.shape
        mask = np.asarray(d["mask"]) > 0.5
        self.mask = (np.broadcast_to(mask, (self.D, self.ny, self.nx)).copy()
                     if mask.ndim == 2 else mask)
        self.mask_full = np.asarray(d["mask_full"], dtype=np.float32)
        lat = np.asarray(d["lat"], dtype=np.float32)
        self.lat = np.broadcast_to(lat, (self.D, self.ny, self.nx)) if lat.ndim == 2 else lat
        self.dx_m = float(d["dx_m"])
        self.dx_km = self.dx_m / 1000.0
        self.size = str(d["size"])
        self.days = np.asarray(d["days"], dtype=np.int64)
        self.doy = np.asarray(d["doy"], dtype=np.float64)
        self.time_unix = np.asarray(d["time_unix"], dtype=np.float64) if "time_unix" in d.files else None
        self.pos = np.asarray(d["pos"], dtype=np.int64)
        self.is_patch = bool(self.pos.size)
        self.step_seconds = self.meta.get("step_seconds") or float("nan")
        self.base_cadence = self.meta.get("base_cadence", "step")
        self._byname = {v.lower(): v for v in self.vars}
        self.crop_stride = 0
        if self.is_patch:
            p0 = self.pos
            dif = np.concatenate([np.diff(np.unique(p0[:, 0])), np.diff(np.unique(p0[:, 1]))])
            self.crop_stride = int(dif.min()) if dif.size else 0
        rng = np.random.default_rng(seed)
        if members_subsample and members_subsample < self.K:
            self.member_idx = np.sort(rng.choice(self.K, members_subsample, replace=False))
        else:
            self.member_idx = np.arange(self.K)

        # -- masks (see eval/diagnostics.py Samples for the measurements) -----
        # Land as the DATA reports it: the loader writes exact 0.0 in every
        # channel at once on land, and truth has no gaps, so this is exact.
        self.data_land = np.all(self.truth == 0.0, axis=1)                  # (D,H,W)
        self.mask_day = self.mask & ~self.data_land                         # (D,H,W)
        self.mask_ocean = self.mask_day.mean(0) > 0.5                       # (H,W)
        self.interior = np.ones((self.ny, self.nx), dtype=bool)
        if border_cut and self.is_patch and 2 * PATCH_BORDER_PX < min(self.ny, self.nx):
            b = PATCH_BORDER_PX
            self.interior[:b, :] = False
            self.interior[-b:, :] = False
            self.interior[:, :b] = False
            self.interior[:, -b:] = False
        self.border_cut = bool((~self.interior).any())
        self.stat_mask = self.mask_day & self.interior
        self.grad_mask = np.stack([K.erode_mask(self.mask_day[i], GRAD_ERODE_PX)
                                   for i in range(self.D)]) & self.interior
        self.clean_day = self.mask_day.reshape(self.D, -1).all(axis=1)
        self.land_any = bool((~self.mask_day).any())

    # -- names ---------------------------------------------------------------
    def name(self, v: str) -> str | None:
        return self._byname.get(v.lower())

    def has(self, *names: str) -> bool:
        return all(self.name(v) is not None for v in names)

    def idx(self, v: str) -> int:
        return self.vars.index(self.name(v) or v)

    def family(self) -> str:
        return "obs" if self.cond_vars else "maskdoy"

    def cond_index(self, v: str) -> int | None:
        """Index into ``cond_vars`` / ``obs_avail`` of the obs that degrades ``v``,
        else the first obs channel, else None."""
        c = self.baseline_map.get(self.name(v) or v)
        if c in self.cond_vars:
            return self.cond_vars.index(c)
        return 0 if self.cond_vars else None

    # -- per-day accessors (anomaly = physical units without climatology) ----
    def gen_norm(self, v: str, d: int, sub: bool = False) -> np.ndarray:
        i = self.idx(v)
        g = self.gen[d, self.member_idx, i] if sub else self.gen[d, :, i]
        return np.asarray(g, dtype=np.float32)

    def truth_norm(self, v: str, d: int) -> np.ndarray:
        return self.truth[d, self.idx(v)]

    def gen_anom(self, v: str, d: int, sub: bool = False) -> np.ndarray:
        return self.gen_norm(v, d, sub) * np.float32(self.std[self.idx(v)])

    def truth_anom(self, v: str, d: int) -> np.ndarray:
        return self.truth_norm(v, d) * np.float32(self.std[self.idx(v)])

    def truth_anom_all(self, v: str) -> np.ndarray:
        return self.truth[:, self.idx(v)] * np.float32(self.std[self.idx(v)])

    def clim_at(self, v: str, d: int) -> np.ndarray:
        return self.clim[d, self.idx(v)]                 # (H,W) or (1,1), broadcastable

    def gen_phys(self, v: str, d: int, sub: bool = False) -> np.ndarray:
        return self.gen_anom(v, d, sub) + self.clim_at(v, d)

    def truth_phys(self, v: str, d: int) -> np.ndarray:
        return self.truth_anom(v, d) + self.clim_at(v, d)

    def baseline_raw(self, v: str, d: int) -> np.ndarray | None:
        """The degraded obs in the truth's units (log10 for log variables),
        NaN where unobserved / unmapped; None if this target has no baseline."""
        i = self.idx(v)
        if self.baseline_map.get(self.vars[i]) is None:
            return None
        b = self.baseline_phys[d, i].astype(np.float64)
        if self.is_log[i]:
            with np.errstate(divide="ignore", invalid="ignore"):
                b = np.where(b > 0, np.log10(b), np.nan)
        return b

    def baseline_pair(self, v: str, d: int, m: np.ndarray):
        """``(base_anom, truth_anom, ok)`` on the pixels ``m & observed``.

        ``absolute`` baselines are compared as ``obs - clim_day`` against the
        truth anomaly. ``anomaly`` baselines (ssha vs ssh) carry no mean
        dynamic topography, so the domain mean over the observed pixels is
        removed from BOTH sides before scoring. Returns None if unmapped."""
        b = self.baseline_raw(v, d)
        if b is None:
            return None
        t = self.truth_anom(v, d).astype(np.float64)
        ok = m & np.isfinite(b)
        if not ok.any():
            return None
        kind = self.baseline_kind.get(self.vars[self.idx(v)], "absolute")
        if kind == "anomaly":
            ba = b - np.nanmean(b[ok])
            ta = t - t[ok].mean()
        else:
            ba = b - self.clim_at(v, d)
            ta = t
        return ba, ta, ok

    def ens_mean_var(self, v: str, d: int):
        return KC.ens_mean_var(self.gen_anom(v, d))

    # -- labels --------------------------------------------------------------
    def run_name(self) -> str:
        ck = self.meta.get("ckpt", "")
        dd = os.path.dirname(ck)
        if os.path.basename(dd) == "checkpoints":
            dd = os.path.dirname(dd)
        return os.path.basename(dd) or "run"

    def cond_desc(self) -> str:
        if self.cond_vars:
            return (f"{len(self.cond_vars)} obs x k={self.k_days}"
                    + (f" + {'+'.join(self.extras)}" if self.extras else ""))
        return "+".join(self.extras) if self.extras else "none"

    def title(self) -> str:
        m = self.meta
        s = m.get("sampler", {})
        return (f"{self.run_name()}  step {m.get('step', 0):,}  |  {self.size} "
                f"{self.ny}x{self.nx}  |  D={self.D} x K={self.K}  |  "
                f"{s.get('num_steps')} steps, s_churn={s.get('s_churn')}  |  cond: {self.cond_desc()}")

    def fold_note(self) -> str:
        if not (self.is_patch and self.crop_stride):
            return ""
        return (f"sample-mean maps in patch geometry are FOLDS of the domain at the "
                f"{self.crop_stride} px crop lattice, both sides -- read the level, not the pattern")

    def mask_note(self) -> str:
        keep = 100.0 * float(self.interior.mean())
        border = (f"{PATCH_BORDER_PX} px frame border dropped both sides ({keep:.0f} % kept) "
                  "for the distributional figures; paired skill reports the halo separately"
                  if self.border_cut else
                  "no frame-border cut" + (" (disabled)" if self.is_patch else
                                           " (full geometry: the frame edge is a real boundary)"))
        return (f"masking: {GRAD_ERODE_PX} px land erosion before any derivative; "
                f"land from the data (all-channel exact zeros) per sample; {border}")

    def day_label(self) -> str:
        return f"target step ({self.base_cadence})"


# ---------------------------------------------------------------------------
# pass 1: paired statistics, streamed over days
# ---------------------------------------------------------------------------
@dataclass
class PairedStats:
    var: str
    rmse_ens: np.ndarray
    mae_ens: np.ndarray
    bias_ens: np.ndarray
    crps: np.ndarray
    spread: np.ndarray
    rmse_member: np.ndarray                   # (D,K)
    rmse_clim: np.ndarray
    mae_clim: np.ndarray
    rmse_obs: np.ndarray
    mae_obs: np.ndarray
    rmse_ens_on_obs_px: np.ndarray
    rmse_obs_px: np.ndarray
    rmse_gap_px: np.ndarray
    coverage: np.ndarray
    obs_age_s: np.ndarray
    obsfit_member: np.ndarray                 # (D,K)
    obsfit_ens: np.ndarray
    obsfit_truth: np.ndarray
    dmean_truth: np.ndarray
    dmean_ens: np.ndarray
    dstd_truth: np.ndarray
    dstd_member: np.ndarray
    land_abs_gen: np.ndarray
    land_quiet_frac: np.ndarray
    n_px: np.ndarray
    rank_counts: np.ndarray
    rank_tv_day: np.ndarray
    bias_map: np.ndarray
    rmse_map: np.ndarray
    spread_map: np.ndarray
    rmse_obs_map: np.ndarray
    crps_climens: np.ndarray
    sq_err_border: float
    n_border: int
    scalars: dict = field(default_factory=dict)

    @staticmethod
    def compute(S: CondSamples, v: str, rng: np.random.Generator) -> "PairedStats":
        D, Kn = S.D, S.K
        i = S.idx(v)
        z = lambda: np.full(D, np.nan)
        rmse_ens, mae_ens, bias_ens, crps, spread = z(), z(), z(), z(), z()
        rmse_member = np.full((D, Kn), np.nan)
        rmse_clim, mae_clim = z(), z()
        rmse_obs, mae_obs, rmse_ens_on_obs = z(), z(), z()
        rmse_obs_px, rmse_gap_px, coverage, obs_age = z(), z(), z(), z()
        obsfit_member = np.full((D, Kn), np.nan)
        obsfit_ens, obsfit_truth = z(), z()
        dmean_t, dmean_e, dstd_t, dstd_m = z(), z(), z(), z()
        land_abs, land_quiet = z(), z()
        n_px = np.zeros(D)
        rank_counts = np.zeros(Kn + 1, dtype=np.int64)
        rank_tv_day = z()
        hw = (S.ny, S.nx)
        sum_err, sum_sq, sum_var, cnt = (np.zeros(hw) for _ in range(4))
        sum_sq_obs, cnt_obs = np.zeros(hw), np.zeros(hw)
        sq_border, n_border = 0.0, 0
        j = S.cond_index(v)

        for d in range(D):
            g = S.gen_anom(v, d).astype(np.float64)              # (K,H,W)
            t = S.truth_anom(v, d).astype(np.float64)
            m = S.stat_mask[d]
            nm = int(m.sum())
            n_px[d] = nm
            if nm == 0:
                continue
            mean = g.mean(0)
            var = g.var(0, ddof=1)
            err = mean - t
            rmse_ens[d] = _rms(err, m)
            mae_ens[d] = float(np.abs(err[m]).mean())
            bias_ens[d] = float(err[m].mean())
            rmse_member[d] = np.sqrt(((g[:, m] - t[m]) ** 2).mean(1))
            spread[d] = float(np.sqrt(var[m].mean()))
            crps[d] = float(KC.crps_ensemble(g, t)[m].mean())
            rk = KC.rank_of_truth(g, t, rng)[m]
            c_d = np.bincount(rk, minlength=Kn + 1)
            rank_counts += c_d
            rank_tv_day[d] = KC.rank_hist_stats(c_d)["tv"]
            rmse_clim[d] = _rms(t, m)
            mae_clim[d] = float(np.abs(t[m]).mean())
            sum_err[m] += err[m]
            sum_sq[m] += err[m] ** 2
            sum_var[m] += var[m]
            cnt[m] += 1
            if S.border_cut:
                ring = S.mask_day[d] & ~S.interior
                sq_border += float((err[ring] ** 2).sum())
                n_border += int(ring.sum())
            # -- observation baseline and fidelity ------------------------
            pair = S.baseline_pair(v, d, m)
            if pair is not None:
                ba, ta, ok = pair
                rmse_obs[d] = _rms(ba - ta, ok)
                mae_obs[d] = float(np.abs((ba - ta)[ok]).mean())
                rmse_ens_on_obs[d] = _rms(err, ok)
                sum_sq_obs[ok] += ((ba - ta)[ok]) ** 2
                cnt_obs[ok] += 1
                # members against the observation itself, where observed
                gk = g
                if S.baseline_kind.get(S.vars[i], "absolute") == "anomaly":
                    gk = g - g[:, ok].mean(1)[:, None, None]
                    ek = mean - mean[ok].mean()
                else:
                    ek = mean
                obsfit_member[d] = np.sqrt(((gk[:, ok] - ba[ok]) ** 2).mean(1))
                obsfit_ens[d] = _rms(ek - ba, ok)
                obsfit_truth[d] = rmse_obs[d]
            if j is not None:
                av = S.obs_avail[d, j] & m
                coverage[d] = av.sum() / nm
                rmse_obs_px[d] = _rms(err, av) if av.any() else np.nan
                gap = m & ~av
                rmse_gap_px[d] = _rms(err, gap) if gap.any() else np.nan
                if S.obs_age_s is not None and S.obs_age_s.shape[1] > j:
                    obs_age[d] = S.obs_age_s[d, j]
            # -- seasonal / domain-level -------------------------------------
            dmean_t[d] = float(t[m].mean())
            dmean_e[d] = float(mean[m].mean())
            dstd_t[d] = float(t[m].std())
            dstd_m[d] = float(g[:, m].std(axis=1).mean())
            # -- land handling (sigma units, all members) --------------------
            if S.land_any:
                land = ~S.mask_day[d]
                if land.any():
                    gl = np.abs(S.gen_norm(v, d)[:, land].astype(np.float64))
                    land_abs[d] = float(gl.mean())
                    land_quiet[d] = float((gl < 0.1).mean())

        # climatological ensemble: the other D-1 truth days as members
        crps_climens = z()
        if D >= 3:
            t_all = S.truth_anom_all(v).astype(np.float64)
            for d in range(D):
                m = S.stat_mask[d]
                if not m.any():
                    continue
                others = np.delete(t_all, d, axis=0)
                crps_climens[d] = float(KC.crps_ensemble(others, t_all[d])[m].mean())

        with np.errstate(divide="ignore", invalid="ignore"):
            bias_map = np.where(cnt > 0, sum_err / cnt, np.nan)
            rmse_map = np.where(cnt > 0, np.sqrt(sum_sq / cnt), np.nan)
            spread_map = np.where(cnt > 0, np.sqrt(sum_var / cnt), np.nan)
            rmse_obs_map = np.where(cnt_obs > 0, np.sqrt(sum_sq_obs / cnt_obs), np.nan)

        P = PairedStats(v, rmse_ens, mae_ens, bias_ens, crps, spread, rmse_member, rmse_clim,
                        mae_clim, rmse_obs, mae_obs, rmse_ens_on_obs, rmse_obs_px, rmse_gap_px,
                        coverage, obs_age, obsfit_member, obsfit_ens, obsfit_truth, dmean_t,
                        dmean_e, dstd_t, dstd_m, land_abs, land_quiet, n_px, rank_counts,
                        rank_tv_day, bias_map, rmse_map, spread_map, rmse_obs_map, crps_climens,
                        sq_border, n_border)
        P.scalars = P._scalars(S, i)
        return P

    def _scalars(self, S: CondSamples, i: int) -> dict:
        Kn = S.K
        w = self.n_px / max(self.n_px.sum(), 1)                 # pixel-weighted day average
        wmean = lambda a: float(np.nansum(a * w) / max(np.nansum(w * np.isfinite(a)), 1e-12)) \
            if np.isfinite(a).any() else float("nan")
        wrms = lambda a: float(np.sqrt(wmean(a ** 2))) if np.isfinite(a).any() else float("nan")
        r = {}
        r["rmse_ens"] = wrms(self.rmse_ens)
        r["mae_ens"] = wmean(self.mae_ens)
        r["bias_ens"] = wmean(self.bias_ens)
        rm = np.sqrt(np.nanmean(self.rmse_member ** 2, axis=0))    # (K,)
        r["rmse_member_mean"] = float(np.sqrt(np.nanmean(rm ** 2)))
        r["rmse_member_min"] = float(np.nanmin(rm))
        r["rmse_member_max"] = float(np.nanmax(rm))
        r["rmse_member_over_ens"] = r["rmse_member_mean"] / r["rmse_ens"] if r["rmse_ens"] > 0 else np.nan
        r["rmse_member_over_ens_target"] = KC.member_over_ensmean_target(Kn)
        r["crps"] = wmean(self.crps)
        r["spread"] = wrms(self.spread)
        r["spread_skill_ratio"] = KC.spread_skill_ratio(r["spread"] ** 2, r["rmse_ens"] ** 2, Kn)
        rk = KC.rank_hist_stats(self.rank_counts)
        r["rank_tv"], r["rank_tails_ratio"], r["rank_slope"] = rk["tv"], rk["tails_ratio"], rk["slope"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log10(self.spread_map * np.sqrt((Kn + 1) / Kn) / self.rmse_map)
        ok = np.isfinite(lr)
        r["frac_px_calibrated"] = float((np.abs(lr[ok]) < np.log10(1.5)).mean()) if ok.any() else np.nan
        r["rmse_border_over_interior"] = (
            float(np.sqrt(self.sq_err_border / self.n_border) / r["rmse_ens"])
            if self.n_border > 0 and r["rmse_ens"] > 0 else np.nan)
        s = S.std[i]
        r["rmse_ens_sigma"] = r["rmse_ens"] / s
        r["crps_sigma"] = r["crps"] / s
        r["spread_sigma"] = r["spread"] / s
        # baselines
        v = S.vars[i]
        r["baseline_var"] = S.baseline_map.get(v) or ""
        r["baseline_kind"] = S.baseline_kind.get(v, "") if S.baseline_map.get(v) else ""
        r["rmse_obs"] = wrms(self.rmse_obs)
        r["mae_obs"] = wmean(self.mae_obs)
        r["rmse_ens_on_obs_px"] = wrms(self.rmse_ens_on_obs_px)
        r["rmse_clim"] = wrms(self.rmse_clim)
        r["mae_clim"] = wmean(self.mae_clim)
        r["ss_rmse_obs"] = (1 - r["rmse_ens_on_obs_px"] / r["rmse_obs"]
                            if r["rmse_obs"] > 0 else np.nan)
        r["ss_rmse_clim"] = 1 - r["rmse_ens"] / r["rmse_clim"] if r["rmse_clim"] > 0 else np.nan
        r["crps_over_mae_obs"] = r["crps"] / r["mae_obs"] if r["mae_obs"] > 0 else np.nan
        r["crps_climens"] = wmean(self.crps_climens)
        r["crpss_climens"] = (1 - r["crps"] / r["crps_climens"]
                              if r["crps_climens"] > 0 else np.nan)
        # fidelity
        r["obsfit_rmse_member"] = float(np.sqrt(np.nanmean(self.obsfit_member ** 2))) \
            if np.isfinite(self.obsfit_member).any() else np.nan
        r["obsfit_rmse_ens"] = wrms(self.obsfit_ens)
        r["obsfit_rmse_truth"] = wrms(self.obsfit_truth)
        r["rmse_obs_px"] = wrms(self.rmse_obs_px)
        r["rmse_gap_px"] = wrms(self.rmse_gap_px)
        r["coverage_mean"] = wmean(self.coverage)
        r["spread_vs_coverage_slope"] = _slope(self.coverage, self.spread / s)
        age_h = self.obs_age_s / 3600.0
        r["spread_vs_age_slope"] = _slope(age_h, self.spread / s) if np.nanstd(age_h) > 0 else np.nan
        # seasonal / domain
        okd = np.isfinite(self.dmean_truth) & np.isfinite(self.dmean_ens)
        r["dmean_corr"] = (float(np.corrcoef(self.dmean_truth[okd], self.dmean_ens[okd])[0, 1])
                           if okd.sum() > 2 and np.std(self.dmean_truth[okd]) > 0 else np.nan)
        r["dmean_rmse_ens"] = float(np.sqrt(np.mean((self.dmean_ens - self.dmean_truth)[okd] ** 2))) \
            if okd.any() else np.nan
        r["dmean_rmse_const"] = float(np.std(self.dmean_truth[okd])) if okd.any() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            r["dstd_ratio_mean"] = float(np.nanmean(self.dstd_member / self.dstd_truth))
        r["land_abs_gen_sigma"] = (float(np.nanmean(self.land_abs_gen))
                                   if np.isfinite(self.land_abs_gen).any() else np.nan)
        r["land_quiet_frac"] = (float(np.nanmean(self.land_quiet_frac))
                                if np.isfinite(self.land_quiet_frac).any() else np.nan)
        return r

    def order(self) -> np.ndarray:
        """Day indices sorted from best to worst ensemble-mean RMSE."""
        r = np.where(np.isfinite(self.rmse_ens), self.rmse_ens, np.inf)
        return np.argsort(r)


def _slope(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


# ---------------------------------------------------------------------------
# pass 2: spectra, streamed over (clean) days
# ---------------------------------------------------------------------------
@dataclass
class SpectralStats:
    var: str
    k: np.ndarray
    psd_truth: np.ndarray            # (n, nk)
    psd_member: np.ndarray           # (n*Ksub, nk)
    psd_ensmean: np.ndarray          # (n, nk)
    psd_err_ens: np.ndarray          # (n, nk)
    psd_err_member: np.ndarray       # (n*Ksub, nk)
    psd_spread: np.ndarray           # (n, nk)  mean over members of PSD(member - mean)
    psd_obs: np.ndarray | None
    psd_err_obs: np.ndarray | None
    gamma2: np.ndarray
    n_days: int
    tile: int
    scalars: dict = field(default_factory=dict)

    @staticmethod
    def compute(S: CondSamples, v: str, tile: int, stride: int) -> "SpectralStats | None":
        # Patch geometry: every sample is its own rectangle with its own land, so
        # the FFT runs on the WHOLE frame of the land-free samples only (both
        # sides -- they are paired). Full geometry: the shared land is fixed, so
        # all-ocean tiles from the shared mask serve every sample.
        if S.is_patch:
            days = np.flatnonzero(S.clean_day)
            tile, stride = min(S.ny, S.nx), min(S.ny, S.nx)
            mask = np.ones((S.ny, S.nx), dtype=bool)        # the selected samples ARE all ocean
        else:
            days = np.arange(S.D)
            mask = S.mask_ocean
        if days.size == 0:
            return None
        pt, pm, pens, pe, pem, psp, po, peo = [], [], [], [], [], [], [], []
        saa = sbb = sab = None
        k = None
        for d in days:
            g = S.gen_anom(v, d).astype(np.float64)
            t = S.truth_anom(v, d).astype(np.float64)
            mean = g.mean(0)
            gs = g[S.member_idx]
            k, a = K.mean_radial_psd(t[None], mask, S.dx_km, tile, stride)
            pt.append(a[0])
            _, a = K.mean_radial_psd(gs, mask, S.dx_km, tile, stride)
            pm.append(a)
            _, a = K.mean_radial_psd(mean[None], mask, S.dx_km, tile, stride)
            pens.append(a[0])
            _, a = K.mean_radial_psd((mean - t)[None], mask, S.dx_km, tile, stride)
            pe.append(a[0])
            _, a = K.mean_radial_psd(gs - t[None], mask, S.dx_km, tile, stride)
            pem.append(a)
            _, a = K.mean_radial_psd(gs - mean[None], mask, S.dx_km, tile, stride)
            psp.append(a.mean(0))
            _, paa, pbb, pab = KC.mean_radial_cross(mean[None], t, mask, S.dx_km, tile, stride)
            saa = paa[0] if saa is None else saa + paa[0]
            sbb = pbb[0] if sbb is None else sbb + pbb[0]
            sab = pab[0] if sab is None else sab + pab[0]
            b = S.baseline_raw(v, d)
            if b is not None:
                kind = S.baseline_kind.get(S.vars[S.idx(v)], "absolute")
                ba = (b - np.nanmean(b[mask])) if kind == "anomaly" else (b - S.clim_at(v, d))
                if np.isfinite(ba[mask]).all() and (not S.is_patch or np.isfinite(ba).all()):
                    ba = np.where(np.isfinite(ba), ba, 0.0)
                    _, a = K.mean_radial_psd(ba[None], mask, S.dx_km, tile, stride)
                    po.append(a[0])
                    tt = t - t[mask].mean() if kind == "anomaly" else t
                    _, a = K.mean_radial_psd((ba - tt)[None], mask, S.dx_km, tile, stride)
                    peo.append(a[0])
        SP = SpectralStats(v, k, np.asarray(pt), np.concatenate(pm, 0), np.asarray(pens),
                           np.asarray(pe), np.concatenate(pem, 0), np.asarray(psp),
                           np.asarray(po) if po else None, np.asarray(peo) if peo else None,
                           KC.coherence(sab, saa, sbb), int(days.size), tile)
        SP.scalars = SP._scalars(S)
        return SP

    def _scalars(self, S: CondSamples) -> dict:
        k = self.k
        med = lambda a: np.median(a, 0)
        pt, pm, pe = med(self.psd_truth), med(self.psd_member), med(self.psd_err_ens)
        pem, psp, pens = med(self.psd_err_member), med(self.psd_spread), med(self.psd_ensmean)
        Kn = S.K
        r = {}
        for name, lo, hi in BANDS:
            b = _band_key(name)
            r[f"psd_ratio_member_{b}"] = K.band_ratio(k, pm, pt, lo, hi)
            r[f"psd_ratio_ensmean_{b}"] = K.band_ratio(k, pens, pt, lo, hi)
            bt = KC.band_power(k, pt, lo, hi)
            be = KC.band_power(k, pe, lo, hi)
            r[f"band_nrmse_ens_{b}"] = float(np.sqrt(be / bt)) if bt > 0 else np.nan
            r[f"band_nrmse_member_{b}"] = float(np.sqrt(KC.band_power(k, pem, lo, hi) / bt)) if bt > 0 else np.nan
            r[f"band_nrmse_obs_{b}"] = (float(np.sqrt(KC.band_power(k, med(self.psd_err_obs), lo, hi) / bt))
                                        if self.psd_err_obs is not None and bt > 0 else np.nan)
            # PSD(member - mean) carries sigma^2 (K-1)/K; the mean's error sigma^2 (K+1)/K
            bs = KC.band_power(k, psp, lo, hi)
            r[f"band_spread_skill_{b}"] = (float(np.sqrt(bs * (Kn + 1) / max(Kn - 1, 1) / be))
                                           if be > 0 and Kn > 1 else np.nan)
        r["coh50_km"] = KC.coherence_half_wavelength(k, self.gamma2)
        resolved = (2 * S.dx_km, 128 * S.dx_km)
        r["logdist_member"] = KC.log_distance(k, pm, pt, *resolved)
        r["logdist_ensmean"] = KC.log_distance(k, pens, pt, *resolved)
        r["n_clean_days"] = self.n_days
        return r


# ---------------------------------------------------------------------------
# figc1: galleries
# ---------------------------------------------------------------------------
def _gallery_row(S: CondSamples, v: str, d: int, axes, anom: bool, first_row: bool):
    cmap, sym = _style(v)
    if anom:
        cmap, sym = "RdBu_r", True
    m = S.mask_day[d]
    g = S.gen_anom(v, d) if anom else S.gen_phys(v, d)
    t = S.truth_anom(v, d) if anom else S.truth_phys(v, d)
    b = S.baseline_raw(v, d)
    if b is not None:
        b = (b - S.clim_at(v, d)) if anom else b
        if anom and S.baseline_kind.get(S.vars[S.idx(v)]) == "anomaly":
            b = S.baseline_raw(v, d)
    mean, var = g.mean(0), g.var(0, ddof=1)
    lo, hi = _limits(t, m, sym)
    panels = [("truth", t, cmap, (lo, hi)),
              ("observation" if b is not None else "(no obs baseline)", b, cmap, (lo, hi)),
              ("ensemble mean", mean, cmap, (lo, hi)),
              ("member 1", g[0], cmap, (lo, hi)),
              ("member 2" if S.K > 1 else "member 1", g[min(1, S.K - 1)], cmap, (lo, hi)),
              ("spread (std)", np.sqrt(var), "magma", (0, _limits(np.sqrt(var), m, False)[1])),
              ("ens mean - truth", mean - t, "PuOr_r", _limits(mean - t, m, True))]
    for ax, (ttl, arr, cm, (a, bb)) in zip(axes, panels):
        if arr is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "no mapped\nobservation", ha="center", va="center", fontsize=8)
            continue
        im = ax.imshow(_masked(arr, m), origin="lower", cmap=cm, vmin=a, vmax=bb,
                       interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if first_row:
            ax.set_title(ttl, fontsize=8, pad=3)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=5)
    axes[0].set_ylabel(f"{v}\n[{_unit(v)}]", fontsize=8)


def fig_gallery(S: CondSamples, P: dict, out: str, units: str = "physical", day: int | None = None):
    """One target step: truth, observation, ensemble mean, two members, spread, error."""
    anom = units == "anomaly"
    if day is None:
        o = P[S.vars[0]].order()
        day = int(o[len(o) // 2])                          # the median-skill day
    fig, axes = plt.subplots(len(S.vars), 7, figsize=(2.3 * 7 + 1.0, 1.9 * len(S.vars) + 1.0),
                             squeeze=False)
    for r, v in enumerate(S.vars):
        _gallery_row(S, v, day, axes[r], anom, r == 0)
    kind = "ANOMALIES (what the model drew)" if anom else "physical units (anomaly + the day's climatology)"
    loc = f", crop at {tuple(S.pos[day])}" if S.is_patch else ""
    fig.suptitle(f"Conditional ensemble, {kind}  --  step {S.days[day]} (doy {S.doy[day]:.0f}{loc})"
                 f"  --  {S.title()}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_days(S: CondSamples, P: dict, out: str):
    """The first variable on its best, median and worst day by ensemble-mean RMSE."""
    v = S.vars[0]
    o = P[v].order()
    picks = [("best", int(o[0])), ("median", int(o[len(o) // 2])), ("worst", int(o[-1]))]
    fig, axes = plt.subplots(3, 7, figsize=(2.3 * 7 + 1.0, 1.9 * 3 + 1.0), squeeze=False)
    for r, (lab, d) in enumerate(picks):
        _gallery_row(S, v, d, axes[r], False, r == 0)
        axes[r][0].set_ylabel(f"{lab}: step {S.days[d]}\nRMSE {P[v].rmse_ens[d]:.3g} {_unit(v)}",
                              fontsize=8)
    fig.suptitle(f"{v}: best / median / worst target step by ensemble-mean RMSE  --  {S.title()}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figc2: skill
# ---------------------------------------------------------------------------
def fig_skill(S: CondSamples, P: dict, out: str):
    nv = len(S.vars)
    fig, axes = plt.subplots(nv, 4, figsize=(19, 3.9 * nv), squeeze=False)
    x_days = S.days
    for r, v in enumerate(S.vars):
        p = P[v]
        sc = p.scalars
        u = _unit(v)
        # (a) RMSE bars
        ax = axes[r][0]
        labels = ["ens mean", "members", "obs baseline", "climatology"]
        vals = [sc["rmse_ens"], sc["rmse_member_mean"], sc["rmse_obs"], sc["rmse_clim"]]
        cols = ["tab:blue", "crimson", "0.45", "0.7"]
        ax.bar(range(4), [0 if not np.isfinite(x) else x for x in vals], color=cols)
        rm = np.sqrt(np.nanmean(p.rmse_member ** 2, axis=0))
        ax.boxplot([rm[np.isfinite(rm)]], positions=[1], widths=0.35, showfliers=False,
                   medianprops=dict(color="k"))
        ax.axhline(sc["rmse_ens"] * sc["rmse_member_over_ens_target"], color="crimson", ls=":",
                   lw=1, label=f"calibrated member RMSE = {sc['rmse_member_over_ens_target']:.2f} x mean")
        for i_, x in enumerate(vals):
            if np.isfinite(x):
                ax.text(i_, x, f"{x:.3g}", ha="center", va="bottom", fontsize=7)
            else:
                ax.text(i_, 0, "n/a", ha="center", va="bottom", fontsize=7, color="0.4")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"RMSE [{u}]", fontsize=8)
        ax.set_title(f"{v}: RMSE vs baselines  (SS_obs {sc['ss_rmse_obs']:+.2f}, "
                     f"SS_clim {sc['ss_rmse_clim']:+.2f})", fontsize=9)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, axis="y", lw=0.4)
        ax.tick_params(labelsize=7)
        # (b) CRPS
        ax = axes[r][1]
        labels = ["CRPS ens", "MAE obs", "MAE clim", "CRPS clim-ens"]
        vals = [sc["crps"], sc["mae_obs"], sc["mae_clim"], sc["crps_climens"]]
        ax.bar(range(4), [0 if not np.isfinite(x) else x for x in vals],
               color=["tab:blue", "0.45", "0.7", "0.55"])
        for i_, x in enumerate(vals):
            ax.text(i_, x if np.isfinite(x) else 0, f"{x:.3g}" if np.isfinite(x) else "n/a",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"[{u}]", fontsize=8)
        ax.set_title(f"probabilistic: CRPS / MAE_obs = {sc['crps_over_mae_obs']:.2f}, "
                     f"CRPSS vs clim-ens = {sc['crpss_climens']:+.2f}", fontsize=9)
        ax.grid(alpha=0.25, axis="y", lw=0.4); ax.tick_params(labelsize=7)
        # (c) spread vs skill per day
        ax = axes[r][2]
        ok = np.isfinite(p.spread) & np.isfinite(p.rmse_ens)
        sc_ = ax.scatter(p.rmse_ens[ok], p.spread[ok] * np.sqrt((S.K + 1) / S.K), c=S.doy[ok],
                         cmap="twilight", s=18, edgecolor="k", linewidth=0.3)
        lim = float(np.nanmax(np.concatenate([p.rmse_ens[ok], p.spread[ok]]))) * 1.1 if ok.any() else 1
        ax.plot([0, lim], [0, lim], "k-", lw=1, label="calibrated")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"RMSE of ensemble mean [{u}]", fontsize=8)
        ax.set_ylabel(f"spread x sqrt((K+1)/K) [{u}]", fontsize=8)
        ax.set_title(f"per-step spread vs skill: ratio {sc['spread_skill_ratio']:.2f} (1 = calibrated)",
                     fontsize=9)
        plt.colorbar(sc_, ax=ax, fraction=0.04).set_label("day of year", fontsize=7)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.4); ax.tick_params(labelsize=7)
        # (d) skill through the record
        ax = axes[r][3]
        ax.plot(x_days, p.rmse_clim, color="0.7", lw=1.2, label="climatology RMSE")
        if np.isfinite(p.rmse_obs).any():
            ax.plot(x_days, p.rmse_obs, color="0.45", lw=1.2, label="obs baseline RMSE")
        ax.plot(x_days, p.rmse_ens, color="tab:blue", lw=1.6, label="ens mean RMSE")
        ax.plot(x_days, p.spread, color="crimson", lw=1.2, ls="--", label="spread")
        ax.plot(x_days, p.crps, color="tab:green", lw=1.2, label="CRPS")
        ax.set_xlabel(S.day_label(), fontsize=8)
        ax.set_ylabel(f"[{u}]", fontsize=8)
        ax.set_title("through the record", fontsize=9)
        ax.legend(fontsize=7, frameon=False, ncol=2); ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
    fig.suptitle("Paired ensemble skill  --  " + S.title() + "\n" + S.mask_note(), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figc3: rank histograms
# ---------------------------------------------------------------------------
def fig_rankhist(S: CondSamples, P: dict, out: str):
    nv = len(S.vars)
    fig, axes = plt.subplots(1, nv, figsize=(4.6 * nv, 4.2), squeeze=False)
    for ax, v in zip(axes[0], S.vars):
        p = P[v]
        c = p.rank_counts.astype(np.float64)
        pr = c / max(c.sum(), 1) * (S.K + 1)
        ax.bar(range(S.K + 1), pr, color="tab:blue", width=0.9)
        ax.axhline(1.0, color="k", lw=1.2)
        sc = p.scalars
        ax.set_title(f"{v}: TV {sc['rank_tv']:.3f}, tails x{sc['rank_tails_ratio']:.2f}, "
                     f"slope {sc['rank_slope']:+.2f}", fontsize=9)
        ax.set_xlabel("rank of truth among members (0 = below all)", fontsize=8)
        ax.set_ylabel("frequency / uniform", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25, axis="y", lw=0.4)
        ins = ax.inset_axes([0.55, 0.62, 0.42, 0.33])
        ins.plot(S.days, p.rank_tv_day, ".-", color="0.3", lw=0.8, ms=3)
        ins.axhline(0.2, color="crimson", lw=0.8, ls=":")
        ins.set_title("per-step TV", fontsize=6)
        ins.tick_params(labelsize=5); ins.set_ylim(bottom=0)
    fig.suptitle("Rank histograms: U-shape = under-dispersed, hump = over-dispersed, slope = bias\n"
                 + S.title(), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figc4: error maps
# ---------------------------------------------------------------------------
def fig_errmaps(S: CondSamples, P: dict, out: str):
    nv = len(S.vars)
    has_obs = any(np.isfinite(P[v].rmse_obs_map).any() for v in S.vars)
    ncol = 5 if has_obs else 4
    fig, axes = plt.subplots(nv, ncol, figsize=(3.6 * ncol + 0.8, 3.0 * nv + 1.0), squeeze=False)
    for r, v in enumerate(S.vars):
        p = P[v]
        u = _unit(v)
        bl = _limits(p.bias_map, np.isfinite(p.bias_map), True)
        vmax = float(np.nanpercentile(np.concatenate([p.rmse_map[np.isfinite(p.rmse_map)],
                                                      p.spread_map[np.isfinite(p.spread_map)]]), 99)) \
            if np.isfinite(p.rmse_map).any() else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log10(p.spread_map * np.sqrt((S.K + 1) / S.K) / p.rmse_map)
        panels = [("bias (mean - truth)", p.bias_map, "PuOr_r", bl, u),
                  ("RMSE of ensemble mean", p.rmse_map, "magma", (0, vmax), u),
                  ("spread", p.spread_map, "magma", (0, vmax), u),
                  ("log10(spread / RMSE), 0 = calibrated", lr, "RdBu_r", (-0.5, 0.5), "")]
        if has_obs:
            panels.append(("RMSE of obs baseline", p.rmse_obs_map, "magma", (0, vmax), u))
        for ax, (ttl, arr, cm, (a, b), uu) in zip(axes[r], panels):
            im = ax.imshow(arr, origin="lower", cmap=cm, vmin=a, vmax=b, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ttl, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(uu, fontsize=7)
        axes[r][0].set_ylabel(v, fontsize=9)
        axes[r][3].set_xlabel(f"{100 * p.scalars['frac_px_calibrated']:.0f} % of pixels within x1.5",
                              fontsize=7, color="0.35")
    fold = ("\n" + S.fold_note()) if S.fold_note() else ""
    fig.suptitle("Error, spread and calibration maps (mean over target steps)  --  " + S.title() + fold,
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figc5: spectra + coherence
# ---------------------------------------------------------------------------
def fig_spectra(S: CondSamples, SP: dict, out: str):
    vars_ok = [v for v in S.vars if SP.get(v) is not None]
    if not vars_ok:
        print("[diag] spectra: no land-free samples at all -- figure skipped")
        return
    ncol = 4
    nrow = -(-len(vars_ok) // ncol)
    fig = plt.figure(figsize=(16, 4.6 * nrow + 4.6))
    gs = GridSpec(nrow + 1, ncol, figure=fig, height_ratios=[1] * nrow + [1.15], hspace=0.45,
                  wspace=0.30, left=0.075, right=0.985, top=0.90, bottom=0.07)
    ratio_ax = fig.add_subplot(gs[nrow, :2])
    coh_ax = fig.add_subplot(gs[nrow, 2:])
    patch_km = 128 * S.dx_km
    for i, v in enumerate(vars_ok):
        sp = SP[v]
        lam = 1.0 / sp.k
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        for arr, col, lab, band in ((sp.psd_truth, "k", "truth", True),
                                    (sp.psd_member, "crimson", "members", True),
                                    (sp.psd_ensmean, "tab:blue", "ensemble mean", False),
                                    (sp.psd_err_ens, "tab:orange", "ens-mean error", False)):
            med = np.median(arr, 0)
            if band:
                ax.fill_between(lam, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0),
                                color=col, alpha=0.18, lw=0)
            ax.plot(lam, med, color=col, lw=1.5, ls="--" if lab == "ens-mean error" else "-", label=lab)
        if sp.psd_obs is not None:
            ax.plot(lam, np.median(sp.psd_obs, 0), color="0.45", lw=1.2, label="obs baseline")
            ax.plot(lam, np.median(sp.psd_err_obs, 0), color="0.45", lw=1.0, ls=":", label="obs error")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.axvline(2 * S.dx_km, color="0.5", ls=":", lw=1)
        ax.axvline(patch_km, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
        if np.isfinite(sp.scalars["coh50_km"]):
            ax.axvline(sp.scalars["coh50_km"], color="tab:green", lw=1, ls="-.")
        ax.set_title(f"{v}  [{_unit(v)}]   coh50 = {sp.scalars['coh50_km']:.0f} km", fontsize=9)
        ax.set_xlabel("wavelength [km]", fontsize=8); ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25, which="both", lw=0.4)
        if i == 0:
            ax.legend(fontsize=6.5, frameon=False)
        pt = np.median(sp.psd_truth, 0)
        ratio_ax.plot(lam, np.median(sp.psd_member, 0) / pt, lw=1.4, label=f"{v} members")
        ratio_ax.plot(lam, np.median(sp.psd_ensmean, 0) / pt, lw=1.0, ls="--",
                      color=ratio_ax.lines[-1].get_color(), label=f"{v} ens mean")
        coh_ax.plot(lam, sp.gamma2, lw=1.4, label=v)
    ratio_ax.axhline(1.0, color="k", lw=1)
    ratio_ax.axvline(2 * S.dx_km, color="0.5", ls=":", lw=1)
    for (name, lo, hi), col in zip(BANDS, ("tab:green", "tab:purple")):
        ratio_ax.axvspan(lo, hi, color=col, alpha=0.09, lw=0)
        coh_ax.axvspan(lo, hi, color=col, alpha=0.09, lw=0)
    ratio_ax.set_xscale("log"); ratio_ax.set_yscale("log"); ratio_ax.invert_xaxis()
    ratio_ax.set_xlabel("wavelength [km]"); ratio_ax.set_ylabel("PSD ratio  / truth")
    ratio_ax.set_title("members: realism (1 = right variance). ens mean: falls where scales are "
                       "unpredictable -- expected", fontsize=8.5)
    ratio_ax.grid(alpha=0.25, which="both", lw=0.4)
    ratio_ax.legend(fontsize=6.5, frameon=False, ncol=2)
    coh_ax.axhline(0.5, color="k", lw=1, ls=":")
    coh_ax.set_xscale("log"); coh_ax.invert_xaxis(); coh_ax.set_ylim(0, 1.02)
    coh_ax.set_xlabel("wavelength [km]"); coh_ax.set_ylabel("coherence^2 (ens mean, truth)")
    coh_ax.set_title("which scales does the ensemble mean place correctly? (0.5 crossing = coh50)",
                     fontsize=8.5)
    coh_ax.grid(alpha=0.25, which="both", lw=0.4); coh_ax.legend(fontsize=7, frameon=False)
    n = SP[vars_ok[0]].n_days
    fig.suptitle(f"Radial power spectra of anomalies ({SP[vars_ok[0]].tile} px tiles, {n}/{S.D} "
                 f"land-free steps)  --  " + S.title(), fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figc6: PDFs
# ---------------------------------------------------------------------------
def _pooled_values(S: CondSamples, v: str, which: str, n_max: int, rng) -> np.ndarray:
    """Ocean-pixel values pooled over days (and members), subsampled to n_max."""
    per = max(n_max // S.D, 1000)
    out = []
    for d in range(S.D):
        m = S.stat_mask[d]
        if which == "truth":
            vals = S.truth_anom(v, d)[m]
        elif which == "member":
            vals = S.gen_anom(v, d, sub=True)[:, m].ravel()
        elif which == "error":
            vals = (S.gen_anom(v, d).mean(0) - S.truth_anom(v, d))[m]
        else:  # deviation of members from their mean
            g = S.gen_anom(v, d, sub=True)
            vals = (g - g.mean(0)).reshape(g.shape[0], -1)[:, m.ravel()].ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size > per:
            vals = rng.choice(vals, per, replace=False)
        out.append(vals)
    return np.concatenate(out) if out else np.zeros(0)


def fig_pdf(S: CondSamples, out: str, rng) -> dict:
    res = {}
    ncol = 4
    nrow = -(-(len(S.vars) + 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.7 * nrow), squeeze=False)
    for i, v in enumerate(S.vars):
        ax = axes.flat[i]
        t = _pooled_values(S, v, "truth", 200_000, rng)
        g = _pooled_values(S, v, "member", 200_000, rng)
        e = _pooled_values(S, v, "error", 100_000, rng)
        dv = _pooled_values(S, v, "dev", 100_000, rng)
        if t.size < 10 or g.size < 10:
            ax.axis("off")
            continue
        grid = np.linspace(*np.percentile(np.concatenate([t, g]), [0.05, 99.95]), 400)
        ax.plot(grid, stats.gaussian_kde(t)(grid), "k", lw=1.6, label="truth")
        ax.plot(grid, stats.gaussian_kde(g)(grid), "crimson", lw=1.6, label="members")
        if e.size > 10 and dv.size > 10:
            ax.plot(grid, stats.gaussian_kde(e)(grid), "tab:orange", lw=0.9, ls="--", label="ens-mean error")
            ax.plot(grid, stats.gaussian_kde(dv)(grid), "tab:blue", lw=0.9, ls=":", label="member - mean")
        ax.set_yscale("log")
        mg, mt = K.moments(g), K.moments(t)
        res[v] = dict(zip(("mean_member", "std_member", "skew_member", "kurt_member"), mg))
        res[v].update(dict(zip(("mean_truth", "std_truth", "skew_truth", "kurt_truth"), mt)))
        res[v]["std_ratio"] = mg[1] / mt[1] if mt[1] > 0 else np.nan
        em = np.concatenate([(S.gen_anom(v, d).mean(0))[S.stat_mask[d]] for d in range(S.D)])
        res[v]["std_ratio_ensmean"] = float(em.std() / mt[1]) if mt[1] > 0 else np.nan
        ax.set_title(f"{v} anomaly [{_unit(v)}]   std member/truth {res[v]['std_ratio']:.2f}, "
                     f"mean/truth {res[v]['std_ratio_ensmean']:.2f}", fontsize=8.5)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    for ax in axes.flat[len(S.vars):]:
        ax.axis("off")
    axes.flat[len(S.vars)].text(0.0, 0.5,
                                "Anomaly units. Members are pooled over\nsteps and members; truth over "
                                "steps.\n\nThe member PDF should match the truth\n(realism). The ens-mean "
                                "error PDF and\nthe member-deviation PDF should match\nEACH OTHER up to "
                                "sqrt((K+1)/(K-1))\n(calibration).",
                                fontsize=8.5, va="center")
    fig.suptitle("Marginal distributions over ocean pixels  --  " + S.title(), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# figc7: EKE
# ---------------------------------------------------------------------------
def _step_rate_fields(fields: dict[str, np.ndarray], mask: np.ndarray) -> float:
    """Fraction of masked pixels on a one-cell step in EVERY channel at once
    (``eval.diagnostics._step_rate`` applied to a dict of (N,H,W) arrays)."""
    sc = None
    for a in fields.values():
        best = np.zeros_like(a)
        for ax in (1, 2):
            d2 = np.abs(np.roll(a, -1, ax) - np.roll(a, 1, ax))
            d4 = np.abs(np.roll(a, -2, ax) - np.roll(a, 2, ax))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(d4 > 0, 2.0 * d2 / d4, 0.0)
            mm = np.broadcast_to(mask, a.shape)
            q = np.quantile(d2[mm], 0.9) if mm.any() else np.inf
            best = np.maximum(best, np.where(d2 > q, r, 0.0))
        sc = best if sc is None else np.minimum(sc, best)
    mm = np.broadcast_to(mask, sc.shape)
    return float(np.mean(sc[mm] > STEP_SCORE)) if mm.any() else float("nan")


def fig_eke(S: CondSamples, out: str) -> dict:
    res = {}
    if not S.has("ssh"):
        print("[diag] eke: no ssh channel -- skipped")
        return res
    has_ag = S.has("uag", "vag")
    parts = ["geostrophic"] + (["ageostrophic", "total"] if has_ag else [])
    acc = {p: {"truth": np.zeros((S.ny, S.nx)), "member": np.zeros((S.ny, S.nx)),
               "ensmean": np.zeros((S.ny, S.nx))} for p in parts}
    cnt = np.zeros((S.ny, S.nx))
    per_day = {p: {"truth": np.full(S.D, np.nan), "member": np.full(S.D, np.nan),
                   "ensmean": np.full(S.D, np.nan)} for p in parts}
    step_t, step_m = [], []
    for d in range(S.D):
        gm = S.grad_mask[d]
        if not gm.any():
            continue
        lat = S.lat[d]
        ssh_t = S.truth_anom("ssh", d).astype(np.float64)
        ssh_g = S.gen_anom("ssh", d, sub=True).astype(np.float64)
        ssh_e = S.gen_anom("ssh", d).mean(0).astype(np.float64)
        vel = {}
        ug, vg = K.geostrophic_uv(ssh_t, S.dx_m, lat)
        vel["truth"] = {"geostrophic": (ug, vg)}
        ue, ve = K.geostrophic_uv(ssh_e, S.dx_m, lat)
        vel["ensmean"] = {"geostrophic": (ue, ve)}
        ugm = np.empty_like(ssh_g); vgm = np.empty_like(ssh_g)
        for j in range(ssh_g.shape[0]):
            ugm[j], vgm[j] = K.geostrophic_uv(ssh_g[j], S.dx_m, lat)
        vel["member"] = {"geostrophic": (ugm, vgm)}
        if has_ag:
            ua_t, va_t = S.truth_anom("uag", d), S.truth_anom("vag", d)
            ua_g, va_g = S.gen_anom("uag", d, sub=True), S.gen_anom("vag", d, sub=True)
            ua_e, va_e = S.gen_anom("uag", d).mean(0), S.gen_anom("vag", d).mean(0)
            vel["truth"]["ageostrophic"] = (ua_t, va_t); vel["truth"]["total"] = (ug + ua_t, vg + va_t)
            vel["member"]["ageostrophic"] = (ua_g, va_g); vel["member"]["total"] = (ugm + ua_g, vgm + va_g)
            vel["ensmean"]["ageostrophic"] = (ua_e, va_e); vel["ensmean"]["total"] = (ue + ua_e, ve + va_e)
        for p in parts:
            for side in ("truth", "member", "ensmean"):
                e = K.eke(*vel[side][p])
                if e.ndim == 3:
                    e = e.mean(0)
                acc[p][side][gm] += e[gm]
                per_day[p][side][d] = float(e[gm].mean())
        cnt[gm] += 1
        step_t.append(_step_rate_fields({v: S.truth_anom(v, d)[None] for v in S.vars}, gm))
        step_m.append(_step_rate_fields({v: S.gen_anom(v, d, sub=True) for v in S.vars}, gm))
    if not cnt.any():
        print("[diag] eke: no pixels survive the gradient mask -- skipped")
        return res
    for p in parts:
        for side in ("truth", "member", "ensmean"):
            res[f"eke_{p}_{side}"] = float(np.nanmean(per_day[p][side]))
        res[f"eke_{p}_ratio"] = res[f"eke_{p}_member"] / res[f"eke_{p}_truth"]
        res[f"eke_{p}_ensmean_ratio"] = res[f"eke_{p}_ensmean"] / res[f"eke_{p}_truth"]
    res["step_rate_truth"] = float(np.mean(step_t))
    res["step_rate_member"] = float(np.mean(step_m))
    res["step_rate_ratio"] = res["step_rate_member"] / max(res["step_rate_truth"], 1e-12)

    main = "total" if has_ag else "geostrophic"
    with np.errstate(divide="ignore", invalid="ignore"):
        maps = {side: np.where(cnt > 0, acc[main][side] / cnt, np.nan)
                for side in ("truth", "member", "ensmean")}
    allv = np.concatenate([m_[np.isfinite(m_)] for m_ in maps.values()])
    vmax = float(np.percentile(allv, 99)) if allv.size else 1.0
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, side, ttl in zip(axes[0][:3], ("truth", "member", "ensmean"),
                             ("truth", "members (mean over members and steps)", "ensemble mean")):
        im = ax.imshow(maps[side], origin="lower", cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"mean {main} EKE, {ttl}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.03).set_label("m2/s2", fontsize=8)
    ax = axes[0][3]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(maps["member"] / maps["truth"])
    lim = float(np.clip(np.nanpercentile(np.abs(lr), 99), 0.3, 2.0)) if np.isfinite(lr).any() else 1.0
    im = ax.imshow(lr, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_title("log10(members / truth)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax = axes[1][0]
    lat1d = np.asarray(S.lat[0]).mean(axis=1)
    varies = float(lat1d.max() - lat1d.min()) > 0.01 and not S.is_patch
    yy = lat1d if varies else np.arange(S.ny)
    rows = np.isfinite(maps["truth"]).any(axis=1)
    for side, col in (("truth", "k"), ("member", "crimson"), ("ensmean", "tab:blue")):
        ax.plot(np.nanmean(maps[side][rows], 1), yy[rows], color=col, lw=1.5, label=side)
    ax.set_xlabel(f"zonal-mean {main} EKE [m2/s2]", fontsize=8)
    ax.set_ylabel("latitude [degN]" if varies else "grid row", fontsize=8)
    ax.set_title("meridional structure", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.4); ax.tick_params(labelsize=7)
    ax = axes[1][1]
    for side, col in (("truth", "k"), ("member", "crimson"), ("ensmean", "tab:blue")):
        ax.plot(S.days, per_day[main][side], ".-", color=col, lw=1.2, ms=4, label=side)
    ax.set_xlabel(S.day_label(), fontsize=8); ax.set_ylabel(f"domain-mean {main} EKE [m2/s2]", fontsize=8)
    ax.set_title("per target step", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.4); ax.tick_params(labelsize=7)
    ax = axes[1][2]
    xs = np.arange(len(parts))
    for off, side, col in ((-0.27, "truth", "0.25"), (0.0, "member", "crimson"), (0.27, "ensmean", "tab:blue")):
        ax.bar(xs + off, [res[f"eke_{p}_{side}"] for p in parts], 0.25, color=col, label=side)
    for i_, p in enumerate(parts):
        ax.text(i_, max(res[f"eke_{p}_truth"], res[f"eke_{p}_member"]) * 1.02,
                f"members x{res[f'eke_{p}_ratio']:.2f}\nmean x{res[f'eke_{p}_ensmean_ratio']:.2f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(parts, fontsize=8)
    ax.set_ylabel("domain-mean EKE [m2/s2]", fontsize=8); ax.set_title("energy budget", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, axis="y", lw=0.4); ax.tick_params(labelsize=7)
    ax = axes[1][3]
    ax.axis("off")
    ax.text(0.0, 0.5,
            "Two ratios, two questions.\n\nmembers / truth: REALISM. Each member\nshould carry the "
            "truth's energy (1.0).\n\nens mean / truth: PREDICTABILITY. The\nmean averages out what "
            "the conditioning\ndoes not determine, so this is < 1 by\nconstruction; how far below "
            "says how\nmuch of the eddy field the observations\npin down.\n\n"
            f"one-cell steps in all channels:\n{res['step_rate_member'] * 1e4:.1f} per 10k px members, "
            f"{res['step_rate_truth'] * 1e4:.1f} truth\n(x{res['step_rate_ratio']:.1f}; drawn coastlines "
            "if >> 1)",
            fontsize=8.5, va="center")
    fold = ("\n" + S.fold_note()) if S.fold_note() else ""
    fig.suptitle(f"Eddy kinetic energy from the ssh anomaly"
                 f"{' + uag/vag channels' if has_ag else ''}  --  " + S.title() + "\n" + S.mask_note() + fold,
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# figc8: cross-channel
# ---------------------------------------------------------------------------
def _pooled_channels(S: CondSamples, which: str, n_max: int, rng) -> np.ndarray:
    """``(Ct, n)`` pixel values across channels, pooled over steps (and members)."""
    per = max(n_max // S.D, 500)
    cols = []
    for d in range(S.D):
        m = S.stat_mask[d]
        idx = np.flatnonzero(m.ravel())
        if idx.size == 0:
            continue
        if which == "truth":
            arr = np.stack([S.truth_anom(v, d).ravel()[idx] for v in S.vars])
        elif which == "member":
            arr = np.concatenate([S.gen_anom(v, d, sub=True).reshape(-1, S.ny * S.nx)[:, idx][None]
                                  for v in S.vars], 0)                    # (Ct, Ksub, n)
            arr = arr.reshape(S.Ct, -1)
        else:
            arr = np.stack([(S.gen_anom(v, d).mean(0) - S.truth_anom(v, d)).ravel()[idx] for v in S.vars])
        if arr.shape[1] > per:
            arr = arr[:, rng.choice(arr.shape[1], per, replace=False)]
        cols.append(arr)
    return np.concatenate(cols, 1)


def fig_crosschannel(S: CondSamples, out: str, rng) -> dict:
    ct = np.corrcoef(_pooled_channels(S, "truth", 200_000, rng))
    cg = np.corrcoef(_pooled_channels(S, "member", 200_000, rng))
    ce = np.corrcoef(_pooled_channels(S, "error", 200_000, rng))
    if S.Ct == 1:
        ct, cg, ce = (np.atleast_2d(x) for x in (ct, cg, ce))
    diff = cg - ct
    res = {"corr_frobenius_error": float(np.linalg.norm(diff)),
           "corr_max_abs_error": float(np.abs(diff).max()),
           "err_corr_max_abs": float(np.abs(ce - np.eye(S.Ct)).max()) if S.Ct > 1 else 0.0}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for ax, (mat, ttl, cmap, lim) in zip(axes[0], ((ct, "truth", "RdBu_r", 1.0),
                                                   (cg, "members", "RdBu_r", 1.0),
                                                   (diff, "members - truth", "PuOr_r", 0.5),
                                                   (ce, "ensemble-mean ERRORS", "RdBu_r", 1.0))):
        im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(S.Ct)); ax.set_xticklabels(S.vars, rotation=45, fontsize=8)
        ax.set_yticks(range(S.Ct)); ax.set_yticklabels(S.vars, fontsize=8)
        ax.set_title(f"channel correlation, {ttl}", fontsize=9)
        for a in range(S.Ct):
            for b in range(S.Ct):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(mat[a, b]) > 0.55 * lim else "black")
        plt.colorbar(im, ax=ax, fraction=0.045)
    if S.has("ssh", "uag", "vag"):
        pts = {"truth": [], "member": []}
        for d in range(S.D):
            gm = S.grad_mask[d]
            if not gm.any():
                continue
            ug, _ = K.geostrophic_uv(S.truth_anom("ssh", d), S.dx_m, S.lat[d])
            pts["truth"].append(np.stack([ug[gm], S.truth_anom("uag", d)[gm]]))
            for j in S.member_idx[:2]:
                ugj, _ = K.geostrophic_uv(S.gen_anom("ssh", d)[j], S.dx_m, S.lat[d])
                pts["member"].append(np.stack([ugj[gm], S.gen_anom("uag", d)[j][gm]]))
        for ax, side in zip(axes[1][:2], ("truth", "member")):
            xy = np.concatenate(pts[side], 1)
            keep = np.isfinite(xy).all(0)
            x, y = xy[0][keep], xy[1][keep]
            if x.size > 400_000:
                sel = rng.choice(x.size, 400_000, replace=False)
                x, y = x[sel], y[sel]
            lim_x = float(np.percentile(np.abs(x), 99.5)); lim_y = float(np.percentile(np.abs(y), 99.5))
            ax.hist2d(x, y, bins=120, range=[[-lim_x, lim_x], [-lim_y, lim_y]], cmap="magma",
                      norm=matplotlib.colors.LogNorm())
            r_ = float(np.corrcoef(x, y)[0, 1])
            res[f"r_ug_uag_{side}"] = r_
            ax.set_xlabel("geostrophic u from ssh [m/s]", fontsize=8); ax.set_ylabel("uag [m/s]", fontsize=8)
            ax.set_title(f"{side}: r = {r_:+.3f}", fontsize=9); ax.tick_params(labelsize=7)
        for ax in axes[1][2:]:
            ax.axis("off")
    else:
        for ax in axes[1]:
            ax.axis("off")
    axes[1][3].text(0.0, 0.5,
                    "Left three: does the ensemble carry the\nsame channel dependence as the truth?\n"
                    f"Frobenius |members - truth| = {res['corr_frobenius_error']:.3f}\n"
                    f"largest single error = {res['corr_max_abs_error']:.3f}\n\n"
                    "Fourth: are the ensemble-mean ERRORS\ncorrelated across channels? Strong\n"
                    "off-diagonals mean one failure shows up\nin every field at once (e.g. a misplaced\n"
                    f"front). max |off-diag| = {res['err_corr_max_abs']:.3f}",
                    fontsize=8.5, va="center")
    fig.suptitle("Cross-channel structure  --  " + S.title(), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# figc9: conditioning fidelity
# ---------------------------------------------------------------------------
def fig_fidelity(S: CondSamples, P: dict, out: str):
    fam = S.family()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    if fam == "obs":
        mapped = [v for v in S.vars if S.baseline_map.get(v)]
        ax = axes[0]
        if mapped:
            xs = np.arange(len(mapped))
            om = [P[v].scalars["obsfit_rmse_member"] / S.std[S.idx(v)] for v in mapped]
            oe = [P[v].scalars["obsfit_rmse_ens"] / S.std[S.idx(v)] for v in mapped]
            ot = [P[v].scalars["obsfit_rmse_truth"] / S.std[S.idx(v)] for v in mapped]
            ax.bar(xs - 0.27, om, 0.25, color="crimson", label="members vs obs")
            ax.bar(xs, oe, 0.25, color="tab:blue", label="ens mean vs obs")
            ax.bar(xs + 0.27, ot, 0.25, color="0.4", label="TRUTH vs obs (the floor)")
            ax.set_xticks(xs); ax.set_xticklabels([f"{v}\n<- {S.baseline_map[v]}" for v in mapped], fontsize=8)
            ax.set_ylabel("RMS misfit at observed pixels [sigma]", fontsize=8)
            ax.set_title("fit to the observation where observed\n(beating the truth's own misfit = "
                         "overfitting a degraded product)", fontsize=8.5)
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.axis("off"); ax.text(0.1, 0.5, "no target has a mapped\nobservation channel", fontsize=9)
        ax = axes[1]
        for v in S.vars:
            p = P[v]
            ok = np.isfinite(p.coverage) & np.isfinite(p.spread)
            sc_ = ax.scatter(p.coverage[ok], p.spread[ok] / S.std[S.idx(v)], s=22, label=v,
                             c=p.obs_age_s[ok] / 3600.0 if np.isfinite(p.obs_age_s[ok]).any() else None,
                             cmap="viridis", edgecolor="k", linewidth=0.3)
        ax.set_xlabel("observed fraction of ocean pixels (newest window step)", fontsize=8)
        ax.set_ylabel("spread [sigma]", fontsize=8)
        ax.set_title("spread vs observation coverage (colour: obs age, h)", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.4)
        try:
            plt.colorbar(sc_, ax=ax, fraction=0.04).set_label("obs age [h]", fontsize=7)
        except Exception:                                   # noqa: BLE001
            pass
        ax = axes[2]
        for v in S.vars:
            p = P[v]
            s = S.std[S.idx(v)]
            ok = np.isfinite(p.rmse_obs_px) & np.isfinite(p.rmse_gap_px)
            ax.scatter(p.rmse_obs_px[ok] / s, p.rmse_gap_px[ok] / s, s=22, label=v, edgecolor="k",
                       linewidth=0.3)
        lim = ax.get_xlim()[1]
        ax.plot([0, lim], [0, lim], "k-", lw=1)
        ax.set_xlabel("ens-mean RMSE at OBSERVED pixels [sigma]", fontsize=8)
        ax.set_ylabel("ens-mean RMSE at GAP pixels [sigma]", fontsize=8)
        ax.set_title("does the error grow where the obs are missing?", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.4)
        ax = axes[3]
        any_age = False
        for v in S.vars:
            p = P[v]
            age = p.obs_age_s / 3600.0
            ok = np.isfinite(age) & np.isfinite(p.spread)
            if ok.sum() > 2 and np.std(age[ok]) > 0:
                any_age = True
                o = np.argsort(age[ok])
                ax.plot(age[ok][o], (p.spread[ok] / S.std[S.idx(v)])[o], ".-", ms=4, lw=0.8, label=f"{v} spread")
                ax.plot(age[ok][o], (p.rmse_ens[ok] / S.std[S.idx(v)])[o], "x--", ms=4, lw=0.8,
                        label=f"{v} RMSE")
        if any_age:
            ax.set_xlabel("age of the newest observation [h]", fontsize=8)
            ax.set_ylabel("[sigma]", fontsize=8); ax.legend(fontsize=6.5, frameon=False, ncol=2)
            ax.set_title("spread and error vs observation age", fontsize=8.5)
        else:
            ax.axis("off")
            ax.text(0.05, 0.5, "observation age does not vary\n(base-cadence observations)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
    else:
        ax = axes[0]
        for v in S.vars:
            p = P[v]
            s = S.std[S.idx(v)]
            o = np.argsort(S.doy)
            ax.plot(S.doy[o], p.dmean_truth[o] / s, "k.-", lw=0.8, ms=4)
            line, = ax.plot(S.doy[o], p.dmean_ens[o] / s, ".-", lw=1.2, ms=4,
                            label=f"{v} (corr {p.scalars['dmean_corr']:+.2f})")
            ax.fill_between(S.doy[o], (p.dmean_ens - p.spread)[o] / s, (p.dmean_ens + p.spread)[o] / s,
                            color=line.get_color(), alpha=0.15, lw=0)
        ax.set_xlabel("day of year of the target step", fontsize=8)
        ax.set_ylabel("domain-mean anomaly [sigma]", fontsize=8)
        ax.set_title("seasonal consistency: truth (black) vs ensemble mean +- spread", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.4)
        ax = axes[1]
        for v in S.vars:
            p = P[v]
            o = np.argsort(S.doy)
            with np.errstate(divide="ignore", invalid="ignore"):
                ax.plot(S.doy[o], (p.dstd_member / p.dstd_truth)[o], ".-", lw=1.0, ms=4, label=v)
        ax.axhline(1.0, color="k", lw=1)
        ax.set_xlabel("day of year", fontsize=8); ax.set_ylabel("domain std member / truth", fontsize=8)
        ax.set_title("does the variability follow the season?", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.4)
        ax = axes[2]
        if S.land_any:
            for v in S.vars:
                land_vals, ocean_vals = [], []
                for d in range(S.D):
                    g = np.abs(S.gen_norm(v, d, sub=True))
                    land = ~S.mask_day[d]
                    if land.any():
                        land_vals.append(g[:, land].ravel())
                    ocean_vals.append(g[:, S.stat_mask[d]].ravel()[:20000])
                if land_vals:
                    lv = np.concatenate(land_vals); ov = np.concatenate(ocean_vals)
                    bins = np.linspace(0, max(np.percentile(np.concatenate([lv, ov]), 99), 1e-3), 60)
                    ax.hist(ov, bins=bins, histtype="step", lw=1.2, density=True, label=f"{v} ocean")
                    ax.hist(lv, bins=bins, histtype="step", lw=1.2, ls="--", density=True,
                            color=ax.patches[-1].get_edgecolor(), label=f"{v} land")
            ax.set_xlabel("|generated value| [sigma]", fontsize=8); ax.set_ylabel("density", fontsize=8)
            ax.set_title("what does the model draw on land? (loss never constrained it)", fontsize=8.5)
            ax.legend(fontsize=6.5, frameon=False, ncol=2); ax.set_yscale("log")
        else:
            ax.axis("off"); ax.text(0.05, 0.5, "no land in these samples", fontsize=9)
        ax = axes[3]
        for v in S.vars:
            o = np.argsort(S.doy)
            ax.plot(S.doy[o], P[v].rank_tv_day[o], ".-", lw=0.9, ms=4, label=v)
        ax.axhline(0.2, color="crimson", lw=0.8, ls=":")
        ax.set_xlabel("day of year", fontsize=8); ax.set_ylabel("per-step rank-histogram TV", fontsize=8)
        ax.set_title("is calibration seasonal?", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.4); ax.set_ylim(bottom=0)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Conditioning fidelity ({'observation-conditioned' if fam == 'obs' else 'mask + day-of-year conditioned'})"
                 f"  --  " + S.title(), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
PAIRED_COLS = ["rmse_ens", "mae_ens", "bias_ens", "rmse_member_mean", "rmse_member_min",
               "rmse_member_max", "rmse_member_over_ens", "rmse_member_over_ens_target", "crps",
               "spread", "spread_skill_ratio", "rank_tv", "rank_tails_ratio", "rank_slope",
               "frac_px_calibrated", "rmse_border_over_interior", "rmse_ens_sigma", "crps_sigma",
               "spread_sigma"]
BASE_COLS = ["baseline_var", "baseline_kind", "rmse_obs", "mae_obs", "rmse_ens_on_obs_px",
             "rmse_clim", "mae_clim", "ss_rmse_obs", "ss_rmse_clim", "crps_over_mae_obs",
             "crps_climens", "crpss_climens"]
FID_COLS = ["obsfit_rmse_member", "obsfit_rmse_ens", "obsfit_rmse_truth", "rmse_obs_px",
            "rmse_gap_px", "coverage_mean", "spread_vs_coverage_slope", "spread_vs_age_slope",
            "dmean_corr", "dmean_rmse_ens", "dmean_rmse_const", "dstd_ratio_mean",
            "land_abs_gen_sigma", "land_quiet_frac"]
PDF_COLS = ["mean_member", "std_member", "skew_member", "kurt_member", "mean_truth", "std_truth",
            "skew_truth", "kurt_truth", "std_ratio", "std_ratio_ensmean"]


def spectral_cols() -> list[str]:
    cols = []
    for name, _lo, _hi in BANDS:
        b = _band_key(name)
        cols += [f"psd_ratio_member_{b}", f"psd_ratio_ensmean_{b}", f"band_nrmse_ens_{b}",
                 f"band_nrmse_member_{b}", f"band_nrmse_obs_{b}", f"band_spread_skill_{b}"]
    return cols + ["coh50_km", "logdist_member", "logdist_ensmean", "n_clean_days"]


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.6g}"
    if isinstance(x, (np.floating,)):
        return _fmt(float(x))
    return str(x)


def _f(x, nd=3) -> str:
    """Report formatting: ``nd`` significant digits, or a plain integer when
    ``nd == 0`` (percentages, wavelengths in km)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.0f}" if nd == 0 else f"{x:.{nd}g}"


def write_report(S: CondSamples, P: dict, SP: dict, pdf: dict, eke: dict, cross: dict,
                 out_dir: str, band_crps: dict | None = None):
    domain_cols = sorted(eke) + sorted(cross)
    domain = dict(eke, **cross)
    cols = (["var", "units", "is_log", "D", "K", "size", "n_px_mean"] + PAIRED_COLS + BASE_COLS
            + spectral_cols() + PDF_COLS + FID_COLS + domain_cols)
    if band_crps:
        cols += sorted(next(iter(band_crps.values())).keys())
    rows = []
    for v in S.vars:
        i = S.idx(v)
        r = {"var": v, "units": _unit(v), "is_log": bool(S.is_log[i]), "D": S.D, "K": S.K,
             "size": S.size, "n_px_mean": float(P[v].n_px.mean())}
        r.update(P[v].scalars)
        r.update(SP[v].scalars if SP.get(v) is not None else {c: np.nan for c in spectral_cols()})
        r.update(pdf.get(v, {}))
        r.update(domain)
        if band_crps:
            r.update(band_crps.get(v, {}))
        rows.append(r)
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(c, np.nan)) for c in cols) + "\n")

    # per-day table
    dcols = ["day", "doy", "var", "n_px", "rmse_ens", "mae_ens", "crps", "spread", "rank_tv",
             "rmse_obs", "rmse_clim", "rmse_obs_px", "rmse_gap_px", "coverage", "obs_age_s",
             "dmean_truth", "dmean_ens", "dstd_truth", "dstd_member"]
    with open(os.path.join(out_dir, "summary_days.csv"), "w") as f:
        f.write(",".join(dcols) + "\n")
        for v in S.vars:
            p = P[v]
            for d in range(S.D):
                vals = [int(S.days[d]), float(S.doy[d]), v, int(p.n_px[d]), p.rmse_ens[d],
                        p.mae_ens[d], p.crps[d], p.spread[d], p.rank_tv_day[d], p.rmse_obs[d],
                        p.rmse_clim[d], p.rmse_obs_px[d], p.rmse_gap_px[d], p.coverage[d],
                        p.obs_age_s[d], p.dmean_truth[d], p.dmean_ens[d], p.dstd_truth[d],
                        p.dstd_member[d]]
                f.write(",".join(_fmt(float(x)) if isinstance(x, (float, np.floating)) else str(x)
                                 for x in vals) + "\n")

    # spectra for the trajectory script
    spec_payload = {"dx_km": np.float64(S.dx_km), "bands": json.dumps(BANDS), "vars": json.dumps(S.vars)}
    for v in S.vars:
        sp = SP.get(v)
        if sp is None:
            continue
        spec_payload["k"] = sp.k
        spec_payload[f"psd_truth_{v}"] = np.median(sp.psd_truth, 0)
        spec_payload[f"psd_member_{v}"] = np.median(sp.psd_member, 0)
        spec_payload[f"psd_ensmean_{v}"] = np.median(sp.psd_ensmean, 0)
        spec_payload[f"psd_err_ens_{v}"] = np.median(sp.psd_err_ens, 0)
        spec_payload[f"psd_spread_{v}"] = np.median(sp.psd_spread, 0)
        spec_payload[f"gamma2_{v}"] = sp.gamma2
        if sp.psd_obs is not None:
            spec_payload[f"psd_obs_{v}"] = np.median(sp.psd_obs, 0)
    if "k" in spec_payload:
        np.savez(os.path.join(out_dir, "spectra.npz"), **spec_payload)

    # -- report.md ---------------------------------------------------------
    m = S.meta
    s = m.get("sampler", {})
    fam = S.family()
    L = [f"# Conditional ensemble diagnostics -- step {m.get('step', 0):,}", "",
         f"- checkpoint: `{m.get('ckpt')}`",
         f"- val_loss at this checkpoint: {m.get('val_loss')}",
         f"- geometry: {S.size} ({S.ny}x{S.nx}), D={S.D} target steps x K={S.K} members, "
         f"split `{m.get('split')}`, steps {S.days.min()}..{S.days.max()} ({S.base_cadence})",
         f"- sampler: {s.get('num_steps')} Heun steps, s_churn={s.get('s_churn')}, "
         f"s_noise={s.get('s_noise')}, s_tmin={s.get('s_tmin')}, s_tmax={s.get('s_tmax')}; "
         f"weights `{m.get('weights')}`",
         f"- conditioning: cond_vars={S.cond_vars}, k_days={S.k_days}, extras={S.extras} "
         f"(family: **{fam}**), stored as `{S.store_cond}`",
         f"- baselines: {S.baseline_map} (kinds {S.baseline_kind}); "
         f"bias obs-truth (physical) = {[_f(b) for b in S.baseline_bias]}",
         f"- latitude: {m.get('lat_note')}; crops: {m.get('lattice_note')}",
         f"- {S.mask_note()}", ""]
    if S.fold_note():
        L += [f"- {S.fold_note()}", ""]
    L += ["Skill numbers are in PHYSICAL anomaly units (norm x std); log10 variables stay in "
          "log space. `[sigma]` columns divide by the training std.", "",
          "## Paired skill (ensemble mean vs truth, ocean pixels)", "",
          "| var | RMSE ens | RMSE member (mean) | RMSE obs | RMSE clim | SS vs obs | SS vs clim | "
          "CRPS | MAE obs | CRPS/MAE_obs | CRPSS vs clim-ens |", "|---|" + "---|" * 10]
    for v in S.vars:
        r = P[v].scalars
        L.append(f"| {v} [{_unit(v)}] | {_f(r['rmse_ens'])} | {_f(r['rmse_member_mean'])} | "
                 f"{_f(r['rmse_obs'])} | {_f(r['rmse_clim'])} | {_f(r['ss_rmse_obs'], 2)} | "
                 f"{_f(r['ss_rmse_clim'], 2)} | {_f(r['crps'])} | {_f(r['mae_obs'])} | "
                 f"{_f(r['crps_over_mae_obs'], 2)} | {_f(r['crpss_climens'], 2)} |")
    L += ["", "## Calibration", "",
          "| var | spread | spread/skill (Fortin) | rank TV | tails ratio | rank slope | "
          "member/mean RMSE | target | px within x1.5 | border/interior RMSE |", "|---|" + "---|" * 9]
    for v in S.vars:
        r = P[v].scalars
        L.append(f"| {v} | {_f(r['spread'])} | {_f(r['spread_skill_ratio'], 2)} | {_f(r['rank_tv'], 3)} | "
                 f"{_f(r['rank_tails_ratio'], 2)} | {_f(r['rank_slope'], 2)} | "
                 f"{_f(r['rmse_member_over_ens'], 2)} | {_f(r['rmse_member_over_ens_target'], 2)} | "
                 f"{_f(100 * r['frac_px_calibrated'], 0)} % | {_f(r['rmse_border_over_interior'], 2)} |")
    L += ["", "## Scale dependence (spectral; band metrics by Parseval, no spatial filter)", "",
          "| var | " + " | ".join(f"NRMSE ens {_band_key(n)} | NRMSE member {_band_key(n)} | "
                                  f"NRMSE obs {_band_key(n)} | spread/skill {_band_key(n)} | "
                                  f"PSD ratio member {_band_key(n)}" for n, _l, _h in BANDS)
          + " | coh50 [km] | logdist member | logdist mean |",
          "|---|" + "---|" * (5 * len(BANDS) + 3)]
    for v in S.vars:
        sp = SP.get(v)
        if sp is None:
            L.append(f"| {v} | " + " | ".join(["n/a"] * (5 * len(BANDS) + 3)) + " |")
            continue
        r = sp.scalars
        cells = []
        for n, _l, _h in BANDS:
            b = _band_key(n)
            cells += [_f(r[f"band_nrmse_ens_{b}"], 2), _f(r[f"band_nrmse_member_{b}"], 2),
                      _f(r[f"band_nrmse_obs_{b}"], 2), _f(r[f"band_spread_skill_{b}"], 2),
                      _f(r[f"psd_ratio_member_{b}"], 2)]
        cells += [_f(r["coh50_km"], 0), _f(r["logdist_member"], 3), _f(r["logdist_ensmean"], 3)]
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L += ["", "## Realism of the members", "",
          "| var | std member/truth | std mean/truth | skew member / truth | kurt member / truth |",
          "|---|---|---|---|---|"]
    for v in S.vars:
        p = pdf.get(v, {})
        L.append(f"| {v} | {_f(p.get('std_ratio'), 2)} | {_f(p.get('std_ratio_ensmean'), 2)} | "
                 f"{_f(p.get('skew_member'), 2)} / {_f(p.get('skew_truth'), 2)} | "
                 f"{_f(p.get('kurt_member'), 2)} / {_f(p.get('kurt_truth'), 2)} |")
    if eke:
        L += ["", "| EKE component | truth | members | ens mean | members/truth | mean/truth |",
              "|---|---|---|---|---|---|"]
        for part in ("geostrophic", "ageostrophic", "total"):
            if f"eke_{part}_truth" in eke:
                L.append(f"| {part} [m2/s2] | {_f(eke[f'eke_{part}_truth'], 4)} | "
                         f"{_f(eke[f'eke_{part}_member'], 4)} | {_f(eke[f'eke_{part}_ensmean'], 4)} | "
                         f"{_f(eke[f'eke_{part}_ratio'], 2)} | {_f(eke[f'eke_{part}_ensmean_ratio'], 2)} |")
        L += ["", f"- one-cell steps in every channel at once: {eke['step_rate_member'] * 1e4:.1f} per 10k px "
                  f"members, {eke['step_rate_truth'] * 1e4:.1f} truth (x{eke['step_rate_ratio']:.1f})"]
    L += ["", f"- channel-correlation Frobenius error (members vs truth): **{cross['corr_frobenius_error']:.3f}**; "
              f"largest single error {cross['corr_max_abs_error']:.3f}; largest cross-channel ERROR "
              f"correlation {cross['err_corr_max_abs']:.3f}"]
    L += ["", "## Conditioning fidelity", ""]
    if fam == "obs":
        L += ["| var | obs channel | coverage | misfit member vs obs | misfit mean vs obs | misfit TRUTH vs obs | "
              "RMSE at obs px | RMSE at gap px | d spread / d coverage | d spread / d age [/h] |",
              "|---|" + "---|" * 9]
        for v in S.vars:
            r = P[v].scalars
            L.append(f"| {v} | {r['baseline_var'] or '-'} | {_f(r['coverage_mean'], 2)} | "
                     f"{_f(r['obsfit_rmse_member'])} | {_f(r['obsfit_rmse_ens'])} | {_f(r['obsfit_rmse_truth'])} | "
                     f"{_f(r['rmse_obs_px'])} | {_f(r['rmse_gap_px'])} | "
                     f"{_f(r['spread_vs_coverage_slope'], 2)} | {_f(r['spread_vs_age_slope'], 3)} |")
    else:
        L += ["| var | corr(domain mean: ens vs truth) | RMSE of domain mean | RMSE of a constant | "
              "domain std member/truth | mean |gen| on land [sigma] | quiet land frac |",
              "|---|" + "---|" * 6]
        for v in S.vars:
            r = P[v].scalars
            L.append(f"| {v} | {_f(r['dmean_corr'], 2)} | {_f(r['dmean_rmse_ens'])} | "
                     f"{_f(r['dmean_rmse_const'])} | {_f(r['dstd_ratio_mean'], 2)} | "
                     f"{_f(r['land_abs_gen_sigma'], 2)} | {_f(r['land_quiet_frac'], 2)} |")
    if band_crps:
        L += ["", "## Band-passed pointwise metrics (opt-in, difference-of-Gaussians filter)", "",
              "| var | " + " | ".join(f"CRPS {_band_key(n)} | rank TV {_band_key(n)} | spread/skill {_band_key(n)}"
                                      for n, _l, _h in BANDS) + " |",
              "|---|" + "---|" * (3 * len(BANDS))]
        for v in S.vars:
            r = band_crps.get(v, {})
            cells = []
            for n, _l, _h in BANDS:
                b = _band_key(n)
                cells += [_f(r.get(f"bp_crps_{b}")), _f(r.get(f"bp_rank_tv_{b}"), 3),
                          _f(r.get(f"bp_spread_skill_{b}"), 2)]
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        L += ["", "The filter's half-power points sit exactly on the band edges; it leaks "
                  "~0.5 amplitude at the edge and < 0.05 a factor 4 outside, by construction "
                  "(see `kernels_cond.dog_transfer`)."]
    L += ["", "## How to read this", "",
          "- **Skill score > 0** means the ensemble mean beats that baseline; the obs baseline is scored "
          "only where the observation exists, and the ensemble mean is scored on those same pixels for "
          "the comparison.",
          "- **CRPS <= MAE of a deterministic baseline** is the probabilistic equivalent of beating it.",
          "- **Spread/skill** is Fortin-corrected, so 1.0 is calibrated for any K (the old 0.5-1.5 gate "
          "was an advisory band). **Rank TV** <= 0.2 was the old gate; the **tails ratio** gives the "
          "sign (> 1 under-dispersed, < 1 over-dispersed) and the **slope** the bias direction.",
          f"- **member/mean RMSE** should sit near sqrt(2K/(K+1)) = {KC.member_over_ensmean_target(S.K):.2f}: "
          "well above means the members carry noise the mean averages out; well below means they are "
          "near-identical.",
          "- **coh50** is the smallest scale at which the ensemble mean is coherent with the truth. "
          "**PSD ratio of the ensemble mean** falling with wavelength is EXPECTED (unpredictable scales "
          "average out); read the **member PSD ratio** for realism and **band spread/skill** for "
          "whether the missing small-scale power reappears as spread.",
          "- **NRMSE by band** = sqrt(band error variance / band truth variance) from the same "
          "Hann-windowed tiles as the spectra (Parseval), so no brick-wall filter leaks.",
          ("- **mask + day-of-year models**: expect SS vs clim ~ 0. The question is whether the ensemble "
           "is a CALIBRATED climatological sampler for that day of year (rank TV, spread/skill) and "
           "whether its domain mean follows the season (corr of domain means). Under `norm_mode: "
           "anomaly` the seasonal cycle is already removed and that correlation tests residual "
           "seasonality only." if fam == "maskdoy" else
           "- **misfit TRUTH vs obs is the floor**: members that fit the observation better than the "
           "truth does are overfitting a degraded product."),
          "- The climatological-ensemble CRPS uses the other D-1 truth steps as members; with closely "
          "spaced steps it is too narrow and CRPSS vs it is optimistic -- advisory.",
          "- CHL-type variables are scored in log10 space.",
          "- Rank-histogram TV is computed over spatially correlated pixels, so its sampling error is far "
          "larger than the pixel count suggests; figc3's inset shows the per-step spread of TV.",
          "", "## What was masked, and why", "",
          f"- {GRAD_ERODE_PX} px Euclidean land erosion before any spatial derivative (EKE, "
          "geostrophy), per sample, from the land the DATA reports (all-channel exact zeros in the truth).",
          ("- " + (f"{PATCH_BORDER_PX} px frame border excluded from the distributional figures (both "
                   "sides). Paired skill is reported on the interior, and `rmse_border_over_interior` "
                   "measures the zero-padding halo directly instead of hiding it."
                   if S.border_cut else
                   "no frame-border cut (full geometry, or disabled with --no-border-cut).")),
          "- Spectra use whole rectangles: at full frame, all-ocean tiles; in patch geometry, only "
          f"land-free samples on BOTH sides ({SP[S.vars[0]].n_days if SP.get(S.vars[0]) else 0}/{S.D} here)."]
    rep = os.path.join(out_dir, "report.md")
    with open(rep, "w") as f:
        f.write("\n".join(L) + "\n")
    return csv_path, rep


# ---------------------------------------------------------------------------
# opt-in: band-passed pointwise metrics
# ---------------------------------------------------------------------------
def band_crps_metrics(S: CondSamples, rng) -> dict:
    out = {}
    for v in S.vars:
        r = {}
        for name, lo, hi in BANDS:
            b = _band_key(name)
            crps_acc, n_acc, counts, sv, se = 0.0, 0, np.zeros(S.K + 1, np.int64), 0.0, 0.0
            for d in range(S.D):
                m = S.stat_mask[d]
                if not m.any():
                    continue
                g = S.gen_anom(v, d).astype(np.float64)
                t = S.truth_anom(v, d).astype(np.float64)
                gf, valid = KC.dog_bandpass(g, S.mask_day[d], S.dx_km, lo, hi)
                tf, _ = KC.dog_bandpass(t, S.mask_day[d], S.dx_km, lo, hi)
                vm = valid & m
                if not vm.any():
                    continue
                crps_acc += float(KC.crps_ensemble(gf, tf)[vm].sum())
                n_acc += int(vm.sum())
                counts += np.bincount(KC.rank_of_truth(gf, tf, rng)[vm], minlength=S.K + 1)
                sv += float(gf.var(0, ddof=1)[vm].sum())
                se += float(((gf.mean(0) - tf) ** 2)[vm].sum())
            r[f"bp_crps_{b}"] = crps_acc / n_acc if n_acc else np.nan
            r[f"bp_rank_tv_{b}"] = KC.rank_hist_stats(counts)["tv"]
            r[f"bp_spread_skill_{b}"] = KC.spread_skill_ratio(sv / n_acc, se / n_acc, S.K) if n_acc else np.nan
        out[v] = r
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="samples_cond_<size>.npz from eval.gen_cond")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--members-subsample", type=int, default=None,
                    help="use only this many members for the distributional figures (speed knob)")
    ap.add_argument("--bands", default=None,
                    help='override the summarised bands, e.g. "mesoscale:40:150,submeso:10:40"')
    ap.add_argument("--day", type=int, default=None, help="sample index for the gallery (default median RMSE)")
    ap.add_argument("--band-crps", action="store_true", help="also compute DoG band-passed CRPS / rank / spread-skill")
    ap.add_argument("--no-eke", action="store_true")
    ap.add_argument("--no-border-cut", action="store_true",
                    help="score the whole patch, including the frame border, instead of the interior")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    global BANDS
    if args.bands:
        BANDS = []
        for item in args.bands.split(","):
            name, lo, hi = item.split(":")
            BANDS.append((f"{name} {lo}-{hi} km", float(lo), float(hi)))

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    S = CondSamples(args.samples, members_subsample=args.members_subsample, seed=args.seed,
                    border_cut=not args.no_border_cut)
    gb = S.gen.nbytes / 1e9
    print(f"[diag] {S.D} steps x {S.K} members x {S.Ct} vars at {S.ny}x{S.nx} ({S.size}), "
          f"gen {S.gen.dtype} {gb:.2f} GB, family {S.family()}, dx={S.dx_km:.3f} km", flush=True)
    p = lambda name: os.path.join(args.out, name)

    print("[diag] paired statistics ...", flush=True)
    P = {v: PairedStats.compute(S, v, rng) for v in S.vars}
    for v in S.vars:
        r = P[v].scalars
        print(f"  {v:>6}: rmse {r['rmse_ens']:.4g}  crps {r['crps']:.4g}  spread/skill "
              f"{r['spread_skill_ratio']:.2f}  rank TV {r['rank_tv']:.3f}  SS_clim {r['ss_rmse_clim']:+.2f}",
              flush=True)
    print("[diag] figc1 galleries ...", flush=True)
    fig_gallery(S, P, p("figc1_gallery.png"), "physical", args.day)
    fig_gallery(S, P, p("figc1_gallery_anom.png"), "anomaly", args.day)
    fig_days(S, P, p("figc1_days.png"))
    print("[diag] figc2 skill, figc3 rank, figc4 maps, figc9 fidelity ...", flush=True)
    fig_skill(S, P, p("figc2_skill.png"))
    fig_rankhist(S, P, p("figc3_rankhist.png"))
    fig_errmaps(S, P, p("figc4_errmaps.png"))
    fig_fidelity(S, P, p("figc9_fidelity.png"))
    print("[diag] spectra ...", flush=True)
    tile = K.psd_tile(S.mask_ocean, S.ny, S.nx)
    SP = {v: SpectralStats.compute(S, v, tile, tile // 2) for v in S.vars}
    if S.is_patch and int(S.clean_day.sum()) < 8:
        print(f"[diag] WARNING: only {int(S.clean_day.sum())} land-free samples for the spectra")
    fig_spectra(S, SP, p("figc5_spectra.png"))
    print("[diag] figc6 pdf ...", flush=True)
    pdf = fig_pdf(S, p("figc6_pdf.png"), rng)
    eke = {}
    if not args.no_eke:
        print("[diag] figc7 eke ...", flush=True)
        eke = fig_eke(S, p("figc7_eke.png"))
    print("[diag] figc8 crosschannel ...", flush=True)
    cross = fig_crosschannel(S, p("figc8_crosschannel.png"), rng)
    bc = None
    if args.band_crps:
        print("[diag] band-passed pointwise metrics ...", flush=True)
        bc = band_crps_metrics(S, rng)
    csv, rep = write_report(S, P, SP, pdf, eke, cross, args.out, bc)
    print(f"[diag] wrote {csv} and {rep}")
    print("\n" + open(rep).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
