"""Diagnostics specific to assimilation output: GenDA-style panels and skill
split by observed / unobserved pixels, against an unguided control.

    python -m sda.diagnostics_sda --samples <dir>/samples_sda_256.npz \
        [--control <dir>/samples_sda_256_free.npz] --out <dir>/figs_sda_256

Everything generic about a paired ensemble (spectra, rank histograms, PDFs,
EKE, cross-channel) is already in ``eval.diagnostics_cond``, which reads the
same file; run it alongside. This script adds what only the SDA layer knows:

* ``panel_day<i>.png`` -- GenDA's 8-row panel (``OSSE_inference.py:421-463``):
  Observed, L4, Truth, Prediction mean, Prediction std, RMSE, |Mean - Truth|,
  Member 1, one column per variable, in units of the prior's per-channel std.
* ``summary_sda.csv`` -- per variable ``rmse_ens, rmse_member, spread,
  spread_skill_ratio, crps, rank_tv`` on {all, obs, gap} pixels, ``coh50_km``
  (finest coherent scale of members vs truth), and, with ``--control``, the same
  for the control plus ``skill_vs_control = 1 - rmse_ens/rmse_ens_ctrl``.
* ``obsfit_sda.csv`` -- per observation term the RMS misfit ``A(x) - y`` of
  members, the ensemble mean and the truth, in normalised units and relative to
  the term's assumed error std. Members far below 1 are over-fitting the
  observation noise; far above 1 are under-guided.
* ``report_sda.md`` -- the numbers, echoed to stdout.

Units: all statistics in normalised (sigma) units so variables are comparable;
``obs`` pixels are those under a *pointwise* term of that variable (a blur term
constrains scales, not pixels). Land is taken from the data (exact zeros in
every channel), the ``PATCH_BORDER_PX`` rim is cut in patch geometry, as in
``eval/diagnostics.py`` (constants copied; importing that module is out of
bounds for this layer).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval import kernels as K  # noqa: E402
from eval import kernels_cond as KC  # noqa: E402
from sda import obs as O  # noqa: E402

PATCH_BORDER_PX = 16      # copied from eval/diagnostics.py:97 (UNet halo, measured +32% EKE)
PANEL_ROWS = ["Observed", "L4", "Truth", "Pred mean", "Pred std", "RMSE", "|mean-truth|", "Member 1"]
SUBSETS = ("all", "obs", "gap")


# ---------------------------------------------------------------------------
class SdaSamples:
    def __init__(self, path: str, seed: int = 0, border_cut: bool = True):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.vars = json.loads(str(d["vars"]))
        self.meta = json.loads(str(d["meta"]))
        self.terms = json.loads(str(d["terms"])) if "terms" in d.files else []
        self.gen = d["gen_norm"]
        self.truth = np.asarray(d["truth_norm"], np.float32)
        self.std = np.asarray(d["std"], np.float64)
        self.clim = np.asarray(d["clim_day"], np.float32)
        self.is_log = np.asarray(d["is_log"], bool)
        self.D, self.K, self.C, self.ny, self.nx = self.gen.shape
        mask = np.asarray(d["mask"]) > 0.5
        self.mask = np.broadcast_to(mask, (self.D, self.ny, self.nx)).copy() if mask.ndim == 2 else mask
        self.y_grid = np.asarray(d["y_grid"], np.float32) if "y_grid" in d.files else np.zeros((self.D, 0, self.ny, self.nx), np.float32)
        self.obs_mask = np.asarray(d["obs_mask"], bool) if "obs_mask" in d.files else np.zeros_like(self.y_grid, bool)
        self.days = np.asarray(d["days"], np.int64)
        self.time_unix = np.asarray(d["time_unix"], np.float64) if "time_unix" in d.files else None
        self.dx_km = float(d["dx_m"]) / 1000.0
        self.size = str(d["size"])
        self.pos = np.asarray(d["pos"], np.int64)
        self.is_patch = bool(self.pos.size)
        self.rng = np.random.default_rng(seed)
        self.data_land = np.all(self.truth == 0.0, axis=1)
        self.mask_day = self.mask & ~self.data_land
        self.interior = np.ones((self.ny, self.nx), bool)
        if border_cut and self.is_patch and 2 * PATCH_BORDER_PX < min(self.ny, self.nx):
            b = PATCH_BORDER_PX
            self.interior[:b] = self.interior[-b:] = False
            self.interior[:, :b] = self.interior[:, -b:] = False
        self.stat_mask = self.mask_day & self.interior
        self.clean_day = self.mask_day.reshape(self.D, -1).all(axis=1)

    def sda(self) -> dict:
        return self.meta.get("sda", {})

    def gen_var(self, ci: int, d: int) -> np.ndarray:
        return np.asarray(self.gen[d, :, ci], np.float32)

    def terms_of(self, var: str, kind: str | None = None):
        return [(i, t) for i, t in enumerate(self.terms)
                if t["var"] == var and (kind is None or t["kind"] == kind)]

    def obs_pixels(self, var: str, d: int) -> np.ndarray | None:
        pt = self.terms_of(var, "pointwise")
        if not pt:
            return None
        m = np.zeros((self.ny, self.nx), bool)
        for i, _ in pt:
            m |= self.obs_mask[d, i]
        return m

    def obs_grid(self, var: str, d: int, kind: str) -> np.ndarray:
        g = np.full((self.ny, self.nx), np.nan, np.float32)
        for i, _ in self.terms_of(var, kind):
            sel = self.obs_mask[d, i]
            g[sel] = self.y_grid[d, i][sel]
        return g

    def title(self) -> str:
        s = self.sda()
        return (f"{os.path.basename(os.path.dirname(os.path.abspath(self.path)))} step {self.meta.get('step')} "
                f"| {self.size} | D={self.D} K={self.K} | guidance {s.get('guidance')} "
                f"{s.get('sampler')} steps={s.get('steps')} corr={s.get('corrections')} "
                f"gamma x{s.get('gamma_scale')} | obs {s.get('obs_name')}")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _acc_new():
    return dict(n=0, se_ens=0.0, se_ens_dm=0.0, se_mem=0.0, var=0.0, crps=0.0, ranks=None)


def _acc_add(acc, x, y, sel, rng, K_):
    """x (K,H,W) members, y (H,W) truth, sel (H,W) bool."""
    if not sel.any():
        return
    xm = x[:, sel].astype(np.float64)
    ym = y[sel].astype(np.float64)
    mean, var = KC.ens_mean_var(xm)
    acc["n"] += int(sel.sum())
    acc["se_ens"] += float(((mean - ym) ** 2).sum())
    # demeaned over the selected pixels: skill on the structure once a uniform
    # offset (the unobservable tide mode in SSH, an SST diurnal bias) is removed
    acc["se_ens_dm"] += float((((mean - mean.mean()) - (ym - ym.mean())) ** 2).sum())
    acc["se_mem"] += float(((xm - ym) ** 2).mean(axis=0).sum())
    acc["var"] += float(var.sum())
    acc["crps"] += float(KC.crps_ensemble(xm, ym, fair=K_ >= 2).sum()) if K_ >= 2 else float(np.abs(xm - ym).sum())
    r = KC.rank_of_truth(xm, ym, rng)
    cnt = np.bincount(r, minlength=K_ + 1)
    acc["ranks"] = cnt if acc["ranks"] is None else acc["ranks"] + cnt


def _acc_final(acc, K_):
    n = acc["n"]
    if n == 0:
        return dict(rmse_ens=np.nan, rmse_ens_dm=np.nan, rmse_member=np.nan, spread=np.nan,
                    spread_skill_ratio=np.nan, crps=np.nan, rank_tv=np.nan, n_px=0)
    mse_ens = acc["se_ens"] / n
    mean_var = acc["var"] / n
    return dict(rmse_ens=np.sqrt(mse_ens), rmse_ens_dm=np.sqrt(acc["se_ens_dm"] / n),
                rmse_member=np.sqrt(acc["se_mem"] / n),
                spread=np.sqrt(mean_var), spread_skill_ratio=KC.spread_skill_ratio(mean_var, mse_ens, K_),
                crps=acc["crps"] / n, rank_tv=KC.rank_hist_stats(acc["ranks"])["tv"], n_px=n)


def paired_metrics(S: SdaSamples) -> dict:
    """``{var: {subset: {metric: value}}}`` plus ``coh50_km`` and ``n_clean_days``."""
    out = {}
    for ci, v in enumerate(S.vars):
        accs = {s: _acc_new() for s in SUBSETS}
        paa = pbb = pab = None
        k = None
        n_clean = 0
        for d in range(S.D):
            x = S.gen_var(ci, d)
            y = S.truth[d, ci]
            sm = S.stat_mask[d]
            op = S.obs_pixels(v, d)
            _acc_add(accs["all"], x, y, sm, S.rng, S.K)
            if op is not None:
                _acc_add(accs["obs"], x, y, sm & op, S.rng, S.K)
                _acc_add(accs["gap"], x, y, sm & ~op, S.rng, S.K)
            if S.clean_day[d] and min(S.ny, S.nx) >= 32:
                tile = K.psd_tile(S.mask_day[d], S.ny, S.nx) if not S.is_patch else min(S.ny, S.nx)
                kk, a, b, c = KC.mean_radial_cross(x, y, S.mask_day[d], S.dx_km, tile=tile,
                                                   stride=max(tile // 2, 1))
                k = kk
                paa = a.sum(0) if paa is None else paa + a.sum(0)
                pbb = b.sum(0) if pbb is None else pbb + b.sum(0)
                pab = c.sum(0) if pab is None else pab + c.sum(0)
                n_clean += 1
        res = {s: _acc_final(accs[s], S.K) for s in SUBSETS}
        if op is None:
            for s in ("obs", "gap"):
                res[s] = {kk_: np.nan for kk_ in res["all"]}
        coh50 = float("nan")
        if paa is not None:
            coh50 = KC.coherence_half_wavelength(k, KC.coherence(pab, paa, pbb))
        res["coh50_km"] = coh50
        res["n_clean_days"] = n_clean
        out[v] = res
    return out


def obs_fit(S: SdaSamples) -> list[dict]:
    """Per term: RMS of ``A(x) - y`` for members, ensemble mean and truth."""
    rows = []
    if not S.terms:
        return rows
    std_t = torch.from_numpy(S.std.astype(np.float32))
    acc = {t["name"]: dict(mem=0.0, ens=0.0, tru=0.0, n=0) for t in S.terms}
    for d in range(S.D):
        ctx = O.ObsContext(ocean=torch.from_numpy(S.mask[d]), clim=torch.from_numpy(S.clim[d]),
                           std=std_t, is_log=list(S.is_log), dx_km=S.dx_km, target=list(S.vars))
        gen_d = torch.from_numpy(np.asarray(S.gen[d], np.float32))
        ens = gen_d.mean(0, keepdim=True)
        tru = torch.from_numpy(S.truth[d])[None]
        for ti, t in enumerate(S.terms):
            m = S.obs_mask[d, ti]
            if not m.any():
                continue
            term = O.ObsTerm(name=t["name"], var=t["var"], channel=int(t["channel"]), kind=t["kind"],
                             mask=torch.from_numpy(m), std_norm=float(t["std_norm"]), gamma=float(t["gamma"]),
                             sigma_px=float(t["sigma_px"]), demean=bool(t["demean"]), anomaly=bool(t.get("anomaly", False)),
                             y_source=t["y_source"])
            y = torch.from_numpy(S.y_grid[d, ti][m])
            with torch.no_grad():
                a = acc[t["name"]]
                a["mem"] += float(((term.apply(gen_d, ctx) - y) ** 2).sum())
                a["ens"] += float(((term.apply(ens, ctx) - y) ** 2).sum())
                a["tru"] += float(((term.apply(tru, ctx) - y) ** 2).sum())
                a["n"] += int(m.sum())
    for t in S.terms:
        a = acc[t["name"]]
        n = max(a["n"], 1)
        sn = float(t["std_norm"])
        rows.append(dict(term=t["name"], var=t["var"], kind=t["kind"], n_px=a["n"], std_norm=sn,
                         obsfit_member=np.sqrt(a["mem"] / n / S.K), obsfit_ensmean=np.sqrt(a["ens"] / n),
                         obsfit_truth=np.sqrt(a["tru"] / n),
                         obsfit_member_rel=np.sqrt(a["mem"] / n / S.K) / sn if sn > 0 else np.nan,
                         obsfit_truth_rel=np.sqrt(a["tru"] / n) / sn if sn > 0 else np.nan))
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_panel(S: SdaSamples, d: int, out: str):
    C = S.C
    fig, axs = plt.subplots(len(PANEL_ROWS), C, figsize=(2.6 * C + 1, 2.3 * len(PANEL_ROWS)),
                            constrained_layout=True, squeeze=False)
    land = ~S.mask_day[d]
    im = None
    for j, v in enumerate(S.vars):
        x = S.gen_var(j, d).astype(np.float32)
        y = S.truth[d, j]
        mean, sd = x.mean(0), x.std(0, ddof=1) if S.K > 1 else np.zeros_like(y)
        rows = [S.obs_grid(v, d, "pointwise"), S.obs_grid(v, d, "blur"), y, mean, sd,
                np.sqrt(((x - y) ** 2).mean(0)), np.abs(mean - y), x[0]]
        for i, (name, f) in enumerate(zip(PANEL_ROWS, rows)):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i < 2 and not np.isfinite(f).any():
                ax.set_axis_off()
                continue
            f = np.where(land, np.nan, f)
            if i in (4, 5, 6):
                ax.imshow(f, origin="lower", cmap="magma", vmin=0, vmax=2)
            else:
                im = ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
            ax.set_title(f"{v} {name}", fontsize=9)
    fig.suptitle(f"{S.title()}\nstep {int(S.days[d])} (sample {d}); colour in sigma units, "
                 f"std/err rows 0..2", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axs[:, -1].tolist(), shrink=0.4, location="right").set_label("sigma units")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_skill(M: dict, Mc: dict | None, vars_, out: str):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    xs = np.arange(len(vars_))
    w = 0.38
    for ax, sub in zip(axs, SUBSETS):
        g = [M[v][sub]["rmse_ens"] for v in vars_]
        sp = [M[v][sub]["spread"] for v in vars_]
        ax.bar(xs - w / 2, g, w, label="guided rmse_ens", color="C0")
        ax.plot(xs - w / 2, sp, "k_", ms=14, mew=2, label="guided spread")
        if Mc is not None:
            gc = [Mc[v][sub]["rmse_ens"] for v in vars_]
            ax.bar(xs + w / 2, gc, w, label="control rmse_ens", color="C7")
            ax.plot(xs + w / 2, [Mc[v][sub]["spread"] for v in vars_], "r_", ms=14, mew=2, label="control spread")
        ax.set_xticks(xs)
        ax.set_xticklabels(vars_, rotation=30)
        ax.set_title(f"{sub} pixels")
        ax.set_ylabel("sigma units")
        ax.axhline(1.0, color="0.7", lw=0.8, ls="--")
    axs[0].legend(fontsize=8)
    fig.suptitle("ensemble-mean RMSE (bars) and spread (ticks); 1.0 = climatology skill", fontsize=10)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_obsfit(rows: list[dict], rows_c: list[dict] | None, out: str):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.2 * len(rows) + 3, 3.8), constrained_layout=True)
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["obsfit_member_rel"] for r in rows], 0.4, label="guided members", color="C0")
    ax.plot(xs - 0.2, [r["obsfit_truth_rel"] for r in rows], "k_", ms=14, mew=2, label="truth")
    if rows_c:
        ax.bar(xs + 0.2, [r["obsfit_member_rel"] for r in rows_c], 0.4, label="control members", color="C7")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["term"] for r in rows], rotation=30)
    ax.set_yscale("log")
    ax.set_ylabel("RMS(A(x) - y) / assumed std")
    ax.set_title("observation misfit relative to the assumed error (1 = consistent)")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
def _stamp(S: SdaSamples, d: int) -> str:
    import datetime as _dt
    tu = getattr(S, "time_unix", None)
    if tu is not None and np.isfinite(tu[d]):
        return _dt.datetime.utcfromtimestamp(float(tu[d])).strftime("%Y-%m-%d %H:%M UTC")
    return f"step {int(S.days[d])}"


def animate_var(S: SdaSamples, Sc: SdaSamples | None, ci: int, out: str, fps: int = 6):
    """Per-variable movie over the case's steps. Row 1: observed pixels, truth,
    guided mean, control mean (if any). Row 2: L4 observation, guided spread,
    |guided mean - truth|, |control mean - truth|. Sigma units, fixed colour
    range, so frames are comparable."""
    import matplotlib.animation as manim

    v = S.vars[ci]
    ncol = 4 if Sc is not None else 3
    fig, axs = plt.subplots(2, ncol, figsize=(3.0 * ncol + 1, 6.4), constrained_layout=True, squeeze=False)
    titles = [["observed", "truth", "guided mean", "control mean"],
              ["L4 obs", "guided spread", "|guided - truth|", "|control - truth|"]]
    ims = []
    blank = np.full((S.ny, S.nx), np.nan, np.float32)
    for i in range(2):
        for j in range(ncol):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{v} {titles[i][j]}", fontsize=9)
            if i == 1 and j >= 1:
                ims.append(ax.imshow(blank, origin="lower", cmap="magma", vmin=0, vmax=2))
            else:
                ims.append(ax.imshow(blank, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3))
    fig.colorbar(ims[1], ax=axs[0].tolist(), shrink=0.7, location="right").set_label("sigma units")
    fig.colorbar(ims[ncol + 1], ax=axs[1].tolist(), shrink=0.7, location="right").set_label("sigma units")
    sup = fig.suptitle("", fontsize=10)

    def frame(d):
        land = ~S.mask_day[d]
        x = S.gen_var(ci, d)
        y = S.truth[d, ci]
        mean = x.mean(0)
        sd = x.std(0, ddof=1) if S.K > 1 else np.zeros_like(y)
        row1 = [S.obs_grid(v, d, "pointwise"), y, mean]
        row2 = [S.obs_grid(v, d, "blur"), sd, np.abs(mean - y)]
        if Sc is not None:
            mc = Sc.gen_var(ci, d).mean(0)
            row1.append(mc)
            row2.append(np.abs(mc - y))
        for im, f in zip(ims, row1 + row2):
            im.set_data(np.where(land, np.nan, f))
        sup.set_text(f"{S.title()}\n{_stamp(S, d)} (sample {d}/{S.D})")
        return ims

    anim = manim.FuncAnimation(fig, frame, frames=S.D, blit=False)
    writer = manim.FFMpegWriter(fps=fps, bitrate=2400) if "ffmpeg" in manim.writers.list() else manim.PillowWriter(fps=fps)
    if not out.endswith(".mp4") or isinstance(writer, manim.PillowWriter):
        out = os.path.splitext(out)[0] + (".gif" if isinstance(writer, manim.PillowWriter) else ".mp4")
    anim.save(out, writer=writer, dpi=110)
    plt.close(fig)
    return out


def animate_panel(S: SdaSamples, out: str, fps: int = 4):
    """The 8-row GenDA panel as a movie (heavier: 8 x C axes per frame)."""
    import matplotlib.animation as manim

    C = S.C
    fig, axs = plt.subplots(len(PANEL_ROWS), C, figsize=(2.6 * C + 1, 2.3 * len(PANEL_ROWS)),
                            constrained_layout=True, squeeze=False)
    blank = np.full((S.ny, S.nx), np.nan, np.float32)
    ims = {}
    for j, v in enumerate(S.vars):
        for i, name in enumerate(PANEL_ROWS):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{v} {name}", fontsize=9)
            cm = ("magma", 0, 2) if i in (4, 5, 6) else ("RdBu_r", -3, 3)
            ims[(i, j)] = ax.imshow(blank, origin="lower", cmap=cm[0], vmin=cm[1], vmax=cm[2])
    sup = fig.suptitle("", fontsize=9)

    def frame(d):
        land = ~S.mask_day[d]
        for j, v in enumerate(S.vars):
            x = S.gen_var(j, d)
            y = S.truth[d, j]
            mean = x.mean(0)
            sd = x.std(0, ddof=1) if S.K > 1 else np.zeros_like(y)
            rows = [S.obs_grid(v, d, "pointwise"), S.obs_grid(v, d, "blur"), y, mean, sd,
                    np.sqrt(((x - y) ** 2).mean(0)), np.abs(mean - y), x[0]]
            for i, f in enumerate(rows):
                ims[(i, j)].set_data(np.where(land, np.nan, f))
        sup.set_text(f"{S.title()}\n{_stamp(S, d)} (sample {d}/{S.D}); colour in sigma units, std/err rows 0..2")
        return list(ims.values())

    anim = manim.FuncAnimation(fig, frame, frames=S.D, blit=False)
    writer = manim.FFMpegWriter(fps=fps, bitrate=3000) if "ffmpeg" in manim.writers.list() else manim.PillowWriter(fps=fps)
    if isinstance(writer, manim.PillowWriter):
        out = os.path.splitext(out)[0] + ".gif"
    anim.save(out, writer=writer, dpi=100)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def write_outputs(S, M, Mc, fit, fit_c, out_dir: str) -> str:
    cols = ["var"]
    for sub in SUBSETS:
        cols += [f"{m}_{sub}" for m in ("rmse_ens", "rmse_ens_dm", "rmse_member", "spread", "spread_skill_ratio", "crps", "rank_tv", "n_px")]
    cols += ["coh50_km", "n_clean_days"]
    if Mc is not None:
        cols += ([f"ctrl_rmse_ens_{s}" for s in SUBSETS] + [f"skill_vs_control_{s}" for s in SUBSETS]
                 + [f"ctrl_rmse_ens_dm_{s}" for s in SUBSETS] + [f"skill_dm_vs_control_{s}" for s in SUBSETS]
                 + ["ctrl_coh50_km"])
    with open(os.path.join(out_dir, "summary_sda.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for v in S.vars:
            row = {"var": v, "coh50_km": M[v]["coh50_km"], "n_clean_days": M[v]["n_clean_days"]}
            for sub in SUBSETS:
                for m, val in M[v][sub].items():
                    row[f"{m}_{sub}"] = val
            if Mc is not None:
                for sub in SUBSETS:
                    rc = Mc[v][sub]["rmse_ens"]
                    row[f"ctrl_rmse_ens_{sub}"] = rc
                    row[f"skill_vs_control_{sub}"] = 1.0 - M[v][sub]["rmse_ens"] / rc if rc > 0 else np.nan
                    rcd = Mc[v][sub]["rmse_ens_dm"]
                    row[f"ctrl_rmse_ens_dm_{sub}"] = rcd
                    row[f"skill_dm_vs_control_{sub}"] = 1.0 - M[v][sub]["rmse_ens_dm"] / rcd if rcd > 0 else np.nan
                row["ctrl_coh50_km"] = Mc[v]["coh50_km"]
            wr.writerow({k: (f"{val:.5g}" if isinstance(val, float) else val) for k, val in row.items()})
    if fit:
        with open(os.path.join(out_dir, "obsfit_sda.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(fit[0].keys()) + (["ctrl_obsfit_member_rel"] if fit_c else []))
            wr.writeheader()
            for i, r in enumerate(fit):
                rr = dict(r)
                if fit_c:
                    rr["ctrl_obsfit_member_rel"] = fit_c[i]["obsfit_member_rel"]
                wr.writerow({k: (f"{val:.5g}" if isinstance(val, float) else val) for k, val in rr.items()})

    lines = [f"# SDA diagnostics", "", S.title(), ""]
    s = S.sda()
    lines += [f"- samples: `{S.path}`", f"- control: `{Mc is not None and S._control_path}`" if Mc is not None else "- control: none",
              f"- terms: {', '.join(s.get('terms', []))}", f"- precision {s.get('precision')}, chunk {s.get('chunk')}, "
              f"peak {s.get('peak_gb', 0):.1f} GB, {s.get('seconds', 0):.0f} s", ""]
    lines += ["## Ensemble-mean RMSE (sigma units), spread and calibration", "",
              "| var | rmse all | rmse demeaned | rmse obs | rmse gap | spread all | SSR all | CRPS all | rank TV | coh50 km"
              + (" | ctrl rmse all | skill vs ctrl | skill demeaned | skill gap" if Mc is not None else "") + " |",
              "|---|---|---|---|---|---|---|---|---|---|" + ("---|---|---|---|" if Mc is not None else "")]
    for v in S.vars:
        a = M[v]
        row = (f"| {v} | {a['all']['rmse_ens']:.3f} | {a['all']['rmse_ens_dm']:.3f} | {a['obs']['rmse_ens']:.3f} | "
               f"{a['gap']['rmse_ens']:.3f} | {a['all']['spread']:.3f} | {a['all']['spread_skill_ratio']:.2f} | "
               f"{a['all']['crps']:.3f} | {a['all']['rank_tv']:.3f} | {a['coh50_km']:.1f}")
        if Mc is not None:
            rc, rg, rcd = Mc[v]["all"]["rmse_ens"], Mc[v]["gap"]["rmse_ens"], Mc[v]["all"]["rmse_ens_dm"]
            row += (f" | {rc:.3f} | {1 - a['all']['rmse_ens'] / rc if rc > 0 else np.nan:.3f} | "
                    f"{1 - a['all']['rmse_ens_dm'] / rcd if rcd > 0 else np.nan:.3f} | "
                    f"{1 - a['gap']['rmse_ens'] / rg if rg > 0 else np.nan:.3f}")
        lines.append(row + " |")
    if fit:
        lines += ["", "## Observation misfit RMS(A(x)-y), normalised units; rel = / assumed std", "",
                  "| term | var | kind | n px | std | members | ens mean | truth | members rel | truth rel"
                  + (" | ctrl members rel" if fit_c else "") + " |",
                  "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if fit_c else "")]
        for i, r in enumerate(fit):
            row = (f"| {r['term']} | {r['var']} | {r['kind']} | {r['n_px']} | {r['std_norm']:.3g} | "
                   f"{r['obsfit_member']:.3g} | {r['obsfit_ensmean']:.3g} | {r['obsfit_truth']:.3g} | "
                   f"{r['obsfit_member_rel']:.2f} | {r['obsfit_truth_rel']:.2f}")
            if fit_c:
                row += f" | {fit_c[i]['obsfit_member_rel']:.2f}"
            lines.append(row + " |")
    lines += ["", "How to read: rmse in sigma units, 1.0 = climatology; control ~ sqrt(2) for independent "
              "draws. `demeaned` removes each step's mean over the scored pixels from both sides "
              "(SSH: the tide mode no demeaned product can constrain). "
              "`obs` = pixels under a pointwise term of that variable, `gap` = the rest. "
              "Members rel ~ 1 means the ensemble fits the observations to their assumed error; "
              "<< 1 over-fits noise, >> 1 under-guided. SSR 1 = calibrated (Fortin 2014). "
              "coh50 = finest scale at which members cohere with truth."]
    rep = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "report_sda.md"), "w") as f:
        f.write(rep)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", help="samples_sda_<size>.npz")
    ap.add_argument("--control", default=None, help="unguided samples_sda npz on the same case")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--n-panels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-border-cut", action="store_true")
    ap.add_argument("--animate", action="store_true",
                    help="write anim_<var>.mp4 per variable (truth / guided / control / errors over the steps)")
    ap.add_argument("--animate-panel", action="store_true", help="also the 8-row panel as anim_panel.mp4")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        raise SystemExit(_selftest())
    if not args.samples or not args.out:
        ap.error("--samples and --out are required")
    os.makedirs(args.out, exist_ok=True)
    S = SdaSamples(args.samples, args.seed, not args.no_border_cut)
    S._control_path = args.control
    M = paired_metrics(S)
    fit = obs_fit(S)
    Mc = fit_c = None
    if args.control:
        Sc = SdaSamples(args.control, args.seed, not args.no_border_cut)
        if not np.array_equal(Sc.days, S.days) or Sc.truth.shape != S.truth.shape:
            raise SystemExit("--control must be built on the same case (days / geometry differ)")
        Mc = paired_metrics(Sc)
        fit_c = obs_fit(Sc) if Sc.terms else None
    Sc_anim = None
    if args.control:
        Sc_anim = Sc
    for d in range(min(args.n_panels, S.D)):
        fig_panel(S, d, os.path.join(args.out, f"panel_day{d}.png"))
    if args.animate:
        for ci in range(S.C):
            f = animate_var(S, Sc_anim, ci, os.path.join(args.out, f"anim_{S.vars[ci]}.mp4"), args.fps)
            print(f"[anim] wrote {f}", flush=True)
    if args.animate_panel:
        f = animate_panel(S, os.path.join(args.out, "anim_panel.mp4"), max(args.fps // 2, 1))
        print(f"[anim] wrote {f}", flush=True)
    fig_skill(M, Mc, S.vars, os.path.join(args.out, "fig_skill_sda.png"))
    fig_obsfit(fit, fit_c, os.path.join(args.out, "fig_obsfit_sda.png"))
    rep = write_outputs(S, M, Mc, fit, fit_c, args.out)
    print(rep)


# ---------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    from diffusion.config import Config
    from sda._testing import _GaussPriorNet, _fake_dataset
    from sda.assimilate import build_parser, run_case
    from sda.case import build_case, fake_obs_spec

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    print("[sda.diagnostics_sda] selftest")
    cfg = Config(target=["ssh", "sst", "sss", "tau"], cond_vars=[], k_days=1, patch=16,
                 channel_mult=[1, 2], norm_mode="zscore", ocean_frac=0.5, sigma_data=1.0,
                 use_ocean_mask=False, use_doy=False)
    ds = _fake_dataset(cfg)
    case = build_case(cfg, ds, ds.spec, fake_obs_spec(), ds.valid_days[[0, 9, 20]], "32", seed=0,
                      log=lambda *a: None)
    for k in ("vars", "config", "obs_spec", "terms", "meta"):
        case[k] = json.loads(case[k])

    def run(**over):
        argv = ["--ckpt", "x", "--case", "x", "--out-dir", "x", "--k", "8", "--steps", "48",
                "--member-chunk", "8", "--precision", "fp32"]
        for k, v in over.items():
            argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        a = build_parser().parse_args(argv)
        a._ckpt_meta = {"ckpt": "/fake/run/ckpt.pt", "step": 10, "weights": "ema"}
        return run_case(_GaussPriorNet(0), cfg, case, a, torch.device("cpu"), log=lambda *a_: None)

    with tempfile.TemporaryDirectory() as tmp:
        pg, pc = os.path.join(tmp, "g.npz"), os.path.join(tmp, "c.npz")
        np.savez(pg, **run(guidance="on"))
        np.savez(pc, **run(guidance="off"))
        out = os.path.join(tmp, "figs")
        main(["--samples", pg, "--control", pc, "--out", out, "--n-panels", "1", "--animate",
              "--animate-panel", "--fps", "2"])
        anims = [f for f in os.listdir(out) if f.startswith("anim_")]
        check(len(anims) == 5 and all(os.path.getsize(os.path.join(out, f)) > 1000 for f in anims),
              f"animations written {sorted(anims)}")
        for fn in ("summary_sda.csv", "obsfit_sda.csv", "report_sda.md", "panel_day0.png",
                   "fig_skill_sda.png", "fig_obsfit_sda.png"):
            check(os.path.exists(os.path.join(out, fn)), f"wrote {fn}")
        with open(os.path.join(out, "summary_sda.csv")) as f:
            rows = {r["var"]: r for r in csv.DictReader(f)}
        tau, ssh = rows["tau"], rows["ssh"]
        check(float(tau["rmse_ens_all"]) < 0.2 * float(tau["ctrl_rmse_ens_all"])
              and float(tau["skill_vs_control_all"]) > 0.8,
              f"fully observed channel: guided rmse {float(tau['rmse_ens_all']):.3f} vs control "
              f"{float(tau['ctrl_rmse_ens_all']):.3f}")
        check(tau["rmse_ens_obs"] != "nan" and ssh["rmse_ens_obs"] == "nan",
              "obs/gap split only for variables under a pointwise term")
        check(0.8 < float(ssh["ctrl_rmse_ens_all"]) / np.sqrt(1 + 1 / 8) < 1.25,
              f"control rmse_ens ~ sqrt(1+1/K) for an exchangeable N(0,1) ensemble ({float(ssh['ctrl_rmse_ens_all']):.3f})")
        with open(os.path.join(out, "obsfit_sda.csv")) as f:
            fit = {r["term"]: r for r in csv.DictReader(f)}
        check(float(fit["full_t"]["obsfit_member_rel"]) < 3.0 and float(fit["full_t"]["ctrl_obsfit_member_rel"]) > 10,
              f"obsfit: guided members within a few assumed stds ({float(fit['full_t']['obsfit_member_rel']):.2f}), "
              f"control far off ({float(fit['full_t']['ctrl_obsfit_member_rel']):.1f})")
    print(f"[sda.diagnostics_sda] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
