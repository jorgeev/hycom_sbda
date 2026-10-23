"""Three read-only products for the UNOBSERVED current components (uag, vag) of
a finished SDA arm, built from ``samples_sda_<size>.npz`` (+ the control):

1. member view     fig_memberview_<var>.png, memberview.csv
   The ensemble mean of a field no observation constrains at fine scale is
   smooth by construction. This product scores the ensemble as a distribution
   instead: per step the MEDOID member (smallest summed distance to the other
   members over the frame), a random member, the mean and the truth side by
   side; CRPS, mean RMSE, member RMSE (mean / min / medoid) and the radial PSD
   of truth, members (average over members) and ensemble mean, with band
   ratios. The medoid is the "representative realisation": a real sample of the
   posterior, not an average.

2. scale split     fig_scalesplit_<var>.png, scalesplit.csv
   Which scales does the mean actually carry? Band-wise correlation of the
   ensemble mean with the truth (Gaussian band-pass, 2-4-9-18-36-72-150 km and
   >150) gives the cut wavelength L where corr(mean, truth) drops below 0.5.
   The figure then shows mean low-passed at L (what the observations
   constrain), the fine-scale spread (what they do not), one member's
   fine-scale part (a sample of it), and the truth split the same way.

3. Ekman check     fig_ekman_<var>.png, ekman.json
   Wind stress is fully observed, so the wind-driven (Ekman) part of the
   ageostrophic current should be nearly deterministic given tau. Low-pass
   (> ``--ekman-km``, default 100 km) uag and vag are regressed on (tau_x,
   tau_y) pixelwise over all steps: once for the truth, once for every member.
   Reported: the 2x2 response matrices (m/s per N/m^2, i.e. in physical units),
   the variance fraction they explain, the member-to-member spread of the
   low-pass component relative to its truth amplitude, and the correlation of
   each member's low-pass uag/vag with the truth's. Members that agree with each
   other and with the truth on this part have learned the tau -> current
   coupling; members that disagree have not.

Usage:
  python -m sda.ageo_products --samples <arm>/samples_sda_full.npz \
      [--control <arm>/samples_sda_full_free.npz] --out <arm>/figs_ageo [--vars uag,vag]
All quantities in the variable's normalised (sigma) units unless stated.
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
from scipy.ndimage import gaussian_filter  # noqa: E402

from eval import kernels as K  # noqa: E402
from eval import kernels_cond as KC  # noqa: E402
from sda.diagnostics_sda import SdaSamples, _stamp  # noqa: E402

BAND_EDGES_KM = [0, 2, 4, 9, 18, 36, 72, 150, np.inf]


def band_labels():
    e = BAND_EDGES_KM
    return [f"{e[i]:g}-{e[i + 1]:g}" if np.isfinite(e[i + 1]) else f">{e[i]:g}" for i in range(len(e) - 1)]


def gap_blur(f: np.ndarray, ok: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return np.where(ok, f, 0.0)
    num = gaussian_filter(np.where(ok, f, 0.0).astype(np.float64), sigma_px, truncate=3.0)
    den = gaussian_filter(ok.astype(np.float64), sigma_px, truncate=3.0)
    return np.where(ok, num / np.maximum(den, 1e-6), 0.0)


def band_decompose(f: np.ndarray, ok: np.ndarray, dx_km: float) -> list[np.ndarray]:
    """Gaussian difference bands on the edges above (sigma = edge/dx)."""
    out, prev = [], np.where(ok, f - f[ok].mean(), 0.0)
    for s in BAND_EDGES_KM[1:-1]:
        b = gap_blur(prev, ok, s / dx_km)
        out.append(prev - b)
        prev = b
    out.append(prev)
    return out


def masked_corr(a, b, ok):
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


# ---------------------------------------------------------------------------
# 1. member view
# ---------------------------------------------------------------------------
def medoid_index(x: np.ndarray, ok: np.ndarray) -> int:
    """x (K,H,W): member with the smallest summed L2 distance to the others."""
    v = x[:, ok].astype(np.float32)
    sq = (v * v).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * v @ v.T
    return int(np.argmin(d2.sum(1)))


def member_view(S: SdaSamples, Sc: SdaSamples | None, var: str, out_dir: str, rng: np.random.Generator):
    ci = S.vars.index(var)
    rows = []
    psd_t, psd_m, psd_mean, k_axis = [], [], [], None
    med_err, rnd_err, mean_err, memb_err, crps = [], [], [], [], []
    for d in range(S.D):
        ok = S.stat_mask[d]
        x = S.gen_var(ci, d)
        y = S.truth[d, ci]
        mean = x.mean(0)
        mi = medoid_index(x, ok)
        ri = int(rng.integers(S.K))
        e = lambda f: float(np.sqrt(((f - y)[ok] ** 2).mean()))
        med_err.append(e(x[mi])); rnd_err.append(e(x[ri])); mean_err.append(e(mean))
        me = np.array([e(x[k]) for k in range(S.K)])
        memb_err.append((me.mean(), me.min(), me.max()))
        crps.append(float(np.mean(KC.crps_ensemble(x[:, ok].astype(np.float64), y[ok].astype(np.float64)))))
        if d % max(1, S.D // 8) == 0:          # spectra on ~8 evenly spaced steps (cost)
            tile = min(512, S.ny, S.nx)
            rp = lambda st: K.mean_radial_psd(np.stack([np.where(ok, f - f[ok].mean(), 0.0) for f in st]), ok, S.dx_km,
                                              tile=tile, stride=tile // 2, min_frac=0.99)
            kk, pt = rp([y]); _, pm = rp([x[k] for k in range(min(S.K, 8))]); _, pe = rp([mean])
            psd_t.append(pt[0]); psd_m.append(pm.mean(0)); psd_mean.append(pe[0]); k_axis = kk
        rows.append(dict(step=int(S.days[d]), medoid=mi, rmse_mean=mean_err[-1], rmse_medoid=med_err[-1],
                         rmse_random=rnd_err[-1], rmse_member_mean=memb_err[-1][0], rmse_member_min=memb_err[-1][1],
                         rmse_member_max=memb_err[-1][2], crps=crps[-1]))
    with open(os.path.join(out_dir, f"memberview_{var}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # figure: a representative step (median mean-RMSE), 4 fields + PSD
    d = int(np.argsort(mean_err)[len(mean_err) // 2])
    ok = S.stat_mask[d]; x = S.gen_var(ci, d); y = S.truth[d, ci]; mean = x.mean(0); mi = rows[d]["medoid"]
    land = ~S.mask_day[d]
    fig = plt.figure(figsize=(16, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    ri = int(rng.integers(S.K))
    panels = [("truth", y), ("ensemble mean", mean), (f"medoid member #{mi + 1}", x[mi]), (f"random member #{ri + 1}", x[ri])]
    for j, (t, f) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(np.where(land, np.nan, f), origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"{var} {t}", fontsize=9)
    ax = fig.add_subplot(gs[1, :2])
    if k_axis is not None:
        lam = 1.0 / np.maximum(np.asarray(k_axis), 1e-9)
        ax.loglog(lam, np.mean(psd_t, 0), "k", label="truth")
        ax.loglog(lam, np.mean(psd_m, 0), "C3", label="members (avg PSD)")
        ax.loglog(lam, np.mean(psd_mean, 0), "C0", label="ensemble mean")
        ax.invert_xaxis(); ax.set_xlabel("wavelength [km]"); ax.set_ylabel("PSD"); ax.legend(fontsize=8)
        ax.set_title(f"{var} radial PSD, averaged over {len(psd_t)} steps", fontsize=9)
    ax = fig.add_subplot(gs[1, 2:])
    steps = np.arange(S.D)
    ax.plot(steps, mean_err, "C0", label="RMSE ensemble mean")
    ax.plot(steps, med_err, "C3", label="RMSE medoid member")
    ax.plot(steps, [m[0] for m in memb_err], "C3", ls="--", label="RMSE member (avg)")
    ax.plot(steps, crps, "C2", label="CRPS")
    if Sc is not None:
        ax.plot(steps, [float(np.sqrt(((Sc.gen_var(ci, dd).mean(0) - Sc.truth[dd, ci])[Sc.stat_mask[dd]] ** 2).mean())) for dd in range(Sc.D)],
                "k", ls=":", label="RMSE control mean")
    ax.axhline(1, color="gray", lw=0.5); ax.set_xlabel("step"); ax.set_ylabel("sigma units"); ax.legend(fontsize=8)
    ax.set_title("per step: the mean wins on RMSE by averaging away what it does not know; the medoid is a real sample", fontsize=9)
    fig.suptitle(f"{S.title()}\nmember view, {var}; fields at {_stamp(S, d)} (median-RMSE step)", fontsize=10)
    fig.savefig(os.path.join(out_dir, f"fig_memberview_{var}.png"), dpi=110); plt.close(fig)
    return dict(rmse_mean=float(np.mean(mean_err)), rmse_medoid=float(np.mean(med_err)),
                rmse_member_avg=float(np.mean([m[0] for m in memb_err])), crps=float(np.nanmean(crps)),
                psd_ratio_members_over_truth=([float(a / b) for a, b in zip(np.mean(psd_m, 0), np.mean(psd_t, 0))] if k_axis is not None else None),
                psd_wavelength_km=(list(map(float, 1.0 / np.maximum(np.asarray(k_axis), 1e-9))) if k_axis is not None else None))


# ---------------------------------------------------------------------------
# 2. scale split
# ---------------------------------------------------------------------------
def scale_split(S: SdaSamples, var: str, out_dir: str, rng: np.random.Generator):
    ci = S.vars.index(var)
    labels = band_labels()
    corr_mean = np.zeros((S.D, len(labels))); corr_memb = np.zeros((S.D, len(labels)))
    amp_truth = np.zeros((S.D, len(labels))); amp_mean = np.zeros((S.D, len(labels))); amp_spread = np.zeros((S.D, len(labels)))
    for d in range(S.D):
        ok = S.stat_mask[d]
        x = S.gen_var(ci, d); y = S.truth[d, ci]; mean = x.mean(0)
        bt = band_decompose(y, ok, S.dx_km); bm = band_decompose(mean, ok, S.dx_km)
        k0 = int(rng.integers(S.K)); bk = band_decompose(x[k0], ok, S.dx_km)
        for i in range(len(labels)):
            corr_mean[d, i] = masked_corr(bm[i], bt[i], ok)
            corr_memb[d, i] = masked_corr(bk[i], bt[i], ok)
            amp_truth[d, i] = bt[i][ok].std(); amp_mean[d, i] = bm[i][ok].std()
        # spread per band: std over members of the band-passed fields, averaged over pixels (4 members for cost)
        ks = rng.choice(S.K, size=min(4, S.K), replace=False)
        bands_k = [band_decompose(x[k], ok, S.dx_km) for k in ks]
        for i in range(len(labels)):
            stack = np.stack([b[i] for b in bands_k])
            amp_spread[d, i] = float(stack[:, ok].std(0).mean())
    cm = corr_mean.mean(0)
    # cut wavelength: first band (from fine to coarse) where corr(mean, truth) >= 0.5
    cut_i = next((i for i in range(len(labels)) if cm[i] >= 0.5), len(labels) - 1)
    cut_km = BAND_EDGES_KM[cut_i]
    with open(os.path.join(out_dir, f"scalesplit_{var}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["band_km", "corr_mean_truth", "corr_member_truth", "amp_truth", "amp_mean", "spread_members"])
        for i, l in enumerate(labels):
            w.writerow([l, f"{cm[i]:.3f}", f"{corr_memb.mean(0)[i]:.3f}", f"{amp_truth.mean(0)[i]:.4f}", f"{amp_mean.mean(0)[i]:.4f}", f"{amp_spread.mean(0)[i]:.4f}"])

    # figure at the median step
    d = S.D // 2; ok = S.stat_mask[d]; land = ~S.mask_day[d]
    x = S.gen_var(ci, d); y = S.truth[d, ci]; mean = x.mean(0)
    s_px = max(cut_km, 1e-3) / S.dx_km
    low = lambda f: gap_blur(f, ok, s_px)
    mean_lo = low(mean); truth_lo = low(y)
    fine_sd = np.std(np.stack([x[k] - low(x[k]) for k in range(S.K)]), 0)
    k0 = int(rng.integers(S.K)); memb_fine = x[k0] - low(x[k0]); truth_fine = y - truth_lo
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.6), constrained_layout=True)
    items = [[(f"mean, scales > {cut_km:g} km (constrained)", mean_lo, "RdBu_r", -3, 3),
              (f"truth, scales > {cut_km:g} km", truth_lo, "RdBu_r", -3, 3),
              ("ensemble mean, all scales", mean, "RdBu_r", -3, 3), ("truth, all scales", y, "RdBu_r", -3, 3)],
             [(f"fine-scale spread (< {cut_km:g} km)", fine_sd, "magma", 0, 2),
              (f"truth fine scales (< {cut_km:g} km)", truth_fine, "RdBu_r", -3, 3),
              (f"member #{k0 + 1} fine scales (a sample)", memb_fine, "RdBu_r", -3, 3), None]]
    for i in range(2):
        for j in range(4):
            ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
            it = items[i][j]
            if it is None:
                ax.bar(np.arange(len(labels)), cm, color="C0", label="corr(mean, truth)")
                ax.plot(np.arange(len(labels)), corr_memb.mean(0), "C3.-", label="corr(one member, truth)")
                ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45, fontsize=7)
                ax.axhline(0.5, color="gray", lw=0.5); ax.set_ylim(-0.1, 1); ax.legend(fontsize=7)
                ax.set_title("band-wise correlation with truth (all steps)", fontsize=9)
                continue
            t, f, cmap, lo, hi = it
            ax.imshow(np.where(land, np.nan, f), origin="lower", cmap=cmap, vmin=lo, vmax=hi); ax.set_title(f"{var} {t}", fontsize=9)
    fig.suptitle(f"{S.title()}\nscale split, {var}: the mean is trusted where corr(mean, truth) >= 0.5 -> cut at {cut_km:g} km; "
                 f"fields at {_stamp(S, d)}", fontsize=10)
    fig.savefig(os.path.join(out_dir, f"fig_scalesplit_{var}.png"), dpi=110); plt.close(fig)
    return dict(cut_km=float(cut_km), corr_mean_truth=dict(zip(labels, map(float, cm))),
                corr_member_truth=dict(zip(labels, map(float, corr_memb.mean(0)))))


# ---------------------------------------------------------------------------
# 3. Ekman check
# ---------------------------------------------------------------------------
def ekman_check(S: SdaSamples, var_u: str, var_v: str, out_dir: str, ekman_km: float):
    cu, cv = S.vars.index(var_u), S.vars.index(var_v)
    ctx, cty = S.vars.index("tau_x"), S.vars.index("tau_y")
    s_px = ekman_km / S.dx_km
    # physical units: sigma * std (anomaly units for the currents; tau is absolute in the state)
    su, sv, stx, sty = (float(S.std[c]) for c in (cu, cv, ctx, cty))
    Xs, Yt, Ym = [], [], []        # design (tau), truth low-pass currents, member low-pass currents
    sub = slice(None, None, 4)     # pixel subsample for the regression
    spread_rel, corr_u, corr_v = [], [], []
    for d in range(S.D):
        ok = S.stat_mask[d]
        tx = S.truth[d, ctx] * stx; ty = S.truth[d, cty] * sty
        tu = gap_blur(S.truth[d, cu] * su, ok, s_px); tv = gap_blur(S.truth[d, cv] * sv, ok, s_px)
        xu = np.stack([gap_blur(S.gen_var(cu, d)[k] * su, ok, s_px) for k in range(S.K)])
        xv = np.stack([gap_blur(S.gen_var(cv, d)[k] * sv, ok, s_px) for k in range(S.K)])
        sel = np.zeros_like(ok); sel[sub, sub] = True; sel &= ok
        Xs.append(np.stack([gap_blur(tx, ok, s_px)[sel], gap_blur(ty, ok, s_px)[sel]], 1))
        Yt.append(np.stack([tu[sel], tv[sel]], 1))
        Ym.append(np.stack([xu[:, sel], xv[:, sel]], 2))          # K, n, 2
        amp = np.sqrt(tu[ok].var() + tv[ok].var()) + 1e-9
        spread_rel.append(float(np.sqrt(xu[:, ok].var(0).mean() + xv[:, ok].var(0).mean()) / amp))
        corr_u.append(float(np.mean([masked_corr(xu[k], tu, ok) for k in range(S.K)])))
        corr_v.append(float(np.mean([masked_corr(xv[k], tv, ok) for k in range(S.K)])))
    X = np.concatenate(Xs); X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.concatenate(Yt)
    def fit(Yc):
        coef, *_ = np.linalg.lstsq(X1, Yc, rcond=None)
        res = Yc - X1 @ coef
        r2 = 1.0 - res.var(0) / (Yc.var(0) + 1e-12)
        return coef[:2].T, r2        # (2x2 response: rows u,v ; cols tau_x,tau_y), R^2 per component
    Mt, r2t = fit(Y)
    Ymm = np.concatenate(Ym, axis=1)   # K, N, 2
    Mk, r2k = zip(*[fit(Ymm[k]) for k in range(S.K)])
    Mk = np.stack(Mk); r2k = np.stack(r2k)
    out = dict(ekman_lowpass_km=ekman_km, units="m/s per N/m^2; rows (u,v), cols (tau_x,tau_y)",
               truth_response=Mt.tolist(), truth_r2=r2t.tolist(),
               member_response_mean=Mk.mean(0).tolist(), member_response_std=Mk.std(0).tolist(),
               member_r2_mean=r2k.mean(0).tolist(),
               lowpass_member_spread_over_truth_amp=float(np.mean(spread_rel)),
               lowpass_corr_member_truth_u=float(np.mean(corr_u)), lowpass_corr_member_truth_v=float(np.mean(corr_v)))
    json.dump(out, open(os.path.join(out_dir, "ekman.json"), "w"), indent=1)

    # figure: low-pass fields at the median step + response matrices
    d = S.D // 2; ok = S.stat_mask[d]; land = ~S.mask_day[d]
    tu = gap_blur(S.truth[d, cu], ok, s_px); xu = np.stack([gap_blur(S.gen_var(cu, d)[k], ok, s_px) for k in range(S.K)])
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.6), constrained_layout=True)
    vmax = max(float(np.abs(tu[ok]).max()), 1e-3)
    panels = [(f"truth {var_u} > {ekman_km:g} km", tu), (f"member-mean {var_u} > {ekman_km:g} km", xu.mean(0)),
              (f"member 1 {var_u} > {ekman_km:g} km", xu[0]), (f"member 2 {var_u} > {ekman_km:g} km", xu[1 % S.K])]
    for j, (t, f) in enumerate(panels):
        ax = axs[0, j]; ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(np.where(land, np.nan, f), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.set_title(t, fontsize=9)
    ax = axs[1, 0]; ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.where(land, np.nan, S.truth[d, ctx]), origin="lower", cmap="RdBu_r", vmin=-3, vmax=3); ax.set_title("tau_x (observed)", fontsize=9)
    ax = axs[1, 1]; ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.where(land, np.nan, xu.std(0)), origin="lower", cmap="magma", vmin=0, vmax=vmax); ax.set_title(f"member spread of {var_u} > {ekman_km:g} km", fontsize=9)
    ax = axs[1, 2]
    lab = ["u<-tau_x", "u<-tau_y", "v<-tau_x", "v<-tau_y"]
    ax.bar(np.arange(4) - 0.2, Mt.ravel(), 0.4, color="k", label="truth")
    ax.bar(np.arange(4) + 0.2, Mk.mean(0).ravel(), 0.4, yerr=Mk.std(0).ravel(), color="C3", label="members (mean +- sd)")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab, fontsize=8); ax.axhline(0, color="gray", lw=0.5); ax.legend(fontsize=8)
    ax.set_title("linear response of low-pass current to tau [m/s per N/m2]", fontsize=9)
    ax = axs[1, 3]
    ax.bar([0, 1], r2t, 0.4, color="k", label="truth R2"); ax.bar([0.4, 1.4], r2k.mean(0), 0.4, color="C3", label="members R2")
    ax.set_xticks([0.2, 1.2]); ax.set_xticklabels([var_u, var_v]); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title(f"variance of low-pass current explained by tau\nmember spread / truth amp = {out['lowpass_member_spread_over_truth_amp']:.2f}; "
                 f"corr(member, truth) u {out['lowpass_corr_member_truth_u']:.2f} v {out['lowpass_corr_member_truth_v']:.2f}", fontsize=8)
    fig.suptitle(f"{S.title()}\nEkman check: is the wind-driven (> {ekman_km:g} km) part of the ageostrophic current determined by the observed tau? "
                 f"fields at {_stamp(S, d)}", fontsize=10)
    fig.savefig(os.path.join(out_dir, f"fig_ekman_{var_u}.png"), dpi=110); plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vars", default="uag,vag")
    ap.add_argument("--ekman-km", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of products to skip: memberview,scalesplit,ekman")
    args = ap.parse_args(argv)
    S = SdaSamples(args.samples)
    Sc = SdaSamples(args.control) if args.control else None
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    skip = set(args.skip.split(",")) if args.skip else set()
    summary = {}
    vars_ = args.vars.split(",")
    for v in vars_:
        if "memberview" not in skip:
            summary[f"memberview_{v}"] = member_view(S, Sc, v, args.out, rng)
            print(f"[ageo] member view {v}: mean RMSE {summary[f'memberview_{v}']['rmse_mean']:.3f} medoid {summary[f'memberview_{v}']['rmse_medoid']:.3f} CRPS {summary[f'memberview_{v}']['crps']:.3f}")
        if "scalesplit" not in skip:
            summary[f"scalesplit_{v}"] = scale_split(S, v, args.out, rng)
            print(f"[ageo] scale split {v}: cut {summary[f'scalesplit_{v}']['cut_km']:g} km; corr(mean,truth) by band "
                  + " ".join(f"{k}:{c:.2f}" for k, c in summary[f"scalesplit_{v}"]["corr_mean_truth"].items()))
    if "ekman" not in skip and len(vars_) >= 2:
        summary["ekman"] = ekman_check(S, vars_[0], vars_[1], args.out, args.ekman_km)
        e = summary["ekman"]
        print(f"[ageo] Ekman: truth R2 {np.round(e['truth_r2'], 2).tolist()} members R2 {np.round(e['member_r2_mean'], 2).tolist()}; "
              f"low-pass spread/amp {e['lowpass_member_spread_over_truth_amp']:.2f}; corr(member,truth) u {e['lowpass_corr_member_truth_u']:.2f} v {e['lowpass_corr_member_truth_v']:.2f}")
    json.dump(summary, open(os.path.join(args.out, "ageo_summary.json"), "w"), indent=1)
    print(f"[ageo] wrote {args.out}")


if __name__ == "__main__":
    main()
