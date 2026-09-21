# sda/ -- score-based data assimilation on the unconditional priors

Reproduces GenDA's inference (Martin et al. 2025; Rozet & Louppe 2023) against
the EDM priors trained by `diffusion/`. New files only: nothing in `diffusion/`
or `eval/` changed, the layer imports downward (`diffusion.*`, `eval.kernels`,
`eval.kernels_cond`) and writes its ensembles in the `eval.diagnostics_cond`
schema so that script scores them unmodified.

## Quickstart

```bash
source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh && conda activate datorch
cd /unity/g2/jvelasco/gitraw/hycom_sbda

# gates (CPU, no store): operator adjoint, closed-form posterior, schema
python -m sda.obs --selftest && python -m sda.assimilate --selftest --with-diagnostics \
    && python -m sda.case --selftest && python -m sda.diagnostics_sda --selftest

# 1. the case: truth + observations for 24 steps at 256x256 (opens the store ONCE, ~1 h cold)
python -m sda.case --ckpt runs/prior_gulfstream/best.pt --obs sda/configs/gs_store_obs.yaml \
    --size 256 --n 24 --out-dir runs/prior_gulfstream/sda_gs_store_obs_step1140000

# 2. control (no guidance) and guided ensembles, same seeds
CUDA_VISIBLE_DEVICES=1 python -m sda.assimilate --ckpt runs/prior_gulfstream/best.pt \
    --case .../case_256.npz --out-dir .../vp --guidance off --out samples_sda_256_free.npz
CUDA_VISIBLE_DEVICES=1 python -m sda.assimilate --ckpt runs/prior_gulfstream/best.pt \
    --case .../case_256.npz --out-dir .../vp --guidance on --verbose

# 3. score
python -m eval.diagnostics_cond --samples .../vp/samples_sda_256.npz --out .../vp/figs_cond_256
python -m sda.diagnostics_sda --samples .../vp/samples_sda_256.npz \
    --control .../vp/samples_sda_256_free.npz --out .../vp/figs_sda_256

# or all of it: sbatch assimilate.slurm runs/prior_gulfstream/best.pt 1 sda/configs/gs_store_obs.yaml 24
```

## Files

| file | what |
|---|---|
| `obs.py` | `ObsTerm`/`ObsContext`, land-safe Gaussian blur, masks (`full`, `none`, `store_finite`, `tracks`, `swath`), YAML spec, `build_terms`, `--selftest` |
| `vpsde.py` | VP-cosine schedule, `PriorDenoiser` (EDMPrecond as `d_fn(x, sigma)`), `vp_sample` (GenDA's predictor-corrector), `edm_sample` |
| `guidance.py` | `GaussianGuidance`: guided denoiser `D + sigma^2 grad log p(y|x)`, autograd through the UNet; `wrap_checkpoint` |
| `case.py` | store -> `case_<size>.npz`; day picking, crop lattice, cadence matching |
| `assimilate.py` | case + checkpoint -> `samples_sda_<size>.npz`; `--guidance on|off`, `--sampler vp|edm`, sharding, `--check-precision` |
| `diagnostics_sda.py` | GenDA panels, movies (`--animate`), obs/gap and demeaned skill vs control, observation misfit; `summary_sda.csv`, `report_sda.md` |
| `_testing.py` | `_ZeroNet`, `_GaussPriorNet` (closed-form posterior), fake spec/dataset/store |
| `configs/gs_store_obs.yaml` | the store's own observation products as `y` (headline) |
| `configs/gs_genda_like.yaml` | GenDA-style synthetic tracks + blurred-truth L4 (ablation) |

## The algorithm

Prior denoiser `D(x, sigma)` = `EDMPrecond` with the zero-width conditioning
tensor the priors were trained with, weights frozen, bf16 autocast, fp32 out.

Guidance (`guidance.py`, GenDA `GaussianScore`): with `x` requiring grad,
`x_hat = D(x, sigma)` (Tweedie; for `eps_edm` this equals GenDA's
`(x - sigma eps)/mu` exactly), `log p = -1/2 sum (y - A(x_hat))^2 / (std^2 + gamma sigma^2)`,
`g = grad_x log p` through the network, `D_post = D + sigma^2 g`. No clamps.

Samplers (`vpsde.py`): `vp` is Rozet & Louppe's loop verbatim -- `mu(t) =
cos(acos(sqrt(eta)) t)^2`, `sigma(t)^2 = 1 - mu^2 + eta^2`, `x_1 ~ N(0, I)`,
256 uniform steps, predictor `x <- r x + (sigma' - r sigma) eps` with
`eps = (mu/sigma)(x/mu - D_post(x/mu, sigma/mu))`, optional Langevin
corrections (`--corrections`, `--tau`). `edm` runs the same `D_post` on a
Karras schedule with Euler (or `--edm-heun`) steps. Both agree on the
closed-form toy in the selftest.

## Observation spec (`configs/*.yaml`)

```yaml
- {name: ssh_aviso, var: ssh, kind: blur, sigma_km: 65, demean: true,
   y_source: "store:ssha_aviso", mask: {source: full}, noise_std: 0.05}
```
`kind: pointwise` selects pixels; `kind: blur` converts the state to physical
units, applies a Gaussian of `sigma_km` (normalised convolution with the ocean
mask; `cutoff_km` is accepted with the parent repo's `/(2 pi)` convention),
converts back, and drops a rim of `border_px` (`auto` = 1 sigma). `demean`
subtracts the mean over the term's pixels in physical units on both `A(x)` and
`y`. `anomaly: true` marks a product that is an anomaly relative to a mean sea
surface (SSHA): the operator then acts on the state anomaly and never adds the
climatology. Without it, the blurred mean dynamic topography sits in `A(x)` but
not in `y` -- 0.25 m RMS across a 256 patch of the Gulf Stream, five times the
AVISO error, and the ensemble was pulled toward that phantom field. With it the
truth misfit is 0.023 m. `y_source: truth` derives `y` from the truth frame and adds
`noise_std` (physical units, divided by the prior's per-channel std) --
`noise_std_norm` gives it directly; `store:<var>` reads the store product at the
matched index (`time_match: nearest` default; `last` = the loader's
no-future-observation rule) and adds no noise unless `add_noise: true`. Masks:
`full`, `none`, `store_finite: {var}` (finite pixels of a store frame -- the
real cloud mask), `tracks` / `swath` (synthetic, deterministic per seed and
day). `optional_terms` are built with `--include-optional` / `--terms`.

Measured on the gulfstream store (2026-09-07, val split), which is why
`gs_store_obs.yaml` looks the way it does:

| channel | what it is | vs hourly truth | term |
|---|---|---|---|
| `ssha_aviso` (daily) | 65 km Gaussian (36 px) of the *daily-mean* SSH, minus a mean SSH = time-mean + 0.053 m; **no tide** | 0.023 m (0.01-0.04) with the anomaly operator, demeaned, on a 256 patch | blur 65 km, demean, anomaly, 0.03 m |
| `sst_radiometer` (daily) | OSTIA-like *daily mean*, sigma 2-4 km | 0.15-0.30 degC, domain bias up to +-0.45 degC (diurnal cycle) | blur 3 km, demean, 0.20 degC |
| `sss_sat` (weekly) | CATDS-like, sigma 51 km (28 px) | 0.035-0.045 psu (0.6 sigma_sss) | blur 51 km, 0.03 psu |
| `sst_sat` (hourly) | GOES-like, native res, real cloud mask; coverage 25% (3-66%) | 0.02-0.05 degC, tails 1.5 degC at fronts | pointwise, store_finite, 0.05 degC |
| `ssha_swot` (hourly) | swaths in 6.6% of hours (55 in val), tide kept, per-swath offset -0.06 +- 0.09 m | 0.007-0.085 m | optional pointwise, demean, anomaly, 0.05 m; `--require-coverage ssha_swot` |
| `tau_x`, `tau_y` | the forcing | -- | from truth, std 1e-2 normalised (GenDA winds) |

The hourly truth SSH carries a near-uniform barotropic tide of +-0.3 m (31% of
the anomaly variance) that AVISO lacks; `demean` on both sides removes it and
the 0.053 m offset together. Observing-system arms are `--terms` subsets, e.g.
`TERMS=sst_sat,tau_x,tau_y` (no altimetry) or `TERMS=ssh_aviso,sst_l4,sss_l4`
(L4 only), one arm directory each.

## Output (`samples_sda_<size>.npz`)

The `eval.diagnostics_cond.CondSamples` key set (`gen_norm (D,K,C,H,W)` fp16,
`truth_norm`, `clim_day`, `std`, `is_log`, `mask`, `lat`, `days`, `pos`, ...):
pointwise terms appear as `cond_vars` / `cond_norm` / `obs_avail` and as the
per-variable `baseline_phys` (`baseline_kind: anomaly` for a demeaned term, so
the mean is removed on both sides there). Extras for `sda.diagnostics_sda`:
`y_grid (D,T,H,W)` and `obs_mask (D,T,H,W)` for every term, `terms` (settings),
`obs_spec`, and `meta.sda` (guidance, sampler, steps, gamma, precision, peak
memory, seconds, `log_p_last`).

## Costs (estimates; the first chunk prints measured time and peak memory)

One forward and one backward through the UNet per member per step. Measured
on an A100-80GB with the 7-channel gulfstream prior at 256^2: a guided step for
24 members takes 0.42 s and 32 GB peak in bf16 (fp32: 0.43 s, 36 GB -- TF32
convolutions make the two nearly equal), so a 256-step, 24-member ensemble is
~110 s per target step; bf16 and fp32 guidance agree to cosine 0.999 and 1% in
norm at sigma/mu = 29, 1.7 and 0.22. Full frame (576x936) is 8.2x the pixels:
`--member-chunk auto` starts at 24 / 24 / 2 (128^2 / 256^2 / full) and halves
on OOM; `--grad-ckpt` cuts memory ~3x for one extra forward per block. Full
frame wants `--day-shard i/3` on three GPUs and `--merge`.

## Reading `report_sda.md`

`rmse all` is the ensemble-mean RMSE in sigma units (1 = climatology; the
unguided control sits near sqrt(2)); `rmse demeaned` removes each step's mean
over the scored pixels from both sides -- for SSH that is the tide mode, which
no demeaned altimetry product can constrain and which was 40% of the SSH error
on the first smoke case. `obs`/`gap` split pixels by whether a pointwise term
observes them. The observation-misfit table is the calibration tool: `truth
rel` is the truth's own misfit against each product divided by the assumed
error, and should be ~1 -- above 1 the assumed error is too small and the
ensemble fits representation error (this is how the stds in
`gs_store_obs.yaml` were set); `members rel` far below `truth rel` means the
same thing from the other side.

## Gotchas

- **Schedule extrapolation.** `sigma/mu` runs from 1000 (t=1) to ~0.01 at 256
  steps, while training sampled sigma in ~[0.01, 10] and the checkpoint's own
  sampler stops at 80. GenDA does the same; the control run (`--guidance off`)
  against `eval_step*/samples_*.npz` diagnostics is the arbiter. `--eta 1e-2`
  caps `sigma/mu` at 100 if it matters.
- **gamma is not a nuisance knob.** On the closed-form toy (`_GaussPriorNet`)
  only the exact Tweedie covariance recovers the posterior; GenDA's
  `gamma sigma^2` under-guides at gamma=1 (mean 0.5x) and over-fits the
  observations below 0.1 (variance collapse). 0.1 is a compromise, tuned
  against a unit-variance prior. Sweep `--gamma-scale`.
- **Blur scale in pixels.** 65 km is 36 px here (kernel radius 108 px): the
  AVISO term cannot live in a 128 patch, and even with the 1 sigma rim it
  constrains only the inner 52% of a 256 patch (61% for the 51 km SSS term).
  Whatever lies outside is unconstrained by that product -- read the SSS and
  SSH rows of the report with that in mind, and treat full frame as the real
  target for the L4 terms.
- **Tight fully-observed terms** (wind stress, std 1e-2) dominate `log p`;
  `--verbose` prints per-term gradient norms on the first chunk. The EDM
  sampler needs >= 64-128 steps with such terms; the VP path is more tolerant.
- **Spread** with `--corrections 0` comes from `x_1` only; `--corrections 1
  --tau 0.3` is the GenDA knob if the rank histogram is U-shaped.
- **Blur as matmul, not conv2d.** `gaussian_blur` applies the two banded blur
  matrices (`M_y x M_x^T`) with replicate padding folded in. The conv2d form
  is exact too, but cuDNN's backward for a 217-tap 1-D kernel took 5.3 s per
  call at 256^2 (forward 36 ms) -- an 80x slowdown of the whole guided step.
  The selftest checks the two agree to 1e-6.
- **Land**: gulfstream has 17 masked pixels only; `--land-mode replace` (fresh
  noise at land pixels each step) is for the GoM priors.
- `Config.assim_config` / `corrector_*` in `diffusion/config.py` are left
  untouched; SDA settings live on the CLI and in `meta.sda`.
