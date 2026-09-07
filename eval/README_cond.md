# Conditional ensemble evaluation

`eval/README.md` describes the unconditional layer: it asks whether unconstrained
samples *look* like the ocean. This layer is for checkpoints whose
`cfg.cond_channels() != 0`, where every sample can be paired with the truth of
the same target step and with the observations it was conditioned on, so the
question becomes whether the ensemble is **right**, and **honest about how right
it is**. The scripts are new files beside the unconditional ones; nothing in the
unconditional layer is imported except the pure numerics in `eval/kernels.py`,
and nothing there was changed.

```bash
source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh && conda activate datorch
cd /unity/g2/jvelasco/gitraw/hycom_sbda

# 0. Gates. Both must pass before any number below is trusted.
python -m eval.kernels --selftest
python -m eval.kernels_cond --selftest
python -m eval.gen_cond --selftest          # the whole generation path on a fake dataset, CPU

# 1. Ensembles + truth + conditioning (GPU; minutes on gom_nemo, ~1 h cold on gulfstream)
CUDA_VISIBLE_DEVICES=1 python -m eval.gen_cond \
    --ckpt /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_joint/ckpt.pt \
    --out-dir /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_joint/eval_cond_step0200000 \
    --size full,256

# 2. Figures + tables (CPU, minutes, rerun freely)
python -m eval.diagnostics_cond \
    --samples .../eval_cond_step0200000/samples_cond_full.npz \
    --out     .../eval_cond_step0200000/figs_cond_full

# or both under SLURM (SIZES/K/STORE_COND are environment knobs)
SIZES=full,256 sbatch eval_cond.slurm .../runs/prior_joint/ckpt.pt
```

## Files

| file | what |
|---|---|
| `kernels_cond.py` | CRPS (fair, sorted form), rank histograms, Fortin spread/skill, cross-spectra and coherence on `kernels.radial_psd`'s bins, Parseval band power, a difference-of-Gaussians band-pass with a quoted transfer function, a synthetic-npz writer. `--selftest` pins each against a closed form. |
| `gen_cond.py` | checkpoint -> `samples_cond_<size>.npz`. `--selftest` runs it on a fake in-memory dataset and a zero network. |
| `diagnostics_cond.py` | npz -> nine figures, `summary.csv`, `summary_days.csv`, `spectra.npz`, `report.md` |
| `step_trajectory_cond.py` | paired skill / calibration / coherence vs training step over `<run>/traj_cond/step*` |
| `sweep_summary_cond.py` | one table over `<run>/sweep_cond/*` arms, read from `summary.csv` only |
| `../eval_cond.slurm`, `../traj_cond.slurm`, `../sweep_cond.slurm` | drivers mirroring `eval.slurm`, `traj.slurm`, `sweep.slurm` |

## Which models this is for

Conditioning in this repo is channel concatenation (`EDMPrecond.forward`,
`diffusion/edm.py:33`): `Cc = len(cond_vars) * k_days + n_extra`, with
`n_extra` the ocean-mask plane and the day-of-year sin/cos planes. Two families
are served, and `report.md` says which it saw:

- **`maskdoy`** -- `cond_vars: []` but `use_ocean_mask` and/or `use_doy` on
  (Cc = 1..3; `runs/prior_joint` is one, Cc = 3). There is no observation to
  score against. Skill against climatology is expected to be about zero; the
  questions are whether the ensemble is a *calibrated* climatological sampler for
  that day of year (rank histogram, spread/skill), whether its domain mean follows
  the season (correlation of domain means with the truth's, `figc9`), and what it
  draws on land, which the masked loss never constrained.
- **`obs`** -- `cond_vars` non-empty (gom_nemo `ssh_128.yaml`, or a gulfstream
  config conditioned on `ssha_aviso` / `sst_radiometer` / `sss_sat`). The newest
  step of each observation channel is stored, and the one that degrades each
  target (`baseline_map`) is scored as a deterministic baseline.

`gen_cond` refuses `cond_channels() == 0` and points at `eval.gen_prior`, the
mirror image of gen_prior's own refusal.

## The npz contract: `samples_cond_<size>.npz`

| key | shape / type | notes |
|---|---|---|
| `vars` | json list (Ct) | target names |
| `cond_vars` | json list (Cobs) | |
| `k_days` | int | |
| `extras` | json list | subset of `[ocean_mask, doy_sin, doy_cos]`, channel order |
| `gen_norm` | `(D,K,Ct,H,W)` **float16** | sigma units; `--gen-dtype float32` to opt out |
| `truth_norm` | `(D,Ct,H,W)` float32 | |
| `cond_norm` | `(D,Cc',H,W)` float32 | the tensor the net saw; `Cc'` = Cc (`--store-cond full`), Cobs+n_extra (`last`) or 0 (`none`) |
| `cond_channel_names` | json list | e.g. `ssha_aviso@t-1, ssha_aviso@t-0, ocean_mask, doy_sin, doy_cos` |
| `obs_avail` | `(D,Cobs,H,W)` bool | newest-step channel `!= 0.0` |
| `obs_age_s` | `(D,Cobs)` float64 | target time minus the mapped observation's time |
| `baseline_phys` | `(D,Ct,H,W)` float32 | the degraded obs in ITS OWN physical units; NaN where unmapped or unobserved |
| `baseline_map`, `baseline_kind`, `baseline_bias` | json, json, `(Ct,)` | see Baselines |
| `clim_day` | `(D,Ct,H,W)` or `(D,Ct,1,1)` float32 | per-day denormalisation mean: `phys = norm*std + clim_day`, then `10**` if `is_log` |
| `std`, `is_log` | `(Ct,)` | |
| `mask`, `lat` | `(D,H,W)` | **per sample** -- a conditional patch is located |
| `mask_full`, `lat_full` | `(NY,NX)` | |
| `dx_m`, `days`, `time_unix`, `doy`, `pos`, `size`, `config`, `meta` | | `pos` is `(D,2)` crop corners, `(0,2)` at full frame |

`meta` carries the checkpoint, step, weights, split, seed, D, K, member batch,
the sampler dict, `store_cond`, `gen_dtype`, `lattice_note`, `pos_mode`,
`base_cadence`, `step_seconds`, `baseline_map`/`kind`, `norm_mode`, and timings.

**float16.** The sampler runs under bf16 autocast, so each denoiser output
carries about three significant digits; fp16 storage (eleven mantissa bits) is
below that noise floor. Without it a gulfstream full-frame eval at D=48, K=16,
Ct=7 is 11.6 GB for `gen_norm` alone. Diagnostics upcast one `(K,H,W)` slice at
a time.

**A conditional patch is somewhere.** `gen_prior` had to treat a generated
patch as "nowhere" (no positional conditioning, so no latitude or climatology of
its own). Here the conditioning and the truth were cut from the full frame at
`pos`, so the patch has a real land mask, a real latitude and a real truth:
`mask` and `lat` are stored per sample and every per-sample statistic uses them.
When the eval patch equals the training patch the loader's own crop lattice is
used; otherwise a lattice is rebuilt for the requested size so the ocean-fraction
filter applies to the window actually cut (`meta.lattice_note` says which).

## Units

The model outputs normalised anomalies. Skill numbers are in **physical anomaly
units** (`norm * std`: m, degC, log10 mg/m3) so RMSE and CRPS read against
physical intuition; the `*_sigma` columns divide by the training std so
variables can be compared. Physical-field panels add `clim_day`, the day's own
climatology (or the scalar mean under `norm_mode: zscore`). Unlike `gen_prior`
there is no shared reference day: every sample is paired with the truth of the
same step, so each day's own climatology is the right one. Log10 variables
(CHL) stay in log space throughout, as in `eval.diagnostics`.

## `obs_avail`, and what 0.0 means

The loader writes an exact 0.0 into a normalised channel wherever the raw value
was non-finite (an observation gap) or the pixel is land
(`diffusion/data.py:564-565`, `:662-663`). In normalised units 0.0 is the mean,
and the network cannot tell a gap from a mean observation. So
`obs_avail = (newest cond channel != 0.0)` is exactly the model's view of what it
was told, which is what the fidelity panels need; land is also "unobserved". The
gulfstream daily/weekly observations are essentially gap-free (`ssha_aviso`,
`sst_radiometer` 100 % finite, `sss_sat` 99.98 %), so coverage- and age-dependence
panels are mostly interesting for gom_nemo `*_gap` configs. `obs_age_s` comes from
the store's timestamps: a daily or weekly observation is repeated across the
hourly steps it covers, so its age cannot be read off the tensor itself.

## Baselines

Two per target. **Climatology** (zero anomaly) needs nothing. **The degraded
observation** is the cond channel that degrades each target: `TRUTH_TO_DEGRADED`
first (`SSH -> SSH_aviso`, ...), then the unique cond var whose lowercase name
starts with the target's (`ssh -> ssha_aviso`, `sst -> sst_radiometer`,
`sss -> sss_sat`); `--baseline-map ssh=ssha_aviso,sss=none` overrides. It is
stored in its own physical units, denormalised with its own statistics, NaN
where unobserved.

**The SSH problem.** `ssha_aviso` is an anomaly about a mean dynamic topography
(mean 0.0004 m); `ssh` is not (mean 0.489 m on gulfstream, 0.095 m on gom_nemo).
`sst`/`sst_radiometer` and `sss`/`sss_sat` agree to 1e-3, so only SSH pairs have
this. `baseline_kind` is `anomaly` when the cond name contains `ssha` or `anom`
(or the target is listed in `--baseline-anom`), and the diagnostics then remove
the domain mean over the observed pixels from BOTH sides before scoring. The
residual spatial MDT structure is an irreducible caveat; `baseline_bias`
(mean obs - truth in physical units) records the offset. Skill scores against the
obs baseline use the ensemble mean on the same observed pixels.

## Metrics, and how to read them

- **Paired skill**: RMSE / MAE / bias of the ensemble mean; per-member RMSE; fair
  CRPS; skill scores `1 - RMSE/RMSE_baseline`; CRPS against the baseline's MAE
  (its probabilistic equivalent); CRPSS against a climatological ensemble made of
  the other D-1 truth steps (advisory: closely spaced steps make it too narrow).
- **Calibration**: spread/skill with the Fortin et al. (2014) `(K+1)/K`
  correction so 1.0 is calibrated for any K; rank histogram TV from uniform
  (the old gate was 0.2), tails ratio (> 1 under-dispersed) and slope (bias
  direction); `RMSE(member)/RMSE(mean)` against its calibrated value
  `sqrt(2K/(K+1))`; fraction of pixels whose spread/RMSE is within a factor 1.5.
- **Scale dependence**: PSD of truth, members, ensemble mean and ensemble-mean
  error from the same Hann-windowed tiles; band NRMSE and band spread/skill are
  Parseval integrals of those spectra, so no brick-wall filter leaks (the reason
  the old stack withdrew its band metrics); coherence between ensemble mean and
  truth, summarised as `coh50`, the finest scale still coherent. The ensemble
  mean's PSD ratio *falling* with wavelength is expected -- unpredictable scales
  average out -- so read the member ratio for realism and band spread/skill for
  whether the missing power reappears as spread. `--band-crps` adds pointwise
  band-passed CRPS / rank / spread-skill through a difference-of-Gaussians filter
  whose half-power points sit exactly on the band edges (transfer quoted in the
  report; it needs a domain several times the band's long edge, so it is `n/a`
  for the mesoscale band on a 128 px patch).
- **Realism** (the five unconditional figures, adapted): members pooled across
  days against the D truths for PDFs, EKE, cross-channel correlation; plus the
  correlation of the ensemble-mean *errors* across channels.
- **Fidelity**: `obs` family -- misfit of members / mean / TRUTH to the
  observation where observed (the truth's own misfit is the floor: beating it is
  overfitting a degraded product), error at observed vs gap pixels, spread vs
  coverage and vs observation age. `maskdoy` family -- correlation of the domain
  mean with the truth's across the record, domain std ratio vs season, |gen| on
  land, per-step rank TV vs season.

## Masking

Land per sample from the data (all-channel exact zeros in the truth), 6 px
Euclidean erosion before any derivative, as measured in `eval/README.md`. In
patch geometry the 16 px frame border is dropped from the distributional figures
(both sides; the zero-padding halo is a property of the network), while paired
skill is reported on the interior and `rmse_border_over_interior` measures the
halo directly; `--no-border-cut` scores the whole frame. Spectra use whole
rectangles: all-ocean tiles at full frame, and in patch geometry only land-free
samples, on both sides since they are paired.

## Cache reuse

`--real-from <earlier out-dir>` reuses truth, conditioning, baselines and
per-day climatology, skipping the store. Because the conditioning depends on the
config, the loader checks target list, `cond_vars`, `k_days`, extras,
`norm_mode`, split, geometry, `baseline_map`, sample count, and that the cache was
written with `--store-cond full` (a `last` cache with `k_days > 1` cannot rebuild
the network input). This is what makes `traj_cond.slurm` a paired comparison:
every rung sees identical observations, and `step_trajectory_cond` additionally
hashes the target days.

## Seeding

Members are drawn with `seed + 1000*day + 100*rep + off` (`rep` = repeat index
of a day in patch mode, `off` = member-batch index), so at full geometry the
members are bit-reproducible against `diffusion.sample.sample_days` for the same
sampler settings.

## Costs

- gom_nemo (464x528, 1872 daily steps): the 3-variable preload of `prior_joint`
  is ~5.5 GB and takes minutes. Gulfstream (576x936, 4367 hourly steps): ~66 GB,
  ~148 s warm, ~1 h cold.
- Generation: ~4.3 s per full 576x936 frame at 32 steps (64-channel genda),
  ~0.1 s per 128 patch at batch 64. Measured on `prior_joint` (128-channel nemo,
  464x528, A100): 0.81 s per full-frame member at 6 steps, i.e. ~4.3 s at 32
  steps; 0.16 s per 256 patch member and 0.06 s per 128 patch member at 6 steps.
  D=48 x K=16 full frames at 32 steps: about an hour.
- Sizes: gom_nemo full D=48, K=16, Ct=3 -> ~1.5 GB; gulfstream full Ct=7 ->
  ~8.5 GB (`D_FULL=32` for ~5.7 GB); 128 patch D=256 -> 0.4-1.2 GB. With a large
  k window on gulfstream, `--store-cond last`.

## Gotchas

- `ZarrWindowDataset` fits and writes climatology / normalisation caches for
  variables it has not seen before (`diffusion/_clim_cache.npz`,
  `_norm_cache.json`; rank 0, atomic). Harmless, but an eval job can touch them.
- `prior_joint/ckpt.pt` predates the `arch` and `norm_mode` fields; they resolve
  to `nemo` and `zscore`, so `clim_day` is `(D,Ct,1,1)`. It was trained at patch
  256, so `--size full,256` is the training geometry and `128` a legitimate
  fully-convolutional extrapolation (record `lattice_note`).
- Rank-histogram TV is computed over spatially correlated pixels; its sampling
  error is far larger than the pixel count suggests. `figc3`'s inset shows the
  per-step distribution.
- The gulfstream conditional path (cross-cadence observations, `obs_age_s`) is
  exercised by the selftests only until a checkpoint exists.
