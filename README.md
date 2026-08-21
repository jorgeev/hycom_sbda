# hycom_sbda

Score-based diffusion priors for gridded ocean state — **self-contained, and
dataset-agnostic by construction.**

Extracted from the `nemo_confusion` concept test (branch `portable-datasets`).
This copy carries the *training core only*: everything needed to train a model
on a zarr store and draw ensembles from it, with no dependency on that project's
Gulf-of-Mexico evaluation stack. It is a plain directory — copy it anywhere,
point a descriptor at your data, and run.

## What it does

An EDM-preconditioned diffusion model (Karras et al. preconditioning, Heun
sampler) in two framings:

- **Conditional super-resolution** — condition on a `k`-step window of
  satellite-like degraded channels, generate the high-resolution truth at the
  latest step. (`configs/ssh_128.yaml`)
- **Unconditional joint prior** — learn `p(x)` over the full state, for use as a
  Bayesian prior in score-based data assimilation, following Martin et al. 2025
  (GenDA) and Rozet & Louppe 2023 (SDA). (`configs/prior_*.yaml`)

Self-contained PyTorch: no `diffusers`, no `einops`. The dataset-builder
dependencies (`xarray`, `cv2`, `pyproj`, `cftime`) are deliberately absent —
nothing here imports them.

## The idea: the store is data, not code

No module knows a store's layout or its variable names. A **descriptor** in
`diffusion/datasets/*.yaml` declares them; an experiment config selects one with
`dataset:`. Adding a new dataset is one YAML file and zero Python edits.

Two descriptors ship, chosen because they differ *structurally* rather than
just in naming — between them they exercise every branch of the loader:

| | `gom_nemo.yaml` | `gulfstream.yaml` |
|---|---|---|
| stores | one | **seven monthly**, concatenated via a manifest |
| cadences | one (daily) | **three** — hourly truth, daily + weekly observations |
| grid | 4 km LAEA, 464×528 | 1.818 km AEQD, 576×936 |
| stats from | `meta/<var>/{mean,std}` | manifest (per-store `meta/` is per-month) |
| splits from | store `splits.json` | manifest `splits.<cadence>_boundary` |
| mask | real land (63.8 % ocean) | `ocean_mask` is all-1; real mask comes from `bathymetry` NaNs |
| time axis | daily with **15 real gaps** | hourly, uniform |

## Quickstart

```bash
source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh && conda activate datorch
cd /unity/g2/jvelasco/gitraw/hycom_sbda

# 1. Always validate a descriptor before spending GPU time on it
python -m diffusion.dataset_spec --selftest diffusion/datasets/gulfstream.yaml

# 2. Smoke test (1 GPU)
CUDA_VISIBLE_DEVICES=1 python -m diffusion.train \
    --config diffusion/configs/prior_gulfstream.yaml \
    --steps 30 --batch 4 --max-days 900 --num-workers 2 --out runs/smoke
CUDA_VISIBLE_DEVICES=1 python -m diffusion.sample \
    --ckpt runs/smoke --k-members 2 --days 2 --split train

# 3. Full run (DDP). The GenDA-parity configs want a 64-sample global batch,
#    so 2 ranks x batch 32 -- 3 ranks cannot split it evenly.
sbatch train.slurm diffusion/configs/prior_gulfstream.yaml 2 1,2
```

`sample.py` writes `ensembles.npz` with the schema
`{days, mask, target_vars, config, gen_<var>, truth_<var>}` — values in physical
units. That file is the integration seam to any downstream evaluation.

## Adding a dataset

Write `diffusion/datasets/<name>.yaml`, then **run the selftest before
training** — it catches the failures that are otherwise expensive or silent:

```bash
python -m diffusion.dataset_spec --selftest diffusion/datasets/<name>.yaml
```

It checks group/variable presence, `(T, NY, NX)` consistency, that the domain
divides by the UNet downsample factor (otherwise full-frame inference dies deep
inside a skip connection), split contents, grid spacing, the resolved mask, and
— for multi-store or multi-cadence families — that reads across a store seam are
contiguous and that no cross-cadence lookup returns a *future* observation.

Descriptor keys:

| key | meaning |
|---|---|
| `stores` / `manifest` | one path, a list, or a manifest whose `stores` entries carry `global_*_offset` |
| `base_cadence`, `cadences` | index spaces: `dynamic_group`, `time_path`, `splits_file`, `offset_key` |
| `variables` | per variable: `cadence`, `log10`, `units`. Omitted variables default to the base cadence |
| `static_group`, `mask_var` | where the land mask lives; `null` synthesises all-ones |
| `mask_from_finite` | ANDs in the finite-pixel pattern of a static field, for stores that encode land as NaN |
| `meta_group`, `stats_source` | `meta` \| `manifest` \| `compute` |
| `splits_source` | `store` (sidecar JSON) \| `manifest` (`splits.<cadence>_boundary`) |

### Three things that will bite you

1. **`k_days`, `crops_per_day`, `val_gap_days`, `n_eval_days` count steps of the
   base cadence, not days.** On gulfstream (`base_cadence: hourly`) they are
   hours. The names are kept only because every saved checkpoint carries them in
   its config dict.
2. **`norm_mode: anomaly` with non-empty `clim_vars` needs ~2 years of record.**
   It fits an annual + semiannual harmonic, which is not identifiable from a
   shorter one — the fit latches onto whatever transient is present and
   subtracts it from the signal you are trying to model. `dataset_spec` warns;
   use `clim_vars: []` (per-pixel time-mean) instead. Gulfstream spans ~6 months,
   so `prior_gulfstream.yaml` sets it empty.
3. **Check your mask is real.** Gulfstream's `ocean_mask` is uniformly 1.0, but
   17 pixels are NaN in every frame of every field; `bathymetry` carries exactly
   that pattern, hence `mask_from_finite: bathymetry`. Without it those pixels
   enter the loss as hard zeros the model is asked to reproduce.

## Layout

```
diffusion/
  dataset_spec.py   DatasetSpec + StoreFamily: multi-store concat, per-variable
                    cadence, timestamp-based index mapping, validation, selftest
  config.py         Config dataclass + YAML loader with CLI overrides
  data.py           ZarrWindowDataset: windowing, normalisation (zscore |
                    anomaly/climatology), patch sampling, denormalisation
  networks.py       SongUNet backbone (attention by level index)
  networks_genda.py GenDA's DDPM++ variant (attention by absolute resolution)
  edm.py            Karras preconditioning, loss, Heun sampler (det. | stochastic)
  ema.py            EMA weights used for sampling
  train.py          DDP loop; --mode diffusion (default) | regression
  sample.py         K-member full-frame ensembles -> ensembles.npz
  utils.py          DDP setup, seeding, CSV logger
  datasets/         dataset descriptors
  configs/          experiment configs
```

## Compatibility with the parent repo

The eleven `diffusion/*.py` files are **byte-identical** to
`nemo_confusion@portable-datasets`, so the two stay diffable and changes can be
moved either way. Only `__init__.py`, this README, `train.slurm`,
`environment.yml` and `.gitignore` are new here.

A config that sets no `dataset:` resolves to `dataset_spec.legacy_spec()` — the
original single-store layout — so configs and checkpoints from the parent repo
load and run unchanged. Verified: 20 fixed-seed steps of `ssh_128.yaml` and
`prior_genda_masked.yaml` reproduce **bit-identically** (668 tensors across
`model` and `ema`, max abs delta 0.0) against the pre-refactor code.

## Not included

The parent repo's evaluation and diagnostics stack — `evaluate.py`, `metrics.py`,
`gate.py`, `baselines.py`, `eddies.py`, `seasonal.py`, `animate.py` — and the
assimilation layer `obs_operator.py` / `assimilate.py`. All of those are still
coupled to Gulf-of-Mexico variable names and SSH-specific physics (geostrophy at
a hardcoded latitude, Okubo-Weiss, AVISO cutoffs). Porting them is a separate
pass; until then, run them from the parent repo against `ensembles.npz`.
