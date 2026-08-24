# Unconditional prior evaluation

What has the prior actually learned? The training loss cannot answer that: it
has been flat since roughly step 380,000 while the model kept training, so the
question has to be asked in sample space.

These scripts draw **completely unconstrained** samples — no conditioning
channels, no observations, no day, no location — and compare them against real
fields from the held-out split.

```bash
source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh && conda activate datorch
cd /unity/g2/jvelasco/gitraw/hycom_sbda

# 0. Always run this first. It gates everything downstream.
python -m eval.kernels --selftest

# 1. Draw samples (~1 h cold / ~10 min with a warm page cache; see below)
CUDA_VISIBLE_DEVICES=1 python -m eval.gen_prior \
    --ckpt /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt \
    --out-dir /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/eval_step1140000

# 2. Figures (seconds to a few minutes, CPU only, rerun freely)
python -m eval.diagnostics \
    --samples .../eval_step1140000/samples_full.npz \
    --out     .../eval_step1140000/figs_full

# or both steps under SLURM
sbatch eval.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt
```

## The three things worth understanding before reading a figure

**1. The model generates anomalies, not fields.** `prior_gulfstream.yaml` uses
`norm_mode: anomaly`, and `sst`/`sss` additionally have a monthly climatology
removed (`clim_vars: [sst, sss]`). That climatology is large — sst runs 20.2 °C
in March to 26.4 °C in July — and the network was deliberately never asked to
model it.

The real val frames span 2017-05-26 to 2017-07-01, so if you denormalised them by
their own timestamps you would put that seasonal spread into the comparison and
the generated sst distribution would look far too narrow for a reason that has
nothing to do with the model. So **both sides are denormalised with the same
`--ref-day` climatology**. A width difference in `fig3_pdf.png` is the model's.

Consequently the figures use two different unit conventions, on purpose:

| figure | units | why |
|---|---|---|
| `fig1_gallery`, `fig3_pdf` | physical, shared reference climatology | readable against physical intuition |
| `fig2_spectra`, `fig4_eke`, `fig5_crosschannel` | anomaly only | the climatology is a fixed spatial field common to both sides; including it would add identical power to both spectra and a shared spatial pattern to both correlation matrices, manufacturing agreement |

**2. Two geometries, because they ask different questions.** The UNet is fully
convolutional, but its single attention block was built for 16×16 tokens
(`img_resolution=128`, 3 downsamples) and at full frame runs on 72×117.

- `samples_128.npz` — exactly the training patch distribution. Nothing is
  extrapolated; this is the cleanest read of what was learned.
- `samples_full.npz` — 576×936, how the prior will be used in assimilation and
  how GenDA runs its own inference. If a statistic disagrees between the two, the
  full-frame attention extrapolation is the first suspect.

**3. A generated patch is nowhere.** The model has no positional conditioning, so
a 128×128 sample has no latitude and no local climatology. Patch mode therefore
uses the domain-mean latitude (35.0 °N) and domain-mean climatology for *both*
generated and real patches. The comparison stays fair; the absolute EKE in patch
mode is not a physical measurement. Read full-frame EKE for that.

## Files

| file | what |
|---|---|
| `kernels.py` | radial PSD, geostrophy, EKE, moments, correlation. `--selftest` |
| `gen_prior.py` | checkpoint → `samples_<size>.npz` |
| `diagnostics.py` | `samples_<size>.npz` → 5 figures + `summary.csv` + `report.md` |

`radial_psd` and `tile_positions` are ported verbatim from
`nemo_confusion/run_diagnostics.py`, and `geostrophic_uv` is adapted from
`nemo_confusion/diffusion/metrics.py` (whose default latitude is
Gulf-of-Mexico specific). They are copied, not imported, because this repo is
standalone by construction — see the top-level README.

## Why `--selftest` is not optional

Both ported kernels fail *quietly*. A sign error in `geostrophic_uv` yields EKE
maps that look completely plausible; a botched window normalisation in
`radial_psd` shifts every spectrum by a constant that reads as a model defect.
The selftest checks the PSD slope against a synthetic power-law field, checks
geostrophy against the store's **own** `ug`/`vg` fields, and checks that the
store really decomposes `u = ug + uag` — which is what lets `fig4` add the
geostrophic velocity implied by the generated ssh to the generated `uag`/`vag`
channels and call the result a total. Those extra fields exist alongside the
seven trained channels and are free ground truth. Current numbers:

```
[psd ] fitted slope -3.037 vs expected -3.000 over 14.9-72.2 km  -> PASS
[geo ] corr(u,ug)=0.9992  corr(v,vg)=0.9975                      -> PASS
[vel ] u = ug + uag to 1.32e-03 of its own rms                   -> PASS
[vel ] v = vg + vag to 2.36e-03 of its own rms                   -> PASS
```

## Gotchas that cost time

- **The dataset preload is ~66 GB**: measured at **~56 min cold, ~2.5 min when
  the OS page cache still holds the store**. Decompression is single-threaded,
  which is what makes the cold case slow. That is why `--size` takes a
  comma-separated list: one load serves every geometry. Generation itself is
  minutes (~4.3 s per full frame, ~0.12 s per patch). Don't be surprised by
  either number -- the first run of the day is the slow one.
- **Do not read the grid from `cfg.zarr_path`.** A checkpoint records it as it
  resolved at training time, and for gulfstream it still holds the *gom_nemo*
  default — `resolve_spec` only back-fills it for single-store families and
  gulfstream has seven. `gen_prior.py` takes `dx_m` and `coords/lat` from the
  resolved `DatasetSpec` instead.
- **Real reference frames are strided across the whole split, not the last N.**
  The base cadence is hourly, so the last 48 steps are one weather state and
  would badly understate how much real fields vary.
- **The last 24 hours of the record are excluded** (`--skip-tail`). The record
  ends 2017-07-01 23:00, so July's monthly climatology is constrained by 24 hours
  alone and those steps are not clean anomalies.
- **The store has 17 non-finite pixels** in every frame (`ocean_mask` is
  uniformly 1.0; the real mask is `bathymetry`'s finite pattern). The loader maps
  them to 0.0, and `np.gradient` smears each into its neighbours — enough to set
  the colour scale of an EKE map. Anything involving a derivative uses
  `kernels.erode_mask`.

## Comparing checkpoints

Both scripts are parameterised only by `--ckpt` and the output directory, so
running them against an older snapshot costs no extra code:

```bash
sbatch eval.slurm .../prior_gulfstream/checkpoints/ckpt_step0400000.pt
```

That directly answers whether the ~740k steps after the val curve went flat
bought anything in sample space.

The real reference fields depend on the dataset, split and seed — never on the
checkpoint — so the second and later runs should not pay the hour again:

```bash
python -m eval.gen_prior --ckpt .../ckpt_step0400000.pt \
    --out-dir .../eval_step0400000 \
    --real-from .../eval_step1140000        # reuses real fields; ~5 min total
```

`--real-from` checks that the cached variable list matches and that it holds at
least as many samples as you asked for; it refuses rather than silently
mismatching. It also reuses the same `ref_day`, which is what makes two
checkpoints' figures directly comparable.
