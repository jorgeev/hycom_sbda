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

A *real* patch, however, is somewhere — and that matters for its land. Each
sample npz stores the real crop corners (`real_pos`) and the full-domain ocean
mask (`mask_full`) so `diagnostics.py` can rebuild each real patch's own land
mask before taking any spatial derivative. See the land-edge EKE gotcha below
for what happens without it.

## Files

| file | what |
|---|---|
| `kernels.py` | radial PSD, geostrophy, EKE, moments, correlation. `--selftest` |
| `gen_prior.py` | checkpoint → `samples_<size>.npz` |
| `diagnostics.py` | `samples_<size>.npz` → 5 figures + `summary.csv` + `report.md` |
| `step_trajectory.py` | many checkpoints' `samples_<size>.npz` → spectra vs training step |
| `sweep_summary.py` | one dispersion table over a sampler sweep |

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

- **Land in the real patches contaminates every real-side statistic, and the
  spectrum worst of all.** In patch geometry a *generated* patch is all ocean by
  construction, while a *real* patch was cropped from somewhere: on gom_nemo the
  `crop_mode: grid` lattice admits anything with `ocean_frac >= 0.5`, so real
  patches average 87.5 % ocean and bottom out at 51 %. Land is an exact `0.0`
  anomaly, so pooling it into the real side drags the real std down, pulls the
  real correlation matrix toward the identity, and — because a land edge is a
  step function — hands the FFT broadband power that is not ocean signal.
  Measured on `prior_genda_masked` at step 1,130,000 this inflated the real
  10–40 km PSD by **40x for SSH** and ~4x for SST and CHL, which reported a
  genuine 5x *excess* of SSH grid-scale power as an apparent 0.18 *deficit* —
  the opposite diagnosis, from a figure that looked entirely reasonable.
  `Samples` therefore carries three things and the figures pick the right one:
  `real_grad_mask` (eroded, per sample) for derivatives, `real_mask` (per
  sample) for pixel statistics, and `real_clean` (per sample) for the spectra,
  which need a whole rectangle and so must *drop* contaminated samples rather
  than mask pixels. The figure title says how many survived; `fig_spectra`
  warns if fewer than 16 do. On gulfstream, whose "land" is 17 isolated pixels,
  the same correction moves nothing — which is exactly why this went unnoticed
  there.

  Read the corrected patch spectra with one caveat: an unconditional draw has no
  location, so the real side cannot be matched to it exactly. Dropping the
  contaminated real patches leaves a deep-basin subsample, while the model's
  draws are a mixture over every lattice position it trained on, shelf included.
  That is a bias — a much smaller one than a 40x inflated denominator, but state
  it rather than quoting the ratio as exact. The full geometry does not have
  this asymmetry (both sides use the same fixed all-ocean tiles), at the cost of
  asking the model for a frame far larger than anything it was trained on.

- **No 256 px window on the Gulf of Mexico is all-ocean.** `K.psd_tile` picks
  the largest all-ocean tile, which is 256 on gulfstream but only **128 on
  gom_nemo** — the basin is not wide enough anywhere for a 1024 km square, nor
  a 768 km one. So the full-frame spectra there resolve no larger scale than the
  patch geometry does (512 km). Relaxing the ocean fraction instead would be
  wrong: land is zeros, so a tile straddling Florida is a hole whose edge
  dominates the FFT. Genuinely larger scales would need a land-tolerant PSD
  estimator, which this repo does not have.


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
- **Land inside a real patch is a trap for derivatives** (bug found in the
  step-1,140,000 patch eval, fixed 2026-08-24). Land pixels hold anomaly
  `== 0.0` exactly, so where a real 128×128 crop contains coastline,
  `np.gradient` sees an ocean→land step of ~0.2 m of ssh over one cell and
  `geostrophic_uv` turns it into O(10) m/s of spurious velocity — up to
  **425 m²/s² of "EKE"** at a small island near full-grid (115, 218), vs ~1
  for a real eddy. Because patch crops come from the training loader's
  stride-32 lattice (`crop_mode: grid`, stride = patch//4), that one island
  landed at the same few patch-relative offsets in every sample, and the
  sample-mean map in `fig4_eke.png` showed it as a *regular grid of identical
  bright comma-shaped blobs* — easy to misread as a model or data defect. The
  full-frame figures never showed it because full mode stores the true ocean
  mask and `grad_mask` erodes it; patch mode stores `mask = ones` (a generated
  patch is nowhere, point 3 above), which silently threw away the *real*
  patches' geography too.

  The fix: `gen_prior.py` now stores `mask_full` in every npz, and
  `diagnostics.py` builds a per-sample eroded mask for the real side of
  `fig4_eke` and the geostrophy panel of `fig5_crosschannel`. Effect on the
  step-1,140,000 numbers: real patch EKE was inflated ~4 % (26 of 512 crops
  contain land), so the total-EKE ratio moved x0.59 → x0.61 — the maps were
  badly polluted but the scalar conclusion stood. A pre-fix patch npz can be
  backfilled without repaying the store load, using the full-geometry npz
  written alongside it:

  ```bash
  python - <<'EOF'
  import numpy as np
  d = dict(np.load("eval_stepNNN/samples_128.npz", allow_pickle=False))
  d["mask_full"] = np.load("eval_stepNNN/samples_full.npz")["mask"]
  np.savez("eval_stepNNN/samples_128.npz", **d)
  EOF
  ```

- **2 px of erosion was not enough, and it was the wrong shape** (found
  2026-08-27 while reading `fig4_eke.png` for `prior_genda_masked` at step
  1,130,000). Two separate problems, both in `kernels.erode_mask`:

  *Depth.* Real geostrophic EKE binned by distance from that patch's own
  coastline runs **80x** the far field at 1–2 px, **10x** at 2–3 px, **3x** at
  3–4 px and **2x** at 4–6 px, reaching background only near 6 px. A 2 px
  erosion keeps everything from 3 px outward, i.e. a rim still 2–3x too bright.
  `diagnostics.GRAD_ERODE_PX` is now **6**, chosen because that is where the
  answer stops moving: real patch-mean EKE is 0.0438 at 2 px, 0.0408 at 6,
  0.0409 at 10, then drifts back *up* to 0.0426 at 24 as the surviving pixels
  become a deep-basin subsample rather than a cleaner one.

  *Shape.* The old implementation was four-neighbour erosion iterated `pix`
  times, which erodes by a **diamond**: it clears the cardinal directions to
  `pix` but the diagonals to only `pix/√2`. At `pix=6` that left a ring of
  pixels 4–6 px (Euclidean) from land whose mean EKE was still 0.075 against
  0.032 at 8–10 px — the exact rim the widening was meant to remove, surviving
  in the corners of the diamond. It now uses
  `scipy.ndimage.distance_transform_edt`, so `pix` is a true Euclidean radius.

- **A generated patch has a bright frame, and a patch-mean map is a fold.**
  Two things make the sample-mean EKE maps in patch geometry misleading, and
  neither shows up in the scalar:

  *Zero-padding halo.* A generated patch contains no land, but the UNet's zero
  padding inflates its EKE near the frame: 0.0855 within 2–4 px of the edge
  against 0.0648 in the interior (**+32 %**), while the real side is flat over
  the same bins (0.0454 vs 0.0407). Split-half reproducibility of the generated
  mean map falls from r = 0.78 with a 2 px cut to 0.45 at 16 px and 0.29 at
  32 px — 0.29 being the real side's own level. `diagnostics.PATCH_BORDER_PX`
  now drops **16 px** from every frame edge, on *both* sides, in patch geometry
  only. Full frame keeps its border: there the edge is a real domain boundary
  that generated and real fields share.

  *Crop-lattice fold.* Real patches come off the training loader's stride-32
  lattice, so patch pixel `(i, j)` and `(i + 32, j)` average almost the same
  set of absolute positions. The sample-mean map is therefore the domain's own
  geography **folded at 32 px**, not a map of anywhere: its shift
  autocorrelation is 0.02 at 24 px, **0.67 at 32 px** and 0.05 at 36 px. The
  *generated* side shows the same 32 px period (0.72), because that lattice is
  the distribution it was trained on — the two folds correlate at **+0.41**
  while what is left after removing them correlates at +0.09. The fold survives
  16 px of land erosion, so the repeated closed rings in the real panel are
  aliased geography, not land contamination. `fig4_eke.png` now says so on the
  panel. Read the *level* of those two maps and the log-ratio; do not read the
  pattern as a place.

  Effect of both corrections together on step 1,130,000: patch geostrophic EKE
  went x1.60 → **x1.58** (real 0.0438 → 0.0408, generated 0.0701 → 0.0643 —
  the two biases partly cancel), and full-frame went x1.45 → **x1.65** (real
  0.0449 → 0.0392, generated unchanged; full frame gets the erosion fix only).
  The two geometries now agree to 4 % where before they disagreed by 10 %. The
  SSH grid-scale energy excess is unchanged and stands.


- **The store's `ocean_mask` misses islands — trust the data, not the mask**
  (found 2026-08-27, from someone recognising Isla de la Juventud in an EKE
  map). On gom_nemo, **143 pixels** are called ocean by `ocean_mask` but are an
  exact `0.0` in every channel of every frame in the record. They fall in 12
  blobs; the largest is **102 px ≈ 1630 km²**, which is the size and position
  of Isla de la Juventud, and the rest are keys and islets. Because the mask
  does not know they are land, `erode_mask` never touches them, `np.gradient`
  runs straight across their coastline, and the real EKE map carried bright
  closed rings around them — sitting right next to the properly eroded islands,
  which made them read as a physical feature rather than a masking hole.

  `Samples` now derives land from the data itself: `all(real_a == 0)` across
  channels, per sample. That test is exact — the loader writes the same `0.0`
  into every channel on land, and no ocean pixel is `0.0` in all of them — and
  it needs no metadata, so it also covers a patch npz written before
  `mask_full` existed. The stored mask is unioned in, never trusted alone.
  Worth **4.8 %** of the full-frame real EKE mean and **3.7 %** at patch level,
  from 0.06 % of the pixels: full frame x1.65 → **x1.73**, patch x1.58 →
  **x1.64**.

- **The model draws coastlines.** The bright closed rings in the *generated*
  EKE map are not a masking artifact and are not maskable: they are what the
  model drew. `prior_genda_masked` is the arm that keeps land in the training
  distribution (`reject_land: false`, `loss_reduction: masked_mean`), and what
  it learned includes the coastline itself — closed curves in open ocean across
  which SSH, SST and CHL all step in a single cell. In one full-frame sample the
  step is 0.19 m of SSH over one 4 km cell, implying **4.9 m/s** of geostrophic
  velocity against a 0.06 m²/s² background.

  `fig4` reports the rate as *one-cell steps in every channel at once, per 10k
  ocean pixels* (`step_rate_gen` / `step_rate_real` in `summary.csv`). At step
  1,130,000: **5.6 vs 0.2 (x23) in patch geometry**, and 0.4 vs 0.3 (x1.2) at
  full frame. Read that full-frame ratio as "this statistic does not separate
  here", not as absence — see the next bullet for why the full-frame real side
  is not a clean control.

  Three other discriminators were tried and do **not** work, so do not
  re-derive them: the model's drawn land is not flat (only 0.05 % of generated
  pixels are quiet in all channels, against 36 % of real pixels being exact
  land zeros); a joint-gradient score calibrated on the real 99.9th percentile
  flags real *fronts* more often than generated edges (1.7 vs 0.6 blobs per
  frame); and the implied-speed tail does not separate at full frame either.
  What works is one-cell-ness, because a front is resolved over two or three
  cells and a coastline is not.

  Suggestive but not established: across the 7 trajectory checkpoints plus the
  `s_churn=0` sweep arm, the patch step rate and the SSH 10–40 km PSD ratio move
  together (Spearman 0.78, n=8), and `s_churn=0` is the extreme of both — 9.4x
  SSH grid-scale power and a 33.5x step rate against ~5x and ~23x for the
  stochastic sampler. Within the 7 checkpoints alone the correlation is only
  0.67, so this rests largely on the one sweep arm. Worth a wider sampler sweep
  before it is treated as the mechanism behind the SSH grid-scale excess.

  It does **not** drive the energy scalar. Dropping the top 1 % of
  joint-gradient pixels moves the full-frame EKE ratio from 1.65 to *1.69* —
  the wrong way. The excess is broadly distributed; the drawn coastlines
  dominate the *map* and are a separate, smaller defect. This is the single
  clearest thing the arm A / arm B pair has produced so far, and it is an
  argument about land handling, which is exactly what the pair isolates.

- **The store has bad pixels that are neither land nor signal.** After masking
  land from the data, the real full-frame side still has **37 isolated
  locations** (97 pixels over 48 frames) implying up to **6.7 m/s** of
  geostrophic velocity. At the worst one the SSH anomaly field is a uniform
  −0.195 m with a five-pixel patch at −0.011 to −0.09 — a 0.19 m jump over one
  cell in an otherwise flat field. They are not land (not zero), not at the
  coast (median 18 px away), and they are why the full-frame real side cannot
  be used as a clean control for the drawn-coastline statistic above. They move
  the tails, not the mean. Not fixed — flagged.


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

### Spectra as a function of training step

Once several checkpoints have been evaluated this way, `step_trajectory.py`
puts them on one axes and reduces each to a scalar, which is what actually
answers *keep training or stop*:

```bash
python -m eval.step_trajectory --run .../runs/prior_genda_masked
```

It draws no samples and opens no store — it walks the run directory, reads the
training step out of each `samples_<size>.npz`'s `meta`, and produces per
geometry:

| figure | what it shows |
|---|---|
| `fig_spectra_vs_step.png` | radial PSD per variable, real in black, one curve per checkpoint coloured by step |
| `fig_ratio_vs_step.png` | the same as generated/real, where "too smooth" is unambiguously below 1 |
| `fig_bands_vs_step.png` | the scalars — band ratios and spectral distance against training step |

plus `trajectory.csv`. The distance is RMS of `log10(gen/real)` over the
resolved band (2dx to the training-patch span), so a factor-2 deficit and a
factor-2 excess score the same and a perfect match is 0.

**It filters by sampler, not by directory name.** A run that also holds a
sampler sweep has several npz files at the *same* training step differing only
in `s_churn`; plotting those against step would read as a training effect and
be wrong. The script keeps only the modal sampler/weights configuration — a
sweep visits each setting once, the training ladder shares one setting across
every checkpoint — and says which points it dropped.

Both band-ratio panels are log-scaled, because a ratio is a log quantity: on a
linear axis gulfstream's `tau` channels (250x the real submesoscale power,
since the real 4 dx spike is an artefact of the upstream forcing interpolation
that the prior cannot and should not reproduce) squash every ocean-state
channel onto the 1.0 line.
