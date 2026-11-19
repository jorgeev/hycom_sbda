# Multi-frame prior (`arch: genllc_video`)

A second, parallel training path for priors that denoise a short **window of
frames** jointly, following GenLLC4320 (Martin et al., the LLC4320 follow-up to
GenDA). It lives beside the single-frame code and does not change it: the only
existing file touched is `diffusion/config.py`, which gained fields whose
defaults describe the single-frame model.

| File | What it is |
|---|---|
| `networks_video.py` | `VideoUNet`: a self-contained port of GenLLC4320's ADM video UNet, plus `CalendarEmbedding` |
| `edm_video.py` | `VideoPrecond`, `VideoEDMLoss` (log-uniform sigma), `ExpertSwitch`, `heun_sampler_video` |
| `data_video.py` | `ZarrVideoDataset`: `n_frames`-step windows on top of `ZarrWindowDataset` |
| `train_video.py` | training loop (DDP, gradient accumulation, `--init-from`) |
| `sample_video.py` | prior ensembles from one network or an expert pair |
| `configs/video_gulfstream_small.yaml` | **the template** -- every knob, commented; small-sigma expert |
| `configs/video_gulfstream_large.yaml` | large-sigma expert |
| `configs/video_smoke.yaml` | tiny model for tests |
| `../train_video.slurm`, `../train_video_lanes.slurm` | local drivers |
| `../aws/launch_video.sh`, `../aws/sample_video.sh` | AWS drivers |

## What the model is

- **State** `(B, T, C, H, W)`: `T = n_frames` hourly snapshots, `frame_stride`
  hours apart, at one crop. Convolutions and spatial attention run per frame;
  frames interact only through per-pixel attention across time.
- **Network**: GenLLC4320's -- 5 levels, 128/256/256/384/512 channels, 3 blocks
  per level, spatial attention at 32 and 16 px, temporal at 64, 32 and 16 px,
  64 channels per head. About 160 M parameters with the template.
- **Inputs beyond the noisy state**, all optional and all from the YAML:
  calendar channels (`use_calendar`), per-frame input fields (`cond_vars`,
  `use_ocean_mask`), per-frame scalar labels (`label_vars`, `tidal_labels`).
- **Two experts**: one network trained on sigma log-uniform in `[0.001, 1]`,
  another on `[0.1, 200]`; the sampler uses the second above `sigma_divide`
  (0.5). Each expert is an ordinary run with its own config and output dir.

Channels are three lists. A store variable in `target` is generated; in
`cond_vars` it is a clean per-frame input field; in `label_vars` it is reduced
to one ocean-mean scalar per frame and fed through the embedding. Wind stress is
generated in the template; moving `tau_x, tau_y` from `target` to `label_vars`
turns it into GenLLC4320-style scalar conditioning.

### Tidal phase labels

The frames are hourly snapshots that contain the tide (about half of the
anomaly-normalised SSH channel), and the calendar channels carry solar time
only, so without help the model has to invent the phase of the lunar tide for
every sample. `tidal_labels` gives it that phase as scalar labels:

```yaml
tidal_labels: [K1, O1, P1, Q1, M2, S2, N2, K2]   # empty (the default) = none
tidal_epoch: 1483232400                          # 2017-01-01T01:00:00Z, unix seconds
```

- **What the network gets.** For every frame, cos and sin of each listed
  constituent's phase at the frame's time, `speed * (t - tidal_epoch)`, divided
  by `sqrt(K)` so the vector has unit length (`diffusion/tides.py`). They are
  appended after the `label_vars` scalars and go through the same embedding.
  Eight constituents are 16 labels and 8,192 extra weights; GPU memory and step
  time do not change.
- **Nothing is read from the store.** The labels depend on the timestamp alone,
  so they are the same for every crop and for the full frame, and they are
  known for any date. `sample_video --labels zero` therefore zeroes the
  `label_vars` scalars only.
- **The order and the epoch match the harmonic fit** written by
  `extract_hycom_tides.py` (`K1_cos, K1_sin, O1_cos, ...`, phase origin at its
  first snapshot). Its fitted tide is `coefficient maps x labels x sqrt(K)`;
  `python -m diffusion.tides --selftest` checks that where the files exist.
- **SSH is not changed.** The tide is still part of the generated field; the
  labels only say where in the cycle each frame is.
- **Configs.** `video_gulfstream_tide_{small,large}.yaml` are the template pair
  plus these keys, with their own output directories.
  `video_gulfstream_tide_nt_{small,large}.yaml` go one step further (below).

### Tide-free SSH (`ssh_nt`), 6 frames

Since 2026-11-19 the gulfstream stores also hold `ssh_nt`: `ssh` minus the
8-constituent harmonic fit (the coherent tide), subtracted on the native grid
before the regrid. Everything not phase-locked to those eight frequencies is
still in it: eddies, the front, incoherent internal tides, near-inertial motion.

`video_gulfstream_tide_nt_{small,large}.yaml` train on it. Relative to the tide
pair they change three keys:

| key | tide pair | tide_nt pair | why |
|---|---|---|---|
| `target` | `[ssh, sst, sss, uag, vag, tau_x, tau_y]` | `[ssh_nt, sst, sss, uag, vag, tau_x, tau_y]` | the tide is no longer generated; still 7 channels |
| `n_frames` | 5 | 6 | |
| `batch` | 3 | 2 | 6 frames x batch 3 does not fit on an 80 GB A100 (table under *Batch size*) |

- The SSH channel's normalisation is fit on `ssh_nt` itself: a new variable name
  is a new cache entry, so no stale statistics are reused.
- With `batch: 2` the global batch of 48 needs 24 GPU-rounds: 8 GPUs x
  `accum: 3`, 4 GPUs x `accum: 6`, or 1 GPU x `accum: 24`.
- `uag`/`vag` (and `ug`/`vg`) were derived from `ssh`, not `ssh_nt`, and still
  contain the tidal currents; the tidal labels still give the model the phase.
- The sampler writes `gen_ssh_nt` and `truth_ssh_nt`. Adding the fitted tide
  back to get full SSH is not built.

Things to know:

- A tide run starts from step 0. Checkpoints trained without the labels have no
  label layer; `--init-from` and the sampler's pair check refuse to mix them.
- P1 and K2 differ from K1 and S2 by 0.082 deg/hour, a 182.6-day beat, about
  the length of the gulfstream record. The training months cover 287 degrees of
  that relative phase and the validation months the other 72, which training
  never shows. The other pairs are well sampled (M2-S2 14.8 days, K1-O1 13.7,
  M2-N2 27.6). Listing six constituents without P1 and K2 avoids it.
- S2 turns exactly twice per solar day, so it repeats the second solar-time
  harmonic of the calendar channels. Harmless.
- `label_dropout > 0` zeroes the whole label vector of a frame, tidal phases
  included.
- These are fixed-speed harmonics with no nodal corrections, as in the fit.
  They are exact for that fit and drift from the real tide over years.

## Run it

```bash
conda activate datorch && cd <repo>

# selftests (CPU; the last one reads the store)
python -m diffusion.tides --selftest
python -m diffusion.networks_video --selftest
python -m diffusion.edm_video --selftest
python -m diffusion.train_video --selftest
python -m diffusion.data_video --selftest diffusion/configs/video_smoke.yaml --max-days 400

# smoke: both experts, then sample from the pair (minutes; --max-days shortens the preload)
CUDA_VISIBLE_DEVICES=2 python -m diffusion.train_video \
    --config diffusion/configs/video_smoke.yaml --max-days 400 --out runs/vs_small
CUDA_VISIBLE_DEVICES=2 python -m diffusion.train_video \
    --config diffusion/configs/video_smoke.yaml --max-days 400 --out runs/vs_large \
    --set expert=large --set train_sigma_min=0.1 --set train_sigma_max=200
CUDA_VISIBLE_DEVICES=2 python -m diffusion.sample_video --small runs/vs_small \
    --large runs/vs_large --split train --max-days 400 --windows 2 --k-members 2
# the same smoke run with tidal phase labels: add this to BOTH training commands
#     --set "tidal_labels=[K1,O1,P1,Q1,M2,S2,N2,K2]"

# the two experts side by side on the local node (one lane each)
sbatch train_video_lanes.slurm \
    "diffusion/configs/video_gulfstream_small.yaml:1:2 \
     diffusion/configs/video_gulfstream_large.yaml:1:3"

# on AWS (once per expert)
CONFIG=diffusion/configs/video_gulfstream_small.yaml bash aws/launch_video.sh
```

`--set key=value` overrides any config field from the command line (YAML
values: `--set "target=[ssh, sst]"`, `--set n_frames=3`).

### Batch size

`global batch = batch (per GPU) x GPUs x accum`. The template is written for a
global batch of 48 and its `steps`, `warmup` and EMA half-life assume it.
`batch: 3` is the most the full-size network fits on an 80 GB A100 at
5 frames x 256^2, so 48 is reached with 16 GPUs, or 8 GPUs and `accum: 2`, or
one GPU and `accum: 16`. `aws/launch_video.sh` prints the product and warns when
it is not 48.

Measured on one A100-80GB, template network, bf16 (2026-11-18):

| setting | peak memory | time per step |
|---|---|---|
| batch 3 | 69 GB | 0.97 s |
| batch 4 | out of memory | |
| batch 4, `grad_checkpoint: true` | 20 GB | 1.65 s |
| 2 ranks x batch 2 x accum 2 (DDP) | 48 GB per GPU | 1.36 s |

That is 0.32 s per sample per GPU, about 900 A100-hours for the 10 M samples of
one expert. A checkpoint (weights, EMA, optimiser) is 2.6 GB.

How it scales (synthetic tensors, same network; real training adds about 1 GB
for the EMA copy):

| frames x variables | largest batch per A100 | with `grad_checkpoint: true` |
|---|---|---|
| 5 x 7 or 8 | 3 (68.5 GB, 0.32 s/sample) | 12 (55.6 GB, 0.40 s/sample) |
| 6 x 7 or 8 | 2 (55.2 GB, 0.38 s/sample) | 12 (66.4 GB, 0.47 s/sample) |
| 12 x 8 | 1 (55.3 GB, 0.75 s/sample) | 6 (66.4 GB, 0.92 s/sample) |

GPU memory follows the number of FRAMES (about 4.4 GB per sample per frame
without checkpointing); the number of variables does not move it. Variables
cost host RAM instead: 9.4 GB per rank per hourly variable.

The launchers set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without
it the 12-frame batch-6 case fails from allocator fragmentation with 22 GB
still free.

Host RAM is per rank and per variable (9.4 GB for each hourly gulfstream
variable), independent of the number of frames.

### Fine-tuning at another patch size

```bash
python -m diffusion.train_video --config <new>.yaml --init-from <run>/ckpt.pt
```

Weights are stored by level, not by resolution, so a checkpoint loads at any
patch with no renaming. The attention lists are absolute resolutions, so scale
them with the patch to keep attention on the same levels -- at 512:
`attn_resolutions: [64, 32]`, `temporal_attn_resolutions: [128, 64, 32]`.
`--init-from` checks this (and the channel lists, widths and `n_frames`) before
loading anything and prints the lists to use. `n_frames` cannot change in a
fine-tune: the temporal attention's position table has that length. Larger
patches also need more GPU memory (`grad_checkpoint: true` trades speed for it)
and noise ranges shifted upward, as in `prior_gulfstream_256_anneal_ft_hisig512.yaml`.

## Sampling output

`sample_video` writes one `.npz` (physical units):

| key | shape |
|---|---|
| `gen_<var>` | `(n_windows, K, T, H, W)` |
| `truth_<var>` | `(n_windows, T, H, W)` |
| `starts`, `time_unix`, `pos`, `mask` | `(n,)`, `(n, T)`, `(n, 2)`, `(n, H, W)` |
| `target_vars`, `config_small`, `config_large`, `meta` | JSON strings |

`--sampler heun` (default) is the EDM Heun sampler with the checkpoint's
settings; `--sampler vp` runs `sda.vpsde.vp_sample`, GenLLC4320's sampler
family, through the same expert switch. `--size full` uses the largest region
that divides by the network's downsample factor (576x928 of 576x936 for five
levels); there is no tiling.

## Not built yet

- Evaluation of multi-frame samples (`eval/` reads single-frame files only).
- Guided assimilation over windows (`sda/`). `ExpertSwitch` already exposes the
  `d_fn(x, sigma)` interface `sda/vpsde.py` and `sda/guidance.py` consume.
- Tiled (multidiffusion) inference for domains or windows larger than one pass.
- `label_vars` scalars at full frame are full-frame means, a different
  statistic from the patch means seen in training (tidal phase labels do not
  depend on the crop).
- Adding the fitted tide back to a generated `ssh_nt` to get full SSH (needed to
  compare with stored `ssh` or to assimilate SWOT, which sees the tide). It is
  `coefficient maps x tidal labels x sqrt(K)`, but the maps exist only on the
  raw 612x1001 grid, not on the store grid.

## Differences from GenLLC4320 worth knowing

| | GenLLC4320 | here |
|---|---|---|
| frames | 12-hour means of tide-removed, inertial-low-passed fields | hourly snapshots as stored |
| forcing | patch-mean wind stress and shortwave as labels | wind stress generated (switchable to labels) |
| tide | removed before training | kept in the fields; its phase optionally given as labels (`tidal_labels`) |
| temporal attention | output replaces the features | added to them (`temporal_attn_residual: false` restores the original) |
| normalisation | per-sample mean removed from SST/SSS | per-pixel monthly climatology (`norm_mode: anomaly`) |
| sampler | VP predictor-corrector | EDM Heun by default, VP available |
| weights used to sample | raw | EMA |
