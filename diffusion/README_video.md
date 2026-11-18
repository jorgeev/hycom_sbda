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
  `use_ocean_mask`), per-frame scalar labels (`label_vars`).
- **Two experts**: one network trained on sigma log-uniform in `[0.001, 1]`,
  another on `[0.1, 200]`; the sampler uses the second above `sigma_divide`
  (0.5). Each expert is an ordinary run with its own config and output dir.

Channels are three lists. A store variable in `target` is generated; in
`cond_vars` it is a clean per-frame input field; in `label_vars` it is reduced
to one ocean-mean scalar per frame and fed through the embedding. Wind stress is
generated in the template; moving `tau_x, tau_y` from `target` to `label_vars`
turns it into GenLLC4320-style scalar conditioning.

## Run it

```bash
conda activate datorch && cd <repo>

# selftests (CPU; the last one reads the store)
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
`batch: 3` is about what the full-size network fits on an 80 GB A100 at
5 frames x 256^2, so 48 is reached with 16 GPUs, or 8 GPUs and `accum: 2`, or
one GPU and `accum: 16`. `aws/launch_video.sh` prints the product and warns when
it is not 48.

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
- Scalar labels at full frame are full-frame means, a different statistic from
  the patch means seen in training.

## Differences from GenLLC4320 worth knowing

| | GenLLC4320 | here |
|---|---|---|
| frames | 12-hour means of tide-removed, inertial-low-passed fields | hourly snapshots as stored |
| forcing | patch-mean wind stress and shortwave as labels | wind stress generated (switchable to labels) |
| temporal attention | output replaces the features | added to them (`temporal_attn_residual: false` restores the original) |
| normalisation | per-sample mean removed from SST/SSS | per-pixel monthly climatology (`norm_mode: anomaly`) |
| sampler | VP predictor-corrector | EDM Heun by default, VP available |
| weights used to sample | raw | EMA |
