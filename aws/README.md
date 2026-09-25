# `aws/` — cloud deployment scaffolding

Everything needed to run `hycom_sbda` on a single multi-GPU AWS instance, with the
zarr stores either **streamed from S3** or **staged to local NVMe**. No conda; deps come
from the image or a Deep Learning AMI. Run all commands from the **repo root**, not from
inside `aws/`.

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA/torch image with pinned deps + source. Build context = the repo root. |
| `launch.sh` | Single-node DDP training launcher: `torchrun`, auto-detects all visible GPUs. |
| `sample.sh` | Single-GPU ensemble generation → `ensembles.npz`. |
| `eval_prior.sh` | Unconditional prior eval (the `eval.slurm` job): samples + figures. |
| `common.sh` | Sourced by the launchers: env defaults, S3 stage-in, result sync-out. |

## How configuration flows

Every path in `diffusion/configs/*.yaml` and `diffusion/datasets/*.yaml` is written
`${VAR:-<on-prem path>}`, expanded by `diffusion/config.py:expandvars` at load time. So the
**same YAML** runs on the COAPS cluster and on AWS — with no env vars set, every placeholder
resolves to exactly the path that used to be hardcoded, which is why `sbatch train.slurm`
still works unchanged.

| Env var | Config field | Fallback when unset |
|---|---|---|
| `HYCOM_STORE` | `datasets/gom_nemo.yaml:stores`, `configs/*.yaml:zarr_path` | `/unity/f1/ozavala/DATA/GOFFISH/GOES/datasets/gom_nemo_diffusive.zarr` |
| `GULFSTREAM_MANIFEST` | `datasets/gulfstream.yaml:manifest` | `/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/datasets/gulfstream_manifest.json` |
| `OUT_DIR` | `configs/*.yaml:out` (run dir root) | `/unity/f1/ozavala/DATA/JorgeVelasco/runs`, or `runs` for `ssh_128` |
| `CACHE_DIR` | `norm_cache`, `clim_cache` | `diffusion` (the source tree — override in a container) |

Launcher-only vars, not seen by Python:

| Env var | Meaning |
|---|---|
| `CONFIG` | **required** by `launch.sh` — path to an experiment config |
| `CKPT` | **required** by `sample.sh` — run dir or `ckpt.pt` |
| `NGPU` | process count; defaults to every GPU `nvidia-smi -L` reports |
| `BATCH`, `STEPS`, `MAX_DAYS`, `NUM_WORKERS`, `OUT` | passed through to `diffusion.train` |
| `ZARR_PATH`, `DATASET`, `K_MEMBERS`, `DAYS`, `SPLIT`, `SAMPLE_OUT` | passed through to `diffusion.sample` |
| `STAGE_DIR` | if set, `s3://` stores are synced here once before launch |
| `OUT_S3` | if set, `OUT_DIR` is restored from and synced back to this prefix |
| `SYNC_EVERY` | seconds between background syncs (default 900; `0` disables) |

For a **private** bucket, set `storage_options` in the config (e.g.
`storage_options: {region_name: us-east-1}`); it is forwarded to fsspec/s3fs. With an
instance **IAM role**, no keys are needed.

## RAM sizing — read this before the first real run

Each DDP rank builds its **own** `ZarrWindowDataset` and preloads every needed channel's
full time extent into `float32` RAM (`diffusion/data.py:_load_zscore` / `_load_anomaly`).
This is host RAM, not VRAM:

| Config | Per channel | Channels | **Per rank** | 8 ranks |
|---|---|---|---|---|
| `prior_gulfstream` (4367 hourly × 576×936) | 9.4 GB | 3 | **28 GB** | ~224 GB |
| `prior_genda_masked` (1872 daily × 464×528) | 1.8 GB | 3 | **5.4 GB** | ~43 GB |
| `ssh_128` (same grid) | 1.8 GB | 6 | **11 GB** | ~86 GB |

A `p4d.24xlarge` (1152 GB) absorbs all of these. The real cost is elsewhere: pointed
straight at `s3://`, that is **N independent full downloads of the store at every start**.

So for anything but a smoke test, **set `STAGE_DIR`** — one `aws s3 sync` to instance NVMe,
shared by all ranks, and a near-instant restart after a spot interruption. Direct `s3://`
streaming stays supported and is the right choice for smoke tests.

## Batch semantics — read this before scaling up

`batch` in the config is **per GPU**; `train.py` derives `global_batch = batch × world_size`
and feeds it to the EMA halflife rule. The GenDA-parity configs are calibrated for a
**global batch of 64** (`batch: 32` × 2 ranks on-prem). Running one unchanged on 8 GPUs
gives a global batch of 256 — a *different training recipe*, not a faster version of the
same one. Set `BATCH=8` on 8 GPUs to hold parity. `launch.sh` warns when it detects the
mismatch.

## 1. Stage the dataset to S3 (once)

The splits sidecars are read from **inside** the store directory, and the gulfstream
manifest names its member stores with paths **relative to its own directory** — so sync
whole directories, never just the arrays:

```bash
# single-store dataset
aws s3 sync /unity/f1/ozavala/DATA/GOFFISH/GOES/datasets/gom_nemo_diffusive.zarr \
            s3://my-bucket/gom_nemo_diffusive.zarr

# multi-store family: the manifest AND its seven member stores, together
aws s3 sync /unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/datasets \
            s3://my-bucket/gulfstream
```

## 2. Build and run

```bash
cd /path/to/hycom_sbda
docker build -f aws/Dockerfile -t hycom_sbda .
```

```bash
docker run --gpus all --ipc=host --shm-size=16g \
  -e HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr \
  -e CONFIG=diffusion/configs/prior_genda_masked.yaml \
  -e OUT_DIR=/mnt/runs -e CACHE_DIR=/mnt/runs/cache -v /mnt/runs:/mnt/runs \
  -e OUT_S3=s3://my-bucket/runs/prior_genda_masked \
  -e BATCH=8 \
  hycom_sbda
```

- **`--ipc=host` is required.** Docker's default private IPC namespace starves the
  DataLoader workers of the shared memory and semaphores they need to bootstrap — they hang
  *silently* at startup, with no crash, no error and no worker PIDs, **even with a generous
  `--shm-size`**. This flag is the fix.
- **`--shm-size=16g` is also required**: DataLoader workers use shared memory; the 64 MB
  default crashes them.
- **Mount `OUT_DIR`** so checkpoints survive the container, and keep `CACHE_DIR` inside it
  so the climatology fit is paid once rather than on every run.
- **`HYCOM_STORE` is resolved inside the container.** An `s3://` URL needs no mount. A host
  path — including a FUSE mount like `/mnt/s3` from `mount-s3` — is invisible unless you
  bind-mount it, and a FUSE mount additionally needs mount propagation
  (`-v /mnt/s3:/mnt/s3:ro,rslave`, or `mount --make-shared /mnt/s3` on the host) or the
  container sees an *empty* directory. Streaming from `s3://` avoids all of it.

### Staged, with S3 persistence — the recommended shape for a real run

```bash
docker run --gpus all --ipc=host --shm-size=16g \
  -e GULFSTREAM_MANIFEST=s3://my-bucket/gulfstream/gulfstream_manifest.json \
  -e STAGE_DIR=/mnt/nvme -v /mnt/nvme:/mnt/nvme \
  -e CONFIG=diffusion/configs/prior_gulfstream.yaml \
  -e OUT_DIR=/mnt/runs -e CACHE_DIR=/mnt/runs/cache -v /mnt/runs:/mnt/runs \
  -e OUT_S3=s3://my-bucket/runs/prior_gulfstream \
  -e BATCH=8 \
  hycom_sbda
```

`OUT_S3` gives you resume for free: `common.sh:restore_out` pulls the previous run back down
before launch, and `diffusion/train.py` resumes implicitly from `<out>/ckpt.pt`. Results are
synced back every `SYNC_EVERY` seconds and once more on exit — including a crash or a spot
reclamation.

### GPU architecture

Check what the instance actually has (`nvidia-smi -L`) before building. The `FROM` tag must
carry kernels for that arch, or every CUDA op fails at runtime:

| Instance | GPU | Compute cap. | Base image |
|---|---|---|---|
| p4d | A100 | sm_80 | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime` (**default**) |
| p5 | H100 | sm_90 | same cu126 tag, or cu128 |
| p6 | B200 | sm_100 | CUDA **12.8+** tag |
| g7 | RTX PRO Blackwell | sm_120 | CUDA **12.8+** tag |

```bash
docker build -f aws/Dockerfile \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -t hycom_sbda .
```

The Dockerfile strips the `torch==` / wheel-index lines from `requirements.txt` before
installing, so the base image's arch-matched torch is never overwritten by the cu126 pin.
The build prints `torch.cuda.get_arch_list()` — confirm your arch is listed.

## 3. Validate a descriptor before spending GPU time

```bash
docker run --rm -e HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr hycom_sbda \
  python -m diffusion.dataset_spec --selftest diffusion/datasets/gom_nemo.yaml
```

## 4. Sample

```bash
docker run --gpus all --ipc=host --shm-size=16g \
  -e HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr \
  -e CKPT=/mnt/runs/prior_genda_masked -v /mnt/runs:/mnt/runs \
  -e OUT_DIR=/mnt/runs -e CACHE_DIR=/mnt/runs/cache \
  -e K_MEMBERS=24 -e SPLIT=val \
  hycom_sbda bash aws/sample.sh
```

A checkpoint records the store path as it **resolved at training time**, so one trained
on the cluster and sampled here needs `ZARR_PATH=` (or `DATASET=`) to relocate it —
`HYCOM_STORE` alone is not enough, because `sample.py` reads the checkpoint's config rather
than the YAML.

## 5. Unconditional prior

`bash aws/eval_prior.sh` is the Docker form of `eval.slurm`: kernel selftest,
unconstrained samples (`samples_full.npz`, `samples_128.npz`), then the five
figures. `sample.sh` is a different job. The image has no `scipy` or
`matplotlib`; the script installs them on first use. One GPU, and the ~66 GB
host-RAM preload of the Gulf Stream record.

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e GULFSTREAM_MANIFEST=s3://my-bucket/gulfstream/gulfstream_manifest.json \
  -e STAGE_DIR=/mnt/nvme -v /mnt/nvme:/mnt/nvme \
  -e CKPT=/mnt/runs/prior_gulfstream/best.pt \
  -e OUT_DIR=/mnt/runs -e CACHE_DIR=/mnt/runs/cache -v /mnt/runs:/mnt/runs \
  -e OUT_S3=s3://my-bucket/runs/prior_gulfstream \
  hycom_sbda bash aws/eval_prior.sh
```

`gen_prior` re-reads the dataset descriptor, so `GULFSTREAM_MANIFEST` (and
`STAGE_DIR`, if you stage) has to be set again. Results land in
`<ckpt dir>/eval_step<N>/`. A later checkpoint skips the store load with
`REAL_FROM=<that dir>`. `SKIP_FIGS=1` writes the npz files only.

## 2b. Deep Learning AMI (no Docker)

```bash
cd hycom_sbda
pip install -r requirements.txt      # torch preinstalled on the DLAMI
export HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr
export OUT_DIR=/mnt/runs CACHE_DIR=/mnt/runs/cache
CONFIG=diffusion/configs/prior_genda_masked.yaml bash aws/launch.sh
```

## Notes / scaling

- **Single-node by design.** This scaffolding targets one multi-GPU box; multi-node (AWS
  Batch / ParallelCluster over EFA) is intentionally not included.
- `launch.sh` deliberately does **not** pin `CUDA_VISIBLE_DEVICES`. `train.slurm` pins it to
  `1,2,3` because on that one COAPS node SLURM gres does not isolate GPUs and GPU 0 is held
  by another process. On a dedicated instance the same line would throw GPUs away.
- No ECR push script, no CI, no image-tagging convention — add them when a second consumer
  of the image exists.
