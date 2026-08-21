#!/usr/bin/env bash
# Single-node DDP training launcher for AWS (e.g. one p4d.24xlarge = 8xA100).
# No conda bootstrap -- run inside the image (aws/Dockerfile), or on a Deep
# Learning AMI after `pip install -r requirements.txt`.
#
# Usage:
#   HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr \
#   CONFIG=diffusion/configs/prior_genda_masked.yaml bash aws/launch.sh
#
#   GULFSTREAM_MANIFEST=s3://my-bucket/gulfstream/gulfstream_manifest.json \
#   STAGE_DIR=/mnt/nvme OUT_S3=s3://my-bucket/runs/prior_gulfstream \
#   CONFIG=diffusion/configs/prior_gulfstream.yaml BATCH=8 bash aws/launch.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # -> repo root; `python -m diffusion.*` needs it
source aws/common.sh

CONFIG="${CONFIG:?set CONFIG, e.g. diffusion/configs/prior_gulfstream.yaml}"
[[ -f "$CONFIG" ]] || { echo "[launch] no such config: $CONFIG" >&2; exit 2; }

# ---------------------------------------------------------------------------
# GPU / process count
# ---------------------------------------------------------------------------
# Deliberately NOT setting CUDA_VISIBLE_DEVICES. train.slurm pins it to "1,2,3"
# because on that one COAPS node SLURM gres does not isolate GPUs and GPU 0 is
# permanently held by another process. On a dedicated instance the same line
# would simply throw GPUs away. Do not reintroduce it here.
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}"

# ---------------------------------------------------------------------------
# Batch semantics -- read this before scaling up
# ---------------------------------------------------------------------------
# `batch` in the config is PER GPU; train.py derives global_batch = batch *
# world_size and feeds it to the EMA halflife rule. The GenDA-parity configs are
# calibrated for a GLOBAL batch of 64 (batch: 32 x 2 ranks on-prem). Running one
# unchanged on 8 GPUs gives a global batch of 256 -- a different training recipe,
# not a faster version of the same one. Set BATCH=8 on 8 GPUs to hold parity.
if [[ -z "${BATCH:-}" ]] && [[ "$NGPU" -gt 2 ]] && [[ "$CONFIG" == *prior_* ]]; then
  echo "[launch] WARNING: $CONFIG targets a global batch of 64, but NGPU=$NGPU"
  echo "[launch]          with the config's per-GPU batch. Set BATCH=$((64 / NGPU))"
  echo "[launch]          to preserve parity, or continue knowingly."
fi

stage_in
restore_out
trap on_exit EXIT
start_background_sync

echo "[launch] CONFIG=$CONFIG  NGPU=$NGPU  BATCH=${BATCH:-<config>}"
echo "[launch] HYCOM_STORE=${HYCOM_STORE:-<config default>}"
echo "[launch] GULFSTREAM_MANIFEST=${GULFSTREAM_MANIFEST:-<config default>}"
echo "[launch] OUT_DIR=$OUT_DIR  CACHE_DIR=$CACHE_DIR  OUT_S3=${OUT_S3:-<none>}"

# --standalone picks a free rendezvous port. train.slurm hardcodes 29517, which
# would collide between two containers sharing the host network.
# `|| rc=$?` rather than a bare call: `set -e` would abort the script the
# instant torchrun returned non-zero, skipping the diagnostic below.
rc=0
torchrun --standalone --nproc_per_node="$NGPU" \
  -m diffusion.train --config "$CONFIG" \
  ${BATCH:+--batch "$BATCH"} \
  ${STEPS:+--steps "$STEPS"} \
  ${MAX_DAYS:+--max-days "$MAX_DAYS"} \
  ${NUM_WORKERS:+--num-workers "$NUM_WORKERS"} \
  ${OUT:+--out "$OUT"} || rc=$?

# Propagate failure. torchrun surfaces a dead rank as a non-zero
# ChildFailedError; swallowing it once let a 3%-trained checkpoint be consumed
# as a finished baseline, fabricating a result that had to be retracted.
if [[ $rc -ne 0 ]]; then
  echo "[launch] Training FAILED (exit $rc) -- checkpoint is NOT usable" >&2
  exit $rc
fi
echo "[launch] Training completed"
