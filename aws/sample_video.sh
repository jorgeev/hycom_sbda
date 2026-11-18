#!/usr/bin/env bash
# Single-GPU prior ensembles of multi-frame windows on AWS -- the video twin of
# aws/sample.sh. Writes ensembles_video.npz next to the (large) expert unless
# SAMPLE_OUT says otherwise.
#
# Usage:
#   SMALL=/mnt/runs/video_gulfstream_small LARGE=/mnt/runs/video_gulfstream_large \
#     bash aws/sample_video.sh
#   SINGLE=/mnt/runs/some_single_network_run bash aws/sample_video.sh
#
# Optional: DATASET, ZARR_PATH (store relocation), K_MEMBERS, WINDOWS, SPLIT,
# SIZE (patch | full | N), SAMPLER (heun | vp), SAMPLER_STEPS, SAMPLE_OUT.
# The climatology cache is looked up under $CACHE_DIR (set by aws/common.sh),
# not at the path the checkpoint recorded when it was trained.
set -euo pipefail
cd "$(dirname "$0")/.."
source aws/common.sh

if [[ -n "${SINGLE:-}" ]]; then
  WHO=(--single "$SINGLE")
else
  : "${SMALL:?set SMALL and LARGE (run dirs or .pt files), or SINGLE}"
  : "${LARGE:?set SMALL and LARGE (run dirs or .pt files), or SINGLE}"
  WHO=(--small "$SMALL" --large "$LARGE")
fi

stage_in
trap on_exit EXIT

echo "[sample] ${WHO[*]}  OUT_S3=${OUT_S3:-<none>}"
python -u -m diffusion.sample_video "${WHO[@]}" \
  ${ZARR_PATH:+--zarr-path "$ZARR_PATH"} \
  ${DATASET:+--dataset "$DATASET"} \
  ${K_MEMBERS:+--k-members "$K_MEMBERS"} \
  ${WINDOWS:+--windows "$WINDOWS"} \
  ${SPLIT:+--split "$SPLIT"} \
  ${SIZE:+--size "$SIZE"} \
  ${SAMPLER:+--sampler "$SAMPLER"} \
  ${SAMPLER_STEPS:+--sampler-steps "$SAMPLER_STEPS"} \
  ${SAMPLE_OUT:+--out "$SAMPLE_OUT"}
