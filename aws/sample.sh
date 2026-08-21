#!/usr/bin/env bash
# Single-GPU ensemble generation on AWS. Writes <ckpt-dir>/ensembles.npz, the
# integration seam to any downstream evaluation.
#
# Usage:
#   HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr \
#   CKPT=/mnt/runs/prior_genda_masked bash aws/sample.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source aws/common.sh

CKPT="${CKPT:?set CKPT to a run directory or a ckpt.pt path}"

stage_in
trap on_exit EXIT

# A checkpoint records the store path as it RESOLVED at training time, so one
# trained on the cluster and sampled here needs the location overridden. The env
# vars alone are not enough: they are read from the YAML, and sample.py reads
# the checkpoint's config instead.
echo "[sample] CKPT=$CKPT  OUT_S3=${OUT_S3:-<none>}"
python -u -m diffusion.sample --ckpt "$CKPT" \
  ${ZARR_PATH:+--zarr-path "$ZARR_PATH"} \
  ${DATASET:+--dataset "$DATASET"} \
  ${K_MEMBERS:+--k-members "$K_MEMBERS"} \
  ${DAYS:+--days "$DAYS"} \
  ${SPLIT:+--split "$SPLIT"} \
  ${SAMPLE_OUT:+--out "$SAMPLE_OUT"}
