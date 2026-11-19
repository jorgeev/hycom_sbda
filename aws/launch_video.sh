#!/usr/bin/env bash
# Single-node DDP launcher for the MULTI-FRAME prior (arch: genllc_video) on AWS.
# The video twin of aws/launch.sh: same env vars, same S3 staging / resume /
# sync-out (aws/common.sh), but it runs diffusion.train_video and understands
# gradient accumulation. One expert per invocation -- run it once with the
# small-sigma config and once with the large-sigma one (separate instances, or
# one after the other; they are independent runs).
#
# Usage (inside the image; see aws/Dockerfile for the required docker flags):
#   GULFSTREAM_MANIFEST=s3://my-bucket/gulfstream/gulfstream_manifest.json \
#   STAGE_DIR=/mnt/nvme OUT_S3=s3://my-bucket/runs/video_gulfstream_small \
#   CONFIG=diffusion/configs/video_gulfstream_small.yaml bash aws/launch_video.sh
#
#   docker run --gpus all --ipc=host --shm-size=16g -e CONFIG=... -e ... \
#     hycom_sbda bash aws/launch_video.sh
#
# Extra env vars on top of aws/launch.sh's:
#   ACCUM          gradient-accumulation rounds per optimiser step
#   GLOBAL_BATCH   the global batch the recipe is written for (default 48)
#   INIT_FROM      checkpoint to seed a fine-tune from (--init-from)
#   SET            extra config overrides, space separated: SET="lr=1e-4 n_frames=3"
set -euo pipefail
cd "$(dirname "$0")/.."          # -> repo root; `python -m diffusion.*` needs it
source aws/common.sh

CONFIG="${CONFIG:?set CONFIG, e.g. diffusion/configs/video_gulfstream_small.yaml}"
[[ -f "$CONFIG" ]] || { echo "[launch] no such config: $CONFIG" >&2; exit 2; }

# Not setting CUDA_VISIBLE_DEVICES: on a dedicated instance every GPU is ours
# (see the note in aws/launch.sh).
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}"

# Avoid allocator fragmentation near the memory limit. With the default
# allocator a 12-frame, batch-6, grad_checkpoint run failed on an 80 GB A100
# with 54.6 GB in use and 22.8 GB "reserved but unallocated"; with expandable
# segments it runs at 66.4 GB (measured 2026-11-19). Respects a value set by
# the caller.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Batch semantics -- read this before scaling up
# ---------------------------------------------------------------------------
# global batch = batch (per GPU) x NGPU x accum. The video configs are written
# for a GLOBAL batch of 48 (GenLLC4320's), and `steps`, `warmup` and the EMA
# half-life are all calibrated to it. A different product is a different recipe,
# not a faster version of the same one, so say what this launch amounts to.
read -r CFG_BATCH CFG_ACCUM < <(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print(c.get("batch", 16), c.get("accum", 1))
PY
)
EFF_BATCH="${BATCH:-$CFG_BATCH}"
EFF_ACCUM="${ACCUM:-$CFG_ACCUM}"
EFF_GLOBAL=$(( EFF_BATCH * NGPU * EFF_ACCUM ))
GLOBAL_BATCH="${GLOBAL_BATCH:-48}"
if [[ "$EFF_GLOBAL" -ne "$GLOBAL_BATCH" ]]; then
  echo "[launch] WARNING: batch $EFF_BATCH x $NGPU GPU(s) x accum $EFF_ACCUM = global batch $EFF_GLOBAL,"
  echo "[launch]          but the recipe targets $GLOBAL_BATCH."
  per=$(( EFF_BATCH * NGPU ))
  if (( GLOBAL_BATCH % per == 0 )); then
    echo "[launch]          Set ACCUM=$(( GLOBAL_BATCH / per )) to hold it, or continue knowingly."
  else
    echo "[launch]          $GLOBAL_BATCH is not a multiple of batch x NGPU = $per; adjust BATCH or NGPU."
  fi
fi

stage_in
restore_out
trap on_exit EXIT
start_background_sync

echo "[launch] CONFIG=$CONFIG  NGPU=$NGPU  BATCH=$EFF_BATCH  ACCUM=$EFF_ACCUM  -> global batch $EFF_GLOBAL"
echo "[launch] GULFSTREAM_MANIFEST=${GULFSTREAM_MANIFEST:-<config default>}"
echo "[launch] OUT_DIR=$OUT_DIR  CACHE_DIR=$CACHE_DIR  OUT_S3=${OUT_S3:-<none>}"

SET_ARGS=()
for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done

rc=0
torchrun --standalone --nproc_per_node="$NGPU" \
  -m diffusion.train_video --config "$CONFIG" \
  ${BATCH:+--batch "$BATCH"} \
  ${ACCUM:+--accum "$ACCUM"} \
  ${STEPS:+--steps "$STEPS"} \
  ${MAX_DAYS:+--max-days "$MAX_DAYS"} \
  ${NUM_WORKERS:+--num-workers "$NUM_WORKERS"} \
  ${OUT:+--out "$OUT"} \
  ${INIT_FROM:+--init-from "$INIT_FROM"} \
  "${SET_ARGS[@]}" || rc=$?

if [[ $rc -ne 0 ]]; then
  echo "[launch] Training FAILED (exit $rc) -- checkpoint is NOT usable" >&2
  exit $rc
fi
echo "[launch] Training completed"
