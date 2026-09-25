#!/usr/bin/env bash
# Unconditional prior evaluation inside the AWS image. Same job as eval.slurm
# (kernel selftest, unconstrained samples, diagnostics) with no conda and no
# SLURM. One GPU. Do not set CUDA_VISIBLE_DEVICES here: on a dedicated instance
# that throws cards away, and --gpus already decides which devices this
# container can see.
#
# The checkpoint stores the descriptor name, not the store path. gen_prior
# re-reads diffusion/datasets/*.yaml, which expands GULFSTREAM_MANIFEST (or
# HYCOM_STORE) from the environment. STAGE_DIR, when set, copies an s3://
# family onto local NVMe once and rewrites that variable before Python starts.
#
# scipy and matplotlib are not in the image. They are installed on first use
# and left alone when already present. SKIP_FIGS=1 draws the npz files only.
#
# Usage (inside the image, or: docker run ... hycom_sbda bash aws/eval_prior.sh):
#   GULFSTREAM_MANIFEST=s3://my-bucket/gulfstream/gulfstream_manifest.json \
#   STAGE_DIR=/mnt/nvme \
#   CKPT=/mnt/runs/prior_gulfstream/best.pt \
#   OUT_S3=s3://my-bucket/runs/prior_gulfstream \
#   bash aws/eval_prior.sh
#
#   # second checkpoint, reusing the first run's real fields (skips the ~66 GB load):
#   CKPT=/mnt/runs/prior_gulfstream/checkpoints/ckpt_step0400000.pt \
#   REAL_FROM=/mnt/runs/prior_gulfstream/eval_step01140000 \
#   bash aws/eval_prior.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source aws/common.sh

CKPT="${CKPT:?set CKPT to a run directory, best.pt, or ckpt.pt}"

stage_in
restore_out
trap on_exit EXIT
start_background_sync

if [[ -d "$CKPT" ]]; then
  if [[ -f "$CKPT/best.pt" ]]; then
    CKPT_PT="$CKPT/best.pt"
  else
    CKPT_PT="$CKPT/ckpt.pt"
  fi
else
  CKPT_PT="$CKPT"
fi
[[ -f "$CKPT_PT" ]] || { echo "[eval] no such checkpoint: $CKPT_PT" >&2; exit 2; }

# Plotting stack. --upgrade-strategy only-if-needed keeps the image's numpy pin.
python -c "import scipy, matplotlib, cmocean" >/dev/null 2>&1 \
  || pip install --upgrade-strategy only-if-needed scipy matplotlib cmocean

if [[ "${SKIP_SELFTEST:-0}" != 1 ]]; then
  python -m eval.kernels --selftest \
    || { echo "[eval] kernel selftest FAILED" >&2; exit 1; }
fi

STEP=$(python - "$CKPT_PT" <<'PY'
import sys, torch
print(f"{int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step']) + 1:07d}")
PY
)

SIZE="${SIZE:-full,128}"
if [[ -n "${EVAL_OUT:-}" ]]; then
  OUT="$EVAL_OUT"
else
  OUT="$(dirname "$CKPT_PT")/eval_step${STEP}"
fi
mkdir -p "$OUT"

# --n must have one value, or one per --size. The eval.slurm defaults apply
# only to the default pair of geometries; any other SIZE uses gen_prior's own
# defaults unless N is set explicitly (e.g. N=48 or N=48,512).
N_ARGS=()
if [[ -n "${N:-}" ]]; then
  N_ARGS=(--n "$N")
elif [[ "$SIZE" == "full,128" ]]; then
  N_ARGS=(--n "${N_FULL:-48},${N_PATCH:-512}")
fi

echo "[eval] ckpt=$CKPT_PT  step=$STEP"
echo "[eval] out=$OUT  size=$SIZE  real_from=${REAL_FROM:-<load store>}"
echo "[eval] GULFSTREAM_MANIFEST=${GULFSTREAM_MANIFEST:-<config default>}"
echo "[eval] OUT_DIR=$OUT_DIR  CACHE_DIR=$CACHE_DIR  OUT_S3=${OUT_S3:-<none>}"

python -u -m eval.gen_prior --ckpt "$CKPT_PT" --out-dir "$OUT" \
  --size "$SIZE" \
  "${N_ARGS[@]}" \
  ${SPLIT:+--split "$SPLIT"} \
  ${REAL_FROM:+--real-from "$REAL_FROM"} \
  ${SEED:+--seed "$SEED"} \
  ${WEIGHTS:+--weights "$WEIGHTS"} \
  ${SAMPLER_STEPS:+--sampler-steps "$SAMPLER_STEPS"} \
  ${S_CHURN:+--s-churn "$S_CHURN"} \
  ${S_NOISE:+--s-noise "$S_NOISE"} \
  ${GEN_BATCH:+--batch "$GEN_BATCH"}

if [[ "${SKIP_FIGS:-0}" == 1 ]]; then
  echo "[eval] SKIP_FIGS=1 -- samples in $OUT"
  exit 0
fi

IFS=',' read -r -a SIZES <<< "$SIZE"
for size in "${SIZES[@]}"; do
  size="${size// /}"
  [[ -n "$size" ]] || continue
  python -u -m eval.diagnostics \
    --samples "$OUT/samples_${size}.npz" \
    --out "$OUT/figs_${size}"
done

echo "[eval] done -- samples and figures in $OUT"
