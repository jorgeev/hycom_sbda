#!/bin/bash
#SBATCH --job-name=hycom_eval_cond
#SBATCH --output=logs/hycom_eval_cond_%j.out
#SBATCH --error=logs/hycom_eval_cond_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#
# CONDITIONAL evaluation: draw K-member ensembles for D held-out target steps at
# each geometry, paired with the truth and the observations they were
# conditioned on, then run the paired diagnostics on each.
#
# Usage: sbatch eval_cond.slurm [ckpt] [gpu] [D_full] [D_patch] [real-from]
#   sbatch eval_cond.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_joint/ckpt.pt
#   SIZES=full,256 K=16 sbatch eval_cond.slurm .../prior_joint/ckpt.pt 1 48 256
#   sbatch eval_cond.slurm .../checkpoints/ckpt_step0150000.pt 1 48 256 .../eval_cond_step0200000
#
# Environment knobs: SIZES (default "full,128"; a mask+doy model trained at
# patch 256 wants "full,256"), K (members, default 16), STORE_COND
# ("full"/"last"/"none", default full; use "last" for a large k window),
# GEN_DTYPE (float16 default).
#
# The 5th argument reuses an earlier run's truth AND conditioning tensors,
# which depend on the dataset, split, seed and the config's conditioning
# fields but never on the weights. It skips the store preload (minutes on
# gom_nemo, ~1 h cold on gulfstream) and guarantees every checkpoint of a run
# is conditioned on identical observations. Pass it for every run after the
# first. It is refused if the cached conditioning does not match this
# checkpoint's config, or was written with STORE_COND=last and k_days > 1.
#
# Results land in <ckpt dir>/eval_cond_step<N>/.
#
# GPU SELECTION: as in eval.slurm -- SLURM's exported CUDA_VISIBLE_DEVICES does
# not identify the GPUs it granted on this node, and physical GPU 0 is held by
# an external vLLM process. Take the pool with --gres and pin by hand.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

CKPT="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_joint/ckpt.pt}"
export CUDA_VISIBLE_DEVICES="${2:-1}"
D_FULL="${3:-48}"
D_PATCH="${4:-256}"
REAL_FROM="${5:-}"
SIZES="${SIZES:-full,128}"
K="${K:-16}"
STORE_COND="${STORE_COND:-full}"
GEN_DTYPE="${GEN_DTYPE:-float16}"

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[eval_cond] ckpt=$CKPT  sizes=$SIZES  K=$K  D=$D_FULL/$D_PATCH  store_cond=$STORE_COND"
echo "[eval_cond] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset})"

# Both kernel modules gate everything downstream: a wrong CRPS normalisation
# or a broken PSD binning produces plausible numbers, not a crash.
python -m eval.kernels --selftest || { echo "[eval_cond] kernel selftest FAILED" >&2; exit 1; }
python -m eval.kernels_cond --selftest || { echo "[eval_cond] kernels_cond selftest FAILED" >&2; exit 1; }

STEP=$(python - "$CKPT" <<'PY'
import sys, torch
print(f"{int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step']) + 1:07d}")
PY
)
OUT="$(dirname "$CKPT")/eval_cond_step${STEP}"
mkdir -p "$OUT"
echo "[eval_cond] step=$STEP  out=$OUT"

# Per-size D: "full" gets D_FULL, every patch size gets D_PATCH.
NS=""
for size in ${SIZES//,/ }; do
    if [ "$size" = "full" ]; then NS="${NS:+$NS,}$D_FULL"; else NS="${NS:+$NS,}$D_PATCH"; fi
done

set -e
python -m eval.gen_cond --ckpt "$CKPT" --out-dir "$OUT" \
    --size "$SIZES" --n "$NS" --k "$K" --store-cond "$STORE_COND" --gen-dtype "$GEN_DTYPE" \
    ${REAL_FROM:+--real-from "$REAL_FROM"}

for size in ${SIZES//,/ }; do
    python -m eval.diagnostics_cond --samples "$OUT/samples_cond_${size}.npz" \
        --out "$OUT/figs_cond_${size}"
done

echo "[eval_cond] done -- figures in $OUT/figs_cond_<size>"
