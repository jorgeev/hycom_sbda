#!/bin/bash
#SBATCH --job-name=hycom_eval
#SBATCH --output=logs/hycom_eval_%j.out
#SBATCH --error=logs/hycom_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#
# Unconditional-prior evaluation: draw samples at both geometries, then run the
# diagnostics on each.
#
# Usage: sbatch eval.slurm [ckpt] [gpu] [n_full] [n_patch] [real-from]
#   sbatch eval.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt
#   sbatch eval.slurm .../checkpoints/ckpt_step0400000.pt 1 48 512 .../eval_step1140000
#
# The 5th argument reuses an earlier run's real reference fields, which depend on
# the dataset and split but never on the checkpoint. Without it every run pays
# the ~1 h store load again; with it a second checkpoint takes about five
# minutes. Pass it for every run after the first.
#
# Results land in <ckpt dir>/eval_step<N>/. Pointing this at an older snapshot
# from checkpoints/ costs nothing extra and answers whether the steps after the
# val curve went flat bought anything in sample space.
#
# GPU SELECTION: same site-specific problem as train.slurm -- SLURM's exported
# CUDA_VISIBLE_DEVICES does not identify the GPUs it granted on this node, and
# physical GPU 0 is held by an external vLLM process using ~76 of 80 GB. Take
# the whole pool with --gres and pin by hand. This job needs one GPU and under
# 10 GB of it.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

CKPT="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt}"
export CUDA_VISIBLE_DEVICES="${2:-1}"
N_FULL="${3:-48}"
N_PATCH="${4:-512}"

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[eval] ckpt=$CKPT"
echo "[eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset})"

# The kernels gate everything downstream: a sign error in geostrophy or a broken
# PSD normalisation would produce figures that look entirely reasonable and are
# entirely wrong. Refuse to spend GPU time if they do not pass.
python -m eval.kernels --selftest || { echo "[eval] kernel selftest FAILED" >&2; exit 1; }

STEP=$(python - "$CKPT" <<'PY'
import sys, torch
print(f"{int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step']) + 1:07d}")
PY
)
OUT="$(dirname "$CKPT")/eval_step${STEP}"
mkdir -p "$OUT"
echo "[eval] step=$STEP  out=$OUT"

set -e
# One invocation, both geometries: the ~1 h dataset preload happens once.
REAL_FROM="${5:-}"
python -m eval.gen_prior --ckpt "$CKPT" --out-dir "$OUT" \
    --size full,128 --n "$N_FULL,$N_PATCH" \
    ${REAL_FROM:+--real-from "$REAL_FROM"}

for size in full 128; do
    python -m eval.diagnostics --samples "$OUT/samples_${size}.npz" \
        --out "$OUT/figs_${size}"
done

echo "[eval] done -- figures in $OUT/figs_full and $OUT/figs_128"
