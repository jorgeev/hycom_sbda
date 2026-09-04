#!/bin/bash
#SBATCH --job-name=hycom_traj
#SBATCH --output=logs/hycom_traj_%j.out
#SBATCH --error=logs/hycom_traj_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#
# Spectra-vs-training-step trajectory for one run.
#
# Usage: sbatch traj.slurm [run_dir] [gpu] [steps...]
#   sbatch traj.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_genda_masked
#   sbatch traj.slurm .../prior_genda_masked 1 1130000 1000000 0800000 0610000
# (also runs directly: bash traj.slurm ...)
#
# Every checkpoint is evaluated at the config's OWN sampler settings, so the
# single variable across the resulting curve is the training step -- which is
# what eval.step_trajectory then filters on. Do not add sampler overrides here;
# that is what sweep.slurm is for.
#
# The newest checkpoint runs first and pays the dataset load; every later point
# reuses its real reference fields through --real-from and costs minutes. Each
# step directory gets a .done marker so a resubmitted job skips what finished.
#
# GPU SELECTION: as in eval.slurm -- SLURM's exported CUDA_VISIBLE_DEVICES does
# not identify the GPUs it granted, so take the pool with --gres and pin by hand.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

RUN="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_genda_masked}"
export CUDA_VISIBLE_DEVICES="${2:-1}"
shift 2 2>/dev/null || shift $# 

# Default ladder: newest first, then spread back over whatever the run retained.
LADDER="$*"
if [ -z "$LADDER" ]; then
    LADDER=$(ls "$RUN/checkpoints" 2>/dev/null \
        | sed -n 's/^ckpt_step\([0-9]\+\)\.pt$/\1/p' | sort -r \
        | awk 'NR==1 {print; last=$1; next} last-$1 >= 70000 {print; last=$1}')
fi
[ -z "$LADDER" ] && { echo "[traj] no checkpoints under $RUN/checkpoints" >&2; exit 1; }

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[traj] run=$RUN"
echo "[traj] ladder: $(echo $LADDER | tr '\n' ' ')"
echo "[traj] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m eval.kernels --selftest || { echo "[traj] kernel selftest FAILED" >&2; exit 1; }

set -e
CACHE=""
for s in $LADDER; do
    out="$RUN/traj/step${s}"
    if [ -f "$out/.done" ]; then
        echo "[traj] step${s} already done, skipping"
        [ -z "$CACHE" ] && CACHE="$out"
        continue
    fi
    echo; echo "=== [step${s}] $(date +%H:%M:%S) ${CACHE:+(real-from $(basename "$CACHE"))} ==="
    # Samples but no .done means a previous run died in the figures. Generation
    # is the expensive half and its output is complete, so do not pay for it
    # again -- especially not for the first checkpoint, which is the one that
    # loads the store.
    if [ -s "$out/samples_full.npz" ] && [ -s "$out/samples_128.npz" ]; then
        echo "[traj] step${s} samples already present, re-running figures only"
    else
        python -m eval.gen_prior --ckpt "$RUN/checkpoints/ckpt_step${s}.pt" \
            --out-dir "$out" --size full,128 ${CACHE:+--real-from "$CACHE"}
    fi
    for size in full 128; do
        python -m eval.diagnostics --samples "$out/samples_${size}.npz" \
            --out "$out/figs_${size}"
    done
    touch "$out/.done"
    [ -z "$CACHE" ] && CACHE="$out"
done

python -m eval.step_trajectory --run "$RUN"

echo "[traj] done -- see $RUN/traj/figs_128 and $RUN/traj/figs_full"
