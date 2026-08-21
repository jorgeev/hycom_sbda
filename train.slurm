#!/bin/bash
#SBATCH --job-name=hycom_sbda
#SBATCH --output=logs/hycom_sbda_%j.out
#SBATCH --error=logs/hycom_sbda_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#
# Usage: sbatch train.slurm <config> [nranks] [gpu-list]
#   sbatch --gres=gpu:a100:2 --time=2-12:00:00 \
#       train.slurm diffusion/configs/prior_gulfstream.yaml 2
#
# Leave [gpu-list] unset under SLURM -- it grants the GPUs and exports the
# right CUDA_VISIBLE_DEVICES itself. Match --gres to <nranks>.
#
# The GenDA-parity configs want a 64-sample GLOBAL batch, which 3 ranks cannot
# split evenly -- run those as 2 ranks x batch 32.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

# GPU selection. Prefer SLURM's own allocation: gres.conf on the COAPS node
# declares only /dev/nvidia{1,2,3}, so the externally-held physical GPU 0 is
# already outside the pool and SLURM sets CUDA_VISIBLE_DEVICES to a correct
# 0-based list within whatever it granted (e.g. "0,1" for --gres=gpu:a100:2).
#
# This used to hardcode `export CUDA_VISIBLE_DEVICES="${3:-1,2,3}"` against a
# time when gres did not isolate. That is now wrong and fails loudly: the value
# indexes *into* the set SLURM already restricted, so on a 2-GPU allocation
# ordinal "3" does not exist and rank 1 dies with "invalid device ordinal"
# while rank 0 comes up fine. Only override when a list is passed explicitly,
# and note those ordinals are allocation-relative, not physical.
if [[ -n "${3:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$3"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Not under SLURM (or gres not in play): fall back to the non-held devices.
    export CUDA_VISIBLE_DEVICES="1,2,3"
fi
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset})"

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

CONFIG="${1:-diffusion/configs/prior_gulfstream.yaml}"
NRANKS="${2:-3}"

torchrun --standalone --nproc_per_node="$NRANKS" --master_port=29517 \
    -m diffusion.train --config "$CONFIG"
rc=$?

# Propagate failure. torchrun surfaces a dead rank as a non-zero
# ChildFailedError; swallowing it once let a 3%-trained checkpoint be consumed
# as a finished baseline, fabricating a result that had to be retracted. A
# non-zero exit here also makes `sbatch --dependency=afterok:` actually hold.
if [[ $rc -ne 0 ]]; then
    echo "Training FAILED (exit $rc) -- checkpoint is NOT usable" >&2
    exit $rc
fi
echo "Training completed"
