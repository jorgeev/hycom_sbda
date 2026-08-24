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
#   sbatch --gres=gpu:a100:3 --time=2-12:00:00 \
#       train.slurm diffusion/configs/prior_gulfstream.yaml 2 2,3
#
# ON THE COAPS NODE, ALWAYS PASS [gpu-list] EXPLICITLY. SLURM's own
# CUDA_VISIBLE_DEVICES is wrong here -- see the block below. Request
# --gres=gpu:a100:3 (the whole pool) and select the physical GPUs by hand.
#
# The GenDA-parity configs want a 64-sample GLOBAL batch, which 3 ranks cannot
# split evenly -- run those as 2 ranks x batch 32.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

# GPU selection. SITE-SPECIFIC and load-bearing: on the COAPS node SLURM's
# exported CUDA_VISIBLE_DEVICES does NOT identify the GPUs it granted, so
# trusting it puts a rank on the externally-held GPU 0 and OOMs.
#
# Measured 2026-08-21 with probe jobs (slurm 24.05.2, ConstrainDevices=yes,
# cgroup v2). gres.conf declares /dev/nvidia{1,2,3}; physical GPU 0 is held by
# an external vLLM process using ~79 of 80 GB. Identifying cards by UUID:
#
#   --gres=gpu:a100:2 -> SLURM grants slots 0,1 (= /dev/nvidia1,2) and exports
#       CUDA_VISIBLE_DEVICES=0,1. But the process actually sees physical
#       {0,1,3} -- GPU 0 leaks in and granted nvidia2 is missing. Ordinals 0,1
#       therefore land on physical 0 and 1: rank 0 gets the vLLM card with
#       1.7 GB free. This is what happened to job 2329.
#   --gres=gpu:a100:3 -> all four devices visible and enumeration matches
#       physical order, so an explicit list means what it says.
#
# Hence: take the whole pool and pin by hand. The ordinals below are PHYSICAL
# only because of that 3-GPU grant; under a partial grant they are scrambled.
#
# (The earlier revision of this comment claimed gres isolated properly and told
# you to leave the list unset. That was wrong and cost a job.)
if [[ -n "${3:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$3"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[gpu] WARNING: no gpu-list passed under SLURM. On the COAPS node" >&2
    echo "[gpu] SLURM's CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} may" >&2
    echo "[gpu] select the held GPU 0. Verify placement by UUID, or pass a list." >&2
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Not under SLURM at all: fall back to the non-held devices.
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
