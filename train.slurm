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
#   sbatch train.slurm diffusion/configs/prior_gulfstream.yaml 2 1,2
#
# The GenDA-parity configs want a 64-sample GLOBAL batch, which 3 ranks cannot
# split evenly -- run those as 2 ranks x batch 32.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

# SITE-SPECIFIC, and load-bearing on the COAPS node this was written for: SLURM
# gres does NOT isolate GPUs there (all devices stay visible) and GPU 0 is
# permanently held by an external process, so ranks must be pinned by hand or
# rank 0 lands on GPU 0 and OOMs immediately. On a cluster where gres isolates
# properly, drop this line (or pass a gpu-list) -- but check before assuming.
export CUDA_VISIBLE_DEVICES="${3:-1,2,3}"

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
