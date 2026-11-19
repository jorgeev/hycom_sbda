#!/bin/bash
#SBATCH --job-name=hycom_vlanes
#SBATCH --output=logs/hycom_vlanes_%j.out
#SBATCH --error=logs/hycom_vlanes_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00
#SBATCH --mem=320G
#
# Video twin of train_lanes.slurm (runs diffusion.train_video): the natural way
# to train the two noise-level experts side by side, one lane each.
#
# Several independent trainings inside ONE allocation. The gres pool is a
# single 3-GPU gres on this node and shrinking it hits the cgroup CPU-fallback
# bug (see train.slurm), so two train.slurm jobs would only serialise; this
# driver takes the pool once and pins each lane to the GPUs you name, exactly
# as train.slurm does for one run. Each lane is train.slurm's torchrun line
# (standalone rendezvous picks its own free port, so lanes do not collide).
#
# Usage: sbatch train_video_lanes.slurm "<config>:<nranks>:<gpu-list> [...]"
#   OUT_DIR=/unity/g2/jvelasco/gitraw/hycom_sbda/aws_runs/runs \
#     sbatch train_video_lanes.slurm \
#       "diffusion/configs/video_gulfstream_small.yaml:1:2 \
#        diffusion/configs/video_gulfstream_large.yaml:1:3"
# Lane output: logs/train_vlane_<config basename>_<jobid>.log. Exit status is
# non-zero if any lane failed (its checkpoint is then NOT a finished run).
# Also runs by hand: bash train_video_lanes.slurm "..." (jobid = manual).
#
# --mem is explicit because the partition's default (DefMemPerCPU=UNLIMITED)
# turns an unset --mem into a request for the whole 1 TB node, which pends on
# "Resources" whenever any other job holds memory here (job 2577, 2026-10-12).
# Each rank preloads ~66 GB of store; 320G covers two single-rank lanes with
# their loader workers. Raise it for more ranks.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

LANES="${1:?usage: train_video_lanes.slurm \"<config>:<nranks>:<gpu-list> ...\"}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1
# Avoid allocator fragmentation near the memory limit (see aws/launch_video.sh).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
JOB="${SLURM_JOB_ID:-manual}"

pids=(); names=()
for lane in $LANES; do
    IFS=: read -r cfg nranks gpus <<<"$lane"
    [[ -f "$cfg" && -n "$nranks" && -n "$gpus" ]] || { echo "[lanes] bad lane spec: $lane" >&2; exit 2; }
    name="$(basename "$cfg" .yaml)"
    log="logs/train_vlane_${name}_${JOB}.log"
    echo "[lanes] $name: $nranks rank(s) on GPU(s) $gpus -> $log"
    CUDA_VISIBLE_DEVICES="$gpus" torchrun --standalone --nproc_per_node="$nranks" \
        -m diffusion.train_video --config "$cfg" > "$log" 2>&1 &
    pids+=($!); names+=("$name")
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[lanes] ${names[$i]} completed"
    else
        echo "[lanes] ${names[$i]} FAILED -- its checkpoint is NOT a finished run" >&2
        rc=1
    fi
done
exit $rc
