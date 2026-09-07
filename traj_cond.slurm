#!/bin/bash
#SBATCH --job-name=hycom_traj_cond
#SBATCH --output=logs/hycom_traj_cond_%j.out
#SBATCH --error=logs/hycom_traj_cond_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#
# Paired-skill-vs-training-step trajectory for one CONDITIONAL run.
#
# Usage: sbatch traj_cond.slurm [run_dir] [gpu] [steps...]
#   sbatch traj_cond.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/<cond_run>
#   SIZES=full,256 sbatch traj_cond.slurm .../<cond_run> 1 0200000 0150000 0100000
# (also runs directly: bash traj_cond.slurm ...)
#
# Every checkpoint is evaluated at the config's OWN sampler settings and, via
# --real-from, on the SAME target steps with the SAME conditioning tensors, so
# the single variable across the curve is the training step -- which is what
# eval.step_trajectory_cond then filters on (it hashes the target days). Do not
# add sampler overrides here; that is what sweep_cond.slurm is for.
#
# The newest checkpoint runs first and pays the store load; every later rung
# reuses its truth/cond cache and costs only the generation. Each step
# directory gets a .done marker so a resubmitted job skips what finished.
#
# Environment knobs: SIZES (default "full,128"), K (default 16), D_FULL (48),
# D_PATCH (256), STORE_COND (full).

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

RUN="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_joint}"
export CUDA_VISIBLE_DEVICES="${2:-1}"
shift 2 2>/dev/null || shift $#
SIZES="${SIZES:-full,128}"
K="${K:-16}"
D_FULL="${D_FULL:-48}"
D_PATCH="${D_PATCH:-256}"
STORE_COND="${STORE_COND:-full}"

LADDER="$*"
if [ -z "$LADDER" ]; then
    LADDER=$(ls "$RUN/checkpoints" 2>/dev/null \
        | sed -n 's/^ckpt_step\([0-9]\+\)\.pt$/\1/p' | sort -r \
        | awk 'NR==1 {print; last=$1; next} last-$1 >= 70000 {print; last=$1}')
fi
[ -z "$LADDER" ] && { echo "[traj_cond] no checkpoints under $RUN/checkpoints" >&2; exit 1; }

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[traj_cond] run=$RUN  sizes=$SIZES  K=$K"
echo "[traj_cond] ladder: $(echo $LADDER | tr '\n' ' ')"
echo "[traj_cond] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m eval.kernels --selftest || { echo "[traj_cond] kernel selftest FAILED" >&2; exit 1; }
python -m eval.kernels_cond --selftest || { echo "[traj_cond] kernels_cond selftest FAILED" >&2; exit 1; }

NS=""
for size in ${SIZES//,/ }; do
    if [ "$size" = "full" ]; then NS="${NS:+$NS,}$D_FULL"; else NS="${NS:+$NS,}$D_PATCH"; fi
done

set -e
CACHE=""
for s in $LADDER; do
    out="$RUN/traj_cond/step${s}"
    if [ -f "$out/.done" ]; then
        echo "[traj_cond] step${s} already done, skipping"
        [ -z "$CACHE" ] && CACHE="$out"
        continue
    fi
    echo; echo "=== [step${s}] $(date +%H:%M:%S) ${CACHE:+(real-from $(basename "$CACHE"))} ==="
    have_all=1
    for size in ${SIZES//,/ }; do
        [ -s "$out/samples_cond_${size}.npz" ] || have_all=0
    done
    if [ "$have_all" = 1 ]; then
        echo "[traj_cond] step${s} samples already present, re-running figures only"
    else
        python -m eval.gen_cond --ckpt "$RUN/checkpoints/ckpt_step${s}.pt" \
            --out-dir "$out" --size "$SIZES" --n "$NS" --k "$K" --store-cond "$STORE_COND" \
            ${CACHE:+--real-from "$CACHE"}
    fi
    for size in ${SIZES//,/ }; do
        python -m eval.diagnostics_cond --samples "$out/samples_cond_${size}.npz" \
            --out "$out/figs_cond_${size}"
    done
    touch "$out/.done"
    [ -z "$CACHE" ] && CACHE="$out"
done

python -m eval.step_trajectory_cond --run "$RUN" --size "$SIZES"

echo "[traj_cond] done -- see $RUN/traj_cond/figs_<size> and trajectory_cond.csv"
