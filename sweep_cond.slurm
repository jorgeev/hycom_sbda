#!/bin/bash
#SBATCH --job-name=hycom_sweep_cond
#SBATCH --output=logs/hycom_sweep_cond_%j.out
#SBATCH --error=logs/hycom_sweep_cond_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#
# Sampler / EMA / checkpoint sweep for a CONDITIONAL model: which knob buys
# calibration (spread/skill -> 1, rank TV -> 0) without costing skill (RMSE,
# CRPS)?
#
# Usage: sbatch sweep_cond.slurm [run_dir] [gpu] [ckpt]
#   sbatch sweep_cond.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/<cond_run> 1
# (also runs directly: bash sweep_cond.slurm)
#
# Every arm reuses the truth + conditioning cached by the newest baseline
# eval_cond_step*/ of the run (--real-from), so nothing here touches the store
# and every arm is scored on identical target steps and observations. Run
# eval_cond.slurm first.
#
# THE GRID mirrors sweep.slurm (see the gamma = min(s_churn/num_steps,
# sqrt(2)-1) coupling documented there) plus one arm that matters more for a
# conditional model: churn0, the deterministic Heun sampler, where members
# differ ONLY by their initial noise. That is the cleanest read of how much
# spread the conditioning itself leaves, before any stochastic re-injection.
#
# Each arm directory gets a .done marker so a resubmitted job skips what
# already finished. Environment knobs: SIZES (default "full,128"), K (16).

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

RUN="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_joint}"
export CUDA_VISIBLE_DEVICES="${2:-1}"   # physical GPU 0 is held by vLLM, as in eval.slurm
CKPT="${3:-$RUN/ckpt.pt}"
[ -f "$RUN/best.pt" ] && [ -z "${3:-}" ] && CKPT="$RUN/best.pt"
SIZES="${SIZES:-full,128}"
K="${K:-16}"

CACHE=$(ls -d "$RUN"/eval_cond_step* 2>/dev/null | sort | tail -1)
[ -z "$CACHE" ] && { echo "[sweep_cond] no $RUN/eval_cond_step*/ to reuse -- run eval_cond.slurm first" >&2; exit 1; }

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[sweep_cond] run=$RUN  ckpt=$CKPT  cache=$CACHE  sizes=$SIZES  K=$K"
echo "[sweep_cond] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m eval.kernels --selftest || { echo "[sweep_cond] kernel selftest FAILED" >&2; exit 1; }
python -m eval.kernels_cond --selftest || { echo "[sweep_cond] kernels_cond selftest FAILED" >&2; exit 1; }

set -e

run_cfg () {
    local id="$1"; shift
    local ckpt="$1"; shift
    local out="$RUN/sweep_cond/$id"
    if [ -f "$out/.done" ]; then
        echo "[sweep_cond] $id already done, skipping"
        return
    fi
    echo; echo "=== [$id] $* ==="
    python -m eval.gen_cond --ckpt "$ckpt" --out-dir "$out" \
        --size "$SIZES" --k "$K" --real-from "$CACHE" "$@"
    for size in ${SIZES//,/ }; do
        python -m eval.diagnostics_cond --samples "$out/samples_cond_${size}.npz" \
            --out "$out/figs_cond_${size}"
    done
    touch "$out/.done"
}

# -- the deterministic sampler: spread from the initial noise alone --------------
run_cfg churn0 "$CKPT" --s-churn 0

# -- sampler sweep, EMA weights (same grid as sweep.slurm) --------------------------
run_cfg s64    "$CKPT" --sampler-steps 64  --s-churn 20      # steps x2, gamma held
run_cfg s128   "$CKPT" --sampler-steps 128 --s-churn 40      # steps x4, gamma held
run_cfg cmax32 "$CKPT" --sampler-steps 32  --s-churn 15      # gamma saturated at 32 steps
run_cfg cmax64 "$CKPT" --sampler-steps 64  --s-churn 30      # gamma saturated, more churn noise
run_cfg n1003  "$CKPT" --s-noise 1.003
run_cfg n1007  "$CKPT" --s-noise 1.007
run_cfg n1010  "$CKPT" --s-noise 1.010
run_cfg combo  "$CKPT" --sampler-steps 64 --s-churn 30 --s-noise 1.007

# -- EMA ablation: raw online weights, baseline sampler ---------------------------
run_cfg raw    "$CKPT" --weights model

# -- an earlier checkpoint, baseline sampler (if the run kept one) ------------------
EARLY=$(ls "$RUN/checkpoints"/ckpt_step*.pt 2>/dev/null | sort | awk 'NR==1')
[ -n "$EARLY" ] && run_cfg early "$EARLY"

# -- one table over everything ------------------------------------------------------
python -m eval.sweep_summary_cond --run "$RUN" --baseline "$(basename "$CACHE")"

echo "[sweep_cond] done -- see $RUN/sweep_cond/summary.md"
