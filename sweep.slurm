#!/bin/bash
#SBATCH --job-name=hycom_sweep
#SBATCH --output=logs/hycom_sweep_%j.out
#SBATCH --error=logs/hycom_sweep_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#
# Sampler / EMA / checkpoint sweep for the under-dispersion diagnosis.
#
# Usage: sbatch sweep.slurm [run_dir] [gpu]
#   sbatch sweep.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream 1
# (also runs directly: bash sweep.slurm)
#
# Every configuration reuses the real reference fields cached by the baseline
# eval (--real-from), so nothing here touches the 66 GB store. Cost per config
# is minutes: ~0.10 s/patch and ~4.3 s/full frame at 32 steps, linear in steps.
#
# THE GRID. The step-1140000 baseline (32 steps, s_churn 10, s_noise 1.0) is
# under-dispersive: normalised gen std 0.75-0.94 vs real 1.02-1.17 at patch
# scale, total EKE ratio 0.61 patch / 0.58 full. The knobs that counteract
# that, and one coupling that shapes the grid: edm_heun_sampler computes
#   gamma = min(s_churn / num_steps, sqrt(2)-1 ~ 0.4142)
# so churn and step count are NOT separable -- at 32 steps anything above
# s_churn ~ 13.26 clamps to the cap (churn 20/40/80 at 32 steps are all the
# same run), and raising steps alone silently lowers gamma. Rows below either
# hold gamma at the baseline 0.3125 while sweeping steps, or saturate the cap.
#
# Each config directory gets a .done marker so a resubmitted job skips what
# already finished.

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

RUN="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream}"
export CUDA_VISIBLE_DEVICES="${2:-1}"   # physical GPU 0 is held by vLLM, as in eval.slurm

CKPT="$RUN/best.pt"
CACHE="$RUN/eval_step1140000"           # baseline eval: real-field cache + reference point

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[sweep] run=$RUN  ckpt=$CKPT  cache=$CACHE"
echo "[sweep] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m eval.kernels --selftest || { echo "[sweep] kernel selftest FAILED" >&2; exit 1; }

set -e

run_cfg () {
    local id="$1"; shift
    local ckpt="$1"; shift
    local out="$RUN/sweep/$id"
    if [ -f "$out/.done" ]; then
        echo "[sweep] $id already done, skipping"
        return
    fi
    echo; echo "=== [$id] $* ==="
    python -m eval.gen_prior --ckpt "$ckpt" --out-dir "$out" \
        --size full,128 --real-from "$CACHE" "$@"
    for size in full 128; do
        python -m eval.diagnostics --samples "$out/samples_${size}.npz" \
            --out "$out/figs_${size}"
    done
    touch "$out/.done"
}

# -- sampler sweep at best.pt (step 1140000), EMA weights ---------------------
run_cfg s64    "$CKPT" --sampler-steps 64  --s-churn 20      # steps x2, gamma held at 0.3125
run_cfg s128   "$CKPT" --sampler-steps 128 --s-churn 40      # steps x4, gamma held at 0.3125
run_cfg cmax32 "$CKPT" --sampler-steps 32  --s-churn 15      # gamma saturated at 32 steps
run_cfg cmax64 "$CKPT" --sampler-steps 64  --s-churn 30      # gamma saturated, more total churn noise
run_cfg n1003  "$CKPT" --s-noise 1.003                       # EDM's own over-denoising remedy
run_cfg n1007  "$CKPT" --s-noise 1.007
run_cfg n1010  "$CKPT" --s-noise 1.010
run_cfg combo  "$CKPT" --sampler-steps 64 --s-churn 30 --s-noise 1.007

# -- EMA ablation: raw online weights, baseline sampler -----------------------
run_cfg raw    "$CKPT" --weights model

# -- earlier checkpoint, baseline sampler: did 400k -> 1140k help dispersion? --
run_cfg step400k "$RUN/checkpoints/ckpt_step0400000.pt"

# -- one table over everything ------------------------------------------------
python -m eval.sweep_summary --run "$RUN"

echo "[sweep] done -- see $RUN/sweep/summary.md"
