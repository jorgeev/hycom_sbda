#!/bin/bash
#SBATCH --job-name=hycom_eval_aws
#SBATCH --output=logs/hycom_eval_aws_%j.out
#SBATCH --error=logs/hycom_eval_aws_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#
# Unconditional-prior evaluation of the AWS-trained gulfstream runs, best.pt
# and ckpt.pt of each, two runs side by side on two GPUs. The eval.slurm job
# for a run that trained here, with three differences:
#
#   1. eval.gen_prior_relocated instead of eval.gen_prior: the AWS checkpoints
#      carry /mnt/runs/cache in their config (see that module's docstring).
#   2. --size full,256: these runs trained at patch 256, so the "training
#      distribution" geometry is 256, not eval.slurm's hardcoded 128. The
#      full-domain geometry is the one every run shares and is compared on.
#   3. Two runs in parallel inside one job. --gres takes the whole pool (the
#      pool is a single gres and shrinking it hits the cgroup CPU-fallback bug),
#      so two separate jobs would only serialise; two lanes pinned by hand on
#      GPUs 1 and 2 use it. GPU 0 is held by an external process.
#
# Within a lane the second checkpoint reuses the first one's real fields
# (--real-from), as eval.slurm recommends.
#
# Usage: sbatch eval_aws.slurm [runs_root]
#   RUNS="prior_gulfstream_geo:1 prior_gulfstream_256:2"  (run:gpu pairs)
#   CKPTS="best.pt ckpt.pt"   SIZES=full,256   N=48,512
#   REAL_FROM=<eval dir>  seeds the first checkpoint of every lane with cached
#   real fields (same variable list required), e.g. for a checkpoint sweep.
#   Per-lane: CKPTS_<run> SIZES_<run> N_<run> REAL_FROM_<run> (see lane()).

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

ROOT="${1:-aws_runs/runs}"
RUNS="${RUNS:-prior_gulfstream_geo:1 prior_gulfstream_256:2}"
CKPTS="${CKPTS:-best.pt ckpt.pt}"
SIZES="${SIZES:-full,256}"
N="${N:-48,512}"
export CACHE_DIR="${CACHE_DIR:-diffusion}"
set -o pipefail

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[eval_aws] root=$ROOT runs=$RUNS ckpts=$CKPTS sizes=$SIZES n=$N cache=$CACHE_DIR"

python -m eval.kernels --selftest || { echo "[eval_aws] kernel selftest FAILED" >&2; exit 1; }

step_of() {
    python - "$1" <<'PY'
import sys, torch
print(f"{int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step']) + 1:07d}")
PY
}

# one checkpoint, one GPU: samples + diagnostics. Prints the eval dir last.
eval_one() {
    local ckpt="$1" gpu="$2" real_from="$3" sizes="$4" n="$5"
    local step out
    step=$(step_of "$ckpt")
    out="$(dirname "$ckpt")/eval_step${step}"
    mkdir -p "$out"
    echo "[lane gpu$gpu] ckpt=$ckpt step=$step out=$out sizes=$sizes n=$n real_from=${real_from:-<load store>}"
    CUDA_VISIBLE_DEVICES="$gpu" python -m eval.gen_prior_relocated \
        --ckpt "$ckpt" --out-dir "$out" --size "$sizes" --n "$n" \
        ${real_from:+--real-from "$real_from"} || return 1
    for size in ${sizes//,/ }; do
        python -m eval.diagnostics --samples "$out/samples_${size}.npz" \
            --out "$out/figs_${size}" || return 1
    done
    echo "$out"
}

# Per-lane overrides: CKPTS_<RUN>, SIZES_<RUN>, N_<RUN>, REAL_FROM_<RUN> with
# the run name's non-alphanumerics mapped to '_', e.g. CKPTS_prior_gulfstream_256.
# Unset ones fall back to the global knobs.
lane() {
    local run="$1" gpu="$2" out
    local key="${run//[^A-Za-z0-9]/_}" v
    v="CKPTS_$key";     local ckpts="${!v:-$CKPTS}"
    v="SIZES_$key";     local sizes="${!v:-$SIZES}"
    v="N_$key";         local n="${!v:-$N}"
    v="REAL_FROM_$key"; local real_from="${!v:-${REAL_FROM:-}}"
    for c in $ckpts; do
        out=$(eval_one "$ROOT/$run/$c" "$gpu" "$real_from" "$sizes" "$n" | tee /dev/stderr | tail -1) || {
            echo "[lane gpu$gpu] $run/$c FAILED" >&2; return 1; }
        real_from="$out"
    done
    echo "[lane gpu$gpu] $run done"
}

pids=()
for pair in $RUNS; do
    run="${pair%%:*}"; gpu="${pair##*:}"
    lane "$run" "$gpu" > "logs/eval_aws_${run}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[eval_aws] all lanes finished, rc=$rc -- see logs/eval_aws_<run>_${SLURM_JOB_ID:-manual}.log"
exit $rc
