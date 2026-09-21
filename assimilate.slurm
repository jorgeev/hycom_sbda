#!/bin/bash
#SBATCH --job-name=hycom_sda
#SBATCH --output=logs/hycom_sda_%j.out
#SBATCH --error=logs/hycom_sda_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#
# SCORE-BASED DATA ASSIMILATION on an unconditional prior: build the OSSE case
# once (truth + observations from the store), draw the unguided CONTROL and the
# GUIDED ensembles with identical seeds, then score both with the paired
# diagnostics and the SDA-specific report.
#
# Usage: sbatch assimilate.slurm [ckpt] [gpu | gpu,gpu,...] [obs yaml] [D]
#   A comma list of GPUs runs one day-shard per GPU in parallel (sda.assimilate
#   --day-shard i/n) and merges them -- how a full-frame run fits the time limit:
#   CONTIG=48 K=16 SIZES=full sbatch assimilate.slurm .../best.pt 2,3
#   sbatch assimilate.slurm /unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt
#   SIZES=256 K=24 STEPS=256 sbatch assimilate.slurm .../prior_gulfstream/best.pt 1 sda/configs/gs_store_obs.yaml 24
#   SIZES=128 OBS=sda/configs/gs_genda_like.yaml sbatch assimilate.slurm .../best.pt 2
#
# Environment knobs: SIZES (default "256"; the 65 km AVISO blur is 36 px, too
# wide for a 128 patch), K (members, 24), STEPS (256), CORR (Langevin
# corrections, 0), TAU (0.3), GAMMA (gamma multiplier, 1.0), SAMPLER (vp|edm),
# TERMS (comma list, default all required terms), EXTRA (any extra flags for
# sda.assimilate, e.g. "--grad-ckpt"; "--include-optional" for sda.case via
# CASE_EXTRA), OUTDIR (override the output root), CONTIG (take the LAST N
# consecutive hours of the split at one crop position instead of D spread
# steps, and write per-variable movies; output root gets a _last<N> suffix),
# SKIP_TAIL (with CONTIG: end the window this many steps before the split's end, 0),
# ANIMATE (1 = movies even without CONTIG), REQUIRE_COVERAGE (store var that must have data at every step,
# e.g. ssha_swot), PLACE_ON (store var each patch crop is placed under, e.g. ssha_swot;
# implies REQUIRE_COVERAGE), ARM (name of this arm's subdirectory, default from settings).
#
# Results land in <ckpt dir>/sda_<obs name>_step<N>/:
#   case_<size>.npz                      built once, reused by every arm
#   <arm>/samples_sda_<size>.npz         guided ensembles (CondSamples schema)
#   <arm>/samples_sda_<size>_free.npz    unguided control, same seeds
#   <arm>/figs_sda_<size>/               panels, summary_sda.csv, report_sda.md
#   <arm>/figs_cond_<size>/              eval.diagnostics_cond on the guided run
#   <arm>/figs_cond_<size>_free/         ... and on the control
#
# GPU SELECTION: as in eval.slurm -- SLURM's exported CUDA_VISIBLE_DEVICES does
# not identify the GPUs it granted on this node, and physical GPU 0 is held by
# an external vLLM process. Take the WHOLE pool with --gres and pin by hand:
# a smaller --gres (e.g. `sbatch --gres=gpu:a100:2`) restricts the job's device
# cgroup to SLURM's own two ids, a pinned physical id outside it has no CUDA
# device, and (before the fail-fast in sda.assimilate) the shard ran on the CPU
# at 17 h/step (job 2513, 2026-09-09).

source /conda/jvelasco/miniforge3/etc/profile.d/conda.sh
conda activate datorch

CKPT="${1:-/unity/f1/ozavala/DATA/JorgeVelasco/runs/prior_gulfstream/best.pt}"
GPUS="${2:-1}"                 # one GPU id, or a comma list -> one day-shard per GPU
IFS=, read -ra GPU_ARR <<< "$GPUS"
NSHARD=${#GPU_ARR[@]}
export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
OBS="${3:-${OBS:-sda/configs/gs_store_obs.yaml}}"
D="${4:-${D:-24}}"
SIZES="${SIZES:-256}"
K="${K:-24}"
STEPS="${STEPS:-256}"
CORR="${CORR:-0}"
TAU="${TAU:-0.3}"
GAMMA="${GAMMA:-1.0}"
SAMPLER="${SAMPLER:-vp}"
TERMS="${TERMS:-}"
EXTRA="${EXTRA:-}"
CASE_EXTRA="${CASE_EXTRA:-}"
REQUIRE_COVERAGE="${REQUIRE_COVERAGE:-}"
PLACE_ON="${PLACE_ON:-}"          # store var to place every crop under (ssha_swot)
CONTIG="${CONTIG:-0}"
ANIMATE="${ANIMATE:-0}"

export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
export PYTHONUNBUFFERED=1

echo "[sda] ckpt=$CKPT obs=$OBS sizes=$SIZES D=$D K=$K steps=$STEPS corr=$CORR tau=$TAU gamma=x$GAMMA sampler=$SAMPLER terms=${TERMS:-all}"
echo "[sda] GPUs=$GPUS shards=$NSHARD (SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset})"

# Gates: a wrong adjoint, a broken CRPS or a mis-normalised blur produce
# plausible numbers, not a crash.
python -m eval.kernels_cond --selftest || { echo "[sda] kernels_cond selftest FAILED" >&2; exit 1; }
python -m sda.obs --selftest           || { echo "[sda] sda.obs selftest FAILED" >&2; exit 1; }
python -m sda.assimilate --selftest    || { echo "[sda] sda.assimilate selftest FAILED" >&2; exit 1; }

STEP=$(python - "$CKPT" <<'PY'
import sys, torch
print(f"{int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step']) + 1:07d}")
PY
)
OBSNAME=$(basename "$OBS" .yaml)
OUT="${OUTDIR:-$(dirname "$CKPT")/sda_${OBSNAME}_step${STEP}}"
if [ "$CONTIG" -gt 0 ]; then OUT="${OUTDIR:-${OUT}_last${CONTIG}}"; ANIMATE=1; fi
ARM="${ARM:-${SAMPLER}_s${STEPS}_c${CORR}_g${GAMMA}${TERMS:+_$(echo "$TERMS" | tr ',' '-')}}"
mkdir -p "$OUT/$ARM"
echo "[sda] step=$STEP  out=$OUT  arm=$ARM"

set -e
# 1. the case: once per geometry, shared by every arm
NEED=""
for size in ${SIZES//,/ }; do
    [ -f "$OUT/case_${size}.npz" ] || NEED="${NEED:+$NEED,}$size"
done
if [ -n "$NEED" ]; then
    if [ "$CONTIG" -gt 0 ]; then
        # the very end of the split: keep the last day too (--skip-tail 0). SKIP_TAIL=N
        # ends the window N steps earlier (e.g. 329 -> 2017-06-16 07:00..06-18 06:00,
        # the last 48 h window with four full-width SWOT swaths; the split's last
        # 48 h carry none).
        python -m sda.case --ckpt "$CKPT" --obs "$OBS" --out-dir "$OUT" --size "$NEED" \
            --contiguous "$CONTIG" --skip-tail "${SKIP_TAIL:-0}" $CASE_EXTRA
    else
        python -m sda.case --ckpt "$CKPT" --obs "$OBS" --out-dir "$OUT" --size "$NEED" --n "$D" \
            ${REQUIRE_COVERAGE:+--require-coverage "$REQUIRE_COVERAGE"} \
            ${PLACE_ON:+--place-on-coverage "$PLACE_ON"} $CASE_EXTRA
    fi
fi

# run_assim <output file name> <flags...>: one process, or NSHARD day-shards in
# parallel (one per GPU) followed by a merge into the same file name.
run_assim() {
    local out="$1"; shift
    if [ "$NSHARD" -le 1 ]; then
        CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -m sda.assimilate "${COMMON[@]}" "$@" --out "$out"
        return
    fi
    local pids=() parts=() i part
    for i in $(seq 0 $((NSHARD - 1))); do
        part="${out%.npz}_shard${i}of${NSHARD}.npz"
        parts+=("$OUT/$ARM/$part")
        if [ -f "$OUT/$ARM/$part" ]; then echo "[sda] shard $i/$NSHARD already at $part, reusing"; continue; fi
        echo "[sda] shard $i/$NSHARD on GPU ${GPU_ARR[$i]} -> $part (log ${part%.npz}.log)"
        CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" python -m sda.assimilate "${COMMON[@]}" "$@" \
            --day-shard "$i/$NSHARD" --out "$part" > "$OUT/$ARM/${part%.npz}.log" 2>&1 &
        pids+=($!)
    done
    local rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    if [ "$rc" -ne 0 ]; then echo "[sda] a shard failed; see $OUT/$ARM/*_shard*.log" >&2; return 1; fi
    tail -n 3 "$OUT/$ARM/${out%.npz}_shard0of${NSHARD}.log" 2>/dev/null || true
    python -m sda.assimilate --merge "${parts[@]}" --out "$OUT/$ARM/$out" && rm -f "${parts[@]}"
}

for size in ${SIZES//,/ }; do
    CASE="$OUT/case_${size}.npz"
    COMMON=(--ckpt "$CKPT" --case "$CASE" --out-dir "$OUT/$ARM" --k "$K" --steps "$STEPS"
            --corrections "$CORR" --tau "$TAU" --gamma-scale "$GAMMA" --sampler "$SAMPLER"
            ${TERMS:+--terms "$TERMS"} $EXTRA)
    # 2. precision check on the first step, then control, then guided
    if [ ! -f "$OUT/$ARM/.precision_${size}.done" ]; then
        python -m sda.assimilate "${COMMON[@]}" --check-precision || echo "[sda] WARNING: bf16 guidance differs from fp32; consider --precision fp32"
        touch "$OUT/$ARM/.precision_${size}.done"
    fi
    if [ ! -f "$OUT/$ARM/samples_sda_${size}_free.npz" ]; then
        run_assim "samples_sda_${size}_free.npz" --guidance off
    fi
    if [ ! -f "$OUT/$ARM/samples_sda_${size}.npz" ]; then
        run_assim "samples_sda_${size}.npz" --guidance on --verbose
    fi
    # 3. diagnostics
    python -m eval.diagnostics_cond --samples "$OUT/$ARM/samples_sda_${size}.npz" --out "$OUT/$ARM/figs_cond_${size}"
    python -m eval.diagnostics_cond --samples "$OUT/$ARM/samples_sda_${size}_free.npz" --out "$OUT/$ARM/figs_cond_${size}_free"
    python -m sda.diagnostics_sda --samples "$OUT/$ARM/samples_sda_${size}.npz" \
        --control "$OUT/$ARM/samples_sda_${size}_free.npz" --out "$OUT/$ARM/figs_sda_${size}" \
        $([ "$ANIMATE" = 1 ] && echo "--animate --animate-panel")
done

echo "[sda] done -- $OUT/$ARM/figs_sda_<size>/report_sda.md"
