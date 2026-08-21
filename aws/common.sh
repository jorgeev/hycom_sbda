#!/usr/bin/env bash
# Shared plumbing for aws/launch.sh and aws/sample.sh: env-var defaults, S3
# staging in, and result persistence out. Sourced, never executed.
#
# The path env vars (HYCOM_STORE, GULFSTREAM_MANIFEST, OUT_DIR, CACHE_DIR) are
# NOT read by these scripts -- they are read by the ${VAR:-default} placeholders
# in the YAML, expanded by diffusion/config.py:expandvars. So they only need to
# be exported; the config resolves them. Unset, every one falls back to the
# on-prem path that used to be hardcoded, which is what keeps train.slurm
# working unchanged.

# OUT_DIR is where runs land. Default to a mounted fast disk rather than the
# image's own filesystem, which is thrown away with the container.
export OUT_DIR="${OUT_DIR:-/mnt/runs}"
# Caches (_norm_cache.json, _clim_cache.npz) are written at run time and must
# not go into the image. Keeping them under OUT_DIR means a mounted volume
# carries them between runs, so the ~minutes of climatology fitting is paid once.
export CACHE_DIR="${CACHE_DIR:-$OUT_DIR/cache}"
mkdir -p "$OUT_DIR" "$CACHE_DIR"

# Optional S3 round-trip for results. OUT_S3 unset = purely local.
OUT_S3="${OUT_S3:-}"
SYNC_EVERY="${SYNC_EVERY:-900}"     # seconds between background syncs; 0 = off
STAGE_DIR="${STAGE_DIR:-}"          # local NVMe to stage s3:// stores into

_have_aws() { command -v aws >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# stage_in -- copy an s3:// store to local disk ONCE, before the ranks start.
# ---------------------------------------------------------------------------
# Why this exists: every DDP rank builds its own ZarrWindowDataset and preloads
# the whole time axis into RAM (diffusion/data.py:_load_zscore/_load_anomaly).
# Pointed straight at s3://, that is N independent full downloads of the store
# on every start. Staging once to instance NVMe makes it one download shared by
# all ranks, and makes a resume after a spot interruption near-instant.
#
# Small stores and smoke tests are fine reading s3:// directly -- leave
# STAGE_DIR unset for those.
stage_in() {
  [[ -n "$STAGE_DIR" ]] || return 0
  if ! _have_aws; then
    echo "[stage] STAGE_DIR set but the aws CLI is missing; skipping" >&2
    return 0
  fi
  mkdir -p "$STAGE_DIR"

  # gom_nemo: a single store directory.
  if [[ "${HYCOM_STORE:-}" == s3://* ]]; then
    local dst="$STAGE_DIR/$(basename "${HYCOM_STORE%/}")"
    echo "[stage] $HYCOM_STORE -> $dst"
    aws s3 sync "$HYCOM_STORE" "$dst"
    export HYCOM_STORE="$dst"
  fi

  # gulfstream: a manifest JSON plus the member stores BESIDE it. The manifest
  # names its members with paths relative to its own directory, so the whole
  # directory has to come across together or the members resolve to nothing.
  if [[ "${GULFSTREAM_MANIFEST:-}" == s3://* ]]; then
    local src_dir="${GULFSTREAM_MANIFEST%/*}"
    local name="${GULFSTREAM_MANIFEST##*/}"
    local dst_dir="$STAGE_DIR/$(basename "$src_dir")"
    echo "[stage] $src_dir -> $dst_dir  (manifest + member stores)"
    aws s3 sync "$src_dir" "$dst_dir"
    export GULFSTREAM_MANIFEST="$dst_dir/$name"
  fi
}

# ---------------------------------------------------------------------------
# restore_out -- pull a previous run back down so training resumes into it.
# ---------------------------------------------------------------------------
# diffusion/train.py resumes implicitly: if <out>/ckpt.pt exists and is
# non-empty it loads it and continues from the saved step. Putting the previous
# checkpoint back in place before launch is therefore the entire resume story.
restore_out() {
  [[ -n "$OUT_S3" ]] || return 0
  _have_aws || { echo "[restore] aws CLI missing; skipping" >&2; return 0; }
  echo "[restore] $OUT_S3 -> $OUT_DIR"
  aws s3 sync "$OUT_S3" "$OUT_DIR"
}

# ---------------------------------------------------------------------------
# sync_out / start_background_sync -- persist results back to S3.
# ---------------------------------------------------------------------------
sync_out() {
  [[ -n "$OUT_S3" ]] || return 0
  _have_aws || return 0
  echo "[sync] $OUT_DIR -> $OUT_S3"
  # Never let a failed sync mask the job's own exit status.
  aws s3 sync "$OUT_DIR" "$OUT_S3" || echo "[sync] FAILED (results remain in $OUT_DIR)" >&2
}

_SYNC_PID=""
start_background_sync() {
  [[ -n "$OUT_S3" && "$SYNC_EVERY" -gt 0 ]] || return 0
  _have_aws || return 0
  ( while sleep "$SYNC_EVERY"; do
      aws s3 sync "$OUT_DIR" "$OUT_S3" >/dev/null 2>&1 || true
    done ) &
  _SYNC_PID=$!
  echo "[sync] background sync every ${SYNC_EVERY}s (pid $_SYNC_PID)"
}

stop_background_sync() {
  [[ -n "$_SYNC_PID" ]] || return 0
  kill "$_SYNC_PID" 2>/dev/null || true
  wait "$_SYNC_PID" 2>/dev/null || true
  _SYNC_PID=""
}

# One final sync on ANY exit, including a crash or a spot reclamation signal --
# a three-day run whose results stay only on an ephemeral disk is a lost run.
on_exit() {
  local rc=$?
  stop_background_sync
  sync_out
  return $rc
}
