#!/bin/bash
#SBATCH --job-name=tau_spectra
#SBATCH --output=tau_interp_check/logs/spectra_%j.out
#SBATCH --error=tau_interp_check/logs/spectra_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${SLURM_TMPDIR:-$HOME/.cache}/matplotlib"
PY=/conda/jvelasco/miniforge3/envs/datorch/bin/python
"$PY" tau_interp_check/compute_spectra.py
"$PY" tau_interp_check/plot_compare.py --npz tau_interp_check/figs/tau_spectra_compare.npz
exec "$PY" tau_interp_check/plot_compare.py --npz tau_interp_check/figs/tau_spectra_compare_tukey02.npz
