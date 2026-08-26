#!/bin/bash
#SBATCH --job-name=tau_bicubic
#SBATCH --output=tau_interp_check/logs/build_bicubic_%j.out
#SBATCH --error=tau_interp_check/logs/build_bicubic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HDF5_USE_FILE_LOCKING=FALSE
PY=/conda/jvelasco/miniforge3/envs/datorch/bin/python
exec "$PY" tau_interp_check/make_bicubic_tau.py --workers "${SLURM_CPUS_PER_TASK}"
