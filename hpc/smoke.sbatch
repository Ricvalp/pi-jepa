#!/usr/bin/env bash
#SBATCH --job-name=pi-jepa-smoke
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=workspace/slurm/%x_%j.out
#SBATCH --error=workspace/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec srun uv run --frozen --no-sync python scripts/check_environment.py \
  --device cuda --cpu-threads "${SLURM_CPUS_PER_TASK:-4}" \
  --output-root "${CHECKS_ROOT:-$PWD/workspace/checks}"
