#!/usr/bin/env bash
#SBATCH --job-name=pi-jepa-data
#SBATCH --partition=SET_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=workspace/slurm/%x_%j.out
#SBATCH --error=workspace/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
: "${DATA_ROOT:?Export DATA_ROOT before submission.}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec srun uv run --frozen --no-sync python -m pi_jepa.data \
  --config "${CONFIG:-configs/base.json}" --data-root "$DATA_ROOT" \
  --dataset "${DATASET:-both}" --workers "${SLURM_CPUS_PER_TASK:-4}"
