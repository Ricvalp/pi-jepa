#!/usr/bin/env bash
#SBATCH --job-name=pi-jepa-data
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=workspace/slurm/%x_%j.out
#SBATCH --error=workspace/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
export DATA_ROOT="${DATA_ROOT:-/hpc/home/phi/rvalperga/data}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python_bin="$PWD/.venv/bin/python"
if [[ ! -x "$python_bin" ]] || ! "$python_bin" -c 'import pi_jepa.data'; then
  printf 'Project environment is missing or incomplete in %s. Before submitting, run: UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra tracking\n' "$PWD" >&2
  exit 1
fi

# Peano has no CPU partition; generation uses CPUs on this GPU allocation.
exec srun "$python_bin" -m pi_jepa.data \
  --config "${CONFIG:-configs/base.json}" --data-root "$DATA_ROOT" \
  --dataset "${DATASET:-both}" --workers "${SLURM_CPUS_PER_TASK:-4}"
