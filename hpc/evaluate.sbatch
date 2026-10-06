#!/usr/bin/env bash
#SBATCH --job-name=pi-jepa-eval
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=workspace/slurm/%x_%j.out
#SBATCH --error=workspace/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
export DATA_ROOT="${DATA_ROOT:-/hpc/home/phi/rvalperga/data/pi-jepa}"
: "${JOINT_CHECKPOINT:?Set JOINT_CHECKPOINT to the completed joint checkpoint.}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
extra_args=()
if [[ -n "${POSTHOC_CHECKPOINT:-}" ]]; then
  extra_args+=(--posthoc-checkpoint "$POSTHOC_CHECKPOINT")
fi

exec srun uv run --frozen --no-sync python -m pi_jepa.evaluate \
  --config "${CONFIG:-configs/base.json}" --data-root "$DATA_ROOT" \
  --output-root "${RESULTS_ROOT:-$PWD/workspace/evaluations}" \
  --dataset controlled --joint-checkpoint "$JOINT_CHECKPOINT" \
  --device cuda --cpu-threads "${SLURM_CPUS_PER_TASK:-8}" "${extra_args[@]}"
