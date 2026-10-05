#!/usr/bin/env bash
#SBATCH --job-name=pi-jepa-train
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=workspace/slurm/%x_%j.out
#SBATCH --error=workspace/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
: "${DATA_ROOT:?Export DATA_ROOT before submission.}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mode="${TRAIN_MODE:-joint}"
dataset="${DATASET:-passive}"
tracking="${WANDB_MODE:-disabled}"
extra_args=()
case "$mode" in
  joint|jepa) ;;
  readout) extra_args+=(--pretrained "${JEPA_CHECKPOINT:?Set JEPA_CHECKPOINT to a completed JEPA checkpoint.}") ;;
  *) printf 'Unsupported TRAIN_MODE: %s\n' "$mode" >&2; exit 2 ;;
esac
case "$tracking" in
  disabled) ;;
  online|offline) extra_args+=(--wandb-project "${WANDB_PROJECT:?Set WANDB_PROJECT for tracked training.}") ;;
  *) printf 'Unsupported WANDB_MODE: %s\n' "$tracking" >&2; exit 2 ;;
esac
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  extra_args+=(--wandb-entity "$WANDB_ENTITY")
fi
runs_root="${RUNS_ROOT:-$PWD/workspace/runs}"
run_name="${dataset}-${mode}-${SLURM_JOB_ID:?}-$(date -u +%Y%m%dT%H%M%S%NZ)"
run_dir="${RUN_DIR:-$runs_root/$run_name}"
if [[ "${RESUME:-0}" == 1 ]]; then
  : "${RUN_DIR:?Set RUN_DIR to the existing run for explicit resume.}"
  extra_args+=(--resume)
fi
printf 'Run directory: %s\n' "$run_dir"

exec srun uv run --frozen --no-sync python -m pi_jepa.train \
  --config "${CONFIG:-configs/wandb.json}" --dataset "$dataset" --mode "$mode" --device cuda \
  --data-root "$DATA_ROOT" --runs-root "$runs_root" --run-dir "$run_dir" \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-8}" --quiet \
  --wandb-mode "$tracking" "${extra_args[@]}"
