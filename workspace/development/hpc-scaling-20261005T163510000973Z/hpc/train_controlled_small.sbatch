#!/usr/bin/env bash
#SBATCH --job-name=pi-jepa-controlled-small
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=workspace/slurm/%x_%j.out
#SBATCH --error=workspace/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the prepared HPC checkout root.}"
: "${DATA_ROOT:?Export DATA_ROOT: parent containing both datasets.}"
: "${CACHE_ROOT:?Export CACHE_ROOT after CPU preparation of the derived caches.}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cache_root="$CACHE_ROOT"
if [[ "${STAGE_CACHE:-0}" == 1 ]]; then
  cache_root="${SLURM_TMPDIR:?STAGE_CACHE=1 requires node-local SLURM_TMPDIR}/pi-jepa-cache-${SLURM_JOB_ID:?}"
  mkdir -p "$cache_root"
  cp -a "$CACHE_ROOT/controlled" "$cache_root/"
fi
runs_root="${RUNS_ROOT:-$PWD/workspace/runs}"
checks_root="${CHECKS_ROOT:-$PWD/workspace/checks}/hpc-${SLURM_JOB_ID:?}"
run_dir="$runs_root/controlled-small-joint-true-fixed-b256-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%S%NZ)"
tracking="${WANDB_MODE:-disabled}"
extra_args=()
case "$tracking" in
  disabled) ;;
  online|offline) extra_args+=(--wandb-project "${WANDB_PROJECT:?Export WANDB_PROJECT for tracked runs.}") ;;
  *) printf 'Unsupported WANDB_MODE: %s\n' "$tracking" >&2; exit 2 ;;
esac
if [[ -n "${WANDB_ENTITY:-}" ]]; then extra_args+=(--wandb-entity "$WANDB_ENTITY"); fi

# Verify this exact model/batch on the allocated GPU before the full run.
srun uv run --frozen --no-sync python scripts/benchmark_training.py \
  --config configs/hpc_true_reset_small.json --dataset controlled --device cuda \
  --data-root "$DATA_ROOT" --cache-root "$cache_root" \
  --expected-gpu H200 --min-gpu-memory-gib 130 \
  --steps 3 --warmup 1 --cpu-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --output-root "$checks_root"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi

printf 'Training run: %s\n' "$run_dir"
exec srun uv run --frozen --no-sync python -m pi_jepa.train \
  --config configs/hpc_true_reset_small.json --dataset controlled --mode joint \
  --initial-conditions true_fixed --device cuda \
  --data-root "$DATA_ROOT" --cache-root "$cache_root" \
  --runs-root "$runs_root" --run-dir "$run_dir" \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-16}" --quiet \
  --wandb-group "${WANDB_GROUP:-hpc-true-reset-controlled}" \
  --wandb-mode "$tracking" "${extra_args[@]}"
