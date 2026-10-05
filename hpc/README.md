# Slurm launchers

Submit from the repository root. GPU scripts use Peano's documented `gpuq` /
`--gres=gpu:1` shape; prefer H200. These options alone do not guarantee an H200:
confirm the site's current node constraints and pass any required `--constraint`,
`--account`, or `--qos` options to `sbatch`. The CPU partition is unknown;
`prepare.sbatch` deliberately requires `sbatch --partition=YOUR_CPU_PARTITION`.
Override resource headers at submission for another cluster. No jobs are submitted
by setup or tests, and the launchers have not been tested on a compute allocation.

The main repository contains the GroupNorm/LayerNorm encoder and fixed true-reset
diagnostic described in the [README](../README.md). Submit these launchers from
this repository's root using its prepared environment. Checkpoints use format 5;
explicit resume requires the same architecture and reset protocol.

On the login host, recreate this project's environment and validate it before
submission (do not copy a workstation `.venv`):

```bash
uv sync --locked --extra tracking
uv run --frozen --no-sync python scripts/check_environment.py --device cpu
mkdir -p workspace/slurm
sbatch hpc/smoke.sbatch
```

Wait for the actual GPU smoke job to succeed before launching long training.
It checks CUDA access, differentiable cart-pole integration, a model update and
inference, and headless PNG/GIF writing in `workspace/checks/`. This renderer uses
Pillow and requires no display server or EGL mapping. Jobs use the prepared
environment with `--frozen --no-sync`; they never install packages. Keep the code,
configuration, and environment stable for queued/running jobs; use a separate
checkout for concurrent development. `uv` and any site-required CUDA/module setup
must already be available to the batch environment.

Export paths so Slurm inherits them. Absolute paths can point to shared scratch;
defaults for outputs are under the checkout's `workspace/`. Slurm opens its log
files before running the script, so create `workspace/slurm` **before** every first
submission from a fresh checkout. W&B is disabled by default. To enable it,
authenticate outside the job (`uv run --frozen --no-sync wandb login`) and export
`WANDB_MODE=online` (or `offline`) and `WANDB_PROJECT`; optionally export
`WANDB_ENTITY`. Never put credentials in launchers.

## Fixed true-reset diagnostic

This joint-only experiment fixes the exact initial state of each **training**
episode and simulates its subsequent targets. It does not read stored future
states or validation/test truth. It is explicitly more supervised than the
learned-reset experiment; see the [scientific description](../docs/experiment.md).

From the repository root, use the existing datasets and run parent:

```bash
export DATA_ROOT="$PWD/workspace/data"
export RUNS_ROOT="$PWD/workspace/runs"
export WANDB_MODE=online
export WANDB_PROJECT=physics-jepa-cartpole
mkdir -p workspace/slurm

DATASET=passive sbatch hpc/true_reset_diagnostic.sbatch

# Submit later, after inspecting the passive diagnostic.
DATASET=controlled sbatch hpc/true_reset_diagnostic.sbatch
```

On a different machine, export absolute shared-scratch paths if appropriate.
Both jobs start fresh models independently. Their default W&B group is
`causal-true-reset-diagnostic`, overrideable with `WANDB_GROUP`. The job name and
run directory distinguish the extra-supervision diagnostic. `CONFIG` can select
a compatible diagnostic configuration; the launcher always explicitly selects
`--mode joint --initial-conditions true_fixed`. No jobs have been submitted.

## Learned-reset comparator and ordinary stages

Use the same GroupNorm/LayerNorm implementation for the matched learned-reset
comparator. `configs/wandb.json` retains learned resets and the same seed, data,
batch size, optimizer settings and update budget as the diagnostic. With the
paths already exported above:

```bash
DATASET=passive TRAIN_MODE=joint sbatch hpc/train.sbatch
```

Repeat with `DATASET=controlled` when needed. The resolved config and protocol
flags identify the learned-reset run. This matched comparison is needed to
separate reset effects from the normalization change; comparing only with an old
BatchNorm run changes both variables.

Both stages train independent physics-informed models from scratch; there is no
checkpoint transfer. `DATASET` defaults to `passive` and `TRAIN_MODE` to `joint`.
Passive additionally supports `TRAIN_MODE=jepa` for the JEPA-only baseline;
post-hoc `TRAIN_MODE=readout` applies only to the controlled corpus.

If you prefer to queue both jobs immediately, ordering them by successful
completion (rather than by scientific result), replace the two submissions with:

```bash
passive_id=$(DATASET=passive TRAIN_MODE=joint sbatch --parsable hpc/train.sbatch)
passive_id=${passive_id%%;*}
DATASET=controlled TRAIN_MODE=joint sbatch --dependency="afterok:$passive_id" hpc/train.sbatch
```

For a new machine without these data, submit preparation before training:

```bash
prepare_id=$(DATASET=both sbatch --parsable --partition=YOUR_CPU_PARTITION hpc/prepare.sbatch)
prepare_id=${prepare_id%%;*}
DATASET=passive TRAIN_MODE=joint sbatch --dependency="afterok:$prepare_id" hpc/train.sbatch
```

`DATA_ROOT` is the container with `passive/` and `controlled/`, not a corpus
directory itself. `afterok` checks successful process completion, not scientific
performance. Each training job prints a unique directory including dataset,
mode, Slurm job ID, and UTC timestamp. Record job IDs and directories. `CONFIG`
can override `configs/wandb.json`. No training job installs dependencies.

After the JEPA run completes, pass its actual checkpoint to the frozen-readout
stage. Do not infer a checkpoint path from the Slurm job ID:

```bash
export JEPA_CHECKPOINT="$RUNS_ROOT/REPLACE_WITH_JEPA_RUN/final.pt"
DATASET=controlled TRAIN_MODE=readout sbatch hpc/train.sbatch
```

For an explicit resume, set `RUN_DIR` to the original directory and `RESUME=1`,
with the same stage, data, configuration, and optional JEPA source. Ensure the
original job has stopped; there must be only one writer per run. New runs refuse
an existing directory.

```bash
export JOINT_CHECKPOINT="$RUNS_ROOT/REPLACE_WITH_JOINT_RUN/final.pt"
export POSTHOC_CHECKPOINT="$RUNS_ROOT/REPLACE_WITH_READOUT_RUN/final.pt"
export RESULTS_ROOT="$PWD/workspace/evaluations"
sbatch hpc/evaluate.sbatch
```

The physical evaluator in `evaluate.sbatch` applies to controlled models. For
passive models, use `pi_jepa.evaluate_latents --dataset passive --checkpoint ...`
as shown in the main README.

Evaluation uses a GPU for CNN inference and writes a separate timestamped result
directory with checkpoint provenance. Preparation uses CPU only. Jobs run one
task and bound library/PyTorch CPU threads by their allocation; no data-loader or
evaluation worker pools are spawned. Preparation bounds rendering/compression
workers by the CPU allocation. Training reserves 24 hours and evaluation
two hours as starting allocations, including final checkpoint/report writes;
adjust after measuring on the actual cluster. `srun` launches the one visible
Python command in each script, and CUDA's assigned device visibility is preserved.
