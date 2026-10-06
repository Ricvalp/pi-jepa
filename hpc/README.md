# H200 training: six matched model-size jobs

The six launchers train **joint PI-JEPA with fixed true training resets**, crossing
passive/controlled data with small/medium/large models. Each requests **one GPU,
16 CPUs, 128 GiB host RAM and 24 hours**, targeting an **H200 with 141 GB GPU
memory**. The host-memory request is separate from GPU memory. All use batch
**256**, seed 42 and 10,000 updates, with the same learning-rate schedule.

All source, configs and launchers are in the main repository. Run the following
commands from its root on Peano, `/hpc/home/phi/rvalperga/pi-jepa`. Install the
locked environment there; datasets remain in the separately configured data root.

## Models and launchers

| Size | Encoder | Passive predictor hidden layers | Controlled predictor width / layers / heads | Passive joint parameters | Controlled joint parameters |
|---|---|---|---|---:|---:|
| [Small](../configs/hpc_true_reset_small.json) | ResNet-18 | 128 × 2 | 192 / 3 / 3 | 12,151,301 | 14,178,757 |
| [Medium](../configs/hpc_true_reset_medium.json) | ResNet-34 | 256 × 3 | 256 / 4 / 4 | 22,731,461 | 27,404,229 |
| [Large](../configs/hpc_true_reset_large.json) | ResNet-50 | 512 × 4 | 384 / 6 / 6 | 31,628,997 | 46,947,269 |

Every encoder uses GroupNorm and a hidden LayerNorm projector. Latent dimension
stays 32; physical readout stays `32→64→5`. Parameter counts exclude optimizer
state and fixed reset buffers. Both encoder and predictor grow with size. The
three [size configs](../configs/) contain identical training settings apart from
`model.size`; corpus is selected literally in each launcher.

| Dataset | Small | Medium | Large |
|---|---|---|---|
| Passive | [train_passive_small.sbatch](train_passive_small.sbatch) | [train_passive_medium.sbatch](train_passive_medium.sbatch) | [train_passive_large.sbatch](train_passive_large.sbatch) |
| Controlled | [train_controlled_small.sbatch](train_controlled_small.sbatch) | [train_controlled_medium.sbatch](train_controlled_medium.sbatch) | [train_controlled_large.sbatch](train_controlled_large.sbatch) |

The scheduler headers use Peano's existing `gpuq` / `--gres=gpu:1` convention.
Those fields alone do **not** select an H200. Verify the allocation policy and
add the site's documented `--constraint`, `--account` or `--qos` when necessary;
no unverified GPU-type constraint is hardcoded. The launchers benchmark the exact
model and batch on their allocated GPU before starting full training. Preflight
requires a device name containing `H200` and at least **130 GiB** of reported
GPU memory, otherwise it exits before training. This distinguishes the decimal
141 GB product capacity from the binary GiB threshold. An H200
allocation has not been tested here, so fit and throughput remain allocation
checks rather than promises.

## Prepare the environment, datasets and caches once

On the login host, update the repository, install its locked environment and
verify both the preparation scripts and project imports:

```bash
cd /hpc/home/phi/rvalperga/pi-jepa
git pull --ff-only
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --extra tracking
test -f scripts/prepare_training_cache.py
test -f scripts/benchmark_training.py
.venv/bin/python -c 'import pi_jepa.data, pi_jepa.training_cache; print(pi_jepa.data.__file__); print(pi_jepa.training_cache.__file__)'
.venv/bin/python scripts/check_environment.py --device cpu
.venv/bin/wandb login
mkdir -p workspace/slurm
```

Updating the repository supplies missing scripts; `uv sync` installs the project
and dependencies into `.venv`. `uv run --no-sync` does neither. The printed module
paths should point inside this repository's `src/pi_jepa/`. The preparation
launcher checks and uses the same `.venv/bin/python` without installing packages
inside the job. W&B login is needed only when using online tracking.

The defaults are `DATA_ROOT=/hpc/home/phi/rvalperga/data/pi-jepa` and
`CACHE_ROOT=$DATA_ROOT/cache`. `DATA_ROOT` must be the actual parent of `passive/`
and `controlled/`. Override it if the existing corpora are elsewhere; for example,
use `/hpc/home/phi/rvalperga/data` if that directory already contains both corpora.
Set `CACHE_ROOT` to an existing compatible cache directory when reusing one.
These settings do not move data or regenerate datasets. The block below selects
the defaults explicitly and replaces earlier values in the current shell:

```bash
export DATA_ROOT="/hpc/home/phi/rvalperga/data/pi-jepa"
export CACHE_ROOT="$DATA_ROOT/cache"
export RUNS_ROOT="$PWD/workspace/runs"
export WANDB_PROJECT=physics-jepa-cartpole
export WANDB_MODE=online
```

If complete datasets already exist, skip generation and prepare the caches below.
For fresh data, generate both corpora **once** with the preparation launcher.
Peano has no CPU queue, so preparation requests a GPU node through `gpuq`.
The processing uses that node's CPUs; the allocated GPU is not used:

```bash
DATASET=both CONFIG=configs/base.json sbatch --partition=gpuq --gres=gpu:1 hpc/prepare.sbatch
```

Each corpus directory must either be absent for fresh generation, or contain a
complete matching dataset with its manifest and episode files. Complete matching
corpora are reused. Incomplete or incompatible outputs are refused; do not
pre-create empty `passive/` or `controlled/` directories or start duplicate
generation jobs. Existing complete production data need no regeneration.

With complete datasets in place (wait for generation to finish successfully if
needed), prepare both caches once on an allocated GPU node before submitting the
six training jobs. The manifest checks below also catch an incorrect data root.
Cache preparation computes on the CPUs:

```bash
test -f "$DATA_ROOT/passive/manifest.json"
test -f "$DATA_ROOT/controlled/manifest.json"
srun --partition=gpuq --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 \
  uv run --frozen --no-sync python scripts/prepare_training_cache.py \
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT" --dataset both \
  --fixed-targets --cpu-threads 8
```

The cache has `CACHE_ROOT/{passive,controlled}/{learning,fixed_targets}/`.
Learning caches store unpacked arrays for memory-mapped window access instead of
repeated full-episode NPZ decompression. Fixed-target caches simulate each
training episode from its true reset once; no stored future truth arrays are
used. Immutable source data are not changed. Complete compatible caches are
reused; training refuses missing, incomplete or mismatched caches rather than
building them inside a GPU job. Allow substantial disk space: uncompressed RGB
is approximately 18 GB per corpus, depending on split sizes and truncation.
Reserve about **40 GiB for both caches**, separately from the source datasets
and training outputs.

Shared caches are used directly by default. If the site provides sufficiently
large node-local storage through `SLURM_TMPDIR`, enable staging explicitly:

```bash
export STAGE_CACHE=1
```

Each job then copies only its selected corpus into a job-specific local cache
before benchmarking. This trades startup I/O and local disk space for local
reads. Six concurrent copies can pressure shared storage; leave staging off
when the shared filesystem already serves the workload well. Raw data remain
at `DATA_ROOT` for provenance and diagnostics. Cache preparation runs once in its
separate allocation and is never repeated by these training launchers.

## Submit the six independent jobs

Create the log directory before `sbatch`: Slurm opens output files before the
script runs. Submit only after cache preparation succeeds:

```bash
mkdir -p workspace/slurm
sbatch hpc/train_passive_small.sbatch
sbatch hpc/train_passive_medium.sbatch
sbatch hpc/train_passive_large.sbatch
sbatch hpc/train_controlled_small.sbatch
sbatch hpc/train_controlled_medium.sbatch
sbatch hpc/train_controlled_large.sbatch
```

The six runs are independent and can be queued together. They do not transfer
weights or require dependencies on each other. If you prefer to inspect passive
results first, submit the first three and leave the controlled commands for
later. Training starts only when you submit its launcher.

Every job first runs `scripts/benchmark_training.py` for one warmup and three
measured updates with its exact size, corpus, batch 256 and BF16 configuration.
It records allocated GPU/model information, runtime and peak memory under
`workspace/checks/hpc-JOBID/benchmark-DATASET-SIZE-b256-UTC/benchmark.json`
(override the parent with `CHECKS_ROOT`). This does
not write to a training run or log to W&B. A failing preflight stops that job;
there is no automatic batch-size reduction. Full training then starts fresh,
with a run name containing dataset, size, fixed-reset protocol, batch, job ID
and UTC time. The benchmark's updated temporary model is not reused.

For an allocation check without the full run, use an existing launcher with
`PREFLIGHT_ONLY=1`, for example:

```bash
PREFLIGHT_ONLY=1 sbatch hpc/train_controlled_large.sbatch
```

This is an optional job submission, not something setup performs automatically.
Inspect the recorded GPU name and available memory alongside the enforced
H200/memory check. A short preflight checks numerical/shape compatibility and
memory for several updates; it does not certify a full 24-hour run or convergence.

W&B is disabled by default inside launchers unless `WANDB_MODE` is exported.
Use `online` or `offline` with `WANDB_PROJECT`, and optionally `WANDB_ENTITY`.
Groups default to `hpc-true-reset-passive` and `hpc-true-reset-controlled`; model
size appears in the config and run name. `WANDB_GROUP` overrides the group.
`WANDB_MODE=disabled` retains local diagnostics without uploading. All six
launchers use the already prepared environment with `--frozen --no-sync`, bound
library threads, and preserve Slurm's CUDA visibility.

## Interpretation and performance choices

BF16 autocast applies to the encoder and predictor. Latents, JEPA/SIGReg losses,
and the physical readout stay float32. Cached simulator targets and the training
physical residual stay float64. Diagnostic residual summaries retain their
existing float32 calculation. cuDNN autotuning is enabled for the fixed image shapes.
Prepared learning/target caches avoid repeated decompression and reset-to-window
integration in the training loop. See the
[performance notes](../docs/hpc_performance.md) and
[metric reference](../docs/metrics.md) for evidence and logging definitions.

Increasing batch 64 to 256 changes the experiment: 10,000 updates now sample
2,560,000 episode-windows rather than 640,000. These are draws with replacement
between updates, not that many unique episodes. SIGReg's finite-sample statistic
also depends on batch size. All six new jobs are matched at batch 256, but are not
an equal-data-exposure comparison with older batch-64 runs. The learning rate
remains `3e-4` with 500-update warmup and cosine decay to `3e-5`; no automatic
linear scaling is applied. Fixed true resets remain additional training
supervision and validation still uses a truth-free reset prior.

Checkpoint format 6 records the model size; resume requires the same architecture,
reset protocol and scientific configuration. The six campaign launchers create
fresh runs. To resume an interrupted run, use the ordinary Python CLI with its
original config, size, dataset and `--resume --run-dir PATH_TO_RUN`, after its
previous writer has stopped.

The existing `train.sbatch`, `true_reset_diagnostic.sbatch`, `prepare.sbatch`,
`smoke.sbatch` and `evaluate.sbatch` remain available for their individual stages.
The preparation launcher uses `gpuq` with one GPU requested, as required on
Peano. Tests and shell parsing are not evidence of an actual
H200 allocation. No scheduler submission or full training is performed here.
