# H200 training: six matched model-size jobs

The six launchers train **joint PI-JEPA with fixed true training resets**, crossing
passive/controlled data with small/medium/large models. Each requests **one GPU,
16 CPUs, 128 GiB host RAM and 24 hours**, targeting an **H200 with 141 GB GPU
memory**. The host-memory request is separate from GPU memory. All use batch
**256**, seed 42 and 10,000 updates, with the same learning-rate schedule.

This HPC version is prepared separately while the existing local run continues.
Locally it lives in
`workspace/development/hpc-scaling-20261005T163510000973Z/`. Transfer this version's
source/configs and lockfile to the cluster, then run the following commands from
that prepared checkout's root. Recreate its environment there; do not copy a
workstation `.venv` or change the source used by an active job.

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

## Prepare the environment and derived caches once

On the login host, install the locked environment and authenticate separately:

```bash
uv sync --locked --extra tracking
uv run --frozen --no-sync python scripts/check_environment.py --device cpu
uv run --frozen --no-sync wandb login
mkdir -p workspace/slurm
```

Set shared paths; `DATA_ROOT` contains `passive/` and `controlled/`. Existing
production datasets are reused, without regeneration:

```bash
export DATA_ROOT="/absolute/shared/path/to/data"
export CACHE_ROOT="/absolute/shared/path/to/pi-jepa-training-cache"
export RUNS_ROOT="/absolute/shared/path/to/pi-jepa-runs"
export WANDB_PROJECT=physics-jepa-cartpole
export WANDB_MODE=online
```

Build the caches **once on a CPU allocation before submitting the six GPU jobs**.
Use your site's CPU allocation command; this command itself does not allocate
resources. Eight threads must fit that allocation:

```bash
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

Shared caches are used directly by default. If the site provides sufficiently
large node-local storage through `SLURM_TMPDIR`, enable staging explicitly:

```bash
export STAGE_CACHE=1
```

Each job then copies only its selected corpus into a job-specific local cache
before benchmarking. This trades startup I/O and local disk space for local
reads. Six concurrent copies can pressure shared storage; leave staging off
when the shared filesystem already serves the workload well. Raw data remain
at `DATA_ROOT` for provenance and diagnostics. Cache preparation belongs on the
CPU allocation and is never repeated by these GPU launchers.

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
later. No jobs are submitted by the implementation or setup instructions.

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
The CPU partition is site-specific; do not submit `prepare.sbatch` without an
appropriate `--partition`. Tests and shell parsing are not evidence of an actual
H200 allocation. No scheduler submission or full training is performed here.
