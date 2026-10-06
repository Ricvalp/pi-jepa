# HPC training configuration and performance checks

The six H200 jobs described in [hpc/README.md](../hpc/README.md) use the same
batch-256, joint, fixed true-reset protocol. Small/medium/large scale both the
encoder and predictor while keeping the 32-dimensional latent and physical
readout interface fixed. The implementation, configuration and launchers are
included in the main repository.

The principal changes address repeated work:

- Read fixed windows from memory-mapped unpacked learning arrays instead of
  decompressing entire RGB episodes each update.
- Precompute physical targets once from fixed true training resets and recorded
  actions, with the same float64 solver. Training gathers target endpoints from
  this derived cache. This optimization is specific to fixed resets; a learned
  reset trajectory changes during training and cannot use that fixed cache.
- Run encoder/predictor operations under BF16 autocast on supporting CUDA devices.
  Cast latent values back to float32 for SIGReg, prediction losses and the physical
  readout. Cached simulator targets and the training physical residual retain
  float64; diagnostic residual summaries keep their existing float32 calculation.
- Enable cuDNN autotuning for repeated image shapes and use a common batch 256
  across all six jobs. Node-local cache staging is optional because shared I/O
  capacity and local disk availability depend on the allocation.

Autocast selects precision by operation while model weights remain float32;
the backward pass runs outside the autocast context. This follows
[PyTorch's AMP guidance](https://docs.pytorch.org/docs/2.14/amp.html), which
describes lower-precision neural operations alongside higher-precision reductions.
The H200 preflight must verify the actual BF16 training path. The local CPU
build did not support the required BF16 convolution backward path; CPU checks
therefore use the explicit benchmark override `--precision float32`. That
override is not present in the six H200 launchers.

## Measured CPU components

The [saved component results](benchmarks/cache_cpu_results.json)
use **two CPU threads**, 16 deterministic real training episodes per corpus,
and a warm OS file cache. Load/solver values below are medians of **three
repetitions** at batch 16; tiny target lookups use 101 repetitions. The episode pixels, forces, parameters and reset values are
unchanged; the derived test copies only limit the selected episode set.

| Corpus | NPZ loading and cropping | Memory-mapped loading and cropping | Loading speed ratio | Float64 target integration | Cached target lookup |
|---|---:|---:|---:|---:|---:|
| Passive | 125.83 ms | 4.00 ms | 31.5× | 163.55 ms | 0.0159 ms |
| Controlled | 107.50 ms | 4.97 ms | 21.6× | 160.81 ms | 0.0158 ms |

Cached target values were **bitwise identical** to recomputed float64 states
for every tested CPU endpoint, with maximum absolute difference zero. Cached
CPU float64 trigonometry can differ from live GPU float64 simulation at roundoff
level; no GPU equivalence measurement was performed here. These
measurements isolate loading/cropping and simulator/lookup costs. They exclude
neural forward/backward passes, GPU transfer, cold-storage I/O, and periodic
diagnostics. They are not measurements of GPU or end-to-end training speedup,
and the ratios should not be extrapolated to batch 256 or another filesystem.

From the production manifest lengths, the complete train/validation learning
arrays require approximately **16.85 GiB passive + 16.70 GiB controlled =
33.55 GiB**, plus small indexes/action arrays. Fixed-target arrays require
approximately **18.80 MiB + 18.62 MiB**, about 38 MiB total. Reserve at least
about 20 GiB for one corpus when staging to node-local storage, or 40 GiB for
both shared caches, allowing room for metadata and filesystem overhead.

## Allocation preflight

These choices do not establish a speedup or fit on a particular GPU without a
measurement. Every launcher runs the exact configured model and batch in its
allocation before starting training, using one warmup and three measured updates.
The preflight records the actual device, timing and peak GPU memory; it never
silently reduces the batch or carries its temporary updated model into the run.
The launchers require an `H200` device name and at least 130 GiB reported memory,
so a generic GPU allocation cannot silently substitute a smaller/different GPU.
Short preflights can expose shape, precision, allocation and memory problems, but
are not convergence tests or reliable full-run duration guarantees. Cache
preparation and optional staging have separate startup costs.

From the repository root, with its environment installed and `DATA_ROOT` /
`CACHE_ROOT` set as in the HPC guide, run a measurement inside a GPU allocation:

```bash
uv run --frozen --no-sync python scripts/benchmark_training.py \
  --config configs/hpc_true_reset_large.json --dataset controlled --device cuda \
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT" \
  --expected-gpu H200 --min-gpu-memory-gib 130 \
  --steps 3 --warmup 1 --cpu-threads 16 --output-root workspace/checks/hpc
```

Keep the output's hardware/configuration with any reported throughput. A CPU
check or a smaller local GPU does not validate batch 256 on an H200. No actual
H200 measurements have been obtained in this workspace. The configs use
`training.cache_batch_size=64` for neural inference during periodic diagnostics;
this is separate from the optimization batch of 256.

## Remaining costs to profile

Training still loads windows in the main process, stacks them on the CPU and
performs synchronous host-to-device transfers. There is no background loader
pool or asynchronous prefetch pipeline. Memory mapping removes decompression,
but CPU assembly, page faults and transfer can still limit GPU utilization.

Validation, fixed-reference diagnostics, the long trajectory figures, logging
and checkpoint writes run synchronously at their configured intervals. Neural
diagnostics use float32, and validation still integrates its truth-free reset
prior with float32 RK4. Fixed training-target caching does not eliminate these
costs. Compare the new `train/performance/*` stage measurements with the
cumulative `train/updates_per_second`, which includes periodic work already
incurred. A fast isolated update does not imply a proportionate full-run gain.

`torch.compile` is not enabled; profile the actual H200 kernels and Python
overhead before deciding whether its compilation/startup costs are worthwhile.
The campaign uses one GPU per independent job, so distributed data parallelism
is not needed for these allocations. These remain measured follow-up choices,
not optimizations assumed to have been implemented.

All three size configurations keep the learning rate and 10,000-update budget
unchanged. Batch 256 processes four times as many episode-window draws as the
older batch-64 experiment and changes the finite-sample SIGReg estimator. Treat
these as a matched six-run scaling experiment, not a pure runtime optimization
of the old learning protocol. The [scientific description](experiment.md) covers
the fixed-reset supervision and the [metric reference](metrics.md) defines
physical, latent and optimization diagnostics.
