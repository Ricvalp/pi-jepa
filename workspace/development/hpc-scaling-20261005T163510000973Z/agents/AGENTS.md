# Instructions for coding agents

These are defaults for research projects. Preserve the repository's existing
conventions where they make sense; do not restructure it just to match this file.
Training, Slurm, and robotics sections apply only when those features are used.
Do not introduce them into an unrelated project.

## Research code, not production scaffolding

- Favor simplicity, readability, and explicit code that a researcher can follow.
  Prefer a clear training loop and small functions over frameworks, registries,
  deep inheritance, or layers of generic wrappers.
- Reuse established libraries for standard algorithms. Make scientific choices
  explicit: objectives, units, normalization, model inputs, and evaluation criteria.
  Ask before resolving ambiguities that would materially change an experiment.
- Avoid speculative features and backward-compatibility layers unless requested.
  Explain intentional interface or checkpoint-format changes.
- Make focused changes, preserve unrelated user work, and test in proportion to
  risk. Do not launch expensive jobs, push, publish, or delete valuable artifacts
  unless requested. Report what was tested and what remains unverified.

## Repository layout and documentation

Use only the directories the project needs. Both `src/<package>/` and a top-level
`<package>/` are fine. Typical locations are:

- `tests/`: small, focused tests; `configs/`: readable experiment configurations.
- `examples/`: tiny, clearly labeled examples; `scripts/`: thin entry points or
  setup utilities; `hpc/`: Slurm launchers and their short usage instructions.
- `workspace/` or a caller-provided root: datasets, caches, runs, and temporary
  artifacts. `datasets/raw/` and `datasets/processed/` are useful when conversion
  is involved, but do not mandate this layout for every project.

Keep scientific logic in Python, not shell scripts. Prefer ordinary command-line
arguments or the project's existing configuration system to a new abstraction.
Accept data and output paths as arguments or environment variables; never require
one person's absolute paths or a particular sibling checkout layout.

Keep the README concise: installation, data preparation, a minimal working
example, training/evaluation where applicable, and the public API needed by users.
Commands should agree with the actual CLI. Explain required working directories,
environment selection, and wrappers. Avoid duplicate guides, speculative roadmaps,
and production governance unless requested.

Track source, tests, small configs, `pyproject.toml`, and `uv.lock`. Ignore virtual
environments, credentials, caches, datasets, checkpoints, videos, W&B output, and
generated logs. Do not accidentally ignore useful source inside a workspace.

## Python environments with uv

- Use a project-owned environment, normally `.venv`, managed by `uv`. Declare
  dependencies in `pyproject.toml` and commit the resolved `uv.lock`. Pin the
  project's Python version, for example with `.python-version`, consistently
  with `requires-python`.
- Make dependency changes intentionally with `uv add`, `uv remove`, or an explicit
  lock update. On another machine, install the committed environment with
  `uv sync --locked`, plus the extras/groups actually defined by this project.
  Recreate environments there; do not transfer a workstation's `.venv`.
- `uv run --locked ...` verifies the lock is current and may synchronize the
  environment. For a prepared HPC environment, use
  `uv run --frozen --no-sync ...` or its explicit Python interpreter. `--frozen`
  does not check lock freshness; `--no-sync` does not install missing dependencies.
  Validate the selected environment before submitting a job.
- `uv` normally targets the current project's environment, not an unrelated
  activated environment. Do not use `--active` merely to suppress a warning;
  use it only when deliberately targeting that environment.
- Separate CPU/CUDA or simulator-specific dependencies only when necessary.
  Install one compatible accelerator profile. For PyTorch-specific indexes,
  use explicit indexes with package-specific source mappings so unrelated
  dependencies still resolve from the intended general index. Do not weaken
  index security just to work around a resolution error.
- Do not install into system Python or shared simulator runtimes. An optional
  dependency should not prevent unrelated tools from importing or running.

## Data, runs, and reproducibility

- Treat production datasets and released checkpoints as immutable. Write derived
  caches separately. Keep caches, logs, temporary files, videos, and run outputs
  below an explicit workspace or caller-provided root.
- Give every new training and evaluation run a descriptive name containing UTC
  date and time, for example `ddim-seed0-20261002T143025123456Z`. Add the Slurm job
  ID or another unique suffix where appropriate. Refuse existing run directories
  for new runs; resume only explicitly, with one writer per run.
- Save the resolved configuration, command, seeds, code commit and dirty status,
  lock digest, data version/manifest and split identifiers. Record the evaluated
  checkpoint's hash. A small JSON file is sufficient; do not build a provenance
  framework. Keep secrets out of saved configs and environment diagnostics.
- Save local metrics and checkpoints inside the run directory; put evaluations
  in their own timestamped directories, linked to the originating checkpoint.
  Runs should remain understandable without access to W&B.
- Fit normalization on training data only. Keep validation and held-out test
  roles distinct. Compare methods with matched data splits, evaluation seeds,
  and budgets, and document deliberate differences.
- Make checkpoints self-contained for inference: weights, architecture/config,
  normalization, training step, and required interface metadata. A resumable
  checkpoint also needs optimizer/scheduler state and relevant RNG/EMA state.
  Define what `best` means, including metric, split, and direction; training or
  validation loss is not automatically the best closed-loop policy. Distinguish
  raw and EMA weights in both checkpoints and evaluation results.
- Prefer safe, tensor-only checkpoint loading where compatible. Load checkpoints
  requiring unrestricted pickle deserialization only from trusted sources.
- For queued/running jobs, keep their code, configs, and environment stable.
  Use a separate checkout when concurrent development would change those inputs.

## Training, evaluation, and W&B (when applicable)

- Print useful losses and provide an optional progress bar. Support quiet or
  non-interactive execution; do not fill Slurm logs with per-batch redraws.
- Make W&B optional and configurable: enabled/disabled, project, optional entity,
  and online/offline mode. Keep it off for tests and smoke runs; enable it
  explicitly for tracked experiments. Use a dedicated project for a campaign
  when requested, not a hardcoded personal account or unrelated project name.
- Authenticate outside scripts; never commit or print API keys. Store W&B files
  under the run directory. Use the local run name as the display name, save the
  W&B run ID for explicit resume, and log the resolved non-secret config.
  Training and local artifact saving must work with W&B disabled.
- Start with train/validation loss, learning rate, and relevant task metrics;
  add gradient norms, timing, or throughput when useful. Use consistent namespaces
  such as `train/`, `val/`, and `eval/`. Document non-obvious metrics, units,
  averaging, and success denominators. Do not call a proxy metric task success.
- Whenever logging is introduced or configured in a project, create a tracked
  Markdown metric reference, normally `docs/metrics.md`, and link it from the
  README. Reuse an existing reference instead of creating a duplicate. This
  applies to local logging as well as W&B or another tracking service.
- Cover every application-logged scalar, histogram, image, video, and failure
  indicator. Give the exact key (or an explicit key template with all possible
  values), a brief plain-language meaning, and the details needed to interpret
  it: formula, units/scaling, aggregation and denominator, data split, evaluation
  protocol, logging frequency, step axis, applicable stages, improvement direction
  or expected range, baselines, and important limitations. Explain missing or
  undefined values and distinguish task metrics from proxies and automatic
  tracker telemetry. Link to the code that computes the values.
- Keep the metric reference current in the same change whenever logging keys,
  calculations, normalization, splits, baselines, schedules, or media change.
  Document renames/removals when needed to interpret older runs. Before finishing
  a logging change, check the reference against the emitted keys and calculations;
  a short list of selected dashboard metrics is not a complete reference.
- Log a few fixed qualitative examples where informative: predictions versus
  ground truth, conditioning context, or action chunks. Preserve training RNG
  state around diagnostics so enabling logging does not change training.
- Evaluate using the actual inference path. Record checkpoint step, EMA/raw
  selection, seeds, episode count, and relevant inference settings. Save a small
  local selection of rollout videos, including successes and failures when
  available; W&B media upload should be optional and bounded.
- When evaluation is expensive, run it asynchronously in bounded subprocesses
  using immutable checkpoint snapshots. Budget CPU/GPU resources explicitly;
  avoid unbounded queues. Clearly report skipped/coalesced busy intervals and
  evaluation failures. Finish outstanding evaluation on normal shutdown or
  explicitly report incomplete results.
- Let the training process own W&B logging. Plot delayed evaluation results
  against their checkpoint's training step using a separate metric axis, not
  the optimization step when results arrived. Do not back-date W&B's global log
  step. Evaluation errors must not silently become zero success rates.

## Testing and CI/CD

- For a new Python project, prefer a small CI workflow on pull requests and
  pushes: a locked environment install, fast CPU tests, and the lint/format/type
  checks the project actually uses. Build the package when it is distributed.
  Do not add a large tooling stack just to satisfy this template.
- Use tiny fixtures and avoid dataset downloads, W&B credentials, GPUs, or native
  simulator installations in default CI. Keep heavy integration tests explicit
  and separate. A skipped simulator test is not evidence that simulation works.
- Cover important scientific interfaces: shapes/units, data splits, normalization,
  checkpoint round trips, and a small training/inference smoke test where useful.
  Run relevant tests after code changes and `bash -n` after shell-script changes.
- Give CI minimal permissions and make its environment reproducible. Automatic
  deployment, package publication, and release workflows are not required for
  research repositories; add them only for an explicitly requested workflow.

## Slurm / sbatch (when applicable)

Keep each launcher readable: resource header, a few paths/environment settings,
one visible training command. Share only genuinely repetitive launch logic; do
not hide experiments behind complicated Bash orchestration.

- Specify partition, nodes/tasks, GPUs, CPUs, memory, time, job name, and stdout/
  stderr paths. Adapt resources and account/QoS to the cluster and experiment.
  `#SBATCH` directives are literal; shell variables in them are not expanded.
  Request GPUs only for stages that need them; CPU-only preparation/evaluation
  can use CPU allocations.
- Create log directories **before** `sbatch`, because Slurm opens logs before
  executing the script. Use `%x_%j` to distinguish job names and IDs.
- Use `set -euo pipefail`, an explicit working directory, and an already prepared
  environment. Do not install or upgrade packages inside experiment jobs.
  Submit from the documented directory. Use `srun` if required by the site.
- Bound data-loader workers, evaluation workers, and BLAS/OpenMP threads within
  the allocation. Respect CUDA visibility; visible device 0 need not be physical
  GPU 0. Do not guess EGL device mapping from Slurm/CUDA index equality.
- Smoke-test on an actual compute allocation before long campaigns: environment,
  accelerator access, a tiny workload, and headless simulation/video if used.
  Include final evaluation and artifact-writing time in the walltime budget.

Example shape, to adapt to the project's real module, CLI, and cluster:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=train-example
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
: "${DATA_ROOT:?Set DATA_ROOT before submission.}"
: "${WANDB_PROJECT:?Set the W&B project for this experiment.}"
export WANDB_PROJECT
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_id="example-seed0-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%S%NZ)"
run_dir="${RUNS_ROOT:-$PWD/workspace/runs}/$run_id"

# The Python entry point creates a fresh run directory and refuses collisions.
# Replace this illustrative module/CLI with the project's actual training API.
exec uv run --frozen --no-sync python -m your_package.train \
  --data-root "$DATA_ROOT" \
  --run-dir "$run_dir" \
  --seed 0 \
  --wandb
```

For Peano, the existing H200 launchers use `gpuq` with `--gres=gpu:1`; prefer H200
over B200 unless requested otherwise. This is a site-specific example, not a
portable hardware guarantee. Verify allocation constraints on another cluster.

Submit after preparing the environment and exporting the required paths/project
(ordinary unexported shell variables are not inherited by the job):

```bash
export DATA_ROOT="/absolute/path/to/data"
export WANDB_PROJECT="your-project-or-campaign"
mkdir -p hpc/logs
sbatch hpc/train.sbatch
```

For actual prerequisites, capture `sbatch --parsable` and submit the dependent
job with `--dependency=afterok:JOBID` (strip any `;cluster` suffix from the returned
ID). `afterok` means the prerequisite exited with status zero, not that it reached
a scientific target. Independent experiments need no dependencies. Record job
IDs and run directories; do not submit a campaign just to test its scripts.

## Optional: robotics projects using phi-* backends

Skip this section if the project does not use these packages. In a policy project,
this repository owns models, training, checkpoint handling, and thin simulator
adapters. `phi-*` projects, whether under `workspace/`, in sibling checkouts, or
installed as packages, are independent backend repositories.

- Treat `phi-*` packages and sibling checkouts as immutable dependencies during
  policy work. Do not edit, vendor, copy, monkey-patch, or commit backend source
  unless the user explicitly requests a backend API or semantic change.
- Import only public paths documented by the pinned backend release. Keep model,
  optimizer, framework conversion, checkpoint, and policy-adapter code here.
  If a capability is missing, report the public API gap instead of importing an
  undocumented private module.
- Keep observation/action profiles explicit: dimensions, ordering, units,
  coordinate conventions, normalization, and execution/replanning semantics.
  Use the backend's documented reset, termination, and success contracts.
- Treat simulator installations as immutable. Never install policy dependencies
  into a shared simulator runtime. Use project-owned environments and per-user
  writable cache/output roots while sharing heavy read-only installations when
  supported. Never use a broad process-kill command on a multi-user workstation.
- Pin every backend to an immutable release or full Git commit and commit the
  resolved lock for shared experiments. Local editable installs are a deliberate
  development convenience, not an immutable experiment pin. Record policy/backend
  commits, lock digests, dataset manifests, checkpoint hashes, and observation/
  action profile identifiers.
