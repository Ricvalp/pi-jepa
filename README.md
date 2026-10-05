# Physics-informed JEPA on dense cart-pole video

Two datasets implement the [collection addendum](physics_jepa_cartpole_dataset_addendum.md)
with the approved **0.85 m rod, ±1.5 m cart limit, and square 96×96 images**:
passive autonomous motion and controlled motion with calibrated actions.
Train a physics-informed JEPA on the passive corpus first, then a separate model
on the controlled corpus. See the [standalone dataset report](docs/new_dataset_report.md),
[scientific description](docs/experiment.md), and [logged metric reference](docs/metrics.md).

The main implementation uses an encoder without BatchNorm and includes a
separately labeled **true-reset diagnostic**. Run all commands below from this
repository's root using its `.venv`. The existing datasets can be reused.
For data or outputs elsewhere, replace `workspace/data` and `workspace/runs`
in the commands with the corresponding directories.

## Environment

Run commands from the repository root. Python 3.11.15 and dependencies are pinned
by `.python-version`, `pyproject.toml`, and `uv.lock`:

```bash
uv sync --locked --extra tracking
uv run --frozen --no-sync wandb login
uv run --frozen --no-sync python scripts/check_environment.py --device cuda
```

W&B is optional; omit `--extra tracking` and login when tracking is disabled.
Use `--device cpu` for a CPU environment check. Prepared jobs use
`--frozen --no-sync` so they do not change the environment. CPU tests:
`uv run --frozen --no-sync pytest -q`.

## Datasets

Both corpora are stored in `workspace/data/{passive,controlled}`. `--data-root`
(or `DATA_ROOT`) always means their **parent directory**, not one corpus folder.
For a new machine, generate and inspect them with:

```bash
uv run --frozen --no-sync python -m pi_jepa.data \
  --config configs/base.json --dataset both --data-root workspace/data --workers 4
uv run --frozen --no-sync python -m pi_jepa.data_qa \
  --dataset both --data-root workspace/data
```

Existing complete matching data are reused; incompatible or incomplete outputs
are refused. The current format is **schema 3**. Old datasets, checkpoints, and
related local outputs were retired for this geometry revision. Actual retained
counts and checks are documented in the [QA report](docs/dataset_qa.md).

The clocks are **2 ms integration, 10 ms saved RGB/state, 20 ms force updates,
20 ms spacing within encoder histories, and 100 ms prediction steps**. Both
camera axes span ±2.6 m with equal pixels per metre and 2× antialiasing. The
0.85 m rod projects to 15.53 pixels at every angle. Pole mass remains 0.2 kg and
gravity 9.81 m/s². Appearance and the world-origin mark are fixed.

Each learning NPZ contains `rgb[N+1,96,96,3]`, `force[N,1]`, float64 timestamps,
commanded `p0=0`, coarse reset mode, identifiers, collection family, valid length,
and termination reason. Known apparatus parameters are supplied only for
controlled train/validation and the fixed passive apparatus. Controlled held-out
learning files also retain their full prechosen `force_program`. Separate private
`truth/` files contain physical states, exact resets, and true parameters.
Ordinary learned-reset training never opens those truth files. The explicit
true-reset diagnostic reads only `exact_initial_state` for training episodes;
it does not read stored future states or validation/test truth. Boundary-truncated
prefixes are retained; no states are clipped or recentered.

For recorded trajectories with applied-force arrows:

```bash
uv sync --project scripts/dataset_videos --locked
uv run --project scripts/dataset_videos --frozen --no-sync \
  python scripts/dataset_videos/render.py --data-root workspace/data --dataset passive
uv run --project scripts/dataset_videos --frozen --no-sync \
  python scripts/dataset_videos/render.py --data-root workspace/data --dataset controlled
```

The default explicitly selects every fourth 100 Hz saved frame at 25 fps,
preserving real time. `--stride 1` displays all records at 100 fps. See the
[video utility guide](scripts/dataset_videos/README.md).

## Train the true-reset diagnostic: passive first, controlled later

The diagnostic asks whether the encoder/readout and latent predictor can learn
physical motion when the simulator starts from the **fixed true training reset**,
instead of a jointly fitted reset that can collapse toward equilibrium. This is
extra supervision, not the original physics-only experiment.

First train the **passive joint model**:

```bash
uv run --frozen --no-sync python -m pi_jepa.train \
  --config configs/true_reset_diagnostic.json --dataset passive --mode joint \
  --data-root workspace/data --runs-root workspace/runs --device cuda
```

After inspecting it, train the **controlled joint model** independently:

```bash
uv run --frozen --no-sync python -m pi_jepa.train \
  --config configs/true_reset_diagnostic.json --dataset controlled --mode joint \
  --data-root workspace/data --runs-root workspace/runs --device cuda
```

These are independent runs from fresh initialization, each with 10,000 updates,
batch size 64, a physical readout, and fixed training reset states. There is no
reset optimizer or checkpoint transfer. The passive predictor receives only the current latent;
its physics loss uses zero force and fixed known `[M,b]=[1,0.25]`. The controlled
predictor receives recent latents, force blocks, and apparatus parameters.

The encoder uses GroupNorm in its ResNet backbone and LayerNorm on the projector's
hidden features. Its final 32-dimensional output stays unnormalized. Six temporal
offsets are encoded separately; normalization never combines different clips or
episodes, and has the same behavior in training and evaluation.

For a **matched learned-reset comparison**, keep this same architecture, seed,
data, and training budget, changing only the reset protocol:

```bash
uv run --frozen --no-sync python -m pi_jepa.train \
  --config configs/wandb.json --dataset passive --mode joint \
  --initial-conditions learned --wandb-group causal-learned-reset \
  --data-root workspace/data --runs-root workspace/runs --device cuda
```

Use `--dataset controlled` for its controlled counterpart. Comparing the new
diagnostic only with an old BatchNorm run confounds the normalization and reset
changes. `configs/base.json` and `configs/wandb.json` retain learned resets by
default. `--initial-conditions true_fixed` explicitly selects the diagnostic in
any otherwise compatible configuration; it is supported only with `--mode joint`.

All commands use W&B project `physics-jepa-cartpole`. The diagnostic configuration
uses group `causal-true-reset-diagnostic`, and its run name includes
`true-fixed-reset-diagnostic`. Override the project with
`--wandb-project YOUR_PROJECT` and optionally `--wandb-entity YOUR_ENTITY`.
`--wandb-mode offline` records locally with the SDK; `--wandb-mode disabled`
keeps diagnostics without uploading. Runs get separate UTC names under
`workspace/runs/`, resolved configuration, provenance, local losses/diagnostics,
and checkpoints. The [metric reference](docs/metrics.md) defines every logged key.

A batch samples distinct episodes uniformly, then one valid 65-frame crop each.
The physical solver integrates from the episode's actual frame-zero reset to the
crop endpoints. In the diagnostic this reset is fixed truth, loaded only for the
training split and stored in the checkpoint; later target states are simulated,
not read from the dataset. The learned-reset comparator instead fits that reset.
An equilibrium shortcut remains possible in the comparator, so low consistency
loss alone does not establish physical learning. Validation retains its
truth-free reset-prior metric in both variants; it is not directly comparable to
the diagnostic's true-reset training loss.

Monitoring also logs `train_eval/physics/autoregressive/h{1,2,3}/mse`: the
readout of autoregressively predicted latents versus the same configured-reset
simulator at 0.1, 0.2, and 0.3 s. `train_eval/physics_phase` shows observed and
predicted readouts with that simulator in cart-position/pole-angle space for
three fixed training episodes: one from each reset family. It shows up to
3.2 seconds of free prediction after the 0.34 s seed, with no future images fed
to the predictor. Shorter episodes stop at their last available latent endpoint.
Figures are logged initially, every 1,000 updates by default, and at the final
update; they are also saved locally under the run's `diagnostics_media/`. These
are diagnostics; no additional loss on predicted readouts is introduced.
Definitions and protocol-dependent interpretation are in the
[metric reference](docs/metrics.md).

`--mode jepa` trains a baseline without physical supervision for either corpus.
Controlled post-hoc fitting uses `--mode readout --pretrained PATH_TO_JEPA_RUN/final.pt`.
Resume explicitly with `--resume --run-dir PATH_TO_RUN` and the original stage,
configuration and optional pretrained checkpoint. There must be one writer per
run. Checkpoints use **format 5**. Start fresh, or explicitly resume a compatible
run with the same architecture and reset protocol. No validation/test metric selects a
"best" checkpoint. Dataset preparation does not launch full training.

## Evaluate and visualize predictions

Passive latent evaluation:

```bash
uv run --frozen --no-sync python -m pi_jepa.evaluate_latents \
  --config configs/base.json --dataset passive --data-root workspace/data \
  --checkpoint PATH_TO_PASSIVE_RUN/final.pt --device cuda
```

Controlled latent and physical evaluation:

```bash
uv run --frozen --no-sync python -m pi_jepa.evaluate_latents \
  --config configs/base.json --dataset controlled --data-root workspace/data \
  --joint-checkpoint PATH_TO_CONTROLLED_RUN/final.pt --device cuda
uv run --frozen --no-sync python -m pi_jepa.evaluate \
  --config configs/base.json --dataset controlled --data-root workspace/data \
  --joint-checkpoint PATH_TO_CONTROLLED_RUN/final.pt --device cuda
```

Add `--posthoc-checkpoint PATH_TO_READOUT_RUN/final.pt` for controlled comparisons.
Latent evaluation compares predictions against encoded actual future frames and
persistence; controlled evaluation also includes shuffled/zero-action controls.
Physical parameter fitting applies only to the controlled corpus, using two
independent calibration episodes before nominal/fitted/oracle forecasts.
Horizon targets are 0.1/0.2/0.4/0.8/1.6/3.2 seconds; unavailable targets after
boundary exits are masked and valid-query counts reported.

To see trajectories rather than error plots, export the joint model's readout
from the resulting latent evaluation directory:

```bash
uv run --frozen --no-sync python scripts/export_readout_trajectories.py \
  --latent-run PATH_TO_LATENT_EVALUATION
uv run --project scripts/dataset_videos --frozen --no-sync \
  python scripts/dataset_videos/compare_readouts.py --trajectories PATH_PRINTED_BY_EXPORTER
```

These separate ground truth, `r(P-predicted z)`, observed-image `r(E(video))`, and
training simulator supervision. Each evaluation has a new timestamped directory
and checkpoint hash. Passive/controlled raw latent errors are not directly
comparable scores.

## Repository and HPC

`src/pi_jepa/` holds scientific code, `tests/` focused checks, `configs/` experiment
settings, `docs/` explanations, and `workspace/` generated outputs. Consult
[agents/AGENTS.md](agents/AGENTS.md) for repository instructions and
[hpc/README.md](hpc/README.md) for Slurm setup and sequential passive/controlled
submission. Licenses and reference implementations are in [THIRD_PARTY.md](THIRD_PARTY.md).
