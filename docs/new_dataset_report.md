# New cart-pole datasets: report for ChatGPT

Prepared on 2026-10-05. This is a self-contained description of the current
physics-informed JEPA experiment and its generated datasets. It supersedes the
0.5 m rod / ±2 m cart-limit values in the original dataset addendum. The approved
revision is **0.85 m rod, ±1.5 m cart limit, 96×96 square RGB, and fixed square
camera spanning ±2.6 m on both axes**. Pole mass and gravity are unchanged.

The old local datasets, checkpoints, evaluation outputs, QA reports, videos,
and exploratory geometry previews were removed before this generation. The
original prompt and addendum remain as historical specifications; this report
records the approved changes. Existing virtual environments were retained.
Remote W&B history was not deleted.

## Purpose and scientific interpretation

We want to learn a visual latent representation `z = E(video history)` and a
predictor `P` that advances it by 0.1 s. A physical readout `r(z)` should decode
cart position/velocity and pendulum angle/angular velocity. The physics-informed
loss supervises `r(E(observed history))` against a differentiable simulator with
known dynamics and learned hidden initial conditions. It does **not** train on
true simulator state labels, and it does not directly supervise `r(P(z))`.
Monitoring additionally measures the consistency of `r(P(...))` with that same
learned-reset simulation. This is a diagnostic, not an extra training loss.

The two corpora isolate two questions:

1. **Passive:** can an action-free predictor learn autonomous visible motion?
   Every trajectory uses fixed cart parameters and exactly zero applied force.
2. **Controlled:** can the predictor model motion using the right known force
   sequence and apparatus parameters, including unseen parameter combinations?

Both now support joint JEPA/physics training. Run passive first, inspect the
predicted/readout trajectories, then train an independent controlled model from
scratch. This is an experiment order, not pretraining followed by weight transfer.
A low latent loss can indicate collapsed representations; a low physical loss
can indicate a nearly constant equilibrium readout, particularly for passive
motion. Neither alone establishes successful learning. Compare forecasts with
latent persistence and (controlled only) shuffled/zero-force baselines; visualize
`r(predicted z)`, `r(encoded actual history)`, and true physical trajectories.

## Physical system and rendering

The state is `x = [p, v, q, w]`, with metres, metres/second, radians, and
radians/second respectively. `q=0` points down; `q=π` is upright. Angles remain
unwrapped during integration. Positive applied force points right.

| Quantity | Value |
|---|---|
| Pole point mass `m` | 0.2 kg |
| Massless rod length `ell` | 0.85 m |
| Gravity `g` | 9.81 m/s² |
| Cart mass `M`, drag `b` | Fixed `[1.0 kg, 0.25 N·s/m]` in passive; varied in controlled |
| Maximum applied force | ±5 N, including all feedback and excitation |
| Cart termination boundary | First recorded finite state with `abs(p) > 1.5 m` |
| Camera | Fixed, square, `x,y ∈ [-2.6,2.6] m` |
| Saved images | uint8 RGB, 96×96, 2× supersampling then antialiased downsampling |
| Metric image scale | `95/5.2 = 18.2692` pixels/metre on **both** axes |
| Rod projection | `0.85 × 95/5.2 = 15.5288` pixels at every angle |
| Appearance | Fixed background, blue cart, red bob; fixed track-origin mark |

The equations implemented for both collection and physical supervision are:

```text
p_dot = v
v_dot = [u - b*v + m*sin(q)*(ell*w² + g*cos(q))] / [M + m*sin(q)²]
q_dot = w
w_dot = -[g*sin(q) + cos(q)*v_dot] / ell
```

The longer rod changes the real dynamics; it is not a graphical stretch. Mass
reduction would not restore the former angular frequency. Normal gravity and
pole mass were deliberately retained. Compared with the old 0.5 m, ±2.8 m camera
collection, the rod has about **83% more pixels along its length**. Cart dimensions
remain physically unchanged; both image axes share the same scale.

The camera does not follow the cart, change zoom by episode, or distort aspect
ratio. A cart, pole and bob visibility check includes the terminal boundary-exit
frame and a half-pixel raster margin; generation fails if a proposed frame would
clip. No hidden external forces, teleports, state clipping, or recentering occur.

## Time conventions

| Clock | Interval | Rate |
|---|---:|---:|
| RK4 numerical substep | 0.002 s | 500 Hz |
| Saved RGB and true state | 0.010 s | 100 Hz |
| Controller / applied-force update | 0.020 s | 50 Hz |
| Spacing between the eight images of one encoder history | 0.020 s | 50 Hz |
| Latent predictor step | 0.100 s | 10 Hz |

Five RK4 substeps advance one recorded interval. `force[k]` acts on
`[t[k], t[k+1])`, advancing `state[k]` to `state[k+1]`. Thus `N` forces accompany
`N+1` images/states. Every complete 20 ms force hold appears in two consecutive
records; a boundary-truncated prefix can end after the first half of a hold.
Numerical integration never straddles a force discontinuity.

## Passive dataset

Every episode starts at commanded cart position `p0=0`, with fixed known
`[M,b]=[1,0.25]`. All actions are exactly zero. Episodes last up to four seconds.
Train/validation/test contain 1,536/96/96 independent episodes. Each split uses
an exact 3:2:1 mixture: 50% downward oscillations, one-third nonlinear excursions,
and one-sixth upright falls.

- Downward: `v0 ∼ U(-0.25,0.25)`, `q0 ∼ U(-1.2,1.2)`,
  `w0 ∼ U(-2,2)`. Reject jointly tiny resets where `abs(q0)<0.10`,
  `abs(w0)<0.20`, and `abs(v0)<0.05`.
- Nonlinear: the same cart-velocity range, random angle sign with
  `abs(q0) ∼ U(1.2,2.6)`, and `w0 ∼ U(-2,2)`.
- Upright: the same cart-velocity range, `q0 = π ± U(0.05,0.20)`,
  and `w0 ∼ U(-0.5,0.5)`.

Only commanded `p0=0` and the coarse reset class are learning metadata. The exact
sampled angle/velocities are hidden. The passive predictor is a
`32 → 128 → 128 → 32` GELU MLP taking only the current latent; it receives no
parameters, forces, class labels, identifiers, or absolute time.

## Controlled dataset

Training samples 64 apparatuses independently with `M ∼ U(0.7,1.3)` and
`b ∼ U(0.05,0.5)`. Validation samples eight additional apparatuses from the same
ranges with independent streams. There are 24 training episodes per apparatus
and 12 validation episodes per apparatus:

| Family | Train per apparatus | Validation per apparatus | Force design |
|---|---:|---:|---|
| Pulse | 8 | 4 | Open-loop pulses; half use opposite-sign pulse pairs |
| Multisine | 4 | 2 | Open-loop sum of three sinusoids |
| Noisy LQR | 8 | 4 | Nominal stabilizing feedback plus recorded excitation |
| LQR release | 4 | 2 | Same, with one feedback interruption |

Open-loop training reuses a library of 32 pulse and 32 multisine programs across
apparatuses, with independent resets. Validation has a separate waveform library.
Pulse durations are 0.10–0.30 s; nominal pulse scale is 0.5–3 N, each nonzero
pulse draws 0.3–1 times that scale, and 20% of sampled segments are zero. Multisine
frequencies lie in `[0.30,0.60]`, `[0.65,1.00]`, and `[1.20,1.70]` Hz; the summed
program is centered and normalized to a sampled 0.8–2.5 N peak.

Feedback uses a single analytic LQR gain designed for nominal `[1,0.25]` and the
**new 0.85 m rod**, scaled per episode by `U(0.85,1.15)`. Half of feedback episodes
use a sinusoidal cart reference with 0.10–0.35 m amplitude and 0.10–0.25 Hz
frequency. Excitation peak is 0.6–1.5 N. Release starts between 1.2 and 2.0 s and
lasts 0.20–0.40 s. The saved force is the entire clipped feedback-plus-excitation
force. LQR is collection machinery, not an imitation-learning target.

Open-loop controlled downward resets use `q0 ∼ U(-1,1)` and
`w0 ∼ U(-1.5,1.5)`. Nonlinear resets match passive nonlinear ranges; upright resets
use `q0 ∼ U(π-0.15,π+0.15)` and `w0 ∼ U(-0.5,0.5)`. All use
`v0 ∼ U(-0.25,0.25)` and `p0=0`. Feedback episodes start upright. Reset classes
are mixed within open-loop waveform families; collection labels are not network
inputs.

## Held-out controlled calibration and queries

There are 12 held-out apparatus pairs `[M,b]`:

- Interior: `[0.78,0.10]`, `[0.88,0.36]`, `[0.97,0.22]`, `[1.08,0.43]`,
  `[1.18,0.16]`, `[1.26,0.31]`.
- Extrapolation: `[0.50,0.02]`, `[1.60,0.80]`, `[0.50,0.275]`,
  `[1.60,0.275]`, `[1.00,0.02]`, `[1.00,0.80]`.

Each has two independent three-second multisine calibration episodes and eight
fresh four-second open-loop queries: four pulse and four multisine, with four
downward, two nonlinear, and two upright resets overall. Calibration starts at
`p0=0`, `v0 ∼ U(-0.15,0.15)`, `q0 ∼ U(-0.6,0.6)`, `w0 ∼ U(-0.5,0.5)`;
its multisine peak is 1–2 N. Query future forces use no true-state feedback.
Held-out apparatus/episode IDs are opaque and random, not encodings of parameters.
They change between generations, so byte hashes are not expected to match solely
from seed 42; the private manifest retains parameter assignments and RNG seeds.

Parameter fitting keeps neural networks frozen and uses calibration observations
only. It jointly estimates apparatus parameters and the unknown state at the first
encoder endpoint (frame 14), including unknown cart position. True parameters
are accessed only for explicitly labeled oracle diagnostics. Prediction errors
from oracle parameters therefore test `P` without relying on parameter fitting.

## Boundary handling and actual collection

An episode is retained through the first finite saved state outside ±1.5 m, then
recording stops. This terminal overshoot is not clipped away. Training and
validation attempts shorter than 65 observations are regenerated within the same
planned family, with discarded attempts recorded. Held-out calibration/query
attempts are retained regardless of length. Predictions whose true future target
falls after termination are masked; attempted-query and valid-target counts are
reported separately.

| Dataset / split | Episodes | Recorded transitions | Boundary exits |
|---|---:|---:|---:|
| passive / train | 1,536 | 614,400 | 0 |
| passive / validation | 96 | 38,400 | 0 |
| passive / test | 96 | 38,400 | 0 |
| controlled / train | 1,536 | 608,663 | 61 |
| controlled / validation | 96 | 38,164 | 3 |
| controlled / calibration | 24 | 7,144 | 1 |
| controlled / query | 96 | 37,502 | 11 |

There are **3,480 retained episodes** and
**1,382,673 recorded transitions**, represented by **1,386,153 RGB images**. Regenerated
short attempts: **0**. Nonfinite exits: **0**. Every saved frame
passed the calibrated geometry and action/state/timestamp alignment checks,
including the first boundary-crossing frame. The maximum stored cart excursion
is 1.525701 m; the smallest object-to-image-edge clearance is
2.0720 pixels (required minimum 0.5 pixels). The rod projects to 15.5288
pixels with maximum numerical length discrepancy 2.49e-14 pixels.

Targets available after the initial context ending at 0.34 s:

| Horizon after observed context | Passive valid / 96 | Controlled valid / 96 |
|---|---:|---:|
| 0.1 s | 96 | 96 |
| 0.2 s | 96 | 96 |
| 0.4 s | 96 | 96 |
| 0.8 s | 96 | 96 |
| 1.6 s | 96 | 96 |
| 3.2 s | 96 | 87 |

RK4 five-2-ms versus ten-1-ms integration over one record differed by at most
1.16e-10 across state coordinates in the three convergence cases. Over a
three-second zero-force trajectory, undamped energy drift was at most
1.36e-11 J; with drag 0.25,
energy decreased at every saved step.

- Passive: [full QA JSON](../workspace/evaluations/dataset-qa-passive-seed42-20261005T103512510534Z/qa.json); [force-annotated video gallery](../workspace/evaluations/dataset-forces-passive-20261005T103524719090Z/index.html).
- Controlled: [full QA JSON](../workspace/evaluations/dataset-qa-controlled-seed42-20261005T103601814980Z/qa.json); [force-annotated video gallery](../workspace/evaluations/dataset-forces-controlled-20261005T103614497474Z/index.html).

Manifest SHA-256 identifiers for this generation:

- `passive`: `a6f95d39fe2599b8ec5dc4338a7b75a3a03051b17ec10fbeb8007b36af341b7a`.
- `controlled`: `bfa6bb17b47f070547370aabe5a43aadef2817fc461b2f5b31fbf651d8650218`.


## Stored files and information separation

Current schema is **3**. Root paths are `workspace/data/passive/` and
`workspace/data/controlled/`, each with its own `manifest.json` recording physics,
camera, clocks, config, code hashes, split memberships, collection counts, and
per-episode visibility margins. Each learning `.npz` stores:

- `rgb[N+1,96,96,3]` (`uint8`), `force[N,1]`, and float64 `t[N+1]`.
- Commanded zero cart reset, coarse reset class, episode/apparatus/waveform IDs,
  collection family, observation count, and termination reason.
- `theta=[M,b]` for known training/validation parameters and the fixed passive
  apparatus; controlled held-out files omit it.
- Full prescribed `force_program` for controlled calibration/query episodes,
  even if their recorded prefix terminates early. This supports shuffled-action
  donors without fabricating absent actions.

Separate `truth/` files contain `state[N+1,4]`, `theta_true`, and exact initial
state; a private held-out manifest records test assignments and generation seeds.
The training loader permits only train/validation learning files and never opens
truth. Truth is read by collection QA, ground-truth scoring, and visualization.
Color is fixed by default, preventing appearance variation from substituting for
motion variation. There are no spatial crops, flips, camera motion, or occlusion
augmentations; training still samples temporal windows from each episode.

## Training interface and objective

An encoder history contains eight images ending at raw index `e`:
`[e-14,e-12,e-10,e-8,e-6,e-4,e-2,e]`. They span 0.14 s, are stacked as 24 channels,
normalized to `[-1,1]`, and encoded to 32 dimensions by a spatial ResNet-18 and
projector. One 65-frame crop supplies six histories ending at relative indices
`[14,24,34,44,54,64]`. Consecutive histories overlap by three images but remain
causal. A minibatch samples independent episodes, then one valid crop per episode.

Controlled `P` is a three-block, width-192, three-head causal transformer using
up to three previous latents. Each transition receives its ten future
recorded-interval forces divided by 5 N, plus normalized mass/drag. Passive `P`
is the action-free MLP described above. Both use
`L_JEPA = MSE(predicted next latent, encoded next history) + 0.1*SIGReg`.
Gradients flow through prediction and target; there is no EMA teacher. SIGReg
operates across independent episodes at each offset, encouraging noncollapsed
representations.

Joint training adds `L_phys` with weight 1. A `32→64→5` readout produces
`[p,v,sin(q),cos(q),w]`, with normalized angular pair. Each training episode has
three learned reset nuisance variables, mapped to
`[p0,v0,q0,w0]=[0,tanh(a),q_base+π*tanh(d),3*tanh(c)]`;
`q_base=π` for upright and zero otherwise. These are initialized from coarse
metadata, never true resets. The simulator integrates from the actual episode
reset to all crop endpoints; a crop is not falsely treated as a new centered
reset. The physical loss compares readout and simulator coordinates after
scaling by `[2,2,1,1,5]`. The 2 m position divisor remains a loss normalization,
not the new track boundary.

The logged forecast-physics diagnostic applies the same scaled residual to
`r(hat z)`, with `hat z` produced autoregressively by `P` at 0.1, 0.2, and
0.3 s after the last observed latent. The simulation target is sampled at the
matching absolute episode endpoints and still starts from the learned episode
reset. It does not restart from `r(z)` at the forecast origin. Controlled
forecasts use the recorded forces and known apparatus parameters; passive
forecasts receive no actions or parameters. This calculation runs without
gradients and leaves the objective above unchanged.

Both joint runs use 10,000 updates, batch size 64, neural AdamW at `3e-4`,
500-update warmup followed by cosine decay to `3e-5`, weight decay 0.05, and neural
gradient clipping at 1. Reset variables use separate Adam at `1e-2` and are not
included in neural clipping. Solver segments are checkpointed for backward memory.
No validation/test metric selects a "best" model; final raw weights are used.
Format-4 checkpoints include dataset and geometry metadata to reject incompatible
old models.

## Commands: passive first, controlled later

Run from the repository root. Prepare the locked environment and authenticate
outside training once:

```bash
uv sync --locked --extra tracking
uv run --frozen --no-sync wandb login
```

Train passive physics-informed JEPA:

```bash
uv run --frozen --no-sync python -m pi_jepa.train \
  --config configs/wandb.json --dataset passive --mode joint \
  --data-root workspace/data --device cuda
```

After that run completes and is inspected, start a fresh controlled model:

```bash
uv run --frozen --no-sync python -m pi_jepa.train \
  --config configs/wandb.json --dataset controlled --mode joint \
  --data-root workspace/data --device cuda
```

W&B project defaults to `physics-jepa-cartpole`; optional overrides are
`--wandb-project NAME` and `--wandb-entity ENTITY`. Add `--wandb-mode disabled`
for local diagnostics only, or `--wandb-mode offline` for offline W&B files.
Every run has its own timestamped folder in `workspace/runs/`, local CSV/JSONL,
media, config/provenance, and checkpoints. Neither command resumes an old run or
loads weights from the other corpus. Slurm equivalents and optional sequential
job dependencies are documented in [hpc/README.md](../hpc/README.md).

The complete [metrics reference](metrics.md) describes training losses,
representation variance/rank, persistence skill, action sensitivity/advantage,
physical residuals, readout variation, gradients, updates, and qualitative media.
Passive joint runs include the existing physical diagnostics but omit all action
controls. Physical residual media use learned simulator resets, not ground truth.
`train_eval/physics/autoregressive/mse` pools all three forecast endpoints and
five normalized physical coordinates; `train_eval/physics/autoregressive/h1/mse`,
`h2/mse`, and `h3/mse` report the separate endpoints. Corresponding
`residual_{p,v,sin,cos,w}` keys identify coordinate contributions. These are
computed on the fixed training reference in joint and controlled readout stages,
at the initial/resumed snapshot, every 500 updates, media updates, and the final
update with default monitoring settings. Validation has no learned reset table,
so no equivalent validation forecast-physics metric is logged. Low values
indicate consistency with the learned-reset simulator, not independently
verified agreement with the hidden true state; equilibrium readout collapse
remains a possible failure mode.

## Evaluate and inspect learned trajectories

```bash
uv run --frozen --no-sync python -m pi_jepa.evaluate_latents \
  --config configs/base.json --dataset passive --data-root workspace/data \
  --checkpoint PATH_TO_PASSIVE_RUN/final.pt --device cuda
```

For controlled evaluation use `--dataset controlled` and
`--joint-checkpoint PATH_TO_CONTROLLED_RUN/final.pt` instead. Evaluation starts
with observed histories ending at frames 14, 24, and 34, then rolls `P` forward
without future images. Future images are encoded only as scoring targets.
Endpoints at 0.1/0.2/0.4/0.8/1.6/3.2 s after context have their own valid-target
counts. Persistence retains the last observed latent. Controlled tests also use
shuffled and zero conditioning forces with matched target/query selection.

To inspect physical trajectories from either joint checkpoint:

```bash
uv run --frozen --no-sync python scripts/export_readout_trajectories.py \
  --latent-run PATH_TO_LATENT_EVALUATION
uv sync --project scripts/dataset_videos --locked
uv run --project scripts/dataset_videos --frozen --no-sync \
  python scripts/dataset_videos/compare_readouts.py --trajectories PATH_PRINTED_BY_EXPORTER
```

The diagrams separate ground truth, the actual trained readout of predicted
latents, observed-history readouts, and training simulator supervision. Forecast
samples are 0.1 s apart; drawing them at a higher video frame rate would not add
predicted states. Dataset source images themselves are sampled at 100 Hz.

## Validation of execution

Both corpora passed a real CUDA joint-training update at the configured batch
size 64 on an RTX 5000 Ada, with diagnostics enabled: finite objectives and
nonzero encoder, predictor, readout, and reset-state gradients. Checkpoints and
local media were written successfully. These are execution checks, not trained
models or convergence evidence:

- [Passive one-update check](../workspace/checks/smoke-passive-joint-batch64-seed42-20261005T103513501572Z/losses.csv).
- [Controlled one-update check](../workspace/checks/smoke-controlled-joint-batch64-seed42-20261005T103615176870Z/losses.csv).

The default test suite passed **94 tests**; the optional offline W&B integration
test was excluded. All four Slurm scripts passed shell syntax checks. Both
one-update checkpoints completed latent evaluation on their 96 held-out queries
without changing weights or buffers, followed by export through the actual
trained physical readout. Dataset generation source and lock hashes match the
saved manifests. These checks do not measure converged prediction quality.

No full 10,000-update campaign was launched during dataset preparation. The
passive and controlled physics-informed commands are in the main README.

## Limits of the evidence

Dataset QA establishes geometry, timing, finite-state alignment, split contents,
and numerical integration, not learned predictive ability. Short training smoke
checks establish executable gradients and logging, not convergence. A fresh
10,000-update campaign and held-out decoded-trajectory inspection are still
required. Passive and controlled losses describe different prediction problems;
treating their raw values as an action or physics ablation would be misleading.
