# W&B metrics reference

This is the complete reference for metrics and media explicitly logged by this
repository. Each table gives a short meaning and the calculation needed to
interpret it. This edition describes dense dataset format v3 and checkpoint
interface v5. Keep this file updated whenever logging changes.

Enable monitoring with [configs/wandb.json](../configs/wandb.json). Add
`--wandb-mode disabled` to keep the same diagnostics locally without the W&B SDK;
`--wandb-mode offline` uses the SDK without uploading. See the
[README](../README.md) for environment setup and training commands.

## Which plots answer which question?

| Question | Useful metrics |
|---|---|
| Does the forecast beat doing nothing? | `val/prediction/autoregressive/h3/all/persistence_skill`: positive means better than holding the last observed latent. |
| Do the correct actions help in controlled monitoring? | `val/prediction/autoregressive/h3/all/action_advantage`: positive means better than the shuffled-action control. |
| Does the model predict motion? | `val/prediction/autoregressive/h3/all/predicted_motion_ratio`: near zero means little predicted displacement. A value near one does not establish correct direction or timing. |
| Has the representation collapsed? | `val/latent/std_mean`, `val/latent/effective_rank`, and `val/latent/temporal_to_between_variance`, together. A high rank can reflect static appearance rather than motion. |
| Is optimization actually changing weights? | `train/gradients/encoder/norm` and `train/updates/encoder_stem/relative_norm`. Nonzero values alone do not establish useful learning. |
| Does the readout agree with its simulator supervision? | `train_eval/physics/residual_*`, `train_eval/physics_fit`, and `train_eval/physics_phase`. Targets use learned resets by default, or fixed true training resets in the explicit diagnostic. |
| Do decoded latent forecasts agree with that same simulator? | `train_eval/physics/autoregressive/h1/mse`, `h2/mse`, and `h3/mse` compare `r(P(...))` with the configured-reset simulator at 0.1, 0.2, and 0.3 s. Lower means closer agreement; its physical meaning depends on the reset protocol below. |

## Axes, splits, stages, and schedule

All application metrics use **`train/update`** as their W&B x-axis. An update is
one optimization step, not an epoch. Initial diagnostics are at update zero;
after resume, the initial snapshot is at the restored update.

| Prefix | Data and model mode |
|---|---|
| `train/` | Current randomly selected training minibatch; default 64 distinct episodes selected uniformly, with one uniformly sampled valid 65-frame crop per episode. Losses, latents, readouts, and gradients describe the forward/backward pass before the update. Weight changes and reset saturation are measured after it. Encoder/predictor use training mode when trainable. |
| `train_eval/` | Fixed reference: 48 independent episodes sampled uniformly from the full training split, with one uniformly sampled crop each. A separate generator seeded with `config.seed + 101` fixes both episode selection and crops. All networks are in evaluation mode. |
| `val/` | One fixed uniformly sampled crop per validation episode, normally all 96 episodes (capped at 96 for larger custom splits). Selection/crops use `config.seed + 102`. All networks are in evaluation mode. No parameters or reset states are fitted to them. |

Each crop gives six 32-dimensional codes, with endpoints relative to the crop
at `14,24,34,44,54,64`. For crop start `s`, the actual episode endpoints are
`s + [14,24,34,44,54,64]`; an endpoint `e` uses images
`[e-14,e-12,e-10,e-8,e-6,e-4,e-2,e]`. Source images are 0.01 seconds apart,
histories span 0.14 seconds, and neighboring codes are **0.10 seconds apart**.
Consecutive histories share three sampled images; all end at their own endpoint.
Denote the code array by `z[B,6,32]`, where `B` is the number of independent
episodes in that metric's batch or split. SIGReg never treats multiple crops of
one episode as independent batch samples. Latent coordinates have arbitrary
learned units. Statistics use population variance/std (`ddof=0`). Episode IDs,
crop starts, raw endpoints, and selection seeds are saved in
`diagnostic_windows.json` so these fixed comparisons can be reproduced.

The format-5 encoder uses GroupNorm in the backbone and LayerNorm in the
projector's hidden layer, with no normalization on the final latent. Temporal
offsets are encoded separately. Encoder outputs therefore do not depend on
other clips through batch statistics, and normalization has the same behavior
in training and evaluation. The controlled predictor still uses dropout during
training, so its train/eval behavior is not identical.

| Stage (`--mode`) | Optimized components and interpretation |
|---|---|
| `joint` | Both corpora: encoder, predictor and readout. With `training.initial_conditions=learned`, also optimize the training reset table. With `true_fixed`, keep true training resets as immutable buffers and omit the reset optimizer. Both JEPA and physical losses are active. |
| `jepa` | Encoder and predictor only. Training physical loss is recorded as zero because it is inactive; readout/physical diagnostics and the validation reset-prior loss are omitted. |
| `readout` | Controlled corpus only: readout and reset table; pretrained encoder/predictor are frozen. Training JEPA/prediction/SIGReg losses are recorded as zero because they are inactive. Validation JEPA losses and prediction diagnostics still evaluate the frozen models. |

**Reset protocol is part of each metric's meaning.** `learned` is the default
and never accesses private truth. The explicit, joint-only `true_fixed`
diagnostic loads only each training episode's `exact_initial_state` and fixes it
throughout training. Its simulator targets follow the true training trajectory
up to numerical precision, rather than a jointly fitted explanation. No stored
future state arrays or validation/test truth are read. This is additional reset
supervision and must be labeled as such; successful training metrics do not
establish generalization or recovery without that supervision. The two protocol
flags below, resolved config, checkpoint and provenance identify the variant.

Default schedule in `configs/wandb.json`:

| Logged values | Frequency |
|---|---|
| Training scalar diagnostics | First update of a fresh/resumed segment, every 50 updates, and final update. Each is a single minibatch measurement, not an average over the preceding 50 updates. |
| `val/loss/*` | Every 500 updates and final update; no update-zero loss entry. |
| Fixed `train_eval/` and `val/` diagnostics | Initial snapshot, every 500 updates, every media update, and final update. |
| Images and histograms | Initial snapshot, every 1,000 updates, and final update. |

`training.val_every` controls validation losses. For monitoring intervals,
`diagnostics.*` overrides `wandb.*` if present; otherwise `log_every` and
`diagnostics_every` fall back to `training.log_every` and `training.val_every`,
and `media_every` defaults to 1,000. Monitoring defaults to `wandb.enabled`
unless `diagnostics.enabled` overrides it. Changing W&B mode to `disabled` does
not turn off diagnostics when they are enabled in the configuration.

All monitoring uses raw weights and learning split files. The `true_fixed`
diagnostic additionally reuses its immutable true **training** reset buffer for
simulator targets. Controlled prediction
uses known training/validation apparatus parameters and all ten subsequent
recorded-interval forces per latent transition. Passive prediction receives only
the current visual latent: no force, parameter, reset family, episode ID, or time
is an input. The passive MLP replaces the controlled causal transformer.
Monitoring does not fit physical parameters or access stored future states,
validation truth, or test trajectories. The separate test-evaluation commands do not currently upload
their results to W&B; there are no application `eval/*` keys here.

## Losses

Lower is better for each loss, subject to the collapse and supervision caveats
below. These are means, not sums; the physical residual uses fixed unit scaling.

| Exact key | Definition and meaning |
|---|---|
| `train/loss/prediction` | Mean squared difference between predicted next code and encoded next clip, averaged over `B × 5 transitions × 32 dimensions`. Controlled transitions use up to three actual preceding codes and aligned actions/parameters (teacher forcing); passive transitions use only the current actual code. Both prediction and target remain in the gradient graph. |
| `train/loss/sigreg` | Unweighted SIGReg statistic encouraging spread toward a standard normal distribution, computed independently across trajectories at each of six offsets, then averaged over offsets and 1,024 random projection directions. Definition below. |
| `train/loss/jepa` | `prediction + 0.1 * sigreg`. The current loss implementation fixes the coefficient at `0.1`; the configuration field `training.lambda_sigreg` is not read by the loss. |
| `train/loss/physics` | Mean squared physical residual over `B × 6 offsets × 5 coordinates`: scaled `r(E(clip))` versus a simulator rollout using known parameters/actions and the configured training reset at raw frame zero (`learned` or `true_fixed`). The solver integrates once from that reset to `s + 64` and gathers `s + ENDPOINTS`; the crop is never treated as a new centered reset. RK4 uses five 2 ms substeps per 10 ms force interval; fixed true-reset targets use float64, learned-reset targets float32. See scaling below. It does not supervise `r(P(...))` directly. |
| `train/loss/total` | `jepa + training.lambda_phys * physics`; the optimized scalar. Default physical weight is `1`. Inactive stage terms are zero. |
| `val/loss/pred` | Same teacher-forced prediction MSE on the fixed validation crops in evaluation mode. Note the key is `pred`, not `prediction`. |
| `val/loss/sigreg` | Same SIGReg statistic on validation codes, with a fresh generator seeded to `config.seed + 100` at each evaluation, so projection directions are repeatable. |
| `val/loss/jepa` | `val/loss/pred + 0.1 * val/loss/sigreg`. |
| `val/loss/physics_reset_prior` | Same scaled physical MSE, but the simulator starts from the reset prior: `p=0`, `v=0`, `q=0` for downward/nonlinear or `π` for upright, and `w=0`. Integrate from actual frame zero through the crop's endpoints. There is no fitted or true validation reset table, including in the `true_fixed` diagnostic. Logged for both corpora in `joint`, and for controlled `readout`. This is not directly comparable to either training physics target protocol and is not ground-truth physical accuracy. |

SIGReg projects each offset's codes onto 1,024 random unit directions. For each
direction it compares the empirical characteristic function (mean cosine and
sine of projected values times `t`) with that of a standard normal:
`(mean(cos(t*x)) - exp(-t²/2))² + mean(sin(t*x))²`. It integrates over 17 knots
on `[0,3]`, using doubled trapezoidal weights for both signs and an additional
`exp(-t²/2)` window, multiplies by `B`, then averages over directions and offsets.
There is no extra quadrature normalization. This is dimensionless; its finite
sample behavior depends on `B`, so it is not a physical error or a task score.

Low prediction loss alone can result from constant codes. Under `learned`, low
physical loss establishes consistency with jointly learned reset states, not
recovery of actual motion. Under `true_fixed`, it measures training-state accuracy
against a correct-reset simulation, while supplying additional supervision.
Use prediction controls, representation statistics and held-out trajectories
alongside the losses in both cases.

## Latent representation statistics

Each suffix below is emitted as **`{split}/latent/{suffix}`**, with
`split = train, train_eval, val`. Unless a ratio/rank/fraction, units are latent
units (standard deviations/norms) or squared latent units (variances/MSEs).

| Suffix | Calculation and interpretation |
|---|---|
| `std_mean` | Across-trajectory std for each offset/dimension, averaged over the `6 × 32` entries. Near zero suggests little representation diversity. |
| `std_min` | Minimum of those 192 standard deviations; detects a nearly constant coordinate at an offset. |
| `std_max` | Maximum of those standard deviations; a large value can reveal uneven scale. |
| `low_std_fraction` | Fraction of the 192 offset/dimension pairs with std strictly below `0.1`; range `[0,1]`. Larger means more nearly constant coordinates at this threshold. |
| `effective_rank` | For each offset, form the `32 × 32` population covariance across trajectories. Normalize its nonnegative eigenvalues to shares `p`; rank is `exp(-sum(p*log(p)))`, or zero for zero covariance. Average over six offsets. Larger means variance is distributed across more directions, at most `min(32,B-1)`. |
| `between_trajectory_variance` | Across-trajectory population variance at each offset/dimension, averaged over the 192 entries. Also equals the square-mean of the corresponding standard deviations, not `std_mean²`. |
| `within_trajectory_variance` | Population variance over the six offsets separately for each trajectory/dimension, then averaged over trajectories and dimensions. Measures temporal spread within videos. |
| `temporal_delta_mse` | Mean `(z[:,1:] - z[:,:-1])²` over trajectories, five adjacent transitions, and dimensions. Measures change over 0.10 seconds. |
| `temporal_to_between_variance` | `temporal_delta_mse / between_trajectory_variance`. Dimensionless; near zero indicates very little temporal change relative to diversity across trajectories. |
| `z_norm_mean` | Mean Euclidean length of the 32-dimensional code over trajectories and six offsets. Monitors scale, not accuracy. |

These are diagnostic quantities, not scores to maximize indiscriminately.
Static appearance can give high variance/rank; noisy codes can give high temporal
variation. The SIGReg target encourages roughly unit coordinate spread, but does
not make any single statistic a sufficient learning criterion.

## Future-latent prediction and controls

Keys expand over the combinations below, except the three action-only scores
are omitted for passive runs:

```text
{split}/prediction/teacher_forced/{reset}/{score}
{split}/prediction/autoregressive/{horizon}/{reset}/{score}
```

- `split = train_eval, val`.
- `reset = all, down, nonlinear, up`: all episodes, downward resets
  (`reset_mode=0`), nonlinear resets (`reset_mode=1`), or upright resets
  (`reset_mode=2`). These describe the episode reset, not the pose within its
  later crop. Each group is averaged independently; `all` pools episodes rather
  than averaging group scores.
- `horizon = h1, h2, h3`: respectively **0.10, 0.20, and 0.30 seconds** after the
  final observed code. Each horizon measures its endpoint, not cumulative error.
- Controlled `score` uses all nine suffixes below. Passive runs emit the six
  suffixes excluding `mse_shuffled`, `action_advantage`, and `sensitivity_mse`.

**Teacher forcing** scores five one-step transitions. The controlled `P` receives
its last one to three actual codes; passive `P` receives only the current actual
code. Persistence uses the actual code immediately before each transition.
**Autoregression** encodes histories ending at crop-relative frames `14,24,34`,
then predicts codes at `44,54,64`, feeding predictions back without future
observations. The controlled context retains at most three codes. The passive
MLP uses only the latest code, initially the observed code at frame `34`.
Persistence holds that frame-34 code for all three future horizons.

Neither protocol uses the readout `r`; targets are the current encoder's actual
future codes. Controlled protocols hold the recipient's true known apparatus
parameters fixed. Shuffled actions come from a deterministic cyclic permutation
of the fixed independent episodes, **which may cross apparatuses and reset
families**. Entire donor force crops are transferred with their temporal order
preserved; individual force entries are never shuffled. Teacher forcing replaces
all action-conditioning blocks. Autoregression retains the recipient's past
forces and replaces only its future crop-relative `force[34:64]` with the donor's
corresponding suffix. Passive runs have no action control.

Training and validation include collection with true-state feedback. Its future
actions can reveal future hidden state, so this action advantage is a learning
diagnostic, not the primary test of intervention prediction. The separate
held-out controlled evaluation uses prescribed open-loop query programs.

Let `A` be actual future codes, `C` model predictions (using correct actions for
controlled runs), `S` shuffled-action predictions when applicable, and `H` the
persistence baseline. Define `mean` over the selected
trajectories and all 32 dimensions, also over five transitions for teacher
forcing. Define `Ec=mean((C-A)²)`, `Es=mean((S-A)²)`, and `Eh=mean((H-A)²)`.

| Score suffix | Formula and interpretation |
|---|---|
| `mse_correct` | `Ec`; squared latent units, lower is better. |
| `mse_shuffled` | `Es`; squared latent units. Control error; compare with `mse_correct`. |
| `mse_persistence` | `Eh`; squared latent units. Error from retaining the previous/last observed code. |
| `persistence_skill` | `1 - Ec/Eh`; dimensionless, higher is better. `1` is perfect, `0` ties persistence, negative is worse than persistence. Undefined if `Eh=0`. |
| `action_advantage` | `(Es-Ec)/Eh`; dimensionless. Positive means correct actions help; zero means equal errors; negative means the shuffled control did better. Undefined if `Eh=0`. |
| `sensitivity_mse` | `mean((S-C)²)`; squared latent units. Measures whether changing actions changes predictions, not whether that change helps. |
| `predicted_motion_mse` | `mean((C-H)²)`; squared latent units. Displacement energy from the persistence reference. Near zero means almost static predictions. |
| `actual_motion_mse` | `Eh`, exactly the same value as `mse_persistence`; actual displacement energy from that reference. |
| `predicted_motion_ratio` | `predicted_motion_mse / Eh`; dimensionless. Near `0` suggests too little motion, near `1` matches displacement energy, above `1` exceeds it. Matching energy does not imply correct direction or timing. Undefined if `Eh=0`. |

Ratios divide pooled MSEs; they are not averages of per-trajectory ratios.
An absent reset subset gives undefined scores. Since the encoder changes during
training, raw latent MSEs can also change with representation scale; compare
them with their same-snapshot controls. These metrics are not physical trajectory
accuracy or balancing success. Monitoring horizons stop at 0.30 seconds because
the fixed diagnostic crop has six latent endpoints. Separate held-out evaluation
uses 0.1, 0.2, 0.4, 0.8, 1.6, and 3.2 second horizons where targets are available,
with valid-query counts and boundary-exit rates. Those test results are local,
not these W&B keys. Raw latent errors from passive versus controlled corpora are
not an apples-to-apples action or physics ablation.

## Physical readout and simulator consistency

`r(z)` outputs `[p,v,sin(q),cos(q),w]`: cart position in metres, velocity in m/s,
angle sine/cosine (dimensionless), and angular velocity in rad/s. The angular
pair is normalized. The physical loss compares
`r(z)/[2,2,1,1,5]` with `[p_sim/2,v_sim/2,sin(q_sim),cos(q_sim),w_sim/5]`.
Divisors correspond to 2 m, 2 m/s, and 5 rad/s scales, so residuals are
dimensionless. Angle error is represented by sine/cosine differences, not radians.

| Key template | Calculation and interpretation |
|---|---|
| `{split}/readout/std_{coordinate}` | Population std pooling **both trajectories and all six offsets**. `split = train, train_eval, val`; `coordinate = p, v, sin, cos, w`. Uses unscaled physical readout units listed above. Low values suggest a nearly constant readout; larger values are not automatically better. |
| `{split}/physics/residual_{coordinate}` | Per-coordinate mean squared scaled residual over trajectories and six offsets. `split = train, train_eval`; same five coordinate suffixes. Lower means closer agreement with the configured-reset simulator (`learned` or `true_fixed`). The mean of the five `train/` residuals equals `train/loss/physics` up to numerical precision. |

These families are omitted during `jepa` training. Passive `joint` runs include
them, using fixed known parameters and zero actions. There are no
`val/physics/residual_*` keys: validation lacks a fitted or true reset table. Its
separate aggregate `val/loss/physics_reset_prior` uses the prior described above.
The two key families in this table decode actual observed clips, not predicted
future codes.

### Physics consistency of predicted future latents

The following additional **diagnostic-only** keys decode the existing
correct-action autoregressive latent forecasts with the same physical readout.
They do not add a term to `train/loss/total` or change any training gradient.

```text
train_eval/physics/autoregressive/{score}
train_eval/physics/autoregressive/{horizon}/{score}
```

- `horizon = h1, h2, h3`: endpoint errors **0.10, 0.20, and 0.30 seconds** after
  the final observed latent at crop-relative frame `34`.
- `score = mse, residual_p, residual_v, residual_sin, residual_cos, residual_w`.
  These are all 24 keys: six pooled scores and six scores at each of three
  horizons. There are no reset-family subgroups, shuffled-action scores, or
  persistence scores in this physical metric family.

For trajectory `i` with crop start `s_i`, reuse the predictions from the
autoregressive protocol above: the controlled predictor starts with observed
latents at `14,24,34` and receives its recorded forces and known parameters;
the passive predictor starts with the observed frame-34 latent and receives
no actions or parameters. Predict successively at `44,54,64`, feeding each
predicted latent back into `P`. No encoded future latent replaces a prediction.
The simulator still starts at the episode's configured reset state (learned or
fixed true training reset), **not** from
the readout of the last observed latent. It uses recorded forces for the
controlled corpus, or zero forces and nominal parameters for the passive
corpus, and integrates to the same absolute episode endpoints as the forecasts.

Let `B` be the fixed training reference's episode count (normally 48, or the
entire training split if it has fewer episodes), `h ∈ {1,2,3}`, and
`t_i,h = 0.01 * (s_i + 34 + 10h)` seconds. Define the five-coordinate residual

\[
e_{i,h}
=D^{-1}r(\hat z_{i,h})
-\iota\!\left(\Phi_{t_{i,h}}(\hat x_{i,0};\theta_i,u_i)\right),
\qquad D=\operatorname{diag}(2,2,1,1,5),
\]

where the simulated coordinates have the same scaling as the existing physical
loss, and `hat z` denotes a predicted latent, not an encoded future observation.
Here `hat x_i,0` means the learned reset in the default protocol and the fixed
true training reset in the explicit diagnostic. Neither is estimated from the
forecast seed for this metric.

| Exact key or template | Calculation and interpretation |
|---|---|
| `train_eval/physics/autoregressive/mse` | `sum(e²) / (B * 3 * 5)`. Pools the three forecast endpoints and all five scaled coordinates. Equals the mean of the three horizon MSEs, and the mean of the five pooled coordinate residuals. |
| `train_eval/physics/autoregressive/residual_{coordinate}` | `sum_i,h(e[i,h,coordinate]²) / (B * 3)`, for `coordinate = p, v, sin, cos, w`. Locates which physical quantity disagrees with the simulator. |
| `train_eval/physics/autoregressive/{horizon}/mse` | `sum_i,coordinate(e[i,h,coordinate]²) / (B * 5)`. Each horizon measures only its endpoint, not a cumulative mean through that horizon. |
| `train_eval/physics/autoregressive/{horizon}/residual_{coordinate}` | `sum_i(e[i,h,coordinate]²) / B`. Coordinate MSE at the specified forecast endpoint; uses the same five suffixes and three horizons listed above. |

All values are nonnegative, dimensionless squared residuals; **lower is
better** for consistency. Sine/cosine residuals measure angular disagreement
without a wrap discontinuity; they are not angular errors in radians. These
metrics have no ratio denominator or empty reset subgroup, so they have no
routine undefined case on a nonempty reference. Nonfinite results follow the
logger's general policy: JSON `null` locally and omission from W&B, never zero.

They are emitted only for the fixed **training** reference in passive/controlled
`joint` and controlled `readout` stages. They are omitted in `jepa` and for
validation, which has no fitted or true reset table. All networks are in evaluation
mode, gradients are disabled, and diagnostic RNG state is preserved. The step
axis is `train/update`, with the existing fixed-diagnostic schedule: initial
snapshot (at the restored update on resume), every 500 updates, every media
update (default 1,000), and the final update when monitoring is enabled.
Configuration overrides described above apply. Scalars are saved in
`diagnostics.jsonl` and sent to W&B when enabled; no additional media or
`losses.csv` columns are introduced.

In the `learned` protocol, target and readout are learned parts of the same
training system: a constant equilibrium readout can also score well, especially
in passive data. In the `true_fixed` diagnostic, the target is instead fixed by
the true training reset and known equations/actions, so these are physical
forecast errors against that simulated trajectory. In both cases this is a
training-split check at short horizons; it does not establish generalization or
accurate long rollouts. Read alongside latent
persistence/action controls, readout variation, and held-out decoded
trajectories. Existing `train_eval/physics/residual_*` scores use **six observed
endpoints**, whereas the pooled forecast scores use **three future endpoints**;
their difference is not a matched-endpoint estimate of the predictor's added
error.

## Gradients, parameter changes, and reset bounds

| Key template or exact key | Calculation and interpretation |
|---|---|
| `train/gradients/{module}/norm` | L2 norm of all parameter gradients in `module = encoder, predictor, readout, initial_conditions`, before clipping. Unused/frozen modules normally report zero; absent modules are omitted. Passive `jepa` runs emit only `encoder` and `predictor`: they have neither a readout nor a reset table. Passive `joint` additionally emits `readout` and `initial_conditions`. Controlled `jepa` retains an inactive readout with zero gradients but has no reset table. In `true_fixed`, `initial_conditions` is parameterless and its norm is correctly zero. Magnitudes depend on parameter scaling; there is no universal target value. |
| `train/gradients/{module}/nonzero_fraction` | Nonzero gradient elements divided by **all parameter elements** in that module, including elements with absent gradients as zeros; range `[0,1]`. Same stage/corpus-dependent module names as the norm. A parameterless fixed-reset module reports zero by convention, not an undefined ratio or a learning failure. |
| `train/gradients/neural_norm_before_clip` | Combined pre-clipping L2 gradient norm of active neural-network parameters. Excludes the reset table. |
| `train/gradients/clipping_applied` | `1` when that norm exceeds `training.grad_clip` (default `1`), otherwise `0`. A flag for this sampled update, not an average clipping frequency. |
| `train/updates/{probe}/norm` | L2 norm of `parameter_after - parameter_before` for the named probe tensor, for this one optimizer step. Parameter units; not a full-network update norm. Probe names below. |
| `train/updates/{probe}/relative_norm` | Probe change norm divided by its pre-update parameter norm. Dimensionless; undefined if the old norm is zero. |
| `train/initial_conditions/saturated_fraction` | Fraction of **all learned training reset-table elements**, not only this batch, satisfying `abs(tanh(raw)) > 0.95` after the update. Range `[0,1]`; high values mean many optimized reset coordinates are near their allowed bounds. Only `joint` and `readout` with learned resets. Omitted in `true_fixed`, whose states do not use a trainable bounded parameterization. |
| `train/initial_conditions/trainable` | Protocol flag: `1` for an optimized learned reset table in `joint`/`readout`, `0` for `true_fixed` or JEPA-only training. Logged at the initial diagnostic snapshot and monitored training steps. Unitless metadata, not an improvement metric. |
| `train/initial_conditions/true_reset_supervision` | Protocol flag: `1` only for the `true_fixed` joint diagnostic, `0` for learned-reset and JEPA-only runs. Same schedule as `train/initial_conditions/trainable`. Identifies the additional training-reset supervision; not a task score. |
| `train/active_encoder` | `1` in `joint`/`jepa`, `0` in `readout`. Stage metadata, not a measured gradient/activity score. |
| `train/active_readout` | `1` in `joint`/`readout`, `0` in `jepa`. Stage metadata. |

Probe names expand as follows; inactive probes are omitted rather than logged
as zero:

| Probe | Tensor | Stages |
|---|---|---|
| `encoder_stem` | `encoder.backbone.conv1.weight` | `joint`, `jepa` |
| `predictor_conditioning` | Controlled `predictor.conditioning[0].weight` | Controlled `joint`, `jepa` |
| `predictor_input` | Passive `predictor.net[0].weight` | Passive `joint`, `jepa` |
| `predictor_output` | Controlled `predictor.output_projection.weight`, or passive `predictor.net[-1].weight` | `joint`, `jepa` |
| `readout_input` | `readout.net[0].weight` | `joint`, `readout` |
| `initial_conditions` | Entire `table.raw` | Learned-reset `joint`, `readout`; omitted for `true_fixed` |

The learned reset table has three raw entries per training trajectory. They map to
`v0=tanh(a)`, `q0=q_base + π*tanh(d)`, and `w0=3*tanh(c)`, with `p0=0`, where
`q_base=π` only for upright metadata (`reset_mode=2`) and zero otherwise.
Thus the bounds concern 1 m/s, a full ±π-radian range around the reset angle,
and 3 rad/s. This state belongs to the real episode reset, not its sampled crop;
no cropped state is constrained to have cart position zero. Initialization uses
coarse reset metadata only, never hidden truth. Its zero initialization makes
the first relative update norm undefined. Reset gradients are not part of
neural clipping. JEPA-only runs in either corpus have no reset table.

The `true_fixed` diagnostic stores the four physical reset coordinates per
training episode as immutable float64 buffers, with no raw parameters, reset
optimizer or reset update probes. Its zero reset gradients are expected. Neural
optimizer/update diagnostics retain their usual interpretation.

## Progress, timing, and failures

| Exact key | Meaning |
|---|---|
| `train/update` | Optimization update associated with this record; custom x-axis for all application metrics. For a failed objective it denotes the attempted update, which did not complete. |
| `train/resumed_from_update` | Restored checkpoint step, or `0` for a new run; emitted only with the initial diagnostic snapshot of a segment. |
| `train/lr` | Current scheduled learning rate of the neural AdamW optimizer. Does not report the separate learned-reset Adam learning rate (`training.initial_lr`, default `0.01`); that optimizer is absent in `true_fixed`. |
| `train/seconds` | Cumulative training-loop wall time in seconds. Includes monitoring/logging overhead already incurred and earlier updates' evaluation/checkpoint work; excludes data loading/caching before the loop. Resume adds the retained CSV elapsed time. |
| `train/updates_per_second` | Completed updates in the current fresh/resumed segment divided by its elapsed wall time at the logging point. A cumulative segment rate, not an instantaneous batch rate. Higher means faster execution, not better learning. |
| `failure/nonfinite_objective` | `1` only when the training objective is NaN/infinite. A failure checkpoint and JSON reason are saved, then training raises an error. Not emitted on healthy steps; absence is not a comprehensive success indicator. Other exceptions do not automatically produce this key. |

## Images and histograms

These are media entries, not scalar scores. For the `{split}` families below,
`split = train_eval, val`; they follow the media schedule above.

| Key | Contents and interpretation |
|---|---|
| `{split}/examples` | First two fixed randomly selected episodes/crops, without guaranteed reset-family coverage; six endpoint images each. Labels show true episode times `(crop_start + [14,24,34,44,54,64])*0.01` seconds, not times measured from a fictitious crop reset. Shows actual conditioning/target observations. |
| `{split}/latent_forecasts` | For the first two fixed crops, heatmaps of actual, model-predicted, and persistence future codes, each minus the last observed code. Controlled runs additionally show shuffled-action predictions; passive runs omit that panel. Dimensions versus three future steps; all panels in one figure share the same symmetric color scale. The scale can change between logging updates. Persistence is zero throughout. |
| `train_eval/physics_fit` | First training-reference trajectory: five scaled coordinates of `r(E(video))` versus simulation from its configured reset. Only `joint`/`readout`; no validation equivalent. Under `learned` this is simulator self-consistency; under `true_fixed` it compares against a true-reset simulation. The title identifies the protocol. It does not plot predicted latent readouts. |
| `train_eval/physics_phase` | Long p–q trajectory comparison for three fixed training episodes: the first episode of each available reset family (downward, nonlinear, upright) in manifest order, selected independently of model errors. Black: dense simulator supervision at 10 ms intervals from the current learned or fixed true reset. Blue: `r(E)` of observed images at 100 ms endpoints. Orange: `r(P)` seeded with encoded observations at 0.14/0.24/0.34 s, then rolled forward freely for up to 32 × 100 ms without future image inputs; the passive predictor uses only the last seed. All curves cover 0.34–3.54 s, or stop at the last available 100 ms endpoint for shorter episodes. Controlled predictions use the recorded forces and known training apparatus parameters. Axes use p in metres and q in radians, with readout q = `atan2(sin,cos)`. Lines break at ±π; open circles mark starts and squares mark ends. Only physical stages (`joint`, controlled `readout`), initially, every `media_every` updates (default 1,000), and finally; no validation version. No stored future truth is loaded. Under `true_fixed`, the simulator is anchored to the true training reset; under `learned`, it is a self-consistency target. Missing reset families produce fewer panels. This is a qualitative diagnostic, with no aggregate score or expected numerical range; a p–q projection omits velocities and cannot alone establish correct timing or full-state recovery. |
| `{split}/latent_values` | Histogram of all `B × 6 × 32` latent values. Shows distribution and scale, but pooling can hide offset/dimension-specific collapse. |
| `{split}/latent_dimension_std` | Histogram of 192 population stds across trajectories, one for each of six offsets and 32 latent dimensions. Corresponds to the `latent/std_*` scalar statistics. |

Histograms discard nonfinite values and use 64 bins. Edges are chosen separately
for each snapshot, so bins are not fixed across updates. If there are no finite
values, the histogram is omitted. Images are saved as PNGs and histogram counts
and edges as NPZs in the run's `diagnostics_media/` directory. Trajectory videos
generated by the separate visualization scripts are local artifacts and are not
currently uploaded by this training logger.

The phase figure is computed by
[`trajectory_diagnostics.py`](../src/pi_jepa/trajectory_diagnostics.py) using the
same causal history encoding and autoregressive rollout as latent evaluation.
Its fixed episode IDs, reset families, available lengths, and endpoints are
recorded in `diagnostic_windows.json` under `training_trajectories`. It caches
at most three uint8 learning-video prefixes (355 frames each), runs without
gradients at media events, and restores module modes and training RNG state.
This longer visual rollout does not change the existing three-step scalar
diagnostics or the training objective.

## Missing values, local files, and W&B bookkeeping

Undefined ratios, empty-group statistics, and other nonfinite scalar values are
written as JSON `null` in `diagnostics.jsonl` and omitted from the W&B payload.
They must not be interpreted as zero or perfect performance. W&B may retain an
older summary value when a later value is omitted; inspect the step history.

Every application scalar above is saved locally in `diagnostics.jsonl`, with
media paths alongside it. `losses.csv` additionally records every update, even
without monitoring. Its `latent_variance` is across-trajectory variance averaged
over offsets/dimensions, matching `train/latent/between_trajectory_variance` on
sampled steps. Its `readout_variance` is the same aggregation of **scaled**
readout coordinates, not the pooled unscaled `readout/std_*`. It is `NaN` for
passive `jepa` runs because no readout exists; passive `joint` records it normally. Controlled `jepa` can still record the
inactive, untrained readout's CSV variance, which is not evidence of physical
learning; readout W&B diagnostics remain omitted. Neither CSV column is
separately uploaded as a W&B metric. `validation.csv` contains validation
losses. These files remain usable with tracking disabled.

W&B also adds its own bookkeeping: `_step` is its monotonically increasing log
record index, `_timestamp` is the record's Unix wall-clock timestamp, and
`_runtime` (also represented in `_wandb.runtime` summary metadata) is SDK run
elapsed time in seconds. These are different from optimization updates and
`train/seconds`. Automatic system charts report host/device telemetry, such as
utilization, memory, temperature, and I/O, with keys/units depending on the SDK
and hardware. They are not application learning metrics or task success; this
repository does not define their sampling or formulas.

W&B identity is saved in `wandb_run.json` and checkpoints. Online resume reuses
the same run, appending history with `train/update` as the custom axis; replayed
updates can have repeated x-coordinates. Local CSV/JSONL rows beyond the restored
checkpoint are removed. Offline resume, or changing tracking mode, starts a
linked segment. Logging and diagnostic media preserve Python, NumPy, and Torch
RNG states. No model/checkpoint/source-code artifacts are automatically uploaded.

## Implementation and maintenance

- [train.py](../src/pi_jepa/train.py): stages, loss weights, schedules, optimizer
  diagnostics, timing, validation losses, and failure indicator.
- [initial_conditions.py](../src/pi_jepa/initial_conditions.py): explicit fixed
  true training-reset diagnostic, immutable buffers and source identity.
- [losses.py](../src/pi_jepa/losses.py): prediction, SIGReg, and physical loss.
- [training_diagnostics.py](../src/pi_jepa/training_diagnostics.py): scalar
  statistics, gradient definitions, prediction protocols, and control formulas.
- [training_monitoring.py](../src/pi_jepa/training_monitoring.py): fixed splits,
  parameter probes, images, and histogram arrays.
- [experiment_logging.py](../src/pi_jepa/experiment_logging.py): local/W&B
  payloads, undefined values, media serialization, step axes, and resume.

When changing any emitted key, formula, units, split, baseline, stage condition,
schedule, or media, update this reference in the same change and check it against
the logging code. Expand the templates with their listed values to match exact
dashboard keys. Preserve a note about renamed/removed keys when older runs need
it.

### Changes from the retired 25 Hz experiment

The main key names are retained, but their dataset/time semantics changed:
`h1/h2/h3` now mean 0.1/0.2/0.3 s (previously 0.32/0.64/0.96 s), and windows now
have 65 dense frames with overlapping causal histories. Validation now contains
96 episodes, and training references/crops are sampled across the full split.
`nonlinear` prediction subgroups were added; upright reset metadata is now `2`
(previously `1`). Shuffled-action donors now cycle across the whole fixed
reference, rather than matching apparatus/reset groups. The fitted initial-angle
range expanded from ±0.8 to ±π radians, and cropped physical supervision must
integrate from the actual frame-zero reset. `val/loss/physics_reset_prior` is now
omitted in `jepa` mode. Passive runs introduce the `predictor_input` update probe
and omit action-control keys; passive JEPA-only runs also omit readout keys. These changes prevent direct comparison
of old versus new metric values without accounting for the changed experiment.

### Changes for the 0.85 m / ±1.5 m dataset

Schema 3 changes the physical rod length and track limit, and uses the square
±2.6 m camera. The physical residual divisors remain `[2,2,1,1,5]`; the 2 m
position divisor is a loss scale, not the new boundary. Logged key names and
monitoring horizons are unchanged. Passive `joint` now emits the existing
readout/physics/reset diagnostics and media, including the validation reset-prior
loss; its predictor remains action-free. In the passive problem, a nearly constant
equilibrium readout can attain a low physics loss, so compare those metrics with
readout motion and decoded held-out trajectories. Interface-4 checkpoints record
the new geometry and dataset identity; old checkpoints are incompatible.

### Added predicted-latent physics diagnostics

`train_eval/physics/autoregressive/*` adds physical consistency checks on
autoregressive predictions at the existing 0.1/0.2/0.3 s monitoring horizons.
Existing observed-latent physics keys, their calculations, the training
objective, and checkpoint format are unchanged. Older logs lack the new keys;
their absence does not mean zero error.

### Causal normalization and explicit true-reset diagnostic

Interface 5 replaces encoder BatchNorm with GroupNorm/hidden LayerNorm and
encodes the six offsets separately. There are no encoder batch-statistic or
running-statistic differences between train and evaluation. Old interface-4
weights cannot be resumed by this architecture.

New protocol flags `train/initial_conditions/trainable` and
`train/initial_conditions/true_reset_supervision` distinguish learned-reset runs
from the joint-only fixed true-reset diagnostic. The diagnostic omits reset
update probes and saturation metrics, retains zero reset-gradient metrics, and
changes existing training physics targets to simulations from fixed true
training resets. Validation still uses its old truth-free prior. These physics
losses must not be compared as though their supervision were identical.
`train_eval/physics_phase` adds an observed/predicted readout trajectory image;
`train_eval/physics_fit` now identifies the reset protocol in its target label.
No prediction-physics training term is added.

After merging the normalization/reset diagnostic into the main repository,
`train_eval/physics_phase` was expanded from two short random crops to three
fixed reset-family examples with up to 3.2 s of autoregressive prediction. Its
key and cadence are unchanged. The earlier audit-only gray private-state curve
is not part of this training logger; its simulator curve uses the actual
physics-loss reset protocol.
