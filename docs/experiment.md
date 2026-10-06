# Scientific experiment

The [dataset addendum](../physics_jepa_cartpole_dataset_addendum.md) replaces the
old collection and timing sections of the [original brief](physics_jepa_cartpole_codex_prompt.md).
The approved 2026-10-05 geometry revision uses a 0.85 m rod and a ±1.5 m cart limit;
it supersedes those values in the original addendum. The standalone
[new dataset report](new_dataset_report.md) records the resulting collection.
Commands are in the [README](../README.md);
[metrics.md](metrics.md) defines the logged diagnostics.

## Physics and grounding

State is `[p,v,q,w]`: cart position (m), velocity (m/s), pole angle (rad), angular
velocity (rad/s). The pole is a point mass `m=0.2 kg` at the end of a massless
`ell=0.85 m` rod; `q=0` is down and `q=pi` upright. Gravity is `9.81 m/s²`.
Cart parameters are mass `M` and viscous drag `b`; positive applied force points
right. Generation and differentiable fitting share the same PyTorch equations
and RK4 implementation. Angle remains unwrapped during integration.

The ordinary `training.initial_conditions=learned` experiment uses weak grounding:
commanded `p0=0`, coarse reset family, calibrated camera,
known geometry, applied forces, and training apparatus parameters are available.
Actual initial angle and velocities are hidden. True states are used for data
collection/QA and scoring, never neural training supervision in that protocol.
The explicit `true_fixed` diagnostic below additionally supervises the exact
training reset state. A fixed track-origin
mark remains visible. The fixed camera uses equal horizontal/vertical pixels per
metre, preventing orientation-dependent apparent rod length. Both axes span
`[-2.6,2.6] m` in 96×96 RGB, so the 0.85 m rod projects to 15.53 pixels at every
angle. Twofold supersampling improves the raster edges without stretching them.

Clocks are 2 ms RK4, 10 ms recorded observations, and 20 ms piecewise-constant
force/controller updates. A saved `force[k]` advances `state[k]` to `state[k+1]`.
Thus N forces accompany N+1 observations; complete action holds have two equal
record entries, and a truncated prefix may contain a final unpaired action.
No numerical step crosses a force discontinuity.

## Two separate prediction problems

Passive collection fixes `[M,b]=[1,0.25]` and every applied force to zero. Each
split mixes 50% downward oscillations, one-third nonlinear excursions, and
one-sixth upright falls. Reset velocities are nonzero in general. The predictor
is an unconditioned `32→128→128→32` GELU MLP receiving only its current latent.
This tests autonomous motion, not balancing or parameter identification. Passive
`--mode joint` fits a physical readout and, with the default `learned` protocol,
one hidden reset state per training episode, using the same objective as controlled joint training with known fixed
parameters and zero forces. An equilibrium readout can satisfy zero-input physics
without explaining visible motion, so low physics loss alone does not establish
grounding. Inspect decoded trajectories, readout variation, representation
statistics, and persistence skill. `--mode jepa` remains a baseline; passive
post-hoc `--mode readout` is not implemented.

Controlled collection draws 64 training and eight independent validation
apparatuses from `M∈[0.7,1.3]`, `b∈[0.05,0.5]`. Each training apparatus has eight
pulse, four multisine, eight noisy-LQR, and four LQR-release episodes; validation
uses half these counts. Half of pulse episodes use opposite-sign impulse pairs.
Training reuses 32 pulse and 32 multisine programs across apparatuses and fresh
resets. Validation/test programs use independent streams. Reset allocations vary
within waveform families, and neither IDs nor collection labels are neural inputs.

LQR is analytic collection machinery, not an expert to imitate. Every apparatus
uses gains designed once at `[1,0.25]`, recomputed for the 0.85 m rod. Excitation, varied references, and controller
release broaden motion; the logged force is the complete clipped feedback plus
excitation. Failed balance is retained. Position is neither clipped nor recentered;
the first saved state beyond `|p|=1.5 m` terminates recording. Finite training prefixes
with at least 65 observations are retained; only shorter attempts are regenerated
within their planned family. Held-out attempts are never resampled for convenience.

All episodes have independent reset streams and stay in one split. Controlled
held-out parameters are only in private truth/configuration, with opaque public
identities. The 12 existing held-out pairs are retained: six interior and six
extrapolation. Each gets two three-second calibration episodes and eight fresh
four-second open-loop queries (four pulses, four multisines; four downward,
two nonlinear, two upright resets). Future query forces have no true-state feedback.

## JEPA histories and objectives

Each input history contains eight saved images at raw indices
`[e-14,e-12,e-10,e-8,e-6,e-4,e-2,e]`, stacked into 24 channels and normalized to
`[-1,1]`. Histories span 0.14 seconds. Six-token crops end at relative records
`[14,24,34,44,54,64]`; each transition advances 0.1 seconds. Consecutive histories
share three images. This overlap is causal but can still make persistence strong.
Independent episodes are sampled uniformly, then one uniformly chosen valid crop
per episode. SIGReg samples are independent episodes, not nearby overlapping crops.

The default small encoder keeps the spatial ResNet-18 shape but replaces all 20 backbone
BatchNorm layers with GroupNorm (32 groups). Its projector is
`Linear(192,256) → LayerNorm(256) → GELU → Linear(256,32)`; there is no final
normalization of the latent. Each of the six history offsets is encoded in an
explicit separate forward call. Both normalization layers operate within a
single clip, so its code does not depend on other episodes or future clips
through minibatch statistics. Train/eval normalization uses the same calculation
and has no running means or variances. SIGReg still compares independent episodes
at each offset; that loss-level comparison is not an encoder input dependency.

This deliberately supersedes the BatchNorm architecture in the original brief.
BatchNorm previously mixed all six offsets in the encoder batch and produced a
large measured train/eval discrepancy at checkpoint 1,000. Removing it addresses
that confound; it does not by itself prevent an ungrounded physical readout.
Checkpoint interface **6** records and validates the full encoder/predictor size
specification. Earlier interfaces are unsupported in this HPC version; models
start from fresh initialization. The [H200 campaign](../hpc/README.md) adds
ResNet-34/ResNet-50 encoders and larger predictors, while keeping latent
dimension 32, the physical readout, and temporal interfaces fixed.

The small controlled predictor retains three width-192, three-head transformer blocks,
causal attention, at most three latent tokens, and AdaLN-zero conditioning. Each
token gets all ten subsequent recorded-interval forces `force[e:e+10]`, divided by
5 N, plus mass/drag normalized using centres `[1,0.275]` and scales `[0.3,0.225]`.
The passive predictor has no force/parameter input or conditioning path.

`L_JEPA=MSE(z_pred,z_next)+0.1*SIGReg(z)` keeps gradients through both endpoints,
without an EMA teacher. SIGReg is unchanged: 1,024 random directions, 17 quadrature
knots over `[0,3]`, Gaussian window, doubled trapezoidal weights, and batch-size
multiplier, applied across episodes independently at each offset.

The joint models' physical readout `r:32→64→5` returns `[p,v,sin(q),cos(q),w]`, normalizing
the angular pair. In the default learned-reset protocol, one table row per training episode parameterizes
`[0,tanh(a),q_base+pi*tanh(d),3*tanh(c)]`, with `q_base=pi` for upright and zero
otherwise. It starts from coarse metadata, not true resets. For a crop starting
at raw index `s`, physics integrates once from raw reset zero through `s+64`, then
gathers the six actual endpoints. Cropping does not reset position to zero.
Twenty-record checkpoint segments trade recomputation for lower backward memory
without changing this objective or the force grid.

The physical loss is MSE between `r(z)/[2,2,1,1,5]` and
`[p_sim/2,v_sim/2,sin(q_sim),cos(q_sim),w_sim/5]` at observed-history endpoints.
It does not directly supervise future `r(P(z))`. Joint training on either corpus optimizes
`L_JEPA+L_phys` from the first update; controlled post-hoc training freezes encoder/predictor
parameters and buffers while fitting the readout/reset table. Frozen codes are
recomputed for sampled crops rather than caching a fixed window for each episode.

Original optimization budgets remain 10,000 updates per stage, batch size 64,
AdamW at `3e-4` with 500-update warmup and cosine decay to `3e-5`; learned reset-table Adam
uses `1e-2`. Neural gradient norm is clipped at 1; the learned table is excluded. No
validation/test criterion chooses checkpoints. Raw final weights are used.

## Fixed true-reset diagnostic

`configs/true_reset_diagnostic.json` selects `training.initial_conditions=true_fixed`.
The equivalent CLI override is `--initial-conditions true_fixed`. This protocol
supports `--mode joint` on either corpus; it is intentionally separate from the
original learned-reset experiment and does not apply to JEPA-only or post-hoc stages.

**Hypothesis:** a major obstacle to physical learning is the jointly optimized
reset table providing incorrect, nearly stationary simulator targets. Holding
the correct training resets fixed tests whether the encoder/readout and latent
predictor can instead learn visible physical motion with the same dynamics model.

The loader opens only each training episode's private `exact_initial_state`
field. No stored future state arrays, validation truth, or test truth are used.
These four physical coordinates are immutable float64 buffers, indexed by
training episode and saved in the checkpoint. They are never encoder/predictor
inputs. There are no trainable reset parameters, reset gradients, or reset
optimizer. The simulator integrates these fixed states in float64, using the
same equations, recorded forces, clocks and physical units as collection.
The learned-reset comparator retains its existing float32 differentiable solver.

For a crop beginning at `s`, the target remains the simulation from the original
reset through the absolute endpoints `s+[14,24,34,44,54,64]`. The joint objective
still sums next-latent prediction, SIGReg and observed-latent physics residuals.
Its physics targets can no longer become easier by changing the reset. Predicted
readouts `r(P(z))` are monitored but are not an extra training loss.

This is **additional true-state supervision at reset**, even though later targets
are recomputed by the simulator. Success would show learning is possible with
that supervision; it would not demonstrate recovery from equations and images
alone. Compare it with a fresh `learned` run using this same GroupNorm/LayerNorm
architecture, seed, data, update budget and optimizer settings. Comparing only
with the old BatchNorm run cannot isolate the effect of fixed resets. The
diagnostic does not promise success: latent prediction, finite context and long
autoregressive rollouts remain learning problems.

The configuration gives diagnostic runs W&B group `causal-true-reset-diagnostic`;
run names include `true-fixed-reset-diagnostic`. Provenance/checkpoints record
the reset protocol, source manifest identity and ordered reset-value digest.
Resume rejects changes to this protocol or its source. Validation still uses
the unchanged coarse reset prior, with no fitted or true validation states;
its physics loss must not be read as a held-out true-state error. Test forecasts
still initialize from observed images, never from the training reset table or
true test resets. The passive and controlled diagnostic runs are independent.

## Calibration and prediction

Apparatus-parameter calibration applies only to controlled data. Passive joint
models decode predicted/observed latents with their own learned physical readout
and are evaluated against held-out passive truth, without parameter fitting.
The two corpora train independent models from scratch; the controlled run does
not initialize from the passive checkpoint.

Calibration keeps networks frozen and fits apparatus parameters plus an unknown
state at the first observed endpoint, frame 14 (0.14 s). Its position is unknown,
since the centred reset occurred earlier. Solver actions start at raw interval14.
Calibration never uses query outcomes or true states/parameters.

Forecasts begin with observed histories ending at frames14,24,34. Controlled
rollouts then use only predicted future latents, known force blocks and the
selected nominal/fitted/oracle parameter condition. Passive rollouts apply the
MLP to the last latent. Future observations are encoded only as scoring targets.
The persistence control keeps the last observed code. Controlled held-out
diagnostics also exchange complete planned programs or set conditioning forces
to zero, including those associated with the observed context. Training-time
autoregressive monitoring instead changes only the future force suffix. Matched
saved queries and donors are reused across
models. Physical decoding of predicted codes is scored separately from decoding
actual observed images; simulator forecasts from inferred states are labeled
separately from learned latent dynamics.

Report horizons0.1,0.2,0.4,0.8,1.6,3.2s after observed context, with valid-target
counts and boundary-exit rates. Short queries stay in attempted denominators;
missing target frames are masked, not synthesized. Positive persistence skill
and correct-action advantage are useful diagnostics but do not alone establish
physical grounding. Passive/controlled raw latent losses measure different
problems and must not be treated as an action or physics ablation.
