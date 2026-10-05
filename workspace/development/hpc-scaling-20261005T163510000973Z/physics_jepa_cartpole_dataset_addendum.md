# Addendum: dense passive and controlled cart-pole datasets

This addendum supplements `physics_jepa_cartpole_codex_prompt.md`. Implement the dataset generation, storage, loaders, and temporal-indexing changes described here in the existing research code.

**This addendum overrides the old data-collection recipe, old timesteps, hard-coded 48-frame trajectories, hard-coded clip endpoints, and hard-coded eight-force conditioning blocks.** Keep the custom point-mass cart-pole equations. Keep the established JEPA loss and the joint-versus-post-hoc comparison unless explicitly changed below.

The purpose is to create two useful corpora:

1. **Passive:** video of freely evolving cart-pole trajectories with exactly zero applied force, for an action-free latent predictor.
2. **Controlled:** video, applied-force sequences, and separately stored true-state transitions, covering both nonlinear motion and upright stabilization.

The main scientific question is prediction of motion, and, for the controlled corpus, prediction of the consequences of chosen forces. The presence of a stabilizing controller during collection does not mean the JEPA should imitate that controller.

Prioritize readable exploratory code. Use one seed, fixed defaults, and a few cheap implementation checks. Do not add an RL training framework, train an expert, run seed sweeps, or launch extra neural training experiments as a prerequisite. This addendum requests data and compatible interfaces; it does not by itself request four new model-training runs.

## 1. Keep the physical system and conventions explicit

Use the existing custom continuous-force system:

```text
state = [p, v, q, w]
p: cart position, m
v: cart velocity, m/s
q: angle, rad; q=0 hanging downward, q=pi upright
w: angular velocity, rad/s
u: actual horizontal force, N; positive to the right

pole point mass m = 0.2 kg
massless rod length ell = 0.5 m
gravity g = 9.81 m/s^2
cart parameters theta = [M, b]
M: cart mass, kg
b: viscous cart drag, N*s/m
```

The right-hand side remains:

```python
D = M + m * sin(q)**2
p_ddot = (u - b*v + m*sin(q)*(ell*w*w + g*cos(q))) / D
q_ddot = -(g*sin(q) + cos(q)*p_ddot) / ell
dx_dt = [v, p_ddot, w, q_ddot]
```

Maintain one implementation used by generation and the differentiable physical solver. Do not replace the point mass with a uniform rod. Do not import a CartPole model with another angle convention.

Keep angle unwrapped internally during integration. Wrap it only for displays, circular errors, or conversion to sine/cosine. A crossing of the -pi/pi boundary is not a physical jump.

## 2. Separate the clocks: this is essential

The earlier code used 20 ms RK4 substeps, saved images every 40 ms, and predicted latents every 320 ms. These are three different sources of apparent large transitions.

Use these new defaults:

| Quantity | Default | Meaning |
|---|---:|---|
| RK4 integration step | 0.002 s | Numerical evolution of the continuous state |
| Recorded state/image interval | 0.010 s | Save true state and render RGB at 100 Hz |
| Force/controller update interval | 0.020 s | Select a force, then hold it for two recorded intervals |
| Frame spacing inside an eight-frame encoder history | 0.020 s | Use every second saved image; history spans 0.14 s |
| JEPA prediction interval | 0.100 s | Ten recorded intervals per latent transition |
| Maximum episode duration | 4.000 s | Up to 401 images/states and 400 recorded-interval forces |

One force is held constant over ten RK4 steps and two recorded intervals. One recorded interval contains five RK4 steps.

**Reducing only RK4's step does not make saved video smoother.** The recorded image interval must also decrease. Conversely, rendering more frequently does not correct an inaccurate integrator.

The predictor interval is separate again: it should be long enough to contain visible motion, while the saved source trajectory remains dense. Do not train only on immediately adjacent 10 ms images and interpret an easy persistence result as good dynamics.

Use integer counters rather than floating-point comparisons to decide when to render or update a force:

```text
physics_steps_per_record = 5
record_intervals_per_force = 2
history_frames = 8
history_frame_stride = 2
latent_stride_records = 10
```

Validate these ratios from the configured timesteps. No interpolation is needed for the default grid.

### Precise action convention

Save:

```text
rgb[k] and state[k] at t[k] = k * 0.01
force[k] acts over [t[k], t[k+1])
state[k+1] = RK4_advance(state[k], force[k], theta, duration=0.01)
```

For N recorded intervals there are **N+1 observations/states and N forces**. Duplicate a held control force into its two recorded-interval entries. Record the actual force after clipping. A boundary-truncated episode may end with an odd number of recorded intervals: allow one final unpaired force entry, without padding or discarding that valid prefix.

At a control boundary, compute the action from the current state before integrating the next interval. Do not associate the action with the state that it has already produced.

Generate in float64 under no-grad. Store true states/forces in float32 or float64, timestamps in float64, RGB in uint8. Training and differentiable fitting can remain float32.

## 3. Shared reset, visibility, and rendering rules

All episodes start from an actual simulator reset at `p0=0`. This preserves the existing commanded-reset coordinate grounding. Initial angle and velocities are sampled privately.

Do not artificially teleport, recenter, or clip the state during an episode. In particular:

- No clipping of cart position or velocity.
- No angle termination when the pole falls.
- No hidden state kicks after reset: deliberate disturbances must be applied as logged forces.
- No controller that catches or snaps the pole upright when LQR fails.
- No interpolated frames manufactured from a sparse old dataset.
- No concatenated resets masquerading as a continuous trajectory.

Use a fixed calibrated camera for both corpora. A view spanning approximately horizontal [-2.6, 2.6] m and vertical [-0.7, 0.7] m is suitable. With a cart boundary at |p|=2.0 m, the 0.5 m pole and its bob remain visible with a small margin. If the existing renderer uses another calibrated view, make the same geometric margin check.

Render the bob at:

```text
bob_x = p + ell*sin(q)
bob_y = -ell*cos(q)
```

Keep a visible track-origin mark. Make the cart, pole, and bob distinguishable; use a pole stroke of at least two final-image pixels and a bob of several pixels. Render at 2x resolution and downsample with antialiasing to 96x96 RGB. Keep the camera and geometry fixed.

For this next collection, use one clear, fixed color/background scheme by default. This reduces nuisance variation while diagnosing motion learning. An optional appearance-randomization flag may reuse the existing independent per-episode color randomization; keep it off in the default manifests and never correlate appearance with parameters or collection family. No random crops, flips, camera shake, occlusion, or coordinate-changing augmentations.

### Boundary handling

Stop an episode at the first recorded state with |p|>2.0 m, or if a state is nonfinite. A boundary exit is censoring by the observation region, not a bounce or an extra physical force.

Retain finite prefixes long enough for a complete training window. Store their true length and termination reason. A controller failure that leaves the pole falling is useful data if the cart remains visible.

For training/validation, require at least 65 observations (0.64 s). If an attempt exits sooner, regenerate within the same planned collection family; log the discarded attempt. Do not reject an otherwise valid trajectory because it is difficult, unstable, or does not balance.

For held-out query evaluation, retain the attempted episode and report early boundary exits; do not repeatedly sample until a convenient long query is obtained. Score only horizons with available targets and report the number of valid queries at each horizon.

## 4. Dataset A: passive, exactly zero force

### 4.1 Start with one fixed apparatus

Default:

```text
M = 1.0 kg
b = 0.25 N*s/m
u[k] = 0.0 exactly, for every interval
```

This is deliberately the clean version for a simple, fully unconditioned predictor:

```text
z_t = encoder(causal_visual_history_t)
z_next_hat = passive_predictor(z_t)
```

Do not pass forces, parameters, reset family, trajectory ID, or episode time into this predictor.

The visual history, rather than one still image, supplies information about velocity. Zero applied force does not imply zero initial motion.

Do not silently vary M,b in this corpus while keeping a fully unconditioned short-history predictor. Different apparatuses can have different futures from the same instantaneous state. A later parameter-varying passive dataset is possible, but would require explicit parameter conditioning or an identification context; that is not the default here.

The passive corpus is a test of autonomous world-model prediction. It is not a balancing benchmark: without force, a perturbed upright pole falls. It also does not establish test-time identification of unfamiliar parameters.

### 4.2 Sample nontrivial initial conditions

Use these three families:

| Family | Fraction | q0 | w0 | v0 |
|---|---:|---|---|---|
| Downward oscillations | 50% | Uniform(-1.2, 1.2) rad | Uniform(-2.0, 2.0) rad/s | Uniform(-0.25, 0.25) m/s |
| Large nonlinear excursions | 33.3% | Random sign times Uniform(1.2, 2.6) rad | Uniform(-2.0, 2.0) rad/s | Uniform(-0.25, 0.25) m/s |
| Falls from upright | 16.7% | pi + random sign times Uniform(0.05, 0.20) rad | Uniform(-0.5, 0.5) rad/s | Uniform(-0.25, 0.25) m/s |

Set p0=0 for all three. Initial cart velocity is allowed; it is set at reset, not imposed by an unlogged force later.

For downward resets only, reject an almost stationary reset when **all three** hold:

```text
abs(q0) < 0.10
abs(w0) < 0.20
abs(v0) < 0.05
```

This avoids filling the corpus with nearly identical equilibrium videos. Do not reject individual slowly moving frames later; turning points belong to the dynamics.

Four-second episodes capture several phases of oscillation and the nonlinear fall. Do not extend them indefinitely until the dataset is dominated by the final quiet state.

### 4.3 Counts and splits

Use independent trajectory-level seeds:

```text
train:      1,536 episodes = 768 downward + 512 nonlinear + 256 upright falls
validation:    96 episodes =  48 downward +  32 nonlinear +  16 upright falls
test:          96 episodes =  48 downward +  32 nonlinear +  16 upright falls
```

All use the nominal apparatus, but different initial conditions and episode seeds. These are independent episodes, not overlapping windows from one long recording.

Save forces as zeros to keep the common file format, but the passive model's loader/interface must not provide them as conditioning.

### 4.4 Minimal passive predictor interface

If implementing the model interface alongside the loader, a readable default is:

```text
MLP: latent_dim -> 128 -> 128 -> latent_dim
activation: GELU after the first two layers
output: unnormalized next latent
```

Keep the existing causal clip encoder and LeJEPA prediction-plus-SIGReg recipe. Predict the next encoded history, without detaching the target or introducing an EMA teacher. This is a requested simple action-free predictor, not an attempt to reproduce LeWorldModel's transformer.

Do not automatically add a passive physical-readout training run. In a zero-input system, a readout that always predicts an equilibrium can satisfy a physics consistency loss despite visible motion. Ordinary JEPA noncollapse does not alone ground that readout.

## 5. Dataset B: controlled video-action-state transitions

### 5.1 What “action-state dataset” means here

Every episode has aligned video, actual applied forces, and true simulator states. The state transitions are available for inspection and evaluation:

```text
(state[k], force[k], state[k+1])
(rgb[k], force[k], rgb[k+1])
```

Keep true states in a separate truth file. The primary JEPA still learns from video and forces, and the physics loss still uses known equations/training parameters rather than true-state supervision. If a supervised state model is wanted later, that is an explicit additional experiment.

### 5.2 Parameter splits

Retain the existing training parameter ranges:

```text
64 training apparatuses:
    M ~ Uniform(0.7, 1.3)
    b ~ Uniform(0.05, 0.5)
    24 episodes per apparatus

8 independent validation apparatuses:
    same parameter ranges
    12 episodes per apparatus
```

Draw M and b independently. Keep them constant within an apparatus and across its episodes. Sample appearance, reset, and force-program seeds independently of M,b.

Reuse the existing fixed 12 held-out apparatus pairs if they are already recorded in config. Otherwise use this explicit list:

```text
interior:
    (0.75, 0.10), (0.85, 0.40), (1.00, 0.20),
    (1.10, 0.45), (1.20, 0.15), (1.25, 0.35)
extrapolation:
    (0.50, 0.25), (1.60, 0.25), (1.00, 0.02),
    (1.00, 0.80), (0.50, 0.02), (1.60, 0.80)
```

Test parameter pairs belong only to the private simulator/evaluator. Training and validation parameter pairs are allowed predictor/solver inputs, as in the original experiment. The encoder does not receive parameters.

### 5.3 Do we need an expert?

**No trained RL expert is required.** Use a small analytic LQR controller solely to keep part of the collection near upright. Combine it with open-loop exploration so the data also contain responses to actions chosen independently of state.

Purely unforced/free-fall or random-force collection will not keep an unstable pole upright for long. Conversely, a quiet balancing controller provides narrow state coverage and an almost deterministic relationship between state and force. The mixture below addresses both limitations.

### 5.4 Exact mixture per training apparatus

| Family | Episodes per apparatus | Purpose |
|---|---:|---|
| Open-loop held pulses | 8 | Varied force responses and departures from equilibrium |
| Open-loop smooth multisines | 4 | Sustained excitation over several time scales |
| Noisy LQR with varied references | 8 | Upright motion, balancing, and ordinary recovery |
| LQR with a brief controller release | 4 | Larger recovery/fall transitions |

Total: 24 episodes per apparatus, 1,536 training episodes.

For each validation apparatus use half those counts: 4,2,4,2, totaling 12.

Do not equate the number of episodes with the number of transitions; boundary-truncated episodes have fewer transitions. Report both.

### 5.5 Open-loop reset distributions

Across the 12 open-loop episodes per training apparatus use:

```text
6 downward:
    q0 ~ Uniform(-1.0, 1.0)
    w0 ~ Uniform(-1.5, 1.5)
    v0 ~ Uniform(-0.25, 0.25)

4 nonlinear:
    q0 = random sign * Uniform(1.2, 2.6)
    w0 ~ Uniform(-2.0, 2.0)
    v0 ~ Uniform(-0.25, 0.25)

2 near upright:
    q0 = pi + Uniform(-0.15, 0.15)
    w0 ~ Uniform(-0.5, 0.5)
    v0 ~ Uniform(-0.25, 0.25)
```

Set p0=0. Allocate the reset families across both pulse and multisine episodes; do not make a particular reset uniquely identify a force waveform.

### 5.6 Open-loop force pulses

Choose the entire waveform before simulating the episode. No state feedback or emergency corrective action is permitted in this family.

Use:

```text
control update grid: 0.02 s
individual pulse duration: 5..15 updates = 0.10..0.30 s
episode force scale: Uniform(0.5, 3.0) N
pulse signs: independently sampled positive/negative
20% probability of a zero-force dwell
force magnitude within a nonzero dwell: Uniform(0.3, 1.0) * episode force scale
actual force bound: [-5, 5] N
```

Sample durations as integers on the control grid. To reduce unbounded cart drift without introducing feedback, make half of these pulse episodes from opposite-sign pulse pairs:

1. Sample one duration and magnitude.
2. Apply +a and -a for the same duration, in a random order.
3. Repeat with new durations/magnitudes; optionally insert zero-force dwells.
4. Trim the final program to the fixed episode length.

The other half use independent pulses. Balanced impulse does not guarantee zero cart displacement; boundary exits remain possible and should be logged.

Do not resample a force at every 2 ms RK4 substep. Rapid independent sign changes can average away before creating informative motion.

### 5.7 Open-loop smooth excitation

Create a smooth waveform on the 20 ms action grid:

```text
raw_u(t) = sum_j a_j * sin(2*pi*f_j*t + phi_j), j=1..3

f_1 ~ Uniform(0.30, 0.60) Hz
f_2 ~ Uniform(0.65, 1.00) Hz
f_3 ~ Uniform(1.20, 1.70) Hz
a_j ~ Uniform(0.3, 1.0)
phi_j ~ Uniform(0, 2*pi)
```

Subtract the waveform's discrete mean over the planned episode and rescale it to a peak magnitude drawn uniformly from 0.8..2.5 N. Hold each sampled value constant until the next 20 ms control update. This preserves the numerical action convention; do not use a continuously varying force inside RK4 while logging only one value.

This frequency range spans slower motion and the characteristic oscillatory scale of the chosen pendulum; it is a proposed collection default, not a proven optimal identification design.

### 5.8 Reuse a small force-program library

Create 32 pulse programs and 32 multisine programs for training, using independent waveform seeds. Reuse each program across different parameter pairs and several fresh initial conditions.

This supplies examples of different systems responding to the same intervention. It also makes it harder to predict from a unique action-program identifier. IDs are metadata only.

Validation and test use independently generated programs with the same amplitude/frequency/dwell distributions. Never reuse the exact stored training program there.

Do not multiply the force by hidden M or b to normalize the true acceleration. The applied force is a calibrated intervention in newtons.

### 5.9 Nominal LQR controller, with explicit signs

Use the **same nominal design parameters** M_nom=1.0, b_nom=0.25 for every training and validation apparatus. This reduces direct encoding of the hidden parameter pair into the controller gains.

For alpha = circular(q-pi), state error e=[p-p_ref, v-v_ref, alpha, w], the upright linearization is:

```text
A = [[0, 1, 0, 0],
     [0, -b/M, m*g/M, 0],
     [0, 0, 0, 1],
     [0, -b/(M*ell), (M+m)*g/(M*ell), 0]]

B = [[0], [1/M], [0], [1/(M*ell)]]
Q = diag([10, 1, 100, 5])
R = [[0.5]]

K = continuous_time_LQR(A_nom, B_nom, Q, R)
u_feedback = -gain_scale * (K @ e)
u_applied = clip(u_feedback + excitation(t), -5, 5)
```

Solve the Riccati equation with SciPy once. Do not train a policy. For these constants, K should be approximately [-4.472, -6.230, 48.734, 10.541]; compute it rather than copying rounded values.

Update feedback at 50 Hz using the true simulator state. This state access is allowed **only for generating collection actions**. The encoder/predictor/training loader does not receive it.

Although using nominal gains reduces a parameter shortcut, future feedback actions can still reveal future hidden state. That is why held-out intervention prediction must use open-loop forces.

### 5.10 Noisy balancing and varied references

Use upright resets:

```text
p0 = 0
q0 = pi + Uniform(-0.15, 0.15)
v0 ~ Uniform(-0.25, 0.25)
w0 ~ Uniform(-0.5, 0.5)
gain_scale ~ Uniform(0.85, 1.15), fixed within the episode
```

For half the episodes use p_ref=v_ref=0. For the other half use a slow reference:

```text
A_ref ~ Uniform(0.10, 0.35) m
f_ref ~ Uniform(0.10, 0.25) Hz
phi_ref ~ Uniform(0, 2*pi)
p_ref(t) = A_ref * sin(2*pi*f_ref*t + phi_ref)
v_ref(t) = A_ref * 2*pi*f_ref * cos(2*pi*f_ref*t + phi_ref)
```

Add a presampled multisine or held-pulse excitation with peak magnitude drawn from 0.6..1.5 N. Choose excitation, reference, and gain-scale seeds independently of apparatus parameters.

Log the total applied force after combining feedback and excitation and clipping. The predictor must not be given only the excitation while the simulator also applies hidden feedback.

Do not discard episodes if the controller fails. Falling and saturated recovery trajectories broaden coverage.

### 5.11 Controller release

Use the same upright resets and nominal feedback. Choose a release start uniformly from 1.2..2.0 s and a release duration from 0.20..0.40 s, rounded to the control grid.

During the release, set the feedback contribution to zero but keep the presampled excitation. Afterwards re-enable the same controller. Do not reset the state or replace the failed recovery with an upright state.

Store every applied force. Controller mode/reference are diagnostic metadata and never model inputs.

## 6. Split and save the data before creating windows

Split by complete episode. For the controlled corpus also split by apparatus as specified above. All windows from an episode remain in the same split.

Suggested minimal layout:

```text
data/passive/
    manifest.json
    train/trajectory_000000.npz
    validation/...
    test/...
    truth/train/trajectory_000000.npz
    truth/validation/...
    truth/test/...

data/controlled/
    manifest.json
    train/...
    validation/...
    calibration/...
    query/...
    truth/...
```

Learning file fields:

```text
rgb: uint8 [N+1, 96, 96, 3]
force: float [N, 1], actual applied N
t: float64 [N+1]
theta: [M,b] for train/validation and fixed passive apparatus only
p0_commanded: 0.0
reset_mode: downward / nonlinear / upright, coarse metadata
episode_id, apparatus_id, waveform_id: metadata, not neural inputs
collection_family, valid_length, termination_reason
```

Truth file fields:

```text
state: [N+1, 4]
theta_true: [M,b]
exact_initial_state: [4]
```

Keep true test theta and test initial conditions exclusively in truth/private simulator configuration. The test-learning manifest must not expose the parameter pair through filenames or apparatus IDs. IDs may group calibration episodes from one apparatus but cannot be decoded into its parameters.

The manifest records units, angle convention, all clocks, camera mapping, seeds, attempted/retained counts, parameter split, collection settings, and format version. Change the format version so old 25 Hz data are not silently reused.

One compressed NPZ per episode is acceptable. At maximum length, uncompressed RGB alone is about 17 GB for a 1,536-episode 96x96 corpus; compression of these simple scenes reduces disk size, but not decoded RAM. Do not load both corpora into RAM at once or cast the entire corpus to float32. A simple bounded cache or loading only requested episodes is sufficient; no dataset framework is required.

## 7. New causal histories and JEPA windows

Preserve the eight-frame, 24-channel encoder interface, but sample its images every second recorded frame.

For a latent endpoint e, the history is:

```python
history_indices = [e-14, e-12, e-10, e-8, e-6, e-4, e-2, e]
```

Require e>=14. Use endpoint strides of 10 recorded intervals:

```text
example six endpoints relative to a window start:
    [14, 24, 34, 44, 54, 64]

history span:       0.14 s
latent transition:  0.10 s
whole six-token window: 0.64 s of recorded data
```

This is an explicit change from the previous nonoverlapping eight-frame clip recipe. Consecutive histories share three sampled images, rather than seven of eight. The partial overlap retains a useful velocity context while shortening the prediction interval. It is not future leakage: every input history ends at its own endpoint.

Do not claim that overlap cannot create a shortcut. Report improvement over latent persistence and physical-state/readout diagnostics. Longer autoregressive forecasts progressively require predicting information absent from the initial context.

For the controlled predictor:

```python
next_endpoint = e + 10
action_block = force[e:next_endpoint]  # exactly ten recorded-interval forces
```

Supply **all ten forces**, even though adjacent entries often repeat. Do not supply one average force, just the first/last force, or five control values without an explicit changed interface.

Update the conditioning input dimension from eight forces plus two parameters to ten forces plus two parameters. Keep force/parameter normalization conventions from the original implementation.

For the passive predictor, do not create an action-conditioning path.

Sample training windows with uniformly chosen valid start indices within sampled episodes. Select independent episodes uniformly, then one window per episode; do not treat strongly overlapping windows as independent SIGReg samples.

Keep a six-endpoint window and the original at-most-three-token transformer context for the controlled model. Compute SIGReg across independent episodes separately at each endpoint offset. Keep the exact loss recipe, optimizer, and anti-collapse conventions; this dataset change is not an implicit replacement of SIGReg.

For an autoregressive evaluation, encode the first three observed histories, then supply only predicted latents and the correct subsequent force blocks. Reuse no future observed latents after the forecast starts.

### Physics-loss compatibility

The integrator used inside the physics objective and adaptation must respect the new action grid and 2 ms substeps. A force discontinuity is an interval boundary: never let one RK4 step straddle it.

Retaining the autodecoded initial-condition table requires care:

- The table's initial state belongs to the actual episode reset at raw frame zero.
- A cropped training window does not command a new p0=0 reset.
- Carry the episode ID and window's raw start/end indices as optimizer/solver metadata, not neural inputs.
- Integrate from the episode's fitted initial condition to the sampled endpoints if retaining the original objective.
- Do not independently relabel each crop as a centered reset.
- Expand the old q0 restriction: the new nonlinear resets cannot be represented by q_base+0.8*tanh(a). Use q_base+pi*tanh(a), with q_base=pi for upright and zero otherwise, or an equivalent full-circle initial-angle parameterization. Initialize from coarse reset metadata, not truth.
- Initial v0,w0 bounds of 1 m/s and 3 rad/s contain the specified training-reset support. States fitted later in a trajectory need wider/unbounded velocity ranges.

Longer episodes and smaller solver steps increase differentiable rollout cost. Do not render frames inside the loss, compare physics at every saved timestamp unnecessarily, or integrate separately for each sampled endpoint. Integrate once per sampled episode and gather endpoints. Do not silently increase the original model-training budget to compensate.

This addendum does not introduce multiple shooting, a new trajectory objective, or hidden-state supervision. If a later change is needed to reduce physics-training cost, make it explicit.

## 8. Calibration and held-out prediction

### Passive test

Use fresh nominal-parameter episodes from all three passive reset families. Measure predictions at 0.1,0.2,0.4,0.8,1.6,3.2 s after observed context where targets are available.

Report latent persistence alongside the learned predictor. The relevant claim is prediction of autonomous motion, not use of interventions or parameter adaptation.

### Controlled calibration

For each test apparatus generate two independent three-second downward episodes:

```text
p0=0
q0 ~ Uniform(-0.6,0.6)
v0 ~ Uniform(-0.15,0.15)
w0 ~ Uniform(-0.5,0.5)
```

Use independent prescribed open-loop multisine force programs, generated before simulation, with peak magnitude 1.0..2.0 N and the frequency bands above.

Keep networks frozen during subsequent parameter fitting. The first valid eight-frame history ends at raw frame 14, at 0.14 s. Fit from that observed endpoint, with an unknown state there. **The cart position at frame 14 is not known to be zero.**

### Controlled query episodes

Per test apparatus generate eight fresh four-second queries:

```text
4 pulse programs, 4 multisine programs
reset allocation: 4 downward, 2 nonlinear, 2 near upright
```

All query waveforms are fixed before the query simulation. No LQR, adaptive force amplitude, rescue feedback, or hidden impulses in this primary forecast set.

Do not feed future actions generated by a true-state feedback policy to the primary predictor evaluation: those actions can reveal future state. A separate feedback-distribution diagnostic can be reported if desired, but label it separately.

Calibration and queries share only apparatus parameters, never initial states, force programs, images, or seeds. Query outcomes cannot select calibration fits or checkpoints.

Use the same saved queries across joint/post-hoc models and nominal/fitted/oracle parameter conditions. Report one-step and autoregressive latent errors, persistence comparisons, and action-shuffling/zero-action diagnostics where already supported.

For action diagnostics, shuffle complete action programs between held-out queries, preserving temporal blocks. A correct-action advantage is informative, but interpret it together with absolute prediction error and state-content diagnostics.

Use horizons 0.1,0.2,0.4,0.8,1.6,3.2 s. Report valid sample counts and boundary-exit rates at each horizon. Report decoded prediction error separately from observed-frame readout error.

The passive and controlled corpora are different prediction problems; their raw latent MSEs are not an apples-to-apples ablation. Do not conclude that actions or physics help merely because one corpus has a lower latent loss.

## 9. Cheap QA and collection statistics

Do these during generation without training preliminary models:

1. Confirm N+1 observations versus N forces, uniform 10 ms record timestamps, and equal force entries in every complete pair on the 20 ms control grid. Allow a final unpaired entry in a truncated episode.
2. Confirm all passive forces are exactly zero.
3. For a few states, compare a recorded interval produced by five 2 ms RK4 steps with a 1 ms reference. In float64, errors should be small relative to the plotted physical scales. If not, fix a genuine simulator/sign/indexing bug; do not silently resample the experiment.
4. Under b=0,u=0, check approximately conserved total mechanical energy for a short numerical trajectory. With positive b,u=0, energy should decrease up to numerical error. Energy is:
   `0.5*(M+m)*v*v + m*ell*v*w*cos(q) + 0.5*m*ell*ell*w*w - m*g*ell*cos(q)`.
5. Compute distributions of p,v,q,w, applied force, controller saturation, episode length, and boundary exits, by collection family. Angle coverage uses circular bins.
6. Compute 50th/95th/99th percentile cart and bob displacements in final-image pixels between saved frames. Report physical increments too. Large increments from high velocity are real motion; they are different from an integrator failure.
7. Report frame-change statistics and the fraction of almost identical successive frames. This is diagnostic, not a requirement to remove equilibrium/turning-point frames.
8. Check that encoder/action indexing matches the explicit history and force slices above; no future observed frame enters a forecast input.
9. Save two representative clips per family and a few angle/cart/force time-series plots. Include one clipped-force/failed-recovery or boundary-truncated example when available.

For human preview, play saved frames at their actual rate of 100 frames/s. If a preview is limited to 25 frames/s, explicitly subsample every fourth saved frame rather than playing all 100 frames/s data at quarter speed. Include a labeled slow-motion preview only as an additional view.

A small numerical check of the proposed dynamics/timing found roughly 0.1 rad or smaller recorded angle increments during falling trajectories and successful noisy nominal-LQR collection across sampled training parameters. Independent pulses sometimes exited the camera region. This motivates dense recording and explicit truncation handling; it does not establish JEPA learning or prove dataset sufficiency.

## 10. Minimal implementation deliverables

Extend the existing small files rather than introducing a collection framework:

```text
physics.py: keep EOM; expose integration/record/action clocks
data.py: reset samplers, force programs, nominal controller, generator, loaders
config.json: passive/controlled options and explicit timing
models.py: passive MLP option; controlled ten-force conditioning dimension
evaluate.py: updated time/index conventions and open-loop query generation
README.md: formats, counts, clocks, commands, weak grounding, actual QA outcome
```

Provide straightforward commands such as:

```bash
python data.py --config config.json --dataset passive
python data.py --config config.json --dataset controlled
python data.py --config config.json --dataset both
```

Adapt argument names to the existing CLI. Use one base seed 42 with independent deterministic streams for apparatus parameters, resets, force programs, and splits. Record seeds in manifests; do not require elaborate bitwise-determinism infrastructure.

Finish by reporting actual retained episode/transition counts, saved-frame rate, numerical checks, family coverage, boundary exits, and example outputs. Do not describe a generated corpus as “working” until a model has actually learned useful dynamics on it. Do not launch new neural training runs unless the surrounding instruction requests them.

## 11. Primary-source rationale and status of the defaults

These are proposed research defaults for our simulator, not a published standard CartPole-JEPA dataset.

- MIT Underactuated Robotics, cart-pole equations and upright LQR: <https://underactuated.mit.edu/acrobot.html>. Our added cart drag is explicit in the stated EOM.
- DINO-WM, ICML 2025, Appendix A.9: <https://arxiv.org/html/2411.04983v2>. It separates dense source trajectories from a frame-skipped prediction interval because neighboring observations can be too similar. We use the same principle, with independently chosen intervals suited to our simulator.
- LeWorldModel, Sections 3.1 and limitations: <https://arxiv.org/html/2603.19312v3>. Offline collection can use exploratory or pseudo-expert behavior without optimality requirements; sufficient dynamics coverage matters. It also notes limitations of Gaussian latent regularization in low-diversity, low-dimensional settings. Better data alone therefore should not be assumed to resolve every learning failure.

Keep these distinctions explicit when documenting the implementation.
