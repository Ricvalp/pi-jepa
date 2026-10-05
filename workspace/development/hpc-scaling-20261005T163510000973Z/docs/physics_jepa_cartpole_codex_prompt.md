# Implement a physics-aligned JEPA with test-time system identification

Build an end-to-end, runnable exploratory research implementation of the experiment below. Prioritize simplicity, readable mathematics, short files, and a direct training loop. This is research code, not a production package.

The experiment asks whether shaping a visual encoder with known equations of motion during JEPA training improves physical-state inference and identification of an unseen apparatus from a few interactions.

## 1. Scope: exactly two trained variants

Implement only:

1. **Joint:** train the visual encoder, JEPA predictor, physical readout, and autodecoded initial conditions jointly with `L_JEPA + lambda_phys * L_phys`.
2. **Post-hoc:** train the same encoder and predictor with `L_JEPA`; freeze them completely; then fit the same physical readout and autodecoded initial conditions using `L_phys`.

Use one seed, identical data, matching initial encoder/predictor weights, equal JEPA update budgets, and equal physical-readout update budgets. The post-hoc variant necessarily has two stages; that is the requested comparison, not an additional experiment.

Go directly to these two variants. Do not add pendulum warmups, physics-only models, supervised state-estimation models, tracker baselines, seed sweeps, hyperparameter searches, curriculum experiments, or intermediate performance gates. A few implementation checks for shapes, gradients, and equation conventions are appropriate; separate pilot training runs are not prerequisites.

If results are poor, finish both variants and report what happened. Fix genuine implementation bugs, but do not silently change the method, add state supervision, invent auxiliary losses, or launch a new experiment suite to obtain a positive result.

Do not stop at a scaffold or a plan. Implement data generation, both training variants, test-time fitting, forecasting, balancing, plots, and clear commands. Run the two configured variants if the environment has an appropriate GPU and the surrounding task authorizes execution; otherwise leave fully runnable commands and clearly state what was actually executed. Do not initiate a remote/cloud compute job without authorization.

## 2. Coherent JEPA recipe and references

Use a scaled **LeWorldModel / LeJEPA** recipe. In particular, use next-latent prediction plus SIGReg, with gradients through both latent endpoints. Do not mix this recipe with an EMA teacher, detached latent targets, BYOL-style training, pixel reconstruction, or a generic covariance penalty renamed SIGReg.

Read the relevant small portions of these primary sources before implementing the JEPA components:

- LeWorldModel paper: <https://arxiv.org/html/2603.19312v1>
- Official LeWorldModel repository: <https://github.com/lucas-maes/le-wm>
- Reference architecture and SIGReg: <https://github.com/lucas-maes/le-wm/blob/main/module.py>
- Reference training: <https://github.com/lucas-maes/le-wm/blob/main/train.py>
- LeJEPA minimal implementation: <https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md>
- Cart-pole equations and angle convention: <https://underactuated.mit.edu/acrobot.html>

Inspect the current reference implementations; record the commit/version used where practical. Reuse the small SIGReg implementation or its official package with appropriate attribution/license handling. Do not import an entire training/planning framework to obtain a few functions.

The dimensions and learning rates below are deliberately scaled research defaults for this task, not claims of reproducing the published architecture or universally optimal hyperparameters.

## 3. Physical system

Use a custom continuous-force, point-mass cart-pole. The pole is a massless rod with a point mass at its end. Do not substitute the uniform-rod equations or unmodified Gym CartPole.

State, ordering, and convention:

```text
x = [p, v, q, w]
p: horizontal cart position [m]
v: cart velocity [m/s]
q: pole angle [rad], zero hanging downward, pi upright
w: angular velocity [rad/s]
u: applied horizontal force [N], positive to the right
```

Known constants:

```text
m = 0.2 kg    # pole point mass
ell = 0.5 m  # pivot-to-point-mass length
g = 9.81 m/s^2
```

Apparatus parameters are `theta = [M, b]`, where `M` is cart mass and `b` is viscous cart drag in N*s/m. Only these two parameters vary. Geometry, gravity, pole mass, force calibration, and camera are fixed. Mass and drag must not be revealed by color or visible dimensions.

Equations:

```text
D = M + m * sin(q)**2
p_ddot = (u - b*v + m*sin(q)*(ell*w*w + g*cos(q))) / D
q_ddot = -(g*sin(q) + cos(q)*p_ddot) / ell
dx_dt = [v, p_ddot, w, q_ddot]
```

Equivalently, the acceleration mass matrix is

```text
[[M+m,        m*ell*cos(q)],
 [m*ell*cos(q), m*ell**2   ]]
```

with right-hand side `[u - b*v + m*ell*w*w*sin(q), -m*g*ell*sin(q)]`.

Implement one batched PyTorch right-hand side and a readable fixed-step RK4 function. Use them for both generation and differentiable fitting, so numerical-model mismatch is not an initial confound. An action is held constant over a video interval of `dt = 0.04 s`; use two RK4 substeps of `0.02 s` per interval. The API should handle a batch of states, a batch of parameter pairs, and a batch of scalar forces.

Generate under `torch.no_grad()` on CPU, optionally in float64. Train and fit in float32. Do not detach the RK4 graph in training or adaptation, and do not use SciPy integration inside a differentiable loss.

## 4. Data: one modest dataset

Default seed: `42`. Keep generation deterministic under that seed without elaborate reproducibility infrastructure.

Use:

- 64 training apparatuses, each with `M ~ Uniform(0.7, 1.3)` and `b ~ Uniform(0.05, 0.5)`.
- 24 short trajectories per training apparatus.
- 8 fresh validation apparatuses in the training range, with 6 trajectories each.
- 12 test apparatuses: 6 fresh interior parameter pairs and 6 modest extrapolation pairs. Include `(0.5, 0.02)`, `(1.6, 0.8)`, and pairs that change one parameter at a time. Fix these pairs in the config; do not choose them from observed model results.

Training trajectories contain 48 frames, indexed `0..47`, at 25 Hz, and 47 forces; their duration is approximately two seconds. Start a new trajectory rather than chopping an arbitrary long trajectory and claiming that its initial cart position is zero.

All resets command `p0 = 0`. Randomize the other initial components and withhold their exact values from learning. Half of training trajectories begin near downward and half near upright. Suitable initial ranges are:

```text
downward: q0 in [-0.5, 0.5], v0 in [-0.3, 0.3], w0 in [-0.5, 0.5]
upright:  q0 in pi + [-0.06, 0.06], v0 in [-0.15, 0.15], w0 in [-0.15, 0.15]
```

The commanded centered reset and coarse reset mode (downward/upright) are allowed metadata. They fix the cart-coordinate origin and initialize a physical prior. Actual randomized angles and velocities are hidden. State this weak grounding explicitly in the README; this is not a claim of completely unanchored state discovery.

For downward data, apply bounded mixtures of sinusoids and force pulses. Reuse some force programs across different apparatuses and initial conditions. For upright data, generate with a true-state LQR controller plus small force excitation. True-state controller access is solely a data-generation mechanism. Record the actual applied force after clipping, not the unclipped desired action. The learning model receives no controller state, reward, or true state.

Use simple independent color/background randomization per trajectory. Keep appearance constant within a trajectory. Start with a fixed calibrated camera and no geometric augmentations, camera jitter, occlusion, or elaborate domain randomization.

Render antialiased RGB at `96 x 96` using Pillow or an equally simple renderer. A fixed view spanning approximately `p in [-2.5, 2.5]` and vertical position `[-0.75, 0.75]` is adequate; fixed different horizontal/vertical pixel scales are acceptable. Render the bob at `(p + ell*sin(q), -ell*cos(q))`. Make the pole/bob readable and include a visible fixed track origin. Do not overlay numeric state/parameter labels. Reject and regenerate a clip that leaves the view rather than hiding the exit with a moving camera.

Save compact NumPy files per apparatus/trajectory or a straightforward equivalent. Avoid a database or streaming framework. Store:

```text
learning data: RGB frames, actual forces, timestamps/dt, known training theta,
               trajectory ID, p0=0, coarse reset mode
evaluation data, separately: true states and true test theta
```

Training/readout-fitting loaders must never load the hidden state arrays. Test parameter fitting must never receive true test parameters. Evaluation can load truth only after producing estimates/predictions. IDs index the initial-condition table; never feed IDs or absolute episode time into the encoder/readout.

## 5. Architecture

### Visual-only causal clip encoder

`z_t = E(o[t-7:t+1])`. It consumes eight consecutive RGB frames ending at time `t`. Never use future frames. Do not give this encoder actions, physical parameters, reset mode, or trajectory ID; actions enter the predictor and physical solver.

For a simple literature-informed backbone, use an unpretrained ResNet-18 with these explicit task adaptations:

- Stack the eight frames chronologically into 24 input channels.
- Replace the input stem with a `3 x 3`, stride-2 convolution for 24 channels, and remove the initial max pool.
- Retain a small spatial grid with adaptive `3 x 3` pooling, flatten it, and map it to 192 features. Do not rely solely on global average pooling for cart-position information.
- Project to `z in R^32` with `Linear(192,256) -> BatchNorm1d(256) -> GELU -> Linear(256,32)`.
- No final LayerNorm, L2 normalization, or sigmoid on `z`.

Use torchvision's ResNet implementation if available; do not build a generalized backbone registry. Channel stacking is the deliberately simple causal-video adaptation. The same complete encoder is used in both variants.

### JEPA predictor

Use a small causal temporal transformer: hidden width 192, 3 attention heads, 3 blocks, feed-forward width 768, dropout 0.1. Project latent inputs into this width and add temporal position embeddings. Keep a maximum context of three latent tokens.

At a prediction step the conditioning vector contains **all eight intervening forces** plus normalized `M,b`. Use fixed normalization `u/5`, `(M-1.0)/0.3`, and `(b-0.275)/0.225`; retain these same scales at test time. Use a small conditioning MLP and token-wise AdaLN-zero in the transformer blocks, following the reference pattern. Zero-initialize its final modulation layer. Use an unnormalized latent output head, with a projection after any final transformer LayerNorm. Keep implementation explicit rather than importing a diffusion model or a planning framework.

### Physical readout

Use one small MLP `32 -> 64 -> 5`, with GELU, to predict

```text
[p_hat, v_hat, sin_q_hat, cos_q_hat, w_hat].
```

Normalize the angular pair with a small epsilon. Convert it to `q_hat = atan2(sin_q_hat, cos_q_hat)` when a four-component physical state is needed. Do not use a raw angle MSE across the periodic boundary.

No image decoder, reward head, parameter-inference network, or supervised readout warmup is required.

## 6. JEPA objective and temporal indexing

For each 48-frame trajectory, form six nonoverlapping causal clips ending at raw frame indices

```text
[7, 15, 23, 31, 39, 47].
```

Their latent timestep is `8 * dt = 0.32 s`. Nonoverlapping clips prevent next-latent prediction from being dominated by seven identical frames. Eight-frame clips still provide the temporal information needed to infer velocities.

To predict from an endpoint `e` to `e+8`, supply forces `u[e:e+8]`. In general, `u[k]` advances state/frame `k` to `k+1`. Preserve this convention in every loader, solver, forecast, and controller.

Teacher-force the five next-latent transitions with a causal, at-most-three-token predictor. A token may see its own intervening force block and known apparatus parameters, but no future observed latent.

For each prediction, explicitly pass only its last one, two, or three latent tokens and use a triangular causal mask within that window. Reset positional indices to `0..window_length-1` in both training and autoregressive evaluation. A five-iteration training loop is fine. Do not assume a full-sequence triangular mask enforces this context limit; even a banded mask can transmit older information indirectly through multiple transformer layers.

```text
L_pred = mean((predicted_future_z - encoded_future_z)**2)
L_JEPA = L_pred + 0.1 * SIGReg(encoded_z)
```

Do not detach `encoded_future_z`. Do not add an EMA encoder. Both endpoints use the same online encoder. Normalize input RGB to a fixed documented range; do not apply spatial crops/flips that change physical coordinates without accounting for them.

Use the reference SIGReg/Epps-Pulley convention: 1024 normalized random projection directions and 17 integration points on `[0,3]`. For latents shaped `(B,6,32)`, compute the statistic across independent trajectories `B` separately at each of the six offsets, then average offsets. Preserve the reference batch-size multiplier and quadrature normalization. Do not flatten temporally correlated codes and silently redefine the statistical sample count.

Run SIGReg in float32. Apply it to learned latents, not to the decoded physical coordinates, which are not supposed to have a Gaussian distribution.

## 7. Physics objective: use the proposed joint method directly

Create one trainable initial-condition entry per training trajectory. Its physical state is `[0, v0, q0, w0]`, with unknown `v0,q0,w0`. Initialize velocities to zero and angle to the commanded coarse reset mode, never to the hidden sampled state. Simple bounded transforms are acceptable, e.g. `v0 = tanh(a)`, `w0 = 3*tanh(c)`, and `q0 = reset_mode_angle + 0.8*tanh(d)`.

Known training `M,b` are fixed inputs to the solver; do not optimize them during training.

For each sampled trajectory, integrate from its learned initial condition under its recorded force sequence. Compare the simulated state at the six encoder endpoints with the readout of their latent codes. Define

```text
iota(x) = [p/2, v/2, sin(q), cos(q), w/5]
iota_hat = [p_hat/2, v_hat/2, normalized_sin_hat,
            normalized_cos_hat, w_hat/5]

L_phys = mean((iota_hat - iota(simulated_x))**2)
```

The mean is over trajectories, endpoints, and the five components. The scales are fixed physical units, not hidden-state-derived statistics.

Joint loss:

```text
L_total = L_JEPA + 1.0 * L_phys
```

Use this physics term from the first update. This is one joint optimization, not a bilevel procedure or alternating trajectory-fitting curriculum. `L_phys` must backpropagate through the physical readout into the visual encoder, and through RK4 into the autodecoded initial conditions. Do not detach either side of this term in the joint variant.

The short training clips keep this initial implementation free of multiple-shooting infrastructure. Do not independently optimize a state at every frame. If a real numerical bug prevents gradients, fix that bug; do not replace the objective with supervised states.

This objective alone does not prove that the readout is physically identified. That is what the held-out evaluation examines. Keep latent/physical-readout variance in the logs, but do not add new anti-shortcut losses or gate experiments.

## 8. Optimization and the two runs

Use single-device PyTorch and ordinary training loops. Start in float32 for clarity. Mixed precision is optional for vision if actually needed, but physics and SIGReg remain float32.

Fixed defaults:

```text
seed = 42
batch_size = 64 independent trajectories
JEPA updates = 10_000 per variant
physical-readout updates = 10_000 per variant
encoder/predictor/readout learning rate = 3e-4
initial-condition learning rate = 1e-2
AdamW weight decay = 0.05 for encoder/predictor matrix/kernel weights
weight decay = 0 for biases, normalization parameters, readout, initial conditions
warmup = first 500 model updates
cosine decay = model LR down to 3e-5
gradient norm clipping = 1.0 for neural networks
```

Use AdamW parameter groups for the networks and a small separate Adam optimizer for the initial-condition table. Both optimizers step from the **same backward pass** in joint training. The initial-condition LR can remain fixed. No gradient clipping is needed for the tiny table unless a concrete numerical issue appears.

Save the initial encoder/predictor state once and reload it for both variants. Initialize the physical readout and initial-condition table identically in the two physical-training stages. Use separate deterministic generators for data sampling and SIGReg directions so the paired runs can use identical minibatch schedules. Do not demand bitwise deterministic GPU kernels.

**Joint:** 10,000 updates of encoder, predictor, readout, and initial-condition table with `L_total`.

**Post-hoc stage 1:** 10,000 updates of encoder and predictor with `L_JEPA`.

**Post-hoc stage 2:** freeze encoder and predictor, including temporal handling, projection heads, BatchNorm running means/variances, and all other buffers. Set them to `eval()`. Fit only a fresh physical readout and initial-condition table for 10,000 updates with `L_phys`. Cache encoder outputs to make this stage cheap. Never accidentally call `encoder.train()` while fitting the readout.

Use the final checkpoints for the main comparison. Validation losses are diagnostics, not triggers for extra sweeps or repeated runs. Save a resumable checkpoint, CSV losses, and enough config metadata to reproduce the run. Avoid WandB dependencies, Lightning, Hydra, distributed training, containers, and elaborate callback abstractions.

## 9. Test-time parameter fitting

For each unseen test apparatus, collect exactly two three-second calibration trajectories near hanging-down, with the same force programs and initial-state distributions for both variants. Commands can be mixtures such as

```text
u_j(t) = 1.6*sin(2*pi*0.65*t + 0.6*j)
         + 0.6*sin(2*pi*1.25*t + 0.8*j),  j in {0,1}.
```

Generate the exact initial state privately; the adaptation code must not receive it. Use centered resets, but fit after the first complete visual context: at frame 7 the cart is no longer necessarily centered.

Freeze encoder, readout, and predictor in `eval()` throughout fitting. Decode the visual states, under `no_grad()`, from causal sliding eight-frame contexts. For fitting, shift each clip's time origin to its first valid endpoint, frame 7. Cache the decoded states. Integrate from a learned four-component state at that endpoint, using forces starting at raw interval 7. Compare predictions with cached readouts at subsequent video timestamps.

Optimize one shared parameter pair per apparatus and one initial state per calibration clip:

```text
min_{M,b,c_0,c_1} mean((cached_iota_hat_j(t)
                       - iota(Phi_[M,b]^t(c_j; u_j)))**2)
```

Use a single fixed initialization and budget: nominal `M=1.0, b=0.25`, initial states from the first decoded endpoint, and 400 Adam updates at LR `0.03`. Bound `M` to `[0.3,2.0]` and `b` to `[0.001,1.2]` with a simple sigmoid parameterization. Initialize its raw coordinates to produce the nominal values. Initial states are four physical coordinates; a wrapped angle is fine because the residual uses sine/cosine.

Cache targets, but retain gradients through RK4 into the fitted parameters and initial states. Test fitting changes no neural-network weights. Batch the two calibration clips together. Batching apparatuses with separate parameter entries is optional if it makes the existing vectorized solver faster, not a reason to add a framework.

Use calibration observations/actions only. Query trajectories, true parameters, hidden states, and balancing outcomes may not enter fitting, initialization, stopping criteria, or checkpoint choice. Save fitted parameters and calibration losses even if optimization performs badly. Do not use a large multistart search.

## 10. Evaluate the nontrivial task

Evaluate only the two trained variants. Within each, use three **evaluation conditions**, not extra training baselines:

1. Nominal parameters `[1.0,0.25]`, without fitting.
2. Parameters fitted from the two calibration videos.
3. True test parameters supplied to the frozen forecasting/controller code as an oracle condition.

The third condition isolates parameter-fitting error from observation/predictor error. It must not influence any fitting or model selection.

### New-action forecasting

Generate fresh four-second query trajectories and force sequences, distinct from calibration trajectories. Reuse identical queries across variants and conditions. Include downward and near-upright trajectories and fresh pulse/sinusoid programs. Choose each entire query force program before simulating its trajectory, independently of hidden query states. These forecast queries use open-loop actions; do not reuse the true-state LQR data generator here, because its future forces could reveal future state through controller feedback.

Warm-start with three observed nonoverlapping latent clips ending at frames `[7,15,23]`. From that point, forecast autoregressively without further observed targets, supplying each block's actual eight forces and the relevant parameter pair.

Report horizons `0.32, 0.64, 1.28, 2.56 s` through both:

- The learned predictor: autoregressive latent rollout, then the frozen physical readout.
- The known physical solver: begin at the visually inferred state at the forecast start and integrate under the same forces.

For the learned rollout, maintain at most three latent context tokens. Recompute each token's conditioning from its own next action block and the apparatus parameters; do not retain conditioning from a different query or substitute observed future latents.

Score against hidden physical states only after predictions are saved. Report cart-position, cart-velocity, circular-angle, and angular-velocity errors separately. Report a normalized aggregate using the same physical scales as training. Encoder/readout state-recovery errors on observed query frames are also useful, especially for velocities.

### Upright balancing

Use a fixed LQR controller based on the relevant parameter pair and the visually inferred state. No RL training, swing-up controller, learned policy, or MPC implementation is required for this first experiment.

For state error `[p, v, delta, w]`, with `delta = atan2(sin(q-pi),cos(q-pi))`, use

```text
A = [[0, 1, 0, 0],
     [0, -b/M, m*g/M, 0],
     [0, 0, 0, 1],
     [0, -b/(M*ell), (M+m)*g/(M*ell), 0]]
B = [[0], [1/M], [0], [1/(M*ell)]]
Q = diag([10, 1, 100, 5])
R = [[0.5]]
```

Solve the continuous Riccati equation with SciPy, set `u = clip(-K @ inferred_state_error, -5, 5)`, and hold the force over each video interval. Networks consume a sliding causal eight-frame history at every control step. Simulation can wait for network inference; real-time wall-clock operation is not a requirement.

Start five paired balancing episodes per test apparatus at `p0=0`, `q0 in pi+[-0.02,0.02]`, `v0,w0 in [-0.04,0.04]`. Use zero force for the first seven intervals to acquire the initial visual context. The controller then receives estimated states only. Use the same privately generated initial states in both variants and parameter conditions.

Episode duration: 10 seconds, including context acquisition. Fail when `abs(p)>2.0 m` or `abs(delta)>0.35 rad`. Report balancing duration, success fraction, and cart/angle RMS error. Failure checks and scoring can use true simulator state; the controller cannot. These are five episodes from one trained seed, not five independently trained models.

Do not condition test control on autodecoded training initial states. Each new control episode starts from its own visual history.

## 11. Minimal outputs and code organization

Prefer a small layout along these lines; adapt names to an existing repository if appropriate:

```text
config.json
physics.py          # RHS, RK4, state/angle utilities, LQR
data.py             # generator, renderer, loading, clip/action indexing
models.py           # encoder, causal conditioned predictor, readout
losses.py           # SIGReg wrapper/reference + physics residual
train.py            # joint / jepa / readout modes
evaluate.py         # parameter fitting, forecasts, balancing
run_experiment.py   # generate once, train the two variants, evaluate
README.md
```

Use dataclasses or ordinary dictionaries, argparse, and a few explicit functions. Avoid extensible registries, abstract experiment interfaces, large configuration stacks, service layers, and unnecessary packaging. Respect existing repository instructions and preserve unrelated work.

Provide one command for the entire configured experiment and individual commands for generation, joint training, JEPA pretraining, frozen readout fitting, and evaluation. Resume should simply reload model/optimizer/table state; no elaborate job manager is needed.

Save:

- Final checkpoints, config, training/physical losses as CSV, and the frozen initial weights.
- A per-apparatus JSON/CSV of nominal/fitted/true parameters and parameter absolute errors.
- Forecast-error curves for the two methods and three parameter conditions, separately for learned and physical-solver forecasts.
- Observed-state recovery errors and balancing results, split into interior and extrapolation apparatuses.
- A handful of trajectory overlays with true and estimated physical states; optionally one short rendered balancing example per variant.
- A short `results.md` stating what ran, whether joint training improved over post-hoc fitting, whether adaptation helped, and any failures. Draw exploratory conclusions from this one seed without claiming statistical significance.

Keep the summary focused on the two scientific questions:

1. Can the proposed model infer unseen mass/drag from two brief visual interactions and use them for forecasting/control?
2. Does joint physical shaping of the encoder improve this over JEPA followed by a frozen-encoder physical readout?

## 12. Small correctness checks, not scientific gates

While implementing, check equation signs at hanging-down equilibrium, tensor/action indexing, angle conversion, and that one joint backward pass gives nonzero gradients to encoder/readout/initial-condition parameters. In the post-hoc stage confirm encoder parameters and BatchNorm buffers remain unchanged. During test fitting confirm only parameter/initial-state variables have gradients.

These are ordinary code checks. Do not create a preliminary benchmark sequence or wait for an accuracy threshold before executing the requested runs. Log numerical failures plainly. Never make hidden states available to training to rescue a failed run.

Finish with a concise account of the implemented files, exact commands, what was executed, and the two-variant results or remaining execution limitation.
