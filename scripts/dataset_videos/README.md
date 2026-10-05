# Dataset and physical-readout videos

From the repository root, prepare this small independent environment once:

```bash
uv sync --project scripts/dataset_videos --locked
uv run --project scripts/dataset_videos --frozen --no-sync \
  python scripts/dataset_videos/render.py --data-root workspace/data --dataset controlled
```

Repeat with `--dataset passive`. Schema-3 datasets are required. A fresh
`workspace/evaluations/dataset-forces-*` folder contains an HTML gallery and two
training examples per collection family, plus a boundary-truncated example when
available. `--all` renders every stored episode; `--output-root` or a fresh
`--output` chooses the destination.

Default playback is real time: **every fourth saved 100 Hz frame at 25 fps**.
`--stride 1` instead displays every record at 100 fps. An optional `--speed 0.5`
is labeled slow motion. Stored RGB is enlarged uniformly, with no aspect-ratio
change or invented intermediate states. The 0.85 m rod uses the isotropic ±2.6 m square camera; recording stops
after the cart first crosses ±1.5 m.

The force number and arrow show the actual outgoing-interval force, including
both feedback and excitation for LQR collection. Positive force points right;
all clips use a ±5 N scale and 20 display pixels per newton. Force `k` applies
from recorded frame `k` to `k+1`; the terminal frame has no outgoing force. Hidden
cart position only anchors the annotation, and is never supplied to a model.
`videos.json` records source/manifest hashes, timing, subsampling, and selection.
MP4 encoding uses the pinned `imageio-ffmpeg` binary in this environment.

## Visualize the trained readout

After a joint checkpoint has been evaluated with `pi_jepa.evaluate_latents`,
export its actual readout and compare with truth:

```bash
uv run --frozen --no-sync python scripts/export_readout_trajectories.py \
  --latent-run PATH_TO_LATENT_EVALUATION
uv run --project scripts/dataset_videos --frozen --no-sync \
  python scripts/dataset_videos/compare_readouts.py --trajectories PATH_PRINTED_BY_EXPORTER
```

The gallery contains the available prefix of each test trajectory and three
training examples (downward, nonlinear, upright). Test panels show truth,
`r(P-predicted z)`, and `r(E(actual video))` separately. Training panels show
truth, the saved readout of observed clips, and simulator supervision from the
learned episode reset. No readout is refitted and the simulator target is not
called ground truth. These physical-readout diagrams use shared metric axes and
display all five readout coordinates.

Predictions are **0.1 seconds apart**, displayed as held samples without
interpolation. Invalid targets beyond a boundary exit are not rendered. Use
`--limit 2` for a small preview; default is every trajectory. Joint checkpoints from either corpus include the physical readout. Passive
JEPA-only checkpoints have no physical readout and cannot use this comparison.
