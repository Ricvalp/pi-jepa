# Dataset collection QA — schema 3, seed 42

Generated on 2026-10-05 with the approved **0.85 m rod, ±1.5 m cart limit,
and 96×96 square isotropic ±2.6 m camera**. This verifies collection and
interfaces; it does not establish useful learned dynamics. See the self-contained
[dataset report for ChatGPT](new_dataset_report.md) for scientific definitions,
stored interfaces, supervision, and training commands.

## Retained corpus and geometric validation

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

## Numerical integration and controller

The recomputed nominal 0.85 m rod LQR gain at `[M,b]=[1,0.25]` is
`[-4.472135955, -6.706516738, 55.231379049, 15.433956187]`. The undamped energy check
started at -1.014537 J and ended at
-1.014537 J. The damped check changed from
-1.014537 J to -1.077055 J;
its largest per-record increase was
-1.7e-08 J.

## Motion and image change

Displacements below are per 10 ms recorded interval in final 96×96 pixels.
Exactly repeated raster images can occur despite nonzero subpixel movement;
they are retained. Predictor steps span ten such intervals (0.1 s).

| Training family | Cart displacement median / p95 / p99 (px) | Bob displacement median / p95 / p99 (px) | Identical successive RGB | Saturated-force intervals |
|---|---|---|---:|---:|
| passive: upright_falls | 0.033 / 0.168 / 0.201 | 0.320 / 0.947 / 0.972 | 26.1% | 0.00% |
| passive: downward_oscillations | 0.032 / 0.097 / 0.121 | 0.184 / 0.468 / 0.533 | 36.2% | 0.00% |
| passive: nonlinear_excursions | 0.035 / 0.154 / 0.188 | 0.477 / 0.860 / 0.917 | 18.3% | 0.00% |
| controlled: pulse | 0.043 / 0.172 / 0.259 | 0.238 / 0.835 / 0.960 | 24.3% | 0.00% |
| controlled: multisine | 0.040 / 0.148 / 0.213 | 0.231 / 0.874 / 0.979 | 26.3% | 0.00% |
| controlled: noisy_lqr | 0.027 / 0.115 / 0.180 | 0.021 / 0.087 / 0.121 | 56.9% | 1.47% |
| controlled: lqr_release | 0.033 / 0.148 / 0.289 | 0.025 / 0.108 / 0.232 | 51.5% | 4.74% |

The full JSON reports also define almost-identical frames as mean absolute RGB
channel change below 0.01 on the uint8 0–255 scale; they contain all state/action/
duration quantiles, circular angle histograms, family counts, and contact sheets.
Video previews explicitly select every fourth 100 Hz recorded frame at 25 fps
(real time), without interpolated states. The source RGB and force arrays are
unchanged by the annotated visualization.

## Training validation

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
