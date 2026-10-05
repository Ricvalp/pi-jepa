"""Physical forecast scores, separate observed readouts, and trajectory overlays."""
from __future__ import annotations

import json
import numpy as np

from pi_jepa.evaluate_latents import ENDPOINTS, FORECAST_START, HORIZONS
from pi_jepa.latent_reporting import availability, clean, groups, write_csv

COMPONENTS = ("position_rmse_m", "velocity_rmse_m_s", "angle_rmse_rad", "angular_velocity_rmse_rad_s")
CONDITIONS = ("nominal", "fitted", "oracle")


def embedding(state):
    return np.stack((state[..., 0] / 2, state[..., 1] / 2, np.sin(state[..., 2]),
                     np.cos(state[..., 2]), state[..., 3] / 5), axis=-1)


def score(predicted, truth):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.shape != truth.shape:
        raise ValueError("Prediction and truth shapes differ")
    residual = predicted - truth
    residual[..., 2] = np.arctan2(np.sin(residual[..., 2]), np.cos(residual[..., 2]))
    count = int(np.prod(predicted.shape[:-1]))
    errors = np.sqrt(np.mean(residual.reshape(-1, 4) ** 2, axis=0)) if count else np.full(4, np.nan)
    aggregate = float(np.sqrt(np.mean((embedding(predicted) - embedding(truth)) ** 2))) if count else float("nan")
    return {**dict(zip(COMPONENTS, errors.tolist())), "normalized_rmse": aggregate,
            "nonfinite_states": int((~np.isfinite(predicted).all(axis=-1)).sum()), "states": count}


def trajectory_overlays(results, metadata, truth, destination):
    """Fixed first episode in each reset family; native samples, no interpolation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = [next(i for i, row in enumerate(metadata) if row["reset_mode"] == mode)
              for mode in sorted({row["reset_mode"] for row in metadata})]
    labels = ("Cart position (m)", "Cart velocity (m/s)", "Angle (rad)", "Angular velocity (rad/s)")
    for variant, result in results.items():
        figure, axes = plt.subplots(4, len(chosen), squeeze=False, figsize=(5 * len(chosen), 10))
        for column, qi in enumerate(chosen):
            for component, label in enumerate(labels):
                axis = axes[component, column]
                def values(states):
                    value = states[:, component].copy()
                    # Unwrap each trajectory to avoid plotting artificial 2pi jumps.
                    return np.unwrap(value) if component == 2 else value
                axis.plot(ENDPOINTS * .01, values(truth[qi]), color="black", label="Ground truth")
                axis.plot(ENDPOINTS * .01, values(result["observed"][qi]), color="green", alpha=.7, label="r(encoded video)")
                for condition, style in (("fitted", "-"), ("oracle", "--")):
                    axis.plot(ENDPOINTS[3:] * .01, values(result[f"{condition}_learned"][qi]),
                              linestyle=style, marker=".", label=f"r(predicted z), {condition}")
                axis.axvline(FORECAST_START * .01, color="gray", linestyle=":")
                axis.set(xlabel="Episode time (s)", ylabel=label)
                if component == 0:
                    axis.set_title(f"{variant}: {metadata[qi]['reset_mode']}")
                    axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(destination / f"trajectory_overlays_{variant}.png", dpi=140)
        plt.close(figure)


def write_report(results, metadata, truth, true_theta, fitted, manifest, cfg, destination, failures):
    forecasts, observed, parameters = [], [], []
    for variant, result in results.items():
        for group, reset, selected in groups(metadata):
            observation_mask = selected[:, None] & np.isfinite(truth).all(-1)
            observed.append({"variant": variant, "group": group, "reset_mode": reset,
                             **score(result["observed"][observation_mask], truth[observation_mask])})
            for horizon in HORIZONS:
                index = round(horizon / .1) - 1
                valid = result["valid"][:, index]
                mask = selected & valid
                counts = availability(metadata, selected, horizon, valid)
                for condition in CONDITIONS:
                    for engine in ("learned", "physical"):
                        forecasts.append({"variant": variant, "group": group, "reset_mode": reset,
                                          "condition": condition, "engine": engine, "horizon_s": horizon,
                                          **counts, **score(result[f"{condition}_{engine}"][mask, index], truth[mask, index + 3])})
        for ai, apparatus in enumerate(manifest["test"]):
            qi = next(i for i, entry in enumerate(metadata) if entry["apparatus_id"] == apparatus["apparatus_id"])
            row = {"variant": variant, "apparatus_id": apparatus["apparatus_id"], "group": apparatus["group"]}
            for coordinate, name, nominal in ((0, "mass", 1.), (1, "drag", .25)):
                actual, estimate = float(true_theta[qi, coordinate]), float(fitted[variant][ai, coordinate])
                row.update({f"true_{name}": actual, f"fitted_{name}": estimate,
                            f"nominal_{name}_absolute_error": abs(nominal - actual),
                            f"fitted_{name}_absolute_error": abs(estimate - actual)})
            parameters.append(row)
    write_csv(destination / "forecast_errors.csv", forecasts)
    write_csv(destination / "observed_state_recovery.csv", observed)
    write_csv(destination / "parameters.csv", parameters)
    (destination / "metrics.json").write_text(json.dumps(clean({"forecasts": forecasts, "observed": observed,
        "parameters": parameters, "failures": failures}), indent=2, allow_nan=False) + "\n")
    trajectory_overlays(results, metadata, truth, destination)
    lines = ["# Dense controlled prediction and readout diagnostics", "",
             f"Frozen raw checkpoints: {', '.join(results)}; {len(metadata)} open-loop held-out queries. "
             "The same episodes and parameter conditions are used for each variant.", "",
             "Calibration fits one parameter pair and two independent unknown states at t=0.14 s per apparatus. "
             "Cart position there is fitted, not set to zero. Only video readouts and applied forces enter this fit. "
             "All nominal/fitted forecasts were saved before test truth was opened; oracle then supplies only true mass/drag. "
             "The physics comparison starts from the same observed readout as latent prediction, never a hidden query state.", "",
             "Three histories ending at 0.14, 0.24 and 0.34 s initialize the latent rollout. Native predictions advance "
             "0.10 s; the physical solver applies each recorded force for 0.01 s using five 0.002 s RK4 steps. "
             "No integrator step crosses a force discontinuity.", "",
             "`forecast_errors.csv` scores r(predicted z) separately from direct physical rollouts, at 0.1, 0.2, 0.4, "
             "0.8, 1.6 and 3.2 s after observed context. `observed_state_recovery.csv` measures r(encoded actual video), "
             "including later observed frames, and is not a prediction metric. It pools available endpoints within "
             "each split/reset group, so longer episodes contribute more endpoint samples.", "",
             "Component errors use physical units and circular angle differences. Normalized RMSE uses "
             "[p/2,v/2,sin(q),cos(q),w/5]. Forecast averages use available targets only, with valid counts and "
             "boundary-exit rates reported against all episodes in each group. Nonfinite predictions are never "
             "silently omitted. Undefined scores are JSON null and blank CSV cells.", "",
             "`latent/` contains oracle one-step/autoregressive latent, persistence, shuffled-whole-program and "
             "zero-action diagnostics. Its one-step protocol uses fresh observed context and is labeled separately. "
             "Raw latent errors are not comparable across independently learned encoders.", "",
             "Trajectory overlays show native samples from a fixed first episode in each reset family. "
             "They show both forecast and observed-readout traces; examples are not selected by error.", "",
             "Calibration failures: " + (json.dumps(failures) if failures else "none"), ""]
    for variant in results:
        lines += [f"![{variant} trajectory examples](trajectory_overlays_{variant}.png)", ""]
    (destination / "results.md").write_text("\n".join(lines))
