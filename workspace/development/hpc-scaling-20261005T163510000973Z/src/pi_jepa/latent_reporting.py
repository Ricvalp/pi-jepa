"""Dense latent forecast summaries with explicit ragged-episode denominators."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def ratio(numerator, denominator):
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else float("nan")


def mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def groups(metadata):
    """Cross apparatus split with all reset families actually present."""
    for group in ["all", *sorted({row["group"] for row in metadata})]:
        for reset in ["all", *sorted({str(row["reset_mode"]) for row in metadata})]:
            selected = np.array([(group == "all" or row["group"] == group) and
                                 (reset == "all" or str(row["reset_mode"]) == reset) for row in metadata])
            if selected.any():
                yield group, reset, selected


def availability(metadata, selected, horizon_s, valid):
    endpoint = 34 + round(horizon_s / .01)
    boundary = np.array([row["recorded_intervals"] < endpoint and "boundary" in row["termination_reason"]
                         for row in metadata])
    total = int(selected.sum())
    return {"queries": total, "valid_queries": int((selected & valid).sum()),
            "boundary_exit_queries": int((selected & boundary).sum()),
            "boundary_exit_rate": float((selected & boundary).sum() / total) if total else float("nan")}


def summarize(bundle, metadata, horizons):
    encoded = np.asarray(bundle["encoded"], dtype=np.float64)
    target, valid = encoded[:, 3:], np.asarray(bundle["valid"], dtype=bool)
    if target.shape[:2] != valid.shape or len(metadata) != len(target):
        raise ValueError("Encoded targets, valid mask, and metadata must align")
    per_query, summary = [], []
    for protocol, keys in (("autoregressive", {"correct": "correct", "persistence": "persistence",
                                               "shuffled": "shuffled", "zero": "zero"}),
                           ("one_step", {"correct": "one_step", "persistence": "one_step_persistence",
                                         "shuffled": "one_step_shuffled", "zero": "one_step_zero"})):
        predictions = {condition: np.asarray(bundle[key], dtype=np.float64) for condition, key in keys.items() if key in bundle}
        errors = {name: ((prediction - target) ** 2).mean(-1) for name, prediction in predictions.items()}
        for seconds in horizons:
            index = round(seconds / .1) - 1
            available = valid[:, index]
            def fields(selected, condition):
                mask = selected & available
                current = errors[condition][mask, index]
                baseline = errors["persistence"][mask, index]
                mse, persistence = mean(current), mean(baseline)
                correct = errors["correct"][mask, index]
                shuffled = errors.get("shuffled")
                variance = float(np.var(target[mask, index], axis=0).mean()) if mask.any() else float("nan")
                sensitivity = ((predictions["correct"][mask, index] - predictions["shuffled"][mask, index]) ** 2).mean(-1) if shuffled is not None else np.array([])
                return {**availability(metadata, selected, seconds, available), "latent_mse": mse,
                        "latent_rmse": float(np.sqrt(mse)), "persistence_mse": persistence,
                        "persistence_skill": 1 - ratio(mse, persistence),
                        "target_variance": variance, "target_drift_mse": persistence,
                        "target_drift_to_variance": ratio(persistence, variance),
                        "action_sensitivity_mse": mean(sensitivity),
                        "action_advantage_mse": mean(shuffled[mask, index] - correct) if shuffled is not None else float("nan"),
                        "fraction_correct_beats_persistence": mean(correct < baseline),
                        "fraction_correct_beats_shuffled": mean(correct < shuffled[mask, index]) if shuffled is not None else float("nan"),
                        "nonfinite_queries": int((~np.isfinite(current)).sum())}
            for group, reset, selected in groups(metadata):
                for condition in predictions:
                    summary.append({"protocol": protocol, "group": group, "reset_mode": reset,
                                    "condition": condition, "horizon_s": seconds, **fields(selected, condition)})
            for qi, row in enumerate(metadata):
                selected = np.arange(len(metadata)) == qi
                for condition in predictions:
                    per_query.append({"protocol": protocol, "query_index": qi, "path": row["path"],
                                      "condition": condition, "horizon_s": seconds, **fields(selected, condition)})
    return per_query, summary


def write_report(bundles, metadata, settings, destination):
    per_query, summary = [], []
    for variant, bundle in bundles.items():
        queries, aggregated = summarize(bundle, metadata, settings["horizons_s"])
        per_query.extend({"variant": variant, **row} for row in queries)
        summary.extend({"variant": variant, **row} for row in aggregated)
    write_csv(destination / "latent_per_query.csv", per_query)
    write_csv(destination / "latent_summary.csv", summary)
    (destination / "metrics.json").write_text(json.dumps(clean({"settings": settings, "summary": summary}), indent=2, allow_nan=False) + "\n")
    lines = ["# Dense latent prediction diagnostics", "",
             f"Dataset: {settings['dataset']}; checkpoints ({', '.join(bundles)}); {len(metadata)} held-out episodes. "
             "Frozen raw weights; no training or readout is used by this evaluation.", "",
             "Histories end at raw frames 14, 24, and 34 (0.14, 0.24, 0.34 s), each using every second image. "
             "Autoregressive forecasts start at frame 34 and advance 0.1 s with no future observed latent in their context. "
             "The separately labeled one-step protocol refreshes its context from actual observed video before every prediction.", "",
             "Errors average over latent coordinates, then over episodes with an available target. "
             "Persistence holds the last observed code: frame 34 for rollouts, the preceding endpoint for one-step diagnostics. "
             "Skill is 1 − mean error / mean persistence error. Undefined ratios are JSON null/blank CSV. "
             "Nonfinite predictions on available targets remain failures and make the aggregate nonfinite; they are not discarded.", "",
             "Valid counts account for boundary truncation. Boundary-exit rate at a horizon is the fraction of all episodes "
             "in that group that exited before its required target; short episodes remain in that denominator. "
             "The first missing target is not padded with a repeated state.", ""]
    if settings["dataset"] == "controlled":
        lines += ["True test mass/drag are used as an explicitly oracle condition. Correct, zero, and shuffled actions share "
                  "the same videos, parameter pairs and targets. Shuffling exchanges complete prechosen action programs "
                  "within apparatus, including forces associated with the observed context; temporal blocks are never permuted. "
                  "This also makes context-force conditioning inconsistent with its observed video, so the advantage "
                  "tests matching the whole intervention history, not exclusively sensitivity to future forces. "
                  "The saved donor map is identical across checkpoints. These are open-loop queries, not feedback actions "
                  "that reveal future state. Action sensitivity alone is not evidence of an accurate action response.", ""]
    else:
        lines += ["Passive models receive latent codes only. No force/parameter conditioning or identification is performed.", ""]
    lines += ["| Model | Protocol | Horizon (s) | Correct MSE | Persistence MSE | Skill | Valid / total | Boundary exits |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return f"{value:.5g}" if np.isfinite(value) else "NA"
    for row in summary:
        if row["condition"] == "correct" and row["group"] == row["reset_mode"] == "all":
            lines.append(f"| {row['variant']} | {row['protocol']} | {row['horizon_s']:.1f} | {number(row['latent_mse'])} | "
                         f"{number(row['persistence_mse'])} | {number(row['persistence_skill'])} | "
                         f"{row['valid_queries']} / {row['queries']} | {row['boundary_exit_rate']:.1%} |")
    lines += ["", "CSV/JSON files include every action condition, apparatus split and reset family. "
              "Raw latent MSE depends on the encoder's scale; it cannot rank passive versus controlled corpora or different "
              "encoders as a direct ablation. Compare persistence skill within each representation and inspect physical "
              "readouts separately. These are one-seed descriptive diagnostics, not evidence of statistical significance.", ""]
    (destination / "results.md").write_text("\n".join(lines))
    return {"per_query": per_query, "summary": summary}
