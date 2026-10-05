"""Frozen controlled calibration and open-loop forecasts with dense histories.

Nominal/fitted predictions are written before test truth is opened. Calibration
fits an unknown state at frame 14, including its cart position, without access to
simulator state. Passive checkpoints use the separate autonomous latent test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pi_jepa.evaluate_latents import (
    ENDPOINTS, FORECAST_START, HISTORY_START, HORIZONS, STEPS, TOKEN_STRIDE,
    add_arguments, dataset_path, encode_endpoints, evaluate_model,
    load_networks as load_latent_networks, load_queries, network_digest,
    parse_config, rollout_latents, shuffle_action_programs, validate_checkpoints,
)
from pi_jepa.models import PhysicalReadout, scale_readout, to_state
from pi_jepa.physics import iota, rollout
from pi_jepa.runs import file_digest, run_directory, write_provenance

CONDITIONS = ("nominal", "fitted", "oracle")


def get_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def load_networks(checkpoint, device):
    encoder, predictor, _ = load_latent_networks(checkpoint, device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    readout = PhysicalReadout().to(device)
    readout.load_state_dict(saved["readout"])
    readout.eval().requires_grad_(False)
    return encoder, predictor, readout


def load_learning(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in ("rgb", "force", "t")}


def fit_parameters(targets, forces, adaptation, physics):
    """Fit one apparatus from two ragged calibration episodes, networks frozen.

    targets[j] is normalized r(E(video)) at raw endpoints 14,24,...; forces[j]
    starts at interval 14 and ends at its last target endpoint. The two unknown
    initial states belong to t=.14 s, not the commanded reset. Their positions
    and velocities are unrestricted parameters. No truth arrays enter this API.
    """
    if len(targets) != 2 or len(forces) != 2:
        raise ValueError("Each apparatus requires two independent calibration episodes")
    for target, force in zip(targets, forces):
        if len(target) < 2 or len(force) != TOKEN_STRIDE * (len(target) - 1):
            raise ValueError("Calibration forces must align with ten-interval endpoint gaps")
    lower, upper = targets[0].new_tensor([.3, .001]), targets[0].new_tensor([2., 1.2])
    nominal = targets[0].new_tensor([1., .25])
    raw = nn.Parameter(torch.logit((nominal - lower) / (upper - lower)))
    first = torch.stack([target[0] for target in targets])
    initial = nn.Parameter(torch.stack((2 * first[:, 0], 2 * first[:, 1],
                                       torch.atan2(first[:, 2], first[:, 3]), 5 * first[:, 4]), dim=-1))
    optimizer = torch.optim.Adam([raw, initial], lr=adaptation["lr"])
    losses, failure = [], None
    for update in range(adaptation["updates"]):
        optimizer.zero_grad(set_to_none=True)
        theta = lower + (upper - lower) * raw.sigmoid()
        clip_losses = []
        for index, (target, force) in enumerate(zip(targets, forces)):
            simulated = rollout(initial[index], theta, force, dt=physics["dt"], substeps=physics["substeps"])
            clip_losses.append((iota(simulated[::TOKEN_STRIDE]) - target).square().mean())
        loss = torch.stack(clip_losses).mean()
        losses.append(float(loss.detach()))
        if not torch.isfinite(loss):
            failure = f"Nonfinite calibration loss at update {update + 1}"
            break
        loss.backward()
        if not torch.isfinite(raw.grad).all() or not torch.isfinite(initial.grad).all():
            failure = f"Nonfinite calibration gradient at update {update + 1}"
            break
        optimizer.step()
    return (lower + (upper - lower) * raw.sigmoid()).detach(), initial.detach(), np.asarray(losses), failure


@torch.no_grad()
def forecast(predictor, readout, warm_codes, forces, theta, cfg, steps=STEPS):
    """Decode native 100 ms latent forecasts and compare a physical rollout."""
    latent = rollout_latents(predictor, warm_codes, forces, theta, steps)
    learned = to_state(readout(latent))
    start = to_state(readout(warm_codes[:, -1]))
    stop = FORECAST_START + TOKEN_STRIDE * steps
    physical = rollout(start, theta, forces[:, FORECAST_START:stop], dt=cfg["physics"]["dt"],
                       substeps=cfg["physics"]["substeps"])[:, TOKEN_STRIDE::TOKEN_STRIDE]
    return learned.cpu().numpy(), physical.cpu().numpy()


def calibrate(modules, root, manifest, cfg, device, batch_size, folder):
    encoder, _, readout = modules
    fitted, initial, curves, failures = [], [], [], []
    for ai, apparatus in enumerate(manifest["test"]):
        targets, actions = [], []
        for path in apparatus["calibration"]:
            episode = load_learning(root / path)
            endpoints = np.arange(HISTORY_START, len(episode["rgb"]), TOKEN_STRIDE)
            with torch.no_grad():
                codes = encode_endpoints(encoder, episode["rgb"], device, batch_size, endpoints)
                targets.append(scale_readout(readout(codes)))
            actions.append(torch.as_tensor(episode["force"][HISTORY_START:endpoints[-1]].reshape(-1),
                                            device=device, dtype=torch.float32))
        pair, states, losses, failure = fit_parameters(targets, actions, cfg["adaptation"], cfg["physics"])
        fitted.append(pair.cpu().numpy())
        initial.append(states.cpu().numpy())
        curves.append(losses)
        if failure:
            failures.append({"apparatus_id": apparatus["apparatus_id"], "failure": failure})
        print(f"  Calibration {ai + 1}/{len(manifest['test'])}: {len(losses)} updates", flush=True)
    padded = np.full((len(curves), max(map(len, curves), default=0)), np.nan)
    for i, curve in enumerate(curves):
        padded[i, :len(curve)] = curve
    np.savez_compressed(folder / "adaptation.npz", fitted=np.asarray(fitted), endpoint_initial_states=np.asarray(initial),
                        losses=padded, first_endpoint=np.array(HISTORY_START))
    (folder / "adaptation_status.json").write_text(json.dumps({"failures": failures,
        "state_time_s": .14, "unknown_initial_coordinates": ["p", "v", "q", "w"]}, indent=2) + "\n")
    return np.asarray(fitted), failures


@torch.no_grad()
def nominal_fitted_predictions(modules, root, metadata, programs, fitted, manifest, cfg, device, batch_size):
    encoder, predictor, readout = modules
    count = len(metadata)
    observed = np.full((count, len(ENDPOINTS), 4), np.nan, dtype=np.float32)
    result = {f"{condition}_{engine}": np.full((count, STEPS, 4), np.nan, dtype=np.float32)
              for condition in ("nominal", "fitted") for engine in ("learned", "physical")}
    valid = np.zeros((count, STEPS), dtype=bool)
    apparatus_index = {entry["apparatus_id"]: i for i, entry in enumerate(manifest["test"])}
    for qi, entry in enumerate(metadata):
        episode = load_learning(root / entry["path"])
        endpoints = ENDPOINTS[ENDPOINTS < len(episode["rgb"])]
        codes = encode_endpoints(encoder, episode["rgb"], device, batch_size, endpoints)
        observed[qi, :len(endpoints)] = to_state(readout(codes)).cpu().numpy()
        if len(codes) < 3:
            continue
        valid[qi, :len(codes) - 3] = True
        force = torch.as_tensor(programs[qi:qi + 1], device=device, dtype=torch.float32)
        pairs = {"nominal": np.array([1., .25]), "fitted": fitted[apparatus_index[entry["apparatus_id"]]]}
        for condition, pair in pairs.items():
            theta = torch.as_tensor(pair[None], device=device, dtype=torch.float32)
            learned, physical = forecast(predictor, readout, codes[None, :3], force, theta, cfg)
            result[f"{condition}_learned"][qi] = learned[0]
            result[f"{condition}_physical"][qi] = physical[0]
    return {"observed": observed, "valid": valid, **result}


def run(cfg, device_name="auto", *, checkpoints, output=None, batch_size=64, shuffle_seed=None):
    if cfg.get("dataset") == "passive":
        from pi_jepa.evaluate_latents import run as latent_run
        return latent_run(cfg, device_name, output, batch_size, shuffle_seed, checkpoints=checkpoints)
    checkpoints = validate_checkpoints(cfg, checkpoints, require_readout=True)
    root, device = dataset_path(cfg), get_device(device_name)
    torch.set_num_threads(cfg.get("evaluation", {}).get("cpu_threads", 4))
    manifest = json.loads((root / "manifest.json").read_text())
    # This call never opens a truth file.
    programs, _, metadata = load_queries(root, oracle=False)
    seed = cfg["seed"] if shuffle_seed is None else shuffle_seed
    shuffled, donors = shuffle_action_programs(programs, metadata, seed)
    with run_directory(cfg["paths"]["results"], f"controlled-physical-eval-seed{cfg['seed']}", run_dir=output) as destination:
        provenance = write_provenance(destination, cfg, checkpoints=checkpoints,
            extra={"kind": "physical_evaluation", "weights": "raw", "split": "test", "device": str(device),
                   "parameter_conditions": list(CONDITIONS), "first_calibration_endpoint": 14})
        (destination / "queries.json").write_text(json.dumps(metadata, indent=2) + "\n")
        settings = {"dataset": "controlled", "config": cfg, "horizons_s": list(HORIZONS),
                    "manifest_sha256": file_digest(root / "manifest.json"),
                    "endpoints": ENDPOINTS.tolist(), "forecast_start": FORECAST_START, "shuffle_seed": seed,
                    "parameter_condition": "nominal/fitted/oracle", "checkpoints": {}}
        results, fitted_values, failures = {}, {}, []
        for variant, checkpoint in checkpoints.items():
            folder = destination / variant
            folder.mkdir()
            modules = load_networks(checkpoint, device)
            before = [network_digest(module) for module in modules]
            fitted, failed = calibrate(modules, root, manifest, cfg, device, batch_size, folder)
            predictions = nominal_fitted_predictions(modules, root, metadata, programs, fitted, manifest,
                                                       cfg, device, batch_size)
            np.savez_compressed(folder / "forecasts_nominal_fitted.npz", **predictions)
            results[variant], fitted_values[variant] = predictions, fitted
            failures.extend({"variant": variant, **item} for item in failed)
            assert before == [network_digest(module) for module in modules]
            del modules
        # All variants' calibration and non-oracle predictions are now immutable.
        _, true_theta, _ = load_queries(root, oracle=True)
        truth = np.full((len(metadata), len(ENDPOINTS), 4), np.nan, dtype=np.float64)
        for qi, entry in enumerate(metadata):
            with np.load(root / entry["truth"], allow_pickle=False) as saved:
                state = saved["state"]
                endpoints = ENDPOINTS[ENDPOINTS < len(state)]
                truth[qi, :len(endpoints)] = state[endpoints]
        np.savez_compressed(destination / "scoring_truth.npz", truth=truth, theta_true=true_theta, endpoints=ENDPOINTS)
        latent_folder = destination / "latent"
        latent_folder.mkdir()
        np.savez_compressed(latent_folder / "inputs.npz", forces=programs, shuffled_forces=shuffled,
                            theta=true_theta, shuffle_source_indices=donors, endpoints=ENDPOINTS)
        (latent_folder / "queries.json").write_text(json.dumps(metadata, indent=2) + "\n")
        bundles = {}
        for variant, checkpoint in checkpoints.items():
            modules = load_networks(checkpoint, device)
            encoder, predictor, readout = modules
            before = [network_digest(module) for module in modules]
            bundle = evaluate_model(encoder, predictor, root, metadata, programs, true_theta, device,
                                    batch_size, shuffled)
            bundles[variant] = bundle
            result = results[variant]
            with torch.no_grad():
                result["oracle_learned"] = to_state(readout(torch.as_tensor(bundle["correct"], device=device))).cpu().numpy()
                result["oracle_physical"] = np.full_like(result["oracle_learned"], np.nan)
                for qi in range(len(metadata)):
                    warm = bundle["encoded"][qi:qi + 1, :3]
                    if not np.isfinite(warm).all():
                        continue
                    # Physical integration receives readout-derived state, never hidden state.
                    x0 = to_state(readout(torch.as_tensor(warm[:, -1], device=device)))
                    actions = torch.as_tensor(programs[qi:qi + 1, FORECAST_START:ENDPOINTS[-1]], device=device, dtype=torch.float32)
                    theta = torch.as_tensor(true_theta[qi:qi + 1], device=device, dtype=torch.float32)
                    path = rollout(x0, theta, actions, dt=cfg["physics"]["dt"], substeps=cfg["physics"]["substeps"])
                    result["oracle_physical"][qi] = path[0, TOKEN_STRIDE::TOKEN_STRIDE].cpu().numpy()
            np.savez_compressed(destination / variant / "forecasts.npz", **result)
            (latent_folder / variant).mkdir()
            np.savez_compressed(latent_folder / variant / "latents.npz", **bundle)
            settings["checkpoints"][variant] = {"path": str(checkpoint), "sha256": file_digest(checkpoint),
                "step": torch.load(checkpoint, map_location="cpu", weights_only=True)["step"], "weights": "raw",
                "parameters_and_buffers_unchanged": before == [network_digest(module) for module in modules]}
            assert settings["checkpoints"][variant]["parameters_and_buffers_unchanged"]
        (destination / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        (latent_folder / "settings.json").write_text(json.dumps({**settings, "parameter_condition": "true_test_parameters"}, indent=2) + "\n")
        from pi_jepa.latent_reporting import write_report as latent_report
        from pi_jepa.reporting import write_report
        latent_report(bundles, metadata, settings, latent_folder)
        write_report(results, metadata, truth, true_theta, fitted_values, manifest, cfg, destination, failures)
        unchanged = all(file_digest(path) == provenance["checkpoints"][name]["sha256"] for name, path in checkpoints.items())
        (destination / "checkpoint_integrity.json").write_text(json.dumps({"unchanged": unchanged}) + "\n")
        if not unchanged:
            raise RuntimeError("Checkpoint changed during evaluation")
    print(f"Evaluation complete: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    config, checkpoints = parse_config(args, parser)
    run(config, args.device, checkpoints=checkpoints, output=args.output,
        batch_size=args.batch_size, shuffle_seed=args.shuffle_seed)


if __name__ == "__main__":
    main()
