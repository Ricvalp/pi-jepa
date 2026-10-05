"""Decode dense latent forecasts with the exact trained physical readout.

Exports carry a valid mask for boundary-truncated queries. Training examples
reconstruct the checkpoint's learned or fixed true-reset simulator target; full hidden states enter only after
model inference. Both passive and controlled PI-JEPA readouts are supported.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pi_jepa.data import ENDPOINTS as TRAIN_ENDPOINTS, RESET_MODES, dataset_root, validate_clocks, validate_manifest
from pi_jepa.checkpoint_interface import validate_checkpoint_interface
from pi_jepa.evaluate_latents import ENDPOINTS, FORECAST_START, STEPS
from pi_jepa.models import Encoder, PhysicalReadout, scale_readout, to_state
from pi_jepa.physics import iota, rollout
from pi_jepa.runs import file_digest, run_directory, write_provenance
from pi_jepa.train import encode_batch
from pi_jepa.initial_conditions import load_initial_conditions_state


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def decode_test_latents(readout, encoded, predicted):
    """Decoded observation at endpoint34, followed by native 0.1 s predictions."""
    if encoded.ndim != 3 or encoded.shape[1:] != (len(ENDPOINTS), 32):
        raise ValueError("Expected observed latents [queries,35,32] at endpoints14..354")
    if predicted.shape != (len(encoded), STEPS, 32):
        raise ValueError("Expected predicted latents [queries,32,32] at endpoints44..354")
    actual = readout(torch.as_tensor(encoded[:, 2:]))
    future = readout(torch.as_tensor(predicted))
    forecast = torch.cat((actual[:, :1], future), dim=1)
    return {"forecast": to_state(forecast).numpy(), "reconstruction": to_state(actual).numpy(),
            "raw_forecast": forecast.numpy(), "raw_reconstruction": actual.numpy()}


@torch.no_grad()
def training_supervision(readout, z, table, ids, theta, forces, dt, substeps, raw_endpoints=None):
    """Training residual operands: integrate from episode reset, gather crop times."""
    raw = readout(z)
    if raw_endpoints is None:
        raw_endpoints = torch.tensor(TRAIN_ENDPOINTS)[None].expand(len(ids), -1)
    states = rollout(table(ids), theta, forces, dt=dt, substeps=substeps)
    states = states.gather(1, raw_endpoints[..., None].expand(-1, -1, 4))
    target = iota(states)
    return {"reconstruction": to_state(raw).numpy(), "supervision": states.numpy(),
            "raw_reconstruction": raw.numpy(),
            "raw_supervision": (target * target.new_tensor([2., 2., 1., 1., 5.])).numpy(),
            "supervised_reconstruction": scale_readout(raw).numpy(), "supervised_target": target.numpy()}


def export(args):
    torch.set_num_threads(2)
    source = args.latent_run.resolve()
    settings = json.loads((source / "settings.json").read_text())
    queries = json.loads((source / "queries.json").read_text())
    config = settings["config"]
    validate_clocks(config)
    passive = config["dataset"] == "passive"
    variant = args.variant or ("passive" if passive else "joint")
    if args.data_root:
        config["paths"]["data"] = str(args.data_root.resolve())
    root = dataset_root(config)
    manifest_path = root / "manifest.json"
    if file_digest(manifest_path) != settings["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the latent evaluation")
    if settings["parameter_condition"] != ("none" if passive else "true_test_parameters"):
        raise ValueError("Use passive latent evaluation or controlled oracle latent evaluation (physical evaluations: latent/)")
    if settings["endpoints"] != ENDPOINTS.tolist():
        raise ValueError("Unexpected dense endpoint alignment")
    if variant not in settings["checkpoints"]:
        raise ValueError(f"Latent evaluation does not contain variant {variant!r}")
    info = settings["checkpoints"][variant]
    checkpoint = Path(info["path"])
    if file_digest(checkpoint) != info["sha256"]:
        raise ValueError("Checkpoint differs from the evaluated latent model")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_interface(saved)
    if saved["config"]["dataset"] != config["dataset"] or manifest["dataset"] != config["dataset"]:
        raise ValueError("Readout checkpoint, latent evaluation, and dataset must select the same corpus")
    if saved["data_manifest_sha256"] != settings["manifest_sha256"]:
        raise ValueError("Checkpoint was trained on a different dataset")
    if saved.get("mode") not in ("joint", "readout") or saved.get("readout") is None or saved.get("initial_conditions") is None:
        raise ValueError("A checkpoint with trained physical readout and initial-condition table is required")
    readout, encoder = PhysicalReadout().eval().requires_grad_(False), Encoder().eval().requires_grad_(False)
    table = load_initial_conditions_state(saved).eval().requires_grad_(False)
    protocol = saved["config"]["training"].get("initial_conditions", "learned")
    fixed_reset = protocol == "true_fixed"
    modules = {"encoder": encoder, "readout": readout, "initial_conditions": table}
    for name, module in modules.items():
        module.load_state_dict(saved[name])
    hashes_before = {name: state_digest(module) for name, module in modules.items()}
    bundle_path = source / variant / "latents.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        test = decode_test_latents(readout, bundle["encoded"], bundle["correct"])
        valid = np.concatenate((np.isfinite(bundle["encoded"][:, 2]).all(-1)[:, None], bundle["valid"]), axis=1)
    if len(test["forecast"]) != len(queries):
        raise ValueError("Query metadata and latent arrays differ in length")
    dt, test_endpoints = config["physics"]["dt"], ENDPOINTS[2:]
    test.update(raw_endpoints=test_endpoints, times_s=(test_endpoints - FORECAST_START) * dt, valid=valid)
    truth = np.full_like(test["forecast"], np.nan)
    force_blocks = []
    input_paths = {manifest_path, source / "settings.json", source / "queries.json", bundle_path}
    for qi, query in enumerate(queries):
        learning_path, truth_path = root / query["path"], root / query["truth"]
        with np.load(learning_path, allow_pickle=False) as learning:
            if passive:
                if np.any(learning["force"]):
                    raise ValueError("Passive readout visualization requires zero recorded forces")
                force_blocks.append(np.zeros((STEPS, 10), dtype=np.float32))
            else:
                force_blocks.append(learning["force_program"][FORECAST_START:ENDPOINTS[-1]].reshape(STEPS, 10))
        with np.load(truth_path, allow_pickle=False) as hidden:
            state = hidden["state"]
            available = test_endpoints[test_endpoints < len(state)]
            truth[qi, :len(available)] = state[available]
        input_paths.update((learning_path, truth_path))
    test.update(truth=truth, force_blocks=np.asarray(force_blocks))

    # First complete window of the first episode in each reset family; no error selection.
    selected = [next((i, row) for i, row in enumerate(manifest["train"]) if row["reset_mode"] == mode)
                for mode in RESET_MODES.values()]
    train_queries = [{**row, "reset_mode": list(RESET_MODES)[row["reset_mode"]], "table_index": i}
                     for i, row in selected]
    frames, forces, theta, ids = [], [], [], []
    for query in train_queries:
        learning_path = root / query["path"]
        with np.load(learning_path, allow_pickle=False) as learning:
            frames.append(learning["rgb"][:65])
            forces.append(learning["force"][:64].reshape(-1))
            theta.append(np.array([1., .25]) if passive else learning["theta"])
            if passive and np.any(forces[-1]):
                raise ValueError("Passive training supervision requires zero recorded forces")
            ids.append(query["table_index"])
            if str(learning["episode_id"].item()) != query["episode_id"]:
                raise ValueError("Training episode identity differs from manifest")
        input_paths.add(learning_path)
    frames = torch.as_tensor(np.stack(frames))
    forces, theta = (torch.as_tensor(np.stack(values), dtype=torch.float32) for values in (forces, theta))
    ids = torch.tensor(ids)
    expected_modes = torch.tensor([RESET_MODES[row["reset_mode"]] for row in train_queries])
    if not torch.equal(table.reset_modes[ids].long(), expected_modes):
        raise ValueError("Training reset table does not match selected episodes")
    with torch.no_grad():
        train = training_supervision(readout, encode_batch(encoder, frames), table, ids, theta, forces,
                                     dt, config["physics"]["substeps"])
    train_truth = []
    for query in train_queries:
        truth_path = root / query["truth"]
        with np.load(truth_path, allow_pickle=False) as hidden:
            train_truth.append(hidden["state"][list(TRAIN_ENDPOINTS)])
        input_paths.add(truth_path)
    train.update(truth=np.asarray(train_truth), times_s=np.asarray(TRAIN_ENDPOINTS) * dt,
                 raw_endpoints=np.asarray(TRAIN_ENDPOINTS), force_blocks=forces[:, 14:64].reshape(3, 5, 10).numpy(),
                 valid=np.ones((3, 6), dtype=bool))
    hashes_after = {name: state_digest(module) for name, module in modules.items()}
    if hashes_before != hashes_after or file_digest(checkpoint) != info["sha256"]:
        raise RuntimeError("Inference changed model parameters, buffers, or checkpoint")
    metadata = {
        "dataset": config["dataset"], "data_schema_version": manifest["schema_version"],
        "physics": manifest["physics"], "camera": manifest["camera"], "cart_limit_m": manifest["cart_limit_m"],
        "checkpoint_step": saved["step"], "weights": "raw", "variant": variant,
        "readout": "Checkpoint PhysicalReadout: Linear(32,64), GELU, Linear(64,5), angular pair normalization",
        "readout_coordinates": ["p", "v", "sin(q)", "cos(q)", "w"],
        "readout_units": ["m", "m/s", "dimensionless", "dimensionless", "rad/s"],
        "state_coordinates": ["p", "v", "q", "w"], "state_units": ["m", "m/s", "rad", "rad/s"],
        "angle_convention": "q=0 down, q=pi up; q=atan2(readout[...,2],readout[...,3])",
        "physical_loss_divisors": [2., 2., 1., 1., 5.], "angular_normalization_eps": 1e-6,
        "test": {"queries": len(queries), "parameter_condition": settings["parameter_condition"],
                 "warm_start_raw_endpoints": [14, 24, 34], "forecast_start_raw_endpoint": 34,
                 "forecast_zero": "Decoded observed history at frame34, not a prediction",
                 "future": ("32 autonomous autoregressive predictions at 0.1 s spacing; no action or parameter inputs"
                            if passive else "32 autoregressive predictions at 0.1 s spacing, true theta, prechosen forces"),
                 "valid": "Only available ground-truth endpoints may be animated; boundary tails stay NaN",
                 "reconstruction": "r(E(actual future video)) is diagnostic only, never forecast input"},
        "train": {"queries": len(train_queries), "selection": "First retained episode in each reset family, first65frames",
                  "reconstruction": "Frozen checkpoint encoder/readout; per-clip GroupNorm and hidden LayerNorm",
                  "initial_conditions_mode": protocol,
                  "supervision_uses_true_reset": fixed_reset,
                  "supervision": ("RK4 from each saved " + ("fixed true" if fixed_reset else "learned") +
                                  " episode-reset state; " + ("fixed theta=[1,.25], zero forces" if passive else
                                                               "training theta and applied forces")),
                  "supervision_is_ground_truth": fixed_reset,
                  "physical_objective": "MSE(scale_readout(r(E(video))), iota(RK4(saved_reset,theta,forces)))",
                  "truth": "Private simulator states loaded only for this comparison"},
        "models_before_sha256": hashes_before, "models_after_sha256": hashes_after,
        "parameters_and_buffers_unchanged": True,
    }
    with run_directory(args.output_root, f"readout-trajectories-seed{config['seed']}", run_dir=args.output) as output:
        write_provenance(output, config, checkpoints={variant: checkpoint}, extra={
            "kind": "readout_trajectory_visualization", "split": ["train", "test"], "weights": "raw",
            "latent_evaluation": str(source), "parameter_condition": settings["parameter_condition"],
            "inputs_sha256": {str(path.resolve()): file_digest(path) for path in sorted(input_paths)},
            "export_script_sha256": file_digest(__file__), "parameters_and_buffers_unchanged": True})
        np.savez_compressed(output / "test_trajectories.npz", **test)
        np.savez_compressed(output / "train_trajectories.npz", **train)
        for name, value in (("test_queries", queries), ("train_queries", train_queries), ("readout_metadata", metadata)):
            (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(output.resolve(), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-run", required=True, type=Path)
    parser.add_argument("--variant", choices=("joint", "posthoc", "passive"), help="Default: passive for passive data, joint otherwise")
    parser.add_argument("--data-root", type=Path, help="Container root holding controlled/ and passive/")
    parser.add_argument("--output", type=Path, help="Fresh export directory")
    parser.add_argument("--output-root", type=Path, default=Path("workspace/evaluations"))
    export(parser.parse_args())


if __name__ == "__main__":
    main()
