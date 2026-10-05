"""Frozen dense-history forecasts; future video is used only for scoring.

Controlled diagnostics use true test parameters explicitly. Passive prediction
has no action/parameter input and no identification stage. Episodes stay ragged:
missing targets after a boundary exit are masked, never invented by padding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from pi_jepa.models import Encoder, Predictor, PassivePredictor, encode_temporal
from pi_jepa.data import validate_clocks, validate_manifest
from pi_jepa.checkpoint_interface import validate_checkpoint_interface
from pi_jepa.runs import file_digest, run_directory, write_provenance

HISTORY_START = 14
TOKEN_STRIDE = 10
FORECAST_START = 34
STEPS = 32
ENDPOINTS = np.arange(HISTORY_START, FORECAST_START + TOKEN_STRIDE * STEPS + 1, TOKEN_STRIDE)
HORIZONS = (.1, .2, .4, .8, 1.6, 3.2)
VARIANTS = {"joint": ("joint",), "posthoc": ("jepa", "readout"), "passive": ("jepa", "joint")}


def dataset_path(config):
    return Path(config["paths"]["data"]) / config.get("dataset", "controlled")


def network_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_networks(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_interface(saved)
    dataset = saved["config"].get("dataset", "controlled")
    encoder = Encoder().to(device)
    predictor = (PassivePredictor() if dataset == "passive" else Predictor()).to(device)
    for name, module in (("encoder", encoder), ("predictor", predictor)):
        module.load_state_dict(saved[name])
        module.eval().requires_grad_(False)
    info = {"path": str(checkpoint), "sha256": file_digest(checkpoint), "step": saved["step"],
            "mode": saved["mode"], "seed": saved["config"]["seed"], "dataset": dataset, "weights": "raw"}
    return encoder, predictor, info


def validate_checkpoints(config, checkpoints, require_readout=False):
    if not checkpoints or not set(checkpoints).issubset(VARIANTS):
        raise ValueError("Provide at least one joint, posthoc, or passive checkpoint")
    paths = {name: Path(path).resolve() for name, path in checkpoints.items()}
    clocks = validate_clocks(config)
    manifest_path = dataset_path(config) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if manifest.get("clocks") != clocks:
        raise ValueError("Dataset manifest clocks differ from the evaluation configuration")
    manifest_digest = file_digest(manifest_path)
    for variant, path in paths.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint_interface(saved)
        if saved.get("mode") not in VARIANTS[variant]:
            raise ValueError(f"{variant} requires a checkpoint from {VARIANTS[variant]}: {path}")
        if saved["config"].get("dataset") != config.get("dataset", "controlled"):
            raise ValueError("Checkpoint and evaluation dataset differ")
        if saved.get("data_manifest_sha256") != manifest_digest:
            raise ValueError(f"Evaluation dataset manifest differs from the checkpoint's training dataset: {path}")
        if require_readout and (saved.get("readout") is None or saved.get("mode") not in ("joint", "readout")):
            raise ValueError("Physical evaluation requires a joint or completed readout-stage checkpoint")
    return paths


@torch.no_grad()
def encode_endpoints(encoder, frames, device, batch_size=64, endpoints=None):
    """Causal histories [e-14,e-12,...,e], with all endpoints in raw indices.

    ``batch_size`` bounds the prepared histories in memory. The shared encoder
    helper still processes one temporal offset per call, with one episode here.
    """
    endpoints = np.arange(HISTORY_START, len(frames), TOKEN_STRIDE) if endpoints is None else np.asarray(endpoints)
    if batch_size < 1 or np.any(endpoints < HISTORY_START) or np.any(endpoints >= len(frames)):
        raise ValueError("Invalid history endpoints or batch size")
    if not len(endpoints):
        return torch.empty((0, 32), device=device)
    codes = []
    for start in range(0, len(endpoints), batch_size):
        clips = np.stack([frames[e - HISTORY_START:e + 1:2] for e in endpoints[start:start + batch_size]])
        x = torch.from_numpy(np.ascontiguousarray(clips)).to(device)
        x = x.permute(0, 1, 4, 2, 3).flatten(1, 2).float().div_(127.5).sub_(1)
        codes.append(encode_temporal(encoder, x.unsqueeze(0))[0])
    return torch.cat(codes)


def shuffle_action_programs(programs, metadata, seed=42):
    """Derange entire prechosen programs within apparatus; never shuffle time.

    Unlike truncation-padded applied-force arrays, these full programs are valid
    counterfactual inputs even when a donor episode exited the camera early.
    """
    programs = np.asarray(programs)
    if programs.ndim != 2 or len(programs) != len(metadata):
        raise ValueError("Expected full [queries,recorded_intervals] programs")
    groups = {}
    for index, entry in enumerate(metadata):
        groups.setdefault(str(entry["apparatus_id"]), []).append(index)
    rng, donors = np.random.default_rng(seed), np.arange(len(programs))
    for key in sorted(groups):
        if len(groups[key]) < 2:
            raise ValueError("Action shuffling requires at least two queries per apparatus")
        cycle = rng.permutation(groups[key])
        donors[cycle] = np.roll(cycle, 1)
    return programs[donors].copy(), donors


@torch.no_grad()
def rollout_latents(predictor, warm_codes, forces=None, theta=None, steps=STEPS, *, passive=False):
    """Only three observed codes enter; all later context codes are predictions."""
    if warm_codes.ndim != 3 or warm_codes.shape[1:] != (3, 32) or steps < 1:
        raise ValueError("Expected [queries,3,32] warm codes and positive steps")
    if not passive and (forces is None or forces.ndim != 2 or
                        forces.shape[0] != len(warm_codes) or
                        forces.shape[1] < FORECAST_START + TOKEN_STRIDE * steps or
                        theta is None or theta.shape != (len(warm_codes), 2)):
        raise ValueError("Insufficient aligned ten-force blocks or apparatus parameters")
    context, outputs = list(warm_codes.unbind(1)), []
    for step in range(steps):
        if passive:
            prediction = predictor(context[-1])
        else:
            blocks = torch.stack([forces[:, HISTORY_START + TOKEN_STRIDE * k:HISTORY_START + TOKEN_STRIDE * (k + 1)]
                                  for k in range(step, step + 3)], dim=1)
            prediction = predictor(torch.stack(context[-3:], dim=1), blocks, theta)[:, -1]
        outputs.append(prediction)
        context.append(prediction)
    return torch.stack(outputs, dim=1)


def persistence_predictions(warm_codes, steps=STEPS):
    if warm_codes.ndim != 3 or warm_codes.shape[1:] != (3, 32) or steps < 1:
        raise ValueError("Expected [queries,3,32] warm codes and positive steps")
    return warm_codes[:, -1:].expand(-1, steps, -1).clone()


@torch.no_grad()
def one_step_latents(predictor, encoded, forces=None, theta=None, *, passive=False):
    """Teacher-forced one-step diagnostics, explicitly separate from rollouts."""
    outputs = []
    for step in range(max(0, encoded.shape[1] - 3)):
        if passive:
            prediction = predictor(encoded[:, step + 2])
        else:
            blocks = torch.stack([forces[:, HISTORY_START + TOKEN_STRIDE * k:HISTORY_START + TOKEN_STRIDE * (k + 1)]
                                  for k in range(step, step + 3)], dim=1)
            prediction = predictor(encoded[:, step:step + 3], blocks, theta)[:, -1]
        outputs.append(prediction)
    return torch.stack(outputs, dim=1) if outputs else encoded[:, :0]


def load_queries(root, oracle=True):
    """Read metadata/actions and optionally oracle theta, never hidden states/RGB.

    RGB is subsequently loaded one episode at a time. Complete query programs
    were chosen before simulation and therefore carry no future feedback state.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    passive = manifest["dataset"] == "passive"
    entries = []
    if passive:
        entries = [dict(entry, group="nominal") for entry in manifest["test"]]
    else:
        for apparatus in manifest["test"]:
            for query_index, (path, truth) in enumerate(zip(apparatus["queries"], apparatus["query_truth"])):
                entries.append({"path": path, "truth": truth, "apparatus_id": apparatus["apparatus_id"],
                                "group": apparatus["group"], "query_index": query_index,
                                "apparatus_truth": apparatus["truth"]})
    programs, parameters, metadata = [], [], []
    theta_cache = {}
    for index, entry in enumerate(entries):
        with np.load(root / entry["path"], allow_pickle=False) as episode:
            t, force = episode["t"], episode["force"].reshape(-1)
            if len(t) != len(force) + 1 or not np.allclose(np.diff(t), .01, atol=1e-10):
                raise ValueError(f"Unexpected dense time/action convention in {entry['path']}")
            program = np.zeros(400) if passive else episode["force_program"].reshape(-1).copy()
            if not passive and not np.array_equal(force, program[:len(force)]):
                raise ValueError("Query forces differ from the prechosen open-loop program")
            metadata.append({**entry, "query_index": index, "valid_length": len(t), "recorded_intervals": len(force),
                             "reset_mode": str(episode["reset_mode"].item()),
                             "collection_family": str(episode["collection_family"].item()),
                             "termination_reason": str(episode["termination_reason"].item())})
        if passive or not oracle:
            theta = np.array([1., .25])
        else:
            truth_path = entry["apparatus_truth"]
            if truth_path not in theta_cache:
                with np.load(root / truth_path, allow_pickle=False) as truth:
                    theta_cache[truth_path] = truth["theta_true"].copy()
            theta = theta_cache[truth_path]
        programs.append(program)
        parameters.append(theta)
    return np.stack(programs), np.stack(parameters), metadata


@torch.no_grad()
def evaluate_model(encoder, predictor, root, metadata, programs, theta, device,
                   batch_size=64, shuffled=None, *, passive=False):
    """Save padded arrays with an explicit target-valid mask; RGB stays bounded."""
    count = len(metadata)
    names = ["correct", "persistence", "one_step", "one_step_persistence"]
    if not passive:
        names += ["shuffled", "zero", "one_step_shuffled", "one_step_zero"]
    bundle = {name: np.full((count, STEPS, 32), np.nan, dtype=np.float32) for name in names}
    bundle["encoded"] = np.full((count, len(ENDPOINTS), 32), np.nan, dtype=np.float32)
    bundle["valid"] = np.zeros((count, STEPS), dtype=bool)
    for index, row in enumerate(metadata):
        with np.load(root / row["path"], allow_pickle=False) as saved:
            frames = saved["rgb"]
        available = ENDPOINTS[ENDPOINTS < len(frames)]
        codes = encode_endpoints(encoder, frames, device, batch_size, available)[None]
        del frames
        bundle["encoded"][index, :len(available)] = codes[0].cpu().numpy()
        if len(available) < 3:
            continue
        n = len(available) - 3
        bundle["valid"][index, :n] = True
        warm = codes[:, :3]
        pair = torch.as_tensor(theta[index:index + 1], device=device, dtype=torch.float32)
        actions = torch.as_tensor(programs[index:index + 1], device=device, dtype=torch.float32)
        choices = {"correct": actions}
        if not passive:
            choices.update(shuffled=torch.as_tensor(shuffled[index:index + 1], device=device, dtype=torch.float32),
                           zero=torch.zeros_like(actions))
        for condition, forces in choices.items():
            bundle[condition][index] = rollout_latents(predictor, warm, forces, pair, passive=passive)[0].cpu().numpy()
            one_step_key = "one_step" if condition == "correct" else f"one_step_{condition}"
            bundle[one_step_key][index, :n] = one_step_latents(predictor, codes, forces, pair, passive=passive)[0].cpu().numpy()
        bundle["persistence"][index] = persistence_predictions(warm)[0].cpu().numpy()
        bundle["one_step_persistence"][index, :n] = codes[0, 2:-1].cpu().numpy()
    return bundle


def run(config, device_name="auto", output=None, batch_size=64, shuffle_seed=None, *, checkpoints):
    paths = validate_checkpoints(config, checkpoints)
    config = deepcopy(config)
    passive = config.get("dataset") == "passive"
    seed = config["seed"] if shuffle_seed is None else shuffle_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    torch.set_num_threads(config.get("evaluation", {}).get("cpu_threads", 4))
    root = dataset_path(config)
    programs, theta, metadata = load_queries(root)
    shuffled, donors = (None, None) if passive else shuffle_action_programs(programs, metadata, seed)
    with run_directory(config["paths"]["results"], f"{config['dataset']}-latent-eval-seed{config['seed']}", run_dir=output) as destination:
        provenance = write_provenance(destination, config, checkpoints=paths,
            extra={"kind": "latent_evaluation", "weights": "raw", "split": "test", "device": str(device),
                   "shuffle_seed": seed, "parameter_condition": "fixed_nominal" if passive else "true_test_parameters"})
        settings = {"dataset": config["dataset"], "config": config, "record_dt": .01, "latent_dt": .1,
                    "manifest_sha256": file_digest(root / "manifest.json"),
                    "endpoints": ENDPOINTS.tolist(), "forecast_start": FORECAST_START,
                    "horizons_s": list(HORIZONS), "queries": len(metadata), "shuffle_seed": seed,
                    "shuffle": "Whole prechosen programs deranged within apparatus, including context forces",
                    "parameter_condition": "none" if passive else "true_test_parameters", "checkpoints": {}}
        inputs = {"forces": programs, "theta": theta, "endpoints": ENDPOINTS}
        if not passive:
            inputs.update(shuffled_forces=shuffled, shuffle_source_indices=donors)
        np.savez_compressed(destination / "inputs.npz", **inputs)
        (destination / "queries.json").write_text(json.dumps(metadata, indent=2) + "\n")
        bundles = {}
        for variant, path in paths.items():
            print(f"Evaluating {variant}: {len(metadata)} held-out {config['dataset']} episodes", flush=True)
            encoder, predictor, info = load_networks(path, device)
            before = (network_digest(encoder), network_digest(predictor))
            bundles[variant] = evaluate_model(encoder, predictor, root, metadata, programs, theta, device,
                                               batch_size, shuffled, passive=passive)
            info["parameters_and_buffers_unchanged"] = before == (network_digest(encoder), network_digest(predictor))
            assert info["parameters_and_buffers_unchanged"]
            settings["checkpoints"][variant] = info
            folder = destination / variant
            folder.mkdir()
            np.savez_compressed(folder / "latents.npz", **bundles[variant])
        (destination / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        from pi_jepa.latent_reporting import write_report
        write_report(bundles, metadata, settings, destination)
        unchanged = all(file_digest(path) == provenance["checkpoints"][name]["sha256"] for name, path in paths.items())
        (destination / "checkpoint_integrity.json").write_text(json.dumps({"unchanged": unchanged}) + "\n")
        if not unchanged:
            raise RuntimeError("Checkpoint changed during evaluation; use immutable snapshots")
    print(f"Evaluation complete: {destination}", flush=True)
    return destination


def add_arguments(parser):
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--dataset", choices=("passive", "controlled"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--output-root", default=os.environ.get("EVAL_ROOT"))
    parser.add_argument("--joint-checkpoint", "--joint")
    parser.add_argument("--posthoc-checkpoint", "--posthoc")
    parser.add_argument("--checkpoint", help="Single checkpoint: passive JEPA/PI-JEPA or controlled joint model")


def parse_config(args, parser):
    config = json.loads(Path(args.config).read_text())
    if args.dataset:
        config["dataset"] = args.dataset
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        config.setdefault("evaluation", {})["cpu_threads"] = args.cpu_threads
    if args.data_root:
        config["paths"]["data"] = args.data_root
    if args.output_root:
        config["paths"]["results"] = args.output_root
    paths = {name: path for name, path in (("joint", args.joint_checkpoint), ("posthoc", args.posthoc_checkpoint)) if path}
    if args.checkpoint:
        if paths:
            parser.error("--checkpoint cannot be combined with named model checkpoints")
        paths = {"passive" if config.get("dataset") == "passive" else "joint": args.checkpoint}
    if not paths:
        parser.error("Provide --checkpoint, --joint-checkpoint, and/or --posthoc-checkpoint")
    return config, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    config, paths = parse_config(args, parser)
    run(config, args.device, args.output, args.batch_size, args.shuffle_seed, checkpoints=paths)


if __name__ == "__main__":
    main()
