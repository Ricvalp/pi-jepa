"""Joint/post-hoc training, plus an explicit training-reset oracle diagnostic."""
import argparse
import csv
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

from pi_jepa.data import TrainDataset, action_blocks, causal_clips, dataset_root, validate_clocks, validate_manifest, ENDPOINTS, WINDOW_FRAMES
from pi_jepa.checkpoint_interface import (CHECKPOINT_FORMAT_VERSION,
    model_interface, geometry_interface, validate_checkpoint_interface)
from pi_jepa.initial_conditions import load_fixed_training_initial_conditions
from pi_jepa.losses import SIGReg, jepa_loss, physics_loss, simulate_window
from pi_jepa.models import (Encoder, PhysicalReadout, Predictor, PassivePredictor,
                           scale_readout, encode_temporal, model_kwargs, model_size)
from pi_jepa.training_runtime import configure_runtime, neural_autocast
from pi_jepa.experiment_logging import TrainingLogger, preserve_rng
from pi_jepa.training_diagnostics import gradient_metrics, readout_metrics, representation_metrics
from pi_jepa.training_monitoring import evaluate_diagnostics, fixed_reference, parameter_probes, update_metrics
from pi_jepa.trajectory_diagnostics import fixed_trajectory_reference, trajectory_phase_figure
from pi_jepa.runs import dataset_identity, file_digest, nonsecret, run_directory, write_provenance



def load_config(path="configs/base.json"):
    return json.loads(Path(path).read_text())


def same_experiment(first, second):
    """Compare scientific settings, independently of storage and monitoring."""
    def science(config):
        config = copy.deepcopy(config)
        config.setdefault("training", {}).setdefault("initial_conditions", "learned")
        config["training"].setdefault("precision", "float32")
        config.setdefault("model", {}).setdefault("size", "small")
        for key in ("paths", "wandb", "diagnostics"):
            config.pop(key, None)
        for key in ("save_every", "log_every", "val_every", "cpu_threads", "cache_batch_size",
                    "cache_learning", "cache_fixed_targets", "cudnn_benchmark"):
            config.get("training", {}).pop(key, None)
        return config
    return science(first) == science(second)


def device_for(name="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


class InitialConditions(nn.Module):
    """One unknown reset state per trajectory, initialized from allowed metadata."""
    def __init__(self, reset_modes):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(len(reset_modes), 3))
        self.register_buffer("reset_modes", torch.as_tensor(reset_modes, dtype=torch.float32))

    def forward(self, ids):
        a, d, c = self.raw[ids].unbind(-1)
        return torch.stack((torch.zeros_like(a), a.tanh(),
                            math.pi * (self.reset_modes[ids] == 2) + math.pi * d.tanh(),
                            3 * c.tanh()), dim=-1)


def parameter_groups(encoder, predictor, readout, mode, weight_decay):
    decay, no_decay = [], []
    if mode != "readout":
        for network in (encoder, predictor):
            for name, value in network.named_parameters():
                # Position embeddings are coordinates, not matrix/kernel weights.
                (decay if value.ndim >= 2 and name.endswith("weight") else no_decay).append(value)
    if mode != "jepa":
        no_decay.extend(readout.parameters())
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def model_lr(step, total, cfg):
    warmup = cfg["warmup"]
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    fraction = (step - warmup) / max(1, total - warmup - 1)
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * fraction))


def initial_weights(config, output, mode):
    """Seed-matched initialization, stored inside each independently owned run."""
    path = output / "initial.pt"
    if not path.exists():
        torch.manual_seed(config["seed"])
        passive = config.get("dataset", "controlled") == "passive"
        kwargs = model_kwargs(config)
        state = {"encoder": Encoder(**kwargs).state_dict(),
                 "predictor": (PassivePredictor(**kwargs) if passive else Predictor(**kwargs)).state_dict(),
                 "readout": None if passive and mode == "jepa" else PhysicalReadout().state_dict(),
                 "seed": config["seed"], "model_size": model_size(config)}
        torch.save(state, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["seed"] != config["seed"]:
        raise ValueError("Initial weights seed differs from the run configuration")
    if state["model_size"] != model_size(config):
        raise ValueError("Initial weights model size differs from the run configuration")
    return state


def load_learning_data(root, split):
    """Keep only a bounded uint8 episode cache; never materialize the full corpus."""
    return TrainDataset(root, split)


def batch_from(data, ids, device, generator=None, starts=None, target_cache=None):
    """One uniformly sampled valid 65-frame window per independent episode.

    Reset IDs and raw endpoints are solver metadata, never neural inputs. The
    force prefix begins at the actual episode reset for the physics objective.
    """
    items = [data[int(index)] for index in ids]
    if starts is None:
        starts = [int(torch.randint(len(item["frames"]) - WINDOW_FRAMES + 1, (),
                                    generator=generator)) for item in items]
    endpoints = torch.tensor(starts)[:, None] + torch.tensor(ENDPOINTS)[None]
    prefix = torch.zeros(len(items), int(endpoints.max()), dtype=torch.float32)
    for row, (item, stop) in enumerate(zip(items, endpoints[:, -1])):
        prefix[row, :stop] = torch.as_tensor(item["forces"][:stop], dtype=torch.float32).flatten()
    result = {
        "frames": torch.stack([torch.as_tensor(item["frames"][start:start + WINDOW_FRAMES])
                               for item, start in zip(items, starts)]),
        "forces": torch.stack([torch.as_tensor(item["forces"][start:start + WINDOW_FRAMES - 1], dtype=torch.float32).flatten()
                               for item, start in zip(items, starts)]),
        "prefix_forces": prefix,
        "raw_endpoints": endpoints,
        "window_start": torch.tensor(starts),
        "trajectory_id": torch.tensor([item["trajectory_id"] for item in items]),
        "reset_mode": torch.tensor([item["reset_mode"] for item in items]),
    }
    if "theta" in items[0]:
        result["theta"] = torch.stack([torch.as_tensor(item["theta"], dtype=torch.float32) for item in items])
    elif data.dataset == "passive":
        # Fixed apparatus constants belong only to the physical simulator. The
        # autonomous latent predictor still receives neither theta nor forces.
        result["theta"] = torch.tensor([1.0, 0.25]).expand(len(items), -1).clone()
    # IDs can be opaque strings. Use equality-only local grouping for diagnostics.
    apparatus = {value: index for index, value in enumerate(dict.fromkeys(item["apparatus_id"] for item in items))}
    result["apparatus_id"] = torch.tensor([apparatus[item["apparatus_id"]] for item in items])
    if target_cache is not None:
        # Lookup before copying metadata to CUDA avoids a device-to-host roundtrip.
        result["simulated_states"] = target_cache.gather(result["trajectory_id"], result["raw_endpoints"])
    return {key: value.to(device) for key, value in result.items()}


def encode_batch(encoder, frames):
    return encode_temporal(encoder, causal_clips(frames))


def prediction_loss(z, predictor, batch, sigreg, generator):
    passive = getattr(predictor, "action_free", False)
    return jepa_loss(z, predictor, None if passive else action_blocks(batch["forces"]),
                     None if passive else batch["theta"], sigreg, generator)


@torch.no_grad()
def validate(encoder, predictor, readout, data, sigreg, config, device, mode):
    modules = [module for module in (encoder, predictor, readout) if module is not None]
    previous_modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        batch = {key: value.to(device) for key, value in data.items()}
        chunks = [encode_batch(encoder, part) for part in batch["frames"].split(config["training"]["cache_batch_size"])]
        z = torch.cat(chunks)
        rng = torch.Generator().manual_seed(config["seed"] + 100)
        jepa, pred, reg = prediction_loss(z, predictor, batch, sigreg, rng)
        result = {"val_jepa": float(jepa), "val_pred": float(pred), "val_sigreg": float(reg)}
        if mode != "jepa":
            # Validation has no optimized table: clearly label this prior residual.
            prior = InitialConditions(batch["reset_mode"].cpu()).to(device)
            simulated = simulate_window(prior(torch.arange(len(z), device=device)),
                                        batch["theta"], batch["prefix_forces"], batch["raw_endpoints"])
            result["val_physics_reset_prior"] = float(physics_loss(readout(z), simulated))
        return result
    finally:
        for module, training in zip(modules, previous_modes):
            module.train(training)


def same_state(network, snapshot):
    return all(torch.equal(value.detach().cpu(), snapshot[key]) for key, value in network.state_dict().items())


def check_dataset_interface(config):
    """Reject stale clocks/configuration before creating a training run."""
    expected = validate_clocks(config)
    manifest = json.loads((dataset_root(config) / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest.get("dataset") != config.get("dataset", "controlled"):
        raise ValueError("Training requires the selected dense corpus")
    saved_config = manifest.get("generation_config", {})
    if "physics" not in saved_config or "data" not in saved_config:
        raise ValueError("Dataset manifest lacks its generation clock configuration")
    actual = validate_clocks(saved_config)
    if actual != expected or manifest.get("clocks") != actual:
        raise ValueError("Dataset manifest clocks differ from the training configuration")


def validate_reset_checkpoint(checkpoint, reset_metadata, fixed_table):
    """Check reset protocol/source before restoring or returning a saved run."""
    if checkpoint.get("initial_conditions_metadata") != reset_metadata:
        raise ValueError("Resume initial-condition protocol or reset source differs")
    if fixed_table is not None:
        stored = checkpoint.get("initial_conditions", {})
        if not isinstance(stored, dict) or any(key not in stored or not torch.equal(value, stored[key])
                for key, value in fixed_table.state_dict().items()):
            raise ValueError("Resume fixed reset states differ from the training reset source")
        if checkpoint.get("initial_optimizer") is not None:
            raise ValueError("A fixed-reset diagnostic cannot have an initial-state optimizer")


def train(config, mode, device="auto", resume=False, *, run_dir=None, pretrained=None, quiet=False):
    """Train one stage in a fresh timestamped run; resume requires its exact path."""
    if mode not in ("joint", "jepa", "readout"):
        raise ValueError("mode must be joint, jepa, or readout")
    if mode == "readout" and pretrained is None:
        raise ValueError("Readout training requires --pretrained with a final JEPA checkpoint")
    if pretrained is not None and mode != "readout":
        raise ValueError("--pretrained is only used by the frozen readout stage")
    config = copy.deepcopy(config)
    model_size(config)  # Validate before creating a run directory.
    protocol = config.setdefault("training", {}).setdefault("initial_conditions", "learned")
    config["training"].setdefault("precision", "float32")
    if config["training"].get("cache_fixed_targets", False) and protocol != "true_fixed":
        raise ValueError("Cached simulator targets require fixed true training resets")
    if config["training"].get("cache_learning") or config["training"].get("cache_fixed_targets"):
        config["paths"].setdefault("cache", "workspace/cache/training")
    if protocol not in ("learned", "true_fixed"):
        raise ValueError("training.initial_conditions must be learned or true_fixed")
    if protocol == "true_fixed" and mode != "joint":
        raise ValueError("true_fixed initial conditions require --mode joint (oracle diagnostic)")
    if mode == "joint" and config["training"]["jepa_updates"] != config["training"]["physical_updates"]:
        raise ValueError("Joint updates must match both requested update budgets.")
    if config.get("dataset", "controlled") == "passive" and mode == "readout":
        raise ValueError("The passive corpus supports --mode jepa or --mode joint")
    for key in config["paths"]:
        config["paths"][key] = str(Path(config["paths"][key]).resolve())
    check_dataset_interface(config)
    fixed_table = None
    reset_metadata = {"mode": "none" if mode == "jepa" else "learned",
        "source": "none" if mode == "jepa" else "commanded_position_and_coarse_reset_mode",
        "diagnostic_oracle": False, "neural_input": False}
    if protocol == "true_fixed":
        fixed_table, reset_metadata = load_fixed_training_initial_conditions(dataset_root(config))
    diagnostic = "-true-fixed-reset-diagnostic" if protocol == "true_fixed" else ""
    prefix = f"{config.get('dataset', 'controlled')}-{mode}-{model_size(config)}{diagnostic}-seed{config['seed']}"
    with run_directory(config["paths"]["runs"], prefix, run_dir, resume) as output:
        if resume:
            saved = json.loads((output / "config.json").read_text())
            if not same_experiment(saved, config):
                raise ValueError("Resume scientific configuration differs from the original run")
            if not (output / "latest.pt").exists():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {output / 'latest.pt'}")
            identity = json.loads((output / "provenance.json").read_text())
            if identity.get("stage") != mode:
                raise ValueError("Resume stage differs from the original run")
            checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            validate_checkpoint_interface(checkpoint)
            total = config["training"]["physical_updates" if mode == "readout" else "jepa_updates"]
            if checkpoint["mode"] != mode or not same_experiment(checkpoint["config"], config) or not 0 <= checkpoint["step"] <= total:
                raise ValueError("Resume checkpoint/config mismatch")
            validate_reset_checkpoint(checkpoint, reset_metadata, fixed_table)
            if (output / "final.pt").exists():
                final = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
                validate_checkpoint_interface(final)
                if final["mode"] != mode or final["step"] != total or not same_experiment(final["config"], config):
                    raise ValueError("Final checkpoint does not match the completed stage")
                validate_reset_checkpoint(final, reset_metadata, fixed_table)
                del final
            del checkpoint
        sources = {"pretrained_jepa": pretrained} if pretrained is not None else None
        provenance = write_provenance(output, config, sources,
            {"stage": mode, "initial_conditions": reset_metadata,
             "diagnostic_oracle_training_resets": protocol == "true_fixed"}, resume)
        if not quiet:
            print(f"Run directory: {output.resolve()}", flush=True)
        return _train(config, mode, device, resume, output, pretrained, quiet, provenance,
                      fixed_table=fixed_table, reset_metadata=reset_metadata)


def _train(config, mode, device, resume, output, pretrained, quiet, provenance, *, fixed_table=None, reset_metadata=None):
    cfg = config["training"]
    torch.set_num_threads(cfg["cpu_threads"])
    device = device_for(device)
    precision = configure_runtime(config, device)
    if mode == "joint" and cfg["jepa_updates"] != cfg["physical_updates"]:
        raise ValueError("Joint updates must match both requested update budgets.")
    checkpoint_path = output / "latest.pt"
    total = cfg["physical_updates"] if mode == "readout" else cfg["jepa_updates"]
    if resume and (output / "final.pt").exists():
        if not quiet:
            print(f"{mode}: final checkpoint already present; skipping completed stage.", flush=True)
        return output / "final.pt"
    initial = initial_weights(config, output, mode)
    initial_digest = file_digest(output / "initial.pt")
    torch.manual_seed(config["seed"] + 1)
    passive = config.get("dataset", "controlled") == "passive"
    kwargs = model_kwargs(config)
    encoder = Encoder(**kwargs).to(device)
    predictor = (PassivePredictor(**kwargs) if passive else Predictor(**kwargs)).to(device)
    readout = None if passive and mode == "jepa" else PhysicalReadout().to(device)
    for name, network in (("encoder", encoder), ("predictor", predictor), ("readout", readout)):
        if network is not None:
            network.load_state_dict(initial[name])
    pretrained_path = Path(pretrained) if pretrained is not None else None
    if mode == "readout":
        pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        validate_checkpoint_interface(pretrained)
        if pretrained["mode"] != "jepa" or pretrained["step"] != cfg["jepa_updates"] or not same_experiment(pretrained["config"], config):
            raise ValueError("Frozen readout requires this configuration's final JEPA checkpoint.")
        source_data = pretrained.get("data_manifest_sha256")
        if source_data is None:
            source_data = dataset_identity(pretrained["config"])["sha256"]
        if source_data != provenance["data"]["sha256"]:
            raise ValueError("Frozen readout dataset differs from the pretrained JEPA dataset")
        encoder.load_state_dict(pretrained["encoder"])
        predictor.load_state_dict(pretrained["predictor"])
        for network in (encoder, predictor):
            network.requires_grad_(False).eval()
    else:
        encoder.train(); predictor.train()
    if readout is not None:
        readout.train(mode != "jepa").requires_grad_(mode != "jepa")
    if not quiet:
        print(f"Opening learning-only {config.get('dataset', 'controlled')} episodes with a bounded cache ...", flush=True)
        if fixed_table is not None:
            print("Oracle diagnostic: physical targets use fixed true training resets; neural inputs remain images/actions.", flush=True)
    if cfg.get("cache_learning", False):
        learning = TrainDataset(dataset_root(config), "train", cache_root=config["paths"]["cache"])
        validation_data = TrainDataset(dataset_root(config), "validation", cache_root=config["paths"]["cache"])
    else:
        learning = load_learning_data(dataset_root(config), "train")
        validation_data = load_learning_data(dataset_root(config), "validation")
    validation = fixed_reference(validation_data, seed=config["seed"] + 102, max_episodes=96)
    if len(learning) < cfg["batch_size"]:
        raise ValueError("SIGReg requires the configured number of distinct trajectories per batch")
    table = (None if mode == "jepa" else
             (fixed_table if fixed_table is not None else InitialConditions(learning.reset_modes)).to(device))
    target_cache = None
    if cfg.get("cache_fixed_targets", False):
        from pi_jepa.training_cache import FixedTargetCache
        target_cache = FixedTargetCache(dataset_root(config), config["paths"]["cache"],
                                       table, reset_metadata)
        table.target_cache = target_cache
    (output / "runtime.json").write_text(json.dumps({
        "device": str(device), "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_precision": precision, "weights_precision": "float32", "readout_precision": "float32",
        "cudnn_benchmark": bool(cfg.get("cudnn_benchmark", False)),
        "learning_cache": bool(cfg.get("cache_learning", False)),
        "fixed_target_cache": target_cache.metadata if target_cache is not None else None,
        "parameters": {name: sum(p.numel() for p in network.parameters()) if network is not None else 0
                       for name, network in (("encoder", encoder), ("predictor", predictor), ("readout", readout))},
    }, indent=2) + "\n")
    learned_resets = table is not None and not getattr(table, "is_fixed", False)
    optimizer = torch.optim.AdamW(parameter_groups(encoder, predictor, readout, mode, cfg["weight_decay"]), lr=cfg["lr"])
    initial_optimizer = torch.optim.Adam(table.parameters(), lr=cfg["initial_lr"]) if learned_resets else None
    sample_rng = torch.Generator().manual_seed(config["seed"] + 2)
    sig_rng = torch.Generator().manual_seed(config["seed"] + 3)
    sigreg = SIGReg().to(device)
    start = 0
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not same_experiment(checkpoint["config"], config) or checkpoint["mode"] != mode:
            raise ValueError("Resume checkpoint/config mismatch")
        for name, network in (("encoder", encoder), ("predictor", predictor), ("readout", readout)):
            if network is not None:
                network.load_state_dict(checkpoint[name])
        if table is not None:
            table.load_state_dict(checkpoint["initial_conditions"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if initial_optimizer is not None:
            initial_optimizer.load_state_dict(checkpoint["initial_optimizer"])
        sample_rng.set_state(checkpoint["sample_rng"]); sig_rng.set_state(checkpoint["sig_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start = checkpoint["step"]
    frozen = [{key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
              for network in (encoder, predictor)] if mode == "readout" else None
    monitoring = config.get("diagnostics", {}).get("enabled", config.get("wandb", {}).get("enabled", False))
    monitor_cfg = {**config.get("wandb", {}), **config.get("diagnostics", {})}
    log_every = monitor_cfg.get("log_every", cfg["log_every"])
    diagnostics_every = monitor_cfg.get("diagnostics_every", cfg["val_every"])
    media_every = monitor_cfg.get("media_every", 1000)
    if min(log_every, diagnostics_every, media_every) < 1:
        raise ValueError("Logging/diagnostic/media intervals must be positive")
    reference = fixed_reference(learning, seed=config["seed"] + 101) if monitoring else None
    trajectories = fixed_trajectory_reference(learning) if monitoring and mode != "jepa" else []
    diagnostic_windows = {}
    for name, batch, seed in (("train_eval", reference, config["seed"] + 101),
                              ("validation", validation, config["seed"] + 102)):
        if batch is not None:
            diagnostic_windows[name] = {"seed": seed, "episodes": len(batch["frames"]),
                **{key: batch[key].tolist() for key in ("trajectory_id", "window_start", "raw_endpoints")}}
    if trajectories:
        diagnostic_windows["training_trajectories"] = {
            "selection": "first training episode of each available reset mode in manifest order",
            "forecast_start_record": 34, "max_forecast_steps": 32,
            "episodes": [{"trajectory_id": int(item["trajectory_id"]),
                          "reset_mode": int(item["reset_mode"]),
                          "available_frames": int(item["available_frames"]),
                          "forecast_steps": int(item["steps"]),
                          "raw_endpoints": item["endpoints"].tolist()} for item in trajectories],
        }
    (output / "diagnostic_windows.json").write_text(json.dumps(diagnostic_windows, indent=2) + "\n")
    networks = [encoder, predictor] if mode == "jepa" else ([readout] if mode == "readout" else [encoder, predictor, readout])
    neural_parameters = [p for network in networks for p in network.parameters() if p.requires_grad]
    probes = parameter_probes(encoder, predictor, readout, table, mode)

    def save(step, name):
        state = {"encoder": encoder.state_dict(), "predictor": predictor.state_dict(), "readout": readout.state_dict() if readout is not None else None,
                 "initial_conditions": table.state_dict() if table is not None else None,
                 "initial_conditions_metadata": reset_metadata, "optimizer": optimizer.state_dict(),
                 "initial_optimizer": initial_optimizer.state_dict() if initial_optimizer else None,
                 "step": step, "mode": mode, "config": nonsecret(config),
                 "sample_rng": sample_rng.get_state(), "sig_rng": sig_rng.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "initial_weights_sha256": initial_digest,
                 "data_manifest_sha256": provenance["data"]["sha256"],
                 "interface": {"format_version": CHECKPOINT_FORMAT_VERSION, **geometry_interface(), **model_interface(config, mode), "dataset": config.get("dataset", "controlled"), "weights": "raw", "ema": False,
                     "training_precision": precision, "weights_dtype": "float32",
                     "normalization": {"rgb": "2 * uint8 / 255 - 1", "force_divisor": None if passive else 5.0,
                         "theta_center": [1.0, 0.275], "theta_scale": [0.3, 0.225],
                         "physical_residual_divisors": [2.0, 2.0, 1.0, 1.0, 5.0], "fitted_statistics": None},
                     "clip_frames": 8, "latent_dim": 32, "latent_endpoints": ENDPOINTS, "window_frames": WINDOW_FRAMES,
                     "history_frame_stride": 2, "latent_stride_records": 10,
                     "force_block_length": None if passive else 10,
                     "integration_dt_seconds": 0.002, "force_dt_seconds": 0.02,
                     "prediction_dt_seconds": 0.1, "initial_state_origin": "episode_reset_frame_zero",
                     "state_order": ["p", "v", "q", "w"],
                     "readout_order": ["p", "v", "sin(q)", "cos(q)", "w"],
                     "theta_order": ["cart_mass_kg", "cart_friction"],
                     "simulator_theta": [1.0, 0.25] if passive else "known_per_episode_training_parameters",
                     "predictor_conditioning": "none" if passive else "force_blocks_and_theta",
                     "action_units": "newton", "frame_dt_seconds": config["physics"]["dt"]}}
        if logger.run_metadata:
            state["tracking"] = logger.run_metadata
        temporary = output / (name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(output / name)

    columns = ["step", "total", "jepa", "prediction", "sigreg", "physics", "latent_variance", "readout_variance", "lr", "seconds"]
    log_path = output / "losses.csv"
    elapsed_offset = 0.0
    if start and log_path.exists():
        with log_path.open() as stream:
            previous = [row for row in csv.DictReader(stream) if int(row["step"]) <= start]
        elapsed_offset = float(previous[-1]["seconds"]) if previous else 0.0
        with log_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(previous)
    validation_path = output / "validation.csv"
    if start and validation_path.exists():
        with validation_path.open() as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            previous_validation = [row for row in reader if int(row["step"]) <= start]
        with validation_path.open("w", newline="") as stream:
            val_writer = csv.DictWriter(stream, fieldnames=fields)
            val_writer.writeheader(); val_writer.writerows(previous_validation)
    append = start > 0 and log_path.exists()
    wall_start = time.monotonic()
    with TrainingLogger(config, mode, output, resume=resume, start_step=start) as logger, log_path.open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not append: writer.writeheader()
        if monitoring:
            with preserve_rng():
                baseline, figures, histograms = evaluate_diagnostics(encoder, predictor, readout, table,
                    reference, validation, config, device, mode, media=True)
                if trajectories:
                    figures["train_eval/physics_phase"] = trajectory_phase_figure(
                        encoder, predictor, readout, table, trajectories, config, device)
                baseline["train/resumed_from_update"] = start
                baseline["train/initial_conditions/trainable"] = float(learned_resets)
                baseline["train/initial_conditions/true_reset_supervision"] = float(fixed_table is not None)
                logger.log(baseline, start, figures=figures, histograms=histograms)
        for step in range(start, total):
            monitor_step = monitoring and (step == start or (step + 1) % log_every == 0 or step + 1 == total)
            monitor_values = {}
            update_started = time.perf_counter()
            ids = torch.randperm(len(learning), generator=sample_rng)[:cfg["batch_size"]]
            batch = batch_from(learning, ids, device, generator=sample_rng, target_cache=target_cache)
            data_seconds = time.perf_counter() - update_started
            compute_started = time.perf_counter()
            lr = model_lr(step, total, cfg)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if initial_optimizer: initial_optimizer.zero_grad(set_to_none=True)
            with neural_autocast(device, precision):
                z = encode_batch(encoder, batch["frames"]).float()
                jepa = pred = reg = phys = z.new_zeros(())
                if mode != "readout":
                    jepa, pred, reg = prediction_loss(z, predictor, batch, sigreg, sig_rng)
            decoded = readout(z) if readout is not None else None
            if mode != "jepa":
                simulated = (batch["simulated_states"]
                             if target_cache is not None else
                             simulate_window(table(batch["trajectory_id"].long()), batch["theta"],
                                             batch["prefix_forces"], batch["raw_endpoints"]))
                phys = physics_loss(decoded, simulated)
            loss = jepa + cfg["lambda_phys"] * phys
            if not torch.isfinite(loss):
                save(step, "numerical_failure.pt")
                (output / "failure.json").write_text(json.dumps({"step": step, "reason": "nonfinite objective"}))
                logger.log({"failure/nonfinite_objective": 1}, step + 1)
                raise FloatingPointError(f"{mode}: nonfinite objective at update {step + 1}; checkpoint saved")
            loss.backward()  # Both optimizers use this SAME graph and backward pass.
            if monitor_step:
                monitor_values.update(gradient_metrics({"encoder": encoder, "predictor": predictor,
                                                        "readout": readout, "initial_conditions": table}))
                before = {name: value.detach().clone() for name, value in probes.items()}
            gradient_norm = torch.nn.utils.clip_grad_norm_(neural_parameters, cfg["grad_clip"], error_if_nonfinite=True)
            if initial_optimizer and not torch.isfinite(table.raw.grad).all():
                raise FloatingPointError(f"{mode}: nonfinite initial-condition gradient")
            optimizer.step()
            if initial_optimizer: initial_optimizer.step()
            if monitor_step:
                monitor_values.update(update_metrics(probes, before))
                monitor_values.update(representation_metrics(z.detach()))
                if mode != "jepa":
                    monitor_values.update(readout_metrics(decoded.detach(), simulated.detach()))
                    if learned_resets:
                        monitor_values["initial_conditions/saturated_fraction"] = float((table.raw.detach().tanh().abs() > .95).float().mean())
                monitor_values["initial_conditions/trainable"] = float(learned_resets)
                monitor_values["initial_conditions/true_reset_supervision"] = float(fixed_table is not None)
                monitor_values.update({"gradients/neural_norm_before_clip": float(gradient_norm),
                    "gradients/clipping_applied": float(gradient_norm > cfg["grad_clip"]),
                    "active_encoder": float(mode != "readout"), "active_readout": float(mode != "jepa")})
            row = {"step": step + 1, "total": float(loss.detach()), "jepa": float(jepa.detach()),
                   "prediction": float(pred.detach()), "sigreg": float(reg.detach()), "physics": float(phys.detach()),
                   "latent_variance": float(z.detach().var(dim=0, unbiased=False).mean()),
                   "readout_variance": float(scale_readout(decoded.detach()).var(dim=0, unbiased=False).mean()) if decoded is not None else float("nan"),
                   "lr": lr, "seconds": elapsed_offset + time.monotonic() - wall_start}
            writer.writerow(row)
            if monitor_step:
                optimization_seconds = time.perf_counter() - compute_started
                duration = data_seconds + optimization_seconds
                monitor_values.update({"performance/data_seconds": data_seconds,
                    "performance/optimization_seconds": optimization_seconds,
                    "performance/update_seconds": duration,
                    "performance/episodes_per_second": cfg["batch_size"] / max(duration, 1e-9)})
                if device.type == "cuda":
                    monitor_values["performance/cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                    monitor_values["performance/cuda_reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
            payload = {f"train/{key}": value for key, value in monitor_values.items()}
            if monitor_step:
                payload.update({f"train/loss/{key}": row[key] for key in ("total", "jepa", "prediction", "sigreg", "physics")})
                payload.update({"train/lr": lr, "train/seconds": row["seconds"],
                                "train/updates_per_second": (step + 1 - start) / max(time.monotonic() - wall_start, 1e-9)})
            if (step + 1) % cfg["log_every"] == 0 or step == start:
                stream.flush()
                if not quiet:
                    print(f"{mode} {step + 1}/{total} total={row['total']:.5f} pred={row['prediction']:.5f} phys={row['physics']:.5f} var(z)={row['latent_variance']:.5f} elapsed={row['seconds']:.1f}s", flush=True)
            if (step + 1) % cfg["val_every"] == 0 or step + 1 == total:
                diagnostics = {"step": step + 1, **validate(encoder, predictor, readout, validation, sigreg, config, device, mode)}
                with (output / "validation.csv").open("a", newline="") as val_stream:
                    val_writer = csv.DictWriter(val_stream, fieldnames=list(diagnostics))
                    if val_stream.tell() == 0: val_writer.writeheader()
                    val_writer.writerow(diagnostics)
                if monitoring:
                    payload.update({f"val/loss/{key.removeprefix('val_')}": value for key, value in diagnostics.items() if key != "step"})
            figures = histograms = None
            if monitoring and ((step + 1) % diagnostics_every == 0 or (step + 1) % media_every == 0 or step + 1 == total):
                with preserve_rng():
                    media = (step + 1) % media_every == 0 or step + 1 == total
                    extra, figures, histograms = evaluate_diagnostics(encoder, predictor, readout, table,
                        reference, validation, config, device, mode,
                        media=media)
                    if media and trajectories:
                        figures["train_eval/physics_phase"] = trajectory_phase_figure(
                            encoder, predictor, readout, table, trajectories, config, device)
                payload.update(extra)
            if payload:
                logger.log(payload, step + 1, figures=figures, histograms=histograms)
            if (step + 1) % cfg["save_every"] == 0: save(step + 1, "latest.pt")
        if frozen is not None:
            assert same_state(encoder, frozen[0]) and same_state(predictor, frozen[1]), "Frozen parameters/buffers changed!"
            (output / "freeze_check.json").write_text(json.dumps({"encoder_and_predictor_unchanged": True}))
        save(total, "latest.pt"); save(total, "final.pt")
    return output / "final.pt"


def add_training_arguments(parser):
    """Common, ordinary CLI overrides used by the stage and campaign entry points."""
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=["controlled", "passive"])
    parser.add_argument("--initial-conditions", choices=["learned", "true_fixed"],
                        help="Use learned resets, or explicit fixed true training resets for a joint diagnostic")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--runs-root", default=os.environ.get("RUNS_ROOT"))
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")


def config_from_arguments(args):
    config = load_config(args.config)
    if getattr(args, "dataset", None) is not None:
        config["dataset"] = args.dataset
    if getattr(args, "initial_conditions", None) is not None:
        config["training"]["initial_conditions"] = args.initial_conditions
    for option, key in (("data_root", "data"), ("runs_root", "runs"), ("cache_root", "cache")):
        value = getattr(args, option, None)
        if value is not None:
            config["paths"][key] = value
    if getattr(args, "batch_size", None) is not None:
        if args.batch_size < 2:
            raise ValueError("--batch-size must be at least two distinct episodes")
        config["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        config["training"]["cpu_threads"] = args.cpu_threads
    for name in ("mode", "project", "entity", "group"):
        value = getattr(args, f"wandb_{name}")
        if value is not None:
            config.setdefault("wandb", {})[name] = value
            config["wandb"]["enabled"] = True
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--mode", choices=["joint", "jepa", "readout"], required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--pretrained", help="Final JEPA checkpoint required for frozen readout training")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(config_from_arguments(args), args.mode, args.device, args.resume,
          run_dir=args.run_dir, pretrained=args.pretrained, quiet=args.quiet)


if __name__ == "__main__":
    main()
