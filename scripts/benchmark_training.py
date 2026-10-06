"""Bounded real-data joint updates: measure actual batch memory and stage times.

No W&B, checkpoints, or updates to an existing model. Run on the allocated GPU
before committing that allocation to training; the warmup model is discarded.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from pi_jepa.data import TrainDataset, dataset_root
from pi_jepa.initial_conditions import load_fixed_training_initial_conditions
from pi_jepa.losses import SIGReg, physics_loss, simulate_window
from pi_jepa.models import Encoder, PassivePredictor, Predictor, PhysicalReadout, model_kwargs, model_size
from pi_jepa.runs import file_digest, run_directory, write_provenance
from pi_jepa.train import batch_from, check_dataset_interface, encode_batch, load_config, parameter_groups, prediction_loss
from pi_jepa.training_runtime import configure_runtime, neural_autocast


def benchmark(config, device, output, steps=3, warmup=1, expected_gpu=None, min_gpu_memory_gib=0):
    if steps < 1 or warmup < 0:
        raise ValueError("Benchmark needs positive measured steps and nonnegative warmup")
    if config["training"]["initial_conditions"] != "true_fixed":
        raise ValueError("HPC benchmark requires the true-fixed joint diagnostic")
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    precision = configure_runtime(config, device)
    check_dataset_interface(config)
    gpu, memory = None, None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu, memory = properties.name, properties.total_memory / 2**30
        if expected_gpu and expected_gpu.casefold() not in gpu.casefold():
            raise ValueError(f"Expected {expected_gpu}, allocation provides {gpu}")
        if memory < min_gpu_memory_gib:
            raise ValueError(f"Allocation has {memory:.1f} GiB, requires at least {min_gpu_memory_gib:g} GiB")
    elif expected_gpu or min_gpu_memory_gib:
        raise ValueError("GPU requirements cannot be checked on a CPU benchmark")
    torch.manual_seed(config["seed"])
    kwargs = model_kwargs(config)
    encoder = Encoder(**kwargs).to(device)
    predictor = (PassivePredictor(**kwargs) if config["dataset"] == "passive" else Predictor(**kwargs)).to(device)
    readout = PhysicalReadout().to(device)
    table, metadata = load_fixed_training_initial_conditions(dataset_root(config))
    target_cache = None
    if config["training"].get("cache_fixed_targets", False):
        from pi_jepa.training_cache import FixedTargetCache
        target_cache = FixedTargetCache(dataset_root(config), config["paths"]["cache"], table, metadata)
    table = table.to(device)
    dataset = TrainDataset(dataset_root(config), "train", cache_root=(config["paths"]["cache"]
                           if config["training"].get("cache_learning", False) else None))
    batch_size = config["training"]["batch_size"]
    if batch_size > len(dataset) or batch_size < 2:
        raise ValueError("Benchmark batch must contain at least two distinct available episodes")
    optimizer = torch.optim.AdamW(parameter_groups(encoder, predictor, readout, "joint",
        config["training"]["weight_decay"]), lr=config["training"]["lr"])
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    sigreg = SIGReg().to(device)
    sample_rng = torch.Generator().manual_seed(config["seed"] + 2)
    sig_rng = torch.Generator().manual_seed(config["seed"] + 3)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    warmup_allocated = warmup_reserved = 0.
    for update in range(warmup + steps):
        if update == warmup and device.type == "cuda":
            warmup_allocated = torch.cuda.max_memory_allocated(device) / 2**30
            warmup_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            torch.cuda.reset_peak_memory_stats(device)
        synchronize()
        start = time.perf_counter()
        ids = torch.randperm(len(dataset), generator=sample_rng)[:batch_size]
        batch = batch_from(dataset, ids, device, generator=sample_rng, target_cache=target_cache)
        synchronize()
        loaded = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with neural_autocast(device, precision):
            z = encode_batch(encoder, batch["frames"]).float()
            jepa, pred, reg = prediction_loss(z, predictor, batch, sigreg, sig_rng)
        decoded = readout(z)
        synchronize()
        neural = time.perf_counter()
        targets = (batch["simulated_states"]
                   if target_cache is not None else simulate_window(table(batch["trajectory_id"]),
                       batch["theta"], batch["prefix_forces"], batch["raw_endpoints"]))
        physical = physics_loss(decoded, targets)
        loss = jepa + config["training"]["lambda_phys"] * physical
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite benchmark objective")
        synchronize()
        physics_done = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config["training"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        synchronize()
        finished = time.perf_counter()
        if update >= warmup:
            rows.append({"data_seconds": loaded - start, "neural_forward_seconds": neural - loaded,
                "physics_seconds": physics_done - neural, "backward_optimizer_seconds": finished - physics_done,
                "update_seconds": finished - start, "loss": float(loss.detach()),
                "episodes_per_second": batch_size / (finished - start)})
        print(f"Benchmark update {update + 1}/{warmup + steps}: {finished-start:.3f}s", flush=True)
    result = {"dataset": config["dataset"], "model_size": model_size(config), "batch_size": batch_size,
        "precision": precision, "device": str(device), "gpu": gpu, "gpu_total_memory_gib": memory,
        "warmup_updates": warmup, "measured_updates": steps, "rows": rows,
        "median": {key: statistics.median(row[key] for row in rows) for key in rows[0]},
        "cuda_peak_allocated_gib": max(warmup_allocated, torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None,
        "cuda_peak_reserved_gib": max(warmup_reserved, torch.cuda.max_memory_reserved(device) / 2**30) if device.type == "cuda" else None,
        "warmup_peak_allocated_gib": warmup_allocated if device.type == "cuda" else None,
        "learning_cache": config["training"].get("cache_learning", False), "fixed_target_cache": target_cache is not None,
        "parameters": {name: sum(p.numel() for p in model.parameters())
                       for name, model in (("encoder", encoder), ("predictor", predictor), ("readout", readout))},
        "data_manifest_sha256": file_digest(dataset_root(config) / "manifest.json"),
        "excludes": "One-time cache preparation, startup, validation, diagnostic media, checkpoint I/O; timings synchronize CUDA at stage boundaries. Cached target lookup is included in data_seconds. This is a fit/speed check, not training convergence."}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("passive", "controlled"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--precision", choices=("float32", "bf16"), help="Explicit override for checks on other hardware")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--expected-gpu")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=0)
    parser.add_argument("--output-root", default="workspace/checks")
    args = parser.parse_args()
    config = load_config(args.config)
    config["dataset"] = args.dataset
    config["paths"]["data"] = str(Path(args.data_root).resolve())
    if args.cache_root:
        config["paths"]["cache"] = str(Path(args.cache_root).resolve())
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.precision is not None:
        config["training"]["precision"] = args.precision
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    config["training"]["cpu_threads"] = args.cpu_threads
    config["wandb"] = {"enabled": False}
    prefix = f"benchmark-{args.dataset}-{model_size(config)}-b{config['training']['batch_size']}"
    with run_directory(args.output_root, prefix) as output:
        write_provenance(output, config, extra={"purpose": "HPC batch memory and throughput preflight"})
        try:
            result = benchmark(config, args.device, output, args.steps, args.warmup,
                               args.expected_gpu, args.min_gpu_memory_gib)
        except Exception as error:
            (output / "failure.json").write_text(json.dumps({"error": type(error).__name__, "message": str(error)}, indent=2) + "\n")
            raise
        print(json.dumps({"output": str(output), "gpu": result["gpu"], "median": result["median"],
                          "cuda_peak_allocated_gib": result["cuda_peak_allocated_gib"]}, indent=2))


if __name__ == "__main__":
    main()
