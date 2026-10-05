"""The cache changes the execution path without changing the FP32 experiment."""
import copy
import json
from pathlib import Path

import pytest
import torch

from test_fixed_resets import reset_dataset
from pi_jepa.initial_conditions import load_fixed_training_initial_conditions
from pi_jepa.training_cache import prepare_learning_cache, prepare_fixed_target_cache
from pi_jepa.train import train


@pytest.mark.parametrize("dataset", ["passive", "controlled"])
def test_prepared_caches_preserve_training_weights_rng_and_reset_states(reset_dataset, tmp_path, dataset):
    config, _ = reset_dataset
    config["dataset"] = dataset
    config["diagnostics"]["enabled"] = False
    plain = train(config, "joint", "cpu", run_dir=tmp_path / "plain", quiet=True)
    root = Path(config["paths"]["data"]) / dataset
    cache = tmp_path / "cache"
    table, metadata = load_fixed_training_initial_conditions(root)
    prepare_learning_cache(root, cache)
    prepare_fixed_target_cache(root, cache, table, metadata)
    cached_config = copy.deepcopy(config)
    cached_config["paths"]["cache"] = str(cache)
    cached_config["training"].update(cache_learning=True, cache_fixed_targets=True)
    cached_config["diagnostics"]["enabled"] = True
    cached = train(cached_config, "joint", "cpu", run_dir=tmp_path / "cached", quiet=True)
    first = torch.load(plain, map_location="cpu", weights_only=True)
    second = torch.load(cached, map_location="cpu", weights_only=True)
    for name in ("encoder", "predictor", "readout", "initial_conditions"):
        for key, value in first[name].items():
            assert torch.equal(value, second[name][key]), (name, key)
    for key in ("sample_rng", "sig_rng", "torch_rng"):
        assert torch.equal(first[key], second[key]), key
    runtime = json.loads((cached.parent / "runtime.json").read_text())
    assert runtime["learning_cache"] and runtime["fixed_target_cache"]["kind"] == "fixed_reset_simulator_targets"
    rows = [json.loads(line) for line in (cached.parent / "diagnostics.jsonl").read_text().splitlines()]
    values = rows[-1]["metrics"]
    assert values["train/performance/data_seconds"] > 0
    assert values["train/performance/optimization_seconds"] > 0
    assert values["train/performance/episodes_per_second"] > 0
    assert "train/performance/cuda_peak_allocated_gib" not in values
    assert "train_eval/physics_phase" in rows[-1]["media"]


def test_target_cache_cannot_be_used_with_moving_resets(reset_dataset, tmp_path):
    config, _ = reset_dataset
    config["training"].update(initial_conditions="learned", cache_fixed_targets=True)
    with pytest.raises(ValueError, match="require fixed true"):
        train(config, "joint", "cpu", run_dir=tmp_path / "invalid", quiet=True)
    assert not (tmp_path / "invalid").exists()
