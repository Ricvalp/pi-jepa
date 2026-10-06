"""Tiny real evaluation runs: isolation, truth access, and checkpoint identity."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pi_jepa import evaluate, evaluate_latents
from pi_jepa.data import generate, CAMERA, CART_LIMIT, SCHEMA_VERSION, validate_clocks
from pi_jepa.checkpoint_interface import CHECKPOINT_FORMAT_VERSION, model_interface, geometry_interface
from pi_jepa.runs import file_digest


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1))
    def forward(self, images):
        return (images.mean((1, 2, 3)) * self.scale)[:, None].expand(-1, 32)


class TinyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.001))
    def forward(self, codes, actions, theta):
        return codes + actions.mean(-1, keepdim=True) * self.scale


class TinyReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1))
    def forward(self, codes):
        p = codes[..., 0] * self.scale
        zero = torch.zeros_like(p)
        return torch.stack((p, zero, zero, torch.ones_like(p), zero), dim=-1)


@pytest.fixture
def evaluation_inputs(tmp_path, monkeypatch):
    root = tmp_path / "data" / "controlled"
    root.mkdir(parents=True)
    paths, truths = [], []
    for i, n in enumerate((54, 400)):
        path, truth = f"query{i}.npz", f"truth{i}.npz"
        rgb = np.full((n + 1, 2, 2, 3), i * 20, dtype=np.uint8)
        np.savez(root / path, rgb=rgb, force=np.full((n, 1), i), force_program=np.full((400, 1), i),
                 t=np.arange(n + 1) * .01, reset_mode="downward", collection_family="pulse",
                 termination_reason="boundary_exit" if i == 0 else "duration")
        np.savez(root / truth, state=np.zeros((n + 1, 4)), theta_true=[1., .25])
        paths.append(path)
        truths.append(truth)
    for i in range(2):
        np.savez(root / f"calibration{i}.npz", rgb=np.zeros((65, 2, 2, 3), dtype=np.uint8),
                 force=np.zeros((64, 1)), t=np.arange(65) * .01)
    np.savez(root / "private.npz", theta_true=[1., .25])
    cfg = json.loads(Path("configs/base.json").read_text())
    cfg.update(seed=42, dataset="controlled",
               paths={"data": str(root.parent), "results": str(tmp_path / "results")},
               adaptation={"updates": 0, "lr": .03}, evaluation={"cpu_threads": 1})
    manifest = {"schema_version": SCHEMA_VERSION, "physics": cfg["physics"], "collection_settings": cfg["data"],
                "camera": CAMERA, "cart_limit_m": CART_LIMIT, "clocks": validate_clocks(cfg),
                "dataset": "controlled", "test": [{"apparatus_id": "opaque", "group": "interior",
                "queries": paths, "query_truth": truths, "truth": "private.npz",
                "calibration": ["calibration0.npz", "calibration1.npz"]}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    checkpoints = {}
    for variant, mode in (("joint", "joint"), ("posthoc", "readout")):
        path = tmp_path / f"{mode}.pt"
        torch.save({"mode": mode, "step": 100, "config": cfg, "encoder": TinyEncoder().state_dict(),
                    "predictor": TinyPredictor().state_dict(), "readout": TinyReadout().state_dict(),
                    "interface": {"format_version": CHECKPOINT_FORMAT_VERSION, **geometry_interface(),
                                  **model_interface(cfg, mode)},
                    "data_manifest_sha256": file_digest(root / "manifest.json")}, path)
        checkpoints[variant] = path
    monkeypatch.setattr(evaluate_latents, "Encoder", TinyEncoder)
    monkeypatch.setattr(evaluate_latents, "Predictor", TinyPredictor)
    monkeypatch.setattr(evaluate, "PhysicalReadout", TinyReadout)
    return cfg, checkpoints


@pytest.mark.parametrize("module", [evaluate, evaluate_latents])
def test_evaluations_record_selected_checkpoint_and_refuse_existing_output(evaluation_inputs, module):
    config, paths = evaluation_inputs
    output = module.run(config, "cpu", checkpoints={"joint": paths["joint"]})
    assert output.parent == Path(config["paths"]["results"])
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["checkpoints"]["joint"]["step"] == 100
    assert provenance["split"] == "test" and provenance["weights"] == "raw"
    assert json.loads((output / "checkpoint_integrity.json").read_text())["unchanged"]
    assert not (output / "posthoc").exists()
    with pytest.raises(FileExistsError):
        module.run(config, "cpu", checkpoints={"joint": paths["joint"]}, output=output)
    if module is evaluate_latents:
        with np.load(output / "joint" / "latents.npz") as arrays:
            assert arrays["valid"].sum(1).tolist() == [2, 32]
    else:
        with np.load(output / "joint" / "forecasts.npz") as arrays:
            assert arrays["valid"].sum(1).tolist() == [2, 32]
            assert arrays["oracle_learned"].shape == (2, 32, 4)


def test_nonoracle_outputs_exist_before_any_hidden_test_file_is_opened(evaluation_inputs, monkeypatch):
    config, paths = evaluation_inputs
    original = np.load
    accesses = []
    def guarded(path, **kwargs):
        path = Path(path)
        if path.name == "private.npz" or path.name.startswith("truth"):
            outputs = list(Path(config["paths"]["results"]).glob("*/joint/forecasts_nominal_fitted.npz"))
            assert len(outputs) == 1
            accesses.append(path.name)
        return original(path, **kwargs)
    monkeypatch.setattr(np, "load", guarded)
    evaluate.run(config, "cpu", checkpoints={"joint": paths["joint"]})
    assert accesses


@pytest.mark.parametrize("module", [evaluate, evaluate_latents])
def test_evaluation_rejects_swapped_checkpoints_and_changed_dataset(evaluation_inputs, module):
    config, paths = evaluation_inputs
    with pytest.raises(ValueError, match="requires"):
        module.run(config, "cpu", checkpoints={"joint": paths["posthoc"]})
    manifest = Path(config["paths"]["data"]) / "controlled" / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset manifest differs"):
        module.run(config, "cpu", checkpoints={"joint": paths["joint"]})


def test_checkpoint_mutation_during_evaluation_is_detected(evaluation_inputs, monkeypatch):
    config, paths = evaluation_inputs
    original = evaluate_latents.evaluate_model
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        torch.save({"corrupted": True}, paths["joint"])
        return result
    monkeypatch.setattr(evaluate_latents, "evaluate_model", mutate)
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        evaluate_latents.run(config, "cpu", checkpoints={"joint": paths["joint"]})


@pytest.mark.parametrize("mutation, message", [
    ({"format_version": 5}, "interface version 6"),
    ({"architecture": {"encoder": "resnet18_24channel_32latent"}}, "architecture differs"),
    ({"encoder_normalization": {}}, "encoder_normalization differs"),
    ({"physics": {"m": .2, "ell": .5, "g": 9.81}}, "Checkpoint physics differs"),
    ({"cart_limit_m": 2.0}, "Checkpoint cart_limit_m differs"),
])
def test_evaluation_rejects_previous_geometry_even_with_matching_manifest(evaluation_inputs, mutation, message):
    config, paths = evaluation_inputs
    saved = torch.load(paths["joint"], weights_only=True)
    saved["interface"].update(mutation)
    torch.save(saved, paths["joint"])
    with pytest.raises(ValueError, match=message):
        evaluate_latents.run(config, "cpu", checkpoints={"joint": paths["joint"]})


def test_partial_dataset_is_not_overwritten(tmp_path):
    config = json.loads(Path("configs/base.json").read_text())
    config["paths"]["data"] = str(tmp_path)
    root = tmp_path / "controlled"
    root.mkdir()
    valuable = root / "existing.npz"
    valuable.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        generate(config, "controlled")
    assert valuable.read_bytes() == b"preserve me"
