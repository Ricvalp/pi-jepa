"""Oracle reset diagnostics are explicit, training-only, frozen, and resumable."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import pi_jepa.train as training
from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION, TrainDataset, validate_clocks
from pi_jepa.initial_conditions import (FixedInitialConditions,
    load_fixed_training_initial_conditions, load_initial_conditions_state)
from pi_jepa.losses import simulate_window
from pi_jepa.physics import rollout
from test_training_monitoring import TinyEncoder, TinyPredictor


@pytest.fixture
def reset_dataset(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(training, "Encoder", TinyEncoder)
    monkeypatch.setattr(training, "Predictor", TinyPredictor)
    config = training.load_config()
    config["dataset"] = "passive"
    config["paths"].update(data=str(tmp_path / "data"), runs=str(tmp_path / "runs"))
    config["training"].update(initial_conditions="true_fixed", batch_size=2,
        jepa_updates=2, physical_updates=2, warmup=1, save_every=1,
        log_every=1, val_every=2, cpu_threads=2, cache_batch_size=2)
    config["wandb"] = {"enabled": False}
    config["diagnostics"] = {"enabled": True, "log_every": 1,
        "diagnostics_every": 2, "media_every": 2}
    states = np.array([[0., .13, .73, -.27], [0., -.17, -1.89, .53],
                       [0., .03, 3.02, -.11], [0., -.07, 1.62, .91]])
    modes = [0, 1, 2, 1]
    rng = np.random.default_rng(31)
    for corpus in ("passive", "controlled"):
        root = Path(config["paths"]["data"]) / corpus
        (root / "train").mkdir(parents=True)
        (root / "validation").mkdir()
        (root / "truth" / "train").mkdir(parents=True)
        manifest = {"schema_version": SCHEMA_VERSION, "dataset": corpus,
            "physics": config["physics"], "collection_settings": config["data"],
            "camera": CAMERA, "cart_limit_m": CART_LIMIT,
            "generation_config": {"physics": config["physics"], "data": config["data"]},
            "clocks": validate_clocks(config), "train": [], "validation": []}
        for split, order in [("train", [2, 0, 3, 1]), ("validation", [0, 1])]:
            for index in order:
                relative = f"{split}/{index}.npz"
                np.savez(root / relative, rgb=rng.integers(0, 256, (81, 8, 8, 3), dtype=np.uint8),
                    force=np.zeros((80, 1)), theta=np.array([1., .25]))
                truth = f"truth/{split}/{index}.npz"
                if split == "train":
                    # Future states deliberately contain nonphysical sentinels:
                    # they must never be used or even requested by the loader.
                    np.savez(root / truth, exact_initial_state=states[index],
                             state=np.full((81, 4), 9999.))
                manifest[split].append({"path": relative, "truth": truth,
                    "reset_mode": modes[index], "apparatus_id": "test-apparatus"})
        (root / "manifest.json").write_text(json.dumps(manifest))
    return config, torch.from_numpy(states[[2, 0, 3, 1]])


def test_fixed_loader_reads_only_training_reset_member_in_manifest_order(reset_dataset, monkeypatch):
    config, expected = reset_dataset
    original_load = np.load
    accessed = []

    class ResetArchive:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __contains__(self, key):
            return key in self.archive
        def __getitem__(self, key):
            assert key == "exact_initial_state"
            return self.archive[key]

    def guarded(path, **kwargs):
        path = Path(path)
        assert path.parent.name == "train" and path.parent.parent.name == "truth"
        accessed.append(path.name)
        return ResetArchive(original_load(path, **kwargs))

    monkeypatch.setattr(np, "load", guarded)
    table, metadata = load_fixed_training_initial_conditions(Path(config["paths"]["data"]) / "passive")
    assert accessed == ["2.npz", "0.npz", "3.npz", "1.npz"]
    assert torch.equal(table.states, expected)
    assert not list(table.parameters()) and not table.states.requires_grad
    assert metadata["split"] == "train" and metadata["diagnostic_oracle"]
    assert len(metadata["source_sha256"]) == 64


def test_fixed_reset_simulates_absolute_episode_time_for_crops(reset_dataset):
    config, expected = reset_dataset
    root = Path(config["paths"]["data"]) / "passive"
    table, _ = load_fixed_training_initial_conditions(root)
    data = TrainDataset(root)
    ids = torch.tensor([3, 0])
    batch = training.batch_from(data, ids, "cpu", starts=[0, 13])
    assert torch.equal(batch["trajectory_id"], ids)
    assert "initial_state" not in batch and "states" not in batch
    actual = simulate_window(table(ids), batch["theta"], batch["prefix_forces"], batch["raw_endpoints"])
    full = rollout(expected[ids], batch["theta"], batch["prefix_forces"])
    target = full.gather(1, batch["raw_endpoints"][..., None].expand(-1, -1, 4))
    assert torch.equal(actual, target)
    assert actual.dtype == torch.float64 and not actual.requires_grad
    assert actual[1, 0, 0] != 0  # A crop must not restart the cart at p=0.


@pytest.mark.parametrize("dataset", ["passive", "controlled"])
def test_fixed_joint_training_and_resume_preserve_resets_and_neural_updates(reset_dataset, monkeypatch, tmp_path, dataset):
    config, expected = reset_dataset
    config["dataset"] = dataset
    original_step = torch.optim.AdamW.step
    calls = 0
    original_log = training.TrainingLogger.log
    phase_captions = []

    def inspect_media(logger, metrics, step, figures=None, histograms=None):
        if figures and "train_eval/physics_phase" in figures:
            phase = figures["train_eval/physics_phase"]
            phase_captions.append(phase._suptitle.get_text())
            assert phase_captions[-1].startswith("Fixed true-reset diagnostic")
            labels = [line.get_label() for axis in phase.axes for line in axis.lines]
            assert "Fixed true-reset simulation" in labels
            assert "r(P): autoregressive forecast" in labels
        return original_log(logger, metrics, step, figures=figures, histograms=histograms)

    monkeypatch.setattr(training.TrainingLogger, "log", inspect_media)

    def interrupted(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional interruption")
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupted)
    output = tmp_path / "fixed-run"
    with pytest.raises(RuntimeError, match="intentional interruption"):
        training.train(config, "joint", "cpu", run_dir=output, quiet=True)
    intermediate = torch.load(output / "latest.pt", weights_only=True)
    assert intermediate["step"] == 1 and intermediate["initial_optimizer"] is None
    assert torch.equal(intermediate["initial_conditions"]["states"], expected)
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    checkpoint = training.train(config, "joint", "cpu", resume=True, run_dir=output, quiet=True)
    final = torch.load(checkpoint, weights_only=True)
    assert final["step"] == 2 and final["initial_optimizer"] is None
    assert torch.equal(final["initial_conditions"]["states"], expected)
    assert torch.equal(load_initial_conditions_state(final).states, expected)
    assert final["initial_conditions_metadata"] == intermediate["initial_conditions_metadata"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["diagnostic_oracle_training_resets"]
    assert provenance["initial_conditions"] == final["initial_conditions_metadata"]
    rows = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    for name in ["encoder", "predictor", "readout"]:
        assert rows[-1]["metrics"][f"train/gradients/{name}/norm"] > 0
        assert any(not torch.equal(value, intermediate[name][key]) for key, value in final[name].items())
    assert rows[-1]["metrics"]["train/gradients/initial_conditions/norm"] == 0
    assert rows[-1]["metrics"]["train/initial_conditions/trainable"] == 0
    assert rows[-1]["metrics"]["train/initial_conditions/true_reset_supervision"] == 1
    assert "train/initial_conditions/saturated_fraction" not in rows[-1]["metrics"]
    assert not any("updates/initial_conditions" in key for key in rows[-1]["metrics"])
    assert phase_captions and "train_eval/physics_phase" in rows[-1]["media"]
    # Completed-run resumes must validate the returned final.pt, not just latest.pt.
    for field, value, message in [
        ("initial_conditions_metadata", {}, "reset source"),
        ("initial_optimizer", {"unexpected": "optimizer"}, "cannot have an initial-state optimizer"),
        ("interface", {"format_version": 4}, "interface version 5"),
    ]:
        altered = copy.deepcopy(final)
        altered[field] = value
        torch.save(altered, checkpoint)
        with pytest.raises(ValueError, match=message):
            training.train(config, "joint", "cpu", resume=True, run_dir=output, quiet=True)
    altered = copy.deepcopy(final)
    altered["initial_conditions"]["states"][0, 2] += .1
    torch.save(altered, checkpoint)
    with pytest.raises(ValueError, match="fixed reset states differ"):
        training.train(config, "joint", "cpu", resume=True, run_dir=output, quiet=True)
    torch.save(final, checkpoint)
    changed = copy.deepcopy(config)
    changed["training"]["initial_conditions"] = "learned"
    with pytest.raises(ValueError, match="scientific configuration"):
        training.train(changed, "joint", "cpu", resume=True, run_dir=output, quiet=True)
    root = Path(config["paths"]["data"]) / dataset
    np.savez(root / "truth/train/2.npz", exact_initial_state=np.array([0., .2, .7, .4]))
    with pytest.raises(ValueError, match="reset source"):
        training.train(config, "joint", "cpu", resume=True, run_dir=output, quiet=True)


@pytest.mark.parametrize("dataset", ["passive", "controlled"])
def test_default_joint_training_never_accesses_truth(reset_dataset, monkeypatch, tmp_path, dataset):
    config, _ = reset_dataset
    config["dataset"] = dataset
    config["training"].pop("initial_conditions")  # Exercise implicit normal default.
    config["training"].update(jepa_updates=1, physical_updates=1)
    config["diagnostics"]["enabled"] = False
    original_load = np.load

    def guarded(path, **kwargs):
        assert "truth" not in Path(path).parts
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    checkpoint = training.train(config, "joint", "cpu", run_dir=tmp_path / "normal", quiet=True)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["training"]["initial_conditions"] == "learned"
    assert saved["initial_optimizer"] is not None and "raw" in saved["initial_conditions"]


def test_fixed_reset_protocol_rejects_invalid_stage_before_creating_outputs(reset_dataset, tmp_path):
    config, _ = reset_dataset
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="require --mode joint"):
        training.train(config, "jepa", "cpu", run_dir=output)
    assert not output.exists()


def test_fixed_reset_loader_refuses_nontraining_paths_and_missing_reset(reset_dataset):
    config, _ = reset_dataset
    root = Path(config["paths"]["data"]) / "passive"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train"][0]["truth"] = "truth/validation/0.npz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="only truth/train"):
        load_fixed_training_initial_conditions(root)
    manifest["train"][0]["truth"] = "truth/train/2.npz"
    manifest_path.write_text(json.dumps(manifest))
    np.savez(root / "truth/train/2.npz", state=np.zeros((81, 4)))
    with pytest.raises(ValueError, match="future states are not a fallback"):
        load_fixed_training_initial_conditions(root)
