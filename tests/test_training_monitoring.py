"""Integration checks: monitoring must preserve optimization and frozen stages."""
import json

import pytest
import torch
from torch import nn

import pi_jepa.train as training
from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION
from pi_jepa.training_monitoring import evaluate_diagnostics


def assert_predicted_physics_keys(metrics):
    expected = {
        f"train_eval/physics/autoregressive{horizon}/{metric}"
        for horizon in ("", "/h1", "/h2", "/h3")
        for metric in ("mse", "residual_p", "residual_v", "residual_sin", "residual_cos", "residual_w")
    }
    actual = {key for key in metrics if "physics/autoregressive" in key}
    assert actual == expected  # No validation learned-reset target exists.
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in expected)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.conv1 = nn.Conv2d(24, 8, 1)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 32)

    def forward(self, x):
        return self.head(self.norm(self.backbone.conv1(x).mean((2, 3))))


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning = nn.Sequential(nn.Linear(12, 32))
        self.output_projection = nn.Linear(32, 32)
        self.dropout = nn.Dropout(.1)

    def forward(self, z, forces, theta):
        conditioning = torch.cat((forces, theta[:, None].expand(-1, z.shape[1], -1)), -1)
        return self.output_projection(self.dropout(z + self.conditioning(conditioning)))


@pytest.fixture
def tiny_training(monkeypatch):
    monkeypatch.setattr(training, "Encoder", TinyEncoder)
    monkeypatch.setattr(training, "Predictor", TinyPredictor)
    class TinyEpisodes:
        reset_modes = [0, 2, 1, 2]

        def __init__(self, root, split):
            generator = torch.Generator().manual_seed(10 if split == "train" else 20)
            self.data = {"frames": torch.randint(0, 256, (4, 81, 8, 8, 3), dtype=torch.uint8, generator=generator),
                         "forces": torch.randn(4, 80, generator=generator) * .1}
            self.passive = str(root).endswith("passive")
            self.dataset = "passive" if self.passive else "controlled"
            if self.passive:
                self.data["forces"].zero_()

        def __len__(self):
            return 4

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.data.items()}
            item.update(trajectory_id=index, reset_mode=self.reset_modes[index], apparatus_id="opaque_id")
            if not self.passive:
                item["theta"] = torch.tensor([1.,.25])
            return item

    monkeypatch.setattr(training, "load_learning_data", TinyEpisodes)


def config_at(path, tracking=None):
    cfg = training.load_config()
    cfg["dataset"] = "controlled"
    cfg["paths"]["runs"] = str(path)
    data = path.parent / "synthetic_data"
    data.mkdir(exist_ok=True)
    for corpus in ("controlled", "passive"):
        (data / corpus).mkdir(exist_ok=True)
        (data / corpus / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION,
            "physics": cfg["physics"], "collection_settings": cfg["data"], "camera": CAMERA,
            "cart_limit_m": CART_LIMIT, "dataset": corpus, "train": [], "validation": [],
            "generation_config": {"physics": cfg["physics"], "data": cfg["data"]}, "clocks": training.validate_clocks(cfg)}))
    cfg["paths"]["data"] = str(data)
    cfg["training"].update(batch_size=4, jepa_updates=2, physical_updates=2, warmup=1,
                           save_every=1, val_every=2, log_every=1, cache_batch_size=2, cpu_threads=2)
    if tracking:
        cfg["wandb"] = {"enabled": True, "mode": tracking, "project": "physics-jepa-tests",
                        "log_every": 1, "diagnostics_every": 2, "media_every": 2}
    return cfg


@pytest.mark.parametrize("dataset", ["controlled", "passive"])
def test_monitoring_and_checkpoint_resume_do_not_change_training(tmp_path, monkeypatch, tiny_training, dataset):
    plain = config_at(tmp_path / "plain")
    monitored = config_at(tmp_path / "monitored", "disabled")
    plain["dataset"] = monitored["dataset"] = dataset
    training.train(plain, "joint", "cpu", run_dir=tmp_path / "plain/joint")
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupted(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        training.train(monitored, "joint", "cpu", run_dir=tmp_path / "monitored/joint")
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    training.train(monitored, "joint", "cpu", resume=True, run_dir=tmp_path / "monitored/joint")
    first = torch.load(tmp_path / "plain/joint/final.pt", weights_only=True)
    second = torch.load(tmp_path / "monitored/joint/final.pt", weights_only=True)
    for key in ("encoder", "predictor", "readout", "initial_conditions"):
        assert all(torch.equal(first[key][name], value) for name, value in second[key].items()), key
    for key in ("torch_rng", "sample_rng", "sig_rng"):
        assert torch.equal(first[key], second[key]), key
    rows = [json.loads(row) for row in (tmp_path / "monitored/joint/diagnostics.jsonl").read_text().splitlines()]
    values = rows[-1]["metrics"]
    assert rows[-1]["step"] == 2
    assert "train/gradients/encoder/norm" in values
    assert "val/prediction/autoregressive/h3/all/persistence_skill" in values
    assert "train_eval/physics/residual_p" in values
    assert_predicted_physics_keys(values)
    assert (tmp_path / "monitored/joint/diagnostics_media").exists()
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


def test_long_trajectory_media_logs_initial_periodic_and_final_updates(tmp_path, tiny_training):
    config = config_at(tmp_path / "phase", "disabled")
    config["training"].update(jepa_updates=3, physical_updates=3, val_every=3)
    config["wandb"].update(diagnostics_every=3, media_every=2)
    checkpoint = training.train(config, "joint", "cpu", run_dir=tmp_path / "phase/joint", quiet=True)
    output = checkpoint.parent
    rows = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if "train_eval/physics_phase" in row.get("media", {})] == [0, 2, 3]
    selection = json.loads((output / "diagnostic_windows.json").read_text())["training_trajectories"]
    assert [item["reset_mode"] for item in selection["episodes"]] == [0, 1, 2]
    assert [item["trajectory_id"] for item in selection["episodes"]] == [0, 2, 1]
    assert len(list((output / "diagnostics_media").glob("*_train_eval_physics_phase.png"))) == 3


@pytest.mark.integration
def test_real_offline_sdk_with_training_and_frozen_readout(tmp_path, tiny_training):
    pytest.importorskip("wandb")
    config = config_at(tmp_path / "offline", "offline")
    pretrained = training.train(config, "jepa", "cpu", run_dir=tmp_path / "offline/jepa")
    training.train(config, "readout", "cpu", run_dir=tmp_path / "offline/readout", pretrained=pretrained)
    frozen = json.loads((tmp_path / "offline/readout/freeze_check.json").read_text())
    assert frozen["encoder_and_predictor_unchanged"]
    for mode in ("jepa", "readout"):
        output = tmp_path / "offline" / mode
        identity = json.loads((output / "wandb_run.json").read_text())
        assert identity["mode"] == "offline"
        rows = [json.loads(row) for row in (output / "diagnostics.jsonl").read_text().splitlines()]
        assert rows[0]["step"] == 0 and rows[-1]["step"] == 2
        assert len(rows[-1]["media"]) >= 6
        assert list((output / "wandb").glob("offline-run-*/*.wandb"))
    readout_rows = [json.loads(row) for row in (tmp_path / "offline/readout/diagnostics.jsonl").read_text().splitlines()]
    assert readout_rows[-1]["metrics"]["train/gradients/encoder/norm"] == 0
    assert readout_rows[-1]["metrics"]["train/gradients/readout/norm"] > 0


def test_local_readout_is_frozen_and_resume_validates_stage_and_manifest(tmp_path, tiny_training):
    config = config_at(tmp_path / "local", "disabled")
    jepa_path = training.train(config, "jepa", "cpu", run_dir=tmp_path / "local/jepa", quiet=True)
    readout_path = training.train(config, "readout", "cpu", run_dir=tmp_path / "local/readout",
                                  pretrained=jepa_path, quiet=True)
    first = torch.load(jepa_path, weights_only=True)
    second = torch.load(readout_path, weights_only=True)
    for key in ("encoder", "predictor"):
        assert all(torch.equal(value, second[key][name]) for name, value in first[key].items())
    assert second["interface"]["weights"] == "raw"
    assert second["interface"]["normalization"]["fitted_statistics"] is None
    for stage, checkpoint in (("jepa", jepa_path), ("readout", readout_path)):
        rows = [json.loads(row) for row in (checkpoint.parent / "diagnostics.jsonl").read_text().splitlines()]
        for row in rows:
            if stage == "readout" and row["step"] in (0, 2):
                assert_predicted_physics_keys(row["metrics"])
            else:
                assert not any("physics/autoregressive" in key for key in row["metrics"])
    with pytest.raises(ValueError, match="stage differs"):
        training.train(config, "joint", "cpu", resume=True, run_dir=jepa_path.parent, quiet=True)
    manifest = tmp_path / "synthetic_data/controlled/manifest.json"
    changed = json.loads(manifest.read_text())
    changed["changed"] = True
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest differs"):
        training.train(config, "jepa", "cpu", resume=True, run_dir=jepa_path.parent, quiet=True)


def test_passive_training_has_no_conditioning_or_readout_and_records_interface(tmp_path, tiny_training):
    config = config_at(tmp_path / "passive", "disabled")
    config["dataset"] = "passive"
    checkpoint = training.train(config, "jepa", "cpu", run_dir=tmp_path / "passive/run", quiet=True)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["readout"] is None
    assert saved["interface"]["dataset"] == "passive"
    assert saved["interface"]["force_block_length"] is None
    assert saved["interface"]["frame_dt_seconds"] == .01
    assert saved["interface"]["prediction_dt_seconds"] == .1
    rows = [json.loads(row) for row in (checkpoint.parent / "diagnostics.jsonl").read_text().splitlines()]
    assert not any("action_advantage" in key for row in rows for key in row["metrics"])
    assert not any("physics/autoregressive" in key for row in rows for key in row["metrics"])


def test_passive_joint_trains_physical_readout_reset_and_unconditioned_dynamics(tmp_path, tiny_training):
    config = config_at(tmp_path / "passive_joint", "disabled")
    config["dataset"] = "passive"
    checkpoint = training.train(config, "joint", "cpu", run_dir=tmp_path / "passive_joint/run", quiet=True)
    saved = torch.load(checkpoint, weights_only=True)
    initial = torch.load(checkpoint.parent / "initial.pt", weights_only=True)
    for name in ("encoder", "predictor", "readout"):
        assert any(not torch.equal(value, initial[name][key]) for key, value in saved[name].items()), name
    assert saved["initial_conditions"]["raw"].abs().sum() > 0
    assert saved["interface"]["format_version"] == 5
    assert saved["interface"]["physics"]["ell"] == .85
    assert saved["interface"]["cart_limit_m"] == 1.5
    assert saved["interface"]["simulator_theta"] == [1., .25]
    assert saved["interface"]["predictor_conditioning"] == "none"
    from pi_jepa.evaluate_latents import validate_checkpoints
    assert validate_checkpoints(config, {"passive": checkpoint}, require_readout=True)["passive"] == checkpoint
    rows = [json.loads(row) for row in (checkpoint.parent / "diagnostics.jsonl").read_text().splitlines()]
    metrics = rows[-1]["metrics"]
    assert metrics["train/gradients/readout/norm"] > 0
    assert metrics["train/gradients/initial_conditions/norm"] > 0
    assert "val/loss/physics_reset_prior" in metrics
    assert "train_eval/physics/residual_p" in metrics
    for row in rows:
        if row["step"] in (0, 2):
            assert_predicted_physics_keys(row["metrics"])
        else:
            assert not any("physics/autoregressive" in key for key in row["metrics"])
    assert not any("action_advantage" in key for row in rows for key in row["metrics"])


@pytest.mark.parametrize("passive", [False, True])
def test_predicted_physics_uses_absolute_future_endpoints_without_future_images(passive):
    class FrameIndexEncoder(nn.Module):
        def forward(self, clips):
            # The final RGB image encodes its absolute record index in each pixel.
            index = (clips[:, -1, 0, 0] + 1) * 127.5
            return index[:, None].expand(-1, 32)

    class ConstantSpeedPredictor(nn.Module):
        action_free = passive

        def forward(self, z, *conditioning):
            return z + 10

    class ConstantSpeedReadout(nn.Module):
        def forward(self, z):
            # Constant cart speed 2 m/s, down pole at rest, zero drag and force.
            # The analytical solution is p(t)=2t, independent of pole/cart mass.
            one, zero = torch.ones_like(z[..., 0]), torch.zeros_like(z[..., 0])
            return torch.stack((.02 * z[..., 0], 2 * one, zero, one, zero), -1)

    class KnownReset(nn.Module):
        def forward(self, ids):
            return torch.tensor([0., 2., 0., 0.]).expand(len(ids), -1)

    starts = torch.tensor([0, 11])
    frames = (starts[:, None] + torch.arange(65)[None]).to(torch.uint8)
    reference = {
        "frames": frames[:, :, None, None, None].expand(-1, -1, 1, 1, 3).clone(),
        "forces": torch.zeros(2, 64), "prefix_forces": torch.zeros(2, 75),
        "raw_endpoints": starts[:, None] + torch.tensor([14, 24, 34, 44, 54, 64]),
        "trajectory_id": torch.arange(2), "theta": torch.tensor([[1., 0.], [1., 0.]]),
        "reset_mode": torch.zeros(2, dtype=torch.long), "apparatus_id": torch.zeros(2),
    }
    modules = (FrameIndexEncoder(), ConstantSpeedPredictor(), ConstantSpeedReadout(), KnownReset())
    config = {"training": {"cache_batch_size": 2}}
    rng = torch.random.get_rng_state().clone()
    original, _, _ = evaluate_diagnostics(*modules, reference, reference, config, "cpu", "joint")
    assert_predicted_physics_keys(original)
    for key in original:
        if "physics/autoregressive" in key:
            assert original[key] == pytest.approx(0., abs=1e-10), key
    # Modify every actual future image while preserving all observed context.
    changed = {key: value.clone() for key, value in reference.items()}
    changed["frames"][:, 35:] += 50
    perturbed, _, _ = evaluate_diagnostics(*modules, changed, changed, config, "cpu", "joint")
    for key in original:
        if "physics/autoregressive" in key:
            assert perturbed[key] == original[key], key
    assert perturbed["train_eval/physics/residual_p"] > original["train_eval/physics/residual_p"]
    assert (perturbed["train_eval/prediction/autoregressive/h1/all/mse_correct"]
            > original["train_eval/prediction/autoregressive/h1/all/mse_correct"])
    assert all(module.training for module in modules)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_windows_carry_full_force_prefix_and_reproducible_random_starts(tiny_training):
    data = training.load_learning_data("controlled", "train")
    first = training.batch_from(data, torch.arange(4), "cpu", torch.Generator().manual_seed(8))
    second = training.batch_from(data, torch.arange(4), "cpu", torch.Generator().manual_seed(8))
    assert first["frames"].shape[1] == 65
    torch.testing.assert_close(first["window_start"], second["window_start"])
    assert first["window_start"].max() > 0
    for row, start in enumerate(first["window_start"]):
        item = data[row]
        end = start + 64
        torch.testing.assert_close(first["frames"][row], item["frames"][start:end+1])
        torch.testing.assert_close(first["forces"][row], item["forces"][start:end])
        torch.testing.assert_close(first["prefix_forces"][row,:end], item["forces"][:end])
        torch.testing.assert_close(first["raw_endpoints"][row], start + torch.tensor([14,24,34,44,54,64]))


def test_training_rejects_incompatible_clocks_before_creating_run(tmp_path, tiny_training):
    config = config_at(tmp_path / "invalid")
    config["physics"]["dt"] = .04
    with pytest.raises(ValueError, match="physical constants/clocks"):
        training.train(config, "joint", "cpu", run_dir=tmp_path / "invalid/new")
    assert not (tmp_path / "invalid/new").exists()


def test_training_rejects_changed_manifest_clock_metadata(tmp_path, tiny_training):
    config = config_at(tmp_path / "manifest_clocks")
    path = tmp_path / "synthetic_data/controlled/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["clocks"]["latent_dt_s"] = .32
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest clocks differ"):
        training.train(config, "joint", "cpu", run_dir=tmp_path / "manifest_clocks/new")
