"""Physical geometry contracts for decoded-trajectory comparison videos."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def visuals():
    path = Path(__file__).resolve().parents[1] / "scripts/dataset_videos/compare_readouts.py"
    spec = importlib.util.spec_from_file_location("readout_video_renderer", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ell", [.85, 1.1])
def test_rendered_rod_length_is_constant_and_matches_horizontal_world_scale(visuals, ell):
    box, bounds = (20, 30, 640, 480), (-2.0, 2.0)
    lengths = []
    for position in (-1.0, 0.0, 1.0):
        for angle in np.linspace(-np.pi, np.pi, 17):
            cart, bob = visuals.rod_points(np.array([position, 0.0, angle, 0.0]), box, bounds, ell)
            lengths.append(np.linalg.norm(np.asarray(bob) - np.asarray(cart)))
    np.testing.assert_allclose(lengths, lengths[0], rtol=1e-12, atol=1e-12)
    origin, _ = visuals.rod_points(np.zeros(4), box, bounds, ell)
    shifted, _ = visuals.rod_points(np.array([ell, 0.0, 0.0, 0.0]), box, bounds, ell)
    # Translation by the physical rod length must equal its projected length.
    assert np.linalg.norm(np.asarray(shifted) - np.asarray(origin)) == pytest.approx(lengths[0])


def test_zero_angle_is_down_and_pi_is_up_in_image_coordinates(visuals):
    box, bounds = (0, 0, 640, 480), (-2.0, 2.0)
    down_cart, down_bob = visuals.rod_points(np.zeros(4), box, bounds, .85)
    up_cart, up_bob = visuals.rod_points(np.array([0.0, 0.0, np.pi, 0.0]), box, bounds, .85)
    np.testing.assert_allclose(down_cart, up_cart)
    assert down_bob[0] == pytest.approx(down_cart[0])
    assert down_bob[1] > down_cart[1]
    assert up_bob[0] == pytest.approx(up_cart[0])
    assert up_bob[1] < up_cart[1]


def test_shared_camera_contains_cart_and_bob_for_every_model(visuals):
    truth = np.array([[-3.0, 0.0, -np.pi / 2, 0.0], [0.0, 0.0, 0.0, 0.0]])
    forecast = np.array([[0.0, 0.0, np.pi, 0.0], [4.0, 0.0, np.pi / 2, 0.0]])
    ell = .85
    lower, upper = visuals.world_bounds([truth, forecast], ell)
    for states in (truth, forecast):
        carts = states[:, 0]
        bobs = carts + ell * np.sin(states[:, 2])
        assert lower <= min(carts.min(), bobs.min())
        assert upper >= max(carts.max(), bobs.max())
    assert lower < -3.85 and upper > 4.85
    box = (0, 0, 456, 378)
    for p in (-3., 4.):
        for angle in np.linspace(-np.pi, np.pi, 33):
            _, bob = visuals.rod_points([p, 0, angle, 0], box, (lower, upper), ell)
            assert 8 < bob[0] < box[2] - 8 and 8 < bob[1] < box[3] - 8


def test_camera_bounds_use_positions_and_angles_not_velocity_units(visuals):
    states = np.array([[0.0, 0.0, 0.2, 0.0], [0.4, 0.0, -1.0, 0.0]])
    different_velocities = states.copy()
    different_velocities[:, [1, 3]] = [[1e5, -1e5], [-1e5, 1e5]]
    np.testing.assert_allclose(visuals.world_bounds([states], .85),
                               visuals.world_bounds([different_velocities], .85))


def test_readout_renderer_requires_explicit_export_geometry(visuals):
    from pi_jepa.data import CAMERA, SCHEMA_VERSION
    metadata = {"dataset": "passive", "data_schema_version": SCHEMA_VERSION,
                "physics": {"ell": .85}, "camera": CAMERA}
    assert visuals.read_geometry(metadata) == (.85, "passive")
    for field in ("dataset", "physics", "camera", "data_schema_version"):
        invalid = dict(metadata)
        invalid.pop(field)
        with pytest.raises(ValueError, match="explicit"):
            visuals.read_geometry(invalid)
    with pytest.raises(ValueError, match="rod length"):
        visuals.read_geometry({**metadata, "physics": {"ell": float("nan")}})


def test_playback_repeats_samples_without_inventing_intermediate_states(visuals, monkeypatch, tmp_path):
    frames, settings = [], {}

    def writer(path, size, **kwargs):
        settings.update(kwargs)
        while True:
            frames.append((yield))

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(write_frames=writer))
    samples = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (11, 22)]
    repeats, fps = visuals.write_video(tmp_path / "unused.mp4", samples, 0.32, 0.5)
    assert repeats == 16
    assert fps == settings["fps"] == 25
    assert len(frames) == 2 * repeats
    for index, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, samples[index // repeats])
    assert len(frames) / fps == pytest.approx(2 * 0.32 / 0.5)


@pytest.fixture(scope="module")
def exporter():
    path = Path(__file__).resolve().parents[1] / "scripts/export_readout_trajectories.py"
    spec = importlib.util.spec_from_file_location("readout_state_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forecast_decoding_keeps_observed_origin_separate_from_future_targets(exporter):
    class Readout(torch.nn.Module):
        def forward(self, z):
            p = z[..., 0]
            return torch.stack((p, p + 1, p.sin(), p.cos(), p - 1), dim=-1)

    encoded = np.zeros((2, 35, 32), dtype=np.float32)
    encoded[..., 0] = np.arange(35)
    predicted = np.zeros((2, 32, 32), dtype=np.float32)
    predicted[..., 0] = np.arange(32) + 20
    result = exporter.decode_test_latents(Readout(), encoded, predicted)
    np.testing.assert_array_equal(result["forecast"][..., 0],
                                  np.broadcast_to([2, *range(20, 52)], (2, 33)))
    np.testing.assert_array_equal(result["reconstruction"][..., 0],
                                  np.broadcast_to(np.arange(2, 35), (2, 33)))
    np.testing.assert_allclose(result["raw_forecast"][..., 2],
                               np.sin(result["forecast"][..., 0]), atol=1e-7)
    np.testing.assert_allclose(result["forecast"][..., 2],
                               np.arctan2(result["raw_forecast"][..., 2],
                                          result["raw_forecast"][..., 3]), atol=1e-7)
    # Replacing all actual future observations cannot change an open-loop forecast.
    encoded[:, 3:] = -999
    altered = exporter.decode_test_latents(Readout(), encoded, predicted)
    np.testing.assert_array_equal(result["forecast"], altered["forecast"])


def test_exported_supervision_matches_training_loss_units_and_trajectory_ids(exporter):
    from pi_jepa.losses import physics_loss
    from pi_jepa.models import PhysicalReadout

    class InitialTable:
        def __init__(self):
            self.ids = None
            self.states = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                        [0.2, 0.3, 0.4, 0.5],
                                        [-0.4, -0.2, 0.3, -0.1]])

        def __call__(self, ids):
            self.ids = ids.clone()
            return self.states[ids]

    readout = PhysicalReadout().eval()
    z = torch.arange(2 * 6 * 32, dtype=torch.float32).reshape(2, 6, 32) / 400
    table = InitialTable()
    ids = torch.tensor([2, 0])
    theta = torch.tensor([[1.0, 0.2], [1.2, 0.3]])
    forces = torch.zeros(2, 64)
    result = exporter.training_supervision(readout, z, table, ids, theta, forces, 0.01, 5)
    torch.testing.assert_close(table.ids, ids)
    with torch.no_grad():
        decoded = readout(z)
        training_loss = physics_loss(decoded, torch.as_tensor(result["supervision"]))
    exported_loss = np.mean((result["supervised_reconstruction"] - result["supervised_target"]) ** 2)
    assert exported_loss == pytest.approx(training_loss.item(), rel=1e-6)
    np.testing.assert_allclose(result["raw_reconstruction"], decoded.numpy())
    # Zero reset/action produces the stationary downward trajectory in row 1.
    np.testing.assert_array_equal(result["supervision"][1], np.zeros((6, 4)))
    assert np.max(np.abs(result["supervision"][0])) > 0.1


def test_exported_cropped_supervision_integrates_from_episode_reset(exporter):
    from pi_jepa.models import PhysicalReadout
    from pi_jepa.physics import rollout

    initial = torch.tensor([[0., .2, .7, -.1]])
    table = lambda ids: initial[ids]
    theta = torch.tensor([[1., .25]])
    forces = torch.ones(1, 104)
    endpoints = torch.tensor([[54, 64, 74, 84, 94, 104]])
    result = exporter.training_supervision(PhysicalReadout().eval(), torch.zeros(1, 6, 32),
        table, torch.tensor([0]), theta, forces, .01, 5, raw_endpoints=endpoints)
    expected = rollout(initial, theta, forces, .01, 5)[:, endpoints[0]]
    np.testing.assert_allclose(result["supervision"], expected.numpy())
    assert result["supervision"][0, 0, 0] != 0  # Crop does not command a new centered reset.


@pytest.mark.parametrize("protocol", ["learned", "true_fixed"])
def test_passive_joint_export_uses_fixed_physics_and_records_render_geometry(exporter, tmp_path, monkeypatch, protocol):
    from pi_jepa.checkpoint_interface import CHECKPOINT_FORMAT_VERSION, model_interface, geometry_interface
    from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION, validate_clocks
    from pi_jepa.models import PhysicalReadout
    from pi_jepa.runs import file_digest
    from pi_jepa.train import InitialConditions, load_config
    from pi_jepa.initial_conditions import FixedInitialConditions

    class TinyEncoder(torch.nn.Module):
        def forward(self, frames):
            return frames.mean((1, 2, 3))[:, None].expand(-1, 32)

    monkeypatch.setattr(exporter, "Encoder", TinyEncoder)
    root, source = tmp_path / "data/passive", tmp_path / "latent"
    root.mkdir(parents=True)
    (source / "passive").mkdir(parents=True)
    cfg = load_config()
    cfg["training"]["initial_conditions"] = protocol
    cfg.update(dataset="passive", paths={"data": str(root.parent), "results": str(tmp_path / "outputs")})
    manifest = {"dataset": "passive", "schema_version": SCHEMA_VERSION, "camera": CAMERA,
                "cart_limit_m": CART_LIMIT, "physics": cfg["physics"], "collection_settings": cfg["data"],
                "clocks": validate_clocks(cfg), "train": []}
    for i in range(3):
        entry = {"path": f"train{i}.npz", "truth": f"truth{i}.npz", "reset_mode": i,
                 "apparatus_id": "nominal", "episode_id": f"episode{i}"}
        manifest["train"].append(entry)
        # No theta array: passive export must use the declared nominal apparatus.
        np.savez(root / entry["path"], rgb=np.zeros((65, 2, 2, 3), dtype=np.uint8),
                 force=np.zeros((64, 1)), episode_id=entry["episode_id"])
        np.savez(root / entry["truth"], state=np.zeros((65, 4)))
    np.savez(root / "query.npz", force=np.zeros((400, 1)))
    np.savez(root / "query_truth.npz", state=np.zeros((401, 4)))
    query = {"path": "query.npz", "truth": "query_truth.npz", "reset_mode": "downward", "apparatus_id": "nominal"}
    manifest["test"] = [query]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "passive_joint.pt"
    table = (FixedInitialConditions(torch.zeros(3, 4), [0, 1, 2]) if protocol == "true_fixed"
             else InitialConditions([0, 1, 2]))
    torch.save({"mode": "joint", "step": 1, "config": cfg, "encoder": {},
                "readout": PhysicalReadout().state_dict(), "initial_conditions": table.state_dict(),
                "interface": {"format_version": CHECKPOINT_FORMAT_VERSION, **geometry_interface(),
                              **model_interface(cfg, "joint")},
                "data_manifest_sha256": file_digest(manifest_path)}, checkpoint)
    settings = {"config": cfg, "manifest_sha256": file_digest(manifest_path), "parameter_condition": "none",
                "endpoints": exporter.ENDPOINTS.tolist(),
                "checkpoints": {"passive": {"path": str(checkpoint), "sha256": file_digest(checkpoint)}}}
    (source / "settings.json").write_text(json.dumps(settings))
    (source / "queries.json").write_text(json.dumps([query]))
    np.savez(source / "passive/latents.npz", encoded=np.zeros((1, 35, 32), dtype=np.float32),
             correct=np.zeros((1, 32, 32), dtype=np.float32), valid=np.ones((1, 32), dtype=bool))
    output = exporter.export(SimpleNamespace(latent_run=source, variant=None, data_root=None,
                            output=None, output_root=tmp_path / "exports"))
    metadata = json.loads((output / "readout_metadata.json").read_text())
    assert metadata["dataset"] == metadata["variant"] == "passive"
    assert metadata["physics"]["ell"] == .85 and metadata["cart_limit_m"] == 1.5
    assert metadata["camera"] == CAMERA
    assert metadata["test"]["parameter_condition"] == "none"
    assert "no action or parameter inputs" in metadata["test"]["future"]
    assert metadata["physical_loss_divisors"] == [2., 2., 1., 1., 5.]
    assert metadata["train"]["initial_conditions_mode"] == protocol
    assert metadata["train"]["supervision_uses_true_reset"] == (protocol == "true_fixed")
    with np.load(output / "test_trajectories.npz") as values:
        assert values["force_blocks"].shape == (1, 32, 10)
        assert not values["force_blocks"].any()
    with np.load(output / "train_trajectories.npz") as values:
        assert values["supervision"].shape == (3, 6, 4)
        np.testing.assert_allclose(values["supervision"][:2], 0)
