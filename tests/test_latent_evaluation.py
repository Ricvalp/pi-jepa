"""Dense histories, open-loop interventions, ragged targets and frozen inference."""
import json

import numpy as np
import pytest
import torch

from pi_jepa import evaluate_latents
from pi_jepa.evaluate_latents import (encode_endpoints, evaluate_model, load_queries, one_step_latents,
    persistence_predictions, rollout_latents, shuffle_action_programs)
from pi_jepa.latent_reporting import summarize


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, images):
        return (images.mean((1, 2, 3)) * self.scale)[:, None].expand(-1, 32)


class TinyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.01))

    def forward(self, codes, actions, theta):
        return codes + actions.mean(-1, keepdim=True) * self.scale


def test_encoder_uses_every_second_image_and_no_future():
    class RecordingEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, images):
            self.calls.append(images.clone())
            endpoint = ((images[:, -3, 0, 0] + 1) * 127.5).round()
            return endpoint[:, None].expand(-1, 32)

    frames = np.empty((101, 2, 3, 3), dtype=np.uint8)
    for t in range(101):
        frames[t] = np.arange(3) * 64 + t
    encoder = RecordingEncoder()
    encoded = encode_endpoints(encoder, frames, torch.device("cpu"), batch_size=4)
    torch.testing.assert_close(encoded[:, 0], torch.arange(14, 101, 10, dtype=torch.float32))
    images = torch.cat(encoder.calls)
    assert len(encoder.calls) == 9 and all(len(call) == 1 for call in encoder.calls)
    assert images.shape == (9, 24, 2, 3)
    for row, endpoint in enumerate(range(14, 101, 10)):
        expected = (torch.arange(endpoint - 14, endpoint + 1, 2)[:, None] + torch.arange(3)[None] * 64).flatten()
        torch.testing.assert_close(((images[row, :, 0, 0] + 1) * 127.5).round(), expected.float())


def test_rollout_only_uses_predicted_context_and_own_ten_force_blocks():
    class RecordingPredictor:
        def __init__(self):
            self.calls = []

        def __call__(self, codes, actions, theta):
            self.calls.append((codes.clone(), actions.clone(), theta.clone()))
            return codes + 100

    predictor = RecordingPredictor()
    warm = torch.tensor([0., 10., 20.])[None, :, None].expand(2, 3, 32).clone().requires_grad_()
    forces = torch.arange(400, dtype=torch.float32)[None].repeat(2, 1)
    theta = torch.tensor([[.8, .2], [1.2, .4]])
    predicted = rollout_latents(predictor, warm, forces, theta, steps=8)
    assert not predicted.requires_grad
    torch.testing.assert_close(predicted[0, :, 0], torch.arange(120., 821., 100))
    history = warm.detach()
    for step, (codes, blocks, pair) in enumerate(predictor.calls):
        torch.testing.assert_close(codes, history[:, -3:])
        torch.testing.assert_close(blocks, forces[:, 14 + step * 10:44 + step * 10].reshape(2, 3, 10))
        torch.testing.assert_close(pair, theta)
        history = torch.cat((history, predicted[:, step:step + 1]), dim=1)


def test_passive_prediction_has_no_conditioning_and_teacher_forcing_is_separate():
    class Autonomous:
        def __call__(self, code):
            return code + 1

    observed = torch.arange(6.)[None, :, None].expand(1, 6, 32) * 10
    autoregressive = rollout_latents(Autonomous(), observed[:, :3], steps=3, passive=True)
    one_step = one_step_latents(Autonomous(), observed, passive=True)
    torch.testing.assert_close(autoregressive[0, :, 0], torch.tensor([21., 22., 23.]))
    torch.testing.assert_close(one_step[0, :, 0], torch.tensor([21., 31., 41.]))
    persistence = persistence_predictions(observed[:, :3], 3)
    torch.testing.assert_close(persistence[0, :, 0], torch.tensor([20., 20., 20.]))


def test_shuffle_deranges_whole_programs_preserving_time_and_apparatus():
    metadata = [{"apparatus_id": "a" if i < 4 else "b"} for i in range(8)]
    forces = np.arange(8)[:, None] * 1000 + np.arange(400)[None]
    shuffled, donors = shuffle_action_programs(forces, metadata)
    repeated, repeated_donors = shuffle_action_programs(forces, metadata)
    np.testing.assert_array_equal(donors, repeated_donors)
    np.testing.assert_array_equal(shuffled, repeated)
    np.testing.assert_array_equal(np.sort(donors), np.arange(8))
    assert np.all(donors != np.arange(8))
    for recipient, donor in enumerate(donors):
        assert metadata[recipient] == metadata[donor]
        np.testing.assert_array_equal(shuffled[recipient], forces[donor])


def test_ragged_predictions_keep_explicit_missing_target_mask(tmp_path):
    metadata = []
    for i, n in enumerate((54, 400)):
        rgb = np.broadcast_to((np.arange(n + 1) % 256).astype(np.uint8)[:, None, None, None], (n + 1, 2, 2, 3))
        np.savez(tmp_path / f"query{i}.npz", rgb=rgb)
        metadata.append({"path": f"query{i}.npz"})
    programs = np.ones((2, 400), dtype=np.float32)
    bundle = evaluate_model(TinyEncoder(), TinyPredictor(), tmp_path, metadata, programs,
                            np.tile([1., .25], (2, 1)), torch.device("cpu"), shuffled=programs * 2)
    assert bundle["valid"].sum(axis=1).tolist() == [2, 32]
    assert np.isfinite(bundle["correct"]).all()
    assert np.isnan(bundle["encoded"][0, 5:]).all()
    assert np.isfinite(bundle["encoded"][0, :5]).all()
    assert bundle["correct"].shape == (2, 32, 32)


def test_summary_uses_available_targets_and_reports_boundary_denominator():
    encoded = np.zeros((2, 35, 1), dtype=np.float32)
    encoded[0, 3:] = 1
    encoded[1, 3:] = 3
    correct = np.zeros((2, 32, 1), dtype=np.float32)
    correct[1] = 3
    valid = np.ones((2, 32), dtype=bool)
    valid[0, 1:] = False
    bundle = {"encoded": encoded, "correct": correct, "persistence": np.zeros_like(correct),
              "one_step": correct, "one_step_persistence": np.zeros_like(correct), "valid": valid}
    metadata = [{"path": "a", "group": "interior", "reset_mode": "downward", "recorded_intervals": 44,
                 "termination_reason": "boundary_exit"},
                {"path": "b", "group": "interior", "reset_mode": "nonlinear", "recorded_intervals": 400,
                 "termination_reason": "duration"}]
    _, rows = summarize(bundle, metadata, [.1, .2])
    selected = [row for row in rows if row["group"] == row["reset_mode"] == "all" and
                row["condition"] == "correct" and row["protocol"] == "autoregressive"]
    assert selected[0]["latent_mse"] == pytest.approx(.5)
    assert selected[0]["persistence_mse"] == pytest.approx(5.)
    assert selected[0]["persistence_skill"] == pytest.approx(.9)
    assert selected[1]["latent_mse"] == 0
    assert selected[1]["valid_queries"] == 1
    assert selected[1]["queries"] == 2
    assert selected[1]["boundary_exit_rate"] == .5


def test_query_loader_oracle_access_never_reads_hidden_states(tmp_path, monkeypatch):
    manifest = {"dataset": "controlled", "test": [{"apparatus_id": "opaque", "group": "interior", "truth": "private.npz",
                 "queries": ["query.npz"], "query_truth": ["state.npz"]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    accesses = []
    class Archive:
        def __init__(self, path):
            self.path = path.name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, key):
            accesses.append((self.path, key))
            if self.path == "private.npz":
                assert key == "theta_true"
                return np.array([.8, .2])
            assert self.path == "query.npz"
            return {"t": np.arange(31) * .01, "force": np.zeros((30, 1)), "force_program": np.zeros((400, 1)),
                    "reset_mode": np.array("downward"), "collection_family": np.array("pulse"),
                    "termination_reason": np.array("boundary_exit")}[key]
    monkeypatch.setattr(np, "load", lambda path, **kwargs: Archive(path))
    actions, theta, metadata = load_queries(tmp_path)
    assert actions.shape == (1, 400)
    np.testing.assert_allclose(theta, [[.8, .2]])
    assert metadata[0]["recorded_intervals"] == 30 and metadata[0]["valid_length"] == 31
    assert [key for path, key in accesses if path == "private.npz"] == ["theta_true"]
    accesses.clear()
    load_queries(tmp_path, oracle=False)
    assert all(path != "private.npz" for path, _ in accesses)


@pytest.mark.parametrize("checkpoints", [{}, {"unknown": "unused.pt"}])
def test_requires_known_nonempty_checkpoint_selection(checkpoints):
    with pytest.raises(ValueError, match="at least one"):
        evaluate_latents.run({}, checkpoints=checkpoints)
