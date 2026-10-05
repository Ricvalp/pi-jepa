"""Tracking is optional, reproducible, and follows checkpoint resume semantics."""
import importlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pi_jepa.experiment_logging import TrainingLogger, preserve_rng


def _consume_rng():
    random.random()
    np.random.rand()
    torch.rand(3)


class FakeRun:
    def __init__(self, kwargs):
        self.id = kwargs["id"]
        self.entity = kwargs["entity"] or "test-user"
        self.settings = SimpleNamespace(mode=kwargs["mode"])
        self.url = "https://example.invalid/run/" + self.id
        self.logs, self.metrics, self.exit_codes = [], [], []

    def define_metric(self, *args, **kwargs):
        _consume_rng()
        self.metrics.append((args, kwargs))

    def log(self, *args, **kwargs):
        _consume_rng()
        self.logs.append((args, kwargs))

    def finish(self, exit_code):
        _consume_rng()
        self.exit_codes.append(exit_code)


class FakeSDK:
    def __init__(self):
        self.calls, self.runs = [], []

    def init(self, **kwargs):
        _consume_rng()
        self.calls.append(kwargs)
        run = FakeRun(kwargs)
        self.runs.append(run)
        return run

    def Image(self, path):
        _consume_rng()
        return ("image", path)

    def Histogram(self, np_histogram):
        _consume_rng()
        return ("histogram", np_histogram)


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = FakeSDK()
    original = importlib.import_module

    def imported(name, *args, **kwargs):
        if name == "wandb":
            _consume_rng()
            return sdk
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", imported)
    return sdk


def _config(mode="online"):
    return {"seed": 42, "wandb": {"enabled": True, "project": "test-project", "mode": mode}}


def _rows(output):
    return [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("config", [{"seed": 42}, _config("disabled")])
def test_disabled_has_no_sdk_import_and_writes_valid_local_scalars(tmp_path, monkeypatch, config):
    def unexpected_import(*args, **kwargs):
        raise AssertionError("Disabled logger imported W&B")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    with TrainingLogger(config, "joint", tmp_path) as logger:
        assert not logger.enabled
        logger.log({"loss": torch.tensor(1.5), "undefined": float("nan"),
                    "overflow": np.float64(np.inf), "valid": True}, step=0)
    row = _rows(tmp_path)[0]
    assert row["step"] == 0
    assert row["metrics"] == {"loss": 1.5, "undefined": None, "overflow": None,
                              "valid": True, "train/update": 0}
    assert not (tmp_path / "wandb_run.json").exists()


def test_checkpoint_resume_trims_local_history_including_partial_final_row(tmp_path):
    config = {"seed": 42}
    with TrainingLogger(config, "joint", tmp_path) as logger:
        for step in (0, 2, 4, 6):
            logger.log({"loss": step}, step)
    with (tmp_path / "diagnostics.jsonl").open("a") as stream:
        stream.write('{"step": 8, "metrics":')
    with TrainingLogger(config, "joint", tmp_path, resume=True, start_step=4) as logger:
        logger.log({"loss": 5}, step=5)
    assert [row["step"] for row in _rows(tmp_path)] == [0, 2, 4, 5]


def test_online_resume_preserves_identity_and_uses_independent_update_axis(tmp_path, fake_sdk):
    config = _config()
    with TrainingLogger(config, "joint", tmp_path) as logger:
        logger.log({"loss": 1.0, "undefined": float("nan")}, step=10)
        old_id = logger.run_metadata["id"]
    with TrainingLogger(config, "joint", tmp_path, resume=True, start_step=5) as logger:
        logger.log({"loss": 0.5}, step=6)
        assert logger.run_metadata["id"] == old_id
    assert fake_sdk.calls[0]["resume"] == "never"
    assert fake_sdk.calls[1]["resume"] == "must"
    assert fake_sdk.calls[1]["entity"] == "test-user"
    assert fake_sdk.calls[0]["name"] == tmp_path.name
    assert fake_sdk.calls[0]["save_code"] is False
    assert fake_sdk.calls[0]["settings"]["disable_code"]
    assert fake_sdk.runs[1].metrics[-1] == (("*",), {"step_metric": "train/update"})
    # No SDK step=6 argument: W&B must append the replay rather than drop it.
    assert fake_sdk.runs[1].logs == [(({"loss": 0.5, "train/update": 6},), {})]
    assert "undefined" not in fake_sdk.runs[0].logs[0][0][0]
    assert [row["step"] for row in _rows(tmp_path)] == [6]
    assert fake_sdk.runs[1].exit_codes == [0]


def test_offline_resume_creates_linked_segment_without_unsupported_resume(tmp_path, fake_sdk):
    config = _config("offline")
    with TrainingLogger(config, "jepa", tmp_path) as logger:
        first = logger.run_metadata
        logger.log({"loss": 1}, step=5)
    with TrainingLogger(config, "jepa", tmp_path, resume=True, start_step=5) as logger:
        second = logger.run_metadata
    assert first["id"] != second["id"]
    assert first["group"] == second["group"]
    assert second["previous_run_id"] == first["id"]
    assert second["segments"][0]["id"] == first["id"]
    assert second["name"] == tmp_path.name
    assert all("resume" not in call for call in fake_sdk.calls)
    assert json.loads((tmp_path / "wandb_run.json").read_text()) == second


def test_different_stages_get_distinct_runs_in_same_group(tmp_path, fake_sdk):
    with TrainingLogger(_config(), "jepa", tmp_path / "jepa") as first:
        pass
    with TrainingLogger(_config(), "joint", tmp_path / "joint") as second:
        pass
    assert first.run_metadata["id"] != second.run_metadata["id"]
    assert first.run_metadata["group"] == second.run_metadata["group"]


def test_no_silent_fallback_on_authentication_failure(tmp_path, fake_sdk, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("API key missing")

    monkeypatch.setattr(fake_sdk, "init", fail)
    with pytest.raises(RuntimeError, match="wandb login") as raised:
        TrainingLogger(_config(), "joint", tmp_path)
    assert "API key missing" in str(raised.value.__cause__)
    assert not (tmp_path / "wandb_run.json").exists()


def test_sdk_silent_mode_fallback_is_rejected(tmp_path, fake_sdk, monkeypatch):
    original = fake_sdk.init

    def disabled(**kwargs):
        run = original(**kwargs)
        run.settings.mode = "disabled"
        return run

    monkeypatch.setattr(fake_sdk, "init", disabled)
    with pytest.raises(RuntimeError, match="Could not initialize"):
        TrainingLogger(_config(), "joint", tmp_path)
    assert fake_sdk.runs[0].exit_codes == [1]


def test_tracking_and_media_preserve_training_rng_states(tmp_path, fake_sdk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.plot([0, 1], [2, 3])
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    with TrainingLogger(_config(), "joint", tmp_path) as logger:
        logger.log({"loss": 1}, step=0, figures={"diagnostics/prediction": figure},
                   histograms={"latents/values": np.arange(5), "empty": np.array([np.nan])})
    plt.close(figure)
    assert python_state == random.getstate()
    assert numpy_state[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    media = _rows(tmp_path)[0]["media"]
    assert len(media) == 2
    assert all((tmp_path / path).is_file() for path in media.values())
    payload = fake_sdk.runs[0].logs[0][0][0]
    assert payload["diagnostics/prediction"][0] == "image"
    assert payload["latents/values"][0] == "histogram"


def test_failure_finishes_run_and_rng_context_restores_on_exception(tmp_path, fake_sdk):
    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="training failed"):
        with TrainingLogger(_config(), "joint", tmp_path):
            with preserve_rng():
                _consume_rng()
                raise RuntimeError("training failed")
    assert torch.equal(state, torch.get_rng_state())
    assert fake_sdk.runs[0].exit_codes == [1]


def test_resume_rejects_changed_project_and_fresh_run_rejects_existing_history(tmp_path, fake_sdk):
    with TrainingLogger(_config(), "joint", tmp_path) as logger:
        logger.log({"loss": 1}, 1)
    altered = _config()
    altered["wandb"]["project"] = "other-project"
    with pytest.raises(ValueError, match="different project"):
        TrainingLogger(altered, "joint", tmp_path, resume=True, start_step=1)
    with pytest.raises(FileExistsError, match="use --resume"):
        TrainingLogger(_config(), "joint", tmp_path)
    assert len(fake_sdk.calls) == 1
