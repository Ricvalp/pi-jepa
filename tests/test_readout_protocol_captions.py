"""Training videos must distinguish oracle reset supervision from learned resets."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def visuals():
    path = Path(__file__).resolve().parents[1] / "scripts/dataset_videos/compare_readouts.py"
    spec = importlib.util.spec_from_file_location("readout_protocol_video", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
    return module


def test_protocol_metadata_preserves_legacy_and_rejects_contradictions(visuals):
    assert visuals.read_training_protocol({}) == "learned"
    assert visuals.read_training_protocol({"train": {"initial_conditions_mode": "true_fixed",
        "supervision_uses_true_reset": True}}) == "true_fixed"
    with pytest.raises(ValueError, match="disagree"):
        visuals.read_training_protocol({"train": {"initial_conditions_mode": "true_fixed",
            "supervision_uses_true_reset": False}})


@pytest.mark.parametrize("protocol", ["learned", "true_fixed"])
def test_gallery_describes_the_actual_training_supervision(visuals, tmp_path, protocol):
    visuals.gallery(tmp_path, [], .5, .85, protocol)
    page = (tmp_path / "index.html").read_text()
    if protocol == "true_fixed":
        assert "True-reset diagnostic" in page
        assert "fixed true training resets" in page
        assert "learned reset table" not in page
        assert "test resets are not supplied to the predictor" in page
    else:
        assert "learned reset table" in page
        assert "True-reset diagnostic" not in page


@pytest.mark.parametrize("kind", ["train", "test"])
def test_fixed_reset_frames_are_explicitly_labelled_as_diagnostic(visuals, monkeypatch, kind):
    captions = []
    original_draw = visuals.ImageDraw.Draw

    class RecordingDraw:
        def __init__(self, image):
            self.draw = original_draw(image)

        def __getattr__(self, name):
            return getattr(self.draw, name)

        def text(self, xy, text, *args, **kwargs):
            captions.append(text)
            return self.draw.text(xy, text, *args, **kwargs)

    monkeypatch.setattr(visuals.ImageDraw, "Draw", RecordingDraw)
    states = np.zeros((2, 4))
    raw = np.stack([visuals.state_coordinates(state) for state in states])
    image = visuals.render_sample([states]*3, [raw]*3, np.array([.1, .2]), 0,
        {"reset_mode": "downward", "path": "train/example.npz"}, kind, .5, .85,
        "passive", "true_fixed")
    assert image.size == visuals.SIZE
    text = " ".join(captions)
    assert "TRUE-RESET DIAGNOSTIC" in text
    assert "learned training reset" not in text and "learned-reset simulator" not in text
    if kind == "train":
        assert "From fixed true training reset" in text
        assert "True future states are not loaded for the training loss" in text
