"""Inference stability and causal gradients for the batch-independent encoder."""
import copy

import pytest
import torch
from torch import nn

from pi_jepa.checkpoint_interface import (CHECKPOINT_FORMAT_VERSION, ENCODER_ARCHITECTURE,
                                         encoder_interface, geometry_interface,
                                         validate_checkpoint_interface)
from pi_jepa.data import causal_clips
from pi_jepa.models import Encoder, encode_temporal


@pytest.fixture
def encoder():
    torch.manual_seed(271)
    return Encoder()


@torch.no_grad()
def test_same_images_have_identical_train_and_eval_encodings(encoder):
    images = torch.randn(2, 24, 96, 96)
    training_codes = encoder.train()(images)
    inference_codes = encoder.eval()(images)
    torch.testing.assert_close(training_codes, inference_codes, rtol=0, atol=0)
    assert not any(isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in encoder.modules())
    assert not list(encoder.buffers())  # No moving statistics that drift between modes.


@pytest.mark.parametrize("training", [True, False])
@torch.no_grad()
def test_encoding_is_independent_of_other_episodes_and_accepts_one_clip(encoder, training):
    encoder.train(training)
    image = torch.randn(1, 24, 96, 96)
    unrelated = torch.randn(3, 24, 96, 96) * 5 + 11
    alone = encoder(image)
    together = encoder(torch.cat((image, unrelated)))[:1]
    assert alone.shape == (1, 32) and torch.isfinite(alone).all()
    torch.testing.assert_close(alone, together, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("training", [True, False])
def test_future_pixels_and_other_episodes_cannot_change_or_receive_gradient_from_past(encoder, training):
    encoder.train(training)
    frames = (torch.rand(2, 65, 96, 96, 3) * 255).requires_grad_()
    codes = encode_temporal(encoder, causal_clips(frames))
    altered = frames.detach().clone()
    altered[:, 15:] = 255 - altered[:, 15:]
    altered[1] = torch.rand_like(altered[1]) * 255
    with torch.no_grad():
        changed_codes = encode_temporal(encoder, causal_clips(altered))
    # The first code's history ends at raw frame14. Changed future pixels and
    # another episode must not affect it, even when training gradients are enabled.
    torch.testing.assert_close(codes[0, 0], changed_codes[0, 0], rtol=0, atol=0)
    assert not torch.allclose(codes[0, -1], changed_codes[0, -1])
    gradient, = torch.autograd.grad(codes[0, 0].square().sum(), frames)
    assert gradient[0, :15].abs().sum() > 0
    assert torch.count_nonzero(gradient[0, 15:]) == 0
    assert torch.count_nonzero(gradient[1]) == 0


def test_temporal_calls_keep_offsets_separate_and_gradients_attached():
    class RecordingEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, images):
            self.calls.append(images.detach().clone())
            return images.mean((1, 2, 3))[:, None].expand(-1, 32)

    clips = torch.arange(5.)[None, :, None, None, None].expand(2, -1, 24, 2, 3).clone().requires_grad_()
    encoder = RecordingEncoder()
    codes = encode_temporal(encoder, clips)
    assert codes.shape == (2, 5, 32)
    assert len(encoder.calls) == 5
    for offset, call in enumerate(encoder.calls):
        assert call.shape == (2, 24, 2, 3)
        torch.testing.assert_close(call, clips[:, offset])
    codes[:, -1].sum().backward()
    assert clips.grad[:, -1].abs().sum() > 0
    assert torch.count_nonzero(clips.grad[:, :-1]) == 0


@torch.no_grad()
def test_final_latent_output_is_not_normalized(encoder):
    final = encoder.projector[-1]
    final.weight.zero_()
    expected = torch.arange(32.) + 3
    final.bias.copy_(expected)
    codes = encoder(torch.randn(1, 24, 96, 96))
    torch.testing.assert_close(codes[0], expected, rtol=0, atol=0)


def current_interface():
    return {"format_version": CHECKPOINT_FORMAT_VERSION, **geometry_interface(),
            **encoder_interface(), "architecture": {"encoder": ENCODER_ARCHITECTURE}}


@torch.no_grad()
def test_checkpoint_roundtrip_preserves_batch_independent_inference(encoder, tmp_path):
    clips = torch.randn(1, 2, 24, 96, 96)
    expected = encode_temporal(encoder.eval(), clips)
    checkpoint = {"interface": current_interface(), "encoder": encoder.state_dict()}
    path = tmp_path / "groupnorm.pt"
    torch.save(checkpoint, path)
    saved = torch.load(path, weights_only=True)
    validate_checkpoint_interface(saved)
    reloaded = Encoder().train()
    reloaded.load_state_dict(saved["encoder"])
    torch.testing.assert_close(encode_temporal(reloaded, clips), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mutation, message", [
    ({"format_version": 4}, "version 5.*BatchNorm checkpoints"),
    ({"architecture": {"encoder": "resnet18_24channel_32latent"}}, "encoder architecture"),
    ({"encoder_normalization": {"backbone": "batch_norm"}}, "encoder_normalization"),
])
def test_checkpoint_rejects_previous_or_mislabelled_encoder(mutation, message):
    interface = copy.deepcopy(current_interface())
    interface.update(mutation)
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_interface({"interface": interface})
