"""Capacity changes preserve causal interfaces and reconstruct from checkpoints."""
import copy

import pytest
import torch
from torch import nn

from pi_jepa.checkpoint_interface import (CHECKPOINT_FORMAT_VERSION, geometry_interface,
                                         model_interface, validate_checkpoint_interface)
from pi_jepa.evaluate_latents import load_networks
from pi_jepa.models import (MODEL_SIZES, Encoder, PassivePredictor, PhysicalReadout,
                            Predictor, model_kwargs, model_size, model_spec)


NEURAL_COUNTS = {
    "small": (12124000, 24864, 2052320),
    "medium": (22580768, 148256, 4821024),
    "large": (30805280, 821280, 16139552),
}


def config_for(size, dataset="passive"):
    return {"seed": 42, "model": {"size": size}, "dataset": dataset}


def checkpoint_interface(config, mode="joint"):
    return {"format_version": CHECKPOINT_FORMAT_VERSION, **geometry_interface(),
            **model_interface(config, mode)}


@pytest.mark.parametrize("size", MODEL_SIZES)
@torch.no_grad()
def test_each_capacity_preserves_per_clip_encoder_and_fixed_latent_interface(size):
    torch.manual_seed(21)
    encoder = Encoder(size)
    image = torch.randn(1, 24, 96, 96)
    trained = encoder.train()(image)
    inferred = encoder.eval()(image)
    assert inferred.shape == (1, 32)
    torch.testing.assert_close(trained, inferred, rtol=0, atol=0)
    unrelated = torch.randn_like(image) * 3 + 5
    batched = encoder(torch.cat((image, unrelated)))[:1]
    torch.testing.assert_close(inferred, batched, rtol=1e-5, atol=3e-6)
    assert not any(isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in encoder.modules())
    assert sum(parameter.numel() for parameter in encoder.parameters()) == NEURAL_COUNTS[size][0]
    final = encoder.projector[-1]
    final.weight.zero_()
    final.bias.fill_(3.)
    torch.testing.assert_close(encoder(image), torch.full((1, 32), 3.), rtol=0, atol=0)
    assert PhysicalReadout()(inferred).shape == (1, 5)
    assert sum(parameter.numel() for parameter in PhysicalReadout().parameters()) == 2437


@pytest.mark.parametrize("size", MODEL_SIZES)
def test_both_predictors_scale_with_unchanged_conditioning_and_causality(size):
    torch.manual_seed(33)
    passive = PassivePredictor(size)
    code = torch.randn(2, 32, requires_grad=True)
    passive(code).square().mean().backward()
    assert code.grad.abs().sum() > 0
    assert sum(p.numel() for p in passive.parameters()) == NEURAL_COUNTS[size][1]
    assert len([layer for layer in passive.net if isinstance(layer, nn.Linear)]) == model_spec(size)["passive_hidden_layers"] + 1
    predictor = Predictor(size).eval()
    for block in predictor.blocks:
        nn.init.normal_(block.modulation[-1].weight, std=.03)
    code = torch.randn(2, 3, 32, requires_grad=True)
    actions = torch.randn(2, 3, 10)
    theta = torch.tensor([[.8, .1], [1.2, .4]])
    prediction = predictor(code, actions, theta)
    assert prediction.shape == (2, 3, 32)
    changed_code, changed_actions = code.detach().clone(), actions.clone()
    changed_code[:, -1] += 5
    changed_actions[:, -1] -= 3
    modified = predictor(changed_code, changed_actions, theta)
    torch.testing.assert_close(prediction[:, :2], modified[:, :2])
    assert not torch.allclose(prediction[:, -1], modified[:, -1])
    prediction.square().mean().backward()
    assert code.grad.abs().sum() > 0
    assert sum(p.numel() for p in predictor.parameters()) == NEURAL_COUNTS[size][2]


@pytest.mark.parametrize("size,dataset", [("small", "passive"), ("medium", "controlled"), ("large", "passive")])
@torch.no_grad()
def test_inference_reconstructs_the_saved_capacity(size, dataset, tmp_path):
    config = config_for(size, dataset)
    encoder = Encoder(size).eval()
    predictor = (PassivePredictor(size) if dataset == "passive" else Predictor(size)).eval()
    image = torch.randn(1, 24, 96, 96)
    expected = encoder(image)
    checkpoint = {"config": config, "mode": "joint", "step": 17,
                  "encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                  "interface": checkpoint_interface(config)}
    path = tmp_path / f"{size}.pt"
    torch.save(checkpoint, path)
    actual_encoder, actual_predictor, info = load_networks(path, torch.device("cpu"))
    assert actual_encoder.size == actual_predictor.size == info["model_size"] == size
    assert not actual_encoder.training and not actual_predictor.training
    assert not any(p.requires_grad for p in actual_encoder.parameters())
    torch.testing.assert_close(actual_encoder(image), expected, rtol=0, atol=0)
    for name, value in predictor.state_dict().items():
        torch.testing.assert_close(actual_predictor.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("source,target", [("small", "medium"), ("medium", "large"), ("large", "small")])
def test_inference_rejects_capacity_metadata_from_a_different_model(source, target, tmp_path):
    saved = {"config": config_for(source), "mode": "joint",
             "interface": checkpoint_interface(config_for(target))}
    path = tmp_path / "conflicting.pt"
    torch.save(saved, path)
    # No weight entries are needed: mismatched size must fail before loading them.
    with pytest.raises(ValueError, match="model_size differs"):
        load_networks(path, torch.device("cpu"))


def test_checkpoint_rejects_changed_structure_or_predictor_capacity():
    config = config_for("medium", "controlled")
    saved = {"config": config, "mode": "joint", "interface": checkpoint_interface(config)}
    wrong_spec = copy.deepcopy(saved)
    wrong_spec["interface"]["model_spec"]["controlled_layers"] = 3
    with pytest.raises(ValueError, match="model_spec differs"):
        validate_checkpoint_interface(wrong_spec)
    wrong_architecture = copy.deepcopy(saved)
    wrong_architecture["interface"]["architecture"]["predictor"] = "passive_mlp_32_256x3_32"
    with pytest.raises(ValueError, match="architecture differs"):
        validate_checkpoint_interface(wrong_architecture)


def test_size_options_are_explicit_and_specs_do_not_share_mutable_state():
    assert model_size({}) == "small" and model_kwargs({}) == {}
    assert model_kwargs(config_for("large")) == {"size": "large"}
    first = model_spec("small")
    first["latent_dim"] = 99
    assert model_spec("small")["latent_dim"] == 32
    with pytest.raises(ValueError, match="model.size"):
        Encoder("unlisted")
