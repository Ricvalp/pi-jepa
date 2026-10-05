"""Small graph/freeze checks, not preliminary training experiments."""
import math

import torch

from pi_jepa.losses import SIGReg, jepa_loss, physics_loss
from pi_jepa.models import Encoder, PhysicalReadout, Predictor
from pi_jepa.physics import rollout
from pi_jepa.train import InitialConditions, parameter_groups, same_state


def test_joint_backward_reaches_both_endpoints_and_initial_conditions():
    torch.manual_seed(3)
    torch.set_num_threads(2)
    encoder, predictor, readout = Encoder(), Predictor(), PhysicalReadout()
    table = InitialConditions([0, 2])
    z = encoder(torch.randn(12, 24, 96, 96)).reshape(2, 6, 32)
    z.retain_grad()
    forces = torch.linspace(-0.7, 0.9, 128).reshape(2, 64)
    theta = torch.tensor([[0.8, 0.1], [1.2, 0.4]])
    from pi_jepa.data import action_blocks
    loss, _, _ = jepa_loss(z, predictor, action_blocks(forces), theta, SIGReg(), torch.Generator().manual_seed(5))
    simulated = rollout(table(torch.arange(2)), theta, forces)
    physical = physics_loss(readout(z), simulated[:, [14, 24, 34, 44, 54, 64]])
    # Isolate physical encoder gradients to ensure joint shaping is connected.
    grad = torch.autograd.grad(physical, encoder.backbone.conv1.weight, retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    (loss + physical).backward()
    for module in (encoder, readout, table):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    assert z.grad[:, -1].abs().sum() > 0  # Targets were not detached.


def test_frozen_encoder_parameters_and_buffers_are_unchanged():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    encoder, predictor, readout = Encoder(), Predictor(), PhysicalReadout()
    for model in (encoder, predictor): model.requires_grad_(False).eval()
    snapshots = [{k: v.clone() for k, v in model.state_dict().items()} for model in (encoder, predictor)]
    with torch.no_grad(): z = encoder(torch.randn(2, 24, 96, 96))
    optimizer = torch.optim.AdamW(parameter_groups(encoder, predictor, readout, "readout", 0.05), lr=3e-4)
    optimizer.zero_grad()
    readout(z).square().mean().backward(); optimizer.step()
    assert same_state(encoder, snapshots[0]) and same_state(predictor, snapshots[1])
    assert all(p.grad is None for model in (encoder, predictor) for p in model.parameters())


def test_initial_conditions_use_only_coarse_reset_prior():
    table = InitialConditions([0, 2])
    torch.testing.assert_close(table(torch.arange(2)), torch.tensor([[0.,0.,0.,0.], [0.,0.,math.pi,0.]]))


def test_cropped_physics_keeps_the_actual_episode_reset_and_matches_gradients():
    from pi_jepa.losses import simulate_window
    theta = torch.tensor([[1., .25], [.8, .1]])
    force = torch.linspace(-.8, .9, 168).reshape(2, 84)
    endpoints = torch.tensor([[14,24,34,44,54,64], [34,44,54,64,74,84]])
    initial = torch.tensor([[0., .2, 1.5, -.3], [0., -.1, 2.1, .4]], requires_grad=True)
    expected = rollout(initial, theta, force).gather(1, endpoints[...,None].expand(-1,-1,4))
    actual = simulate_window(initial, theta, force, endpoints)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.square().mean(), initial, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.square().mean(), initial)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual[1,0,0] != 0  # A crop is not a newly centered cart reset.


def test_initial_angle_can_represent_nonlinear_resets():
    table = InitialConditions([0,1,2])
    with torch.no_grad():
        table.raw[:,1] = torch.atanh(torch.tensor(2.6 / math.pi))
    state = table(torch.arange(3))
    torch.testing.assert_close(state[:,2], torch.tensor([2.6,2.6,math.pi+2.6]))
