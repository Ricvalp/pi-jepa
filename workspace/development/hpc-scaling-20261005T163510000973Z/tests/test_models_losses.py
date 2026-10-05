"""Checks of the statistical convention, causal windows, and gradient paths."""

import math

import pytest
import torch
from torch import nn

from pi_jepa.losses import SIGReg, jepa_loss, physics_loss
from pi_jepa.models import Encoder, PhysicalReadout, Predictor, PassivePredictor, scale_readout, to_state


def test_sigreg_matches_reference_independent_offset_statistic():
    torch.manual_seed(31)
    z = torch.randn(7, 6, 32, requires_grad=True)
    regularizer = SIGReg()
    actual = regularizer(z, torch.Generator().manual_seed(8))
    directions = torch.randn(32, 1024, generator=torch.Generator().manual_seed(8))
    directions /= directions.norm(dim=0)
    t = torch.linspace(0, 3, 17)
    target = (-t.square() / 2).exp()
    # Independently evaluate each offset with trapezoidal quadrature. The factor
    # 2 accounts for negative t, and 7 is B rather than B*6.
    per_offset = []
    for offset in range(6):
        phase = (z[:, offset] @ directions)[..., None] * t
        error = (phase.cos().mean(0) - target).square() + phase.sin().mean(0).square()
        per_offset.append(7 * 2 * torch.trapezoid(error * target, t).mean())
    expected = torch.stack(per_offset).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_predictor_causal_mask_and_context_limit():
    torch.manual_seed(12)
    predictor = Predictor().eval()
    # Gates are initially zero, so test causality after making attention active.
    for block in predictor.blocks:
        nn.init.normal_(block.modulation[-1].weight, std=0.04)
    z = torch.randn(2, 3, 32)
    forces = torch.randn(2, 3, 10)
    theta = torch.tensor([[0.7, 0.05], [1.3, 0.5]])
    expected = predictor(z, forces, theta)
    changed_z, changed_forces = z.clone(), forces.clone()
    changed_z[:, 2] += 5
    changed_forces[:, 2] -= 3
    actual = predictor(changed_z, changed_forces, theta)
    torch.testing.assert_close(expected[:, :2], actual[:, :2])
    assert not torch.allclose(expected[:, 2], actual[:, 2])
    with pytest.raises(ValueError, match="1 <= L <= 3"):
        predictor(torch.zeros(2, 4, 32), torch.zeros(2, 4, 10), theta)


def test_teacher_forcing_windows_and_target_gradients():
    class RecordingPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.windows = []

        def forward(self, z, forces, theta):
            self.windows.append((z[0, :, 0].tolist(), forces[0, :, 0].tolist()))
            return z * 0

    class ZeroReg(nn.Module):
        def forward(self, z, generator=None):
            return z.sum() * 0

    z = torch.arange(6.0)[None, :, None].expand(2, -1, 32).clone().requires_grad_()
    blocks = torch.arange(5.0)[None, :, None].expand(2, -1, 10)
    predictor = RecordingPredictor()
    total, _, _ = jepa_loss(z, predictor, blocks, torch.ones(2, 2), ZeroReg())
    total.backward()
    expected = [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert predictor.windows == [(window, window) for window in expected]
    # The final code only acts as a target, so this detects detached targets.
    assert z.grad[:, -1].abs().sum() > 0


def test_encoder_readout_shapes_and_physics_gradient():
    encoder, readout = Encoder(), PhysicalReadout()
    z = encoder(torch.randn(2, 24, 96, 96))
    assert z.shape == (2, 32)
    decoded = readout(z)
    assert decoded.shape == (2, 5)
    torch.testing.assert_close(decoded[:, 2:4].norm(dim=-1), torch.ones(2))
    simulated = torch.tensor([[0, 0, 0.2, 0], [0, 0, math.pi + 0.02, 0]], requires_grad=True)
    physics_loss(decoded, simulated).backward()
    assert encoder.backbone.conv1.weight.grad.abs().sum() > 0
    assert readout.net[0].weight.grad.abs().sum() > 0
    assert simulated.grad.abs().sum() > 0
    assert encoder.backbone.avgpool.output_size == (3, 3)


def test_angular_readout_is_periodic():
    q = torch.tensor([-math.pi + 0.01, math.pi - 0.01])
    decoded = torch.stack([torch.ones(2) * 2, torch.ones(2) * 4, q.sin(), q.cos(), torch.ones(2) * 5], -1)
    torch.testing.assert_close(to_state(decoded)[:, 2], q)
    torch.testing.assert_close(scale_readout(decoded)[:, [0, 1, 4]], torch.tensor([[1., 2., 1.], [1., 2., 1.]]))
    simulated = to_state(decoded)
    shifted = simulated + torch.tensor([0, 0, 2 * math.pi, 0])
    torch.testing.assert_close(physics_loss(decoded, shifted), torch.tensor(0.), atol=1e-12, rtol=0)


def test_passive_predictor_is_unconditioned_and_targets_stay_attached():
    predictor = PassivePredictor()
    z = torch.randn(3, 6, 32, requires_grad=True)
    assert predictor(z).shape == z.shape
    loss, _, _ = jepa_loss(z, predictor, None, None, SIGReg(), torch.Generator().manual_seed(2))
    loss.backward()
    assert z.grad[:, -1].abs().sum() > 0
    with pytest.raises(TypeError):
        predictor(z, torch.zeros(3, 5, 10), torch.ones(3, 2))
    with pytest.raises(ValueError, match="no action"):
        jepa_loss(z, predictor, torch.zeros(3, 5, 10), None, SIGReg())
