"""Equation, differentiation, periodic-angle, and causal-indexing checks."""

import numpy as np
import pytest
import torch

from pi_jepa.data import (CAMERA, CART_LIMIT, SCHEMA_VERSION, action_blocks,
    causal_clips, render, require_visible, sample_appearance, world_to_pixel,
    pulse_program, simulate_batch, validate_clocks, validate_manifest)
from pi_jepa.physics import ELL, G, M_POLE, angle_difference, iota, lqr_gain, rhs, rollout


def test_downward_equilibrium_and_forcing_signs():
    x = torch.zeros(2, 4, dtype=torch.float64)
    theta = torch.tensor([[1.0, 0.25], [0.7, 0.1]], dtype=x.dtype)
    assert torch.equal(rhs(x, theta, torch.zeros(2)), x)
    derivative = rhs(x, theta, torch.ones(2))
    assert torch.all(derivative[:, 1] > 0)
    assert torch.all(derivative[:, 3] < 0)


def test_rhs_matches_point_mass_matrix():
    x = torch.tensor([[0.3, 0.2, 0.5, 0.7], [-0.3, -0.4, 2.9, -0.5]], dtype=torch.float64)
    theta = torch.tensor([[0.75, 0.08], [1.4, 0.7]], dtype=x.dtype)
    forces = torch.tensor([1.2, -2.0], dtype=x.dtype)
    for state, parameters, force, derivative in zip(x, theta, forces, rhs(x, theta, forces)):
        _, v, q, w = state
        mass, drag = parameters
        matrix = torch.tensor([[mass + M_POLE, M_POLE * ELL * q.cos()], [M_POLE * ELL * q.cos(), M_POLE * ELL ** 2]])
        external = torch.stack((force - drag * v + M_POLE * ELL * w.square() * q.sin(), -M_POLE * G * ELL * q.sin()))
        expected = torch.linalg.solve(matrix, external)
        torch.testing.assert_close(derivative[[1, 3]], expected)


def test_rollout_gradients_reach_initial_state_and_parameters():
    initial = torch.tensor([[0.0, 0.1, 0.3, -0.2]], requires_grad=True)
    theta = torch.tensor([[1.0, 0.25]], requires_grad=True)
    forces = torch.linspace(-1, 1, 12)[None]
    states = rollout(initial, theta, forces)
    assert states.shape == (1, 13, 4)
    iota(states[:, -1]).square().sum().backward()
    assert initial.grad.abs().sum() > 0
    assert torch.all(theta.grad.abs() > 0)


def test_angle_boundary_and_lqr_linearization():
    residual = angle_difference(torch.tensor(-np.pi + 0.01), torch.tensor(np.pi - 0.01))
    torch.testing.assert_close(residual, torch.tensor(0.02), atol=1e-6, rtol=1e-5)
    theta = torch.tensor([1.0, 0.25], dtype=torch.float64)
    equilibrium = torch.tensor([0.0, 0.0, np.pi, 0.0], dtype=torch.float64)
    a = torch.autograd.functional.jacobian(lambda x: rhs(x, theta, torch.tensor(0.0)), equilibrium).numpy()
    b = torch.autograd.functional.jacobian(lambda u: rhs(equilibrium, theta, u), torch.tensor(0.0, dtype=torch.float64)).numpy()[:, None]
    assert np.max(np.linalg.eigvals(a - b @ lqr_gain(theta.numpy())).real) < 0


def test_clip_and_action_alignment():
    frames = torch.arange(65, dtype=torch.uint8)[None, :, None, None, None].expand(1, 65, 2, 2, 3)
    clips = causal_clips(frames)
    assert clips.shape == (1, 6, 24, 2, 2)
    chronological = ((clips[0, :, :, 0, 0] + 1) * 127.5).round().long()
    expected = torch.tensor([[e - 14 + 2 * k for k in range(8)] for e in (14, 24, 34, 44, 54, 64)])
    assert torch.equal(chronological, expected.repeat_interleave(3, dim=-1))
    blocks = action_blocks(torch.arange(64)[None])
    assert blocks.shape == (1, 5, 10)
    assert torch.equal(blocks[0], torch.arange(14, 64).reshape(5, 10))


def test_renderer_shape_dtype():
    appearance = sample_appearance(np.random.default_rng(42))
    frame = render([0.0, 0.0, 0.0, 0.0], appearance)
    assert frame.shape == (96, 96, 3)
    assert frame.dtype == np.uint8


def test_isotropic_projection_keeps_rod_length_and_visible_margin():
    for p in (-CART_LIMIT, 0., CART_LIMIT):
        pivot = np.asarray(world_to_pixel(p, 0.))
        for q in np.linspace(-np.pi, np.pi, 100):
            bob = np.asarray(world_to_pixel(p + ELL * np.sin(q), -ELL * np.cos(q)))
            np.testing.assert_allclose(np.linalg.norm(bob - pivot), ELL * CAMERA['pixels_per_metre'], atol=1e-12)
            assert np.all(bob - 2.5 >= 0) and np.all(bob + 2.5 < 96)
            assert require_visible([[p, 0., q, 0.]]) >= .5


def test_generation_holds_actions_and_preserves_finite_boundary_prefix():
    plan = {"initial": np.array([0., 4., .2, 0.]), "theta": np.array([1., .25]),
            "program": np.ones(100)}
    states, forces, reason = simulate_batch([plan])[0]
    assert reason == "boundary_exit"
    assert len(states) == len(forces) + 1
    assert np.all(np.abs(states[:-1, 0]) <= CART_LIMIT)
    assert abs(states[-1, 0]) > CART_LIMIT
    require_visible(states)
    np.testing.assert_array_equal(forces[0:len(forces)//2*2:2], forces[1:len(forces)//2*2:2])
    reference = rollout(torch.from_numpy(plan["initial"]), torch.from_numpy(plan["theta"]), torch.from_numpy(forces))
    np.testing.assert_allclose(states, reference.numpy(), rtol=1e-12, atol=1e-12)


def test_passive_forces_are_exactly_zero_and_angles_unwrapped():
    initial = np.array([0., .1, np.pi + .1, .3])
    states, force, reason = simulate_batch([{"initial": initial, "theta": np.array([1., .25]), "program": np.zeros(100)}])[0]
    assert np.array_equal(force, np.zeros_like(force))
    assert states[:, 2].max() > np.pi
    assert np.abs(np.diff(states[:, 2])).max() < .2
    assert reason == "duration"


def test_pulse_library_has_calibrated_nontrivial_holds():
    pulse = pulse_program(np.random.default_rng(42), updates=200, paired=True)
    assert pulse.shape == (200,)
    assert np.max(np.abs(pulse)) <= 3.
    assert np.any(pulse > 0) and np.any(pulse < 0)
    changes = np.r_[0, np.flatnonzero(np.diff(pulse) != 0) + 1, len(pulse)]
    assert np.diff(changes)[:-1].min() >= 5


def test_visibility_rejects_cropped_terminal_frame():
    # Valid cart position before the boundary, followed by a deliberately huge
    # overshoot: generation must fail rather than silently crop or drop it.
    states = [[CART_LIMIT - .01, 0., np.pi / 2, 0.], [2., 0., np.pi / 2, 0.]]
    with pytest.raises(ValueError, match='frame 1 violates camera visibility'):
        require_visible(states, 'overshoot')


def test_generation_contract_rejects_old_physics_camera_and_schema():
    from copy import deepcopy
    import json
    from pathlib import Path
    config = json.loads(Path('configs/base.json').read_text())
    clocks = validate_clocks(config)
    assert ELL == .85 and CART_LIMIT == 1.5 and SCHEMA_VERSION == 3
    assert clocks['record_dt_s'] == .01 and clocks['integration_dt_s'] == .002
    manifest = {'schema_version': SCHEMA_VERSION, 'physics': config['physics'],
                'collection_settings': config['data'], 'camera': CAMERA,
                'cart_limit_m': CART_LIMIT}
    validate_manifest(manifest)
    for field, value in (('ell', .5), ('dt', .02)):
        bad = deepcopy(config)
        bad['physics'][field] = value
        with pytest.raises(ValueError):
            validate_clocks(bad)
    for key, value in (('cart_limit_m', 2.), ('camera', {**CAMERA, 'x_limits_m': [-2.8, 2.8]})):
        bad = deepcopy(config)
        bad['data'][key] = value
        with pytest.raises(ValueError):
            validate_clocks(bad)
    for key, value in (('schema_version', 2), ('cart_limit_m', 2.),
                       ('camera', {**CAMERA, 'y_limits_m': [-.7, .7]})):
        bad = deepcopy(manifest)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_manifest(bad)
