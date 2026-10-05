"""Check diagnostic reference scales, conditioning, and absence of side effects."""
import math

import numpy as np
import pytest
import torch
from torch import nn

from pi_jepa.models import Predictor
from pi_jepa.training_diagnostics import (
    gradient_metrics, prediction_diagnostics, predicted_physics_metrics, readout_metrics,
    representation_metrics,
)


class ActionPredictor(nn.Module):
    """Known dynamics: one latent increment equals the next force-block mean."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, z, forces, theta):
        self.calls.append((z.clone(), forces.clone(), theta.clone()))
        return z + forces.mean(-1, keepdim=True)


def motion_batch():
    slope = torch.arange(1.0, 5.0)
    z = slope[:, None, None] * torch.arange(6.0)[None, :, None]
    z = z.expand(-1, -1, 32).clone()
    forces = slope[:, None].expand(-1, 64).clone()
    theta = torch.tensor([[0.8, 0.1], [0.8, 0.1], [1.2, 0.4], [1.2, 0.4]])
    modes, apparatus = torch.tensor([0, 0, 2, 2]), torch.tensor([0, 0, 1, 1])
    return z, forces, theta, modes, apparatus


def test_representation_collapse_is_distinct_from_temporally_static_latents():
    collapsed = representation_metrics(torch.zeros(4, 6, 32, requires_grad=True))
    assert collapsed["latent/std_mean"] == 0
    assert collapsed["latent/low_std_fraction"] == 1
    assert collapsed["latent/effective_rank"] == 0
    assert math.isnan(collapsed["latent/temporal_to_between_variance"])

    # Full-rank across trajectories, but no motion whatsoever within each one.
    static = torch.cat([torch.eye(32), -torch.eye(32)])[:, None].expand(-1, 6, -1)
    metrics = representation_metrics(static)
    assert metrics["latent/effective_rank"] == pytest.approx(32)
    assert metrics["latent/std_mean"] == pytest.approx(1 / math.sqrt(32))
    assert metrics["latent/between_trajectory_variance"] == pytest.approx(1 / 32)
    assert metrics["latent/within_trajectory_variance"] == 0
    assert metrics["latent/temporal_delta_mse"] == 0


def test_representation_does_not_count_time_offsets_as_independent_trajectories():
    # At a given time all trajectories share a code. Flattening B*T would
    # incorrectly report healthy spread and effective rank from time variation.
    z = torch.eye(32)[:6][None].expand(4, -1, -1)
    metrics = representation_metrics(z)
    assert metrics["latent/std_mean"] == 0
    assert metrics["latent/effective_rank"] == 0
    assert metrics["latent/within_trajectory_variance"] > 0
    assert metrics["latent/temporal_delta_mse"] > 0
    assert math.isnan(metrics["latent/temporal_to_between_variance"])


def test_readout_residual_uses_physical_units_and_periodic_angle():
    decoded = torch.tensor([2., 4., 1., 4., 5.]).expand(2, 6, -1).requires_grad_()
    simulated = torch.zeros(2, 6, 4)
    metrics = readout_metrics(decoded, simulated)
    for name, expected in zip(("p", "v", "sin", "cos", "w"), (1, 4, 1, 9, 1)):
        assert metrics[f"physics/residual_{name}"] == pytest.approx(expected)
        assert metrics[f"readout/std_{name}"] == 0
    simulated[..., 2] += 2 * math.pi
    periodic = readout_metrics(decoded, simulated)
    assert periodic == pytest.approx(metrics, abs=1e-6)
    assert decoded.grad is None
    assert all(key.startswith("readout/") for key in readout_metrics(decoded))


def test_predicted_physics_scales_coordinates_and_averages_trajectories_and_horizons():
    readout = nn.Linear(32, 5, bias=False).eval()
    with torch.no_grad():
        readout.weight.zero_()
        readout.weight[:, :5] = torch.eye(5)
    simulated = torch.tensor([
        [[1., 2., 0., 3.], [2., 3., .5, 4.], [3., 4., 1., 5.]],
        [[-1., -2., -.5, -3.], [-2., -3., -1., -4.], [-3., -4., -1.5, -5.]],
    ])
    raw_target = torch.stack((simulated[..., 0], simulated[..., 1],
                              simulated[..., 2].sin(), simulated[..., 2].cos(),
                              simulated[..., 3]), -1)
    # Independent factors give known squared-error means: batch=(1²+3²)/2=5;
    # horizons=(1²,2²,4²); coordinate=(1²,...,5²). Undo fixed physical scaling.
    scaled_errors = (torch.tensor([1., 3.])[:, None, None]
                     * torch.tensor([1., 2., 4.])[None, :, None]
                     * torch.arange(1., 6.)[None, None, :])
    predicted = torch.zeros(2, 3, 32)
    predicted[..., :5] = raw_target + scaled_errors * torch.tensor([2., 2., 1., 1., 5.])
    metrics = predicted_physics_metrics(readout, predicted, simulated)
    assert len(metrics) == 24
    for horizon, multiplier in enumerate((5., 20., 80.), start=1):
        prefix = f"physics/autoregressive/h{horizon}/"
        assert metrics[prefix + "mse"] == pytest.approx(11 * multiplier)
        for coordinate, name in enumerate(("p", "v", "sin", "cos", "w"), start=1):
            assert metrics[prefix + f"residual_{name}"] == pytest.approx(coordinate**2 * multiplier)
    assert metrics["physics/autoregressive/mse"] == pytest.approx(385.)
    for coordinate, name in enumerate(("p", "v", "sin", "cos", "w"), start=1):
        assert metrics[f"physics/autoregressive/residual_{name}"] == pytest.approx(35 * coordinate**2)
    simulated[..., 2] += 2 * math.pi
    assert predicted_physics_metrics(readout, predicted, simulated) == pytest.approx(metrics, abs=1e-5)


def test_predicted_physics_preserves_gradients_rng_buffers_and_input_tensors():
    class StatefulReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(32, 5), nn.BatchNorm1d(5), nn.Dropout(.5))

        def forward(self, z):
            return self.net(z.flatten(0, 1)).reshape(*z.shape[:2], 5)

    readout = StatefulReadout().eval()
    predicted = torch.randn(2, 3, 32, requires_grad=True)
    simulated = torch.randn(2, 3, 4, requires_grad=True)
    next(readout.parameters()).grad = torch.ones_like(next(readout.parameters()))
    state = {key: value.clone() for key, value in readout.state_dict().items()}
    gradients = {name: value.grad.clone() if value.grad is not None else None
                 for name, value in readout.named_parameters()}
    inputs = (predicted.clone(), simulated.clone())
    rng = torch.random.get_rng_state().clone()
    metrics = predicted_physics_metrics(readout, predicted, simulated)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert not any(module.training for module in readout.modules())
    for key, value in readout.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    for name, parameter in readout.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    for actual, original in zip((predicted, simulated), inputs):
        assert actual.grad is None
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert all(isinstance(value, float) for value in metrics.values())
    readout.train()
    with pytest.raises(ValueError, match="already in eval mode"):
        predicted_physics_metrics(readout, predicted, simulated)


@pytest.mark.parametrize("latent_shape,state_shape", [
    ((2, 6, 32), (2, 6, 4)),  # Observed context must not dilute the forecast metric.
    ((0, 3, 32), (0, 3, 4)),
    ((2, 3, 32), (2, 6, 4)),  # Explicit future-only state alignment is required.
])
def test_predicted_physics_rejects_wrong_horizons_or_empty_batches(latent_shape, state_shape):
    with pytest.raises(ValueError, match="Expected"):
        predicted_physics_metrics(nn.Linear(32, 5).eval(), torch.zeros(latent_shape),
                                  torch.zeros(state_shape))


def test_gradient_metrics_include_missing_gradients_and_preserve_them():
    linear = nn.Linear(2, 2)
    linear(torch.tensor([[3., 4.]])).sum().backward()
    weight_gradient = linear.weight.grad.clone()
    bias_gradient = linear.bias.grad.clone()
    # One zero and one missing gradient must count in the denominator.
    linear.weight.grad[0, 0] = 0
    weight_gradient[0, 0] = 0
    linear.bias.grad = None
    metrics = gradient_metrics({"test": linear, "empty": nn.Identity()})
    assert metrics["gradients/test/norm"] == pytest.approx(math.sqrt(41))
    assert metrics["gradients/test/nonzero_fraction"] == pytest.approx(3 / 6)
    assert metrics["gradients/empty/norm"] == 0
    assert metrics["gradients/empty/nonzero_fraction"] == 0
    torch.testing.assert_close(linear.weight.grad, weight_gradient)
    assert linear.bias.grad is None
    assert bias_gradient.tolist() == [1, 1]


def test_prediction_scores_match_exact_action_dynamics_and_pooled_baselines():
    predictor = ActionPredictor().eval()
    batch = motion_batch()
    metrics, plots = prediction_diagnostics(predictor, *batch)
    teacher = "prediction/teacher_forced/all/"
    assert metrics[teacher + "mse_correct"] == 0
    assert metrics[teacher + "mse_persistence"] == pytest.approx(7.5)
    assert metrics[teacher + "mse_shuffled"] == 3
    assert metrics[teacher + "persistence_skill"] == 1
    assert metrics[teacher + "action_advantage"] == pytest.approx(3 / 7.5)
    assert metrics[teacher + "predicted_motion_ratio"] == 1
    assert metrics["prediction/teacher_forced/down/mse_persistence"] == 2.5
    assert metrics["prediction/teacher_forced/up/mse_persistence"] == 12.5
    for horizon in (1, 2, 3):
        prefix = f"prediction/autoregressive/h{horizon}/all/"
        assert metrics[prefix + "mse_correct"] == 0
        assert metrics[prefix + "mse_persistence"] == pytest.approx(7.5 * horizon**2)
        assert metrics[prefix + "sensitivity_mse"] == pytest.approx(3 * horizon**2)
        assert metrics[prefix + "action_advantage"] == pytest.approx(3 / 7.5)
    assert set(plots) == {"actual", "correct", "shuffled", "persistence"}
    assert all(isinstance(value, np.ndarray) and value.shape == (4, 3, 32)
               for value in plots.values())
    np.testing.assert_array_equal(plots["correct"], plots["actual"])
    # Each teacher-forced call uses the proper truncated chronological context.
    for step, (codes, actions, parameters) in enumerate(predictor.calls[:5]):
        start = max(0, step - 2)
        torch.testing.assert_close(codes, batch[0][:, start:step + 1])
        torch.testing.assert_close(actions[:, -1], batch[1][:, 14 + 10*step:24 + 10*step])
        torch.testing.assert_close(parameters, batch[2])
    # First shuffled AR call preserves both historical action blocks.
    first_shuffled = predictor.calls[13]
    torch.testing.assert_close(first_shuffled[1][:, :2], predictor.calls[10][1][:, :2])
    assert not torch.equal(first_shuffled[1][:, 2], predictor.calls[10][1][:, 2])
    torch.testing.assert_close(first_shuffled[2], batch[2])


def test_autoregression_never_observes_future_latents_and_zero_baseline_is_undefined():
    batch = list(motion_batch())
    predictor = ActionPredictor().eval()
    _, original = prediction_diagnostics(predictor, *batch)
    batch[0] = batch[0].clone()
    batch[0][:, 3:] += 1000
    _, changed = prediction_diagnostics(predictor, *batch)
    for key in ("correct", "shuffled", "persistence"):
        np.testing.assert_array_equal(original[key], changed[key])
    assert not np.array_equal(original["actual"], changed["actual"])
    batch[0].zero_()
    metrics, _ = prediction_diagnostics(predictor, *batch)
    prefix = "prediction/teacher_forced/all/"
    assert metrics[prefix + "mse_persistence"] == 0
    assert math.isnan(metrics[prefix + "persistence_skill"])
    assert math.isnan(metrics[prefix + "action_advantage"])
    assert math.isnan(metrics[prefix + "predicted_motion_ratio"])


def test_prediction_diagnostics_preserve_rng_buffers_modes_and_gradients():
    predictor = Predictor().eval()
    # Also exercise a stateful normalization buffer inside the actual predictor.
    class FeatureBatchNorm(nn.BatchNorm1d):
        def forward(self, x):
            return super().forward(x.transpose(1, 2)).transpose(1, 2)
    predictor.output_projection = nn.Sequential(predictor.output_projection,
                                               FeatureBatchNorm(32))
    predictor.eval()
    batch = list(motion_batch())
    batch[0].requires_grad_()
    next(predictor.parameters()).grad = torch.ones_like(next(predictor.parameters()))
    state = {key: value.clone() for key, value in predictor.state_dict().items()}
    gradients = {name: parameter.grad.clone() if parameter.grad is not None else None
                 for name, parameter in predictor.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    metrics, _ = prediction_diagnostics(predictor, *batch)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert not any(module.training for module in predictor.modules())
    for key, value in predictor.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    for name, parameter in predictor.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert batch[0].grad is None
    assert all(isinstance(value, float) for value in metrics.values())
    predictor.train()
    with pytest.raises(ValueError, match="already in eval mode"):
        prediction_diagnostics(predictor, *batch)


def test_shuffling_rejects_unmatched_singletons():
    batch = motion_batch()
    with pytest.raises(ValueError, match=">=2 trajectories"):
        prediction_diagnostics(ActionPredictor().eval(), *[value[:1] for value in batch])


def test_passive_diagnostics_have_no_action_or_readout_claims():
    from pi_jepa.models import PassivePredictor
    predictor = PassivePredictor().eval()
    z, _, _, modes, apparatus = motion_batch()
    metrics, plots = prediction_diagnostics(predictor, z, None, None, modes, apparatus)
    assert not any("shuffled" in key or "action" in key or "sensitivity" in key for key in metrics)
    assert "shuffled" not in plots
    assert "prediction/autoregressive/h3/nonlinear/mse_correct" in metrics
    assert math.isnan(metrics["prediction/autoregressive/h3/nonlinear/mse_correct"])
