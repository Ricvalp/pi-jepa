"""Native-step forecasting, unknown calibration position, and circular scores."""
import numpy as np
import torch

from pi_jepa.evaluate import fit_parameters, forecast
from pi_jepa.physics import iota, rollout
from pi_jepa.reporting import score


def test_forecast_uses_native_timing_and_each_context_tokens_actions():
    class Predictor:
        def __init__(self):
            self.calls = []
        def __call__(self, codes, actions, theta):
            self.calls.append((codes.clone(), actions.clone()))
            return codes + 1
    def readout(codes):
        zeros, ones = torch.zeros_like(codes[..., 0]), torch.ones_like(codes[..., 0])
        return torch.stack((codes[..., 0], zeros, zeros, ones, zeros), -1)
    predictor = Predictor()
    warm = torch.arange(3, dtype=torch.float32)[None, :, None].expand(1, 3, 32)
    actions = torch.arange(400, dtype=torch.float32)[None] / 100
    theta = torch.tensor([[1., .25]])
    learned, physical = forecast(predictor, readout, warm, actions, theta,
                                 {"physics": {"dt": .01, "substeps": 5}}, steps=4)
    assert learned.shape == physical.shape == (1, 4, 4)
    np.testing.assert_allclose(learned[0, :, 0], np.arange(3, 7))
    for step, (codes, blocks) in enumerate(predictor.calls):
        torch.testing.assert_close(codes[0, :, 0], torch.arange(step, step + 3, dtype=torch.float32))
        torch.testing.assert_close(blocks, actions[:, 14 + 10 * step:44 + 10 * step].reshape(1, 3, 10))
    expected = rollout(torch.tensor([[2., 0., 0., 0.]]), theta, actions[:, 34:74], .01, 5)[:, 10::10]
    np.testing.assert_allclose(physical, expected.numpy())


def test_calibration_fits_unknown_state_at_point14_not_commanded_reset():
    initial = torch.tensor([[.7, .1, .2, .1], [-.4, -.1, -.2, .1]])
    theta = torch.tensor([.9, .15])
    actions = [torch.ones(20), torch.ones(30) * -.3]
    targets = [iota(rollout(state, theta, forces, .01, 5)[::10]) for state, forces in zip(initial, actions)]
    fitted, states, losses, failure = fit_parameters(targets, actions, {"lr": .01, "updates": 2},
                                                     {"dt": .01, "substeps": 5})
    assert failure is None and losses.shape == (2,)
    assert not fitted.requires_grad and not states.requires_grad
    assert states[0, 0] > .6 and states[1, 0] < -.3
    assert torch.isfinite(states).all()
    assert not torch.equal(fitted, torch.tensor([1., .25]))


def test_physical_scores_wrap_angles_and_keep_nonfinite_failures():
    truth = np.array([[1., 2., np.pi - .01, 3.]])
    prediction = np.array([[1., 2., -np.pi + .01, 3.]])
    metrics = score(prediction, truth)
    assert np.isclose(metrics["angle_rmse_rad"], .02)
    assert metrics["position_rmse_m"] == metrics["velocity_rmse_m_s"] == 0
    prediction[0, 0] = np.nan
    assert score(prediction, truth)["nonfinite_states"] == 1
    assert np.isnan(score(prediction, truth)["normalized_rmse"])
