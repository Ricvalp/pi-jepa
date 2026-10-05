"""Review of held-out target availability and calibration action discontinuities."""
import numpy as np
import torch

from pi_jepa.evaluate_latents import ENDPOINTS, FORECAST_START, HORIZONS
from pi_jepa.physics import rk4, rollout


def test_horizons_are_after_three_causal_histories():
    np.testing.assert_array_equal(ENDPOINTS[:3], [14, 24, 34])
    assert FORECAST_START == 34
    assert [ENDPOINTS[2 + round(t / .1)] for t in HORIZONS] == [44, 54, 74, 114, 194, 354]
    assert all(ENDPOINTS[i + 1] - ENDPOINTS[i] == 10 for i in range(len(ENDPOINTS) - 1))


def test_force_discontinuities_are_exact_record_interval_boundaries():
    initial = torch.tensor([0., 0., .3, 0.], dtype=torch.float64)
    theta = torch.tensor([1., .25], dtype=torch.float64)
    forces = torch.tensor([2., 2., -3., -3.], dtype=torch.float64)
    actual = rollout(initial, theta, forces, dt=.01, substeps=5)
    expected = [initial]
    for force in forces:
        state = expected[-1]
        for _ in range(5):
            state = rk4(state, theta, force, dt=.002, substeps=1)
        expected.append(state)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=1e-14)
