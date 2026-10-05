"""Cheap independent physical checks and isotropic camera margins."""
import numpy as np

from pi_jepa.data import CAMERA, CART_LIMIT, require_visible, world_to_pixel
from pi_jepa.data_qa import numerical_checks
from pi_jepa.physics import ELL


def test_energy_and_integration_precision():
    report = numerical_checks()
    assert max(report['rk4_2ms_vs_1ms_max_absolute_error_p_v_q_w']) < 1e-5
    np.testing.assert_allclose(report['nominal_lqr_gain'], [-4.472135955, -6.706516738, 55.231379049, 15.433956187], atol=1e-8)


def test_fixed_rod_pixel_length_and_bob_visibility():
    angles = np.linspace(-np.pi, np.pi, 101)
    for p in (-CART_LIMIT, 0., CART_LIMIT):
        origin = np.asarray(world_to_pixel(p, 0.))[:, None]
        bob = np.asarray(world_to_pixel(p + ELL*np.sin(angles), -ELL*np.cos(angles)))
        np.testing.assert_allclose(np.linalg.norm(bob - origin, axis=0), ELL*CAMERA['pixels_per_metre'], atol=1e-12)
        assert np.min(bob) - 2.5 > 0
        assert np.max(bob) + 2.5 < 95
        states = np.stack((np.full_like(angles, p), np.zeros_like(angles), angles, np.zeros_like(angles)), axis=-1)
        assert require_visible(states) > 2.
