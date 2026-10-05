"""Differentiable point-mass cart-pole, with q=0 down and q=pi upright.

Equations/convention: https://underactuated.mit.edu/acrobot.html (read 2026-10-01).
Cart drag is the additional generalized force -b*v. State order is [p,v,q,w].
"""

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

M_POLE = 0.2
ELL = 0.85
G = 9.81
INTEGRATION_DT = 0.002
DT = 0.010
ACTION_DT = 0.020
SUBSTEPS = 5


def rhs(x, theta, u):
    """Batched derivative; broadcast leading dimensions of x, theta and u."""
    _, v, q, w = x.unbind(-1)
    mass, drag = theta.unbind(-1)
    s, c = q.sin(), q.cos()
    acceleration = (u - drag * v + M_POLE * s * (ELL * w.square() + G * c)) / (
        mass + M_POLE * s.square()
    )
    angular_acceleration = -(G * s + c * acceleration) / ELL
    return torch.stack((v, acceleration, w, angular_acceleration), dim=-1)


def rk4(x, theta, u, dt=DT, substeps=SUBSTEPS):
    """Advance one recorded video interval while holding u constant."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = rhs(x, theta, u)
        k2 = rhs(x + h * k1 / 2, theta, u)
        k3 = rhs(x + h * k2 / 2, theta, u)
        k4 = rhs(x + h * k3, theta, u)
        x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def rollout(x0, theta, forces, dt=DT, substeps=SUBSTEPS):
    """Include x0: forces[...,k] advances output[...,k,:] to k+1."""
    states = [x0]
    for u in forces.unbind(-1):
        states.append(rk4(states[-1], theta, u, dt, substeps))
    return torch.stack(states, dim=-2)


def iota(x):
    """Fixed physical-unit embedding, including a periodic angular residual."""
    p, v, q, w = x.unbind(-1)
    return torch.stack((p / 2, v / 2, q.sin(), q.cos(), w / 5), dim=-1)


def angle_difference(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def lqr_gain(theta):
    """Continuous-time upright LQR; returns K with shape (1,4)."""
    mass, drag = np.asarray(theta, dtype=np.float64)
    a = np.array([
        [0, 1, 0, 0],
        [0, -drag / mass, M_POLE * G / mass, 0],
        [0, 0, 0, 1],
        [0, -drag / (mass * ELL), (mass + M_POLE) * G / (mass * ELL), 0],
    ])
    b = np.array([[0], [1 / mass], [0], [1 / (mass * ELL)]])
    q, r = np.diag([10.0, 1.0, 100.0, 5.0]), np.array([[0.5]])
    solution = solve_continuous_are(a, b, q, r)
    return np.linalg.solve(r, b.T @ solution)
