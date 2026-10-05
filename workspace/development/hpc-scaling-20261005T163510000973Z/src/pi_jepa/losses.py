"""Joint JEPA and physical losses; neither endpoint is detached.

SIGReg is adapted from Lucas Maes's MIT-licensed LeWorldModel module.py,
commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac. See THIRD_PARTY.md and
licenses/LeWorldModel-MIT.txt for attribution and the original license.
"""

import torch
from torch import nn

from pi_jepa.models import scale_readout
from pi_jepa.physics import iota


class SIGReg(nn.Module):
    """Reference Epps–Pulley statistic over B independent trajectories per offset.

    Input is [B,6,32], never [B*6,32]. The same 1024 unit directions are used
    for all six offsets. Doubled trapezoid weights integrate both signs of t:
    endpoints dt and interior 2*dt on 17 knots over [0,3], with no additional
    quadrature normalization. The reference statistic is multiplied by B.
    """

    def __init__(self):
        super().__init__()
        self.num_proj = 1024
        t = torch.linspace(0, 3, 17, dtype=torch.float32)
        dt = 3 / 16
        weights = torch.full((17,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z, generator=None):
        if z.ndim != 3 or z.shape[1:] != (6, 32):
            raise ValueError("SIGReg expects [B,6,32], keeping trajectories independent")
        # An explicit CPU generator gives paired runs the same direction stream.
        # Float32 is enforced even if a caller enables autocast for the vision net.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            directions = torch.randn(32, self.num_proj, generator=generator, dtype=torch.float32)
            directions = directions / directions.norm(p=2, dim=0)
            directions = directions.to(z.device)
            x_t = (z @ directions).unsqueeze(-1) * self.t.float()
            # [B,6,1024,17] -> [6,1024,17], averaging only independent B.
            error = (x_t.cos().mean(0) - self.phi.float()).square() + x_t.sin().mean(0).square()
            return ((error @ self.weights.float()) * z.shape[0]).mean()


def jepa_loss(z, predictor, blocks, theta, sigreg, generator=None):
    """Five teacher-forced transitions; force block k is u[e_k:e_k+10].

    Each call gets just its last 1–3 observed codes. This removes indirect
    access to older codes through a multilayer transformer's earlier tokens.
    Returns (L_JEPA, L_pred, L_SIGReg); targets remain in the autograd graph.
    """
    if z.ndim != 3 or z.shape[1:] != (6, 32):
        raise ValueError("Expected codes [B,6,32]")
    if getattr(predictor, "action_free", False):
        if blocks is not None or theta is not None:
            raise ValueError("Passive prediction accepts no action or parameter conditioning")
        predicted = predictor(z[:, :-1])
    else:
        if blocks is None or blocks.shape != (z.shape[0], 5, 10):
            raise ValueError("Expected action blocks [B,5,10]")
        predictions = []
        for step in range(5):
            start = max(0, step - 2)
            prediction = predictor(z[:, start:step + 1], blocks[:, start:step + 1], theta)
            predictions.append(prediction[:, -1])
        predicted = torch.stack(predictions, dim=1)
    prediction_loss = (predicted - z[:, 1:]).square().mean()
    regularization = sigreg(z, generator=generator)
    return prediction_loss + 0.1 * regularization, prediction_loss, regularization


def physics_loss(decoded, simulated_states):
    """Mean fixed-unit residual, differentiable through readout and simulator."""
    return (scale_readout(decoded) - iota(simulated_states)).square().mean()


def simulate_window(initial_state, theta, prefix_forces, raw_endpoints, checkpoint_records=20):
    """Integrate from the real episode reset, then gather each crop's endpoints.

    Prefix forces start at raw record zero; shorter episodes are zero-padded only
    after their last requested endpoint. Checkpointing 20-record solver segments
    bounds the saved backward graph without changing states, forces, or loss.
    No integration step straddles a recorded-interval force discontinuity.
    """
    from torch.utils.checkpoint import checkpoint
    from pi_jepa.physics import rollout

    state = initial_state
    chunks = [state[:, None]]
    for start in range(0, prefix_forces.shape[1], checkpoint_records):
        forces = prefix_forces[:, start:start + checkpoint_records]
        if torch.is_grad_enabled() and (state.requires_grad or theta.requires_grad):
            segment = checkpoint(rollout, state, theta, forces, use_reentrant=False)
        else:
            segment = rollout(state, theta, forces)
        chunks.append(segment[:, 1:])
        state = segment[:, -1]
    states = torch.cat(chunks, dim=1)
    return states.gather(1, raw_endpoints[..., None].expand(-1, -1, 4))
