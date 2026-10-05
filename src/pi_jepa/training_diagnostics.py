"""Observational training diagnostics without hidden states or optimizer changes.

Latent statistics keep independent trajectories on the batch axis. Latent
prediction scores use known training/validation parameters, never fitted
parameters or the physical readout. Predicted physics diagnostics decode those
forecasts and compare them with the training simulator targets, using learned
resets or explicitly fixed true training resets. All routines
detach inputs; no random stream is consumed.
"""
from __future__ import annotations

import torch

from pi_jepa.models import scale_readout
from pi_jepa.physics import iota


def _ratio(numerator, denominator):
    """Leave an undefined reference scale visible rather than clamping it."""
    return float(numerator / denominator) if float(denominator) > 0 else float("nan")


@torch.no_grad()
def representation_metrics(z):
    """Population statistics for [trajectories, six offsets, 32 features].

    Standard deviations and covariance ranks are measured across trajectories
    independently at each offset. Effective rank is exp(entropy(eigenvalue
    shares)); a zero covariance has rank zero. Between variance is the average
    across-trajectory variance at fixed offsets; within variance is the average
    across-time variance within trajectories. Low standard deviation means <.1.
    """
    if z.ndim != 3 or z.shape[1:] != (6, 32) or len(z) == 0:
        raise ValueError("Expected nonempty latent codes [B,6,32]")
    z = z.detach().double()
    centered = z - z.mean(0, keepdim=True)
    variance = centered.square().mean(0)
    std = variance.sqrt()
    covariance = torch.einsum("btd,bte->tde", centered, centered) / len(z)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    totals = eigenvalues.sum(-1)
    shares = eigenvalues / totals.clamp_min(torch.finfo(z.dtype).tiny)[:, None]
    entropy = -(shares * shares.clamp_min(torch.finfo(z.dtype).tiny).log()).sum(-1)
    ranks = torch.where(totals > 0, entropy.exp(), torch.zeros_like(totals))
    between = variance.mean()
    temporal_delta = z.diff(dim=1).square().mean()
    values = {
        "std_mean": std.mean(), "std_min": std.min(), "std_max": std.max(),
        "low_std_fraction": (std < 0.1).double().mean(),
        "effective_rank": ranks.mean(),
        "between_trajectory_variance": between,
        "within_trajectory_variance": z.var(dim=1, unbiased=False).mean(),
        "temporal_delta_mse": temporal_delta,
        "temporal_to_between_variance": _ratio(temporal_delta, between),
        "z_norm_mean": z.norm(dim=-1).mean(),
    }
    return {f"latent/{key}": float(value) for key, value in values.items()}


@torch.no_grad()
def readout_metrics(decoded, simulated=None):
    """Readout spread and optionally its scaled physical residual per coordinate.

    The simulator input is the same initial-condition trajectory used in the
    loss. Learned resets measure self-consistency; only the explicit true-fixed
    training diagnostic supplies an independently fixed physical target.
    """
    if decoded.ndim != 3 or decoded.shape[1:] != (6, 5) or len(decoded) == 0:
        raise ValueError("Expected readouts [B,6,5]")
    decoded = decoded.detach().float()
    names = ("p", "v", "sin", "cos", "w")
    spread = decoded.flatten(0, 1).std(dim=0, unbiased=False)
    metrics = {f"readout/std_{name}": float(value) for name, value in zip(names, spread)}
    if simulated is not None:
        if simulated.shape != (*decoded.shape[:2], 4):
            raise ValueError("Expected simulated states [B,6,4]")
        residual = (scale_readout(decoded) - iota(simulated.detach().float())).square().mean((0, 1))
        metrics.update({f"physics/residual_{name}": float(value)
                        for name, value in zip(names, residual)})
    return metrics


@torch.no_grad()
def predicted_physics_metrics(readout, predicted, simulated):
    """Decode three AR predictions and compare with aligned training targets.

    Inputs cover only the future endpoints (crop records44,54,64); observed
    context is excluded. Means use the same five scaled coordinates as the
    training physics loss. This diagnostic never fits or updates any state.
    """
    if any(module.training for module in readout.modules()):
        raise ValueError("Predicted physics diagnostics require a readout already in eval mode")
    if predicted.ndim != 3 or predicted.shape[1:] != (3, 32) or len(predicted) == 0:
        raise ValueError("Expected predicted latents [B,3,32]")
    if simulated.shape != (*predicted.shape[:2], 4):
        raise ValueError("Expected aligned simulated future states [B,3,4]")
    decoded = readout(predicted.detach())
    squared = (scale_readout(decoded.float()) - iota(simulated.detach().float())).square()
    # [3,5]: average independent trajectories first, retaining horizon/coordinate.
    residuals = squared.mean(0)
    metrics = {}
    for suffix, values in [("", residuals.mean(0)),
                           *((f"/h{step + 1}", values) for step, values in enumerate(residuals))]:
        prefix = f"physics/autoregressive{suffix}"
        metrics[f"{prefix}/mse"] = float(values.mean())
        for name, value in zip(("p", "v", "sin", "cos", "w"), values):
            metrics[f"{prefix}/residual_{name}"] = float(value)
    return metrics


@torch.no_grad()
def gradient_metrics(named_modules):
    """L2 norm and nonzero fraction of parameter gradients before clipping.

    Absent gradients count as zero; a module with no parameters has zero norm
    and zero nonzero fraction. Existing gradients are never changed.
    """
    metrics = {}
    for name, module in named_modules.items():
        squared_norm, nonzero, total = 0.0, 0, 0
        if module is None:
            continue
        for parameter in module.parameters():
            total += parameter.numel()
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    gradient = gradient.to_dense()
                squared_norm += float(gradient.double().square().sum())
                nonzero += int(torch.count_nonzero(gradient))
        metrics[f"gradients/{name}/norm"] = squared_norm ** 0.5
        metrics[f"gradients/{name}/nonzero_fraction"] = nonzero / total if total else 0.0
    return metrics


def _donor_indices(apparatus_ids, reset_modes, device):
    """Cycle whole programs between fixed independent episodes, with no RNG use.

    Calibrated forces are transferable between apparatuses/reset families; donors
    need not have matching theta. This is a distribution diagnostic, not test
    intervention evidence (training collection can contain feedback actions).
    """
    count = len(apparatus_ids)
    if count < 2:
        raise ValueError("Action shuffling requires >=2 trajectories")
    return torch.arange(count, device=device).roll(-1)


def _teacher_forced(predictor, z, forces, theta):
    from pi_jepa.data import action_blocks
    if getattr(predictor, "action_free", False):
        return predictor(z[:, :-1])
    blocks = action_blocks(forces)
    predictions = []
    for step in range(5):
        start = max(0, step - 2)
        predictions.append(predictor(z[:, start:step + 1],
                                    blocks[:, start:step + 1], theta)[:, -1])
    return torch.stack(predictions, dim=1)


def _autoregressive(predictor, warm, forces, theta):
    from pi_jepa.data import action_blocks
    # Warm observations end at raw crop records14,24,34, then only predictions.
    passive = getattr(predictor, "action_free", False)
    context = list(warm.unbind(1))
    predictions = []
    blocks = None if passive else action_blocks(forces)
    for step in range(3):
        if passive:
            prediction = predictor(context[-1])
        else:
            prediction = predictor(torch.stack(context[-3:], dim=1),
                                   blocks[:, step:step + 3], theta)[:, -1]
        context.append(prediction)
        predictions.append(prediction)
    return torch.stack(predictions, dim=1)


def _score(actual, correct, shuffled, persistence):
    """Ratios of pooled MSEs, not averages of unstable per-episode ratios."""
    correct_error = (correct - actual).square().mean()
    baseline_error = (persistence - actual).square().mean()
    predicted_motion = (correct - persistence).square().mean()
    scores = {
        "mse_correct": float(correct_error), "mse_persistence": float(baseline_error),
        "persistence_skill": 1 - _ratio(correct_error, baseline_error),
        "predicted_motion_mse": float(predicted_motion),
        "actual_motion_mse": float(baseline_error),
        "predicted_motion_ratio": _ratio(predicted_motion, baseline_error),
    }
    if shuffled is not None:
        shuffled_error = (shuffled - actual).square().mean()
        scores.update({"mse_shuffled": float(shuffled_error),
                       "action_advantage": _ratio(shuffled_error - correct_error, baseline_error),
                       "sensitivity_mse": float((shuffled - correct).square().mean())})
    return scores


@torch.no_grad()
def prediction_diagnostics(predictor, z, forces, theta, reset_modes, apparatus_ids):
    """Fixed-window train/validation teacher forcing and 0.1/0.2/0.3s forecasts.

    AR observes histories ending at14,24,34 only. Controlled shuffling replaces
    future records34:64 with another episode's entire force suffix, preserving
    action blocks; past forces remain correct. Training feedback data may reveal
    hidden state through future actions, so these are learning diagnostics, not
    primary held-out intervention results. Passive predictions receive only z.
    Splits are all/down/nonlinear/up; missing families yield NaN, not zero error.
    """
    if any(module.training for module in predictor.modules()):
        raise ValueError("Prediction diagnostics require a predictor already in eval mode")
    if z.ndim != 3 or z.shape[1:] != (6, 32) or len(z) == 0:
        raise ValueError("Expected nonempty latent codes [B,6,32]")
    passive = getattr(predictor, "action_free", False)
    if not passive and (forces.shape != (len(z), 64) or theta.shape != (len(z), 2)):
        raise ValueError("Expected aligned window forces [B,64] and known parameters [B,2]")
    reset_modes = torch.as_tensor(reset_modes, device=z.device)
    if reset_modes.shape != (len(z),) or len(apparatus_ids) != len(z):
        raise ValueError("Expected one reset mode and apparatus ID per trajectory")
    if not torch.all((reset_modes >= 0) & (reset_modes <= 2)):
        raise ValueError("Reset modes must be0 down,1 nonlinear,2 up")
    z = z.detach()
    forces = None if passive else forces.detach()
    theta = None if passive else theta.detach()
    teacher_correct = _teacher_forced(predictor, z, forces, theta)
    teacher_shuffled = shuffled = None
    if not passive:
        donor_forces = forces[_donor_indices(apparatus_ids, reset_modes, z.device)]
        teacher_shuffled = _teacher_forced(predictor, z, donor_forces, theta)
        shuffled_forces = forces.clone()
        shuffled_forces[:, 34:64] = donor_forces[:, 34:64]
    actual = z[:, 3:]
    correct = _autoregressive(predictor, z[:, :3], forces, theta)
    if not passive:
        shuffled = _autoregressive(predictor, z[:, :3], shuffled_forces, theta)
    persistence = z[:, 2:3].expand(-1, 3, -1)
    metrics = {}
    splits = {"all": torch.ones(len(z), dtype=torch.bool, device=z.device),
              "down": reset_modes == 0, "nonlinear": reset_modes == 1, "up": reset_modes == 2}
    for split, mask in splits.items():
        scores = _score(z[mask, 1:], teacher_correct[mask],
                        None if passive else teacher_shuffled[mask], z[mask, :-1])
        metrics.update({f"prediction/teacher_forced/{split}/{key}": value for key, value in scores.items()})
        for horizon in range(3):
            scores = _score(actual[mask, horizon], correct[mask, horizon],
                            None if passive else shuffled[mask, horizon], persistence[mask, horizon])
            metrics.update({f"prediction/autoregressive/h{horizon + 1}/{split}/{key}": value
                            for key, value in scores.items()})
    plots = {name: value.float().cpu().numpy().copy() for name, value in
             {"actual": actual, "correct": correct, "shuffled": shuffled, "persistence": persistence}.items()
             if value is not None}
    return metrics, plots
