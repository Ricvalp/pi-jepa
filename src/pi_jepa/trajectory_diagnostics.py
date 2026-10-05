"""Fixed training-episode phase portraits, using learning files and current resets."""
import numpy as np
import torch

from pi_jepa.evaluate_latents import (ENDPOINTS, FORECAST_START, encode_endpoints,
                                      rollout_latents)
from pi_jepa.models import to_state
from pi_jepa.physics import DT, rollout


RESET_NAMES = ("Downward oscillation", "Nonlinear swing", "Near-upright reset")
COLORS = ("#171717", "#2374ba", "#d05a19")


def fixed_trajectory_reference(data):
    """Copy the first training episode of each available reset mode, capped at 3.54 s.

    Selection uses manifest order, never errors or episode duration. Only the
    learning dataset is accessed. Short episodes keep their actual endpoints;
    missing observations and actions are never padded or extrapolated.
    """
    modes = torch.as_tensor(data.reset_modes)
    reference = []
    for mode in range(3):
        indices = torch.nonzero(modes == mode).flatten()
        if not len(indices):
            continue
        item = data[int(indices[0])]
        available = len(item["frames"])
        if len(item["forces"]) != available - 1:
            raise ValueError("Trajectory diagnostics require one force per recorded frame interval")
        count = min(available, int(ENDPOINTS[-1]) + 1)
        endpoints = torch.as_tensor(ENDPOINTS[ENDPOINTS < count]).clone()
        theta = torch.tensor([1., .25]) if data.dataset == "passive" else item["theta"]
        reference.append({
            "trajectory_id": int(item["trajectory_id"]), "reset_mode": mode,
            "available_frames": available,
            "frames": torch.as_tensor(item["frames"][:count]).detach().cpu().clone(),
            "forces": torch.as_tensor(item["forces"][:count - 1], dtype=torch.float32).flatten().detach().cpu().clone(),
            "theta": torch.as_tensor(theta, dtype=torch.float32).detach().cpu().clone(),
            "endpoints": endpoints, "steps": max(0, len(endpoints) - 3),
        })
    return reference


def _phase_curve(ax, states, color, label, markers=False):
    """Break at the angular wrap; a jump from pi to -pi is not physical motion."""
    states = np.asarray(states)
    p, q = states[:, 0], (states[:, 2] + np.pi) % (2 * np.pi) - np.pi
    cuts = np.flatnonzero(np.abs(np.diff(q)) > np.pi) + 1
    for part, indices in enumerate(np.split(np.arange(len(q)), cuts)):
        ax.plot(p[indices], q[indices], color=color, linewidth=1.8,
                marker="o" if markers else None, markersize=3,
                label=label if part == 0 else None)
    ax.scatter(p[0], q[0], marker="o", facecolors="white", edgecolors=color, s=40, zorder=5)
    ax.scatter(p[-1], q[-1], marker="s", color=color, s=22, zorder=5)


@torch.no_grad()
def trajectory_phase_figure(encoder, predictor, readout, table, reference, config, device):
    """Plot r(E), a 32-step r(P) rollout, and the current simulator target.

    Modules and every child's train/eval flag are restored even on failure. No
    model, reset, gradient, or reference tensor is changed. The training caller
    wraps diagnostics in its existing preserve_rng context.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not reference:
        raise ValueError("Trajectory phase diagnostics require a training reference episode")
    networks = (encoder, predictor, readout, table)
    modes = [(module, module.training) for network in networks for module in network.modules()]
    fig = None
    try:
        for network in networks:
            network.eval()
        fixed = config["training"].get("initial_conditions", "learned") == "true_fixed"
        target_label = "Fixed true-reset simulation" if fixed else "Learned-reset simulation"
        labels = (target_label, "r(E): observed images", "r(P): autoregressive forecast")
        fig, axes = plt.subplots(1, len(reference), figsize=(4.8 * len(reference), 5.6),
                                 squeeze=False, sharey=True)
        for ax, row in zip(axes[0], reference):
            endpoints = row["endpoints"].cpu().numpy()
            mode, episode = row["reset_mode"], row["trajectory_id"]
            title = f"{RESET_NAMES[mode]} · train episode {episode}"
            if len(endpoints) < 3:
                ax.set_title(title)
                ax.text(.5, .5, "Insufficient observed history for the 0.34 s forecast", ha="center",
                        va="center", wrap=True, transform=ax.transAxes)
                continue
            last = int(endpoints[-1])
            z = encode_endpoints(encoder, row["frames"].numpy(), device,
                                 batch_size=config["training"].get("cache_batch_size", 8),
                                 endpoints=endpoints)[None]
            theta = row["theta"][None].to(device)
            forces = row["forces"][None, :last].to(device)
            steps = len(endpoints) - 3
            if steps:
                passive = getattr(predictor, "action_free", False)
                predicted = rollout_latents(predictor, z[:, :3],
                    None if passive else forces, None if passive else theta,
                    steps=steps, passive=passive)
                forecast = torch.cat((z[:, 2:3], predicted), dim=1)
            else:
                forecast = z[:, 2:3]
            observed = to_state(readout(z[:, 2:]))[0].cpu().numpy()
            forecast = to_state(readout(forecast))[0].cpu().numpy()
            initial = table(torch.tensor([episode], dtype=torch.long, device=device))
            simulated = rollout(initial, theta, forces)[0, FORECAST_START:].cpu().numpy()
            for states, color, label, markers in zip((simulated, observed, forecast), COLORS,
                                                     labels, (False, True, True)):
                _phase_curve(ax, states, color, label, markers)
            suffix = "; truncated episode" if row["available_frames"] <= int(ENDPOINTS[-1]) else ""
            ax.set_title(f"{title}\n0.34–{last * DT:.2f} s; {steps} forecast steps{suffix}", fontsize=10)
            ax.set_xlabel("Cart position p [m]")
            ax.set_ylim(-np.pi - .2, np.pi + .2)
            ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                          [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
            ax.axhline(0, color="#cccccc", linewidth=.7, zorder=0)
            ax.grid(alpha=.18)
        axes[0, 0].set_ylabel("Pole angle q [rad]")
        fig.suptitle(("Fixed true-reset diagnostic" if fixed else "Learned-reset physics target") +
                     f": {len(reference)} training examples", fontsize=14)
        handles = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(COLORS, labels)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .095),
                   ncol=3, frameon=False, fontsize=9)
        fig.text(.5, .025, "Open circle = start; square = end. q=0 is downward; ±π is upright. Lines break at the angular wrap.\n"
                 "P starts from observations ending at 0.34 s and receives no future images; model samples are 100 ms apart.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.065, right=.99, bottom=.255, top=.83, wspace=.15)
        return fig
    except BaseException:
        if fig is not None:
            plt.close(fig)
        raise
    finally:
        for module, training in modes:
            module.training = training
