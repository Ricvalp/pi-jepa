"""Fixed learning-only diagnostic batches and visualizations for the training loop."""
import numpy as np
import torch

from pi_jepa.data import causal_clips, ENDPOINTS
from pi_jepa.models import encode_temporal, scale_readout
from pi_jepa.physics import iota
from pi_jepa.losses import simulate_window
from pi_jepa.training_diagnostics import (prediction_diagnostics, predicted_physics_metrics,
                                         readout_metrics, representation_metrics)


def fixed_reference(data, seed=143, max_episodes=48):
    """Fixed independent episodes and uniform crops, with a separate RNG stream."""
    from pi_jepa.train import batch_from
    rng = torch.Generator().manual_seed(seed)
    ids = torch.randperm(len(data), generator=rng)[:max_episodes]
    return batch_from(data, ids, "cpu", generator=rng)


def parameter_probes(encoder, predictor, readout, table, mode):
    """Small explicitly named tensors, not a claim of full-network update norms."""
    selected = {}
    if mode != "readout":
        selected["encoder_stem"] = encoder.backbone.conv1.weight
        if getattr(predictor, "action_free", False):
            selected["predictor_input"] = predictor.net[0].weight
            selected["predictor_output"] = predictor.net[-1].weight
        else:
            selected["predictor_conditioning"] = predictor.conditioning[0].weight
            selected["predictor_output"] = predictor.output_projection.weight
    if mode != "jepa":
        selected["readout_input"] = readout.net[0].weight
        if hasattr(table, "raw") and table.raw.requires_grad:
            selected["initial_conditions"] = table.raw
    return selected


def update_metrics(probes, before):
    result = {}
    for name, parameter in probes.items():
        change = (parameter.detach() - before[name]).norm().item()
        norm = before[name].norm().item()
        result[f"updates/{name}/norm"] = change
        result[f"updates/{name}/relative_norm"] = change / norm if norm > 0 else float("nan")
    return result


def figures_for(z, plot_data, frames, decoded, simulated, name, raw_endpoints,
                *, initial_conditions="learned"):
    """Fixed examples; future targets are used only to display forecast accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = {}
    rows = min(2, len(frames))
    fig, axes = plt.subplots(rows, 6, figsize=(10, 2 * rows), squeeze=False)
    for row in range(rows):
        for col, endpoint in enumerate(ENDPOINTS):
            axes[row, col].imshow(frames[row, endpoint].numpy())
            axes[row, col].axis("off")
            axes[row, col].set_title(f"t={int(raw_endpoints[row, col]) * .01:.2f}s", fontsize=8)
    fig.suptitle(f"{name}: fixed episode/window examples")
    fig.tight_layout(); figures[f"{name}/examples"] = fig
    keys = [key for key in ("actual", "correct", "shuffled", "persistence") if key in plot_data]
    fig, axes = plt.subplots(rows, len(keys), figsize=(3 * len(keys), 3 * rows), squeeze=False)
    initial = z[:2, 2].cpu().numpy()
    arrays = {key: np.asarray(value)[:2] - initial[:, None]
              for key, value in plot_data.items() if key in ("actual", "correct", "shuffled", "persistence")}
    limit = max(1e-6, max(float(np.abs(value).max()) for value in arrays.values()))
    for row in range(rows):
        for col, key in enumerate(keys):
            axes[row, col].imshow(arrays[key][row].T, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, col].set(title=key, xlabel="Future latent step", ylabel="Latent dimension")
    fig.suptitle(f"{name}: change from last observed latent (shared color scale ±{limit:.3g})")
    fig.tight_layout(); figures[f"{name}/latent_forecasts"] = fig
    if simulated is not None:
        fixed_reset = initial_conditions == "true_fixed"
        target_label = "Fixed true-reset simulation" if fixed_reset else "Learned-reset simulation"
        fig, axes = plt.subplots(1, 5, figsize=(13, 3))
        observed, target = scale_readout(decoded[:1]).cpu().numpy()[0], iota(simulated[:1]).cpu().numpy()[0]
        for col, label in enumerate(("p/2", "v/2", "sin(q)", "cos(q)", "w/5")):
            axes[col].plot(raw_endpoints[0].cpu().numpy() * .01, observed[:, col], label="Readout")
            axes[col].plot(raw_endpoints[0].cpu().numpy() * .01, target[:, col], linestyle="--", label=target_label)
            axes[col].set(title=label, xlabel="Time (s)")
        axes[0].legend(fontsize=7)
        fig.suptitle("True-reset diagnostic: training physical fit" if fixed_reset else
                     "Training physical self-consistency; simulated targets are not ground truth")
        fig.tight_layout(); figures[f"{name}/physics_fit"] = fig

    return figures


@torch.no_grad()
def evaluate_diagnostics(encoder, predictor, readout, table, reference, validation, config, device, mode, media=False):
    modules = [module for module in (encoder, predictor, readout) if module is not None]
    modes = [module.training for module in modules]
    for module in modules: module.eval()
    metrics, figures, histograms = {}, {}, {}
    try:
        for name, data in (("train_eval", reference), ("val", validation)):
            batch = {key: value.to(device) for key, value in data.items() if key != "frames"}
            codes = []
            for frames in data["frames"].split(config["training"]["cache_batch_size"]):
                clips = causal_clips(frames.to(device))
                codes.append(encode_temporal(encoder, clips))
            z = torch.cat(codes)
            diagnostic, plot_data = prediction_diagnostics(predictor, z, batch["forces"], batch.get("theta"),
                                                           batch["reset_mode"], batch["apparatus_id"])
            diagnostic.update(representation_metrics(z))
            decoded, simulated = readout(z) if readout is not None else None, None
            if mode != "jepa":
                if name == "train_eval":
                    cache = getattr(table, "target_cache", None)
                    simulated = (cache.gather(batch["trajectory_id"], batch["raw_endpoints"], device=device)
                                 if cache is not None else
                                 simulate_window(table(batch["trajectory_id"].long()), batch["theta"],
                                                 batch["prefix_forces"], batch["raw_endpoints"]))
                    # Reuse the exact correct-action autoregressive forecasts
                    # already scored in latent space; no extra predictor pass.
                    predicted = torch.as_tensor(plot_data["correct"], device=device, dtype=z.dtype)
                    diagnostic.update(predicted_physics_metrics(readout, predicted, simulated[:, 3:]))
                diagnostic.update(readout_metrics(decoded, simulated))
            metrics.update({f"{name}/{key}": value for key, value in diagnostic.items()})
            if media:
                figures.update(figures_for(z, plot_data, data["frames"], decoded, simulated, name, data["raw_endpoints"],
                    initial_conditions=config["training"].get("initial_conditions", "learned")))
                histograms[f"{name}/latent_values"] = z.cpu().numpy()
                histograms[f"{name}/latent_dimension_std"] = z.std(dim=0, unbiased=False).cpu().numpy()
    finally:
        for module, training in zip(modules, modes): module.train(training)
    return metrics, figures, histograms
