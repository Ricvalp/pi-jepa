"""Small local/Slurm smoke test; no dataset, downloads, or tracking service."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image
import torch

from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION, render
from pi_jepa.models import Encoder, PhysicalReadout, Predictor
from pi_jepa.physics import ELL, G, M_POLE, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("workspace/checks"))
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access an accelerator")

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_root / f"environment-{args.device}-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)

    theta = torch.tensor([[1.0, 0.2], [1.1, 0.3]], device=device, requires_grad=True)
    initial = torch.tensor([[0.0, 0.0, 0.2, 0.0], [0.1, 0.0, -0.3, 0.0]], device=device)
    forces = torch.linspace(-0.5, 0.5, 12, device=device).repeat_interleave(2).expand(2, -1)
    states = rollout(initial, theta, forces)
    if states.shape != (2, 25, 4) or not torch.isfinite(states).all():
        raise RuntimeError("Differentiable simulator produced invalid states")
    physical_gradient = torch.autograd.grad(states[:, -1].square().mean(), theta, retain_graph=True)[0]
    if not torch.isfinite(physical_gradient).all() or not torch.count_nonzero(physical_gradient):
        raise RuntimeError("Differentiable simulator parameter gradients are invalid")

    appearance = {"background": [245, 245, 245], "cart": [32, 92, 172], "bob": [166, 52, 45]}
    frames = np.stack([
        np.stack([render(state, appearance) for state in trajectory[:15:2]])
        for trajectory in states.detach().cpu().numpy()
    ])
    images = [Image.fromarray(frame) for frame in frames[0]]
    images[0].save(destination / "frame.png")
    images[0].save(destination / "rollout.gif", save_all=True, append_images=images[1:],
                   duration=20, loop=0)
    clips = torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3)
    clips = clips.reshape(2, 24, 96, 96).float().div(127.5).sub(1.0)

    encoder, predictor, readout = Encoder().to(device), Predictor().to(device), PhysicalReadout().to(device)
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(predictor.parameters()) + list(readout.parameters()), lr=1e-3
    )
    z = encoder(clips)
    predicted = predictor(z[:, None], forces[:, None, 14:24], theta)
    decoded = readout(z)
    loss = predicted.square().mean() + decoded.square().mean() + states[:, -1].square().mean()
    loss.backward()
    for name, model in (("encoder", encoder), ("predictor", predictor), ("readout", readout)):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"{name} gradients are missing or non-finite")
        if not any(torch.count_nonzero(g) for g in gradients):
            raise RuntimeError(f"{name} has no nonzero gradient")
    if theta.grad is None or not torch.isfinite(theta.grad).all():
        raise RuntimeError("Differentiable simulator parameter gradients are invalid")
    optimizer.step()
    encoder.eval()
    predictor.eval()
    readout.eval()
    with torch.no_grad():
        inferred = readout(predictor(encoder(clips)[:, None], forces[:, None, 14:24], theta.detach()))
    if inferred.shape != (2, 1, 5) or not torch.isfinite(inferred).all():
        raise RuntimeError("Inference output is invalid")
    if device.type == "cuda":
        torch.cuda.synchronize()

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "accelerator": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "seed": 0,
        "dataset_schema": SCHEMA_VERSION,
        "physics": {"ell_m": ELL, "pole_mass_kg": M_POLE, "gravity_m_s2": G,
                    "cart_limit_m": CART_LIMIT},
        "camera": CAMERA,
        "training_loss": float(loss.detach().cpu()),
        "latent_shape": list(z.shape),
        "inference_shape": list(inferred.shape),
        "physics_parameter_gradient_norm": float(physical_gradient.norm().detach().cpu()),
        "headless_rendering": "PNG and animated GIF saved",
        "output": str(destination.resolve()),
    }
    (destination / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
