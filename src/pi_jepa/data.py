"""Dense passive/controlled cart-pole corpora, private truth, and causal windows."""

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from pi_jepa.physics import (
    ACTION_DT, DT, ELL, G, INTEGRATION_DT, M_POLE, SUBSTEPS,
    angle_difference, lqr_gain, rk4,
)

ENDPOINTS = (14, 24, 34, 44, 54, 64)
HISTORY_STRIDE = 2
LATENT_STRIDE = 10
WINDOW_FRAMES = 65
SCHEMA_VERSION = 3
CART_LIMIT = 1.5
CAMERA_HALF_EXTENT = 2.6
BOB_RADIUS_PIXELS = 2.5
# A half-pixel allowance also covers rounding on the 2x rendering grid.
RASTER_MARGIN_PIXELS = 0.5
RESET_MODES = {"downward": 0, "nonlinear": 1, "upright": 2}
FIXED_APPEARANCE = {"background": [244, 246, 249], "cart": [35, 99, 172], "bob": [206, 73, 50]}
CAMERA = {
    "width": 96, "height": 96, "supersampling": 2,
    "x_limits_m": [-CAMERA_HALF_EXTENT, CAMERA_HALF_EXTENT],
    "y_limits_m": [-CAMERA_HALF_EXTENT, CAMERA_HALF_EXTENT],
    "pixels_per_metre": 95 / (2 * CAMERA_HALF_EXTENT), "isotropic": True,
    "note": "Square calibrated view; vertical margins preserve equal horizontal/vertical scales.",
}


def dataset_root(config, dataset=None):
    """The configured data path contains the two separately stored corpora."""
    selected = dataset or config.get("dataset", "controlled")
    if selected not in ("passive", "controlled"):
        raise ValueError("Select passive or controlled")
    return Path(config["paths"]["data"]) / selected


def causal_clips(frames):
    """B,65,H,W,3 uint8 -> B,6,24,H,W; each eight-image history is causal."""
    if frames.ndim != 5 or frames.shape[1] != WINDOW_FRAMES or frames.shape[-1] != 3:
        raise ValueError("Expected RGB windows shaped (B,65,H,W,3)")
    indices = torch.tensor([[e - 14 + 2 * k for k in range(8)] for e in ENDPOINTS], device=frames.device)
    clips = frames[:, indices]
    b, n, _, h, w, _ = clips.shape
    return clips.permute(0, 1, 2, 5, 3, 4).reshape(b, n, 24, h, w).float().div(127.5).sub(1)


def action_blocks(forces):
    """Each raw e:e+10 force block advances endpoint e to e+10."""
    if forces.shape[-1] < ENDPOINTS[-1]:
        raise ValueError("A window needs 64 recorded-interval forces")
    return torch.stack([forces[..., e:e + LATENT_STRIDE] for e in ENDPOINTS[:-1]], dim=-2)


class TrainDataset(Dataset):
    """Learning files only, bounded uint8 episode cache, no true-state access."""

    def __init__(self, root, split="train", cache_size=8):
        split = "validation" if split == "val" else split
        if split not in ("train", "validation"):
            raise ValueError("TrainDataset only permits train or validation")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        validate_manifest(manifest)
        self.dataset = manifest["dataset"]
        self.entries = manifest[split]
        self.reset_modes = torch.tensor([e["reset_mode"] for e in self.entries])
        self.cache_size = max(0, int(cache_size))
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        index = int(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        entry = self.entries[index]
        with np.load(self.root / entry["path"], allow_pickle=False) as data:
            result = {
                "frames": torch.from_numpy(data["rgb"]),
                "forces": torch.from_numpy(data["force"].reshape(-1)).float(),
                "trajectory_id": index, "reset_mode": entry["reset_mode"],
                "apparatus_id": entry["apparatus_id"],
            }
            if self.dataset == "controlled":
                result["theta"] = torch.from_numpy(data["theta"]).float()
        if self.cache_size:
            self._cache[index] = result
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result


def sample_appearance(rng):
    """Optional independent episode colors; default collection uses fixed colors."""
    palette = [(32, 92, 172), (166, 52, 45), (26, 128, 95), (131, 58, 166), (177, 102, 22)]
    return {"background": rng.integers(230, 256, size=3).tolist(),
            "cart": list(palette[int(rng.integers(len(palette)))]),
            "bob": list(palette[int(rng.integers(len(palette)))])}


def world_to_pixel(x, y):
    """Final-image coordinates: the same pixels/metre along both axes."""
    ppm = CAMERA["pixels_per_metre"]
    return ((np.asarray(x) + CAMERA_HALF_EXTENT) * ppm,
            (CAMERA_HALF_EXTENT - np.asarray(y)) * ppm)


def visibility_margins(states):
    """Minimum final-pixel edge clearance of the full cart, pole and bob.

    Include the cart/wheels, the two-pixel pole stroke, and the bob radius.
    Coordinates use image pixel centres [0,95]; retaining another half pixel
    covers raster rounding. The finite boundary-exit frame is checked too.
    """
    states = np.asarray(states, dtype=np.float64)
    p, q = states[..., 0], states[..., 2]
    px, py = world_to_pixel(p, np.zeros_like(p))
    bx, by = world_to_pixel(p + ELL * np.sin(q), -ELL * np.cos(q))
    ppm = CAMERA["pixels_per_metre"]
    left = np.minimum.reduce((px - .17 * ppm, px - 1., bx - BOB_RADIUS_PIXELS))
    right = np.maximum.reduce((px + .17 * ppm, px + 1., bx + BOB_RADIUS_PIXELS))
    top = np.minimum.reduce((py - .05 * ppm, py - 1., by - BOB_RADIUS_PIXELS))
    bottom = np.maximum.reduce((py + .15 * ppm, py + 1., by + BOB_RADIUS_PIXELS))
    return np.minimum.reduce((left, CAMERA["width"] - 1 - right,
                              top, CAMERA["height"] - 1 - bottom))


def require_visible(states, episode_id="episode"):
    """Fail on invalid framing; never clip, modify or resample a trajectory."""
    margins = visibility_margins(states)
    if not np.isfinite(margins).all() or np.min(margins) < RASTER_MARGIN_PIXELS:
        frame = int(np.argmin(margins))
        raise ValueError(f"{episode_id}: frame {frame} violates camera visibility "
                         f"(edge clearance {margins[frame]:.6g} pixels)")
    return float(np.min(margins))


def render(state, appearance=None):
    """Fixed isotropic 96x96 RGB camera, 2x supersampling and antialiasing."""
    appearance = FIXED_APPEARANCE if appearance is None else appearance
    p, _, q, _ = np.asarray(state)
    scale = CAMERA["supersampling"]
    image = Image.new("RGB", (CAMERA["width"] * scale, CAMERA["height"] * scale), tuple(appearance["background"]))
    draw = ImageDraw.Draw(image)

    def xy(x, y):
        px, py = world_to_pixel(x, y)
        return float(px * scale), float(py * scale)

    def box(x0, y0, x1, y1):
        return (*xy(x0, y1), *xy(x1, y0))

    draw.line([xy(-CAMERA_HALF_EXTENT, -0.12), xy(CAMERA_HALF_EXTENT, -0.12)], fill=(135, 140, 148), width=scale)
    for tick in np.arange(-CART_LIMIT, CART_LIMIT + .01, 0.5):
        draw.line([xy(tick, -0.13), xy(tick, -0.22)], fill=(135, 140, 148), width=scale)
    draw.line([xy(0, -0.13), xy(0, -0.36)], fill=(40, 45, 55), width=2 * scale)
    draw.rectangle(box(p - 0.17, -0.10, p + 0.17, 0.05), fill=tuple(appearance["cart"]))
    for wheel in (p - 0.11, p + 0.11):
        draw.ellipse(box(wheel - 0.045, -0.15, wheel + 0.045, -0.06), fill=(45, 49, 55))
    bob_x, bob_y = p + ELL * np.sin(q), -ELL * np.cos(q)
    draw.line([xy(p, 0), xy(bob_x, bob_y)], fill=(35, 38, 45), width=2 * scale)
    bx, by = xy(bob_x, bob_y)
    radius = BOB_RADIUS_PIXELS * scale
    draw.ellipse((bx - radius, by - radius, bx + radius, by + radius), fill=tuple(appearance["bob"]), outline=(30, 34, 40), width=scale)
    px, py = xy(p, 0)
    draw.ellipse((px - scale, py - scale, px + scale, py + scale), fill=(220, 220, 220))
    return np.asarray(image.resize((CAMERA["width"], CAMERA["height"]), Image.Resampling.LANCZOS))


def validate_camera(camera):
    required = ("width", "height", "supersampling", "x_limits_m", "y_limits_m", "isotropic")
    if any(camera.get(key) != CAMERA[key] for key in required):
        raise ValueError(f"Dense schema {SCHEMA_VERSION} requires the fixed isotropic camera: {CAMERA}")
    if "pixels_per_metre" in camera and not np.isclose(camera["pixels_per_metre"], CAMERA["pixels_per_metre"]):
        raise ValueError("Camera pixels_per_metre does not match its extent")


def validate_clocks(config):
    physics, data = config["physics"], config["data"]
    expected = {"m": M_POLE, "ell": ELL, "g": G, "dt": DT,
                "integration_dt": INTEGRATION_DT, "action_dt": ACTION_DT,
                "substeps": SUBSTEPS, "force_limit": 5.0}
    if any(not np.isclose(physics.get(k, np.nan), v) for k, v in expected.items()):
        raise ValueError(f"Dense corpus requires these physical constants/clocks: {expected}")
    if not np.isclose(data.get("cart_limit_m", np.nan), CART_LIMIT):
        raise ValueError(f"Dense schema {SCHEMA_VERSION} requires cart_limit_m={CART_LIMIT}")
    validate_camera(data.get("camera", {}))
    ratios = {"physics_steps_per_record": physics["dt"] / physics["integration_dt"],
              "record_intervals_per_force": physics["action_dt"] / physics["dt"],
              "history_frame_stride": data["history_dt"] / physics["dt"],
              "latent_stride_records": data["latent_dt"] / physics["dt"]}
    if any(not np.isclose(v, round(v)) for v in ratios.values()):
        raise ValueError("All clocks must have integer grid ratios")
    ratios = {k: round(v) for k, v in ratios.items()}
    if tuple(ratios.values()) != (5, 2, 2, 10):
        raise ValueError("Expected integration/force/history/latent ratios 5,2,2,10")
    return {"integration_dt_s": INTEGRATION_DT, "record_dt_s": DT,
            "action_dt_s": ACTION_DT, "history_dt_s": .02, "latent_dt_s": .1,
            "episode_seconds": data["episode_seconds"], "history_frames": 8,
            "window_frames": WINDOW_FRAMES, "endpoints": list(ENDPOINTS), **ratios}


def validate_manifest(manifest):
    """Reject stale datasets even when the tensor shapes still match."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Regenerate data: dense schema version {SCHEMA_VERSION} is required")
    validate_clocks({"physics": manifest.get("physics", {}),
                     "data": manifest.get("collection_settings", {})})
    validate_camera(manifest.get("camera", {}))
    if not np.isclose(manifest.get("cart_limit_m", np.nan), CART_LIMIT):
        raise ValueError("Dataset cart boundary does not match the current experiment")


def _seed(seed, *parts):
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0])


def sample_reset(rng, mode, passive=False, calibration=False):
    """Private resets at commanded p=0; unwrapped angles and no state kicks."""
    if calibration:
        return np.array([0., rng.uniform(-.15, .15), rng.uniform(-.6, .6), rng.uniform(-.5, .5)])
    while True:
        v = rng.uniform(-.25, .25)
        if mode == 0:
            q = rng.uniform(-1.2, 1.2) if passive else rng.uniform(-1., 1.)
            w = rng.uniform(-2., 2.) if passive else rng.uniform(-1.5, 1.5)
            if passive and abs(q) < .10 and abs(w) < .20 and abs(v) < .05:
                continue
        elif mode == 1:
            q = rng.choice([-1, 1]) * rng.uniform(1.2, 2.6)
            w = rng.uniform(-2., 2.)
        else:
            q = np.pi + (rng.choice([-1, 1]) * rng.uniform(.05, .20) if passive else rng.uniform(-.15, .15))
            w = rng.uniform(-.5, .5)
        return np.array([0., v, q, w], dtype=np.float64)


def pulse_program(rng, updates=200, paired=False):
    """Prechosen pulses in newtons on the 20 ms grid; no state feedback."""
    scale = rng.uniform(.5, 3.)
    program = []
    while len(program) < updates:
        duration = int(rng.integers(5, 16))
        if rng.random() < .2:
            program.extend([0.] * duration)
            continue
        amplitude = rng.choice([-1., 1.]) * rng.uniform(.3, 1.) * scale
        program.extend([amplitude] * duration)
        if paired:
            program.extend([-amplitude] * duration)
    return np.asarray(program[:updates], dtype=np.float64).clip(-5, 5)


def multisine_program(rng, updates=200, peak_range=(.8, 2.5)):
    t = np.arange(updates) * ACTION_DT
    frequencies = rng.uniform([.30, .65, 1.20], [.60, 1., 1.70])
    amplitudes = rng.uniform(.3, 1., 3)
    phases = rng.uniform(0., 2 * np.pi, 3)
    wave = (amplitudes[:, None] * np.sin(2 * np.pi * frequencies[:, None] * t + phases[:, None])).sum(0)
    wave -= wave.mean()
    return wave * (rng.uniform(*peak_range) / np.abs(wave).max())


@lru_cache(maxsize=1)
def nominal_lqr_gain():
    """Compute the one nominal collection gain once per generation process."""
    return lqr_gain([1., .25])


@torch.no_grad()
def simulate_batch(plans):
    """Shared float64 RK4; force selected before each two-record hold.

    Return only finite prefixes. The first finite boundary-exit state is retained;
    a nonfinite candidate state is discarded with its unmatched force interval.
    """
    batch = len(plans)
    intervals = len(plans[0]["program"]) * 2
    x = torch.as_tensor(np.stack([p["initial"] for p in plans]), dtype=torch.float64)
    theta = torch.as_tensor(np.stack([p["theta"] for p in plans]), dtype=torch.float64)
    programs = torch.as_tensor(np.stack([p["program"] for p in plans]), dtype=torch.float64)
    gain = torch.from_numpy(nominal_lqr_gain()).reshape(-1)
    feedback = torch.tensor([p.get("feedback", False) for p in plans])
    scales = torch.tensor([p.get("gain_scale", 1.) for p in plans], dtype=torch.float64)
    reference = torch.tensor([p.get("reference", [0., 0., 0.]) for p in plans], dtype=torch.float64)
    release = torch.tensor([p.get("release", [-1, -1]) for p in plans])
    states = np.empty((batch, intervals + 1, 4), dtype=np.float64)
    forces = np.zeros((batch, intervals), dtype=np.float64)
    states[:, 0] = x.numpy()
    active = torch.ones(batch, dtype=torch.bool)
    lengths = np.ones(batch, dtype=np.int64)
    reasons = np.full(batch, "duration", dtype="<U16")
    for k in range(intervals):
        if k % 2 == 0:
            update = k // 2
            u = programs[:, update].clone()
            if feedback.any():
                amplitude, frequency, phase = reference.unbind(-1)
                argument = 2 * torch.pi * frequency * (k * DT) + phase
                p_ref = amplitude * argument.sin()
                v_ref = amplitude * 2 * torch.pi * frequency * argument.cos()
                error = torch.stack((x[:, 0] - p_ref, x[:, 1] - v_ref,
                                     angle_difference(x[:, 2], torch.pi), x[:, 3]), -1)
                enabled = feedback & ~((update >= release[:, 0]) & (update < release[:, 1]))
                u = u - enabled * scales * (error * gain).sum(-1)
            u = u.clamp(-5, 5)
        candidate = rk4(x, theta, u, DT, SUBSTEPS)
        finite = torch.isfinite(candidate).all(-1)
        accepted = active & finite
        states[:, k + 1] = torch.where(accepted[:, None], candidate, x).numpy()
        forces[:, k] = torch.where(accepted, u, 0.).numpy()
        lengths += accepted.numpy()
        boundary = accepted & (candidate[:, 0].abs() > CART_LIMIT)
        reasons[(active & ~finite).numpy()] = "nonfinite"
        reasons[boundary.numpy()] = "boundary_exit"
        x = torch.where(accepted[:, None], candidate, x)
        active = active & finite & ~boundary
        if not active.any():
            break
    return [(states[i, :n].copy(), forces[i, :n - 1].copy(), str(reasons[i]))
            for i, n in enumerate(lengths)]


def _episode_plan(seed, split_code, episode, apparatus, theta, family, mode, program, waveform_id,
                  seconds=4., passive=False, calibration=False):
    stream = _seed(seed, split_code, episode, 12)
    reset_seed = _seed(stream, 1)
    control_seed = _seed(stream, 2)
    appearance_seed = _seed(stream, 3)
    rng = np.random.default_rng(control_seed)
    plan = {"episode": episode, "apparatus_id": apparatus, "theta": np.asarray(theta),
            "family": family, "mode": mode, "program": np.asarray(program),
            "waveform_id": waveform_id, "reset_seed": reset_seed, "control_seed": control_seed,
            "appearance_seed": appearance_seed, "attempt": 0, "passive": passive,
            "calibration": calibration, "seconds": seconds}
    plan["initial"] = sample_reset(np.random.default_rng(reset_seed), mode, passive, calibration)
    if family in ("noisy_lqr", "lqr_release"):
        plan["feedback"] = True
        plan["gain_scale"] = float(rng.uniform(.85, 1.15))
        # Half of each feedback family uses a varied reference.
        plan["reference"] = ([float(rng.uniform(.10, .35)), float(rng.uniform(.10, .25)),
                              float(rng.uniform(0., 2 * np.pi))] if episode % 2 else [0., 0., 0.])
        if family == "lqr_release":
            start = round(rng.uniform(1.2, 2.) / ACTION_DT)
            duration = round(rng.uniform(.20, .40) / ACTION_DT)
            plan["release"] = [start, start + duration]
    return plan


def _save_episode(task):
    root, split, plan, result, appearance_randomization, private, keep_program = task
    states, force, reason = result
    root = Path(root)
    stem = plan["episode_id"]
    minimum_margin = require_visible(states, stem)
    path = f"{split}/{stem}.npz"
    truth_path = f"truth/{split}/{stem}.npz"
    appearance = sample_appearance(np.random.default_rng(plan["appearance_seed"])) if appearance_randomization else FIXED_APPEARANCE
    payload = {"rgb": np.stack([render(state, appearance) for state in states]),
               "force": force[:, None], "t": np.arange(len(states), dtype=np.float64) * DT,
               "p0_commanded": np.float64(0.), "reset_mode": np.asarray(list(RESET_MODES)[plan["mode"]]),
               "episode_id": np.asarray(stem), "apparatus_id": np.asarray(plan["apparatus_id"]),
               "waveform_id": np.asarray(plan["waveform_id"]), "collection_family": np.asarray(plan["family"]),
               "valid_length": np.int64(len(states)), "termination_reason": np.asarray(reason)}
    if not private:
        payload["theta"] = plan["theta"].astype(np.float64)
    if keep_program:
        payload["force_program"] = np.repeat(plan["program"], 2)[:, None]
    np.savez_compressed(root / path, **payload)
    np.savez_compressed(root / truth_path, state=states, theta_true=plan["theta"], exact_initial_state=states[0])
    entry = {"path": path, "truth": truth_path, "episode_id": stem,
             "apparatus_id": plan["apparatus_id"], "reset_mode": plan["mode"],
             "collection_family": plan["family"], "waveform_id": plan["waveform_id"],
             "valid_length": len(states), "termination_reason": reason,
             "duration_s": (len(states) - 1) * DT,
             "minimum_visibility_margin_pixels": minimum_margin}
    if not private:
        entry["seeds"] = {k: plan[k] for k in ("reset_seed", "control_seed", "appearance_seed")}
    if plan.get("feedback"):
        entry["controller"] = {"gain_scale": plan["gain_scale"], "reference": plan["reference"],
                               "release_updates": plan.get("release")}
    return entry


def _collect(root, split, plans, manifest, config, executor, private=False, minimum=True, keep_program=False):
    (root / split).mkdir(parents=True, exist_ok=True)
    (root / "truth" / split).mkdir(parents=True, exist_ok=True)
    results = simulate_batch(plans)
    attempts = len(plans)
    for index, (plan, result) in enumerate(zip(plans, results, strict=True)):
        while minimum and len(result[0]) < WINDOW_FRAMES:
            manifest["discarded_attempts"].append({"split": split, "episode_id": plan["episode_id"],
                "collection_family": plan["family"], "valid_length": len(result[0]),
                "termination_reason": result[2], "attempt": plan["attempt"]})
            plan["attempt"] += 1
            if plan["attempt"] > 1000:
                raise RuntimeError("1000 short attempts within the same planned family")
            rng = np.random.default_rng(_seed(plan["reset_seed"], plan["attempt"]))
            plan["initial"] = sample_reset(rng, plan["mode"], plan["passive"], plan["calibration"])
            result = simulate_batch([plan])[0]
            results[index] = result
            attempts += 1
    tasks = [(str(root), split, plan, result, config["data"]["appearance_randomization"], private, keep_program)
             for plan, result in zip(plans, results, strict=True)]
    entries = []
    iterator = map(_save_episode, tasks) if executor is None else executor.map(_save_episode, tasks, chunksize=2)
    for n, entry in enumerate(iterator, 1):
        entries.append(entry)
        if n % 128 == 0 or n == len(plans):
            print(f"{manifest['dataset']}/{split}: saved {n}/{len(plans)} episodes", flush=True)
    manifest[split] = entries
    manifest["counts"][split] = {"episodes": len(entries), "transitions": sum(e["valid_length"] - 1 for e in entries),
        "attempts": attempts, "discarded_early": attempts - len(entries),
        "boundary_exits": sum(e["termination_reason"] == "boundary_exit" for e in entries),
        "nonfinite_exits": sum(e["termination_reason"] == "nonfinite" for e in entries)}


def _passive_plans(config, split, code):
    total = config["data"][f"passive_{split}_episodes"]
    if total % 6:
        raise ValueError("Passive split count must be divisible by 6 for the exact 3:2:1 mixture")
    modes = [0] * (total // 2) + [1] * (total // 3) + [2] * (total // 6)
    np.random.default_rng(_seed(config["seed"], code, 80)).shuffle(modes)
    updates = round(config["data"]["episode_seconds"] / ACTION_DT)
    plans = []
    for i, mode in enumerate(modes):
        family = ("downward_oscillations", "nonlinear_excursions", "upright_falls")[mode]
        plan = _episode_plan(config["seed"], code, i, "nominal", [1., .25], family, mode,
                             np.zeros(updates), "zero", passive=True)
        plan["episode_id"] = f"trajectory_{i:06d}"
        plans.append(plan)
    return plans


def _controlled_training_plans(config, split, code, root):
    data, seed = config["data"], config["seed"]
    prefix = "train" if split == "train" else "val"
    per_app = data[f"{prefix}_trajectories"]
    if per_app != (24 if split == "train" else 12):
        raise ValueError("Controlled collection requires 24 train / 12 validation episodes per apparatus")
    updates = round(data["episode_seconds"] / ACTION_DT)
    library = []
    library_metadata = []
    size = data["waveform_library_size"]
    if size != 32:
        raise ValueError("Use 32 programs of each open-loop family")
    for family in ("pulse", "multisine"):
        programs = []
        for i in range(size):
            wave_seed = _seed(seed, code, 100 if family == "pulse" else 101, i)
            rng = np.random.default_rng(wave_seed)
            paired = i % 2 == 0
            program = pulse_program(rng, updates, paired) if family == "pulse" else multisine_program(rng, updates)
            programs.append(program)
            library_metadata.append({"id": f"{split}_{family}_{i:02d}", "family": family,
                                     "paired": paired if family == "pulse" else None, "seed": wave_seed})
        library.append(np.stack(programs))
    programs_path = root / "programs"
    programs_path.mkdir(exist_ok=True)
    np.savez_compressed(programs_path / f"{split}.npz", pulse=library[0], multisine=library[1], t=np.arange(updates) * ACTION_DT)
    (programs_path / f"{split}.json").write_text(json.dumps(library_metadata, indent=2) + "\n")
    parameter_rng = np.random.default_rng(_seed(seed, code, 90))
    plans = []
    half = split == "validation"
    for app in range(data[f"{prefix}_apparatuses"]):
        theta = parameter_rng.uniform([.7, .05], [1.3, .5])
        # Both force families contain downward/nonlinear/upright resets.
        pulse_modes = [0, 0, 1, 2] if half else [0, 0, 0, 0, 1, 1, 1, 2]
        smooth_modes = [0, 1] if half else [0, 0, 1, 2]
        # Validation has six open-loop resets: 3 downward,2 nonlinear,1 upright.
        # Move its sole upright reset between waveform types across apparatuses.
        if half and app % 2:
            pulse_modes[-1], smooth_modes[-1] = smooth_modes[-1], pulse_modes[-1]
        reset_allocation_rng = np.random.default_rng(_seed(seed, code, app, 81))
        reset_allocation_rng.shuffle(pulse_modes)
        reset_allocation_rng.shuffle(smooth_modes)
        assignments = [("pulse", m) for m in pulse_modes] + [("multisine", m) for m in smooth_modes]
        assignments += [("noisy_lqr", 2)] * (4 if half else 8) + [("lqr_release", 2)] * (2 if half else 4)
        for j, (family, mode) in enumerate(assignments):
            episode = app * per_app + j
            if family in ("pulse", "multisine"):
                fi = int(family == "multisine")
                # Balanced and independent pulses alternate exactly within each apparatus.
                local = j if fi == 0 else j - len(pulse_modes)
                width = len(pulse_modes) if fi == 0 else len(smooth_modes)
                wi = (app * width + local) % size
                program = library[fi][wi]
                waveform = f"{split}_{family}_{wi:02d}"
            else:
                excitation_rng = np.random.default_rng(_seed(seed, code, episode, 102))
                if episode % 2:
                    program = pulse_program(excitation_rng, updates, paired=True)
                    program *= excitation_rng.uniform(.6, 1.5) / max(np.abs(program).max(), 1e-12)
                else:
                    program = multisine_program(excitation_rng, updates, peak_range=(.6, 1.5))
                waveform = f"{split}_excitation_{episode:06d}"
            plan = _episode_plan(seed, code, episode, f"{split}_a{app:03d}", theta,
                                 family, mode, program, waveform)
            plan["episode_id"] = f"trajectory_{episode:06d}"
            plans.append(plan)
    return plans


def _controlled_test(config, root, manifest, executor):
    seed, data = config["seed"], config["data"]
    calibration, queries, apps, private_apps = [], [], [], []
    order = np.random.default_rng(_seed(seed, 30, 90)).permutation(len(data["test_pairs"]))
    for ordinal, pair_index in enumerate(order):
        theta = data["test_pairs"][pair_index]
        app_id = secrets.token_hex(8)
        truth_path = f"truth/apparatus_{app_id}.npz"
        app = {"apparatus_id": app_id, "group": data["test_groups"][pair_index], "truth": truth_path,
               "calibration": [], "queries": [], "calibration_truth": [], "query_truth": []}
        np.savez_compressed(root / truth_path, theta_true=np.asarray(theta, dtype=np.float64))
        private_app = {"apparatus_id": app_id, "theta_true": theta, "source_pair_index": int(pair_index), "episodes": []}
        for split, count, code in (("calibration", 2, 31), ("query", 8, 32)):
            modes = [0, 0] if split == "calibration" else [0, 0, 1, 2, 0, 0, 1, 2]
            for j in range(count):
                episode = ordinal * count + j
                wave_seed = _seed(seed, code, episode, 100)
                rng = np.random.default_rng(wave_seed)
                family = "multisine" if split == "calibration" or j >= 4 else "pulse"
                updates = round((3. if split == "calibration" else 4.) / ACTION_DT)
                program = (multisine_program(rng, updates, peak_range=(1., 2.) if split == "calibration" else (.8, 2.5))
                           if family == "multisine" else pulse_program(rng, updates, paired=j % 2 == 0))
                episode_id = secrets.token_hex(12)
                plan = _episode_plan(seed, code, episode, app_id, theta, family, modes[j], program,
                                     f"program_{secrets.token_hex(8)}", seconds=updates * ACTION_DT,
                                     calibration=split == "calibration")
                plan["episode_id"] = episode_id
                (calibration if split == "calibration" else queries).append(plan)
                key = "calibration" if split == "calibration" else "queries"
                truth_key = "calibration_truth" if split == "calibration" else "query_truth"
                app[key].append(f"{split}/{episode_id}.npz")
                app[truth_key].append(f"truth/{split}/{episode_id}.npz")
                private_app["episodes"].append({"episode_id": episode_id, "split": split, "waveform_seed": wave_seed,
                    **{k: plan[k] for k in ("reset_seed", "control_seed", "appearance_seed")}})
        apps.append(app)
        private_apps.append(private_app)
    # Test calibration/query attempts remain visible, including all early exits.
    _collect(root, "calibration", calibration, manifest, config, executor, private=True, minimum=False, keep_program=True)
    _collect(root, "query", queries, manifest, config, executor, private=True, minimum=False, keep_program=True)
    manifest["test"] = sorted(apps, key=lambda app: app["apparatus_id"])
    manifest["calibration"].sort(key=lambda episode: episode["episode_id"])
    manifest["query"].sort(key=lambda episode: episode["episode_id"])
    (root / "truth" / "private_test_manifest.json").write_text(json.dumps(private_apps, indent=2) + "\n")


def generate(config, dataset=None):
    """Write fresh dense corpora; never silently reuse an older format."""
    selected = dataset or config.get("dataset", "controlled")
    if selected == "both":
        return {name: generate(config, name) for name in ("passive", "controlled")}
    clocks = validate_clocks(config)
    root = dataset_root(config, selected)
    public_data = {key: value for key, value in config["data"].items() if key not in ("test_pairs", "test_groups")}
    signature = {"seed": config["seed"], "physics": config["physics"], "data": public_data,
                 "private_test_configuration_sha256": hashlib.sha256(json.dumps(config["data"]["test_pairs"]).encode()).hexdigest()}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        validate_manifest(saved)
        if saved.get("schema_version") != SCHEMA_VERSION or saved.get("generation_config") != signature:
            raise ValueError(f"Dataset {root} differs; use a fresh data root")
        splits = ("train", "validation", "test") if selected == "passive" else ("train", "validation", "calibration", "query")
        if not all((root / entry[key]).is_file() for split in splits for entry in saved[split] for key in ("path", "truth")):
            raise FileNotFoundError("Existing manifest references missing episode files")
        print(f"Reusing complete dense corpus at {root}", flush=True)
        return saved
    root.mkdir(parents=True, exist_ok=False)
    (root / "truth").mkdir()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    lock = Path("uv.lock")
    manifest = {"schema_version": SCHEMA_VERSION, "dataset": selected, "generation_config": signature,
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "code_commit": commit, "code_dirty": dirty,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("data.py", "physics.py")},
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "seed": config["seed"], "seed_streams": "Independent SeedSequence streams per split/parameter/reset/program/appearance; held-out mapping/seeds private.",
        "physics": config["physics"], "clocks": clocks, "camera": CAMERA,
        "cart_limit_m": CART_LIMIT,
        "termination_policy": "Stop at first recorded |p| > cart_limit_m; retain that finite state and its force interval.",
        "visibility_policy": {"bob_radius_pixels": BOB_RADIUS_PIXELS,
            "minimum_required_edge_clearance_pixels": RASTER_MARGIN_PIXELS,
            "checks": "Full cart/pole/bob geometry and pole stroke, all frames including terminal overshoot; fail instead of cropping or resampling."},
        "units": {"state": ["m", "m/s", "rad", "rad/s"], "force": "N", "theta": ["kg", "N*s/m"]},
        "angle_convention": "q=0 down, q=pi upright; unwrapped during integration",
        "action_convention": "force[k] acts on [t[k],t[k+1]); paired 20 ms hold, odd final entry allowed",
        "collection_settings": public_data, "appearance": "random independent episode colors" if public_data["appearance_randomization"] else FIXED_APPEARANCE,
        "parameter_split": "fixed nominal [1.0,0.25] all splits" if selected == "passive" else "independent train/validation apparatuses; held-out pairs private",
        "nominal_lqr_gain": nominal_lqr_gain().tolist(), "counts": {}, "discarded_attempts": []}
    workers = int(config["data"].get("generation_workers", 4))
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for split, code in (("train", 10), ("validation", 20)):
            plans = (_passive_plans(config, split, code) if selected == "passive"
                     else _controlled_training_plans(config, split, code, root))
            _collect(root, split, plans, manifest, config, executor)
        if selected == "passive":
            plans = _passive_plans(config, "test", 30)
            _collect(root, "test", plans, manifest, config, executor, minimum=False)
        else:
            _controlled_test(config, root, manifest, executor)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset": selected, "root": str(root), "counts": manifest["counts"]}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--dataset", choices=("passive", "controlled", "both"), default="both")
    parser.add_argument("--workers", type=int, help="Bounded rendering/compression processes")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    if args.data_root:
        config["paths"]["data"] = args.data_root
    if args.workers is not None:
        config["data"]["generation_workers"] = args.workers
    generate(config, args.dataset)
