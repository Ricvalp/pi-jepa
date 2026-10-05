"""Animate exact checkpoint readouts and simulator truth on shared metric axes.

Input: the directory produced by scripts/export_readout_trajectories.py.
Only the exported sample points are shown; samples are held without interpolation.
"""
import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from render import digest, font

SIZE = (1536, 960)
INK = (24, 38, 58)
MUTED = (96, 113, 131)
TRUTH = (0, 112, 178)
PREDICTED = (212, 88, 24)
RECONSTRUCTION = (0, 141, 103)


def read_geometry(metadata):
    """The export must describe its apparatus; never guess a legacy rod length."""
    try:
        dataset = metadata["dataset"]
        ell = float(metadata["physics"]["ell"])
        camera = metadata["camera"]
        schema = metadata["data_schema_version"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Readout export requires explicit dataset/schema/physics/camera metadata; re-export trajectories") from exc
    if dataset not in ("passive", "controlled") or schema != 3 or not math.isfinite(ell) or ell <= 0:
        raise ValueError("Invalid readout dataset schema or physical rod length")
    if not isinstance(camera, dict) or not camera.get("isotropic") or not all(key in camera for key in ("width", "height", "x_limits_m", "y_limits_m")):
        raise ValueError("Readout export requires the calibrated isotropic dataset camera")
    return ell, dataset


def read_training_protocol(metadata):
    """Legacy exports used learned resets; new diagnostics state the protocol."""
    training = metadata.get("train", {})
    mode = training.get("initial_conditions_mode", "learned")
    if mode not in ("learned", "true_fixed"):
        raise ValueError("Unknown training initial-condition protocol in readout metadata")
    true_reset = training.get("supervision_uses_true_reset", mode == "true_fixed")
    if not isinstance(true_reset, bool) or true_reset != (mode == "true_fixed"):
        raise ValueError("Readout initial-condition mode and true-reset supervision flag disagree")
    return mode


def world_bounds(series, ell):
    """One horizontal camera for all methods and all times of a trajectory."""
    positions = []
    for states in series:
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[-1] != 4 or not np.isfinite(states).all():
            raise ValueError("Expected finite physical states [T,4]")
        positions.extend([states[:, 0], states[:, 0] + ell * np.sin(states[:, 2])])
    x = np.concatenate(positions)
    center = (float(x.min()) + float(x.max())) / 2
    span = max(2 * ell + 1.4, float(x.max()) - float(x.min()) + 1.1)
    return center - span / 2, center + span / 2


def project_point(x, y, box, bounds, ell):
    left, top, right, bottom = box
    xmin, xmax = bounds
    # Equal px/m on both axes, with room for the exported rod at any angle.
    scale = min((right - left - 36) / (xmax - xmin), (bottom - top - 36) / (2 * ell + .6))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    return center_x + (x - (xmin + xmax) / 2) * scale, center_y - y * scale


def rod_points(state, box, bounds, ell):
    p, _, q, _ = state
    return (project_point(float(p), 0, box, bounds, ell),
            project_point(float(p + ell * np.sin(q)), float(-ell * np.cos(q)), box, bounds, ell))


def draw_cartpole(draw, state, box, bounds, color, ell, ghost=False):
    p = float(state[0])
    pivot, bob = rod_points(state, box, bounds, ell)
    first = project_point(p - .17, .035, box, bounds, ell)
    second = project_point(p + .17, -.075, box, bounds, ell)
    width = 2 if ghost else 5
    draw.rectangle((*first, *second), outline=color, fill=None if ghost else color, width=2)
    draw.line((*pivot, *bob), fill=color, width=width)
    radius = 7 if ghost else 8
    draw.ellipse((bob[0] - radius, bob[1] - radius, bob[0] + radius, bob[1] + radius),
                 outline=color, fill=None if ghost else color, width=2)
    draw.ellipse((pivot[0] - 3, pivot[1] - 3, pivot[0] + 3, pivot[1] + 3), fill=INK)
    if not ghost:
        for dx in (-.11, .11):
            x, y = project_point(p + dx, -.1, box, bounds, ell)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK)


def state_coordinates(state):
    p, v, q, w = state
    return np.array([p, v, np.sin(q), np.cos(q), w])


def draw_scene(draw, state, truth, history, box, bounds, color, show_truth, ell):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill=(246, 249, 252))
    for x in np.arange(math.ceil(bounds[0] * 2) / 2, bounds[1] + .001, .5):
        a = project_point(x, -.095, box, bounds, ell)
        b = project_point(x, -.135, box, bounds, ell)
        draw.line((*a, *b), fill=(161, 174, 186), width=1)
        draw.text((b[0], b[1] + 5), f"{x:g}", anchor="mt", font=font(11), fill=MUTED)
    yrail = project_point(0, -.095, box, bounds, ell)[1]
    draw.line((left + 10, yrail, right - 10, yrail), fill=(173, 185, 197), width=2)
    if len(history) > 1:
        points = [rod_points(s, box, bounds, ell)[1] for s in history]
        draw.line(points, fill=tuple(int(.55*c + .45*255) for c in color), width=2)
    if show_truth:
        draw_cartpole(draw, truth, box, bounds, (139, 184, 212), ell, ghost=True)
    draw_cartpole(draw, state, box, bounds, color, ell)
    draw.text((left + 12, bottom - 31), "x in metres  |  equal scale on both axes", font=font(13), fill=MUTED)


def render_sample(series, raw_series, times, index, record, kind, speed, ell, dataset,
                  initial_conditions_mode="learned"):
    true_reset = initial_conditions_mode == "true_fixed"
    if kind == "test":
        labels = ("Ground truth", "r(z predicted by P)", "r(E(actual video))")
        subtitles = ("Held-out simulator trajectory", "Autoregressive latent rollout", "Observed-video diagnostic")
        colors = (TRUTH, PREDICTED, RECONSTRUCTION)
        title = "HELD-OUT TRAJECTORY / " + str(record["reset_mode"]).upper() + " RESET"
        clock = f"forecast horizon {times[index]:.2f} s"
    else:
        labels = ("Ground truth", "r(E(training video))", "Simulator supervision")
        subtitles = ("Recorded simulator trajectory" if true_reset else "Hidden states: visualization only",
                     "Final encoder in eval mode",
                     "From fixed true training reset" if true_reset else "From learned training reset state")
        colors = (TRUTH, RECONSTRUCTION, PREDICTED)
        title = "TRAINING READOUT / " + str(record["reset_mode"]).upper() + " RESET"
        clock = f"trajectory time {times[index]:.2f} s"
    if true_reset:
        title = "TRUE-RESET DIAGNOSTIC / " + kind.upper() + " / " + str(record["reset_mode"]).upper() + " RESET"
    image = Image.new("RGB", SIZE, (236, 242, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SIZE[0], 108), fill=INK)
    draw.text((26, 19), title, font=font(27, True), fill="white")
    description = record.get("path", "") + (f" / {record['group']}" if record.get("group") else "")
    draw.text((28, 65), description, font=font(18), fill=(193, 209, 224))
    draw.text((1080, 24), clock, font=font(22, True), fill="white")
    draw.text((1082, 66), f"sample {index+1}/{len(times)}   |   {speed:g}x playback", font=font(17), fill=(193, 209, 224))
    bounds = world_bounds(series, ell)
    for panel, (states, raw, label, subtitle, color) in enumerate(zip(series, raw_series, labels, subtitles, colors)):
        x = 20 + panel * 506
        draw.rounded_rectangle((x, 126, x + 488, 808), radius=14, fill="white")
        draw.text((x + 20, 144), label, font=font(25, True), fill=color)
        draw.text((x + 20, 186), subtitle, font=font(16), fill=MUTED)
        draw_scene(draw, states[index], series[0][index], states[:index+1],
                   (x + 16, 220, x + 472, 598), bounds, color, show_truth=panel != 0, ell=ell)
        draw.text((x + 20, 613), "Physical readout coordinates", font=font(17, True), fill=INK)
        for row, (name, units, value) in enumerate(zip(("p", "v", "sin(q)", "cos(q)", "w"),
                                                      ("m", "m/s", "", "", "rad/s"), raw[index])):
            yy = 645 + row * 28
            draw.text((x + 22, yy), name, font=font(18), fill=MUTED)
            draw.text((x + 246, yy), f"{value:+.3f} {units}", font=font(18, True), fill=color)
    draw.text((29, 826), f"Blue outline in the other panels = ground truth at the same time. Rod length = {ell:g} m.", font=font(19), fill=INK)
    if kind == "test":
        detail = ("Warm observations end at t=0.34 s. Autonomous P receives no actions, parameters, or future observations."
                  if dataset == "passive" else
                  "Warm observations end at t=0.34 s. Future observations do not enter P; known test mass/drag condition its rollout.")
    else:
        detail = ("Diagnostic: simulator targets use the fixed true training reset. True future states are not loaded for the training loss."
                  if true_reset else
                  "Training loss compares scaled r(E(video)) with the learned-reset simulator; the simulator target is not ground truth.")
    draw.text((29, 861), detail, font=font(18), fill=MUTED)
    draw.text((29, 896), "One state sample every 0.10 s; samples are held without interpolation. All panels use the same metric camera.", font=font(18), fill=MUTED)
    return image


def write_video(path, images, sample_interval, speed):
    import imageio_ffmpeg
    # 25fps container, repeated identical frames: no invented intermediate motion.
    repeats = max(1, round(25 * sample_interval / speed))
    fps = repeats * speed / sample_interval
    writer = imageio_ffmpeg.write_frames(str(path), SIZE, fps=fps, codec="libx264", quality=8,
                                        output_params=["-threads", "1", "-movflags", "+faststart"], ffmpeg_log_level="error")
    writer.send(None)
    try:
        for image in images:
            pixels = np.asarray(image)
            for _ in range(repeats):
                writer.send(pixels)
    finally:
        writer.close()
    return repeats, fps


def gallery(destination, records, speed, ell, initial_conditions_mode="learned"):
    true_reset = initial_conditions_mode == "true_fixed"
    sections = []
    for kind, heading in (("test", "Predicted latent trajectories: r(P(...))"), ("train", "Readout used during physical training")):
        cards = []
        for row in records:
            if row["kind"] != kind:
                continue
            cards.append(f'<article><h3>{html.escape(row["title"])}</h3><p>{html.escape(row["source"])}</p>'
                         f'<video controls loop playsinline preload="none" poster="{row["stem"]}.jpg" src="{row["stem"]}.mp4"></video>'
                         f'<p><a href="{row["stem"]}.mp4" download>Download video</a></p></article>')
        if kind == "test":
            content = '<section>' + ''.join(cards[:2]) + '</section>'
            if len(cards) > 2:
                content += f'<details><summary>Show the other {len(cards)-2} held-out trajectories</summary><section>' + ''.join(cards[2:]) + '</section></details>'
        else:
            content = '<section>' + ''.join(cards) + '</section>'
        sections.append(f'<h2 id="{kind}">{heading}</h2>' + content)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    page += '<title>Trained physical readout: trajectory comparison</title><style>body{font:17px system-ui;color:#18263a;background:#ecf2f8;margin:28px auto;padding:0 22px;max-width:1460px}p{line-height:1.6}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,620px),1fr));gap:24px}article{background:white;padding:20px;border-radius:12px}video{width:100%}h3{margin-top:0}a{color:#0070b2}code{background:#dee8f2;padding:3px 6px}</style>'
    page += ('<h1>True-reset diagnostic: ground truth and the trained physical readout</h1>'
             if true_reset else '<h1>Ground truth and the trained physical readout</h1>')
    if true_reset:
        page += '<p><strong>Diagnostic model trained using fixed true initial states of training episodes.</strong> '
        page += 'Those states initialize the simulator targets; future true states and test resets are not supplied to the predictor.</p>'
    page += '<p><a href="#test">Predicted trajectories</a> · <a href="#train">Training simulator supervision</a></p>'
    page += f'<p>Same final checkpoint readout used during training: <code>r: 32 → 64 → 5</code> with GELU and a normalized sine/cosine pair. '
    page += 'Its outputs are <code>[p, v, sin(q), cos(q), w]</code>; the drawing uses <code>q = atan2(sin(q), cos(q))</code>. '
    page += f'These are state diagrams of those exact values, on equal metric axes, with a fixed {ell:g} m rod. The dataset pixels and trained weights are unchanged.</p>'
    page += f'<p>Playback is {speed:g}×; model samples are 0.10 s apart and held without interpolation. '
    page += 'The center panel of each test video is the requested <code>r(z_predicted)</code>. The right panel independently decodes actual video to reveal what the same readout can reconstruct; it does not feed the forecast.</p>'
    page += ('<p>Training videos show the final-checkpoint encoder in eval mode and the simulator rollout from the fixed true training resets. '
             if true_reset else
             '<p>Training videos show the final-checkpoint encoder in eval mode and the simulator rollout from its learned reset table. ')
    page += 'The original loss is <code>MSE(r(z)/[2,2,1,1,5], [p_sim/2,v_sim/2,sin(q_sim),cos(q_sim),w_sim/5])</code>. '
    page += ('Only initial states are supplied by the diagnostic; simulator targets are recomputed using the physical equations.</p>'
             if true_reset else 'The simulator-supervision panel is labeled separately from ground truth.</p>')
    page += ''.join(sections) + '</html>'
    (destination / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--limit", type=int, help="Optional maximum trajectories per split for a small preview")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    metadata_path = args.trajectories / "readout_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    ell, dataset = read_geometry(metadata)
    initial_conditions_mode = read_training_protocol(metadata)
    destination = args.trajectories / "videos"
    destination.mkdir(exist_ok=False)
    records = []
    for kind in ("test", "train"):
        source = args.trajectories / f"{kind}_trajectories.npz"
        queries = json.loads((args.trajectories / f"{kind}_queries.json").read_text())
        with np.load(source, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        times = values["times_s"]
        interval = float(times[1] - times[0])
        if not np.allclose(np.diff(times), interval):
            raise ValueError("Uniform sample times are required")
        for i, query in enumerate(queries[:args.limit] if args.limit is not None else queries):
            if kind == "test":
                series = [values[key][i] for key in ("truth", "forecast", "reconstruction")]
                raws = [np.stack([state_coordinates(s) for s in series[0]]), values["raw_forecast"][i], values["raw_reconstruction"][i]]
            else:
                series = [values[key][i] for key in ("truth", "reconstruction", "supervision")]
                raws = [np.stack([state_coordinates(s) for s in series[0]]), values["raw_reconstruction"][i], values["raw_supervision"][i]]
            if any(s.shape != (len(times), 4) for s in series) or any(r.shape != (len(times), 5) for r in raws):
                raise ValueError("Trajectory coordinates do not match the sample times")
            valid = values["valid"][i].astype(bool)
            if not valid.any():
                print(f"Skipped {query['path']}: no valid observed forecast context", flush=True)
                continue
            if not np.array_equal(np.flatnonzero(valid), np.arange(valid.sum())):
                raise ValueError("Valid video samples must be a contiguous prefix")
            series, raws = [s[valid] for s in series], [r[valid] for r in raws]
            display_times = times[valid]
            stem = f"{kind}_{i:03d}_{query['reset_mode']}"
            label = str(query["reset_mode"]).title() + f" / apparatus {query['apparatus_id']}"
            if query.get("group"):
                label += f" / {query['group']}"
            print(f"{kind} {i+1}/{len(queries)}: {query['path']}", flush=True)
            repeat, fps = write_video(destination / f"{stem}.mp4",
                (render_sample(series, raws, display_times, j, query, kind, args.speed, ell, dataset,
                               initial_conditions_mode) for j in range(len(display_times))), interval, args.speed)
            render_sample(series, raws, display_times, min(3, len(display_times)-1), query, kind, args.speed, ell,
                          dataset, initial_conditions_mode).save(destination / f"{stem}.jpg", quality=93)
            records.append({"kind": kind, "stem": stem, "title": label, "source": query["path"],
                            "samples": len(display_times), "frames": repeat * len(display_times), "fps": fps,
                            "source_arrays_sha256": digest(source)})
    gallery(destination, records, args.speed, ell, initial_conditions_mode)
    (destination / "videos.json").write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "script_sha256": digest(__file__),
        "uv_lock_sha256": digest(Path(__file__).parent / "uv.lock"),
        "readout": "the saved trained PhysicalReadout, no refitting", "interpolation": "none",
        "dataset": dataset, "rod_length_m": ell, "metadata_sha256": digest(metadata_path),
        "initial_conditions_mode": initial_conditions_mode,
        "supervision_uses_true_reset": initial_conditions_mode == "true_fixed",
        "playback_speed": args.speed, "projection": "isotropic, common axes for all three methods within each video",
        "videos": records}, indent=2) + "\n")
    print(f"Gallery: {destination.resolve() / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
