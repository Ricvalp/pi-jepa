"""Render recorded cart-pole RGB with the actual force on each outgoing interval.

This standalone uv project leaves a running training environment unchanged.
From the repository root:
  uv run --project scripts/dataset_videos --locked python scripts/dataset_videos/render.py --data-root workspace/data --dataset controlled
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = (1280, 768)
IMAGE_ORIGIN = (124, 166)
IMAGE_SIZE = 432
FORCE_LIMIT = 5.0
INK = (25, 39, 59)
MUTED = (95, 110, 128)
BLUE = (0, 113, 179)
ORANGE = (209, 85, 25)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_clips(manifest, all_clips=False):
    """Two training examples per collection family and one difficult prefix."""
    if manifest.get("schema_version") != 3:
        raise ValueError("Dataset schema 3 is required (0.85 m rod, 1.5 m cart limit)")
    splits = ("train", "validation", "test") if manifest["dataset"] == "passive" else ("train", "validation", "calibration", "query")
    rows = [(split, entry) for split in splits for entry in manifest.get(split, [])]
    selected, counts = [], {}
    for split, entry in rows:
        family = entry["collection_family"]
        if all_clips or (split == "train" and counts.get(family, 0) < 2):
            selected.append((split, entry)); counts[family] = counts.get(family, 0) + 1
    difficult = next(((split, entry) for split, entry in rows
                      if entry["termination_reason"] not in ("max_duration", "completed", "duration")), None)
    if difficult and difficult not in selected:
        selected.append(difficult)
    return [{"split": split, "kind": entry["collection_family"], "path": entry["path"],
             "apparatus_id": entry["apparatus_id"], "group": entry.get("group"),
             "truth_path": entry["truth"], "termination_reason": entry["termination_reason"],
             "camera": manifest["camera"]} for split, entry in selected]


def current_force(forces, index):
    if index < 0 or index > len(forces):
        raise IndexError("Frame index is outside the trajectory")
    return float(forces[index]) if index < len(forces) else None


def load_clip(data_root, record):
    """Truth cart position places the overlay only; motion/forces are recorded data."""
    data_root = Path(data_root)
    with np.load(data_root / record["path"], allow_pickle=False) as saved:
        mode = str(saved["reset_mode"].item())
        clip = {"frames": saved["rgb"].copy(), "forces": saved["force"].reshape(-1).copy(),
                "timestamps": saved["t"].copy(), "dt": .01,
                "reset_mode": {"downward": 0, "nonlinear": 1, "upright": 2}.get(mode, mode)}
    with np.load(data_root / record["truth_path"], allow_pickle=False) as saved:
        clip["positions"] = saved["state"][:, 0].copy()
    count = len(clip["frames"])
    if clip["frames"].shape != (count, 96, 96, 3) or clip["frames"].dtype != np.uint8:
        raise ValueError("Expected recorded uint8 RGB frames [T,96,96,3]")
    if count < 2 or clip["forces"].shape != (count - 1,) or clip["timestamps"].shape != (count,) or clip["positions"].shape != (count,):
        raise ValueError("Frames, forces, timestamps and cart positions are not aligned")
    clip["dt"], clip["reset_mode"] = float(clip["dt"]), int(clip["reset_mode"])
    if clip["dt"] <= 0 or not np.allclose(np.diff(clip["timestamps"]), clip["dt"], rtol=1e-5, atol=1e-6):
        raise ValueError("Expected a positive, uniform recorded frame interval")
    if any(not np.isfinite(clip[key]).all() for key in ("forces", "positions", "timestamps")):
        raise ValueError("Nonfinite recorded force, position or timestamp")
    if np.max(np.abs(clip["forces"])) > FORCE_LIMIT + 1e-5:
        raise ValueError("Recorded forces exceed the fixed +/-5 N visualization scale")
    clip["record"] = record
    return clip


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def title(clip):
    record = clip["record"]
    return f"{record['split'].upper()} / {record['kind'].replace('_', ' ').upper()}"


def draw_arrow(draw, origin, force):
    """Length is proportional to |u|, with one fixed 20px/N scale for all clips."""
    if force is None or abs(force) < 1e-8:
        return
    color = BLUE if force > 0 else ORANGE
    x, y = origin
    length = abs(force) * 20
    sign = 1 if force > 0 else -1
    end = x + sign * length
    head = min(12, length * .48)
    draw.line((x, y, end, y), fill="white", width=11)
    draw.line((x, y, end, y), fill=color, width=6)
    draw.polygon([(end, y), (end - sign * head, y - head * .7), (end - sign * head, y + head * .7)], fill=color)


def render_frame(clip, index):
    force = current_force(clip["forces"], index)
    now = float(clip["timestamps"][index])
    end_time = float(clip["timestamps"][-1])
    record = clip["record"]
    image = Image.new("RGB", SIZE, (238, 243, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SIZE[0], 103), fill=INK)
    draw.text((30, 17), title(clip), font=font(29, True), fill="white")
    group = f" / {record['group']}" if record["group"] else ""
    draw.text((32, 61), f"Apparatus {record['apparatus_id']}{group}  |  {record['path']}", font=font(18), fill=(191, 210, 229))
    draw.text((990, 25), f"t = {now:4.2f} s", font=font(29, True), fill="white")
    draw.text((993, 64), f"frame {index:03d} / {len(clip['frames']) - 1:03d}", font=font(17), fill=(191, 210, 229))
    draw.rounded_rectangle((24, 124, 656, 742), radius=16, fill="white")
    draw.rounded_rectangle((676, 124, 1256, 354), radius=16, fill="white")
    draw.rounded_rectangle((676, 374, 1256, 742), radius=16, fill="white")
    resized = Image.fromarray(clip["frames"][index]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    image.paste(resized, IMAGE_ORIGIN)
    scale = IMAGE_SIZE / 96
    xlim = record["camera"]["x_limits_m"]
    px = IMAGE_ORIGIN[0] + (((float(clip["positions"][index]) - xlim[0]) / (xlim[1]-xlim[0]) * 95 + .5) * scale - .5)
    ylim = record["camera"]["y_limits_m"]
    pivot_y = ylim[1] / (ylim[1] - ylim[0]) * 95
    py = IMAGE_ORIGIN[1] + (pivot_y + .5) * scale - .5
    draw_arrow(draw, (px, py), force)
    draw.text((46, 618), "Recorded RGB + applied-force overlay", font=font(22, True), fill=INK)
    draw.text((46, 657), "Arrow: + right, - left; length proportional to |u|", font=font(19), fill=MUTED)
    control = "Recorded frames; fixed metric camera and geometry"
    draw.text((46, 695), control, font=font(19), fill=MUTED)
    draw.text((701, 142), "FORCE APPLIED TO THE CART", font=font(19, True), fill=MUTED)
    color = MUTED if force is None or force == 0 else BLUE if force > 0 else ORANGE
    value = "END OF TRAJECTORY" if force is None else f"{force:+.3f} N"
    draw.text((700, 180), value, font=font(31 if force is None else 48, True), fill=color)
    direction = "No next action is recorded" if force is None else "RIGHT (+)" if force > 0 else "LEFT (-)" if force < 0 else "ZERO FORCE"
    draw.text((704, 244), direction, font=font(20, True), fill=color)
    gx0, gx1, gy = 740, 1200, 294
    zero = (gx0 + gx1) / 2
    draw.line((gx0, gy, gx1, gy), fill=(215, 225, 235), width=10)
    if force is not None:
        endpoint = zero + force / FORCE_LIMIT * (gx1 - gx0) / 2
        draw.line((zero, gy, endpoint, gy), fill=color, width=10)
    draw.line((zero, gy - 11, zero, gy + 11), fill=INK, width=2)
    for x, label in ((gx0, "-5 N"), (zero, "0"), (gx1, "+5 N")):
        draw.text((x, gy + 16), label, anchor="mt", font=font(17), fill=MUTED)

    draw.text((701, 391), "RECORDED FORCE PROGRAM", font=font(19, True), fill=INK)
    x0, y0, x1, y1 = 742, 443, 1215, 640
    def point(t, u):
        return (x0 + (float(t) - float(clip["timestamps"][0])) / (end_time - float(clip["timestamps"][0])) * (x1 - x0),
                (y0 + y1) / 2 - float(u) / FORCE_LIMIT * (y1 - y0) / 2)
    for u in (-5, 0, 5):
        yy = point(now, u)[1]
        draw.line((x0, yy, x1, yy), fill=(217, 226, 235), width=1)
        draw.text((x0 - 13, yy), f"{u:+d}" if u else "0", anchor="rm", font=font(16), fill=MUTED)
    for k, u in enumerate(clip["forces"]):
        a, b = point(clip["timestamps"][k], u), point(clip["timestamps"][k + 1], u)
        draw.line((*a, *b), fill=(148, 165, 181), width=2)
        if k:
            previous = point(clip["timestamps"][k], clip["forces"][k - 1])
            draw.line((*previous, *a), fill=(148, 165, 181), width=2)
        if k < index:
            draw.line((*a, *b), fill=BLUE, width=3)
    cursor_x = point(now, 0)[0]
    draw.line((cursor_x, y0, cursor_x, y1), fill=ORANGE, width=2)
    if force is not None:
        cx, cy = point(now, force)
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=color)
    for time in (float(clip["timestamps"][0]), (end_time + float(clip["timestamps"][0])) / 2, end_time):
        draw.text((point(time, 0)[0], y1 + 12), f"{time:.2f}s", anchor="mt", font=font(16), fill=MUTED)
    interval = "Terminal frame: no outgoing force interval" if force is None else f"u[{index}] acts on [{now:.2f}, {float(clip['timestamps'][index+1]):.2f}) s"
    draw.text((700, 692), interval, font=font(18), fill=INK)
    return image


def write_mp4(path, frames, fps):
    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(str(path), SIZE, fps=fps, codec="libx264", quality=8,
                                        output_params=["-threads", "1", "-movflags", "+faststart"],
                                        ffmpeg_log_level="error")
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.asarray(frame))
    finally:
        writer.close()


def gallery(destination, summaries, speed):
    cards = []
    for row in summaries:
        label = html.escape(row["title"])
        cards.append(f'<article><h2>{label}</h2><p>{html.escape(row["source"])}</p>'
                     f'<video controls loop playsinline preload="metadata" poster="{row["stem"]}.jpg" src="{row["stem"]}.mp4"></video>'
                     f'<p><a href="{row["stem"]}.mp4" download>Download MP4</a> · '
                     f'{row["physical_duration_s"]:.2f}s simulated · force {row["min_force_n"]:+.3f} to {row["max_force_n"]:+.3f} N</p></article>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    page += '<title>Cart-pole dataset: applied forces</title><style>body{font:17px system-ui;background:#eef3f8;color:#19273b;max-width:1280px;margin:32px auto;padding:0 22px}h1{font-size:32px}p{line-height:1.5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}article{background:white;padding:20px;border-radius:12px}h2{font-size:20px}video{width:100%;border-radius:8px}a{color:#0071b3}</style>'
    page += f'<h1>Recorded cart-pole trajectories and applied forces</h1><p>{len(summaries)} fixed examples. Playback: {speed:g}× real time. '
    page += 'Blue/right is positive force; orange/left is negative. The arrow and number show the force applied from the displayed frame to the next frame. '
    page += 'Default previews explicitly use every fourth 100 Hz saved frame at 25 fps. Every clip uses the same ±5 N scale. The last frame has no next action. RGB frames and actions are the recorded dataset; private cart position only anchors the annotation.</p>'
    page += '<main>' + ''.join(cards) + '</main></html>'
    (destination / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "workspace/data")))
    parser.add_argument("--output-root", type=Path, default=Path("workspace/evaluations"))
    parser.add_argument("--output", type=Path, help="Explicit fresh output directory")
    parser.add_argument("--dataset", choices=["passive", "controlled"], default="controlled")
    parser.add_argument("--speed", type=float, default=1., help="Playback speed; default real time, extra slow-motion optional")
    parser.add_argument("--stride", type=int, choices=[1, 4], default=4, help="1: all records at 100fps; 4: explicit subsampling at 25fps")
    parser.add_argument("--all", action="store_true", help="Render every trajectory instead of two examples per family")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    args.data_root = args.data_root / args.dataset
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or args.output_root / f"dataset-forces-{args.dataset}-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = args.data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = select_clips(manifest, args.all)
    summaries = []
    for number, record in enumerate(records, start=1):
        clip = load_clip(args.data_root, record)
        stem = record["path"].replace("/", "_").removesuffix(".npz")
        fps = args.speed / (clip["dt"] * args.stride)
        indices = list(range(0, len(clip["frames"]), args.stride))
        print(f"[{number}/{len(records)}] {record['path']} -> {stem}.mp4", flush=True)
        write_mp4(destination / f"{stem}.mp4", (render_frame(clip, k) for k in indices), fps)
        preview_index = int(np.argmax(np.abs(clip["forces"])))
        render_frame(clip, preview_index).save(destination / f"{stem}.jpg", quality=92)
        summaries.append({"stem": stem, "title": title(clip), "source": record["path"],
                          "record": record, "frame_count": len(indices), "recorded_frame_count": len(clip["frames"]), "record_stride": args.stride, "dt_s": clip["dt"],
                          "physical_duration_s": float(clip["timestamps"][-1] - clip["timestamps"][0]),
                          "fps": fps, "playback_speed": args.speed,
                          "min_force_n": float(clip["forces"].min()), "max_force_n": float(clip["forces"].max()),
                          "source_sha256": digest(args.data_root / record["path"]),
                          "truth_sha256": digest(args.data_root / record["truth_path"])})
    gallery(destination, summaries, args.speed)
    metadata = {"utc": stamp, "command": [sys.executable, *sys.argv], "data_root": str(args.data_root.resolve()),
                "manifest_sha256": digest(manifest_path), "manifest_schema": manifest["schema_version"],
                "dataset_seed": manifest.get("seed", 42),
                "script_sha256": digest(__file__), "uv_lock_sha256": digest(Path(__file__).parent / "uv.lock"),
                "force_timing": "u[k] applies on [timestamp[k], timestamp[k+1]); terminal frame has no outgoing action",
                "force_units": "N", "force_limit_n": FORCE_LIMIT,
                "arrow_scale_pixels_per_newton": 20,
                "truth_usage": "cart position only, for overlay placement; not supplied to a model",
                "selection": "all" if args.all else "two training examples per family plus one difficult prefix when available",
                "videos": summaries}
    (destination / "videos.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Video gallery: {destination.resolve() / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
