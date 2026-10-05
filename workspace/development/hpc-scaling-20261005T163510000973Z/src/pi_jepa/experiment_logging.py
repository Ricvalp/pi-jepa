"""Local training diagnostics and optional, explicitly enabled W&B logging."""
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import random
import re
import uuid

import numpy as np
import torch

from pi_jepa.runs import nonsecret


@contextmanager
def preserve_rng_state():
    """Monitoring must not consume random numbers used by paired training runs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# A short name is convenient when wrapping diagnostic forward passes as well.
preserve_rng = preserve_rng_state


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("TrainingLogger metrics must be scalar")
        value = value.detach().item()
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (bool, int, float)):
        raise TypeError(f"TrainingLogger expected a numeric scalar, got {type(value).__name__}")
    return value if math.isfinite(value) else None


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TrainingLogger:
    """Log diagnostics locally even when W&B is disabled.

    `train/update` is the chart axis, separate from the SDK's increasing history
    index. On checkpoint resume, local diagnostics after the checkpoint are
    removed. Existing online history cannot be deleted by this logger; replayed
    updates are appended with the same `train/update` values. Offline resumes
    create separate, explicitly linked segments because W&B cannot resume an
    offline run. No model, source code, or checkpoint artifacts are uploaded.
    """

    def __init__(self, config, stage, output, resume=False, start_step=0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output / "diagnostics.jsonl"
        self.metadata_path = self.output / "wandb_run.json"
        self.run = None
        self.sdk = None
        self.run_metadata = None
        self._finished = False
        settings = config.get("wandb", {})
        mode = settings.get("mode", "online")
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb.mode must be online, offline, or disabled")
        self.enabled = bool(settings.get("enabled", False)) and mode != "disabled"
        if start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if self.history_path.exists() and not resume:
            raise FileExistsError(f"{self.history_path} exists; use --resume or a new run directory")
        if self.enabled:
            with preserve_rng_state():
                self._start(config, settings, stage, mode, resume, start_step)
        try:
            self._prepare_history(resume, start_step)
        except BaseException:
            self.finish(exit_code=1)
            raise

    def _prepare_history(self, resume, start_step):
        if not self.history_path.exists():
            return
        if not resume:
            raise FileExistsError(f"{self.history_path} exists; use --resume or a new run directory")
        # A crash may leave a partially written final line. Earlier malformed
        # rows are genuine corruption and must not be silently discarded.
        lines = self.history_path.read_text().splitlines()
        retained = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise
            if int(row["step"]) <= start_step:
                retained.append(json.dumps(row, allow_nan=False) + "\n")
        temporary = self.history_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(retained))
        temporary.replace(self.history_path)

    def _start(self, config, settings, stage, mode, resume, start_step):
        previous = None
        if self.metadata_path.exists():
            if not resume:
                raise FileExistsError(f"{self.metadata_path} exists; use --resume or a new run directory")
            previous = json.loads(self.metadata_path.read_text())
        project = settings.get("project") or "physics-jepa-cartpole"
        entity = settings.get("entity")
        group = settings.get("group") or f"{self.output.parent.name}-seed{config['seed']}"
        name = self.output.name
        if previous:
            for key, requested in (("project", project), ("group", group), ("stage", stage)):
                if previous[key] != requested:
                    raise ValueError(f"Cannot resume W&B with a different {key}; use a new runs directory")
            if entity is not None and previous.get("entity") not in (None, entity):
                raise ValueError("Cannot resume W&B with a different entity; use a new runs directory")
            entity = entity or previous.get("entity")
        online_resume = bool(previous and previous["mode"] == "online" and mode == "online")
        run_id = previous["id"] if online_resume else uuid.uuid4().hex[:12]
        segments = list(previous.get("segments", [])) if previous else []
        if previous and not online_resume:
            segments.append({key: previous.get(key) for key in ("id", "mode", "start_step", "name")})

        try:
            self.sdk = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("W&B logging was requested but wandb is not installed. "
                               "Run uv sync --locked --extra tracking, "
                               "or use --wandb-mode disabled.") from exc
        kwargs = {
            "project": project, "entity": entity, "group": group, "name": name,
            "id": run_id, "job_type": stage, "mode": mode,
            "dir": str(self.output.resolve()), "config": nonsecret({**config, "stage": stage}),
            "save_code": False, "settings": {"disable_code": True, "disable_git": True},
        }
        if mode == "online":
            kwargs["resume"] = "must" if online_resume else "never"
            kwargs["force"] = True
        try:
            self.run = self.sdk.init(**kwargs)
            # Do not turn an explicitly requested online experiment into an
            # untracked run if login or SDK environment settings disabled it.
            actual_mode = getattr(getattr(self.run, "settings", None), "mode", mode)
            if actual_mode != mode:
                raise RuntimeError(f"W&B initialized in {actual_mode!r} mode; requested {mode!r}")
            self.run.define_metric("train/update")
            self.run.define_metric("*", step_metric="train/update")
            self.run_metadata = {
                "id": self.run.id, "project": project,
                "entity": getattr(self.run, "entity", None) or entity,
                "mode": mode, "name": name, "group": group, "stage": stage,
                "start_step": int(start_step), "segments": segments,
                "previous_run_id": previous["id"] if previous and not online_resume else None,
                "url": getattr(self.run, "url", None) if mode == "online" else None,
                "resume_policy": "same online run" if mode == "online" else "new segment on checkpoint resume",
            }
            _write_json(self.metadata_path, self.run_metadata)
        except Exception as exc:
            self.finish(exit_code=1)
            raise RuntimeError("Could not initialize requested W&B logging. For online logging, "
                               "run .venv/bin/wandb login (or set WANDB_API_KEY) and check the "
                               "project/entity and saved run identity. Use --wandb-mode offline "
                               "to record locally without authentication.") from exc

    def log(self, metrics, step, figures=None, histograms=None):
        """Write scalars and optional matplotlib figures / histogram arrays."""
        if self._finished:
            raise RuntimeError("Cannot log after TrainingLogger.finish()")
        if step < 0:
            raise ValueError("step must be nonnegative")
        scalars = {str(key): _scalar(value) for key, value in metrics.items()}
        scalars["train/update"] = int(step)
        row = {"step": int(step), "metrics": scalars}
        payload = {key: value for key, value in scalars.items() if value is not None}
        with preserve_rng_state():
            media = {}
            if figures or histograms:
                media_dir = self.output / "diagnostics_media"
                media_dir.mkdir(parents=True, exist_ok=True)
                for key, figure in (figures or {}).items():
                    filename = f"{step:07d}_{_media_name(key)}.png"
                    path = media_dir / filename
                    figure.savefig(path, dpi=120, bbox_inches="tight")
                    media[key] = str(path.relative_to(self.output))
                    if self.run is not None:
                        payload[key] = self.sdk.Image(str(path))
                for key, values in (histograms or {}).items():
                    if isinstance(values, torch.Tensor):
                        values = values.detach().cpu().numpy()
                    values = np.asarray(values).reshape(-1)
                    values = values[np.isfinite(values)]
                    if not len(values):
                        continue
                    path = media_dir / f"{step:07d}_{_media_name(key)}.npz"
                    counts, edges = np.histogram(values, bins=64)
                    np.savez_compressed(path, counts=counts, edges=edges)
                    media[key] = str(path.relative_to(self.output))
                    if self.run is not None:
                        payload[key] = self.sdk.Histogram(np_histogram=(counts, edges))
            if media:
                row["media"] = media
            with self.history_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            if self.run is not None:
                # Passing the optimizer update as SDK `step` would discard
                # replayed metrics when resuming an older checkpoint.
                self.run.log(payload)
            if figures:
                import matplotlib.pyplot as plt
                for figure in figures.values():
                    plt.close(figure)

    def finish(self, exit_code=0):
        if self._finished:
            return
        self._finished = True
        if self.run is not None:
            with preserve_rng_state():
                self.run.finish(exit_code=exit_code)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish(exit_code=0 if exc_type is None else 1)
        return False


def _media_name(key):
    """Metric names are not filesystem paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key)).strip(".") or "media"
