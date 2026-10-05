"""Small run-directory and provenance helpers shared by training and evaluation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def run_directory(root, prefix, run_dir=None, resume=False):
    """Create a fresh run, or lock an explicitly selected existing run for resume."""
    if resume and run_dir is None:
        raise ValueError("Resume requires an explicit --run-dir")
    if run_dir is None:
        suffix = f"-{os.environ['SLURM_JOB_ID']}" if os.environ.get("SLURM_JOB_ID") else ""
        output = Path(root) / f"{prefix}-{utc_stamp()}{suffix}"
    else:
        output = Path(run_dir)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"Resume run does not exist: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".writer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer owns run directory: {output}") from exc
        try:
            lock.seek(0); lock.truncate()
            lock.write(f"pid={os.getpid()}\n"); lock.flush()
            yield output
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


_SECRET = re.compile(r"(^|[_-])(api[_-]?key|password|passwd|secret|token|credentials?)([_-]|$)", re.I)


def nonsecret(value):
    """Exclude credential fields; this project never needs credentials in config."""
    if isinstance(value, dict):
        return {str(key): "<redacted>" if _SECRET.search(str(key)) else nonsecret(item)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [nonsecret(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def _command():
    result, redact_next = [], False
    for word in sys.argv:
        if redact_next:
            result.append("<redacted>"); redact_next = False
        elif _SECRET.search(word.split("=", 1)[0]):
            result.append(word.split("=", 1)[0] + "=<redacted>" if "=" in word else word)
            redact_next = "=" not in word
        else:
            result.append(word)
    return [sys.executable, *result]


def _code_state():
    repository = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    lock = repository / "uv.lock"
    return {"commit": commit, "dirty": bool(status) if status is not None else None,
            "uv_lock_sha256": file_digest(lock) if lock.exists() else None}


def dataset_identity(config):
    from pi_jepa.data import dataset_root

    path = dataset_root(config) / "manifest.json"
    manifest = json.loads(path.read_text())
    splits = {}
    for name in ("train", "validation", "test", "calibration", "query"):
        entries = manifest.get(name, [])
        serialized = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        splits[name] = {"entries": len(entries), "sha256": hashlib.sha256(serialized).hexdigest()}
    return {"manifest": str(path.resolve()), "sha256": file_digest(path),
            "schema_version": manifest.get("schema_version"),
            "dataset": manifest.get("dataset"), "splits": splits}


def write_provenance(output, config, checkpoints=None, extra=None, resume=False):
    """Save locally inspectable identity; resume cannot silently switch datasets."""
    output = Path(output)
    data = dataset_identity(config)
    record = {"utc": utc_stamp(), "command": _command(), "config": nonsecret(config),
              "seed": config["seed"], "code": _code_state(), "data": data,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "checkpoints": {}}
    for name, checkpoint in (checkpoints or {}).items():
        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        record["checkpoints"][name] = {"path": str(path.resolve()), "sha256": file_digest(path),
            "step": saved.get("step"), "mode": saved.get("mode"),
            "seed": saved.get("config", {}).get("seed"),
            "weights": saved.get("interface", {}).get("weights", "raw")}
        del saved
    if extra:
        record.update(nonsecret(extra))
    path = output / "provenance.json"
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Run has no provenance.json to validate: {output}")
        previous = json.loads(path.read_text())
        if previous["data"]["sha256"] != data["sha256"]:
            raise ValueError("Resume dataset manifest differs from the original run")
        old_sources = {key: value["sha256"] for key, value in previous["checkpoints"].items()}
        new_sources = {key: value["sha256"] for key, value in record["checkpoints"].items()}
        if old_sources != new_sources:
            raise ValueError("Resume source checkpoint differs from the original run")
        previous.setdefault("resumes", []).append(record)
        record = previous
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
    (output / "config.json").write_text(json.dumps(nonsecret(config), indent=2) + "\n")
    return record
