"""Explicit derived caches for learning images and fixed-reset simulator targets.

Preparation is a separate CPU step. Training only opens completed caches and
rejects stale inputs; production datasets are never written. The image cache
contains learning files only. Target preparation receives the already authorized
fixed training reset table and never opens any private truth file itself.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from pi_jepa import physics
from pi_jepa.runs import file_digest


CACHE_VERSION = 1


def _read_manifest(root):
    from pi_jepa.data import validate_manifest
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    return root, manifest


def _identity(root, manifest, splits):
    """Hash compressed learning sources once at preparation/open, never per batch."""
    return {"cache_version": CACHE_VERSION, "dataset": manifest["dataset"],
        "manifest_sha256": file_digest(root / "manifest.json"),
        "learning_sources": {split: [{"path": entry["path"],
            "sha256": file_digest(root / entry["path"])} for entry in manifest[split]]
            for split in splits}}


def _cache_path(root, manifest, cache_root, kind):
    path = Path(cache_root).resolve() / manifest["dataset"] / kind
    if path.is_relative_to(root):
        raise ValueError("Derived training caches must be outside the production corpus")
    return path


def _index(path):
    index = path / "index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Prepare the completed training cache first: {index}")
    return json.loads(index.read_text())


def _require_identity(metadata, expected, path):
    if metadata.get("identity") != expected:
        raise ValueError(f"Training cache source/manifest/solver identity differs: {path}; use a fresh cache root")


@contextmanager
def _writer(path):
    """One preparer per cache; publish an entire completed directory atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f".{path.name}.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def prepare_learning_cache(root, cache_root, splits=("train", "validation")):
    """Decompress RGB once into per-episode NPY files; never access truth."""
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= {"train", "validation"}:
        raise ValueError("Learning cache supports distinct train/validation splits only")
    root, manifest = _read_manifest(root)
    path = _cache_path(root, manifest, cache_root, "learning")
    identity = _identity(root, manifest, splits)
    with _writer(path):
        if path.exists():
            metadata = _index(path)
            _require_identity(metadata, identity, path)
            for split in splits:
                _check_learning_files(path, metadata["episodes"][split])
            return path
        temporary = Path(tempfile.mkdtemp(prefix=".learning-", dir=path.parent))
        try:
            metadata = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                "kind": "learning_npy", "splits": list(splits), "episodes": {}}
            for split in splits:
                (temporary / split).mkdir()
                metadata["episodes"][split] = []
                for index, entry in enumerate(manifest[split]):
                    files = {"frames": f"{split}/{index:06d}_rgb.npy",
                             "forces": f"{split}/{index:06d}_force.npy"}
                    with np.load(root / entry["path"], allow_pickle=False) as archive:
                        rgb = archive["rgb"]
                        forces = np.asarray(archive["force"], dtype=np.float32).reshape(-1)
                        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3 or len(forces) != len(rgb) - 1:
                            raise ValueError(f"Invalid learning episode: {entry['path']}")
                        np.save(temporary / files["frames"], rgb, allow_pickle=False)
                        np.save(temporary / files["forces"], forces, allow_pickle=False)
                        if manifest["dataset"] == "controlled":
                            files["theta"] = f"{split}/{index:06d}_theta.npy"
                            np.save(temporary / files["theta"], np.asarray(archive["theta"], dtype=np.float32), allow_pickle=False)
                    metadata["episodes"][split].append({"source": entry["path"], "files": files,
                        "frame_shape": list(rgb.shape),
                        "sizes": {key: (temporary / name).stat().st_size for key, name in files.items()}})
            (temporary / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return path


def _check_learning_files(path, entries):
    for entry in entries:
        for key, name in entry["files"].items():
            candidate = path / name
            if not candidate.is_file() or candidate.stat().st_size != entry["sizes"][key]:
                raise ValueError(f"Training learning cache is incomplete: {candidate}")


class LearningCache:
    """Read prepared NPY episodes through copy-on-write, disk-preserving mappings.

    Mapping mode 'c' makes NumPy/PyTorch views writable without writing to cache
    files. Normal training stacks only requested 65-frame crops into owned batch
    tensors. An accidental in-place operation cannot corrupt the shared cache.
    """
    def __init__(self, root, split, cache_root):
        root, manifest = _read_manifest(root)
        if split not in ("train", "validation"):
            raise ValueError("Learning cache supports train/validation only")
        self.cache_path = _cache_path(root, manifest, cache_root, "learning")
        self.metadata = _index(self.cache_path)
        if split not in self.metadata.get("splits", []):
            raise ValueError(f"Prepared learning cache does not contain {split}")
        _require_identity(self.metadata, _identity(root, manifest, self.metadata["splits"]), self.cache_path)
        self.entries = self.metadata["episodes"][split]
        if [entry["source"] for entry in self.entries] != [entry["path"] for entry in manifest[split]]:
            raise ValueError("Learning cache episode order differs from the dataset")
        _check_learning_files(self.cache_path, self.entries)

    def __getitem__(self, index):
        entry = self.entries[int(index)]
        # Only RGB needs a mapping. Copy the tiny force/theta arrays so a large
        # minibatch does not retain three open file mappings per episode.
        arrays = {key: np.load(self.cache_path / name, mmap_mode="c" if key == "frames" else None, allow_pickle=False)
                  for key, name in entry["files"].items()}
        if (arrays["frames"].dtype != np.uint8 or list(arrays["frames"].shape) != entry["frame_shape"]
                or arrays["forces"].dtype != np.float32 or arrays["forces"].shape != (len(arrays["frames"]) - 1,)):
            raise ValueError("Cached learning array shape/dtype differs from its index")
        if "theta" in arrays and (arrays["theta"].shape != (2,) or arrays["theta"].dtype != np.float32):
            raise ValueError("Cached apparatus parameters must be float32 [2]")
        return {key: torch.from_numpy(value) for key, value in arrays.items()}


def _fixed_identity(root, manifest, table, reset_metadata):
    import hashlib
    if not getattr(table, "is_fixed", False) or list(table.parameters()):
        raise ValueError("Simulator target caching requires fixed true training resets")
    if reset_metadata.get("mode") != "true_fixed" or reset_metadata.get("split") != "train":
        raise ValueError("Fixed target cache requires explicit training reset provenance")
    states = table.states.detach().cpu()
    if states.dtype != torch.float64 or states.shape != (len(manifest["train"]), 4):
        raise ValueError("Fixed reset table shape/dtype differs from the training manifest")
    return {**_identity(root, manifest, ("train",)),
        "reset_source_sha256": reset_metadata["source_sha256"],
        "initial_states_sha256": hashlib.sha256(states.numpy().astype("<f8", copy=False).tobytes()).hexdigest(),
        "solver_sha256": file_digest(physics.__file__),
        "solver": {"m": physics.M_POLE, "ell": physics.ELL, "g": physics.G,
                   "dt": physics.DT, "substeps": physics.SUBSTEPS, "torch_version": str(torch.__version__)},
        "numerics": "CPU float64 reset/state with float32 recorded forces and float32 public theta",
        "state_order": ["p", "v", "q", "w"], "split": "train"}


@torch.no_grad()
def prepare_fixed_target_cache(root, cache_root, table, reset_metadata, batch_size=64):
    """Precompute current fixed-reset targets using learning forces and parameters.

    The float32 action/parameter conversion intentionally matches batch_from;
    converting those original NPZ arrays directly to float64 would change the
    existing diagnostic's numerical protocol for controlled episodes.
    """
    if batch_size < 1:
        raise ValueError("Target preparation batch size must be positive")
    root, manifest = _read_manifest(root)
    path = _cache_path(root, manifest, cache_root, "fixed_targets")
    identity = _fixed_identity(root, manifest, table, reset_metadata)
    with _writer(path):
        if path.exists():
            metadata = _index(path)
            _require_identity(metadata, identity, path)
            _check_target_files(path, metadata)
            return path
        temporary = Path(tempfile.mkdtemp(prefix=".fixed-targets-", dir=path.parent))
        try:
            forces, theta, lengths = [], [], []
            for entry in manifest["train"]:
                with np.load(root / entry["path"], allow_pickle=False) as archive:
                    force = np.asarray(archive["force"], dtype=np.float32).reshape(-1)
                    parameter = (np.array([1., .25], dtype=np.float32) if manifest["dataset"] == "passive"
                                 else np.asarray(archive["theta"], dtype=np.float32))
                forces.append(torch.from_numpy(force))
                theta.append(torch.from_numpy(parameter))
                lengths.append(len(force) + 1)
            offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
            np.save(temporary / "offsets.npy", offsets, allow_pickle=False)
            targets = np.lib.format.open_memmap(temporary / "states.npy", mode="w+",
                dtype=np.float64, shape=(int(offsets[-1]), 4))
            initial = table.states.detach().cpu()
            for start in range(0, len(lengths), batch_size):
                stop = min(start + batch_size, len(lengths))
                max_force = max(lengths[start:stop]) - 1
                force_batch = torch.zeros(stop - start, max_force, dtype=torch.float32)
                for row, force in enumerate(forces[start:stop]):
                    force_batch[row, :len(force)] = force
                simulated = physics.rollout(initial[start:stop], torch.stack(theta[start:stop]), force_batch)
                for row, index in enumerate(range(start, stop)):
                    targets[offsets[index]:offsets[index + 1]] = simulated[row, :lengths[index]].numpy()
            targets.flush()
            del targets
            metadata = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                "kind": "fixed_reset_simulator_targets", "episodes": len(lengths), "states": int(offsets[-1]),
                "source": "Computed from fixed resets and learning-only applied forces/public parameters; no stored future truth read",
                "files_sha256": {name: file_digest(temporary / name) for name in ("states.npy", "offsets.npy")}}
            (temporary / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return path


def _check_target_files(path, metadata):
    for name, digest in metadata["files_sha256"].items():
        if not (path / name).is_file() or file_digest(path / name) != digest:
            raise ValueError(f"Fixed simulator target cache is incomplete or changed: {path / name}")


class FixedTargetCache:
    """Gather exact requested episode-time states, without a solver or gradients."""
    def __init__(self, root, cache_root, table, reset_metadata):
        root, manifest = _read_manifest(root)
        self.cache_path = _cache_path(root, manifest, cache_root, "fixed_targets")
        self.metadata = _index(self.cache_path)
        _require_identity(self.metadata, _fixed_identity(root, manifest, table, reset_metadata), self.cache_path)
        _check_target_files(self.cache_path, self.metadata)
        self.states = np.load(self.cache_path / "states.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.cache_path / "offsets.npy", mmap_mode="r", allow_pickle=False)
        if (self.states.dtype != np.float64 or self.states.shape != (self.metadata["states"], 4)
                or self.offsets.shape != (len(manifest["train"]) + 1,) or self.offsets.dtype != np.int64
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.states) or np.any(np.diff(self.offsets) < 1)):
            raise ValueError("Invalid packed simulator target cache")

    def gather(self, ids, raw_endpoints, device=None):
        ids = torch.as_tensor(ids).detach().cpu().numpy()
        endpoints = torch.as_tensor(raw_endpoints).detach().cpu().numpy()
        if ids.ndim != 1 or endpoints.ndim != 2 or len(endpoints) != len(ids):
            raise ValueError("Target lookup expects episode IDs [B] and raw endpoints [B,K]")
        if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(endpoints.dtype, np.integer):
            raise ValueError("Episode IDs and target endpoints must be integers")
        if np.any(ids < 0) or np.any(ids >= len(self.offsets) - 1):
            raise ValueError("Target lookup episode ID is outside the training split")
        lengths = np.diff(self.offsets)[ids]
        if np.any(endpoints < 0) or np.any(endpoints >= lengths[:, None]):
            raise ValueError("Target lookup cannot extrapolate beyond an episode")
        values = self.states[self.offsets[ids, None] + endpoints]
        return torch.from_numpy(values).to(device=device)
