"""Explicit, training-only oracle resets for the fixed-reset diagnostic.

Ordinary training never calls this loader. Only the exact_initial_state member
of training truth files is read; future states and held-out truth remain unused.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pi_jepa.runs import file_digest


class FixedInitialConditions(nn.Module):
    """Immutable physical resets in manifest training order, with no parameters.

    Float64 preserves the recorded simulator reset exactly and makes the target
    simulation use float64. Neural inputs and network parameters remain float32.
    """

    is_fixed = True

    def __init__(self, states, reset_modes):
        super().__init__()
        states = torch.as_tensor(states, dtype=torch.float64).detach().clone()
        if states.shape != (len(reset_modes), 4) or not torch.isfinite(states).all():
            raise ValueError("Fixed initial states must be finite [episodes,4] values")
        self.register_buffer("states", states)
        self.register_buffer("reset_modes", torch.as_tensor(reset_modes, dtype=torch.float32).clone())

    def forward(self, ids):
        return self.states[ids]


def load_fixed_training_initial_conditions(root):
    """Load only explicit training reset metadata, never future/held-out truth.

    The digest binds each reset to its ordered manifest episode and its source path.
    It intentionally hashes the used reset values, not unrelated future truth.
    """
    from pi_jepa.data import validate_manifest

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    states, paths, modes = [], [], []
    permitted = root / "truth" / "train"
    for entry in manifest["train"]:
        path = (root / entry["truth"]).resolve()
        if not path.is_relative_to(permitted):
            raise ValueError("Fixed reset diagnostic may read only truth/train files")
        with np.load(path, allow_pickle=False) as archive:
            if "exact_initial_state" not in archive:
                raise ValueError("Training truth lacks exact_initial_state; future states are not a fallback")
            state = np.asarray(archive["exact_initial_state"], dtype=np.float64)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid physical reset in training truth: {entry['truth']}")
        states.append(state)
        paths.append({"episode": entry["path"], "reset_source": entry["truth"]})
        modes.append(entry["reset_mode"])
    if not states:
        raise ValueError("Fixed reset diagnostic requires training episodes")
    states = np.asarray(states, dtype="<f8")
    digest = hashlib.sha256(json.dumps(paths, sort_keys=True, separators=(",", ":")).encode())
    digest.update(states.tobytes(order="C"))
    metadata = {
        "mode": "true_fixed", "source": "training_truth_exact_initial_state_only",
        "diagnostic_oracle": True, "split": "train", "episodes": len(states),
        "state_order": ["p", "v", "q", "w"], "dtype": "float64",
        "source_sha256": digest.hexdigest(), "manifest_sha256": file_digest(manifest_path),
        "ordering": "manifest.train", "neural_input": False,
    }
    return FixedInitialConditions(states, modes), metadata


def load_initial_conditions_state(checkpoint):
    """Rebuild a checkpoint's simulator reset table without accessing any truth."""
    state = checkpoint.get("initial_conditions")
    if state is None:
        return None
    protocol = checkpoint["config"]["training"].get("initial_conditions", "learned")
    if protocol == "true_fixed":
        if checkpoint.get("mode") != "joint":
            raise ValueError("Fixed true resets are supported only for the joint diagnostic")
        table = FixedInitialConditions(state["states"], state["reset_modes"])
    elif protocol == "learned":
        # Import lazily so the ordinary training module can retain its public API.
        from pi_jepa.train import InitialConditions
        table = InitialConditions(state["reset_modes"])
    else:
        raise ValueError("Unknown initial-condition protocol in checkpoint")
    table.load_state_dict(state)
    return table
