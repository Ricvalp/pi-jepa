"""The encoder, physical apparatus, and image geometry required by a checkpoint."""
from copy import deepcopy

from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION
from pi_jepa.physics import ELL, G, M_POLE
from pi_jepa.models import ENCODER_ARCHITECTURE

CHECKPOINT_FORMAT_VERSION = 5


def encoder_interface():
    return {"encoder_normalization": {
        "backbone": "group_norm", "groups": 32,
        "projector_hidden": "layer_norm", "output": "none",
        "temporal_batching": "one_offset_across_episodes",
    }}


def geometry_interface():
    return {
        "data_schema_version": SCHEMA_VERSION,
        "physics": {"m": M_POLE, "ell": ELL, "g": G},
        "camera": deepcopy(CAMERA),
        "cart_limit_m": CART_LIMIT,
    }


def validate_checkpoint_interface(checkpoint):
    """Reject old BatchNorm weights before inference or optimizer state restoration."""
    interface = checkpoint.get("interface", {})
    if interface.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint interface version 5 is required for the GroupNorm/LayerNorm encoder; "
                         "old BatchNorm checkpoints are unsupported; start a fresh run")
    if interface.get("architecture", {}).get("encoder") != ENCODER_ARCHITECTURE:
        raise ValueError("Checkpoint encoder architecture differs from the GroupNorm/LayerNorm encoder")
    for key, expected in encoder_interface().items():
        if interface.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from the causal, batch-independent encoder")
    for key, expected in geometry_interface().items():
        if interface.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from the current dataset/physical renderer")
