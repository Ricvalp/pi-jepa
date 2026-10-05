"""The encoder, physical apparatus, and image geometry required by a checkpoint."""
from copy import deepcopy

from pi_jepa.data import CAMERA, CART_LIMIT, SCHEMA_VERSION
from pi_jepa.physics import ELL, G, M_POLE
from pi_jepa.models import encoder_architecture, model_size, model_spec

CHECKPOINT_FORMAT_VERSION = 6


def encoder_interface():
    return {"encoder_normalization": {
        "backbone": "group_norm", "groups": 32,
        "projector_hidden": "layer_norm", "output": "none",
        "temporal_batching": "one_offset_across_episodes",
    }}


def model_interface(config, mode):
    """Resolve both neural capacities from the saved scientific configuration."""
    size = model_size(config)
    spec = model_spec(size)
    passive = config.get("dataset", "controlled") == "passive"
    predictor = (f"passive_mlp_32_{spec['passive_hidden']}x{spec['passive_hidden_layers']}_32" if passive else
                 f"causal_transformer_d{spec['controlled_width']}_l{spec['controlled_layers']}"
                 f"_h{spec['controlled_heads']}_32latent_10forces")
    return {"model_size": size, "model_spec": spec, **encoder_interface(),
            "architecture": {"encoder": encoder_architecture(size), "predictor": predictor,
                             "readout": None if passive and mode == "jepa" else "mlp_32_to_5"}}


def geometry_interface():
    return {
        "data_schema_version": SCHEMA_VERSION,
        "physics": {"m": M_POLE, "ell": ELL, "g": G},
        "camera": deepcopy(CAMERA),
        "cart_limit_m": CART_LIMIT,
    }


def validate_checkpoint_interface(checkpoint):
    """Reject missing or conflicting capacity metadata before restoring weights."""
    interface = checkpoint.get("interface", {})
    if interface.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint interface version 6 is required for explicit model sizes; "
                         "older checkpoints are unsupported; start a fresh run")
    if "config" not in checkpoint or checkpoint.get("mode") not in ("joint", "jepa", "readout"):
        raise ValueError("Checkpoint must include its model configuration and training mode")
    for key, expected in model_interface(checkpoint["config"], checkpoint["mode"]).items():
        if interface.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from its configured model size and architecture")
    for key, expected in geometry_interface().items():
        if interface.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from the current dataset/physical renderer")
