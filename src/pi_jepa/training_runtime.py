"""Explicit neural precision settings; physical targets keep simulator precision."""
from contextlib import nullcontext

import torch


def configure_runtime(config, device):
    settings = config["training"]
    precision = settings.get("precision", "float32")
    if precision not in ("float32", "bf16"):
        raise ValueError("training.precision must be float32 or bf16")
    if device.type == "cuda":
        with torch.cuda.device(device):
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support requested BF16 training")
        torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", False)
    return precision


def neural_autocast(device, precision):
    """Autocast E/P forwards; master weights, readout and JEPA losses stay FP32."""
    if precision == "float32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("Unknown neural precision")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
