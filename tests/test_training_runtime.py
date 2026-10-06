"""Mixed precision keeps JEPA statistics and physical supervision at their intended precision."""
import importlib.util
from pathlib import Path

import pytest
import torch

from pi_jepa.initial_conditions import FixedInitialConditions
from pi_jepa.losses import SIGReg, physics_loss, simulate_window
from pi_jepa.models import Encoder, PassivePredictor, PhysicalReadout, Predictor
from pi_jepa.train import encode_batch, parameter_groups, prediction_loss
from pi_jepa.training_runtime import configure_runtime, neural_autocast


class TinyVisualEncoder(torch.nn.Module):
    """Small visual network for portable CPU autocast/backward contract checks."""
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(24, 64)
        self.hidden = torch.nn.LayerNorm(64)
        self.output = torch.nn.Linear(64, 32)

    def forward(self, clips):
        return self.output(torch.nn.functional.gelu(self.hidden(self.input(clips.mean((2, 3))))))


@pytest.mark.parametrize("dataset", ["passive", "controlled"])
def test_joint_bf16_update_preserves_statistics_and_simulator_precision(dataset, monkeypatch):
    # This workstation's oneDNN supports BF16 forward but rejects BF16 backward
    # on AVX2. Use the portable ATen CPU path for this small precision-contract
    # test; it is not a measurement of the workstation or CUDA performance.
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    torch.manual_seed(711)
    device = torch.device("cpu")
    precision = configure_runtime({"training": {"precision": "bf16"}}, device)
    encoder = TinyVisualEncoder()
    predictor = PassivePredictor() if dataset == "passive" else Predictor()
    readout = PhysicalReadout()
    optimizer = torch.optim.AdamW(parameter_groups(encoder, predictor, readout, "joint", .05), lr=3e-4)
    table = FixedInitialConditions([[0., .2, .3, .4], [0., -.1, -.5, .2]], [0, 1])
    frames = torch.randint(0, 256, (2, 65, 8, 8, 3), dtype=torch.uint8)
    forces = torch.zeros(2, 64) if dataset == "passive" else torch.randn(2, 64) * .1
    theta = torch.tensor([[1., .25], [1., .25] if dataset == "passive" else [.9, .3]])
    batch = {"forces": forces, "theta": theta}
    encoder_dtypes, predictor_dtypes = [], []
    encoder.register_forward_hook(lambda module, inputs, output: encoder_dtypes.append(output.dtype))
    predictor.register_forward_hook(lambda module, inputs, output: predictor_dtypes.append(output.dtype))
    regularizer = SIGReg()
    before = encoder.input.weight.detach().clone()
    with neural_autocast(device, precision):
        codes = encode_batch(encoder, frames).float()
        jepa, predicted, regularization = prediction_loss(codes, predictor, batch, regularizer,
                                                        torch.Generator().manual_seed(19))
    decoded = readout(codes)
    targets = simulate_window(table(torch.arange(2)), theta, forces,
                              torch.tensor([[14, 24, 34, 44, 54, 64]]).expand(2, -1))
    physical = physics_loss(decoded, targets)
    assert encoder_dtypes == [torch.bfloat16] * 6
    assert predictor_dtypes and set(predictor_dtypes) == {torch.bfloat16}
    assert codes.dtype == decoded.dtype == torch.float32
    assert jepa.dtype == predicted.dtype == regularization.dtype == torch.float32
    assert targets.dtype == physical.dtype == torch.float64
    expected_regularizer = regularizer(codes.detach(), torch.Generator().manual_seed(19))
    torch.testing.assert_close(regularization.detach(), expected_regularizer.detach(), rtol=0, atol=0)
    total = jepa + physical
    total.backward()
    assert torch.isfinite(total)
    for network in (encoder, predictor, readout):
        parameters = list(network.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in parameters)
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum() for value in gradients) > 0
    assert not list(table.parameters()) and not table.states.requires_grad
    optimizer.step()
    assert not torch.equal(before, encoder.input.weight)
    assert all(torch.isfinite(parameter).all() for group in optimizer.param_groups for parameter in group["params"])


@torch.no_grad()
def test_real_encoder_bf16_forward_retains_float32_master_weights():
    encoder = Encoder()
    with neural_autocast(torch.device("cpu"), "bf16"):
        code = encoder(torch.randn(2, 24, 96, 96))
    assert code.shape == (2, 32) and code.dtype == torch.bfloat16
    assert torch.isfinite(code).all()
    assert all(parameter.dtype == torch.float32 for parameter in encoder.parameters())


def test_float32_runtime_does_not_enable_autocast_and_rejects_unsupported_precision():
    device = torch.device("cpu")
    layer = torch.nn.Linear(4, 3)
    inputs = torch.randn(2, 4)
    precision = configure_runtime({"training": {}}, device)
    with neural_autocast(device, precision):
        assert not torch.is_autocast_enabled("cpu")
        output = layer(inputs)
    assert output.dtype == torch.float32
    with pytest.raises(ValueError, match="float32 or bf16"):
        configure_runtime({"training": {"precision": "float16"}}, device)


@pytest.mark.parametrize("requirements", [{"expected_gpu": "H200"}, {"min_gpu_memory_gib": 100.}])
def test_cpu_benchmark_cannot_validate_gpu_allocation(monkeypatch, tmp_path, requirements):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_training.py"
    spec = importlib.util.spec_from_file_location("runtime_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "check_dataset_interface", lambda config: None)
    config = {"training": {"initial_conditions": "true_fixed", "cpu_threads": 2, "precision": "bf16"}}
    with pytest.raises(ValueError, match="GPU requirements cannot be checked on a CPU"):
        module.benchmark(config, "cpu", tmp_path, steps=1, warmup=0, **requirements)
