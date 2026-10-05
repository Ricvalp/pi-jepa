"""Long qualitative forecasts stay causal, aligned, and outside optimization."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from pi_jepa.experiment_logging import preserve_rng
from pi_jepa.physics import rollout
from pi_jepa.trajectory_diagnostics import fixed_trajectory_reference, trajectory_phase_figure


class Episodes:
    """Learning-only fixture with deliberately ragged episodes and no truth files."""
    def __init__(self, modes=(2, 2, 0, 1), lengths=(65, 401, 401, 75), dataset="controlled"):
        self.reset_modes, self.dataset = list(modes), dataset
        self.accesses, self.items = [], []
        for index, (mode, length) in enumerate(zip(modes, lengths)):
            rgb = (torch.arange(length) % 256).to(torch.uint8)[:, None, None, None].expand(-1, 2, 3, 3).clone()
            item = {"trajectory_id": index, "reset_mode": mode, "frames": rgb,
                    "forces": torch.arange(length - 1).float() * .01}
            if dataset == "controlled":
                item["theta"] = torch.tensor([1. + index * .1, .25])
            else:
                item["forces"].zero_()
            self.items.append(item)

    def __getitem__(self, index):
        self.accesses.append(index)
        return self.items[index]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2))
        self.dropout = nn.Dropout(.5)

    def forward(self, clips):
        value = self.dropout(clips[:, -3:].mean((1, 2, 3)) * self.scale)
        return value[:, None].expand(-1, 32)


class ControlledPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.01))
        self.dropout = nn.Dropout(.5)
        self.calls = []

    def forward(self, codes, actions, theta):
        self.calls.append((codes.detach().clone(), actions.detach().clone(), theta.detach().clone()))
        return self.dropout(codes + self.gain * actions.mean(-1, keepdim=True))


class AutonomousPredictor(nn.Module):
    action_free = True

    def __init__(self):
        super().__init__()
        self.increment = nn.Parameter(torch.tensor(.02))
        self.calls = []

    def forward(self, code):
        self.calls.append(code.detach().clone())
        return code + self.increment


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, codes):
        p = codes[..., 0] * self.scale
        q, zero = p * .2, torch.zeros_like(p)
        return torch.stack((p, zero, q.sin(), q.cos(), zero), dim=-1)


class Resets(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.states = nn.Parameter(torch.tensor([[0., .1, .15, .05]]).repeat(count, 1))

    def forward(self, ids):
        return self.states[ids]


CONFIG = {"training": {"initial_conditions": "learned", "cache_batch_size": 8}}


def labelled_curve(figure, label, column=0):
    return next(line for line in figure.axes[column].lines if line.get_label() == label)


def test_references_use_first_episode_per_mode_without_duration_selection_or_truth(monkeypatch):
    def no_truth(*args, **kwargs):
        raise AssertionError("No external truth file may be opened")
    monkeypatch.setattr(np, "load", no_truth)
    episodes = Episodes()
    reference = fixed_trajectory_reference(episodes)
    assert episodes.accesses == [2, 3, 0]
    assert [r["trajectory_id"] for r in reference] == [2, 3, 0]
    assert [r["reset_mode"] for r in reference] == [0, 1, 2]
    assert [r["steps"] for r in reference] == [32, 4, 3]
    assert [len(r["frames"]) for r in reference] == [355, 75, 65]
    assert [len(r["forces"]) for r in reference] == [354, 74, 64]
    assert reference[0]["frames"].dtype == torch.uint8
    assert reference[0]["endpoints"].tolist() == list(range(14, 355, 10))
    assert reference[2]["available_frames"] == 65  # Does not choose the later401-frame upright episode.
    episodes.items[2]["frames"].zero_()
    episodes.items[2]["theta"].zero_()
    assert reference[0]["frames"].sum() > 0
    torch.testing.assert_close(reference[0]["theta"], torch.tensor([1.2, .25]))


def test_missing_families_are_omitted_and_passive_parameters_are_nominal():
    reference = fixed_trajectory_reference(Episodes((1, 1), (401, 75), "passive"))
    assert len(reference) == 1 and reference[0]["trajectory_id"] == 0
    torch.testing.assert_close(reference[0]["theta"], torch.tensor([1., .25]))


@pytest.mark.parametrize("passive", [False, True])
def test_forecast_does_not_use_future_images_but_observed_readout_does(passive):
    data = Episodes((0,), (65,), "passive" if passive else "controlled")
    first = fixed_trajectory_reference(data)
    second = copy.deepcopy(first)
    second[0]["frames"][35:] = 255 - second[0]["frames"][35:]
    encoder, readout, table = TinyEncoder(), Readout(), Resets(1)
    predictor = AutonomousPredictor() if passive else ControlledPredictor()
    before = trajectory_phase_figure(encoder, predictor, readout, table, first, CONFIG, "cpu")
    after = trajectory_phase_figure(encoder, predictor, readout, table, second, CONFIG, "cpu")
    try:
        for coordinate in ("get_xdata", "get_ydata"):
            actual = getattr(labelled_curve(before, "r(P): autoregressive forecast"), coordinate)()
            changed = getattr(labelled_curve(after, "r(P): autoregressive forecast"), coordinate)()
            np.testing.assert_array_equal(actual, changed)
        old_observed = labelled_curve(before, "r(E): observed images").get_xdata()
        new_observed = labelled_curve(after, "r(E): observed images").get_xdata()
        assert old_observed[0] == new_observed[0]  # Both start from exactly frame34.
        assert not np.array_equal(old_observed[1:], new_observed[1:])
    finally:
        plt.close(before)
        plt.close(after)


def test_controlled_context_force_blocks_and_dense_simulator_prefix_are_aligned():
    reference = fixed_trajectory_reference(Episodes((0,), (65,)))
    row = reference[0]
    encoder, predictor, readout, table = TinyEncoder(), ControlledPredictor(), Readout(), Resets(1)
    figure = trajectory_phase_figure(encoder, predictor, readout, table, reference, CONFIG, "cpu")
    try:
        assert len(predictor.calls) == 3
        for step, (codes, actions, theta) in enumerate(predictor.calls):
            assert codes.shape == (1, 3, 32)
            torch.testing.assert_close(actions[0], row["forces"][14 + step*10:44 + step*10].reshape(3, 10))
            torch.testing.assert_close(theta[0], row["theta"])
        target = rollout(table(torch.tensor([0])), row["theta"][None], row["forces"][None]).detach().numpy()[0, 34:]
        curve = labelled_curve(figure, "Learned-reset simulation")
        np.testing.assert_allclose(curve.get_xdata(), target[:, 0])
        np.testing.assert_allclose(curve.get_ydata(), (target[:, 2] + np.pi) % (2*np.pi) - np.pi)
        assert len(curve.get_xdata()) == 31  # Every10ms, frame34 through64 inclusive.
        predicted = labelled_curve(figure, "r(P): autoregressive forecast").get_xdata()
        observed = labelled_curve(figure, "r(E): observed images").get_xdata()
        assert len(predicted) == len(observed) == 4 and predicted[0] == observed[0]
        assert "0.34–0.64 s" in figure.axes[0].get_title()
        assert "truncated episode" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


def test_three_long_examples_show32_future_steps_and_fixed_reset_labels():
    reference = fixed_trajectory_reference(Episodes((0, 1, 2), (401, 401, 401), "passive"))
    predictor = AutonomousPredictor()
    config = {"training": {"initial_conditions": "true_fixed", "cache_batch_size": 8}}
    figure = trajectory_phase_figure(TinyEncoder(), predictor, Readout(), Resets(3), reference, config, "cpu")
    try:
        assert len(figure.axes) == 3 and len(predictor.calls) == 96
        for column, axis in enumerate(figure.axes):
            assert "0.34–3.54 s" in axis.get_title()
            assert len(labelled_curve(figure, "r(P): autoregressive forecast", column).get_xdata()) == 33
            assert len(labelled_curve(figure, "r(E): observed images", column).get_xdata()) == 33
            assert len(labelled_curve(figure, "Fixed true-reset simulation", column).get_xdata()) == 321
        assert "Fixed true-reset diagnostic" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_diagnostic_preserves_mixed_modes_weights_gradients_references_and_rng():
    reference = fixed_trajectory_reference(Episodes((0,), (65,)))
    original_reference = copy.deepcopy(reference)
    networks = (TinyEncoder().train(), ControlledPredictor().train(), Readout().eval(), Resets(1).train())
    networks[0].dropout.eval()  # Preserve the child's distinct mode too.
    modes = [module.training for network in networks for module in network.modules()]
    states = [copy.deepcopy(network.state_dict()) for network in networks]
    for network in networks:
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter)
    torch_before, numpy_before = torch.get_rng_state(), np.random.get_state()
    with preserve_rng():
        figure = trajectory_phase_figure(*networks, reference, CONFIG, "cpu")
    plt.close(figure)
    assert modes == [module.training for network in networks for module in network.modules()]
    assert torch.equal(torch_before, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
    for network, state in zip(networks, states):
        assert all(torch.equal(value, state[name]) for name, value in network.state_dict().items())
        assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in network.parameters())
    for key, value in reference[0].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, original_reference[0][key])
        else:
            assert value == original_reference[0][key]


def test_failures_restore_modes_and_close_partial_figure(monkeypatch):
    reference = fixed_trajectory_reference(Episodes((0,), (65,)))
    encoder, predictor, readout, table = TinyEncoder().train(), ControlledPredictor().eval(), Readout(), Resets(1)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic encoder failure")
    monkeypatch.setattr(encoder, "forward", fail)
    before = plt.get_fignums()
    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        trajectory_phase_figure(encoder, predictor, readout, table, reference, CONFIG, "cpu")
    assert encoder.training and not predictor.training and readout.training and table.training
    assert plt.get_fignums() == before
