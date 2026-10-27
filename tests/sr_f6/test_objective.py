import math

import pytest
import torch

from sr_f6.objective import sequence_loss
from sr_f12.training import sequence_objective


@pytest.mark.parametrize("successes", range(9))
@pytest.mark.parametrize("length", (1, 7, 1536))
@pytest.mark.parametrize("outcome", (0, 1))
def test_zero_auxiliary_matches_actual_native_loss_and_gradient(successes, length, outcome):
    current = torch.linspace(-3, -0.2, length, dtype=torch.float32, requires_grad=True)
    sampler = current.detach() + torch.linspace(-0.003, 0.003, length)
    reference = current.detach() + torch.linspace(-0.1, 0.1, length)
    p = successes / 8
    advantage = (outcome - p) / math.sqrt(p * (1 - p)) if 0 < p < 1 else 0.0
    result = sequence_loss(current, sampler, reference, advantage, 0.0)
    observed = torch.autograd.grad(result["loss"], current)[0]
    expected = sequence_objective(current, sampler, reference, advantage)
    expected_gradient = torch.autograd.grad(expected["loss"], current)[0]
    torch.testing.assert_close(result["loss"], expected["loss"])
    torch.testing.assert_close(observed, expected_gradient)


def test_legal_g8_extreme_is_not_rejected_by_old_f5_quarter_bound():
    current = torch.full((8,), -0.5, dtype=torch.float32, requires_grad=True)
    result = sequence_loss(current, current.detach(), current.detach(), 0.0, 0.4375)
    gradient = torch.autograd.grad(result["loss"], current)[0]
    torch.testing.assert_close(gradient, torch.full_like(current, -0.4375 / 8))


def test_three_channels_sum_to_total_gradient_with_kl_once():
    current = torch.tensor([-1.0, -0.7, -0.4], requires_grad=True)
    sampler = current.detach() + 0.001
    reference = current.detach() + 0.15
    result = sequence_loss(current, sampler, reference, -1.0, 0.25)
    channels = [
        torch.autograd.grad(result[name], current, retain_graph=True)[0]
        for name in ("native", "aux", "kl")
    ]
    total = torch.autograd.grad(result["loss"], current)[0]
    torch.testing.assert_close(total, sum(channels))
    torch.testing.assert_close(result["kl"], 0.02 * result["kl_unweighted"])
    torch.testing.assert_close(channels[0], result["weights"] / 3)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_probabilities_rejected(bad):
    current = torch.tensor([-1.0, bad], requires_grad=True)
    with pytest.raises(FloatingPointError):
        sequence_loss(current, current.detach(), current.detach(), 0.0)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), 0.5))
def test_nonfinite_or_unregistered_auxiliary_rejected(bad):
    current = torch.tensor([-1.0], requires_grad=True)
    with pytest.raises(ValueError):
        sequence_loss(current, current.detach(), current.detach(), 0.0, bad)


def test_reference_overflow_is_not_silently_clamped():
    current = torch.tensor([-1000.0], requires_grad=True)
    with pytest.raises(FloatingPointError):
        sequence_loss(current, current.detach(), torch.tensor([-1.0]), 0.0)


def test_probability_dtype_detachment_and_alignment_are_enforced():
    current = torch.tensor([-1.0], requires_grad=True)
    with pytest.raises(ValueError, match="float32"):
        sequence_loss(current.double(), current.detach().double(), current.detach().double(), 0.0)
    with pytest.raises(ValueError, match="detached"):
        sequence_loss(current, current, current.detach(), 0.0)
    with pytest.raises(ValueError, match="align"):
        sequence_loss(current, torch.tensor([-1.0, -2.0]), current.detach(), 0.0)
    with pytest.raises(ValueError, match="detached scalar"):
        sequence_loss(
            current, current.detach(), current.detach(), torch.tensor(0.0, requires_grad=True)
        )
