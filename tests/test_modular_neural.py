"""Contracts for the diagnostic CNN-perceiver/MLP-reasoner model."""

from __future__ import annotations

import inspect

import pytest

from compbias.models.modular_neural import (
    ModularPerceiverReasoner,
    parameter_groups,
    set_training_mode,
)


def test_public_model_api_supports_controlled_state_injection() -> None:
    """The reasoner must accept a perceived state without receiving an image."""

    forward_parameters = inspect.signature(ModularPerceiverReasoner.forward).parameters

    assert "images" in forward_parameters
    assert "perceived_state" in forward_parameters


@pytest.mark.neural
def test_parameter_groups_are_disjoint_complete_and_architecturally_distinct() -> None:
    torch = pytest.importorskip("torch")
    model = ModularPerceiverReasoner(
        image_channels=1,
        image_size=16,
        num_perceived_states=2,
        num_reasoning_actions=2,
        hidden_dim=8,
    )

    groups = parameter_groups(model)

    assert set(groups) == {"perception", "reasoning"}
    assert all(isinstance(parameters, tuple) for parameters in groups.values())
    perception_ids = {id(parameter) for parameter in groups["perception"]}
    reasoning_ids = {id(parameter) for parameter in groups["reasoning"]}
    assert perception_ids
    assert reasoning_ids
    assert perception_ids.isdisjoint(reasoning_ids)
    assert perception_ids | reasoning_ids == {id(parameter) for parameter in model.parameters()}
    assert any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    assert any(isinstance(module, torch.nn.Linear) for module in model.modules())


@pytest.mark.neural
@pytest.mark.parametrize(
    ("mode", "perception_trainable", "reasoning_trainable"),
    [
        ("perception_only", True, False),
        ("reasoning_only", False, True),
        ("joint", True, True),
    ],
)
def test_training_mode_freezes_exactly_the_requested_parameter_block(
    mode: str,
    perception_trainable: bool,
    reasoning_trainable: bool,
) -> None:
    pytest.importorskip("torch")
    model = ModularPerceiverReasoner(
        image_channels=1,
        image_size=16,
        num_perceived_states=2,
        num_reasoning_actions=2,
        hidden_dim=8,
    )

    returned = set_training_mode(model, mode)
    groups = parameter_groups(model)

    assert returned is model
    assert {parameter.requires_grad for parameter in groups["perception"]} == {perception_trainable}
    assert {parameter.requires_grad for parameter in groups["reasoning"]} == {reasoning_trainable}


@pytest.mark.neural
def test_injected_state_prevents_reasoner_from_rereading_the_image() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = ModularPerceiverReasoner(
        image_channels=1,
        image_size=16,
        num_perceived_states=2,
        num_reasoning_actions=2,
        hidden_dim=8,
    ).eval()
    injected_state = torch.tensor([0, 1], dtype=torch.long)
    image_a = torch.zeros((2, 1, 16, 16))
    image_b = torch.ones((2, 1, 16, 16))

    with torch.no_grad():
        output_a = model(image_a, perceived_state=injected_state)
        output_b = model(image_b, perceived_state=injected_state)

    assert set(output_a) >= {"perception_logits", "reasoning_logits"}
    assert output_a["perception_logits"].shape == (2, 2)
    assert output_a["reasoning_logits"].shape == (2, 2)
    torch.testing.assert_close(output_a["reasoning_logits"], output_b["reasoning_logits"])


def test_unknown_training_mode_is_rejected() -> None:
    pytest.importorskip("torch")
    model = ModularPerceiverReasoner(
        image_channels=1,
        image_size=16,
        num_perceived_states=2,
        num_reasoning_actions=2,
        hidden_dim=8,
    )

    with pytest.raises(ValueError, match="mode"):
        set_training_mode(model, "everything")
