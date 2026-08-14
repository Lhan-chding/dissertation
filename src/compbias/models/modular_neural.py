"""Small, explicitly modular neural model used by the diagnostic experiments.

The architecture is intentionally unremarkable: a CNN produces a categorical
perceived state and an MLP maps that state to a reasoning action.  Keeping the
two blocks explicit lets experiments freeze either side and, importantly, lets
the reasoner consume an externally injected state without consulting an image.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised in base-only installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


TrainingMode = Literal["perception_only", "reasoning_only", "joint"]
_VALID_TRAINING_MODES = frozenset({"perception_only", "reasoning_only", "joint"})
_ModuleBase = nn.Module if nn is not None else object


class ModularPerceiverReasoner(_ModuleBase):
    """A CNN perceiver followed by an image-blind MLP reasoner.

    ``perceived_state`` may be either integer state indices or a matrix of
    state probabilities.  When it is supplied, the reasoner is fed only that
    tensor; changes to ``images`` therefore cannot alter its output.
    """

    def __init__(
        self,
        *,
        image_channels: int,
        image_size: int,
        num_perceived_states: int,
        num_reasoning_actions: int,
        hidden_dim: int,
    ) -> None:
        if nn is None or torch is None:
            raise ModuleNotFoundError(
                "ModularPerceiverReasoner requires the optional 'torch' dependency; "
                "install compbias[neural]"
            )
        super().__init__()
        dimensions = {
            "image_channels": image_channels,
            "image_size": image_size,
            "num_perceived_states": num_perceived_states,
            "num_reasoning_actions": num_reasoning_actions,
            "hidden_dim": hidden_dim,
        }
        invalid = [
            name for name, value in dimensions.items() if not isinstance(value, int) or value < 1
        ]
        if invalid:
            raise ValueError(f"model dimensions must be positive integers: {', '.join(invalid)}")

        self.image_channels = image_channels
        self.image_size = image_size
        self.num_perceived_states = num_perceived_states
        self.num_reasoning_actions = num_reasoning_actions
        self.perception = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_perceived_states),
        )
        self.reasoning = nn.Sequential(
            nn.Linear(num_perceived_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_reasoning_actions),
        )

    def _injected_state_features(self, perceived_state: Any) -> Any:
        if perceived_state.ndim == 0:
            perceived_state = perceived_state.unsqueeze(0)
        if perceived_state.ndim == 2:
            if perceived_state.shape[-1] != self.num_perceived_states:
                raise ValueError(
                    "a two-dimensional perceived_state must have "
                    f"{self.num_perceived_states} columns"
                )
            return perceived_state.to(dtype=next(self.parameters()).dtype)
        if perceived_state.ndim != 1:
            raise ValueError("perceived_state must contain indices or state probabilities")
        indices = perceived_state.to(dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= self.num_perceived_states):
            raise ValueError("perceived_state contains an out-of-range state index")
        return torch.nn.functional.one_hot(indices, num_classes=self.num_perceived_states).to(
            dtype=next(self.parameters()).dtype
        )

    def forward(
        self,
        images: Any | None = None,
        perceived_state: Any | None = None,
    ) -> Mapping[str, Any]:
        """Return perception and reasoning logits for natural or injected states."""

        if images is None and perceived_state is None:
            raise ValueError("images or perceived_state must be provided")

        perception_logits = self.perception(images) if images is not None else None
        if perceived_state is None:
            state_features = torch.softmax(perception_logits, dim=-1)
        else:
            state_features = self._injected_state_features(perceived_state)
            if (
                perception_logits is not None
                and perception_logits.shape[0] != state_features.shape[0]
            ):
                raise ValueError("images and perceived_state must have the same batch size")
            if perception_logits is None:
                perception_logits = torch.zeros(
                    (state_features.shape[0], self.num_perceived_states),
                    dtype=state_features.dtype,
                    device=state_features.device,
                )

        reasoning_logits = self.reasoning(state_features)
        return {
            "perception_logits": perception_logits,
            "reasoning_logits": reasoning_logits,
        }


def parameter_groups(model: ModularPerceiverReasoner) -> dict[str, tuple[Any, ...]]:
    """Return complete, non-overlapping perception and reasoning parameters."""

    if not isinstance(model, ModularPerceiverReasoner):
        raise TypeError("model must be a ModularPerceiverReasoner")
    return {
        "perception": tuple(model.perception.parameters()),
        "reasoning": tuple(model.reasoning.parameters()),
    }


def set_training_mode(
    model: ModularPerceiverReasoner,
    mode: TrainingMode | str,
) -> ModularPerceiverReasoner:
    """Freeze exactly the parameter block excluded by ``mode``."""

    if mode not in _VALID_TRAINING_MODES:
        choices = ", ".join(sorted(_VALID_TRAINING_MODES))
        raise ValueError(f"unknown training mode {mode!r}; expected one of: {choices}")
    groups = parameter_groups(model)
    train_perception = mode in {"perception_only", "joint"}
    train_reasoning = mode in {"reasoning_only", "joint"}
    for parameter in groups["perception"]:
        parameter.requires_grad_(train_perception)
    for parameter in groups["reasoning"]:
        parameter.requires_grad_(train_reasoning)
    return model


__all__ = [
    "ModularPerceiverReasoner",
    "TrainingMode",
    "parameter_groups",
    "set_training_mode",
]
