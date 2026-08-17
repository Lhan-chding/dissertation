"""Runtime-only model introspection helpers."""

from __future__ import annotations


def introspect_model(model) -> dict[str, object]:
    return {
        "language_layers": model.config.num_hidden_layers,
        "vision_config": vars(model.config.vision_config),
        "module_names": tuple(name for name, _module in model.named_modules()),
    }
