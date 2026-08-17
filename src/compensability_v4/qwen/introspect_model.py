"""Runtime introspection for the frozen Qwen model.

No layer or module count here is a Qwen constant: all values are read from the
loaded model and its named-module tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _config_value(config: object, name: str) -> object | None:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def language_layer_count(config: object) -> int:
    """Resolve the decoder depth from flat or composite Qwen runtime config."""

    direct = _config_value(config, "num_hidden_layers")
    text_config = _config_value(config, "text_config")
    nested = None if text_config is None else _config_value(text_config, "num_hidden_layers")
    candidates = tuple(value for value in (direct, nested) if value is not None)
    invalid = any(isinstance(value, bool) or not isinstance(value, int) for value in candidates)
    if not candidates or invalid:
        raise RuntimeError("runtime config has no valid num_hidden_layers")
    if any(value <= 0 for value in candidates):
        raise RuntimeError("runtime language layer count must be positive")
    if len(set(candidates)) != 1:
        raise RuntimeError("runtime config has conflicting language layer counts")
    return candidates[0]


def _config_mapping(config: object) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        value = config.to_dict()  # type: ignore[attr-defined]
    elif isinstance(config, Mapping):
        value = dict(config)
    elif hasattr(config, "__dict__"):
        value = vars(config)
    else:
        raise RuntimeError("model config is not introspectable")
    if not isinstance(value, Mapping):
        raise RuntimeError("model config did not produce a mapping")
    # The JSON round-trip prevents live config objects leaking into evidence.
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str, allow_nan=False))


def introspect_model(model: object) -> dict[str, object]:
    """Return a detached architecture manifest derived from the live model."""

    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("loaded model has no runtime config")
    language_layers = language_layer_count(config)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise RuntimeError("runtime config has no vision_config")
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise RuntimeError("loaded model does not expose named_modules()")
    module_names = tuple(name for name, _ in named_modules())
    if not module_names or len(module_names) != len(set(module_names)):
        raise RuntimeError("runtime named-module manifest is empty or non-unique")
    return {
        "model_class": type(model).__qualname__,
        "language_layers": language_layers,
        "vision_config": _config_mapping(vision_config),
        "model_config": _config_mapping(config),
        "module_names": module_names,
        "module_count": len(module_names),
    }


def write_model_introspection(model: object, artifact_dir: Path) -> dict[str, object]:
    """Write the two phase-1 architecture evidence artifacts."""

    result = introspect_model(model)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model_introspection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    module_names = result["module_names"]
    if not isinstance(module_names, tuple):
        raise AssertionError("introspection invariant violated")
    (artifact_dir / "module_manifest.txt").write_text(
        "\n".join(module_names) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["introspect_model", "language_layer_count", "write_model_introspection"]
