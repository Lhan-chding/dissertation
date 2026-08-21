"""Server-side offline gate contracts for v5 training entrypoints."""

from __future__ import annotations

from compensability_v5.training.train_support_lora import _offline_config_enabled


def test_offline_config_accepts_bool_string_and_integer_truthy_flags() -> None:
    assert _offline_config_enabled({"offline": True}) is True
    assert (
        _offline_config_enabled({"offline": {"HF_HUB_OFFLINE": True, "TRANSFORMERS_OFFLINE": "1"}})
        is True
    )
    assert (
        _offline_config_enabled({"offline": {"HF_HUB_OFFLINE": 1, "TRANSFORMERS_OFFLINE": 1}})
        is True
    )


def test_offline_config_rejects_incomplete_or_disabled_flags() -> None:
    assert _offline_config_enabled({"offline": {"HF_HUB_OFFLINE": "1"}}) is False
    assert _offline_config_enabled({"offline_only": False}) is False
