from __future__ import annotations

from pathlib import Path

import pytest

from compensability.study_c3.qwen_backend import adapter_artifact_sha256


def test_adapter_hash_ignores_optimizer_state_and_binds_model_and_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-48"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"rank": 8}\n', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"first")
    first = adapter_artifact_sha256(checkpoint)

    (checkpoint / "optimizer.pt").write_bytes(b"second")
    assert adapter_artifact_sha256(checkpoint) == first

    (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
    assert adapter_artifact_sha256(checkpoint) != first


def test_adapter_hash_fails_closed_on_missing_or_ambiguous_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        adapter_artifact_sha256(checkpoint)
    (checkpoint / "adapter_model.bin").write_bytes(b"one")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"two")
    with pytest.raises(ValueError, match="exactly one"):
        adapter_artifact_sha256(checkpoint)
