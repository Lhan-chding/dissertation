from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from compensability_v5.qwen import study_b_token_counter

ROOT = Path(__file__).resolve().parents[2]


def _load_support_freeze_cli():
    path = ROOT / "scripts/v5/05_build_budget_matched_support.py"
    specification = importlib.util.spec_from_file_location(
        "test_v5_build_budget_matched_support_cli", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_support_freeze_cli_can_import_registered_qwen_token_counter() -> None:
    cli = _load_support_freeze_cli()

    counter = cli._load_token_counter(
        "compensability_v5.qwen.study_b_token_counter:count_completion_tokens"
    )

    assert counter is study_b_token_counter.count_completion_tokens


def test_count_completion_tokens_uses_exact_tokenizer_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    class _Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool):
            calls.append((text, add_special_tokens))
            return [11, 22, 33, 44]

    monkeypatch.setattr(study_b_token_counter, "_load_tokenizer", lambda: _Tokenizer())
    study_b_token_counter._get_tokenizer.cache_clear()
    try:
        assert study_b_token_counter.count_completion_tokens("9,2,3,4") == 4
    finally:
        study_b_token_counter._get_tokenizer.cache_clear()

    assert calls == [("9,2,3,4", False)]


def test_count_completion_tokens_rejects_empty_text_and_malformed_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def encode(self, text: str, *, add_special_tokens: bool):
            del text, add_special_tokens
            return self.payload

    with pytest.raises(ValueError, match="non-empty"):
        study_b_token_counter.count_completion_tokens("")

    monkeypatch.setattr(study_b_token_counter, "_load_tokenizer", lambda: _Tokenizer([1, True]))
    study_b_token_counter._get_tokenizer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="malformed token ids"):
            study_b_token_counter.count_completion_tokens("9,2,3,4")
    finally:
        study_b_token_counter._get_tokenizer.cache_clear()
