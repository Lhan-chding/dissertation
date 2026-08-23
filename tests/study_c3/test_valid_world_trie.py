from __future__ import annotations

import json
from pathlib import Path

import pytest

from compensability.study_c3 import qwen_backend
from compensability.study_c3.io import sha256_file
from compensability.study_c3.valid_world_trie import ValidWorldTrie


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


def test_trie_is_complete_and_does_not_assume_numbers_are_single_tokens() -> None:
    tokenizer = CharacterTokenizer()
    trie = ValidWorldTrie.build(tokenizer, minimum=2, maximum=18)

    assert trie.world_count == 17**4 == 83_521
    assert len(tokenizer.encode("10", add_special_tokens=False)) == 2
    for action in trie.iter_actions():
        assert trie.accepts(tokenizer.encode(action, add_special_tokens=False))


def test_trie_blocks_invalid_actions_and_allows_only_eos_after_completion() -> None:
    tokenizer = CharacterTokenizer()
    trie = ValidWorldTrie.build(tokenizer, minimum=2, maximum=18)
    valid = tokenizer.encode("2,3,4,5", add_special_tokens=False)

    assert trie.allowed_next(valid) == frozenset({tokenizer.eos_token_id})
    assert trie.accepts([*valid, tokenizer.eos_token_id])
    assert not trie.accepts(tokenizer.encode("1,3,4,5", add_special_tokens=False))
    assert not trie.accepts(tokenizer.encode("2,3,4,19", add_special_tokens=False))
    assert not trie.accepts(tokenizer.encode("[2,3,4,5]", add_special_tokens=False))
    assert not trie.accepts(valid + tokenizer.encode(" prose", add_special_tokens=False))


def test_mask_records_before_and_after_state() -> None:
    tokenizer = CharacterTokenizer()
    trie = ValidWorldTrie.build(tokenizer, minimum=2, maximum=18)
    prefix = tokenizer.encode("2,3,4,", add_special_tokens=False)
    logits = [float(index) for index in range(256)]

    masked, record = trie.mask_logits(prefix, logits)

    assert record["prefix_token_ids"] == prefix
    assert record["allowed_token_ids"] == sorted(trie.allowed_next(prefix))
    assert record["pre_mask_argmax_token_id"] == 255
    assert record["post_mask_argmax_token_id"] in trie.allowed_next(prefix)
    assert sum(value != float("-inf") for value in masked) == len(trie.allowed_next(prefix))


def test_invalid_prefix_fails_closed() -> None:
    trie = ValidWorldTrie.build(CharacterTokenizer(), minimum=2, maximum=18)
    with pytest.raises(ValueError, match="invalid constrained-decoding prefix"):
        trie.allowed_next([999_999])


def test_frozen_trie_loader_binds_build_and_validation_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trie = ValidWorldTrie.build(CharacterTokenizer(), minimum=2, maximum=18)
    payload = tmp_path / "trie.json"
    build_manifest = tmp_path / "manifest.json"
    validation_manifest = tmp_path / "validation_manifest.json"
    payload.write_text(json.dumps(trie.to_payload()), encoding="utf-8")
    build_manifest.write_text(
        json.dumps(
            {
                "status": "STUDY_C3_VALID_WORLD_TRIE_COMPLETE",
                "trie_sha256": sha256_file(payload),
                "world_count": 83_521,
            }
        ),
        encoding="utf-8",
    )
    validation_manifest.write_text(
        json.dumps(
            {
                "status": "STUDY_C3_VALID_WORLD_TRIE_VALIDATION_COMPLETE",
                "trie_sha256": sha256_file(payload),
                "legal_action_count": 83_521,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(qwen_backend, "TRIE_PAYLOAD", payload, raising=False)
    monkeypatch.setattr(qwen_backend, "TRIE_MANIFEST", build_manifest, raising=False)
    monkeypatch.setattr(
        qwen_backend, "TRIE_VALIDATION_MANIFEST", validation_manifest, raising=False
    )

    loaded = qwen_backend.load_frozen_valid_world_trie()
    assert loaded.world_count == 83_521

    payload.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        qwen_backend.load_frozen_valid_world_trie()
