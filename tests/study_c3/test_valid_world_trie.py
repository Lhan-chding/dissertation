from __future__ import annotations

import pytest

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
    assert trie.accepts(valid + [tokenizer.eos_token_id])
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

