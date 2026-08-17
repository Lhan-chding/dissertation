"""Immutable cached-continuation provenance and parity execution gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _immutable_mapping(value: Mapping[str, Any]) -> MappingProxyType:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("generation-config keys must be strings")
        if isinstance(item, Mapping):
            frozen[key] = _immutable_mapping(item)
        elif isinstance(item, list | tuple):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class CachedGenerationState:
    """One sample's fully consumed token/cache state plus audit provenance."""

    sample_id: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[tuple[int, ...], ...]
    image_token_positions: tuple[int, ...]
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int
    generation_config: Mapping[str, Any]
    rng_seed: int
    past_key_values: object
    chat_messages: tuple[Mapping[str, Any], ...] = ()
    generated_token_ids: tuple[int, ...] = ()
    processor: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("cached sample_id must be non-empty")
        if not self.token_ids or any(isinstance(item, bool) or item < 0 for item in self.token_ids):
            raise ValueError("cached token ids must be non-empty non-negative integers")
        if len(self.attention_mask) != len(self.token_ids):
            raise ValueError("cached attention mask must align with token ids")
        if any(item not in (0, 1) for item in self.attention_mask):
            raise ValueError("cached attention mask must be binary")
        if not self.position_ids or any(
            len(axis) != len(self.token_ids) for axis in self.position_ids
        ):
            raise ValueError("cached position ids must align with token ids on every axis")
        if any(
            position < 0 or position >= len(self.token_ids)
            for position in self.image_token_positions
        ):
            raise ValueError("cached image-token positions are outside the token sequence")
        if isinstance(self.visual_token_count, bool) or self.visual_token_count < 0:
            raise ValueError("visual token count must be a non-negative integer")
        if self.visual_token_count and not self.image_token_positions:
            raise ValueError("non-zero visual token count requires saved image-token positions")
        if len(self.image_grid_thw) != 3 or any(value <= 0 for value in self.image_grid_thw):
            raise ValueError("image_grid_thw must contain three positive integers")
        if isinstance(self.rng_seed, bool) or not isinstance(self.rng_seed, int):
            raise TypeError("rng_seed must be an integer")
        if self.past_key_values is None:
            raise ValueError("cached state must retain past_key_values")
        if any(token_id not in self.token_ids for token_id in self.generated_token_ids):
            # This inexpensive check catches obvious provenance corruption. The
            # ordered suffix relation is checked by the producer.
            raise ValueError("generated token provenance is inconsistent with cached tokens")
        object.__setattr__(self, "generation_config", _immutable_mapping(self.generation_config))
        object.__setattr__(
            self,
            "chat_messages",
            tuple(_immutable_mapping(message) for message in self.chat_messages),
        )


def build_suffix_token_ids(
    prefix_token_ids: Sequence[int], full_token_ids: Sequence[int]
) -> tuple[int, ...]:
    prefix = tuple(int(token_id) for token_id in prefix_token_ids)
    full = tuple(int(token_id) for token_id in full_token_ids)
    if full[: len(prefix)] != prefix:
        raise ValueError("full token sequence must preserve the cached prefix")
    suffix = full[len(prefix) :]
    if not suffix:
        raise ValueError("full token sequence must add a non-empty suffix")
    return suffix


def assert_cache_parity(
    cache_output: Sequence[int],
    full_output: Sequence[int],
    *,
    sample_id: str | None = None,
) -> None:
    cached = tuple(int(token_id) for token_id in cache_output)
    reencoded = tuple(int(token_id) for token_id in full_output)
    if cached != reencoded:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(cached, reencoded, strict=False))
                if pair[0] != pair[1]
            ),
            min(len(cached), len(reencoded)),
        )
        label = "" if sample_id is None else f" for sample {sample_id!r}"
        raise RuntimeError(
            f"cache continuation parity check failed{label} at generated token {mismatch}"
        )


@dataclass(frozen=True, slots=True)
class ParityGate:
    """Proof object required before I4 outputs may enter primary artifacts."""

    sample_id: str
    cached_token_ids: tuple[int, ...]
    full_reencode_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        assert_cache_parity(
            self.cached_token_ids,
            self.full_reencode_token_ids,
            sample_id=self.sample_id,
        )


def require_state_sample(state: CachedGenerationState, sample_id: str) -> None:
    """Prevent accidental cache reuse across scenes."""

    if state.sample_id != sample_id:
        raise RuntimeError(
            f"cached state belongs to sample {state.sample_id!r}, not requested {sample_id!r}"
        )


def _token_ids(value: object) -> tuple[int, ...]:
    to_list = getattr(value, "tolist", None)
    converted = to_list() if callable(to_list) else value
    if isinstance(converted, Sequence) and converted and isinstance(converted[0], Sequence):
        if len(converted) != 1:
            raise RuntimeError("cache continuation supports exactly one sample")
        converted = converted[0]
    if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes)):
        raise RuntimeError("chat template did not produce token ids")
    return tuple(int(token_id) for token_id in converted)


def append_turn_and_continue(
    model: object,
    cached_state: CachedGenerationState,
    new_user_text: str,
    *,
    sample_id: str | None = None,
    processor: object | None = None,
    max_new_tokens: int = 64,
    parity_reference_token_ids: Sequence[int] | None = None,
) -> dict[str, object]:
    """Append an exact chat-template suffix and greedily continue from cache.

    A parity reference is optional here so the function can create diagnostic
    evidence.  ``InterfaceRunner`` requires a successful parity proof before an
    I4 result may be marked primary.
    """

    if not isinstance(new_user_text, str) or not new_user_text.strip():
        raise ValueError("new cache-continuation user turn must be non-empty")
    expected_sample = cached_state.sample_id if sample_id is None else sample_id
    require_state_sample(cached_state, expected_sample)
    runtime_processor = processor or cached_state.processor
    if runtime_processor is None:
        raise RuntimeError("cache continuation requires the original processor")
    if not cached_state.chat_messages:
        raise RuntimeError("cached state has no exact chat-message provenance")
    user_message = {"role": "user", "content": new_user_text}
    messages = [*cached_state.chat_messages, user_message]
    full_ids = _token_ids(
        runtime_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    try:
        suffix_ids = build_suffix_token_ids(cached_state.token_ids, full_ids)
    except ValueError as error:
        raise RuntimeError(
            "assistant decode/re-encode did not preserve the cached prefix; "
            "I4 must remain diagnostic for this sample"
        ) from error

    import torch

    device = getattr(model, "device", None)
    input_ids = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [cached_state.attention_mask + (1,) * len(suffix_ids)],
        dtype=torch.long,
        device=device,
    )
    from .manual_generation import manual_greedy_generate

    tokenizer = getattr(runtime_processor, "tokenizer", runtime_processor)
    result = manual_greedy_generate(
        model,
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_new_tokens,
        eos_token_ids=getattr(tokenizer, "eos_token_id", None),
        rng_seed=cached_state.rng_seed,
        generation_config=cached_state.generation_config,
        past_key_values=cached_state.past_key_values,
        prior_token_ids=cached_state.token_ids,
        prior_position_ids=cached_state.position_ids,
    )
    decoded = tokenizer.decode(result.generated_token_ids, skip_special_tokens=True)
    parity_verified = parity_reference_token_ids is not None
    if parity_reference_token_ids is not None:
        assert_cache_parity(
            result.generated_token_ids,
            parity_reference_token_ids,
            sample_id=cached_state.sample_id,
        )
    continued_messages = (*messages, {"role": "assistant", "content": decoded})
    continued_state = CachedGenerationState(
        sample_id=cached_state.sample_id,
        token_ids=result.all_token_ids,
        attention_mask=result.attention_mask,
        position_ids=result.position_ids,
        image_token_positions=cached_state.image_token_positions,
        image_grid_thw=cached_state.image_grid_thw,
        visual_token_count=cached_state.visual_token_count,
        generation_config=result.generation_config,
        rng_seed=result.rng_seed,
        past_key_values=result.past_key_values,
        chat_messages=continued_messages,
        generated_token_ids=result.generated_token_ids,
        processor=runtime_processor,
    )
    return {
        "text": decoded,
        "generated_token_ids": result.generated_token_ids,
        "suffix_token_ids": suffix_ids,
        "state": continued_state,
        "parity_verified": parity_verified,
    }


__all__ = [
    "CachedGenerationState",
    "ParityGate",
    "append_turn_and_continue",
    "assert_cache_parity",
    "build_suffix_token_ids",
    "require_state_sample",
]
