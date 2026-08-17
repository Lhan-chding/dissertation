"""Cache-continuation provenance and parity gates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class CachedGenerationState:
    sample_id: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[tuple[int, ...], ...]
    image_token_positions: tuple[int, ...]
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int
    generation_config: MappingProxyType
    rng_seed: int
    past_key_values: object


def build_suffix_token_ids(prefix_token_ids: list[int], full_token_ids: list[int]) -> tuple[int, ...]:
    if full_token_ids[: len(prefix_token_ids)] != prefix_token_ids:
        raise ValueError("full token sequence must preserve the cached prefix")
    return tuple(full_token_ids[len(prefix_token_ids) :])


def assert_cache_parity(cache_output, full_output) -> None:
    if tuple(cache_output) != tuple(full_output):
        raise RuntimeError("cache continuation parity check failed")
