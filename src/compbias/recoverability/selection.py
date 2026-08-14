"""Deterministic fail-closed selection of independent semantic scenes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class SceneCandidate:
    scene_id: str
    family: str
    stage1_parse_success: bool
    natural_perception_error: bool
    operator_sensitive: bool
    design_recoverability_validated: bool

    def __post_init__(self) -> None:
        for value, label in ((self.scene_id, "scene_id"), (self.family, "family")):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{label} must be a bounded safe identifier")
        flags = (
            self.stage1_parse_success,
            self.natural_perception_error,
            self.operator_sensitive,
            self.design_recoverability_validated,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("candidate eligibility flags must be boolean")


def _rank(scene_id: str, family: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{family}:{scene_id}".encode()).digest()


def select_fixed_family_quotas(
    candidates: Sequence[SceneCandidate],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> tuple[SceneCandidate, ...]:
    """Select exact family quotas without extension, top-up, or redistribution."""

    if type(seed) is not int or seed < 1:
        raise ValueError("seed must be a positive integer")
    if not isinstance(quotas, Mapping) or not quotas:
        raise ValueError("quotas must be a non-empty mapping")
    validated_quotas: dict[str, int] = {}
    for family, count in quotas.items():
        if not isinstance(family, str) or _IDENTIFIER.fullmatch(family) is None:
            raise ValueError("quota family must be a bounded safe identifier")
        if type(count) is not int or count < 1:
            raise ValueError("each family quota must be a positive integer")
        validated_quotas[family] = count
    if any(not isinstance(item, SceneCandidate) for item in candidates):
        raise TypeError("candidates must contain SceneCandidate instances")
    identifiers = [item.scene_id for item in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate scene identifiers must be unique")
    unknown = sorted({item.family for item in candidates} - set(validated_quotas))
    if unknown:
        raise ValueError(f"candidate family is not registered in quotas: {unknown[0]}")
    selected: list[SceneCandidate] = []
    for family in sorted(validated_quotas):
        eligible = [
            item
            for item in candidates
            if item.family == family
            and item.stage1_parse_success
            and item.natural_perception_error
            and item.operator_sensitive
            and item.design_recoverability_validated
        ]
        eligible.sort(key=lambda item: (_rank(item.scene_id, family, seed), item.scene_id))
        quota = validated_quotas[family]
        if len(eligible) < quota:
            raise ValueError(
                f"quota unmet for {family}: required {quota}, observed {len(eligible)}"
            )
        selected.extend(eligible[:quota])
    return tuple(
        sorted(
            selected,
            key=lambda item: (item.family, _rank(item.scene_id, item.family, seed)),
        )
    )
