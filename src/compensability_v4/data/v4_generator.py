"""Deterministic CPU generator for design-recoverable v4 support scenes."""

from __future__ import annotations

import random
from collections.abc import Iterable

from compensability_v4.schemas.scene import RecoveryScene
from compensability_v4.theory.candidate_space import unique_constraint_projection

from .splits import DatasetSplit


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def generate_v4_scenes(
    *,
    count: int,
    seed: int,
    split: DatasetSplit,
    value_domain: Iterable[int] = range(1, 10),
    resized_height: int = 280,
    resized_width: int = 280,
) -> tuple[RecoveryScene, ...]:
    """Generate independent scenes with an auditable, uniquely recoverable fact graph."""

    number = _positive_integer(count, "count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    try:
        canonical_split = split if isinstance(split, DatasetSplit) else DatasetSplit(split)
    except (TypeError, ValueError) as error:
        raise ValueError("split must be registered by v4") from error
    domain = tuple(sorted(set(value_domain)))
    if len(domain) < 2 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in domain
    ):
        raise ValueError("value_domain must contain at least two distinct integers")
    rng = random.Random(seed)
    scenes: list[RecoveryScene] = []
    for index in range(number):
        truth = tuple(rng.choice(domain) for _ in range(4))
        observed_first = rng.choice(tuple(value for value in domain if value != truth[0]))
        observed = (observed_first, truth[1], truth[2], truth[3])
        facts = (
            {"type": "known_value", "index": 1, "value": truth[1]},
            {"type": "known_value", "index": 2, "value": truth[2]},
            {"type": "known_value", "index": 3, "value": truth[3]},
            {
                "type": "pair_sum",
                "left_index": 0,
                "right_index": 1,
                "total": truth[0] + truth[1],
            },
        )
        if unique_constraint_projection(observed, facts, domain) != truth:
            raise AssertionError("generator constructed a non-recoverable scene")
        prefix = f"v4-s{seed:08d}-{index:06d}"
        scenes.append(
            RecoveryScene(
                scene_id=prefix,
                split=canonical_split,
                semantic_scene_id=f"semantic-{prefix}",
                numeric_table_id=f"numbers-{prefix}",
                constraint_graph_id=f"graph-{prefix}",
                truth=truth,  # type: ignore[arg-type]
                facts=facts,
                resized_height=resized_height,
                resized_width=resized_width,
                image_path=f"images/{prefix}.png",
            )
        )
    return tuple(scenes)


def generate_observed_world(
    scene: RecoveryScene, *, error_index: int, replacement_value: int
) -> tuple[int, int, int, int]:
    if not isinstance(scene, RecoveryScene):
        raise TypeError("scene must be a RecoveryScene")
    if (
        isinstance(error_index, bool)
        or not isinstance(error_index, int)
        or not 0 <= error_index < 4
    ):
        raise ValueError("error_index must lie in [0, 3]")
    if isinstance(replacement_value, bool) or not isinstance(replacement_value, int):
        raise TypeError("replacement_value must be an integer")
    if replacement_value == scene.truth[error_index]:
        raise ValueError("replacement_value must create an actual error")
    observed = list(scene.truth)
    observed[error_index] = replacement_value
    return tuple(observed)  # type: ignore[return-value]
