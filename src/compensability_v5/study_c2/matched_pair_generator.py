"""Deterministic matched collision/separating Study C2 prompt construction."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Mapping

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation

_SPLIT_BASE_COUNTS = {
    "support_audit": 48,
    "train": 96,
    "dev": 24,
    "test": 48,
    "positive_control": 16,
}


class C2DataError(ValueError):
    """Study C2 data violates matching, isolation, or domain constraints."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _operation(operator: str, first: int, second: int) -> dict[str, object]:
    return {"operator": operator, "indices": [first, second]}


def _answer(world: tuple[int, int, int, int], operation: Mapping[str, object]) -> int:
    return apply_answer_operation(WorldAction(world), operation)


def _facts(truth: tuple[int, int, int, int], family: str) -> tuple[str, ...]:
    if family == "trend":
        return (
            f"v2-v1={truth[1] - truth[0]}",
            f"v3-v2={truth[2] - truth[1]}",
            f"v4-v3={truth[3] - truth[2]}",
            f"v1={truth[0]}",
        )
    return (
        f"v1+v2={truth[0] + truth[1]}",
        f"v2+v3={truth[1] + truth[2]}",
        f"v3+v4={truth[2] + truth[3]}",
        f"v1={truth[0]}",
    )


def _prompt(
    observation: tuple[int, int, int, int],
    facts: tuple[str, ...],
    operation: Mapping[str, object],
) -> str:
    return (
        "One value in the observed four-value world is wrong. Use the reliable facts to "
        "recover the true world.\n"
        f"Observed world: {','.join(map(str, observation))}\n"
        f"Reliable facts: {'; '.join(facts)}\n"
        f"Question operation metadata: {operation['operator']} indices={operation['indices']}\n"
        "The first line is the only action. Return exactly v1,v2,v3,v4 with values in [2,18]."
    )


def build_matched_reward_pairs(*, seed: int) -> tuple[dict[str, object], ...]:
    """Build the frozen 232-base/464-prompt isolated matched dataset."""

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    ordinal = 0
    for split, base_count in _SPLIT_BASE_COUNTS.items():
        for local_index in range(base_count):
            family = (
                "duplicate_encoding"
                if split == "positive_control"
                else ("cross_series" if local_index % 2 == 0 else "trend")
            )
            truth = tuple(rng.randint(2, 18) for _ in range(4))
            error_index = rng.randrange(4)
            possible_deltas = tuple(
                delta for delta in (-3, -2, -1, 1, 2, 3) if 2 <= truth[error_index] + delta <= 18
            )
            error_delta = rng.choice(possible_deltas)
            observation_values = list(truth)
            observation_values[error_index] += error_delta
            observation = tuple(observation_values)
            other = tuple(index for index in range(4) if index != error_index)
            operator = "sum" if ordinal % 2 == 0 else "difference"
            collision_operation = _operation(operator, other[0], other[1])
            separating_operation = _operation(operator, error_index, other[0])
            facts = _facts(truth, family)
            pair_id = f"c2-{split}-{local_index:04d}"
            for condition, operation in (
                ("collision", collision_operation),
                ("separating", separating_operation),
            ):
                prompt = _prompt(observation, facts, operation)
                rows.append(
                    {
                        "schema_version": 2,
                        "dataset_seed": seed,
                        "pair_id": pair_id,
                        "scene_id": f"{pair_id}-{condition}",
                        "split": split,
                        "condition": condition,
                        "family": family,
                        "truth": list(truth),
                        "observation": list(observation),
                        "facts": list(facts),
                        "error_index": error_index,
                        "error_delta": error_delta,
                        "operation": operation,
                        "gold_answer": _answer(truth, operation),
                        "observed_answer": _answer(observation, operation),
                        "prompt": prompt,
                        "prompt_sha256": _sha256(prompt),
                    }
                )
            ordinal += 1
    validate_matched_pairs(rows)
    return tuple(rows)


def validate_matched_pairs(rows: Iterable[Mapping[str, object]]) -> None:
    materialized = tuple(rows)
    counts = Counter(str(row.get("split")) for row in materialized)
    expected = {split: count * 2 for split, count in _SPLIT_BASE_COUNTS.items()}
    if counts != expected:
        raise C2DataError(f"Study C2 split counts drifted: {dict(counts)}")
    pairs: dict[str, list[Mapping[str, object]]] = {}
    for row in materialized:
        pairs.setdefault(str(row.get("pair_id")), []).append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row.get("condition") for row in pair} != {
            "collision",
            "separating",
        }:
            raise C2DataError(f"matched pair {pair_id} is incomplete")
        for field in ("truth", "observation", "facts", "family", "error_index", "error_delta"):
            if pair[0].get(field) != pair[1].get(field):
                raise C2DataError(f"matched pair {pair_id} drifted in {field}")
        collision = next(row for row in pair if row["condition"] == "collision")
        separating = next(row for row in pair if row["condition"] == "separating")
        if collision["observed_answer"] != collision["gold_answer"]:
            raise C2DataError(f"collision pair {pair_id} does not collide")
        if separating["observed_answer"] == separating["gold_answer"]:
            raise C2DataError(f"separating pair {pair_id} does not separate")


def validate_data_isolation(
    rows: Iterable[Mapping[str, object]], *, forbidden_scene_ids: set[str]
) -> None:
    scene_ids = [str(row.get("scene_id")) for row in rows]
    if len(scene_ids) != len(set(scene_ids)):
        raise C2DataError("Study C2 scene IDs are duplicated")
    overlap = sorted(set(scene_ids) & forbidden_scene_ids)
    if overlap:
        raise C2DataError(f"Study C2 scene overlap detected: {overlap[:3]}")


__all__ = [
    "C2DataError",
    "build_matched_reward_pairs",
    "validate_data_isolation",
    "validate_matched_pairs",
]
