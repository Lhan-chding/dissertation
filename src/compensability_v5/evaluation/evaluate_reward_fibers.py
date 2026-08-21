"""Candidate-neighborhood metrics for answer-reward fibers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{label} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def evaluate_reward_fibers(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate one-edit answer fibers without merging direct/relational families."""

    required = {"scene_id", "family", "operation", "truth", "candidates"}
    scene_rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for source in records:
        missing = required - set(source)
        if missing:
            raise ValueError(f"reward-fiber record missing fields: {sorted(missing)}")
        scene_id, family, operation = source["scene_id"], source["family"], source["operation"]
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or scene_id in identifiers
            or not isinstance(family, str)
            or not family
            or not isinstance(operation, str)
            or not operation
        ):
            raise ValueError("reward-fiber metadata is malformed or duplicated")
        identifiers.add(scene_id)
        truth = _world(source["truth"], "truth")
        candidates = source["candidates"]
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not candidates
        ):
            raise ValueError("candidates must be a non-empty sequence")
        seen: set[tuple[int, int, int, int]] = set()
        total_mass = 0.0
        fiber_mass = 0.0
        fiber_size = 0
        true_probability: float | None = None
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise TypeError("candidate rows must be mappings")
            if set(candidate) != {"world", "prior_probability", "answer_correct"}:
                raise ValueError(
                    "candidate schema must contain only world/probability/answer_correct"
                )
            world = _world(candidate["world"], f"candidates[{index}].world")
            probability = _probability(
                candidate["prior_probability"], f"candidates[{index}].prior_probability"
            )
            if not isinstance(candidate["answer_correct"], bool):
                raise TypeError("candidate answer_correct must be boolean")
            if world in seen:
                raise ValueError("candidate worlds must be unique")
            seen.add(world)
            total_mass += probability
            if candidate["answer_correct"]:
                fiber_size += 1
                fiber_mass += probability
            if world == truth:
                true_probability = probability
                if not candidate["answer_correct"]:
                    raise ValueError("the true world must belong to the correct-answer fiber")
        if abs(total_mass - 1.0) > 1e-8:
            raise ValueError("candidate prior probabilities must sum to one")
        if true_probability is None or fiber_size == 0 or fiber_mass <= 0.0:
            raise ValueError("candidate set must contain truth and a positive answer fiber")
        scene_rows.append(
            {
                "scene_id": scene_id,
                "family": family,
                "operation": operation,
                "candidate_count": len(candidates),
                "fiber_size": fiber_size,
                "singleton_fiber": fiber_size == 1,
                "true_state_prior_probability": true_probability,
                "answer_fiber_prior_mass": fiber_mass,
                "fiber_purity": true_probability / fiber_mass,
            }
        )
    if not scene_rows:
        raise ValueError("reward-fiber records must be non-empty")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(str(row["family"]), str(row["operation"]))].append(row)
    strata: list[dict[str, object]] = []
    for (family, operation), group in sorted(grouped.items()):
        strata.append(
            {
                "family": family,
                "operation": operation,
                "scene_count": len(group),
                "mean_fiber_size": sum(int(row["fiber_size"]) for row in group) / len(group),
                "singleton_fiber_rate": sum(bool(row["singleton_fiber"]) for row in group)
                / len(group),
                "mean_true_state_prior_probability": sum(
                    float(row["true_state_prior_probability"]) for row in group
                )
                / len(group),
                "mean_answer_fiber_prior_mass": sum(
                    float(row["answer_fiber_prior_mass"]) for row in group
                )
                / len(group),
                "mean_fiber_purity": sum(float(row["fiber_purity"]) for row in group)
                / len(group),
            }
        )
    return {
        "schema_version": 1,
        "status": "V5_REWARD_FIBERS_EVALUATED",
        "candidate_domain": "frozen_one_edit_values_2_18",
        "scene_count": len(scene_rows),
        "rows": scene_rows,
        "strata": strata,
    }


def reward_fiber_fixture() -> tuple[dict[str, object], ...]:
    return (
        {
            "scene_id": "fixture-fiber",
            "family": "pair_sum",
            "operation": "sum_0_1",
            "truth": [5, 2, 7, 4],
            "candidates": [
                {"world": [5, 2, 7, 4], "prior_probability": 0.2, "answer_correct": True},
                {"world": [4, 3, 7, 4], "prior_probability": 0.3, "answer_correct": True},
                {"world": [4, 2, 7, 4], "prior_probability": 0.5, "answer_correct": False},
            ],
        },
    )


__all__ = ["evaluate_reward_fibers", "reward_fiber_fixture"]
