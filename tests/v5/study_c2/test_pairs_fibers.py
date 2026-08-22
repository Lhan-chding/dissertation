from __future__ import annotations

from collections import Counter

import pytest

from compensability_v5.study_c2.fiber import full_reward_fiber_size, one_edit_fiber_size
from compensability_v5.study_c2.matched_pair_generator import (
    C2DataError,
    build_matched_reward_pairs,
    validate_data_isolation,
)


def test_registered_split_counts_and_matched_collision_invariants() -> None:
    rows = build_matched_reward_pairs(seed=2026082402)
    counts = Counter(str(row["split"]) for row in rows)

    assert counts == {
        "support_audit": 96,
        "train": 192,
        "dev": 48,
        "test": 96,
        "positive_control": 32,
    }
    pairs: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    assert all({str(row["condition"]) for row in pair} == {"collision", "separating"} for pair in pairs.values())
    for pair in pairs.values():
        left, right = pair
        for field in ("truth", "observation", "facts", "family", "error_index", "error_delta"):
            assert left[field] == right[field]
        collision = next(row for row in pair if row["condition"] == "collision")
        separating = next(row for row in pair if row["condition"] == "separating")
        assert collision["observed_answer"] == collision["gold_answer"]
        assert separating["observed_answer"] != separating["gold_answer"]


def test_dataset_isolation_fails_closed() -> None:
    rows = build_matched_reward_pairs(seed=2026082402)
    validate_data_isolation(rows, forbidden_scene_ids={"legacy-scene"})
    with pytest.raises(C2DataError, match="overlap"):
        validate_data_isolation(rows, forbidden_scene_ids={str(rows[0]["scene_id"])})


def test_full_domain_and_one_edit_fibers_are_exact() -> None:
    operation = {"operator": "sum", "indices": [0, 1]}
    assert full_reward_fiber_size(operation, 5) == 2 * 17 * 17
    assert one_edit_fiber_size((2, 3, 8, 9), operation) == 1

    range_operation = {"operator": "max_minus_min", "indices": [0, 1, 2, 3]}
    brute = sum(
        max(a, b, c, d) - min(a, b, c, d) == 3
        for a in range(2, 19)
        for b in range(2, 19)
        for c in range(2, 19)
        for d in range(2, 19)
    )
    assert full_reward_fiber_size(range_operation, 3) == brute
