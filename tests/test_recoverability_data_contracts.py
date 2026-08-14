from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

from compbias.recoverability.compatibility import (
    CompatibilityQuery,
    KnownValueConstraint,
    PairSumConstraint,
    analyze_compatibility,
)
from compbias.recoverability.interventions import (
    CueCondition,
    Stage2Evidence,
    build_stage2_payload,
)
from compbias.recoverability.leakage import (
    CueAuditRecord,
    assert_cue_group_nonleaking,
    assert_disjoint_numeric_tables,
    reject_forbidden_payload_content,
)
from compbias.recoverability.operators import Operation, apply_operation
from compbias.recoverability.selection import SceneCandidate, select_fixed_family_quotas
from compbias.recoverability.worlds import (
    SemanticWorld,
    validate_counterfactual_pair,
)


def _world(
    scene_id: str,
    values: tuple[int, int, int, int],
    *,
    split: str = "causal_test",
    constraint: PairSumConstraint | None = None,
) -> SemanticWorld:
    return SemanticWorld(
        scene_id=scene_id,
        chart_type="grouped_bar",
        operation=Operation.DIFFERENCE,
        values=values,
        redundancy_family="cross_series",
        split=split,
        visible_constraints=(constraint,) if constraint else (),
    )


def test_semantic_world_is_immutable_and_derives_its_answer() -> None:
    world = _world(
        "scene_000001",
        (8, 4, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 12),
    )

    assert world.gold_answer == 4
    with pytest.raises(FrozenInstanceError):
        world.values = (9, 4, 5, 9)  # type: ignore[misc]
    with pytest.raises(ValueError, match="constraint"):
        _world(
            "scene_000002",
            (8, 4, 5, 9),
            constraint=PairSumConstraint("sum_ab", 0, 1, 13),
        )


def test_counterfactual_pair_requires_two_complete_legal_worlds() -> None:
    original = _world(
        "scene_000010",
        (8, 4, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 12),
    )
    counterfactual = _world(
        "scene_000010_cf",
        (9, 4, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 13),
    )

    pair = validate_counterfactual_pair(
        original,
        counterfactual,
        changed_value_indices=(0,),
        changed_constraint_ids=("sum_ab",),
    )

    assert pair.original_answer == 4
    assert pair.counterfactual_answer == 5
    assert pair.answer_delta == 1
    assert pair.original.split == pair.counterfactual.split


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"values": (8, 4, 6, 9)}, "undeclared value"),
        ({"split": "dev"}, "same split"),
        ({"operation": Operation.SUM}, "same operation"),
    ],
)
def test_counterfactual_pair_rejects_incomplete_or_undeclared_changes(
    replacement: dict[str, object], message: str
) -> None:
    original = _world(
        "scene_000020",
        (8, 4, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 12),
    )
    changes: dict[str, object] = {
        "scene_id": "scene_000020_cf",
        "values": (9, 4, 5, 9),
        "visible_constraints": (PairSumConstraint("sum_ab", 0, 1, 13),),
    }
    changes.update(replacement)
    counterfactual = replace(original, **changes)

    with pytest.raises(ValueError, match=message):
        validate_counterfactual_pair(
            original,
            counterfactual,
            changed_value_indices=(0,),
            changed_constraint_ids=("sum_ab",),
        )


def test_counterfactual_pair_rejects_same_answer_and_illegal_constraint() -> None:
    original = _world(
        "scene_000030",
        (8, 4, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 12),
    )
    same_answer = _world(
        "scene_000030_cf",
        (9, 5, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 14),
    )
    with pytest.raises(ValueError, match="answer must change"):
        validate_counterfactual_pair(
            original,
            same_answer,
            changed_value_indices=(0, 1),
            changed_constraint_ids=("sum_ab",),
        )

    with pytest.raises(ValueError, match="constraint"):
        replace(
            original,
            scene_id="scene_000030_bad",
            values=(9, 4, 5, 9),
            visible_constraints=(PairSumConstraint("sum_ab", 0, 1, 99),),
        )


def test_stage2_payload_contains_only_visible_post_error_information() -> None:
    evidence = Stage2Evidence(
        target_facts=(8, None, 5, 9),
        redundant_facts=(PairSumConstraint("sum_ab", 0, 1, 12),),
        axis_facts=("integer_ticks",),
    )
    original = evidence

    payload = build_stage2_payload(
        evidence=evidence,
        operation=Operation.DIFFERENCE,
        question="What is A minus B?",
        cue_condition=CueCondition.VALID,
        cue_constraint_ids=("sum_ab",),
    )

    assert payload.image_available is False
    assert payload.evidence is evidence
    assert payload.operation is Operation.DIFFERENCE
    assert payload.cue_condition is CueCondition.VALID
    assert not hasattr(payload, "gold_answer")
    assert not hasattr(payload, "gold_scene")
    assert evidence == original
    assert "gold_answer" not in inspect.signature(build_stage2_payload).parameters
    assert "gold_scene" not in inspect.signature(build_stage2_payload).parameters


@pytest.mark.parametrize(
    "payload",
    [
        {"recoverable": True},
        {"nested": {"gold_answer": 5}},
        {"nested": [{"compensation_category": "repair"}]},
        {"scene_id": "scene_answer_5"},
        {"cue": {"gold_target_value": 8}},
        {"route": "gold_reasoning_route"},
    ],
)
def test_recursive_leakage_guard_rejects_gold_or_answer_coded_content(
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="forbidden|answer-coded"):
        reject_forbidden_payload_content(payload)


def test_recursive_leakage_guard_accepts_visible_evidence_only() -> None:
    reject_forbidden_payload_content(
        {
            "question": "What is A minus B?",
            "cue": {"constraint_id": "sum_ab", "indices": [0, 1], "total": 12},
            "scene_id": "scene_000123",
        }
    )


def test_cue_group_requires_four_distinct_balanced_answers() -> None:
    records = tuple(
        CueAuditRecord(
            cue_pattern="pair_sum_two_fields",
            answer=answer,
            count=10,
        )
        for answer in (1, 2, 3, 4)
    )

    result = assert_cue_group_nonleaking(
        records,
        min_distinct_answers=4,
        max_answer_imbalance=0,
    )

    assert result.distinct_answers == 4
    assert result.total_records == 40
    with pytest.raises(ValueError, match="distinct answers"):
        assert_cue_group_nonleaking(
            records[:3], min_distinct_answers=4, max_answer_imbalance=0
        )


def test_numeric_tables_are_disjoint_across_splits() -> None:
    assert_disjoint_numeric_tables(
        {
            "train": ((1, 2, 3, 4), (5, 6, 7, 8)),
            "dev": ((2, 3, 4, 5),),
            "test": ((3, 4, 5, 6),),
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_numeric_tables(
            {"train": ((1, 2, 3, 4),), "test": ((1, 2, 3, 4),)}
        )


def test_fixed_family_selection_is_deterministic_and_exact() -> None:
    candidates = tuple(
        SceneCandidate(
            scene_id=f"{family}_{index:03d}",
            family=family,
            operator_sensitive=True,
            exactly_recoverable=True,
        )
        for family in ("cross_series", "trend", "duplicate_encoding")
        for index in range(8)
    )
    quotas = {"cross_series": 3, "trend": 3, "duplicate_encoding": 2}

    first = select_fixed_family_quotas(candidates, quotas=quotas, seed=2026081603)
    second = select_fixed_family_quotas(
        tuple(reversed(candidates)), quotas=quotas, seed=2026081603
    )

    assert first == second
    assert len(first) == 8
    assert {family: sum(item.family == family for item in first) for family in quotas} == quotas
    assert len({item.scene_id for item in first}) == len(first)


def test_fixed_family_selection_fails_closed_without_redistribution() -> None:
    candidates = (
        SceneCandidate("cross_1", "cross_series", True, True),
        SceneCandidate("cross_2", "cross_series", True, True),
        SceneCandidate("trend_1", "trend", True, True),
        SceneCandidate("duplicate_1", "duplicate_encoding", False, True),
    )

    with pytest.raises(ValueError, match="quota unmet.*duplicate_encoding"):
        select_fixed_family_quotas(
            candidates,
            quotas={"cross_series": 2, "trend": 1, "duplicate_encoding": 1},
            seed=2026081603,
        )


def test_valid_visible_constraints_create_singleton_compatibility_set() -> None:
    query = CompatibilityQuery(
        observed_values=(8, 4, 5, 9),
        operation=Operation.DIFFERENCE,
        constraints=(
            KnownValueConstraint("known_b", 1, 4),
            PairSumConstraint("sum_ab", 0, 1, 12),
        ),
        value_domain=tuple(range(1, 19)),
        max_mismatches=0,
    )

    report = analyze_compatibility(query)
    worlds = report.compatible_values
    answers = {apply_operation(world, Operation.DIFFERENCE) for world in worlds}

    assert worlds == ((8, 4, 5, 9),)
    assert answers == {4}
