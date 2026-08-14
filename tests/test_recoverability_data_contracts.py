from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from compbias.recoverability.compatibility import (
    ArithmeticProgressionConstraint,
    CompatibilityQuery,
    KnownValueConstraint,
    PairSumConstraint,
    analyze_compatibility,
)
from compbias.recoverability.interventions import (
    CueCondition,
    Stage2Evidence,
    build_stage2_payload,
    serialize_stage2_payload,
)
from compbias.recoverability.leakage import (
    assert_cue_group_nonleaking,
    assert_disjoint_numeric_tables,
    build_cue_audit_record,
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
        question_id="difference_ab",
        values=values,
        redundancy_family="cross_series",
        split=split,
        visible_constraints=(constraint,) if constraint else (),
        value_domain=tuple(range(1, 19)),
    )


def test_semantic_world_is_immutable_and_derives_its_answer() -> None:
    world = _world(
        "scene_000001",
        (8, 4, 5, 9),
        constraint=PairSumConstraint("sum_ab", 0, 1, 12),
    )

    assert world.gold_answer == 4
    assert "gold_answer" not in inspect.signature(SemanticWorld).parameters
    with pytest.raises(FrozenInstanceError):
        world.values = (9, 4, 5, 9)  # type: ignore[misc]
    with pytest.raises(ValueError, match="constraint"):
        _world(
            "scene_000002",
            (8, 4, 5, 9),
            constraint=PairSumConstraint("sum_ab", 0, 1, 13),
        )


@pytest.mark.parametrize(
    ("family", "constraint"),
    [
        ("cross_series", PairSumConstraint("sum_ab", 0, 1, 12)),
        ("duplicate_encoding", KnownValueConstraint("duplicate_a", 0, 8)),
        ("trend", ArithmeticProgressionConstraint("trend_abc", (0, 2, 3))),
    ],
)
def test_each_registered_redundancy_family_has_an_executable_constraint(
    family: str,
    constraint: PairSumConstraint | KnownValueConstraint | ArithmeticProgressionConstraint,
) -> None:
    world = SemanticWorld(
        scene_id=f"scene_{family}",
        chart_type="line" if family == "trend" else "grouped_bar",
        operation=Operation.DIFFERENCE,
        question_id="difference_ab",
        values=(8, 4, 5, 2),
        redundancy_family=family,
        split="causal_test",
        visible_constraints=(constraint,),
        value_domain=tuple(range(1, 19)),
    )

    assert constraint.accepts(world.values)


@pytest.mark.parametrize("values", [(True, 4, 5, 9), (19, 4, 5, 9)])
def test_semantic_world_rejects_bool_or_out_of_domain_values(
    values: tuple[object, object, object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SemanticWorld(
            scene_id="scene_invalid_values",
            chart_type="grouped_bar",
            operation=Operation.DIFFERENCE,
            question_id="difference_ab",
            values=values,  # type: ignore[arg-type]
            redundancy_family="cross_series",
            split="causal_test",
            visible_constraints=(),
            value_domain=tuple(range(1, 19)),
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
        ({"values": (9, 4, 6, 9)}, "undeclared value"),
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
        observed_values=(7, 4, 5, 9),
        redundant_facts=(PairSumConstraint("sum_ab", 0, 1, 12),),
        axis_facts=("integer_ticks",),
        max_mismatches=1,
    )
    original = evidence

    payload = build_stage2_payload(
        evidence=evidence,
        operation=Operation.DIFFERENCE,
        question="What is A minus B?",
        cue_condition=CueCondition.VALID,
        cue_constraint_ids=("sum_ab",),
        randomized_cue_id="cue_000001",
        dsl_instructions="recoverability_dsl_v1",
    )

    assert payload.image_available is False
    assert payload.evidence == evidence
    assert payload.operation is Operation.DIFFERENCE
    assert payload.cue_condition is CueCondition.VALID
    assert payload.evidence.max_mismatches == 1
    assert payload.randomized_cue_id == "cue_000001"
    assert payload.dsl_instructions == "recoverability_dsl_v1"
    assert not hasattr(payload, "gold_answer")
    assert not hasattr(payload, "gold_scene")
    assert evidence == original
    assert "gold_answer" not in inspect.signature(build_stage2_payload).parameters
    assert "gold_scene" not in inspect.signature(build_stage2_payload).parameters
    assert "error_position" not in inspect.signature(Stage2Evidence).parameters
    serialized = serialize_stage2_payload(payload)
    assert set(serialized) == {
        "evidence",
        "operation",
        "question",
        "randomized_cue_id",
        "dsl_instructions",
        "image_available",
    }
    assert serialized["evidence"]["redundant_facts"] == [
        {
            "constraint_id": "sum_ab",
            "kind": "pair_sum",
            "left_index": 0,
            "right_index": 1,
            "total": 12,
        }
    ]
    reject_forbidden_payload_content(serialized)
    assert asdict(payload)["image_available"] is False


def test_ablated_stage2_payload_removes_all_cue_facts_and_arm_labels() -> None:
    evidence = Stage2Evidence(
        observed_values=(7, 4, 5, 9),
        redundant_facts=(PairSumConstraint("sum_ab", 0, 1, 12),),
        axis_facts=("integer_ticks",),
        max_mismatches=1,
    )
    payload = build_stage2_payload(
        evidence=evidence,
        operation=Operation.DIFFERENCE,
        question="What is A minus B?",
        cue_condition=CueCondition.ABLATED,
        cue_constraint_ids=(),
        randomized_cue_id="cue_000002",
        dsl_instructions="recoverability_dsl_v1",
    )

    public = serialize_stage2_payload(payload)

    assert public["evidence"]["redundant_facts"] == []
    encoded = json.dumps(public, sort_keys=True)
    assert "ablated" not in encoded
    assert "cue_condition" not in encoded
    assert "cue_constraint_ids" not in encoded


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
    with pytest.raises(ValueError, match=r"forbidden|answer-coded"):
        reject_forbidden_payload_content(payload)


def test_recursive_leakage_guard_accepts_visible_evidence_only() -> None:
    reject_forbidden_payload_content(
        {
            "question": "What is A minus B?",
            "cue": {"constraint_id": "sum_ab", "indices": [0, 1], "total": 12},
            "scene_id": "scene_000123",
        }
    )
    reject_forbidden_payload_content({"question": "Return the answer as an integer."})


def test_cue_group_requires_four_distinct_balanced_answers() -> None:
    records = tuple(
        build_cue_audit_record(
            public_cue={"kind": "pair_sum", "indices": [0, 1], "total": 12},
            question_signature="difference_ab",
            answer=answer,
            count=100,
            cue_only_correct_count=15,
        )
        for answer in (1, 2, 3, 4)
    )

    result = assert_cue_group_nonleaking(
        records,
        min_distinct_answers=4,
        max_answer_imbalance=0,
        max_cue_only_accuracy_upper=0.30,
    )

    assert result.distinct_answers == 4
    assert result.total_records == 400
    assert result.cue_only_accuracy_upper < 0.30
    with pytest.raises(ValueError, match="distinct answers"):
        assert_cue_group_nonleaking(
            records[:3],
            min_distinct_answers=4,
            max_answer_imbalance=0,
            max_cue_only_accuracy_upper=0.30,
        )


def test_cue_audit_groups_by_full_numeric_cue_and_question() -> None:
    records = (
        build_cue_audit_record(
            public_cue={"kind": "pair_sum", "total": 12},
            question_signature="difference_ab",
            answer=1,
            count=100,
            cue_only_correct_count=10,
        ),
        build_cue_audit_record(
            public_cue={"kind": "pair_sum", "total": 13},
            question_signature="difference_ab",
            answer=2,
            count=100,
            cue_only_correct_count=10,
        ),
    )

    with pytest.raises(ValueError, match="same full cue and question"):
        assert_cue_group_nonleaking(
            records,
            min_distinct_answers=2,
            max_answer_imbalance=0,
            max_cue_only_accuracy_upper=0.30,
        )


def test_cue_only_accuracy_upper_bound_is_a_fail_closed_gate() -> None:
    records = tuple(
        build_cue_audit_record(
            public_cue={"kind": "pair_sum", "total": 12},
            question_signature="difference_ab",
            answer=answer,
            count=100,
            cue_only_correct_count=35,
        )
        for answer in (1, 2, 3, 4)
    )

    with pytest.raises(ValueError, match="accuracy upper"):
        assert_cue_group_nonleaking(
            records,
            min_distinct_answers=4,
            max_answer_imbalance=0,
            max_cue_only_accuracy_upper=0.30,
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
        assert_disjoint_numeric_tables({"train": ((1, 2, 3, 4),), "test": ((1, 2, 3, 4),)})


def test_fixed_family_selection_is_deterministic_and_exact() -> None:
    candidates = tuple(
        SceneCandidate(
            scene_id=f"{family}_{index:03d}",
            family=family,
            stage1_parse_success=True,
            natural_perception_error=True,
            operator_sensitive=True,
            design_recoverability_validated=True,
        )
        for family in ("cross_series", "trend", "duplicate_encoding")
        for index in range(8)
    )
    quotas = {"cross_series": 3, "trend": 3, "duplicate_encoding": 2}

    first = select_fixed_family_quotas(candidates, quotas=quotas, seed=2026081603)
    second = select_fixed_family_quotas(tuple(reversed(candidates)), quotas=quotas, seed=2026081603)

    assert first == second
    assert len(first) == 8
    assert {family: sum(item.family == family for item in first) for family in quotas} == quotas
    assert len({item.scene_id for item in first}) == len(first)
    assert tuple(item.scene_id for item in first) == (
        "cross_series_000",
        "cross_series_003",
        "cross_series_005",
        "duplicate_encoding_006",
        "duplicate_encoding_002",
        "trend_006",
        "trend_000",
        "trend_003",
    )
    signature = inspect.signature(SceneCandidate).parameters
    assert "gold_answer" not in signature
    assert "final_answer" not in signature
    assert "answer_correct" not in signature


def test_fixed_family_selection_fails_closed_without_redistribution() -> None:
    candidates = (
        SceneCandidate("cross_1", "cross_series", True, True, True, True),
        SceneCandidate("cross_2", "cross_series", True, True, True, True),
        SceneCandidate("trend_1", "trend", True, True, True, True),
        SceneCandidate("cross_surplus", "cross_series", True, True, True, True),
        SceneCandidate("duplicate_1", "duplicate_encoding", True, True, False, True),
    )

    with pytest.raises(ValueError, match=r"quota unmet.*duplicate_encoding"):
        select_fixed_family_quotas(
            candidates,
            quotas={"cross_series": 2, "trend": 1, "duplicate_encoding": 1},
            seed=2026081603,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"stage1_parse_success": False},
        {"natural_perception_error": False},
        {"operator_sensitive": False},
        {"design_recoverability_validated": False},
    ],
)
def test_fixed_selection_requires_every_pre_outcome_eligibility_clause(
    updates: dict[str, bool],
) -> None:
    eligible = SceneCandidate("cross_eligible", "cross_series", True, True, True, True)
    ineligible = replace(
        SceneCandidate("cross_ineligible", "cross_series", True, True, True, True),
        **updates,
    )

    selected = select_fixed_family_quotas(
        (ineligible, eligible),
        quotas={"cross_series": 1},
        seed=2026081603,
    )

    assert selected == (eligible,)


def test_fixed_selection_changes_with_seed_but_never_counts() -> None:
    candidates = tuple(
        SceneCandidate(f"cross_{index}", "cross_series", True, True, True, True)
        for index in range(12)
    )
    first = select_fixed_family_quotas(candidates, quotas={"cross_series": 4}, seed=2026081603)
    second = select_fixed_family_quotas(candidates, quotas={"cross_series": 4}, seed=2026081604)

    assert len(first) == len(second) == 4
    assert first != second


def test_fixed_selection_rejects_duplicate_unknown_or_boolean_control_fields() -> None:
    candidate = SceneCandidate("cross_1", "cross_series", True, True, True, True)
    with pytest.raises(ValueError, match="unique"):
        select_fixed_family_quotas((candidate, candidate), quotas={"cross_series": 1}, seed=1)
    with pytest.raises(ValueError, match="not registered"):
        select_fixed_family_quotas(
            (replace(candidate, family="unknown"),),
            quotas={"cross_series": 1},
            seed=1,
        )
    with pytest.raises(ValueError, match="seed"):
        select_fixed_family_quotas(
            (candidate,),
            quotas={"cross_series": 1},
            seed=True,  # type: ignore[arg-type]
        )


def test_valid_visible_constraints_create_singleton_compatibility_set() -> None:
    query = CompatibilityQuery(
        observed_values=(7, 4, 5, 9),
        operation=Operation.DIFFERENCE,
        constraints=(
            KnownValueConstraint("known_b", 1, 4),
            PairSumConstraint("sum_ab", 0, 1, 12),
        ),
        value_domain=tuple(range(1, 19)),
        max_mismatches=1,
    )

    report = analyze_compatibility(query)
    worlds = report.compatible_values
    answers = {apply_operation(world, Operation.DIFFERENCE) for world in worlds}

    assert worlds == ((8, 4, 5, 9),)
    assert answers == {4}

    ablated = analyze_compatibility(replace(query, constraints=()))
    assert ablated.exactly_recoverable is False
    assert len(ablated.compatible_answers) > 1


def test_valid_and_counterfactual_cues_resolve_from_the_same_erroneous_mediator() -> None:
    observed = (7, 4, 5, 9)
    common = KnownValueConstraint("known_b", 1, 4)
    valid = analyze_compatibility(
        CompatibilityQuery(
            observed_values=observed,
            operation=Operation.DIFFERENCE,
            constraints=(common, PairSumConstraint("sum_ab", 0, 1, 12)),
            value_domain=tuple(range(1, 19)),
            max_mismatches=1,
        )
    )
    counterfactual = analyze_compatibility(
        CompatibilityQuery(
            observed_values=observed,
            operation=Operation.DIFFERENCE,
            constraints=(common, PairSumConstraint("sum_ab", 0, 1, 13)),
            value_domain=tuple(range(1, 19)),
            max_mismatches=1,
        )
    )

    assert valid.compatible_values == ((8, 4, 5, 9),)
    assert valid.compatible_answers == (4,)
    assert counterfactual.compatible_values == ((9, 4, 5, 9),)
    assert counterfactual.compatible_answers == (5,)
