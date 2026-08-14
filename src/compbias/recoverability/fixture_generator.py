"""Deterministic 50-scene pre-model audit fixture for Recoverability v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import product

from .compatibility import (
    ArithmeticProgressionConstraint,
    CompatibilityQuery,
    KnownValueConstraint,
    PairSumConstraint,
    analyze_compatibility,
)
from .interventions import (
    CueCondition,
    Stage2Evidence,
    build_stage2_payload,
    serialize_stage2_payload,
)
from .leakage import reject_forbidden_payload_content
from .operators import Operation, apply_operation
from .worlds import CounterfactualPair, SemanticWorld, validate_counterfactual_pair

Constraint = PairSumConstraint | KnownValueConstraint | ArithmeticProgressionConstraint
DOMAIN = tuple(range(1, 19))


@dataclass(frozen=True, slots=True)
class FixtureScene:
    world: SemanticWorld
    observed_values: tuple[int, int, int, int]
    counterfactual: CounterfactualPair
    valid_surface: str
    sham_surface: str
    sham_constraints: tuple[Constraint, ...]
    stage2_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class FixtureAudit:
    records: int
    family_counts: tuple[tuple[str, int], ...]
    operation_counts: tuple[tuple[str, int], ...]
    exactly_recoverable_valid: int
    nonrecoverable_ablated: int
    legal_counterfactuals: int
    valid_sham_surface_matched: int
    nonrecoverable_sham: int
    gold_free_stage2_payloads: int
    unique_numeric_tables: int
    fixture_sha256: str
    audit_passed: bool


@dataclass(frozen=True, slots=True)
class _TrendPattern:
    values: tuple[int, int, int, int]
    original_indices: tuple[tuple[int, int, int], tuple[int, int, int]]
    counterfactual_indices: tuple[tuple[int, int, int], tuple[int, int, int]]
    operation: Operation


_TREND_PATTERNS = (
    _TrendPattern((2, 1, 2, 3), ((1, 0, 3), (1, 2, 3)), ((0, 2, 1), (1, 2, 3)), Operation.SUM),
    _TrendPattern(
        (2, 1, 3, 2), ((1, 0, 2), (1, 3, 2)), ((0, 3, 1), (1, 3, 2)), Operation.DIFFERENCE
    ),
    _TrendPattern((2, 1, 3, 3), ((1, 0, 2), (1, 0, 3)), ((0, 3, 2), (0, 2, 3)), Operation.SUM),
    _TrendPattern(
        (2, 1, 3, 5), ((1, 0, 2), (1, 2, 3)), ((1, 0, 3), (1, 2, 3)), Operation.DIFFERENCE
    ),
    _TrendPattern((2, 1, 5, 3), ((1, 3, 2), (1, 0, 3)), ((1, 0, 2), (1, 3, 2)), Operation.SUM),
    _TrendPattern(
        (2, 2, 1, 3), ((2, 0, 3), (2, 1, 3)), ((0, 1, 2), (2, 1, 3)), Operation.DIFFERENCE
    ),
    _TrendPattern((2, 2, 2, 4), ((0, 2, 1), (0, 1, 2)), ((1, 0, 3), (2, 0, 3)), Operation.SUM),
    _TrendPattern(
        (2, 2, 3, 1), ((2, 0, 3), (2, 1, 3)), ((0, 1, 3), (2, 1, 3)), Operation.DIFFERENCE
    ),
    _TrendPattern((2, 2, 3, 4), ((0, 2, 3), (1, 2, 3)), ((1, 0, 3), (1, 2, 3)), Operation.SUM),
    _TrendPattern(
        (2, 2, 4, 2), ((0, 3, 1), (0, 1, 3)), ((1, 0, 2), (2, 0, 3)), Operation.DIFFERENCE
    ),
    _TrendPattern(
        (2, 4, 5, 6), ((0, 1, 3), (1, 2, 3)), ((0, 1, 2), (1, 2, 3)), Operation.MAX_MINUS_MIN
    ),
    _TrendPattern(
        (2, 2, 4, 6), ((0, 2, 3), (1, 2, 3)), ((1, 0, 2), (1, 2, 3)), Operation.DIFFERENCE
    ),
    _TrendPattern(
        (2, 3, 4, 5), ((0, 1, 2), (1, 2, 3)), ((0, 2, 3), (1, 2, 3)), Operation.MAX_MINUS_MIN
    ),
    _TrendPattern(
        (2, 3, 5, 4), ((0, 1, 3), (1, 3, 2)), ((0, 3, 2), (1, 3, 2)), Operation.MAX_MINUS_MIN
    ),
    _TrendPattern(
        (2, 4, 3, 5), ((0, 2, 1), (2, 1, 3)), ((0, 1, 3), (2, 1, 3)), Operation.MAX_MINUS_MIN
    ),
    _TrendPattern(
        (2, 4, 5, 3), ((0, 3, 1), (2, 1, 3)), ((0, 1, 2), (2, 1, 3)), Operation.MAX_MINUS_MIN
    ),
)


def _constraint_payload(constraint: Constraint) -> dict[str, object]:
    if isinstance(constraint, PairSumConstraint):
        return {
            "constraint_id": constraint.constraint_id,
            "kind": "pair_sum",
            "left_index": constraint.left_index,
            "right_index": constraint.right_index,
            "total": constraint.total,
        }
    if isinstance(constraint, KnownValueConstraint):
        return {
            "constraint_id": constraint.constraint_id,
            "kind": "known_value",
            "index": constraint.index,
            "value": constraint.value,
        }
    return {
        "constraint_id": constraint.constraint_id,
        "kind": "arithmetic_progression",
        "indices": list(constraint.indices),
    }


def _surface(constraints: tuple[Constraint, ...]) -> str:
    return json.dumps(
        [_constraint_payload(item) for item in constraints],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _matched_nonrecoverable_sham(
    observed: tuple[int, int, int, int],
    operation: Operation,
    constraints: tuple[Constraint, ...],
) -> tuple[Constraint, ...]:
    target_width = len(_surface(constraints).encode("utf-8"))
    option_groups: list[tuple[Constraint, ...]] = []

    def compatible_answers(candidate: tuple[Constraint, ...]) -> set[int]:
        worlds = {observed}
        for index in range(4):
            for value in DOMAIN:
                changed = list(observed)
                changed[index] = value
                worlds.add(tuple(changed))  # type: ignore[arg-type]
        return {
            apply_operation(world, operation)
            for world in worlds
            if all(item.accepts(world) for item in candidate)
        }

    for item in constraints:
        if isinstance(item, PairSumConstraint):
            option_groups.append(
                tuple(
                    PairSumConstraint(item.constraint_id, left, right, total)
                    for left in range(4)
                    for right in range(left + 1, 4)
                    for total in range(2, 37)
                    if (left, right, total) != (item.left_index, item.right_index, item.total)
                )
            )
        elif isinstance(item, KnownValueConstraint):
            option_groups.append(
                tuple(
                    KnownValueConstraint(item.constraint_id, index, value)
                    for index in range(4)
                    for value in DOMAIN
                    if (index, value) != (item.index, item.value)
                )
            )
        else:
            indices = tuple(range(4))
            option_groups.append(
                tuple(
                    ArithmeticProgressionConstraint(item.constraint_id, candidate)
                    for candidate in product(indices, repeat=3)
                    if len(set(candidate)) == 3 and candidate != item.indices
                )
            )
    for candidate in product(*option_groups):
        typed = tuple(candidate)
        if len(_surface(typed).encode("utf-8")) > target_width:
            continue
        if len(compatible_answers(typed)) > 1:
            return typed
    raise RuntimeError("no matched nonrecoverable sham exists for fixture scene")


def _queries(
    observed: tuple[int, int, int, int],
    operation: Operation,
    constraints: tuple[Constraint, ...],
) -> tuple[CompatibilityQuery, CompatibilityQuery]:
    return (
        CompatibilityQuery(observed, operation, constraints, DOMAIN, 1),
        CompatibilityQuery(observed, operation, (), DOMAIN, 1),
    )


def _build_scene(
    *,
    scene_id: str,
    family: str,
    chart_type: str,
    operation: Operation,
    values: tuple[int, int, int, int],
    counterfactual_values: tuple[int, int, int, int],
    constraints: tuple[Constraint, ...],
    counterfactual_constraints: tuple[Constraint, ...],
    seed: int,
) -> FixtureScene:
    observed = (values[0] - 1, values[1], values[2], values[3])
    question_id = f"{operation.value}_registered"
    world = SemanticWorld(
        scene_id=scene_id,
        chart_type=chart_type,
        operation=operation,
        question_id=question_id,
        values=values,
        redundancy_family=family,
        split="local_fixture",
        visible_constraints=constraints,
        value_domain=DOMAIN,
    )
    counterfactual_world = SemanticWorld(
        scene_id=f"{scene_id}_cf",
        chart_type=chart_type,
        operation=operation,
        question_id=question_id,
        values=counterfactual_values,
        redundancy_family=family,
        split="local_fixture",
        visible_constraints=counterfactual_constraints,
        value_domain=DOMAIN,
    )
    pair = validate_counterfactual_pair(
        world,
        counterfactual_world,
        changed_value_indices=(0,),
        changed_constraint_ids=tuple(
            left.constraint_id
            for left, right in zip(constraints, counterfactual_constraints, strict=True)
            if left != right
        ),
    )
    evidence = Stage2Evidence(
        observed_values=observed,
        redundant_facts=constraints,
        axis_facts=("integer_ticks", "image_cut"),
        max_mismatches=1,
    )
    payload = build_stage2_payload(
        evidence=evidence,
        operation=operation,
        question=f"Apply {operation.value} to the registered target values.",
        cue_condition=CueCondition.VALID,
        cue_constraint_ids=tuple(item.constraint_id for item in constraints),
        randomized_cue_id=f"cue_{seed}_{scene_id}",
        dsl_instructions="recoverability_dsl_v1",
    )
    serialized = serialize_stage2_payload(payload)
    reject_forbidden_payload_content(serialized)
    sham_constraints = _matched_nonrecoverable_sham(observed, operation, constraints)
    valid_surface = _surface(constraints)
    sham_surface = _surface(sham_constraints)
    return FixtureScene(
        world=world,
        observed_values=observed,
        counterfactual=pair,
        valid_surface=valid_surface,
        sham_surface=sham_surface.ljust(len(valid_surface)),
        sham_constraints=sham_constraints,
        stage2_payload=serialized,
    )


def _cross_scene(index: int, seed: int) -> FixtureScene:
    operation = (Operation.SUM, Operation.DIFFERENCE, Operation.MAX_MINUS_MIN)[index % 3]
    a = 10 + index % 7
    values = (a, 1 + index % 6, 2 + (index * 2) % 6, 3 + (index * 3) % 6)
    counterfactual = (a + 1, values[1], values[2], values[3])
    constraints = (
        PairSumConstraint("pair_ab", 0, 1, values[0] + values[1]),
        PairSumConstraint("pair_ac", 0, 2, values[0] + values[2]),
    )
    cf_constraints = (
        PairSumConstraint("pair_ab", 0, 1, counterfactual[0] + counterfactual[1]),
        PairSumConstraint("pair_ac", 0, 2, counterfactual[0] + counterfactual[2]),
    )
    return _build_scene(
        scene_id=f"fixture_cross_{index:03d}",
        family="cross_series",
        chart_type="grouped_bar",
        operation=operation,
        values=values,
        counterfactual_values=counterfactual,
        constraints=constraints,
        counterfactual_constraints=cf_constraints,
        seed=seed,
    )


def _duplicate_scene(index: int, seed: int) -> FixtureScene:
    operation = (Operation.MAX_MINUS_MIN, Operation.SUM, Operation.DIFFERENCE)[index % 3]
    a = 9 + index % 8
    values = (a, 1 + index % 7, 2 + (index * 3) % 7, 1 + (index * 5) % 7)
    counterfactual = (a + 1, values[1], values[2], values[3])
    constraints = (KnownValueConstraint("duplicate_a", 0, values[0]),)
    cf_constraints = (KnownValueConstraint("duplicate_a", 0, counterfactual[0]),)
    return _build_scene(
        scene_id=f"fixture_duplicate_{index:03d}",
        family="duplicate_encoding",
        chart_type="grouped_bar",
        operation=operation,
        values=values,
        counterfactual_values=counterfactual,
        constraints=constraints,
        counterfactual_constraints=cf_constraints,
        seed=seed,
    )


def _trend_scene(index: int, seed: int) -> FixtureScene:
    pattern = _TREND_PATTERNS[index]
    counterfactual = (pattern.values[0] + 1, *pattern.values[1:])
    constraints = tuple(
        ArithmeticProgressionConstraint(f"trend_{number}", indices)
        for number, indices in enumerate(pattern.original_indices)
    )
    cf_constraints = tuple(
        ArithmeticProgressionConstraint(f"trend_{number}", indices)
        for number, indices in enumerate(pattern.counterfactual_indices)
    )
    return _build_scene(
        scene_id=f"fixture_trend_{index:03d}",
        family="trend",
        chart_type="line",
        operation=pattern.operation,
        values=pattern.values,
        counterfactual_values=counterfactual,
        constraints=constraints,
        counterfactual_constraints=cf_constraints,
        seed=seed,
    )


def generate_fixture_50(*, seed: int) -> tuple[FixtureScene, ...]:
    """Build the fixed 50-scene CPU-only design fixture."""

    if type(seed) is not int or seed < 1:
        raise ValueError("seed must be a positive integer")
    scenes = tuple(
        [
            *(_cross_scene(index, seed) for index in range(17)),
            *(_duplicate_scene(index, seed) for index in range(17)),
            *(_trend_scene(index, seed) for index in range(16)),
        ]
    )
    return tuple(
        sorted(
            scenes,
            key=lambda item: hashlib.sha256(f"{seed}:{item.world.scene_id}".encode()).digest(),
        )
    )


def _record(scene: FixtureScene) -> dict[str, object]:
    return {
        "scene_id": scene.world.scene_id,
        "family": scene.world.redundancy_family,
        "operation": scene.world.operation.value,
        "stage2_payload": scene.stage2_payload,
        "audit": {
            "gold_answer": scene.world.gold_answer,
            "gold_values": list(scene.world.values),
            "observed_values": list(scene.observed_values),
        },
        "counterfactual": {
            "scene_id": scene.counterfactual.counterfactual.scene_id,
            "gold_answer": scene.counterfactual.counterfactual_answer,
            "gold_values": list(scene.counterfactual.counterfactual.values),
            "answer_delta": scene.counterfactual.answer_delta,
        },
    }


def serialize_fixture(scenes: tuple[FixtureScene, ...]) -> tuple[str, ...]:
    if not isinstance(scenes, tuple) or any(not isinstance(item, FixtureScene) for item in scenes):
        raise TypeError("scenes must be a tuple of FixtureScene instances")
    return tuple(
        json.dumps(_record(scene), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for scene in sorted(scenes, key=lambda item: item.world.scene_id)
    )


def fixture_sha256(scenes: tuple[FixtureScene, ...]) -> str:
    digest = hashlib.sha256()
    for line in serialize_fixture(scenes):
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def audit_fixture(scenes: tuple[FixtureScene, ...]) -> FixtureAudit:
    if len(scenes) != 50 or len({item.world.scene_id for item in scenes}) != 50:
        raise ValueError("fixture must contain exactly 50 uniquely identified scenes")
    valid = 0
    ablated = 0
    legal_cf = 0
    matched = 0
    sham_nonrecoverable = 0
    gold_free = 0
    for scene in scenes:
        valid_query, ablated_query = _queries(
            scene.observed_values,
            scene.world.operation,
            scene.world.visible_constraints,
        )
        valid_report = analyze_compatibility(valid_query)
        ablated_report = analyze_compatibility(ablated_query)
        valid += int(
            valid_report.exactly_recoverable
            and valid_report.compatible_answers == (scene.world.gold_answer,)
        )
        ablated += int(len(ablated_report.compatible_answers) > 1)
        cf_report = analyze_compatibility(
            CompatibilityQuery(
                scene.observed_values,
                scene.world.operation,
                scene.counterfactual.counterfactual.visible_constraints,
                DOMAIN,
                1,
            )
        )
        legal_cf += int(
            cf_report.exactly_recoverable
            and cf_report.compatible_answers == (scene.counterfactual.counterfactual_answer,)
            and scene.counterfactual.original_answer != scene.counterfactual.counterfactual_answer
        )
        matched += int(
            len(scene.valid_surface.encode("utf-8")) == len(scene.sham_surface.encode("utf-8"))
        )
        sham_report = analyze_compatibility(
            CompatibilityQuery(
                scene.observed_values,
                scene.world.operation,
                scene.sham_constraints,
                DOMAIN,
                1,
            )
        )
        sham_nonrecoverable += int(
            sham_report.status == "ok" and len(sham_report.compatible_answers) > 1
        )
        serialized_stage2 = json.dumps(scene.stage2_payload, sort_keys=True)
        gold_free += int(
            "gold_answer" not in serialized_stage2 and "gold_scene" not in serialized_stage2
        )
    family_counts = Counter(item.world.redundancy_family for item in scenes)
    operation_counts = Counter(item.world.operation.value for item in scenes)
    unique_tables = len({item.world.values for item in scenes})
    audit_passed = (
        valid == ablated == legal_cf == matched == sham_nonrecoverable == gold_free == 50
        and family_counts == Counter({"cross_series": 17, "duplicate_encoding": 17, "trend": 16})
        and min(operation_counts.values()) >= 16
        and unique_tables == 50
    )
    return FixtureAudit(
        records=50,
        family_counts=tuple(sorted(family_counts.items())),
        operation_counts=tuple(sorted(operation_counts.items())),
        exactly_recoverable_valid=valid,
        nonrecoverable_ablated=ablated,
        legal_counterfactuals=legal_cf,
        valid_sham_surface_matched=matched,
        nonrecoverable_sham=sham_nonrecoverable,
        gold_free_stage2_payloads=gold_free,
        unique_numeric_tables=unique_tables,
        fixture_sha256=fixture_sha256(scenes),
        audit_passed=audit_passed,
    )
