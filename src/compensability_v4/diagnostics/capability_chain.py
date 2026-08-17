"""Strict T1--T6 minimal-output parsing and scoring."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from compbias.recoverability.compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
)
from compbias.recoverability.phase_c_screen import build_family_constraints
from compensability_v4.eval.statistics import scene_clustered_bootstrap_ci
from compensability_v4.theory.candidate_space import (
    enumerate_one_edit_candidates,
    unique_constraint_projection,
)
from compensability_v4.theory.constraint_system import (
    World,
    satisfies_all_facts,
    validate_world,
)

_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_WORLD = re.compile(
    r"((?:0|-?[1-9][0-9]*)),((?:0|-?[1-9][0-9]*)),"
    r"((?:0|-?[1-9][0-9]*)),((?:0|-?[1-9][0-9]*))\Z"
)
_LABEL = re.compile(r"[A-Z]\Z")


class CapabilityTaskType(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def parse_capability_output(task_type: CapabilityTaskType, output: object) -> object | None:
    if not isinstance(task_type, CapabilityTaskType):
        raise TypeError("task_type must be a CapabilityTaskType")
    if not isinstance(output, str):
        return None
    if task_type is CapabilityTaskType.T1:
        return output if output in {"YES", "NO"} else None
    if task_type is CapabilityTaskType.T2:
        return output if output in {"CONFLICT", "CONSISTENT"} else None
    if task_type is CapabilityTaskType.T3:
        return int(output) if output in {"0", "1", "2", "3"} else None
    if task_type is CapabilityTaskType.T4:
        return int(output) if _INTEGER.fullmatch(output) else None
    if task_type is CapabilityTaskType.T5:
        return output if _LABEL.fullmatch(output) else None
    match = _WORLD.fullmatch(output)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())


@dataclass(frozen=True, slots=True)
class CapabilityTask:
    scene_id: str
    task_type: CapabilityTaskType
    expected_output: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _identifier(self.scene_id, "scene_id"))
        if not isinstance(self.task_type, CapabilityTaskType):
            try:
                object.__setattr__(self, "task_type", CapabilityTaskType(self.task_type))
            except (TypeError, ValueError) as error:
                raise ValueError("task_type must be one of T1 through T6") from error
        if parse_capability_output(self.task_type, self.expected_output) is None:
            raise ValueError("expected_output is not canonical for task_type")


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    scene_id: str
    task_type: CapabilityTaskType
    expected_output: str
    raw_output: str
    parsed_output: object | None
    parse_success: bool
    is_correct: bool

    @property
    def establishes_full_recovery(self) -> bool:
        return self.task_type is CapabilityTaskType.T6 and self.is_correct


def evaluate_capability_record(task: CapabilityTask, raw_output: object) -> CapabilityRecord:
    if not isinstance(task, CapabilityTask):
        raise TypeError("task must be a CapabilityTask")
    parsed = parse_capability_output(task.task_type, raw_output)
    expected = parse_capability_output(task.task_type, task.expected_output)
    raw = raw_output if isinstance(raw_output, str) else ""
    return CapabilityRecord(
        scene_id=task.scene_id,
        task_type=task.task_type,
        expected_output=task.expected_output,
        raw_output=raw,
        parsed_output=parsed,
        parse_success=parsed is not None,
        is_correct=parsed is not None and parsed == expected,
    )


@dataclass(frozen=True, slots=True)
class CapabilitySummary:
    task_type: CapabilityTaskType
    number_of_scenes: int
    parse_rate: float
    accuracy: float


def summarize_capability_records(
    records: Iterable[CapabilityRecord],
) -> tuple[CapabilitySummary, ...]:
    record_tuple = tuple(records)
    summaries: list[CapabilitySummary] = []
    for task_type in CapabilityTaskType:
        group = tuple(record for record in record_tuple if record.task_type is task_type)
        if not group:
            continue
        if len({record.scene_id for record in group}) != len(group):
            raise ValueError(f"duplicate scene records for {task_type.value}")
        summaries.append(
            CapabilitySummary(
                task_type=task_type,
                number_of_scenes=len(group),
                parse_rate=sum(record.parse_success for record in group) / len(group),
                accuracy=sum(record.is_correct for record in group) / len(group),
            )
        )
    return tuple(summaries)


def build_t6_expected_output(world: World) -> str:
    return ",".join(str(value) for value in world)


_LEGACY_FAMILIES = ("cross_series", "duplicate_encoding", "trend")
_LEGACY_FAMILY_COUNTS = {"cross_series": 208, "duplicate_encoding": 182, "trend": 190}
_LEGACY_VALUE_DOMAIN = tuple(range(2, 19))


def _fact_mapping(constraint: object) -> Mapping[str, object]:
    if isinstance(constraint, PairSumConstraint):
        value = {
            "type": "pair_sum",
            "left_index": constraint.left_index,
            "right_index": constraint.right_index,
            "total": constraint.total,
            "fact_id": constraint.constraint_id,
        }
    elif isinstance(constraint, KnownValueConstraint):
        value = {
            "type": "known_value",
            "index": constraint.index,
            "value": constraint.value,
            "fact_id": constraint.constraint_id,
        }
    elif isinstance(constraint, ArithmeticProgressionConstraint):
        value = {
            "type": "arithmetic_progression",
            "indices": constraint.indices,
            "fact_id": constraint.constraint_id,
        }
    else:
        raise TypeError("legacy capability constraint is not registered")
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class LegacyCapabilityScene:
    scene_id: str
    family: str
    truth: World
    observed: World
    facts: tuple[Mapping[str, object], ...]
    error_index: int
    value_domain: tuple[int, ...]


def _legacy_scene(row: Mapping[str, object]) -> LegacyCapabilityScene:
    required_true = (
        "parse_success",
        "natural_perception_error",
        "one_position_error",
        "operator_sensitive",
        "design_recoverability_validated",
        "eligible",
    )
    if any(row.get(field) is not True for field in required_true):
        raise ValueError("legacy capability row violates the eligible predicate")
    scene_id = _identifier(row.get("scene_id"), "scene_id")
    family = row.get("family")
    if family not in _LEGACY_FAMILIES:
        raise ValueError("legacy capability family is not registered")
    if row.get("chart_type") not in {"grouped_bar", "line"}:
        raise ValueError("legacy capability chart type is not registered")
    if row.get("operation") not in {"sum", "difference", "max_minus_min"}:
        raise ValueError("legacy capability operation is not registered")
    truth = validate_world(row.get("values"), "values")
    observed = validate_world(row.get("perceived_values"), "perceived_values")
    mismatches = tuple(
        index
        for index, (true_value, observed_value) in enumerate(zip(truth, observed, strict=True))
        if true_value != observed_value
    )
    if len(mismatches) != 1:
        raise ValueError("legacy capability scene must contain exactly one observed error")
    facts = tuple(_fact_mapping(item) for item in build_family_constraints(str(family), truth))
    domain = tuple(sorted(set(_LEGACY_VALUE_DOMAIN) | set(observed)))
    if unique_constraint_projection(observed, facts, domain) != truth:
        raise ValueError("legacy capability facts do not uniquely recover the hidden truth")
    return LegacyCapabilityScene(
        scene_id=scene_id,
        family=str(family),
        truth=truth,
        observed=observed,
        facts=facts,
        error_index=mismatches[0],
        value_domain=domain,
    )


def load_legacy_capability_scenes(
    path: Path,
    *,
    expected_scenes: int = 580,
    expected_family_counts: Mapping[str, int] = _LEGACY_FAMILY_COUNTS,
) -> tuple[LegacyCapabilityScene, ...]:
    """Load the exact eligible legacy slice after its bytes pass the outer SHA gate."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy capability input must be a regular file")
    scenes: list[LegacyCapabilityScene] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("legacy capability record must be a JSON object")
            if row.get("eligible") is True:
                scenes.append(_legacy_scene(row))
    frozen = tuple(sorted(scenes, key=lambda item: item.scene_id))
    if len(frozen) != expected_scenes or len({item.scene_id for item in frozen}) != len(frozen):
        raise ValueError("legacy capability eligible scene count or identifiers differ")
    if Counter(item.family for item in frozen) != Counter(dict(expected_family_counts)):
        raise ValueError("legacy capability family counts differ from frozen evidence")
    return frozen


@dataclass(frozen=True, slots=True)
class CapabilityCall:
    call_id: str
    scene_id: str
    family: str
    task_type: CapabilityTaskType
    expected_output: str
    messages: tuple[Mapping[str, str], ...]
    fact_type: str | None = None
    candidate_worlds: tuple[World, ...] = ()


@dataclass(frozen=True, slots=True)
class CapabilityExecutionRecord:
    call_id: str
    scene_id: str
    family: str
    task_type: CapabilityTaskType
    fact_type: str | None
    expected_output: str
    raw_output: str
    parsed_output: object | None
    parse_success: bool
    is_correct: bool


def _rank(seed: int, *parts: object) -> bytes:
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).digest()


def _messages(prompt: str, payload: Mapping[str, object]) -> tuple[Mapping[str, str], ...]:
    user = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return (
        MappingProxyType({"role": "system", "content": prompt}),
        MappingProxyType({"role": "user", "content": user}),
    )


def _negative_t1_world(scene: LegacyCapabilityScene, fact: Mapping[str, object]) -> World:
    if not satisfies_all_facts(scene.observed, (fact,)):
        return scene.observed
    for candidate in enumerate_one_edit_candidates(scene.truth, scene.value_domain):
        if candidate != scene.truth and not satisfies_all_facts(candidate, (fact,)):
            return candidate
    raise RuntimeError("could not construct a negative T1 candidate")


def _t5_mapping(
    scene: LegacyCapabilityScene,
    *,
    labels: tuple[str, ...],
    true_slot: int,
    seed: int,
) -> tuple[tuple[str, World], ...]:
    distractors = tuple(
        candidate
        for candidate in enumerate_one_edit_candidates(scene.observed, scene.value_domain)
        if candidate not in {scene.observed, scene.truth}
        and not satisfies_all_facts(candidate, scene.facts)
    )
    ranked = sorted(distractors, key=lambda world: (_rank(seed, scene.scene_id, world), world))
    worlds = (scene.truth, scene.observed, *ranked[:2])
    if len(worlds) != 4 or len(set(worlds)) != 4:
        raise RuntimeError("T5 requires four unique matched one-edit candidates")
    remaining = sorted(
        worlds[1:], key=lambda world: (_rank(seed, "assign", scene.scene_id, world), world)
    )
    assigned: list[World] = []
    for index in range(4):
        assigned.append(scene.truth if index == true_slot else remaining.pop(0))
    return tuple(zip(labels, assigned, strict=True))


def build_capability_calls(
    scenes: Iterable[LegacyCapabilityScene],
    *,
    prompts: Mapping[str, str],
    candidate_labels: Iterable[str],
    seed: int,
) -> tuple[CapabilityCall, ...]:
    """Build exactly one deterministic T1--T6 call per frozen legacy scene."""

    frozen = tuple(sorted(scenes, key=lambda item: item.scene_id))
    if not frozen or len({item.scene_id for item in frozen}) != len(frozen):
        raise ValueError("capability scenes must be non-empty with unique identifiers")
    labels = tuple(candidate_labels)
    if len(labels) != 4 or len(set(labels)) != 4:
        raise ValueError("T5 requires exactly four unique candidate labels")
    if set(prompts) != {task.value for task in CapabilityTaskType} or any(
        not isinstance(value, str) or not value.strip() for value in prompts.values()
    ):
        raise ValueError("capability prompts must contain exactly T1 through T6")
    calls: list[CapabilityCall] = []
    for ordinal, scene in enumerate(frozen):
        fact_index = int.from_bytes(_rank(seed, "fact", scene.scene_id)[:8], "big")
        fact = scene.facts[fact_index % len(scene.facts)]
        positive = ordinal % 2 == 0
        t1_world = scene.truth if positive else _negative_t1_world(scene, fact)
        t5 = _t5_mapping(scene, labels=labels, true_slot=ordinal % 4, seed=seed)
        shared = {"facts": [dict(item) for item in scene.facts]}
        payloads: dict[CapabilityTaskType, tuple[str, Mapping[str, object]]] = {
            CapabilityTaskType.T1: (
                "YES" if positive else "NO",
                {"candidate_world": list(t1_world), "fact": dict(fact)},
            ),
            CapabilityTaskType.T2: (
                "CONFLICT",
                {"observed_world": list(scene.observed), **shared},
            ),
            CapabilityTaskType.T3: (
                str(scene.error_index),
                {"observed_world": list(scene.observed), **shared},
            ),
            CapabilityTaskType.T4: (
                str(scene.truth[scene.error_index]),
                {
                    "observed_world": list(scene.observed),
                    "error_index": scene.error_index,
                    **shared,
                },
            ),
            CapabilityTaskType.T5: (
                labels[ordinal % 4],
                {
                    "candidates": [{"label": label, "world": list(world)} for label, world in t5],
                    **shared,
                },
            ),
            CapabilityTaskType.T6: (
                build_t6_expected_output(scene.truth),
                {"observed_world": list(scene.observed), **shared},
            ),
        }
        for task_type in CapabilityTaskType:
            expected, payload = payloads[task_type]
            CapabilityTask(scene.scene_id, task_type, expected)
            calls.append(
                CapabilityCall(
                    call_id=f"{scene.scene_id}.{task_type.value}",
                    scene_id=scene.scene_id,
                    family=scene.family,
                    task_type=task_type,
                    expected_output=expected,
                    messages=_messages(prompts[task_type.value], payload),
                    fact_type=str(fact["type"]) if task_type is CapabilityTaskType.T1 else None,
                    candidate_worlds=(
                        tuple(world for _label, world in t5)
                        if task_type is CapabilityTaskType.T5
                        else ()
                    ),
                )
            )
    return tuple(calls)


def evaluate_capability_call(call: CapabilityCall, raw_output: object) -> CapabilityExecutionRecord:
    if not isinstance(call, CapabilityCall):
        raise TypeError("call must be a CapabilityCall")
    record = evaluate_capability_record(
        CapabilityTask(call.scene_id, call.task_type, call.expected_output), raw_output
    )
    return CapabilityExecutionRecord(
        call_id=call.call_id,
        scene_id=call.scene_id,
        family=call.family,
        task_type=call.task_type,
        fact_type=call.fact_type,
        expected_output=record.expected_output,
        raw_output=record.raw_output,
        parsed_output=record.parsed_output,
        parse_success=record.parse_success,
        is_correct=record.is_correct,
    )


def _gap_payload(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    bootstrap_resamples: int,
) -> dict[str, object]:
    if before.keys() != after.keys() or not before:
        raise ValueError("paired gap requires complete paired scene records")
    rows = tuple(
        {"scene_id": scene_id, "difference": after[scene_id] - before[scene_id]}
        for scene_id in sorted(before)
    )
    interval = scene_clustered_bootstrap_ci(
        rows,
        metric="difference",
        n_resamples=bootstrap_resamples,
        seed=2026081701,
    )
    return {
        "estimate": interval.estimate,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "confidence": interval.confidence,
        "number_of_scenes": interval.number_of_scenes,
    }


def summarize_capability_run(
    records: Iterable[CapabilityExecutionRecord], *, bootstrap_resamples: int = 10_000
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    frozen = tuple(records)
    if not frozen or len({item.call_id for item in frozen}) != len(frozen):
        raise ValueError("capability records must be non-empty with unique call identifiers")
    by_scene: dict[str, dict[CapabilityTaskType, CapabilityExecutionRecord]] = {}
    scene_family: dict[str, str] = {}
    for record in frozen:
        previous = scene_family.setdefault(record.scene_id, record.family)
        task_records = by_scene.setdefault(record.scene_id, {})
        if previous != record.family or record.task_type in task_records:
            raise ValueError("capability records are duplicated or cross family boundaries")
        task_records[record.task_type] = record
    if any(set(group) != set(CapabilityTaskType) for group in by_scene.values()):
        raise ValueError("capability records do not contain complete T1 through T6 pairs")
    summaries: list[dict[str, object]] = []
    for family in sorted(set(scene_family.values())):
        for task_type in CapabilityTaskType:
            group = tuple(
                item for item in frozen if item.family == family and item.task_type is task_type
            )
            summaries.append(
                {
                    "family": family,
                    "task_type": task_type.value,
                    "number_of_scenes": len(group),
                    "parse_rate": sum(item.parse_success for item in group) / len(group),
                    "accuracy": sum(item.is_correct for item in group) / len(group),
                }
            )

    def correctness(task: CapabilityTaskType, scene_ids: Iterable[str]) -> dict[str, float]:
        return {scene_id: float(by_scene[scene_id][task].is_correct) for scene_id in scene_ids}

    scene_ids = tuple(sorted(by_scene))
    gaps: dict[str, object] = {
        "G_search": _gap_payload(
            correctness(CapabilityTaskType.T6, scene_ids),
            correctness(CapabilityTaskType.T5, scene_ids),
            bootstrap_resamples=bootstrap_resamples,
        ),
        "G_loc": _gap_payload(
            correctness(CapabilityTaskType.T3, scene_ids),
            correctness(CapabilityTaskType.T4, scene_ids),
            bootstrap_resamples=bootstrap_resamples,
        ),
        "by_family": {},
        "T5_establishes_full_recovery": False,
        "subjective_success_threshold_applied": False,
    }
    family_gaps: dict[str, object] = {}
    for family in sorted(set(scene_family.values())):
        family_ids = tuple(item for item in scene_ids if scene_family[item] == family)
        family_gaps[family] = {
            "G_search": _gap_payload(
                correctness(CapabilityTaskType.T6, family_ids),
                correctness(CapabilityTaskType.T5, family_ids),
                bootstrap_resamples=bootstrap_resamples,
            ),
            "G_loc": _gap_payload(
                correctness(CapabilityTaskType.T3, family_ids),
                correctness(CapabilityTaskType.T4, family_ids),
                bootstrap_resamples=bootstrap_resamples,
            ),
        }
    gaps["by_family"] = family_gaps
    return tuple(summaries), gaps
