"""Frozen twelve-call diagnostic for four-integer world recovery."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .compatibility import VisibleConstraint
from .phase_c_arms import _constraint_payload
from .phase_c_screen import build_family_constraints
from .phase_c_screen_result import FrozenEligibleScene

WORLD_RECOVERY_CONDITIONS = ("no_cue", "valid_cue")
_FAMILIES = ("cross_series", "duplicate_encoding", "trend")
_DOMAIN = tuple(range(2, 19))
_CSV = re.compile(r"\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\Z")
_FENCE = re.compile(r"\A```[A-Za-z0-9_-]*\s*\n(?P<body>[\s\S]*?)\n```\s*\Z")
_EXAMPLE_TUPLES = frozenset(
    {
        (12, 5, 18, 7),
        (12, 5, 18, 9),
        (6, 13, 9, 4),
        (7, 13, 9, 4),
        (4, 8, 10, 13),
        (4, 7, 10, 13),
        (6, 14, 4, 14),
    }
)
_NO_CUE_TEMPLATE = "redundant_facts: []\nobserved_values: [{{a}},{{b}},{{c}},{{d}}]\n"
_VALID_CUE_TEMPLATE = (
    "redundant_facts:\n{{facts_as_json_array}}\nobserved_values: [{{a}},{{b}},{{c}},{{d}}]\n"
)


class _AmbiguousWorldRecoveryCase(ValueError):
    """A source scene that does not identify one world under the new task."""


@dataclass(frozen=True, slots=True)
class PhaseCWorldRecoveryConfig:
    schema_version: int
    status: str
    qualification_id: str
    dataset_id: str
    source_screen_result: str
    output_subdirectory: str
    families: tuple[str, ...]
    conditions: tuple[str, ...]
    cases_per_family: int
    model_call_cap: int
    seed: int
    max_new_tokens: int
    do_sample: bool
    format_retries: int
    hypothesis_tested: bool
    scale_authorized: bool
    training_authorized: bool
    rl_authorized: bool


@dataclass(frozen=True, slots=True)
class PhaseCWorldRecoveryCall:
    call_id: str
    scene_id: str
    family: str
    case_index: int
    condition: str
    observed_values: tuple[int, int, int, int]
    true_values: tuple[int, int, int, int]
    error_index: int
    facts: tuple[VisibleConstraint, ...]
    facts_json: str
    messages: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class WorldRecoveryParse:
    exact_format_compliance: bool
    values: tuple[int, int, int, int] | None
    parse_failure: bool


@dataclass(frozen=True, slots=True)
class PhaseCWorldRecoveryRecord:
    call_id: str
    scene_id: str
    family: str
    case_index: int
    condition: str
    raw_text: str
    observed_values: tuple[int, int, int, int]
    true_values: tuple[int, int, int, int]
    semantic_values: tuple[int, int, int, int] | None
    exact_format_compliance: bool
    semantic_parse_success: bool
    true_world_recovery: bool
    observation_copy: bool
    all_facts_satisfied: bool | None
    edit_count: int | None
    minimal_valid_repair: bool | None
    correct_error_localization: bool
    wrong_single_edit: bool
    over_edit: bool
    fact_ignored_copy: bool
    unsupported_no_cue_edit: bool
    parse_failure: bool


def load_phase_c_world_recovery_config(path: Path) -> PhaseCWorldRecoveryConfig:
    mapping = load_yaml_mapping(path, label="Phase C world recovery configuration")
    fields = set(PhaseCWorldRecoveryConfig.__dataclass_fields__)
    reject_unknown_fields(mapping, fields, label="Phase C world recovery configuration")
    if set(mapping) != fields:
        raise ValueError("Phase C world recovery configuration is incomplete")
    candidate = PhaseCWorldRecoveryConfig(
        **{
            **mapping,
            "families": tuple(mapping["families"]),
            "conditions": tuple(mapping["conditions"]),
        }
    )
    expected = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_WORLD_RECOVERY_NOT_HYPOTHESIS_TEST",
        "qualification_id": "recoverability-phase-c-world-recovery-v1r1",
        "dataset_id": "CVA-Recoverability-Causal-v3",
        "source_screen_result": "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "output_subdirectory": "cva_recoverability_causal_v3/phase_c_world_recovery_v1r1",
        "families": _FAMILIES,
        "conditions": WORLD_RECOVERY_CONDITIONS,
        "cases_per_family": 2,
        "model_call_cap": 12,
        "seed": 2026082101,
        "max_new_tokens": 32,
        "do_sample": False,
        "format_retries": 0,
        "hypothesis_tested": False,
        "scale_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
    }
    if any(getattr(candidate, key) != value for key, value in expected.items()):
        raise ValueError("Phase C world recovery differs from the frozen twelve-call diagnostic")
    return candidate


def _rank(seed: int, *parts: object) -> bytes:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode()).digest()


def _mismatches(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (first, second) in enumerate(zip(left, right, strict=True))
        if first != second
    )


def _candidate_worlds(
    observed: tuple[int, int, int, int],
    constraints: tuple[VisibleConstraint, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    candidates = {observed}
    for index in range(4):
        for value in _DOMAIN:
            changed = list(observed)
            changed[index] = value
            candidates.add((changed[0], changed[1], changed[2], changed[3]))
    return tuple(
        sorted(
            values
            for values in candidates
            if all(constraint.accepts(values) for constraint in constraints)
        )
    )


def _validated_constraints(scene: FrozenEligibleScene) -> tuple[VisibleConstraint, ...]:
    constraints = build_family_constraints(scene.family, scene.true_values)
    if not constraints or any(not item.accepts(scene.true_values) for item in constraints):
        raise ValueError("world recovery facts do not hold in the hidden truth")
    if all(item.accepts(scene.perceived_values) for item in constraints):
        raise ValueError("world recovery observed values must violate a fact")
    if _candidate_worlds(scene.perceived_values, constraints) != (scene.true_values,):
        raise _AmbiguousWorldRecoveryCase(
            "world recovery facts must identify one unique valid world"
        )
    return constraints


def _render_user(
    observed: tuple[int, int, int, int],
    constraints: tuple[VisibleConstraint, ...],
    *,
    template: str,
) -> tuple[str, str]:
    payload = [_constraint_payload(item) for item in constraints]
    for item in payload:
        item.pop("constraint_id", None)
    facts_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    rendered = template.replace("{{facts_as_json_array}}", facts_json)
    for name, value in zip(("a", "b", "c", "d"), observed, strict=True):
        rendered = rendered.replace("{{" + name + "}}", str(value))
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("world recovery user template contains an unresolved placeholder")
    return rendered.rstrip(), facts_json


def build_phase_c_world_recovery_calls(
    scenes: tuple[FrozenEligibleScene, ...],
    *,
    config: PhaseCWorldRecoveryConfig,
    system_prompt: str,
    no_cue_template: str = _NO_CUE_TEMPLATE,
    valid_cue_template: str = _VALID_CUE_TEMPLATE,
) -> tuple[PhaseCWorldRecoveryCall, ...]:
    if not isinstance(config, PhaseCWorldRecoveryConfig):
        raise TypeError("config must be a PhaseCWorldRecoveryConfig")
    if not isinstance(scenes, tuple) or not scenes:
        raise ValueError("eligible scenes must be a non-empty tuple")
    if any(not isinstance(scene, FrozenEligibleScene) for scene in scenes):
        raise TypeError("eligible scenes contain an invalid item")
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("eligible scene identifiers must be unique")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system prompt must be non-empty text")
    if not isinstance(no_cue_template, str) or not isinstance(valid_cue_template, str):
        raise TypeError("world recovery user templates must be text")

    grouped: dict[str, list[tuple[FrozenEligibleScene, tuple[VisibleConstraint, ...]]]] = (
        defaultdict(list)
    )
    for scene in scenes:
        if scene.true_values in _EXAMPLE_TUPLES or scene.perceived_values in _EXAMPLE_TUPLES:
            continue
        try:
            constraints = _validated_constraints(scene)
        except _AmbiguousWorldRecoveryCase:
            continue
        grouped[scene.family].append((scene, constraints))

    selected: list[tuple[FrozenEligibleScene, tuple[VisibleConstraint, ...], int]] = []
    for family in config.families:
        ranked = sorted(
            grouped[family],
            key=lambda item: (
                _rank(config.seed, "case", family, item[0].scene_id),
                item[0].scene_id,
            ),
        )
        if len(ranked) < config.cases_per_family:
            raise ValueError(f"at least two eligible cases are required for {family}")
        selected.extend(
            (scene, facts, index)
            for index, (scene, facts) in enumerate(ranked[: config.cases_per_family])
        )

    calls: list[PhaseCWorldRecoveryCall] = []
    for scene, valid_facts, case_index in selected:
        error_indices = _mismatches(scene.true_values, scene.perceived_values)
        if len(error_indices) != 1:
            raise ValueError("world recovery case must contain exactly one observed error")
        for condition in config.conditions:
            facts = valid_facts if condition == "valid_cue" else ()
            template = valid_cue_template if condition == "valid_cue" else no_cue_template
            user, facts_json = _render_user(
                scene.perceived_values,
                facts,
                template=template,
            )
            calls.append(
                PhaseCWorldRecoveryCall(
                    call_id=f"{scene.scene_id}.world-{condition}",
                    scene_id=scene.scene_id,
                    family=scene.family,
                    case_index=case_index,
                    condition=condition,
                    observed_values=scene.perceived_values,
                    true_values=scene.true_values,
                    error_index=error_indices[0],
                    facts=facts,
                    facts_json=facts_json,
                    messages=(
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                    ),
                )
            )
    frozen = tuple(sorted(calls, key=lambda call: call.call_id))
    if len(frozen) != config.model_call_cap:
        raise RuntimeError("world recovery plan must contain exactly twelve calls")
    return frozen


def parse_world_recovery_output(raw_text: str) -> WorldRecoveryParse:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be text")
    exact = _CSV.fullmatch(raw_text) is not None
    normalized = raw_text.strip()
    fenced = _FENCE.fullmatch(normalized)
    if fenced is not None:
        normalized = fenced.group("body")
    candidates: list[tuple[int, int, int, int]] = []
    for line in (item.strip() for item in normalized.splitlines() if item.strip()):
        if line.startswith("[") and line.endswith("]"):
            line = line[1:-1].strip()
        match = _CSV.fullmatch(line)
        if match is not None:
            values = tuple(int(value) for value in match.groups())
            candidates.append((values[0], values[1], values[2], values[3]))
    values = candidates[0] if len(candidates) == 1 else None
    return WorldRecoveryParse(
        exact_format_compliance=exact,
        values=values,
        parse_failure=values is None,
    )


def evaluate_phase_c_world_recovery_call(
    call: PhaseCWorldRecoveryCall,
    raw_text: str,
) -> PhaseCWorldRecoveryRecord:
    if not isinstance(call, PhaseCWorldRecoveryCall):
        raise TypeError("call must be a PhaseCWorldRecoveryCall")
    parsed = parse_world_recovery_output(raw_text)
    values = parsed.values
    edit_count = None if values is None else len(_mismatches(values, call.observed_values))
    true_recovery = values == call.true_values
    copied = values == call.observed_values
    valid_condition = call.condition == "valid_cue"
    facts_satisfied = (
        None
        if not valid_condition or values is None
        else all(item.accepts(values) for item in call.facts)
    )
    minimal = (
        None
        if not valid_condition or values is None
        else bool(facts_satisfied and edit_count is not None and edit_count <= 1)
    )
    changed = () if values is None else _mismatches(values, call.observed_values)
    correct_localization = bool(true_recovery and changed == (call.error_index,))
    return PhaseCWorldRecoveryRecord(
        call_id=call.call_id,
        scene_id=call.scene_id,
        family=call.family,
        case_index=call.case_index,
        condition=call.condition,
        raw_text=raw_text,
        observed_values=call.observed_values,
        true_values=call.true_values,
        semantic_values=values,
        exact_format_compliance=parsed.exact_format_compliance,
        semantic_parse_success=not parsed.parse_failure,
        true_world_recovery=true_recovery,
        observation_copy=copied,
        all_facts_satisfied=facts_satisfied,
        edit_count=edit_count,
        minimal_valid_repair=minimal,
        correct_error_localization=correct_localization,
        wrong_single_edit=bool(edit_count == 1 and not true_recovery),
        over_edit=bool(edit_count is not None and edit_count > 1),
        fact_ignored_copy=bool(valid_condition and copied),
        unsupported_no_cue_edit=bool(
            call.condition == "no_cue" and values is not None and not copied
        ),
        parse_failure=parsed.parse_failure,
    )


def _pair_category(
    no_cue: PhaseCWorldRecoveryRecord,
    valid: PhaseCWorldRecoveryRecord,
) -> str:
    if valid.parse_failure:
        return "cue_format_failure"
    if valid.true_world_recovery and not no_cue.true_world_recovery:
        return "cue_corrected"
    if valid.true_world_recovery and no_cue.true_world_recovery:
        return "both_correct"
    if valid.observation_copy:
        return "cue_ignored"
    if valid.over_edit:
        return "cue_overedited"
    if valid.wrong_single_edit:
        return "cue_wrong_single_edit"
    return "other"


def _condition_summary(
    records: tuple[PhaseCWorldRecoveryRecord, ...],
) -> dict[str, object]:
    total = len(records)
    return {
        "model_calls": total,
        "semantic_parse_successes": sum(item.semantic_parse_success for item in records),
        "exact_format_compliant": sum(item.exact_format_compliance for item in records),
        "true_world_recoveries": sum(item.true_world_recovery for item in records),
        "observation_copies": sum(item.observation_copy for item in records),
        "parse_failures": sum(item.parse_failure for item in records),
    }


def summarize_phase_c_world_recovery(
    records: tuple[PhaseCWorldRecoveryRecord, ...],
    *,
    config: PhaseCWorldRecoveryConfig,
) -> dict[str, object]:
    if len(records) != config.model_call_cap:
        raise ValueError("summary requires the exact frozen twelve-call diagnostic")
    if len({item.call_id for item in records}) != len(records):
        raise ValueError("world recovery record identifiers must be unique")
    grouped: dict[str, dict[str, PhaseCWorldRecoveryRecord]] = defaultdict(dict)
    for record in records:
        grouped[record.scene_id][record.condition] = record
    if len(grouped) != 6 or any(set(pair) != set(config.conditions) for pair in grouped.values()):
        raise ValueError("world recovery matched pairs are incomplete")

    pair_rows = []
    for scene_id in sorted(grouped):
        pair = grouped[scene_id]
        pair_rows.append(
            {
                "scene_id": scene_id,
                "family": pair["valid_cue"].family,
                "case_index": pair["valid_cue"].case_index,
                "category": _pair_category(pair["no_cue"], pair["valid_cue"]),
            }
        )
    category_counts = dict(sorted(Counter(row["category"] for row in pair_rows).items()))
    by_condition = {
        condition: _condition_summary(
            tuple(item for item in records if item.condition == condition)
        )
        for condition in config.conditions
    }
    by_family = {
        family: {
            "role": (
                "full_trusted_state_restatement_instruction_following_control"
                if family == "duplicate_encoding"
                else "nontrivial_constraint_recovery_diagnostic"
            ),
            "valid_cue": _condition_summary(
                tuple(
                    item
                    for item in records
                    if item.family == family and item.condition == "valid_cue"
                )
            ),
            "pair_categories": dict(
                sorted(
                    Counter(row["category"] for row in pair_rows if row["family"] == family).items()
                )
            ),
        }
        for family in config.families
    }
    return {
        "model_calls": len(records),
        "selected_cases": len(grouped),
        "conditions": list(config.conditions),
        "pair_category_counts": category_counts,
        "pairs": pair_rows,
        "by_condition": by_condition,
        "by_family": by_family,
        "nontrivial_families": ["cross_series", "trend"],
        "duplicate_encoding_role": ("full_trusted_state_restatement_instruction_following_control"),
        "format_retries": 0,
        "hypothesis_tested": False,
        "scale_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
        "training_invoked": False,
    }


def record_payload(record: PhaseCWorldRecoveryRecord) -> dict[str, object]:
    if not isinstance(record, PhaseCWorldRecoveryRecord):
        raise TypeError("record must be a PhaseCWorldRecoveryRecord")
    return asdict(record)


def hidden_manifest_payload(call: PhaseCWorldRecoveryCall) -> dict[str, object]:
    return {
        "scene_id": call.scene_id,
        "family": call.family,
        "case_index": call.case_index,
        "observed_values": list(call.observed_values),
        "true_values": list(call.true_values),
        "error_index": call.error_index,
        "facts": json.loads(call.facts_json),
    }


def public_manifest_payload(call: PhaseCWorldRecoveryCall) -> dict[str, object]:
    return {
        "scene_id": call.scene_id,
        "family": call.family,
        "case_index": call.case_index,
        "observed_values": list(call.observed_values),
        "facts": json.loads(call.facts_json),
    }


__all__ = [
    "WORLD_RECOVERY_CONDITIONS",
    "PhaseCWorldRecoveryCall",
    "PhaseCWorldRecoveryConfig",
    "PhaseCWorldRecoveryRecord",
    "WorldRecoveryParse",
    "build_phase_c_world_recovery_calls",
    "evaluate_phase_c_world_recovery_call",
    "hidden_manifest_payload",
    "load_phase_c_world_recovery_config",
    "parse_world_recovery_output",
    "public_manifest_payload",
    "record_payload",
    "summarize_phase_c_world_recovery",
]
