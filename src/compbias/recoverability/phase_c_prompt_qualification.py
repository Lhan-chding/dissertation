"""Low-cost, format-separated qualification of the repaired Phase-C prompt."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .compatibility import VisibleConstraint
from .dsl.executor import TrustedBinding
from .dsl.result_program import evaluate_result_program, parse_result_program
from .dsl.schema import ProgramOperation, ProgramStep
from .operators import apply_operation
from .phase_c_arms import _constraint_payload
from .phase_c_screen import build_family_constraints
from .phase_c_screen_result import FrozenEligibleScene

PROMPT_QUALIFICATION_CONDITIONS = ("no_cue", "valid_cue")
_FAMILIES = ("cross_series", "duplicate_encoding", "trend")
_OPERATIONS = ("difference", "sum", "max_minus_min")
_VARIABLES = ("a", "b", "c", "d")


@dataclass(frozen=True, slots=True)
class PhaseCPromptQualificationConfig:
    schema_version: int
    status: str
    qualification_id: str
    dataset_id: str
    source_screen_result: str
    output_subdirectory: str
    families: tuple[str, ...]
    operations: tuple[str, ...]
    conditions: tuple[str, ...]
    scenes_per_cell: int
    forks_per_condition: int
    model_call_cap: int
    seed: int
    max_new_tokens: int
    temperature: float
    top_p: float
    format_retries: int
    hypothesis_tested: bool
    scale_authorized: bool
    training_authorized: bool
    rl_authorized: bool


@dataclass(frozen=True, slots=True)
class PhaseCPromptQualificationCall:
    call_id: str
    scene_id: str
    family: str
    chart_type: str
    operation: str
    condition: str
    fork_index: int
    sampling_seed: int
    observed_values: tuple[int, int, int, int]
    expected_values: tuple[int, int, int, int]
    expected_answer: int
    messages: tuple[dict[str, object], ...]
    format_retries: int


@dataclass(frozen=True, slots=True)
class PhaseCPromptQualificationRecord:
    call_id: str
    scene_id: str
    family: str
    chart_type: str
    operation: str
    condition: str
    fork_index: int
    sampling_seed: int
    raw_text: str
    expected_values: tuple[int, int, int, int]
    expected_answer: int
    strict_parse_success: bool
    strict_execution_success: bool
    strict_answer_correct: bool
    semantic_world_extracted: bool
    semantic_values: tuple[int, int, int, int] | None
    semantic_world_exact: bool
    semantic_answer: int | None
    semantic_answer_correct: bool
    error_code: str | None


def load_phase_c_prompt_qualification_config(
    path: Path,
) -> PhaseCPromptQualificationConfig:
    mapping = load_yaml_mapping(path, label="Phase C prompt qualification configuration")
    fields = set(PhaseCPromptQualificationConfig.__dataclass_fields__)
    reject_unknown_fields(mapping, fields, label="Phase C prompt qualification configuration")
    if set(mapping) != fields:
        raise ValueError("Phase C prompt qualification configuration is incomplete")
    converted = {
        **mapping,
        "families": tuple(mapping["families"]),
        "operations": tuple(mapping["operations"]),
        "conditions": tuple(mapping["conditions"]),
    }
    candidate = PhaseCPromptQualificationConfig(**converted)
    canonical = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_PROMPT_QUALIFICATION_NOT_HYPOTHESIS_TEST",
        "qualification_id": "recoverability-phase-c-prompt-qualification-v1",
        "dataset_id": "CVA-Recoverability-Causal-v3",
        "source_screen_result": (
            "configs/recoverability/phase_c_screen_v2_frozen_result.yaml"
        ),
        "output_subdirectory": (
            "cva_recoverability_causal_v3/phase_c_prompt_qualification_v1"
        ),
        "families": _FAMILIES,
        "operations": _OPERATIONS,
        "conditions": PROMPT_QUALIFICATION_CONDITIONS,
        "scenes_per_cell": 1,
        "forks_per_condition": 2,
        "model_call_cap": 36,
        "seed": 2026082001,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
        "format_retries": 0,
        "hypothesis_tested": False,
        "scale_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
    }
    if any(getattr(candidate, key) != value for key, value in canonical.items()):
        raise ValueError("Phase C prompt qualification differs from the frozen 36-call diagnostic")
    return candidate


def _rank(seed: int, *parts: object) -> bytes:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode()).digest()


def _registered_steps(operation: str) -> tuple[ProgramStep, ...]:
    if operation == "sum":
        return (ProgramStep(ProgramOperation.ADD, ("a", "b"), "result"),)
    if operation == "difference":
        return (ProgramStep(ProgramOperation.SUBTRACT, ("a", "b"), "result"),)
    return (
        ProgramStep(ProgramOperation.MAX, _VARIABLES, "high"),
        ProgramStep(ProgramOperation.MIN, _VARIABLES, "low"),
        ProgramStep(ProgramOperation.SUBTRACT, ("high", "low"), "result"),
    )


def _example(operation: str) -> str:
    variables = {"a": 9, "b": 4, "c": 6, "d": 2}
    if operation == "sum":
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
    elif operation == "difference":
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
    else:
        steps = [
            {"op": "max", "inputs": list(_VARIABLES), "output": "high"},
            {"op": "min", "inputs": list(_VARIABLES), "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
    return json.dumps(
        {"variables": variables, "steps": steps, "return": "result"},
        separators=(",", ":"),
    )


def _messages(
    *,
    observed: tuple[int, int, int, int],
    constraints: tuple[VisibleConstraint, ...],
    operation: str,
) -> tuple[dict[str, object], ...]:
    system = (
        "You are solving a four-integer recovery task without an image. Array index 0 means a, "
        "1 means b, 2 means c, and 3 means d. observed_values gives the initial a,b,c,d. "
        "At most one observed value may be wrong. Every redundant_fact must hold: known_value "
        "means the value at index equals value; pair_sum means the values at left_index and "
        "right_index add to total; arithmetic_progression means the values at the three listed "
        "indices, in that listed order, have equal consecutive differences. If redundant_facts "
        "is empty, use observed_values unchanged. Infer exactly one final integer world. Return "
        "one JSON object with exactly variables, steps, and return. variables must be an object "
        "with exactly the keys a,b,c,d and an integer value for each key. Every step must have "
        "exactly \"op\", \"inputs\", and "
        "\"output\". return must be the string \"result\". The exact operation-specific template "
        f"is: {_example(operation)}. Replace only the four variable values. No Markdown, no code "
        "fence, no prose, and no extra keys."
    )
    evidence = {
        "observed_values": list(observed),
        "redundant_facts": [_constraint_payload(item) for item in constraints],
        "max_mismatches": 1,
        "operation": operation,
        "image_available": False,
    }
    return (
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        },
    )


def _select_scenes(
    scenes: tuple[FrozenEligibleScene, ...],
    *,
    config: PhaseCPromptQualificationConfig,
) -> tuple[FrozenEligibleScene, ...]:
    if not isinstance(scenes, tuple) or not scenes:
        raise ValueError("eligible scenes must be a non-empty tuple")
    if any(not isinstance(scene, FrozenEligibleScene) for scene in scenes):
        raise TypeError("eligible scenes contain an invalid item")
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("eligible scene identifiers must be unique")
    grouped: dict[tuple[str, str], list[FrozenEligibleScene]] = defaultdict(list)
    for scene in scenes:
        grouped[(scene.family, scene.operation)].append(scene)
    selected: list[FrozenEligibleScene] = []
    for family in config.families:
        for operation in config.operations:
            candidates = grouped[(family, operation)]
            if not candidates:
                raise ValueError(f"no eligible scene for {family}|{operation}")
            ranked = sorted(
                candidates,
                key=lambda scene: (
                    _rank(config.seed, "scene", family, operation, scene.scene_id),
                    scene.scene_id,
                ),
            )
            selected.extend(ranked[: config.scenes_per_cell])
    return tuple(selected)


def build_phase_c_prompt_qualification_calls(
    scenes: tuple[FrozenEligibleScene, ...],
    *,
    config: PhaseCPromptQualificationConfig,
) -> tuple[PhaseCPromptQualificationCall, ...]:
    if not isinstance(config, PhaseCPromptQualificationConfig):
        raise TypeError("config must be a PhaseCPromptQualificationConfig")
    selected = _select_scenes(scenes, config=config)
    calls: list[PhaseCPromptQualificationCall] = []
    for scene in selected:
        valid_constraints = build_family_constraints(scene.family, scene.true_values)
        for condition in config.conditions:
            constraints = valid_constraints if condition == "valid_cue" else ()
            expected = scene.true_values if condition == "valid_cue" else scene.perceived_values
            for fork_index in range(config.forks_per_condition):
                seed_bytes = _rank(
                    config.seed, scene.scene_id, condition, fork_index
                )
                sampling_seed = int.from_bytes(seed_bytes[:8], "big") % (2**31 - 1)
                calls.append(
                    PhaseCPromptQualificationCall(
                        call_id=(
                            f"{scene.scene_id}.prompt-{condition}.fork-{fork_index:02d}"
                        ),
                        scene_id=scene.scene_id,
                        family=scene.family,
                        chart_type=scene.chart_type,
                        operation=scene.operation,
                        condition=condition,
                        fork_index=fork_index,
                        sampling_seed=sampling_seed,
                        observed_values=scene.perceived_values,
                        expected_values=expected,
                        expected_answer=apply_operation(expected, scene.operation),
                        messages=_messages(
                            observed=scene.perceived_values,
                            constraints=constraints,
                            operation=scene.operation,
                        ),
                        format_retries=0,
                    )
                )
    frozen = tuple(
        sorted(calls, key=lambda call: (_rank(config.seed, "call", call.call_id), call.call_id))
    )
    if len(frozen) != config.model_call_cap:
        raise RuntimeError("Phase C prompt qualification did not freeze exactly 36 calls")
    return frozen


def _json_fragment(text: str) -> object | None:
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if 0 <= first < last:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, UnicodeError):
            continue
    return None


def _balanced_value_after_variables(text: str) -> object | None:
    marker = '"variables"'
    start = text.find(marker)
    if start < 0:
        return None
    colon = text.find(":", start + len(marker))
    if colon < 0:
        return None
    opening = next((index for index in range(colon + 1, len(text)) if text[index] in "[{"), -1)
    if opening < 0:
        return None
    pairs = {"[": "]", "{": "}"}
    closing = pairs[text[opening]]
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == text[opening]:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[opening : index + 1])
                except (json.JSONDecodeError, UnicodeError):
                    return None
    return None


def _semantic_world(raw_text: str) -> tuple[int, int, int, int] | None:
    payload = _json_fragment(raw_text)
    raw_variables: object | None = payload.get("variables") if isinstance(payload, dict) else None
    if raw_variables is None:
        raw_variables = _balanced_value_after_variables(raw_text)
    values: object
    if isinstance(raw_variables, dict) and set(raw_variables) == set(_VARIABLES):
        values = tuple(raw_variables[name] for name in _VARIABLES)
    elif isinstance(raw_variables, list) and len(raw_variables) == 4:
        values = tuple(raw_variables)
    else:
        return None
    if not isinstance(values, tuple) or any(type(value) is not int for value in values):
        return None
    return values[0], values[1], values[2], values[3]


def evaluate_phase_c_prompt_qualification_call(
    call: PhaseCPromptQualificationCall,
    raw_text: str,
) -> PhaseCPromptQualificationRecord:
    if not isinstance(call, PhaseCPromptQualificationCall):
        raise TypeError("call must be a PhaseCPromptQualificationCall")
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be text")
    bindings = {
        variable: TrustedBinding(f"qualification.{variable}", expected)
        for variable, expected in zip(_VARIABLES, call.expected_values, strict=True)
    }
    evaluation = evaluate_result_program(raw_text, constraint_bindings=bindings)
    strict_contract = False
    if evaluation.program_parse_success:
        try:
            program = parse_result_program(raw_text)
            strict_contract = (
                tuple(name for name, _value in program.variables) == _VARIABLES
                and program.steps == _registered_steps(call.operation)
                and program.return_variable == "result"
            )
        except ValueError:
            strict_contract = False
    strict_answer_correct = bool(
        strict_contract
        and evaluation.program_execution_success
        and evaluation.executed_result == call.expected_answer
    )
    semantic_values = _semantic_world(raw_text)
    semantic_answer = (
        apply_operation(semantic_values, call.operation) if semantic_values is not None else None
    )
    semantic_exact = semantic_values == call.expected_values
    semantic_correct = (
        semantic_answer == call.expected_answer if semantic_answer is not None else False
    )
    error_code = evaluation.error_code
    if error_code is None and not strict_contract:
        error_code = "strict_program_contract_mismatch"
    elif error_code is None and not strict_answer_correct:
        error_code = "strict_answer_mismatch"
    return PhaseCPromptQualificationRecord(
        call_id=call.call_id,
        scene_id=call.scene_id,
        family=call.family,
        chart_type=call.chart_type,
        operation=call.operation,
        condition=call.condition,
        fork_index=call.fork_index,
        sampling_seed=call.sampling_seed,
        raw_text=raw_text,
        expected_values=call.expected_values,
        expected_answer=call.expected_answer,
        strict_parse_success=evaluation.program_parse_success,
        strict_execution_success=evaluation.program_execution_success,
        strict_answer_correct=strict_answer_correct,
        semantic_world_extracted=semantic_values is not None,
        semantic_values=semantic_values,
        semantic_world_exact=semantic_exact,
        semantic_answer=semantic_answer,
        semantic_answer_correct=semantic_correct,
        error_code=error_code,
    )


def _rates(records: tuple[PhaseCPromptQualificationRecord, ...]) -> dict[str, object]:
    total = len(records)
    extracted = sum(record.semantic_world_extracted for record in records)
    return {
        "model_calls": total,
        "strict_schema_parse_rate": sum(record.strict_parse_success for record in records)
        / total,
        "strict_execution_rate": sum(record.strict_execution_success for record in records)
        / total,
        "strict_answer_accuracy_over_all": sum(
            record.strict_answer_correct for record in records
        )
        / total,
        "semantic_world_extraction_rate": extracted / total,
        "semantic_world_exact_rate_over_all": sum(
            record.semantic_world_exact for record in records
        )
        / total,
        "semantic_world_exact_rate_among_extracted": (
            sum(record.semantic_world_exact for record in records) / extracted if extracted else 0.0
        ),
        "semantic_answer_accuracy_over_all": sum(
            record.semantic_answer_correct for record in records
        )
        / total,
        "semantic_answer_accuracy_among_extracted": (
            sum(record.semantic_answer_correct for record in records) / extracted
            if extracted
            else 0.0
        ),
        "error_counts": dict(
            sorted(Counter(record.error_code or "none" for record in records).items())
        ),
    }


def summarize_phase_c_prompt_qualification(
    records: tuple[PhaseCPromptQualificationRecord, ...],
    *,
    config: PhaseCPromptQualificationConfig,
) -> dict[str, object]:
    if not isinstance(records, tuple) or len(records) != config.model_call_cap:
        raise ValueError("summary requires the exact frozen 36-call diagnostic")
    if len({record.call_id for record in records}) != len(records):
        raise ValueError("prompt qualification call identifiers must be unique")
    by_condition = {
        condition: _rates(tuple(record for record in records if record.condition == condition))
        for condition in config.conditions
    }
    by_cell = {
        f"{family}|{operation}|{condition}": _rates(
            tuple(
                record
                for record in records
                if (record.family, record.operation, record.condition)
                == (family, operation, condition)
            )
        )
        for family in config.families
        for operation in config.operations
        for condition in config.conditions
    }
    return {
        **_rates(records),
        "by_condition": by_condition,
        "by_family_operation_condition": by_cell,
        "selected_scenes": len({record.scene_id for record in records}),
        "conditions": list(config.conditions),
        "format_retries": 0,
        "hypothesis_tested": False,
        "scale_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
        "training_invoked": False,
    }


def record_payload(record: PhaseCPromptQualificationRecord) -> dict[str, object]:
    return asdict(record)


__all__ = [
    "PROMPT_QUALIFICATION_CONDITIONS",
    "PhaseCPromptQualificationCall",
    "PhaseCPromptQualificationConfig",
    "PhaseCPromptQualificationRecord",
    "build_phase_c_prompt_qualification_calls",
    "evaluate_phase_c_prompt_qualification_call",
    "load_phase_c_prompt_qualification_config",
    "record_payload",
    "summarize_phase_c_prompt_qualification",
]
