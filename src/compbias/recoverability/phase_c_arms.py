"""Frozen six-arm complete-crossover plan and executor-authoritative scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

from .compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
    VisibleConstraint,
)
from .dsl.executor import TrustedBinding
from .dsl.result_program import evaluate_result_program, parse_result_program
from .dsl.schema import ProgramOperation, ProgramStep
from .operators import Operation, apply_operation
from .phase_c_postscreen_amendment import PhaseCPostscreenAmendment
from .phase_c_screen import build_family_constraints
from .phase_c_screen_result import FrozenEligibleScene

PHASE_C_ARMS = (
    "ablated",
    "valid",
    "sham",
    "counterfactual",
    "oracle_perception",
    "operator_swap",
)
_DOMAIN = tuple(range(2, 19))
_VARIABLES = ("a", "b", "c", "d")


@dataclass(frozen=True, slots=True)
class PhaseCArmCall:
    call_id: str
    scene_id: str
    family: str
    arm: str
    fork_index: int
    sampling_seed: int
    operation: str
    expected_values: tuple[int, int, int, int]
    expected_answer: int
    messages: tuple[dict[str, object], ...]
    trusted_bindings: tuple[tuple[str, str, int], ...]
    required_provenance_prefix: str | None
    image_available: bool
    format_retries: int


@dataclass(frozen=True, slots=True)
class PhaseCArmRecord:
    call_id: str
    scene_id: str
    family: str
    arm: str
    fork_index: int
    sampling_seed: int
    operation: str
    raw_text: str
    program_parse_success: bool
    program_execution_success: bool
    executed_result: int | None
    expected_answer: int
    answer_correct: bool
    required_cue_on_dataflow: bool
    consumed_constraint_ids: tuple[str, ...]
    faithful_success: bool
    error_code: str | None


def _constraint_payload(constraint: VisibleConstraint) -> dict[str, object]:
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
    if isinstance(constraint, ArithmeticProgressionConstraint):
        return {
            "constraint_id": constraint.constraint_id,
            "kind": "arithmetic_progression",
            "indices": list(constraint.indices),
        }
    raise TypeError("constraint is not registered")


def _compatible_answers(
    observed: tuple[int, int, int, int],
    operation: str,
    constraints: tuple[VisibleConstraint, ...],
) -> tuple[int, ...]:
    worlds = (
        values
        for values in _single_error_worlds(observed)
        if all(constraint.accepts(values) for constraint in constraints)
    )
    return tuple(sorted({apply_operation(values, operation) for values in worlds}))


def _single_error_worlds(
    observed: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    worlds = {observed}
    for index in range(4):
        for value in _DOMAIN:
            candidate = list(observed)
            candidate[index] = value
            worlds.add(tuple(candidate))  # type: ignore[arg-type]
    return tuple(sorted(worlds))


def _rank(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(":".join((str(seed), *(str(part) for part in parts))).encode()).digest()


def _counterfactual(
    scene: FrozenEligibleScene, *, seed: int
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[VisibleConstraint, ...],
]:
    candidates: list[
        tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[VisibleConstraint, ...],
        ]
    ] = []
    original_answer = apply_operation(scene.true_values, scene.operation)
    for values in _single_error_worlds(scene.perceived_values):
        if (
            values == scene.true_values
            or apply_operation(values, scene.operation) == original_answer
        ):
            continue
        try:
            constraints = build_family_constraints(scene.family, values)
        except ValueError:
            continue
        if _compatible_answers(scene.perceived_values, scene.operation, constraints) != (
            apply_operation(values, scene.operation),
        ):
            continue
        candidates.append((scene.perceived_values, values, constraints))
    if candidates:
        return min(candidates, key=lambda item: _rank(seed, scene.scene_id, "cf", item[1]))

    mismatch = next(
        index
        for index, (truth, perceived) in enumerate(
            zip(scene.true_values, scene.perceived_values, strict=True)
        )
        if truth != perceived
    )
    error_delta = scene.perceived_values[mismatch] - scene.true_values[mismatch]
    for values in _registered_family_worlds(scene.family):
        if (
            values == scene.true_values
            or apply_operation(values, scene.operation) == original_answer
        ):
            continue
        perceived_value = values[mismatch] + error_delta
        if perceived_value not in _DOMAIN:
            continue
        counterfactual_observed = list(values)
        counterfactual_observed[mismatch] = perceived_value
        observed = tuple(counterfactual_observed)  # type: ignore[assignment]
        if apply_operation(observed, scene.operation) == apply_operation(values, scene.operation):
            continue
        constraints = build_family_constraints(scene.family, values)
        if _compatible_answers(observed, scene.operation, constraints) != (
            apply_operation(values, scene.operation),
        ):
            continue
        candidates.append((observed, values, constraints))
    if not candidates:
        raise ValueError(f"no legal coherent counterfactual exists for {scene.scene_id}")
    return min(
        candidates,
        key=lambda item: (
            sum(left != right for left, right in zip(scene.true_values, item[1], strict=True)),
            _rank(seed, scene.scene_id, "cf-fallback", item[1]),
        ),
    )


@lru_cache(maxsize=3)
def _registered_family_worlds(family: str) -> tuple[tuple[int, int, int, int], ...]:
    if family != "trend":
        return tuple(product(_DOMAIN, repeat=4))
    worlds: list[tuple[int, int, int, int]] = []
    for raw in product(_DOMAIN, repeat=4):
        values = raw[0], raw[1], raw[2], raw[3]
        try:
            build_family_constraints(family, values)
        except ValueError:
            continue
        worlds.append(values)
    return tuple(worlds)


def _sham_constraints(
    scene: FrozenEligibleScene,
    *,
    count: int,
    seed: int,
) -> tuple[VisibleConstraint, ...]:
    bases: list[VisibleConstraint] = []
    worlds = _single_error_worlds(scene.perceived_values)
    if scene.family == "cross_series":
        for left in range(4):
            for right in range(left + 1, 4):
                for values in worlds:
                    bases.append(
                        PairSumConstraint(
                            "sham-base", left, right, values[left] + values[right]
                        )
                    )
    elif scene.family == "duplicate_encoding":
        for index in range(4):
            for value in _DOMAIN:
                bases.append(KnownValueConstraint("sham-base", index, value))
    else:
        for indices in permutations(range(4), 3):
            bases.append(ArithmeticProgressionConstraint("sham-base", indices))
    ranked = sorted(bases, key=lambda item: _rank(seed, scene.scene_id, "sham", repr(item)))
    for base in ranked:
        if isinstance(base, PairSumConstraint):
            candidate = tuple(
                PairSumConstraint(
                    f"sham-{index:02d}", base.left_index, base.right_index, base.total
                )
                for index in range(count)
            )
        elif isinstance(base, KnownValueConstraint):
            candidate = tuple(
                KnownValueConstraint(f"sham-{index:02d}", base.index, base.value)
                for index in range(count)
            )
        else:
            candidate = tuple(
                ArithmeticProgressionConstraint(f"sham-{index:02d}", base.indices)
                for index in range(count)
            )
        if len(_compatible_answers(scene.perceived_values, scene.operation, candidate)) > 1:
            return candidate
    raise ValueError(f"no matched nonrecoverable sham exists for {scene.scene_id}")


def _operator_swap(scene: FrozenEligibleScene, *, seed: int) -> str:
    alternatives = [
        operation.value
        for operation in Operation
        if operation.value != scene.operation
        and apply_operation(scene.true_values, operation) != apply_operation(
            scene.true_values, scene.operation
        )
    ]
    if not alternatives:
        raise ValueError(f"no non-null operator swap exists for {scene.scene_id}")
    return min(alternatives, key=lambda item: _rank(seed, scene.scene_id, "operator", item))


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


def _messages(
    *,
    observed: tuple[int, int, int, int],
    constraints: tuple[VisibleConstraint, ...],
    operation: str,
    randomized_cue_id: str,
) -> tuple[dict[str, object], ...]:
    evidence = {
        "observed_values": list(observed),
        "redundant_facts": [_constraint_payload(item) for item in constraints],
        "max_mismatches": 1,
        "operation": operation,
        "randomized_cue_id": randomized_cue_id,
        "image_available": False,
    }
    system = (
        "You are a strict text-only integer recovery interface. The image is unavailable. "
        "Infer one four-integer world that differs from observed_values in at most one position "
        "and satisfies every visible redundant fact. Then return exactly one JSON result-pointer "
        "program with keys variables, steps, and return. variables must be a,b,c,d. Use add(a,b) "
        "for sum, subtract(a,b) for difference, or max(a,b,c,d), min(a,b,c,d), then subtract "
        "for max_minus_min. Return no answer field, prose, code fence, or extra key."
    )
    return (
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        },
    )


def _arm_spec(
    scene: FrozenEligibleScene,
    *,
    arm: str,
    seed: int,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[VisibleConstraint, ...],
    str,
    str | None,
]:
    valid = build_family_constraints(scene.family, scene.true_values)
    if arm == "ablated":
        return scene.perceived_values, scene.perceived_values, (), scene.operation, None
    if arm == "valid":
        return scene.perceived_values, scene.true_values, valid, scene.operation, "cue."
    if arm == "sham":
        sham = _sham_constraints(scene, count=len(valid), seed=seed)
        return scene.perceived_values, scene.perceived_values, sham, scene.operation, None
    if arm == "counterfactual":
        observed, values, constraints = _counterfactual(scene, seed=seed)
        return observed, values, constraints, scene.operation, "cue."
    if arm == "oracle_perception":
        return scene.true_values, scene.true_values, (), scene.operation, "oracle."
    if arm == "operator_swap":
        swapped = _operator_swap(scene, seed=seed)
        return scene.perceived_values, scene.true_values, valid, swapped, "cue."
    raise ValueError("Phase C arm is not registered")


def build_phase_c_arm_calls(
    scenes: tuple[FrozenEligibleScene, ...],
    *,
    amendment: PhaseCPostscreenAmendment,
) -> tuple[PhaseCArmCall, ...]:
    """Build and deterministically randomize all complete scene-arm-fork calls."""

    if not isinstance(amendment, PhaseCPostscreenAmendment):
        raise TypeError("amendment must be a PhaseCPostscreenAmendment")
    if not isinstance(scenes, tuple) or not scenes:
        raise ValueError("eligible scenes must be a non-empty tuple")
    if any(not isinstance(scene, FrozenEligibleScene) for scene in scenes):
        raise TypeError("eligible scenes contain an invalid item")
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("eligible scene identifiers must be unique")
    if amendment.arms != PHASE_C_ARMS or amendment.forks_per_arm != 8:
        raise ValueError("Phase C arm contract differs from the amendment")
    calls: list[PhaseCArmCall] = []
    for scene in scenes:
        for arm in PHASE_C_ARMS:
            observed, expected, constraints, operation, prefix = _arm_spec(
                scene, arm=arm, seed=amendment.seed
            )
            expected_answer = apply_operation(expected, operation)
            for fork_index in range(amendment.forks_per_arm):
                seed_bytes = _rank(amendment.seed, scene.scene_id, arm, fork_index)
                sampling_seed = int.from_bytes(seed_bytes[:8], "big") % (2**31 - 1)
                cue_id = hashlib.sha256(seed_bytes).hexdigest()[:24]
                if prefix == "cue.":
                    cue_digest = hashlib.sha256(
                        ":".join(item.constraint_id for item in constraints).encode()
                    ).hexdigest()[:24]
                    provenance = f"cue.{cue_digest}"
                elif prefix == "oracle.":
                    provenance = "oracle.perception"
                else:
                    provenance = "stage1.perception"
                bindings = tuple(
                    (variable, f"{provenance}.{index}", value)
                    for index, (variable, value) in enumerate(
                        zip(_VARIABLES, expected, strict=True)
                    )
                )
                call_id = f"{scene.scene_id}.{arm}.fork-{fork_index:02d}"
                calls.append(
                    PhaseCArmCall(
                        call_id=call_id,
                        scene_id=scene.scene_id,
                        family=scene.family,
                        arm=arm,
                        fork_index=fork_index,
                        sampling_seed=sampling_seed,
                        operation=operation,
                        expected_values=expected,
                        expected_answer=expected_answer,
                        messages=_messages(
                            observed=observed,
                            constraints=constraints,
                            operation=operation,
                            randomized_cue_id=cue_id,
                        ),
                        trusted_bindings=bindings,
                        required_provenance_prefix=prefix,
                        image_available=False,
                        format_retries=0,
                    )
                )
    return tuple(
        sorted(calls, key=lambda call: (_rank(amendment.seed, "call", call.call_id), call.call_id))
    )


def evaluate_phase_c_arm_call(call: PhaseCArmCall, raw_text: str) -> PhaseCArmRecord:
    if not isinstance(call, PhaseCArmCall):
        raise TypeError("call must be a PhaseCArmCall")
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be text")
    bindings = {
        variable: TrustedBinding(constraint_id, expected)
        for variable, constraint_id, expected in call.trusted_bindings
    }
    evaluation = evaluate_result_program(raw_text, constraint_bindings=bindings)
    contract_match = False
    if evaluation.program_parse_success:
        try:
            program = parse_result_program(raw_text)
            contract_match = (
                tuple(name for name, _value in program.variables) == _VARIABLES
                and program.steps == _registered_steps(call.operation)
                and program.return_variable == "result"
            )
        except ValueError:
            contract_match = False
    answer_correct = bool(
        contract_match
        and evaluation.program_execution_success
        and evaluation.executed_result == call.expected_answer
        and evaluation.final_answer == call.expected_answer
    )
    cue_ok = call.required_provenance_prefix is None or any(
        identifier.startswith(call.required_provenance_prefix)
        for identifier in evaluation.consumed_constraint_ids
    )
    faithful = bool(answer_correct and cue_ok)
    error_code = evaluation.error_code
    if error_code is None and not contract_match:
        error_code = "program_contract_mismatch"
    elif error_code is None and not answer_correct:
        error_code = "executor_answer_mismatch"
    elif error_code is None and not cue_ok:
        error_code = "required_cue_absent_from_dataflow"
    return PhaseCArmRecord(
        call_id=call.call_id,
        scene_id=call.scene_id,
        family=call.family,
        arm=call.arm,
        fork_index=call.fork_index,
        sampling_seed=call.sampling_seed,
        operation=call.operation,
        raw_text=raw_text,
        program_parse_success=evaluation.program_parse_success,
        program_execution_success=evaluation.program_execution_success,
        executed_result=evaluation.executed_result,
        expected_answer=call.expected_answer,
        answer_correct=answer_correct,
        required_cue_on_dataflow=cue_ok,
        consumed_constraint_ids=evaluation.consumed_constraint_ids,
        faithful_success=faithful,
        error_code=error_code,
    )


__all__ = [
    "PHASE_C_ARMS",
    "FrozenEligibleScene",
    "PhaseCArmCall",
    "PhaseCArmRecord",
    "build_phase_c_arm_calls",
    "evaluate_phase_c_arm_call",
]
