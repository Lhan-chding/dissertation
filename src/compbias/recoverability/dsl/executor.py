"""Deterministic executor and dataflow audit for restricted programs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .parser import ProgramParseError, parse_program
from .schema import Program, ProgramOperation, ProgramStep

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class ProgramExecutionError(ValueError):
    """A parsed operation cannot be executed in its declared integer domain."""


@dataclass(frozen=True, slots=True)
class ProgramExecutionResult:
    program_execution_success: bool
    executed_result: int
    final_answer: int
    program_answer_match: bool
    consumed_constraint_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProgramEvaluation:
    program_parse_success: bool
    program_execution_success: bool
    executed_result: int | None
    final_answer: int | None
    program_answer_match: bool
    consumed_constraint_ids: tuple[str, ...]
    error_code: str | None


def _require_arity(step: ProgramStep, values: tuple[int, ...], expected: int) -> None:
    if len(values) != expected:
        raise ProgramExecutionError(
            f"{step.operation.value} arity must equal {expected}, observed {len(values)}"
        )


def _execute_step(step: ProgramStep, values: tuple[int, ...]) -> int:
    operation = step.operation
    if operation is ProgramOperation.READ:
        _require_arity(step, values, 1)
        return values[0]
    if operation is ProgramOperation.ADD:
        _require_arity(step, values, 2)
        return values[0] + values[1]
    if operation is ProgramOperation.SUBTRACT:
        _require_arity(step, values, 2)
        return values[0] - values[1]
    if operation in {
        ProgramOperation.MAX,
        ProgramOperation.MIN,
        ProgramOperation.ARGMAX,
        ProgramOperation.ARGMIN,
    }:
        if not values:
            raise ProgramExecutionError(f"{operation.value} arity must be positive")
        if operation is ProgramOperation.MAX:
            return max(values)
        if operation is ProgramOperation.MIN:
            return min(values)
        indices = range(len(values))
        if operation is ProgramOperation.ARGMAX:
            return max(indices, key=lambda index: values[index])
        return min(indices, key=lambda index: values[index])
    if operation is ProgramOperation.SOLVE_SUM_CONSTRAINT:
        _require_arity(step, values, 2)
        return values[0] - values[1]
    if operation is ProgramOperation.SOLVE_DIFFERENCE_CONSTRAINT:
        _require_arity(step, values, 2)
        return values[0] + values[1]
    if operation is ProgramOperation.INTERPOLATE_ARITHMETIC_PROGRESSION:
        _require_arity(step, values, 2)
        total = values[0] + values[1]
        if total % 2:
            raise ProgramExecutionError(
                "interpolate_arithmetic_progression requires an integer midpoint"
            )
        return total // 2
    if operation is ProgramOperation.LOOKUP_DUPLICATE:
        _require_arity(step, values, 1)
        return values[0]
    if operation is ProgramOperation.COMPARE:
        _require_arity(step, values, 2)
        return int(values[0] > values[1]) - int(values[0] < values[1])
    raise ProgramExecutionError("operation is not executable")


def _bindings(
    program: Program, constraint_bindings: Mapping[str, str]
) -> dict[str, frozenset[str]]:
    if not isinstance(constraint_bindings, Mapping):
        raise TypeError("constraint_bindings must be a mapping")
    variable_names = {name for name, _value in program.variables}
    result: dict[str, frozenset[str]] = {name: frozenset() for name in variable_names}
    for variable, constraint_id in constraint_bindings.items():
        if variable not in variable_names:
            raise ProgramExecutionError("constraint binding references an unknown variable")
        if not isinstance(constraint_id, str) or _IDENTIFIER.fullmatch(constraint_id) is None:
            raise ProgramExecutionError("constraint binding identifier is invalid")
        result[variable] = frozenset({constraint_id})
    return result


def execute_program(
    program: Program,
    *,
    constraint_bindings: Mapping[str, str],
) -> ProgramExecutionResult:
    """Execute without mutation and trace only constraints reaching the final result."""

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    values = program.variable_dict()
    provenance = _bindings(program, constraint_bindings)
    if not program.steps:
        raise ProgramExecutionError("program must contain at least one executable step")
    for step in program.steps:
        inputs = tuple(values[name] for name in step.inputs)
        values[step.output] = _execute_step(step, inputs)
        provenance[step.output] = frozenset().union(*(provenance[name] for name in step.inputs))
    final_output = program.steps[-1].output
    executed = values[final_output]
    consumed = tuple(sorted(provenance[final_output]))
    return ProgramExecutionResult(
        program_execution_success=True,
        executed_result=executed,
        final_answer=program.answer,
        program_answer_match=executed == program.answer,
        consumed_constraint_ids=consumed,
    )


def evaluate_program(
    raw: str,
    *,
    constraint_bindings: Mapping[str, str],
) -> ProgramEvaluation:
    """Preserve parse and execution failures as data instead of repairing output."""

    try:
        program = parse_program(raw)
    except ProgramParseError:
        return ProgramEvaluation(
            program_parse_success=False,
            program_execution_success=False,
            executed_result=None,
            final_answer=None,
            program_answer_match=False,
            consumed_constraint_ids=(),
            error_code="program_parse_failure",
        )
    try:
        result = execute_program(program, constraint_bindings=constraint_bindings)
    except ProgramExecutionError:
        return ProgramEvaluation(
            program_parse_success=True,
            program_execution_success=False,
            executed_result=None,
            final_answer=program.answer,
            program_answer_match=False,
            consumed_constraint_ids=(),
            error_code="program_execution_failure",
        )
    return ProgramEvaluation(
        program_parse_success=True,
        program_execution_success=True,
        executed_result=result.executed_result,
        final_answer=result.final_answer,
        program_answer_match=result.program_answer_match,
        consumed_constraint_ids=result.consumed_constraint_ids,
        error_code=None,
    )
