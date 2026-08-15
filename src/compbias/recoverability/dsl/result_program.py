"""Strict result-pointer DSL whose numeric answer is produced only by execution."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .executor import ProgramExecutionError, TrustedBinding, execute_program
from .schema import Program, ProgramOperation, ProgramStep

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_MAX_BYTES = 16_384
_MAX_VARIABLES = 64
_MAX_STEPS = 32
_MAX_INPUTS = 16


class ResultProgramParseError(ValueError):
    """The model output is not one closed result-pointer program."""


@dataclass(frozen=True, slots=True)
class ResultProgram:
    variables: tuple[tuple[str, int], ...]
    steps: tuple[ProgramStep, ...]
    return_variable: str


@dataclass(frozen=True, slots=True)
class ResultProgramExecutionResult:
    program_execution_success: bool
    executed_result: int
    final_answer: int
    return_variable: str
    consumed_constraint_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResultProgramEvaluation:
    program_parse_success: bool
    program_execution_success: bool
    executed_result: int | None
    final_answer: int | None
    return_variable: str | None
    consumed_constraint_ids: tuple[str, ...]
    error_code: str | None


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ResultProgramParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ResultProgramParseError(f"non-finite JSON constant is forbidden: {value}")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ResultProgramParseError(f"{label} must be a bounded DSL identifier")
    return value


def _load(raw: str) -> object:
    if not isinstance(raw, str):
        raise ResultProgramParseError("program output must be text")
    if len(raw.encode("utf-8")) > _MAX_BYTES:
        raise ResultProgramParseError("program output exceeds the 16384 bytes limit")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ResultProgramParseError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ResultProgramParseError("program output must be one exact JSON object") from error


def _parse_step(raw: object, *, available: set[str], outputs: set[str]) -> ProgramStep:
    if not isinstance(raw, dict) or set(raw) != {"op", "inputs", "output"}:
        raise ResultProgramParseError("each step must contain exactly op, inputs, and output")
    try:
        operation = ProgramOperation(raw["op"])
    except (TypeError, ValueError) as error:
        raise ResultProgramParseError("step op is not in the whitelist") from error
    raw_inputs = raw["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs or len(raw_inputs) > _MAX_INPUTS:
        raise ResultProgramParseError("step inputs must be a bounded non-empty list")
    inputs = tuple(_identifier(item, "step input") for item in raw_inputs)
    if any(item not in available for item in inputs):
        raise ResultProgramParseError("step contains a forward or unknown input reference")
    output = _identifier(raw["output"], "step output")
    if output in available or output in outputs:
        raise ResultProgramParseError("step output must not overwrite or duplicate a value")
    return ProgramStep(operation=operation, inputs=inputs, output=output)


def parse_result_program(raw: str) -> ResultProgram:
    """Parse a program with an output pointer and no model-supplied numeric answer."""

    payload = _load(raw)
    if not isinstance(payload, dict) or set(payload) != {"variables", "steps", "return"}:
        raise ResultProgramParseError(
            "result program must contain exactly variables, steps, and return"
        )
    raw_variables = payload["variables"]
    if not isinstance(raw_variables, dict) or len(raw_variables) > _MAX_VARIABLES:
        raise ResultProgramParseError("variables must be a bounded JSON object")
    variables: list[tuple[str, int]] = []
    for raw_name, raw_value in raw_variables.items():
        name = _identifier(raw_name, "variable name")
        if type(raw_value) is not int:
            raise ResultProgramParseError(f"variable {name} must be an exact integer")
        variables.append((name, raw_value))
    variables.sort()
    available = {name for name, _value in variables}
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps or len(raw_steps) > _MAX_STEPS:
        raise ResultProgramParseError("steps must contain from one to 32 entries")
    steps: list[ProgramStep] = []
    outputs: set[str] = set()
    for raw_step in raw_steps:
        step = _parse_step(raw_step, available=available, outputs=outputs)
        steps.append(step)
        outputs.add(step.output)
        available.add(step.output)
    returned = _identifier(payload["return"], "return")
    if returned != steps[-1].output:
        raise ResultProgramParseError("return must reference the final step output")
    return ResultProgram(tuple(variables), tuple(steps), returned)


def execute_result_program(
    program: ResultProgram,
    *,
    constraint_bindings: Mapping[str, TrustedBinding],
) -> ResultProgramExecutionResult:
    """Execute once and make the trusted result the sole numeric final answer."""

    if not isinstance(program, ResultProgram):
        raise TypeError("program must be a ResultProgram")
    legacy = Program(variables=program.variables, steps=program.steps, answer=0)
    result = execute_program(legacy, constraint_bindings=constraint_bindings)
    return ResultProgramExecutionResult(
        program_execution_success=True,
        executed_result=result.executed_result,
        final_answer=result.executed_result,
        return_variable=program.return_variable,
        consumed_constraint_ids=result.consumed_constraint_ids,
    )


def evaluate_result_program(
    raw: str,
    *,
    constraint_bindings: Mapping[str, TrustedBinding],
) -> ResultProgramEvaluation:
    """Preserve strict parse/execution failures without retrying or repairing output."""

    try:
        program = parse_result_program(raw)
    except ResultProgramParseError:
        return ResultProgramEvaluation(False, False, None, None, None, (), "program_parse_failure")
    try:
        result = execute_result_program(program, constraint_bindings=constraint_bindings)
    except ProgramExecutionError:
        return ResultProgramEvaluation(
            True,
            False,
            None,
            None,
            program.return_variable,
            (),
            "program_execution_failure",
        )
    return ResultProgramEvaluation(
        True,
        True,
        result.executed_result,
        result.final_answer,
        result.return_variable,
        result.consumed_constraint_ids,
        None,
    )
