"""Fail-closed JSON parser for the restricted reasoning DSL."""

from __future__ import annotations

import json
import re

from .schema import Program, ProgramOperation, ProgramStep

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_MAX_BYTES = 16_384
_MAX_VARIABLES = 64
_MAX_STEPS = 32
_MAX_INPUTS = 16


class ProgramParseError(ValueError):
    """The model output is not a valid closed DSL program."""


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProgramParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ProgramParseError(f"non-finite JSON constant is forbidden: {value}")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ProgramParseError(f"{label} must be a bounded DSL identifier")
    return value


def _exact_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ProgramParseError(f"{label} must be an exact integer")
    return value


def _load(raw: str) -> object:
    if not isinstance(raw, str):
        raise ProgramParseError("program output must be text")
    if len(raw.encode("utf-8")) > _MAX_BYTES:
        raise ProgramParseError("program output exceeds the 16384 bytes limit")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ProgramParseError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ProgramParseError("program output must be one exact JSON object") from error


def _parse_step(
    raw: object,
    *,
    available: set[str],
    outputs: set[str],
) -> ProgramStep:
    if not isinstance(raw, dict) or set(raw) != {"op", "inputs", "output"}:
        raise ProgramParseError("each step must contain exactly op, inputs, and output")
    try:
        operation = ProgramOperation(raw["op"])
    except (TypeError, ValueError) as error:
        raise ProgramParseError("step op is not in the whitelist") from error
    inputs_raw = raw["inputs"]
    if not isinstance(inputs_raw, list) or not inputs_raw or len(inputs_raw) > _MAX_INPUTS:
        raise ProgramParseError("step inputs must be a bounded non-empty list")
    inputs = tuple(_identifier(item, "step input") for item in inputs_raw)
    if any(item not in available for item in inputs):
        raise ProgramParseError("step contains a forward or unknown input reference")
    output = _identifier(raw["output"], "step output")
    if output in available or output in outputs:
        raise ProgramParseError("step output must not overwrite or duplicate a value")
    return ProgramStep(operation=operation, inputs=inputs, output=output)


def parse_program(raw: str) -> Program:
    """Parse exactly one closed JSON object into an immutable Program."""

    payload = _load(raw)
    if not isinstance(payload, dict) or set(payload) != {"variables", "steps", "answer"}:
        raise ProgramParseError("program must contain exactly variables, steps, and answer")
    variables_raw = payload["variables"]
    if not isinstance(variables_raw, dict) or len(variables_raw) > _MAX_VARIABLES:
        raise ProgramParseError("variables must be a bounded JSON object")
    variables: list[tuple[str, int]] = []
    for raw_name, raw_value in variables_raw.items():
        name = _identifier(raw_name, "variable name")
        variables.append((name, _exact_integer(raw_value, f"variable {name}")))
    variables.sort()
    available = {name for name, _value in variables}
    steps_raw = payload["steps"]
    if not isinstance(steps_raw, list) or len(steps_raw) > _MAX_STEPS:
        raise ProgramParseError("steps must be a list with at most 32 entries")
    steps: list[ProgramStep] = []
    outputs: set[str] = set()
    for raw_step in steps_raw:
        step = _parse_step(raw_step, available=available, outputs=outputs)
        steps.append(step)
        outputs.add(step.output)
        available.add(step.output)
    return Program(
        variables=tuple(variables),
        steps=tuple(steps),
        answer=_exact_integer(payload["answer"], "answer"),
    )
