"""Minimal-output capability chain contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class CapabilityTaskType(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"


@dataclass(frozen=True, slots=True)
class CapabilityTask:
    scene_id: str
    task_type: CapabilityTaskType
    expected_output: str


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    scene_id: str
    task_type: CapabilityTaskType
    raw_output: str
    parsed_output: object | None
    parse_success: bool
    is_correct: bool
    establishes_full_recovery: bool


_CSV = re.compile(r"\A(-?\d+),(-?\d+),(-?\d+),(-?\d+)\Z")


def _parse(task_type: CapabilityTaskType, raw_output: str) -> object | None:
    if task_type in {CapabilityTaskType.T1, CapabilityTaskType.T2, CapabilityTaskType.T5}:
        allowed = {"YES", "NO", "CONFLICT", "CONSISTENT", "A", "B", "C", "D"}
        return raw_output if raw_output in allowed else None
    if task_type in {CapabilityTaskType.T3, CapabilityTaskType.T4}:
        return int(raw_output) if raw_output.isdigit() and len(raw_output) == 1 else None
    match = _CSV.fullmatch(raw_output)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())


def evaluate_capability_record(task: CapabilityTask, raw_output: str) -> CapabilityRecord:
    parsed = _parse(task.task_type, raw_output)
    expected = _parse(task.task_type, task.expected_output)
    is_correct = parsed is not None and parsed == expected
    return CapabilityRecord(
        scene_id=task.scene_id,
        task_type=task.task_type,
        raw_output=raw_output,
        parsed_output=parsed,
        parse_success=parsed is not None,
        is_correct=is_correct,
        establishes_full_recovery=task.task_type is CapabilityTaskType.T6 and is_correct,
    )
