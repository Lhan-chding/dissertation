"""Strict T1--T6 minimal-output parsing and scoring."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from compensability_v4.theory.constraint_system import World

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
