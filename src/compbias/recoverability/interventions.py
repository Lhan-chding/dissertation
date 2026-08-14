"""Gold-free Stage-2 payloads for matched cue interventions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
)
from .operators import Operation

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class CueCondition(str, Enum):
    ABLATED = "ablated"
    VALID = "valid"
    SHAM = "sham"
    COUNTERFACTUAL = "counterfactual"
    ORACLE_PERCEPTION = "oracle_perception"
    OPERATOR_SWAP = "operator_swap"


@dataclass(frozen=True, slots=True)
class Stage2Evidence:
    observed_values: tuple[int, int, int, int]
    redundant_facts: tuple[
        PairSumConstraint | KnownValueConstraint | ArithmeticProgressionConstraint, ...
    ]
    axis_facts: tuple[str, ...]
    max_mismatches: int

    def __post_init__(self) -> None:
        if not isinstance(self.observed_values, tuple) or len(self.observed_values) != 4:
            raise ValueError("observed_values must contain exactly four positions")
        if any(type(value) is not int for value in self.observed_values):
            raise TypeError("observed_values must contain exact integers")
        if not isinstance(self.redundant_facts, tuple) or any(
            not isinstance(
                item,
                (PairSumConstraint, KnownValueConstraint, ArithmeticProgressionConstraint),
            )
            for item in self.redundant_facts
        ):
            raise TypeError("redundant_facts must contain registered constraints")
        identifiers = tuple(item.constraint_id for item in self.redundant_facts)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("redundant constraint identifiers must be unique")
        if not isinstance(self.axis_facts, tuple) or any(
            not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None
            for item in self.axis_facts
        ):
            raise ValueError("axis_facts must contain safe identifiers")
        if type(self.max_mismatches) is not int or not 0 <= self.max_mismatches <= 4:
            raise ValueError("max_mismatches must be an integer from zero to four")


@dataclass(frozen=True, slots=True)
class Stage2Payload:
    evidence: Stage2Evidence
    operation: Operation
    question: str
    cue_condition: CueCondition
    cue_constraint_ids: tuple[str, ...]
    randomized_cue_id: str
    dsl_instructions: str
    image_available: bool = False


def build_stage2_payload(
    *,
    evidence: Stage2Evidence,
    operation: Operation,
    question: str,
    cue_condition: CueCondition,
    cue_constraint_ids: tuple[str, ...],
    randomized_cue_id: str,
    dsl_instructions: str,
) -> Stage2Payload:
    """Build the operational mediator interface without accepting hidden gold fields."""

    if not isinstance(evidence, Stage2Evidence):
        raise TypeError("evidence must be Stage2Evidence")
    try:
        registered_operation = Operation(operation)
    except (TypeError, ValueError) as error:
        raise ValueError("operation is not registered") from error
    try:
        registered_condition = CueCondition(cue_condition)
    except (TypeError, ValueError) as error:
        raise ValueError("cue_condition is not registered") from error
    if not isinstance(question, str) or not question or len(question.encode("utf-8")) > 4096:
        raise ValueError("question must be non-empty and at most 4096 UTF-8 bytes")
    if "\x00" in question:
        raise ValueError("question must not contain NUL")
    if not isinstance(cue_constraint_ids, tuple) or any(
        not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None
        for identifier in cue_constraint_ids
    ):
        raise ValueError("cue_constraint_ids must contain safe identifiers")
    if len(set(cue_constraint_ids)) != len(cue_constraint_ids):
        raise ValueError("cue_constraint_ids must be unique")
    for value, label in (
        (randomized_cue_id, "randomized_cue_id"),
        (dsl_instructions, "dsl_instructions"),
    ):
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise ValueError(f"{label} must be a bounded safe identifier")
    available = {item.constraint_id for item in evidence.redundant_facts}
    if not set(cue_constraint_ids) <= available:
        raise ValueError("cue_constraint_ids must reference visible redundant facts")
    if registered_condition is CueCondition.ABLATED and cue_constraint_ids:
        raise ValueError("ablated payload must not expose cue constraints")
    selected = frozenset(cue_constraint_ids)
    public_evidence = Stage2Evidence(
        observed_values=evidence.observed_values,
        redundant_facts=tuple(
            item for item in evidence.redundant_facts if item.constraint_id in selected
        ),
        axis_facts=evidence.axis_facts,
        max_mismatches=evidence.max_mismatches,
    )
    return Stage2Payload(
        evidence=public_evidence,
        operation=registered_operation,
        question=question,
        cue_condition=registered_condition,
        cue_constraint_ids=cue_constraint_ids,
        randomized_cue_id=randomized_cue_id,
        dsl_instructions=dsl_instructions,
    )


def _constraint_payload(
    constraint: PairSumConstraint | KnownValueConstraint | ArithmeticProgressionConstraint,
) -> dict[str, Any]:
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


def serialize_stage2_payload(payload: Stage2Payload) -> dict[str, Any]:
    """Serialize the exact public Stage-2 interface with no hidden evaluation fields."""

    if not isinstance(payload, Stage2Payload):
        raise TypeError("payload must be Stage2Payload")
    return {
        "evidence": {
            "observed_values": list(payload.evidence.observed_values),
            "redundant_facts": [
                _constraint_payload(item) for item in payload.evidence.redundant_facts
            ],
            "axis_facts": list(payload.evidence.axis_facts),
            "max_mismatches": payload.evidence.max_mismatches,
        },
        "operation": payload.operation.value,
        "question": payload.question,
        "randomized_cue_id": payload.randomized_cue_id,
        "dsl_instructions": payload.dsl_instructions,
        "image_available": False,
    }
