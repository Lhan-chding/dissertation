"""Immutable schema for the recoverability reasoning DSL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProgramOperation(str, Enum):
    READ = "read"
    ADD = "add"
    SUBTRACT = "subtract"
    MAX = "max"
    MIN = "min"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
    SOLVE_SUM_CONSTRAINT = "solve_sum_constraint"
    SOLVE_DIFFERENCE_CONSTRAINT = "solve_difference_constraint"
    INTERPOLATE_ARITHMETIC_PROGRESSION = "interpolate_arithmetic_progression"
    LOOKUP_DUPLICATE = "lookup_duplicate"
    COMPARE = "compare"


@dataclass(frozen=True, slots=True)
class ProgramStep:
    operation: ProgramOperation
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True, slots=True)
class Program:
    variables: tuple[tuple[str, int], ...]
    steps: tuple[ProgramStep, ...]
    answer: int

    def variable_dict(self) -> dict[str, int]:
        return dict(self.variables)
