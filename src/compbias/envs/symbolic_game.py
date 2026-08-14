"""Executable 2x2 perception--reasoning coordination environment.

The symbolic environment assigns the two perception modes and two reasoning
modes from the formal coordination model to concrete actions.  Its expected
reward delegates to :mod:`compbias.theory.coordination`, which remains the
authoritative implementation of the model equation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from compbias.theory.coordination import CoordinationParams
from compbias.theory.coordination import reward as coordination_reward


class PerceptionMode(str, Enum):
    """The registered perception actions in the symbolic game."""

    TRUTHFUL = "T"
    ERRONEOUS = "E"


class ReasoningMode(str, Enum):
    """The registered reasoning actions in the symbolic game."""

    CANONICAL = "C"
    COMPENSATOR = "K"


class CoordinationSolution(str, Enum):
    """Semantic interpretation of an executed action pair."""

    TRUTHFUL = "truthful"
    COMPENSATORY = "compensatory"
    MISMATCHED = "mismatched"


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number")
    try:
        checked = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(checked):
        raise ValueError(f"{field} must be finite")
    return checked


def _probability(value: object, field: str) -> float:
    checked = _finite_real(value, field)
    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return checked


def _mismatch_penalty(value: object, field: str) -> float:
    checked = _finite_real(value, field)
    if checked <= 0.0:
        raise ValueError(f"{field} must be strictly positive")
    return checked


def _perception_mode(value: object) -> PerceptionMode:
    try:
        return PerceptionMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid perception mode: {value!r}") from error


def _reasoning_mode(value: object) -> ReasoningMode:
    try:
        return ReasoningMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid reasoning mode: {value!r}") from error


def _solution_for(
    perception: PerceptionMode,
    reasoning: ReasoningMode,
) -> CoordinationSolution:
    if perception is PerceptionMode.TRUTHFUL and reasoning is ReasoningMode.CANONICAL:
        return CoordinationSolution.TRUTHFUL
    if perception is PerceptionMode.ERRONEOUS and reasoning is ReasoningMode.COMPENSATOR:
        return CoordinationSolution.COMPENSATORY
    return CoordinationSolution.MISMATCHED


@dataclass(frozen=True, slots=True)
class SymbolicOutcome:
    """One immutable execution record for a concrete action pair."""

    perception_mode: PerceptionMode
    reasoning_mode: ReasoningMode
    reward: float
    solution: CoordinationSolution

    def __post_init__(self) -> None:
        perception = _perception_mode(self.perception_mode)
        reasoning = _reasoning_mode(self.reasoning_mode)
        object.__setattr__(self, "perception_mode", perception)
        object.__setattr__(self, "reasoning_mode", reasoning)
        object.__setattr__(self, "reward", _finite_real(self.reward, "reward"))
        try:
            solution = CoordinationSolution(self.solution)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid coordination solution: {self.solution!r}") from error
        if solution is not _solution_for(perception, reasoning):
            raise ValueError("solution is inconsistent with the executed action pair")
        object.__setattr__(self, "solution", solution)

    @property
    def is_coordinated(self) -> bool:
        """Whether the pair is one of the two high-reward coordinated modes."""

        return self.solution is not CoordinationSolution.MISMATCHED


@dataclass(frozen=True, slots=True)
class SymbolicGame:
    """Closed 2x2 game with executable and distributional reward APIs."""

    delta: float
    epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", _mismatch_penalty(self.delta, "delta"))
        object.__setattr__(self, "epsilon", _mismatch_penalty(self.epsilon, "epsilon"))

    def expected_reward(self, p_truthful: object, q_canonical: object) -> float:
        """Return expected reward for independent mixed perception/reasoning policies."""

        checked_p = _probability(p_truthful, "p_truthful")
        checked_q = _probability(q_canonical, "q_canonical")
        params = CoordinationParams(delta=self.delta, epsilon=self.epsilon)
        result = coordination_reward(checked_p, checked_q, params)
        if not isinstance(result, float):  # pragma: no cover - scalar inputs guarantee this
            raise RuntimeError("coordination reward returned a non-scalar result")
        return result

    def reward(
        self,
        perception_mode: PerceptionMode | str,
        reasoning_mode: ReasoningMode | str,
    ) -> float:
        """Return the reward for one pure perception/reasoning action pair."""

        perception = _perception_mode(perception_mode)
        reasoning = _reasoning_mode(reasoning_mode)
        return self.expected_reward(
            float(perception is PerceptionMode.TRUTHFUL),
            float(reasoning is ReasoningMode.CANONICAL),
        )

    def execute(
        self,
        perception_mode: PerceptionMode | str,
        reasoning_mode: ReasoningMode | str,
    ) -> SymbolicOutcome:
        """Execute a pure action pair and retain its semantic interpretation."""

        perception = _perception_mode(perception_mode)
        reasoning = _reasoning_mode(reasoning_mode)
        return SymbolicOutcome(
            perception_mode=perception,
            reasoning_mode=reasoning,
            reward=self.reward(perception, reasoning),
            solution=_solution_for(perception, reasoning),
        )

    @property
    def reward_matrix(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return rows ``(T, E)`` by columns ``(C, K)`` as immutable tuples."""

        return tuple(
            tuple(self.execute(perception, reasoning).reward for reasoning in ReasoningMode)
            for perception in PerceptionMode
        )  # type: ignore[return-value]


__all__ = [
    "CoordinationSolution",
    "PerceptionMode",
    "ReasoningMode",
    "SymbolicGame",
    "SymbolicOutcome",
]
