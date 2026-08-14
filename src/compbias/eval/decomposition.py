"""Additive and Bregman decompositions of outcome error."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class AdditiveErrors:
    y_true: float
    y_perceived: float
    y_model: float
    e_p: float
    e_r: float
    e_outcome: float


@dataclass(frozen=True, slots=True)
class SquaredDecomposition:
    l_p: float
    l_r: float
    coupling: float
    l_outcome: float
    per_sample_outcome: np.ndarray
    per_sample_identity_rhs: np.ndarray

    def __post_init__(self) -> None:
        outcome = np.array(self.per_sample_outcome, dtype=np.float64, copy=True)
        identity_rhs = np.array(self.per_sample_identity_rhs, dtype=np.float64, copy=True)
        outcome.setflags(write=False)
        identity_rhs.setflags(write=False)
        object.__setattr__(self, "per_sample_outcome", outcome)
        object.__setattr__(self, "per_sample_identity_rhs", identity_rhs)


@dataclass(frozen=True, slots=True)
class CouplingMetrics:
    l_p: float
    l_r: float
    coupling: float
    l_outcome: float
    normalized_cancellation: float
    n_samples: int


@dataclass(frozen=True, slots=True)
class BregmanDecomposition:
    outcome: float
    perception: float
    reasoning: float
    interaction: float


class DifferentiablePotential(Protocol):
    def value(self, x: np.ndarray) -> float: ...

    def gradient(self, x: np.ndarray) -> np.ndarray: ...


def _numeric_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def additive_errors(
    *,
    scene: Mapping[str, object],
    perceived_scene: Mapping[str, object],
    reasoning_action: Mapping[str, object],
    solver: Callable[..., object],
) -> AdditiveErrors:
    """Decompose using the canonical perceived answer as the reasoning boundary."""

    if not callable(solver):
        raise TypeError("solver must be callable")
    y_true = _numeric_scalar(solver(dict(scene)), "canonical answer")
    y_perceived = _numeric_scalar(solver(dict(perceived_scene)), "perceived answer")
    y_model = _numeric_scalar(solver(dict(perceived_scene), dict(reasoning_action)), "model answer")
    e_p = y_perceived - y_true
    e_r = y_model - y_perceived
    return AdditiveErrors(
        y_true=y_true,
        y_perceived=y_perceived,
        y_model=y_model,
        e_p=e_p,
        e_r=e_r,
        e_outcome=y_model - y_true,
    )


def _matched_arrays(e_p: object, e_r: object) -> tuple[np.ndarray, np.ndarray]:
    perception = np.asarray(e_p, dtype=np.float64)
    reasoning = np.asarray(e_r, dtype=np.float64)
    if perception.shape != reasoning.shape:
        raise ValueError("perception and reasoning errors must have the same shape")
    if perception.size == 0:
        raise ValueError("error arrays must not be empty")
    if not np.all(np.isfinite(perception)) or not np.all(np.isfinite(reasoning)):
        raise ValueError("error arrays must contain only finite values")
    return perception.copy(), reasoning.copy()


def squared_decomposition(e_p: object, e_r: object) -> SquaredDecomposition:
    """Compute ``E[(e_p + e_r)^2]`` and its signed coupling term."""

    perception, reasoning = _matched_arrays(e_p, e_r)
    perception_loss = perception * perception
    reasoning_loss = reasoning * reasoning
    cross = perception * reasoning
    outcome = (perception + reasoning) ** 2
    identity_rhs = perception_loss + reasoning_loss + 2.0 * cross
    return SquaredDecomposition(
        l_p=float(np.mean(perception_loss)),
        l_r=float(np.mean(reasoning_loss)),
        coupling=float(np.mean(cross)),
        l_outcome=float(np.mean(outcome)),
        per_sample_outcome=outcome,
        per_sample_identity_rhs=identity_rhs,
    )


def _record_mapping(record: object) -> Mapping[str, object]:
    if isinstance(record, Mapping):
        return record
    if is_dataclass(record) and not isinstance(record, type):
        return {field.name: getattr(record, field.name) for field in fields(record)}
    raise TypeError("each decomposition record must be a mapping or dataclass")


def coupling_metrics(records: Iterable[object]) -> CouplingMetrics:
    """Summarize squared-error coupling and its cosine-normalized cancellation."""

    rows = tuple(_record_mapping(record) for record in records)
    if not rows:
        raise ValueError("records must not be empty")
    e_p = [_numeric_scalar(row.get("e_p"), "e_p") for row in rows]
    e_r = [_numeric_scalar(row.get("e_r"), "e_r") for row in rows]
    result = squared_decomposition(e_p, e_r)
    denominator = math.sqrt(result.l_p * result.l_r)
    normalized = 0.0 if denominator == 0.0 else -result.coupling / denominator
    return CouplingMetrics(
        l_p=result.l_p,
        l_r=result.l_r,
        coupling=result.coupling,
        l_outcome=result.l_outcome,
        normalized_cancellation=float(normalized),
        n_samples=len(rows),
    )


def _point(value: object, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be non-empty and finite")
    return point.copy()


def _potential_value(phi: DifferentiablePotential, point: np.ndarray) -> float:
    return _numeric_scalar(phi.value(point.copy()), "potential value")


def _potential_gradient(phi: DifferentiablePotential, point: np.ndarray) -> np.ndarray:
    gradient = np.asarray(phi.gradient(point.copy()), dtype=np.float64)
    if gradient.shape != point.shape:
        raise ValueError("potential gradient must have the same shape as its input")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("potential gradient must be finite")
    return gradient.copy()


def _bregman(
    x: np.ndarray,
    y: np.ndarray,
    phi: DifferentiablePotential,
    *,
    phi_x: float | None = None,
    phi_y: float | None = None,
    grad_y: np.ndarray | None = None,
) -> float:
    x_value = _potential_value(phi, x) if phi_x is None else phi_x
    y_value = _potential_value(phi, y) if phi_y is None else phi_y
    y_gradient = _potential_gradient(phi, y) if grad_y is None else grad_y
    return float(x_value - y_value - np.vdot(y_gradient, x - y))


def bregman_decomposition(
    *,
    y_true: object,
    y_perceived: object,
    y_model: object,
    phi: DifferentiablePotential,
) -> BregmanDecomposition:
    """Return the signed Bregman three-point decomposition."""

    if not callable(getattr(phi, "value", None)) or not callable(getattr(phi, "gradient", None)):
        raise TypeError("phi must expose callable value and gradient methods")
    true = _point(y_true, "y_true")
    perceived = _point(y_perceived, "y_perceived")
    model = _point(y_model, "y_model")
    if true.shape != perceived.shape or true.shape != model.shape:
        raise ValueError("all Bregman points must have the same shape")

    phi_true = _potential_value(phi, true)
    phi_perceived = _potential_value(phi, perceived)
    phi_model = _potential_value(phi, model)
    grad_perceived = _potential_gradient(phi, perceived)
    grad_model = _potential_gradient(phi, model)

    outcome = _bregman(true, model, phi, phi_x=phi_true, phi_y=phi_model, grad_y=grad_model)
    perception = _bregman(
        true,
        perceived,
        phi,
        phi_x=phi_true,
        phi_y=phi_perceived,
        grad_y=grad_perceived,
    )
    reasoning = _bregman(
        perceived,
        model,
        phi,
        phi_x=phi_perceived,
        phi_y=phi_model,
        grad_y=grad_model,
    )
    interaction = float(np.vdot(true - perceived, grad_perceived - grad_model))
    return BregmanDecomposition(
        outcome=outcome,
        perception=perception,
        reasoning=reasoning,
        interaction=interaction,
    )
