"""Outcome loss must decompose into local errors and their coupling."""

from dataclasses import dataclass

import numpy as np
import pytest

from compbias.eval.decomposition import (
    additive_errors,
    bregman_decomposition,
    coupling_metrics,
    squared_decomposition,
)


def _solver(state: dict[str, float], action: dict[str, float] | None = None) -> float:
    operation = action or {"operation": "add", "operand": 3}
    if operation["operation"] != "add":
        raise ValueError("unsupported operation")
    return state["value"] + operation["operand"]


def test_additive_errors_use_canonical_perceived_answer_as_the_boundary() -> None:
    errors = additive_errors(
        scene={"value": 7},
        perceived_scene={"value": 8},
        reasoning_action={"operation": "add", "operand": 2},
        solver=_solver,
    )

    assert errors.y_true == 10
    assert errors.y_perceived == 11
    assert errors.y_model == 10
    assert errors.e_p == 1
    assert errors.e_r == -1
    assert errors.e_outcome == 0
    assert errors.e_outcome == errors.e_p + errors.e_r


def test_fixed_coupling_fixture_cancels_exactly_per_sample_and_in_aggregate() -> None:
    e_p = np.asarray([1.0, -1.0])
    e_r = np.asarray([-1.0, 1.0])

    result = squared_decomposition(e_p, e_r)

    assert result.l_p == pytest.approx(1.0)
    assert result.l_r == pytest.approx(1.0)
    assert result.coupling == pytest.approx(-1.0)
    assert result.l_outcome == pytest.approx(0.0)
    assert result.l_outcome == pytest.approx(result.l_p + result.l_r + 2 * result.coupling)
    assert np.allclose(result.per_sample_outcome, result.per_sample_identity_rhs)


def test_coupling_metrics_report_normalized_cancellation() -> None:
    records = (
        {"sample_id": "a", "e_p": 1.0, "e_r": -1.0},
        {"sample_id": "b", "e_p": -1.0, "e_r": 1.0},
    )

    metrics = coupling_metrics(records)

    assert metrics.l_p == pytest.approx(1.0)
    assert metrics.l_r == pytest.approx(1.0)
    assert metrics.coupling == pytest.approx(-1.0)
    assert metrics.normalized_cancellation == pytest.approx(1.0)
    assert metrics.l_outcome == pytest.approx(0.0)


@dataclass(frozen=True)
class QuadraticPotential:
    def value(self, x: np.ndarray) -> float:
        return float(0.5 * np.dot(x, x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return x.copy()


def test_bregman_three_point_identity_has_signed_interaction() -> None:
    result = bregman_decomposition(
        y_true=np.asarray([0.0]),
        y_perceived=np.asarray([1.0]),
        y_model=np.asarray([0.0]),
        phi=QuadraticPotential(),
    )

    assert result.outcome == pytest.approx(0.0)
    assert result.perception == pytest.approx(0.5)
    assert result.reasoning == pytest.approx(0.5)
    assert result.interaction == pytest.approx(-1.0)
    assert result.outcome == pytest.approx(
        result.perception + result.reasoning + result.interaction
    )


def test_zero_local_losses_have_defined_zero_normalized_cancellation() -> None:
    metrics = coupling_metrics(({"sample_id": "a", "e_p": 0.0, "e_r": 0.0},))

    assert metrics.normalized_cancellation == 0.0
    assert np.isfinite(metrics.normalized_cancellation)
