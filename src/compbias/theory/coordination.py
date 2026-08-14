"""Two-policy coordination dynamics and their qualitative phase structure.

The state ``(p, q)`` contains the probabilities that the perception and
reasoning modules choose their truth-aligned actions.  The unregularized
dynamics are the two-population replicator equations for the reward matrix

``[[1, 1 - epsilon], [1 - delta, 1]]``.

Optional KL penalties pull each policy towards an interior reference policy.
The implementation evaluates the entropy-gradient contribution in a
boundary-safe form, so the closed unit square remains an invariant domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit, logit, xlogy

Stability = Literal["stable", "unstable", "saddle", "critical"]

__all__ = [
    "BasinMap",
    "BifurcationBranch",
    "CoordinationParams",
    "FixedPoint",
    "PhasePortrait",
    "basin_map",
    "fixed_points",
    "jacobian",
    "phase_portrait",
    "reward",
    "symmetric_bifurcation_branch",
    "symmetric_bifurcation_root",
    "vector_field",
]


def _finite_scalar(name: str, value: object) -> float:
    """Return a finite real scalar, raising a stable public error otherwise."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        scalar = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real number")
    return scalar


def _immutable_array(values: ArrayLike, *, dtype: object) -> NDArray:
    """Make an owning, read-only array for an immutable result object."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CoordinationParams:
    """Parameters of the perception--reasoning coordination game."""

    delta: float
    epsilon: float
    beta_p: float = 0.0
    beta_q: float = 0.0
    p_ref: float = 0.5
    q_ref: float = 0.5

    def __post_init__(self) -> None:
        values = {
            name: _finite_scalar(name, getattr(self, name))
            for name in ("delta", "epsilon", "beta_p", "beta_q", "p_ref", "q_ref")
        }
        if values["delta"] <= 0.0 or values["epsilon"] <= 0.0:
            raise ValueError("delta and epsilon must be strictly positive")
        if values["beta_p"] < 0.0 or values["beta_q"] < 0.0:
            raise ValueError("KL coefficients must be nonnegative")
        if not 0.0 < values["p_ref"] < 1.0 or not 0.0 < values["q_ref"] < 1.0:
            raise ValueError("reference probabilities must lie strictly between zero and one")
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class FixedPoint:
    """A fixed state and its linear/local stability classification."""

    state: tuple[float, float]
    stability: Stability


@dataclass(frozen=True, slots=True)
class PhasePortrait:
    """Mesh-aligned samples of the coordination vector field."""

    p: NDArray[np.float64]
    q: NDArray[np.float64]
    dp: NDArray[np.float64]
    dq: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("p", "q", "dp", "dq"):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), dtype=np.float64),
            )


@dataclass(frozen=True, slots=True)
class BasinMap:
    """Mesh-aligned labels for the two attracting coordination outcomes."""

    p: NDArray[np.float64]
    q: NDArray[np.float64]
    labels: NDArray[np.str_]

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _immutable_array(self.p, dtype=np.float64))
        object.__setattr__(self, "q", _immutable_array(self.q, dtype=np.float64))
        object.__setattr__(self, "labels", _immutable_array(self.labels, dtype=str))


@dataclass(frozen=True, slots=True)
class BifurcationBranch:
    """Center and symmetry-related branches of the KL pitchfork."""

    beta_over_a: NDArray[np.float64]
    center: NDArray[np.float64]
    positive: NDArray[np.float64]
    negative: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("beta_over_a", "center", "positive", "negative"):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), dtype=np.float64),
            )


def _probability_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real probabilities") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite probabilities")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} probabilities must lie in the closed unit interval")
    return result


def _state_components(state: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = _probability_array("state", state)
    if values.ndim == 0 or values.shape[0] != 2:
        raise ValueError("state must have leading dimension two")
    return values[0], values[1]


def _scalar_state(state: ArrayLike) -> tuple[float, float]:
    values = _probability_array("state", state)
    if values.shape != (2,):
        raise ValueError("state must be a length-two vector")
    return float(values[0]), float(values[1])


def _logit_flow(probability: NDArray[np.float64], reference: float) -> NDArray[np.float64]:
    """Evaluate ``x(1-x)(logit(x)-logit(reference))`` without 0*inf."""
    complement = 1.0 - probability
    entropy_gradient = complement * xlogy(probability, probability) - probability * xlogy(
        complement, complement
    )
    return entropy_gradient - probability * complement * float(logit(reference))


def _field_components(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    params: CoordinationParams,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    interaction = params.delta + params.epsilon
    p_variance = p * (1.0 - p)
    q_variance = q * (1.0 - q)
    dp = p_variance * (interaction * q - params.delta)
    dq = q_variance * (interaction * p - params.epsilon)
    if params.beta_p:
        dp = dp - params.beta_p * _logit_flow(p, params.p_ref)
    if params.beta_q:
        dq = dq - params.beta_q * _logit_flow(q, params.q_ref)
    return np.asarray(dp, dtype=np.float64), np.asarray(dq, dtype=np.float64)


def reward(
    p: ArrayLike,
    q: ArrayLike,
    params: CoordinationParams,
) -> float | NDArray[np.float64]:
    """Return the expected outcome reward at state ``(p, q)``."""
    p_values = _probability_array("p", p)
    q_values = _probability_array("q", q)
    try:
        p_values, q_values = np.broadcast_arrays(p_values, q_values)
    except ValueError as error:
        raise ValueError("p and q must be broadcast-compatible") from error
    result = (
        1.0
        - params.delta * p_values * (1.0 - q_values)
        - params.epsilon * (1.0 - p_values) * q_values
    )
    if result.ndim == 0:
        return float(result)
    return np.array(result, dtype=np.float64, copy=True)


def vector_field(
    _time: float,
    state: ArrayLike,
    params: CoordinationParams,
) -> NDArray[np.float64]:
    """Evaluate the KL-regularized replicator vector field."""
    p, q = _state_components(state)
    dp, dq = _field_components(p, q, params)
    return np.stack((dp, dq), axis=0)


def _regularized_diagonal(
    probability: float,
    other: float,
    mismatch: float,
    beta: float,
    reference: float,
    interaction: float,
) -> float:
    if not beta:
        advantage = interaction * other - mismatch
        return (1.0 - 2.0 * probability) * advantage
    if probability in (0.0, 1.0):
        return np.inf
    advantage = (
        interaction * other
        - mismatch
        - beta * (float(logit(probability)) - float(logit(reference)))
    )
    return (1.0 - 2.0 * probability) * advantage - beta


def jacobian(state: ArrayLike, params: CoordinationParams) -> NDArray[np.float64]:
    """Return the analytic state Jacobian of :func:`vector_field`.

    A positive infinity on a KL-regularized policy boundary records the true
    one-sided divergent derivative of the entropy-gradient flow.
    """
    p, q = _scalar_state(state)
    interaction = params.delta + params.epsilon
    return np.array(
        [
            [
                _regularized_diagonal(
                    p,
                    q,
                    params.delta,
                    params.beta_p,
                    params.p_ref,
                    interaction,
                ),
                p * (1.0 - p) * interaction,
            ],
            [
                q * (1.0 - q) * interaction,
                _regularized_diagonal(
                    q,
                    p,
                    params.epsilon,
                    params.beta_q,
                    params.q_ref,
                    interaction,
                ),
            ],
        ],
        dtype=np.float64,
    )


def symmetric_bifurcation_root(beta_over_a: float) -> float:
    """Return the positive nonzero branch of ``2 r atanh(m) = m``.

    The branch reaches the boundary ``m=1`` at zero regularization and merges
    continuously into the center root at the critical ratio ``r=1/2``.
    """
    ratio = _finite_scalar("beta_over_a", beta_over_a)
    if ratio < 0.0:
        raise ValueError("beta_over_a must be nonnegative")
    if ratio == 0.0:
        return 1.0
    if ratio >= 0.5:
        return 0.0
    if ratio <= 0.025:
        # The mathematical root is closer to one than float64 can represent.
        return 1.0

    inverse_temperature = 1.0 / (2.0 * ratio)

    def residual(logit_half: float) -> float:
        if logit_half == 0.0:
            return 1.0 - 2.0 * ratio
        return float(np.tanh(logit_half) / logit_half - 2.0 * ratio)

    logit_half = brentq(
        residual,
        0.0,
        inverse_temperature,
        xtol=5e-15,
        rtol=1e-14,
    )
    return float(np.tanh(logit_half))


def symmetric_bifurcation_branch(beta_over_a: ArrayLike) -> BifurcationBranch:
    """Evaluate all three symmetric bifurcation branches on a ratio vector."""
    ratios = np.asarray(beta_over_a, dtype=np.float64)
    if ratios.ndim != 1:
        raise ValueError("beta_over_a must be a one-dimensional vector")
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("beta_over_a must contain finite nonnegative ratios")
    positive = np.fromiter(
        (symmetric_bifurcation_root(float(ratio)) for ratio in ratios),
        dtype=np.float64,
        count=ratios.size,
    )
    center = np.zeros_like(ratios)
    return BifurcationBranch(
        beta_over_a=ratios,
        center=center,
        positive=positive,
        negative=-positive,
    )


def _is_symmetric(params: CoordinationParams) -> bool:
    return (
        params.delta == params.epsilon
        and params.beta_p == params.beta_q
        and params.p_ref == 0.5
        and params.q_ref == 0.5
    )


def _coupled_q(p: ArrayLike, params: CoordinationParams) -> NDArray[np.float64]:
    argument = (
        float(logit(params.q_ref))
        + ((params.delta + params.epsilon) * np.asarray(p) - params.epsilon) / params.beta_q
    )
    return np.asarray(expit(argument), dtype=np.float64)


def _reduced_residual(p: ArrayLike, params: CoordinationParams) -> NDArray[np.float64]:
    p_values = np.asarray(p, dtype=np.float64)
    q_values = _coupled_q(p_values, params)
    argument = (
        float(logit(params.p_ref))
        + ((params.delta + params.epsilon) * q_values - params.delta) / params.beta_p
    )
    return p_values - np.asarray(expit(argument), dtype=np.float64)


def _reduced_derivative(p: ArrayLike, params: CoordinationParams) -> NDArray[np.float64]:
    p_values = np.asarray(p, dtype=np.float64)
    q_values = _coupled_q(p_values, params)
    mapped_p = p_values - _reduced_residual(p_values, params)
    interaction = params.delta + params.epsilon
    slope = interaction * interaction / (params.beta_p * params.beta_q)
    return 1.0 - slope * mapped_p * (1.0 - mapped_p) * q_values * (1.0 - q_values)


def _bracket_roots(
    grid: NDArray[np.float64],
    values: NDArray[np.float64],
    function: object,
) -> list[float]:
    roots = [float(grid[index]) for index in np.flatnonzero(values == 0.0)]
    for index in np.flatnonzero(values[:-1] * values[1:] < 0.0):
        left, right = float(grid[index]), float(grid[index + 1])
        roots.append(float(brentq(function, left, right, xtol=5e-15, rtol=1e-14)))  # type: ignore[arg-type]
    return roots


def _general_coupled_interior(params: CoordinationParams) -> list[tuple[float, float]]:
    grid = np.linspace(0.0, 1.0, 4097)
    residuals = _reduced_residual(grid, params)

    def residual_function(value: float) -> float:
        return float(_reduced_residual(value, params))

    roots = _bracket_roots(grid, residuals, residual_function)

    derivatives = _reduced_derivative(grid, params)

    def derivative_function(value: float) -> float:
        return float(_reduced_derivative(value, params))

    for critical in _bracket_roots(grid, derivatives, derivative_function):
        if abs(residual_function(critical)) <= 1e-10:
            roots.append(critical)

    states: list[tuple[float, float]] = []
    for p in sorted(roots):
        q = float(_coupled_q(p, params))
        if 0.0 < p < 1.0 and 0.0 < q < 1.0:
            _append_unique(states, (p, q))
    return states


def _interior_states(params: CoordinationParams) -> list[tuple[float, float]]:
    interaction = params.delta + params.epsilon
    if params.beta_p == 0.0 and params.beta_q == 0.0:
        return [(params.epsilon / interaction, params.delta / interaction)]
    if params.beta_p == 0.0:
        q = params.delta / interaction
        p = (
            params.epsilon + params.beta_q * (float(logit(q)) - float(logit(params.q_ref)))
        ) / interaction
        return [(p, q)] if 0.0 < p < 1.0 else []
    if params.beta_q == 0.0:
        p = params.epsilon / interaction
        q = (
            params.delta + params.beta_p * (float(logit(p)) - float(logit(params.p_ref)))
        ) / interaction
        return [(p, q)] if 0.0 < q < 1.0 else []
    if _is_symmetric(params):
        root = symmetric_bifurcation_root(params.beta_p / params.delta)
        if root == 0.0:
            return [(0.5, 0.5)]
        return [
            ((1.0 - root) / 2.0, (1.0 - root) / 2.0),
            (0.5, 0.5),
            ((1.0 + root) / 2.0, (1.0 + root) / 2.0),
        ]
    return _general_coupled_interior(params)


def _edge_states(params: CoordinationParams) -> list[tuple[float, float]]:
    interaction = params.delta + params.epsilon
    states: list[tuple[float, float]] = []
    if params.beta_p:
        base = float(logit(params.p_ref))
        for q in (0.0, 1.0):
            p = float(expit(base + (interaction * q - params.delta) / params.beta_p))
            if 0.0 < p < 1.0:
                states.append((p, q))
    if params.beta_q:
        base = float(logit(params.q_ref))
        for p in (0.0, 1.0):
            q = float(expit(base + (interaction * p - params.epsilon) / params.beta_q))
            if 0.0 < q < 1.0:
                states.append((p, q))
    return states


def _append_unique(
    states: list[tuple[float, float]],
    candidate: tuple[float, float],
) -> None:
    if not any(np.allclose(candidate, state, atol=1e-11, rtol=0.0) for state in states):
        states.append((float(candidate[0]), float(candidate[1])))


def _one_sided_jacobian(state: tuple[float, float], params: CoordinationParams) -> NDArray:
    point = np.asarray(state, dtype=np.float64)
    baseline = vector_field(0.0, point, params)
    result = np.empty((2, 2), dtype=np.float64)
    maximum_step = 1e-7
    for column in range(2):
        shifted = point.copy()
        if point[column] == 0.0:
            shifted[column] = maximum_step
            result[:, column] = (vector_field(0.0, shifted, params) - baseline) / maximum_step
        elif point[column] == 1.0:
            shifted[column] = 1.0 - maximum_step
            result[:, column] = (baseline - vector_field(0.0, shifted, params)) / maximum_step
        else:
            step = min(maximum_step, point[column] / 2.0, (1.0 - point[column]) / 2.0)
            shifted[column] += step
            upper = vector_field(0.0, shifted, params)
            shifted[column] -= 2.0 * step
            lower = vector_field(0.0, shifted, params)
            result[:, column] = (upper - lower) / (2.0 * step)
    return result


def _classify(state: tuple[float, float], params: CoordinationParams) -> Stability:
    matrix = jacobian(state, params)
    if not np.all(np.isfinite(matrix)):
        matrix = _one_sided_jacobian(state, params)
    real_parts = np.real(np.linalg.eigvals(matrix))
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(real_parts))))
    if np.max(real_parts) < -tolerance:
        return "stable"
    if np.min(real_parts) > tolerance:
        return "unstable"
    if np.min(real_parts) < -tolerance and np.max(real_parts) > tolerance:
        return "saddle"
    return "critical"


def fixed_points(params: CoordinationParams) -> tuple[FixedPoint, ...]:
    """Enumerate boundary and interior fixed points on the closed unit square."""
    states = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    for state in (*_edge_states(params), *_interior_states(params)):
        _append_unique(states, state)
    return tuple(FixedPoint(state=state, stability=_classify(state, params)) for state in states)


def _grid_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    result = _probability_array(name, values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return np.array(result, dtype=np.float64, copy=True)


def phase_portrait(
    p_values: ArrayLike,
    q_values: ArrayLike,
    params: CoordinationParams,
) -> PhasePortrait:
    """Sample the vector field on the Cartesian product of two policy grids."""
    p_grid = _grid_vector("p_values", p_values)
    q_grid = _grid_vector("q_values", q_values)
    p_mesh, q_mesh = np.meshgrid(p_grid, q_grid, indexing="xy")
    dp, dq = _field_components(p_mesh, q_mesh, params)
    return PhasePortrait(p=p_mesh, q=q_mesh, dp=dp, dq=dq)


def _has_symmetric_separatrix(params: CoordinationParams) -> bool:
    if not (
        params.delta == params.epsilon
        and params.beta_p == params.beta_q
        and params.q_ref == 1.0 - params.p_ref
    ):
        return False
    center = min(
        fixed_points(params),
        key=lambda point: np.linalg.norm(np.asarray(point.state) - 0.5),
    )
    return center.stability in {"saddle", "critical"}


def _attractor_label(state: tuple[float, float], tolerance: float) -> str:
    score = state[0] + state[1] - 1.0
    if score > tolerance:
        return "truthful"
    if score < -tolerance:
        return "compensatory"
    return "separatrix"


def _integrate_endpoint(
    initial: tuple[float, float],
    params: CoordinationParams,
    horizon: float,
) -> tuple[float, float]:
    solution = solve_ivp(
        vector_field,
        (0.0, horizon),
        np.asarray(initial, dtype=np.float64),
        args=(params,),
        rtol=1e-9,
        atol=1e-11,
        max_step=min(0.1, horizon),
    )
    if not solution.success:
        raise RuntimeError(f"coordination integration failed: {solution.message}")
    endpoint = np.clip(solution.y[:, -1], 0.0, 1.0)
    return float(endpoint[0]), float(endpoint[1])


def basin_map(
    p_values: ArrayLike,
    q_values: ArrayLike,
    params: CoordinationParams,
    *,
    horizon: float = 30.0,
    separatrix_tolerance: float = 1e-8,
) -> BasinMap:
    """Integrate a grid of initial states and label their limiting basins."""
    duration = _finite_scalar("horizon", horizon)
    tolerance = _finite_scalar("separatrix_tolerance", separatrix_tolerance)
    if duration <= 0.0:
        raise ValueError("horizon must be strictly positive")
    if tolerance < 0.0:
        raise ValueError("separatrix_tolerance must be nonnegative")

    p_grid = _grid_vector("p_values", p_values)
    q_grid = _grid_vector("q_values", q_values)
    p_mesh, q_mesh = np.meshgrid(p_grid, q_grid, indexing="xy")
    points = fixed_points(params)
    attractors = tuple(point for point in points if point.stability == "stable")
    symmetric_separatrix = _has_symmetric_separatrix(params)
    labels = np.empty(p_mesh.shape, dtype="<U12")

    for index in np.ndindex(p_mesh.shape):
        initial = (float(p_mesh[index]), float(q_mesh[index]))
        if symmetric_separatrix and abs(sum(initial) - 1.0) <= tolerance:
            labels[index] = "separatrix"
            continue
        endpoint = _integrate_endpoint(initial, params, duration)
        if not attractors:
            labels[index] = "separatrix"
            continue
        destination = min(
            attractors,
            key=lambda point: np.linalg.norm(np.subtract(endpoint, point.state)),
        )
        labels[index] = _attractor_label(destination.state, tolerance)
    return BasinMap(p=p_mesh, q=q_mesh, labels=labels)
