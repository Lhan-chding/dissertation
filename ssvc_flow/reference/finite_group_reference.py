"""CPU mathematical reference. Not a model runner or a training implementation.

Order: X, S, W, I. Functions operate on a fixed group or a finite IID bank.
Averages over overlapping bank subsets do not create independent observations.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_counts(counts: Sequence[int]) -> tuple[int, int, int, int]:
    if len(counts) != 4 or any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("counts must be four nonnegative Python integers: X,S,W,I")
    if sum(counts) == 0:
        raise ValueError("empty bank")
    return tuple(counts)  # type: ignore[return-value]


def _choose(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def bank_group_rates(counts: Sequence[int], k: int) -> dict[str, float]:
    """Average group events over all size-k subsets of a bank, without replacement.

    If the n bank draws were IID from one fixed prompt policy, each expression
    is an unbiased estimator of that prompt's future size-k group probability.
    This does not make the overlapping subsets statistically independent.
    """
    nx, ns, nw, ni = _validate_counts(counts)
    n = nx + ns + nw + ni
    if type(k) is not int or not 1 <= k <= n:
        raise ValueError("require integer 1 <= k <= bank size; no extrapolation")
    nb, nv = ns + nw, nx + ns + nw
    den = _choose(n, k)
    c = lambda m: _choose(m, k) / den
    out = {
        "no_X": c(n - nx),
        "X_informative": 1.0 - c(nx) - c(n - nx),
        "validity_mixed": 1.0 - c(nv) - c(ni),
        "aux_only_activation": c(nb + ni) - c(nb) - c(ni),
        "three_level_X_B_I": (
            1.0 - c(nb + ni) - c(nx + ni) - c(nx + nb)
            + c(nx) + c(nb) + c(ni)
        ),
    }
    if any(v < -1e-12 or v > 1 + 1e-12 for v in out.values()):
        raise ArithmeticError("group probability outside [0,1]")
    return {name: max(0.0, min(1.0, value)) for name, value in out.items()}


def population_group_rates(p: Sequence[float], k: int) -> dict[str, float]:
    """Exact formula for ONE known prompt distribution, not pooled prompt means."""
    if len(p) != 4 or any(not math.isfinite(v) or v < 0 for v in p):
        raise ValueError("require four nonnegative finite probabilities")
    if abs(sum(p) - 1) > 1e-10 or type(k) is not int or k < 1:
        raise ValueError("probabilities must sum to 1 and k must be positive")
    x, s, w, i = p
    b, v = s + w, 1 - i
    return {
        "no_X": (1 - x) ** k,
        "X_informative": 1 - x**k - (1-x)**k,
        "validity_mixed": 1 - v**k - i**k,
        "aux_only_activation": (1-x)**k - b**k - i**k,
        "three_level_X_B_I": (
            1 - (1-x)**k - (1-b)**k - (1-i)**k + x**k + b**k + i**k
        ),
    }


def normalized_advantage_and_derivative(
    categories: Sequence[str], lam: float, alpha: float = 2.0,
    epsilon: float = 1e-4,
) -> tuple[list[float], list[float]]:
    """Exact fixed-group joint z-score and lambda derivative for alpha*X+lambda*V.

    At epsilon=0 and a zero-variance tie, A is defined to be zero but the
    derivative is generally not defined: return NaNs rather than inventing one.
    """
    if not categories or any(c not in {"X", "S", "W", "I"} for c in categories):
        raise ValueError("categories must be a nonempty X/S/W/I sequence")
    if not all(math.isfinite(v) for v in [lam, alpha, epsilon]):
        raise ValueError("finite parameters required")
    if lam < 0 or alpha <= 0 or epsilon < 0:
        raise ValueError("require lambda>=0, alpha>0, epsilon>=0")
    n = len(categories)
    valid = [float(c != "I") for c in categories]
    rewards = [alpha * (c == "X") + lam * v for c, v in zip(categories, valid)]
    mean_r, mean_v = sum(rewards)/n, sum(valid)/n
    rr = [r-mean_r for r in rewards]
    vv = [v-mean_v for v in valid]
    variance = sum(r*r for r in rr)/n
    sd = math.sqrt(variance + epsilon*epsilon)
    if sd == 0:
        # If validity is also constant, the group remains a tie for every lambda.
        if all(v == 0.0 for v in vv):
            return [0.0]*n, [0.0]*n
        return [0.0]*n, [math.nan]*n
    covariance = sum(r*v for r, v in zip(rr, vv))/n
    return ([r/sd for r in rr],
            [v/sd-r*covariance/(sd**3) for r, v in zip(rr, vv)])


def zero_epsilon_sensitivity_energy(categories: Sequence[str], lam: float,
                                    alpha: float = 2.0) -> float:
    """(1/K)||dA/dlambda||^2 = alpha^2*x*b*i / Var(r)^2, Var(r)>0."""
    _, deriv = normalized_advantage_and_derivative(categories, lam, alpha, 0.0)
    if any(math.isnan(v) for v in deriv):
        return math.nan
    n = len(categories)
    x = categories.count("X")/n
    i = categories.count("I")/n
    b = (categories.count("S") + categories.count("W"))/n
    v = 1-i
    var_r = alpha**2*x*(1-x) + 2*alpha*lam*x*i + lam**2*v*i
    if var_r == 0:
        return math.nan
    return alpha**2*x*b*i/(var_r**2)
