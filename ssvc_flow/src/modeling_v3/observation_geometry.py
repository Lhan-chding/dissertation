"""V3 signed observations with lossless raw records and explicit joint covariance.

This finite-observation module never receives a complete action distribution or
an exact event response. Inputs are paid, identified, per-draw contributions for
ONE prompt. Candidate-major flattening keeps shared baseline, proposal and alias
correlations. Covariance is per draw unless the attribute says ``of_mean``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

EVENTS = ("X", "S", "W", "I")
METHODS = (
    "RAW4",
    "EQUAL_ZERO_SUM",
    "DROP_W",
    "PRESERVE_XI",
    "PILOT_COV_ZERO_SUM",
    "PILOT_SHRINK_ZERO_SUM",
    "CROSSFIT_COV_ZERO_SUM",
    "KNOWN_COV_ORACLE",
)
SIMPLE_METHODS = METHODS[:4]


def _readonly(value, dtype=float):
    out = np.array(value, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


ONES = _readonly(np.ones(4))
B_EQUAL = _readonly(ONES / 4)
B_DROP_W = _readonly([0.0, 0.0, 1.0, 0.0])
B_PRESERVE_XI = _readonly([0.0, 0.5, 0.5, 0.0])


def _finite(value, *, name="values"):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f"Finite {name} required")
    return value


def _events(value):
    value = _finite(value, name="four-event values")
    if value.ndim < 1 or value.shape[-1] != 4:
        raise ValueError("Final axis must be the four X/S/W/I events")
    return value


def helmert():
    """4 by 3 orthonormal zero-mass basis; never silently applied to RAW4."""
    return np.array(
        [
            [1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
            [-1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
            [0.0, -2 / math.sqrt(6), 1 / math.sqrt(12)],
            [0.0, 0.0, -3 / math.sqrt(12)],
        ]
    )


def encode_raw(raw):
    """Return three Helmert coordinates AND the separately preserved raw mass."""
    raw = _events(raw)
    return raw @ helmert(), raw.sum(axis=-1)


def decode_raw(coordinates, mass_residual):
    coordinates = _finite(coordinates, name="Helmert coordinates")
    mass = _finite(mass_residual, name="raw mass residual")
    if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
        raise ValueError("Three Helmert coordinates required")
    if mass.shape != coordinates.shape[:-1]:
        raise ValueError("Raw mass must align with the Helmert coordinates")
    return coordinates @ helmert().T + mass[..., None] / 4


def _log_values(*values):
    values = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in values])
    for value in values:
        if np.any(np.isnan(value) | np.isposinf(value)):
            raise ValueError("Log probabilities cannot contain NaN or positive infinity")
        if np.any(value > 1e-12):
            raise ValueError("Normalized log probabilities must be nonpositive")
    return values


def stable_pair_weight(log_u, log_b):
    """(pi_u-pi_b)/((pi_u+pi_b)/2), bounded by TWO, without tiny subtraction.

    At a point with zero probability under both endpoints, the contrast weight
    is defined as zero; an actual sampler cannot return that action.
    """
    u, b = _log_values(log_u, log_b)
    with np.errstate(invalid="ignore", over="ignore"):
        difference = u - b
        result = 2 * np.tanh(difference / 2)
    return np.where(np.isneginf(u) & np.isneginf(b), 0.0, result)


def stable_origin_weight(log_u, log_b, log_origin):
    """Unclipped signed ORIGIN weight using signed log/expm1 arithmetic.

    This verifies support at scored actions only. The collection service must
    separately certify global contrast support before a ContributionBatch may
    enter a primary estimator. Nonfinite weights fail, never clip or normalize.
    """
    u, b, origin = _log_values(log_u, log_b, log_origin)
    if np.isneginf(origin).any():
        raise ValueError("SUPPORT_MISMATCH: sampled action has zero proposal probability")
    same = u == b
    maximum = np.maximum(u, b)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        delta = u - b
        # Combining in log space also handles representable differences of two
        # ratios that would each overflow on exponentiation.
        log_abs = maximum - origin + np.log(-np.expm1(-np.abs(delta)))
        result = np.sign(delta) * np.exp(log_abs)
    result = np.where(same, 0.0, result)
    if not np.isfinite(result).all():
        raise FloatingPointError("NONFINITE_WEIGHT: no clipping permitted")
    return result


def stable_three_policy_weight(log_probabilities, *, candidate_index=1, baseline_index=0):
    """Equal THREE-policy mixture weight; its bound is three, not two."""
    (logs,) = _log_values(log_probabilities)
    if logs.ndim < 1 or logs.shape[-1] != 3:
        raise ValueError("Exactly three policy log probabilities on the final axis required")
    if (
        type(candidate_index) is not int
        or type(baseline_index) is not int
        or candidate_index not in range(3)
        or baseline_index not in range(3)
    ):
        raise ValueError("Three-policy endpoint indices must be integers in [0,2]")
    maximum = logs.max(axis=-1)
    if np.isneginf(maximum).any():
        raise ValueError("SUPPORT_MISMATCH: all three policies assign zero probability")
    shifted = logs - maximum[..., None]
    denominator = np.exp(shifted).sum(axis=-1)
    u, b = shifted[..., candidate_index], shifted[..., baseline_index]
    with np.errstate(invalid="ignore"):
        difference = u - b
        numerator = np.sign(difference) * np.exp(np.maximum(u, b)) * (-np.expm1(-abs(difference)))
    numerator = np.where(u == b, 0.0, numerator)
    return 3 * numerator / denominator


def sample_mixture_sources(n, components, rng):
    """One independent uniform component per sequence, not fixed-size quotas."""
    if type(n) is not int or n < 1 or components not in (2, 3):
        raise ValueError("Positive draw count and two or three mixture components required")
    source = np.asarray(rng.integers(0, components, size=n))
    if (
        source.shape != (n,)
        or source.dtype.kind not in "iu"
        or np.any((source < 0) | (source >= components))
    ):
        raise ValueError("RNG returned invalid component identities")
    return source


def event_contributions(weights, labels):
    """One known X/S/W/I label per sample, including failed and truncated actions."""
    weights = _finite(weights, name="unclipped weights")
    labels = np.asarray(labels)
    if (
        labels.shape != weights.shape
        or labels.dtype.kind not in "iu"
        or np.any((labels < 0) | (labels > 3))
    ):
        raise ValueError(
            "One exhaustive X/S/W/I integer label per weight required; no dropped samples"
        )
    return weights[..., None] * np.eye(4)[labels]


def fixed_coefficient(method):
    if method not in SIMPLE_METHODS:
        raise ValueError("Fixed coefficient requires a simple observation method")
    return {
        "RAW4": np.zeros(4),
        "EQUAL_ZERO_SUM": B_EQUAL,
        "DROP_W": B_DROP_W,
        "PRESERVE_XI": B_PRESERVE_XI,
    }[method].copy()


def _coefficient(b):
    b = _events(b)
    if b.ndim not in (1, 2):
        raise ValueError("Coefficient shape must be (4,) or (contrasts,4)")
    sums = b.sum(axis=-1)
    zero = np.all(b == 0, axis=-1)
    if np.any(~zero & ~np.isclose(sums, 1, rtol=0, atol=1e-10)):
        raise ValueError("Correction coefficients must sum to one, or be all zero for RAW4")
    return b


def correction(z, b):
    z, b = _events(z), _coefficient(b)
    if b.ndim == 2 and (z.ndim < 2 or z.shape[-2] != b.shape[0]):
        raise ValueError("Per-contrast coefficients must align with contributions")
    return z - z.sum(axis=-1, keepdims=True) * b


def _covariance(covariance):
    cov = _finite(covariance, name="covariance")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not len(cov):
        raise ValueError("Nonempty square covariance required")
    scale = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if not np.allclose(cov, cov.T, atol=scale * 1e-10, rtol=1e-10):
        raise ValueError("Symmetric covariance required")
    cov = (cov + cov.T) / 2
    if np.linalg.eigvalsh(cov).min() < -scale * 1e-9:
        raise ValueError("Covariance must be positive semidefinite")
    return cov


def block_transform(b, contrasts=None):
    b = _coefficient(b)
    if b.ndim == 1:
        contrasts = 1 if contrasts is None else contrasts
        if type(contrasts) is not int or contrasts < 1:
            raise ValueError("Positive contrast count required")
        b = np.broadcast_to(b, (contrasts, 4))
    elif contrasts is not None and contrasts != len(b):
        raise ValueError("Coefficient/contrast count mismatch")
    out = np.zeros((4 * len(b), 4 * len(b)))
    for j, row in enumerate(b):
        out[4 * j : 4 * j + 4, 4 * j : 4 * j + 4] = np.eye(4) - np.outer(row, ONES)
    return out


def covariance_after(cov, b, n=1):
    """Propagate a per-draw JOINT covariance; divide by n exactly once."""
    if type(n) is not int or n < 1:
        raise ValueError("Positive integer main sample count required")
    cov = _covariance(cov)
    if len(cov) % 4:
        raise ValueError("Covariance dimensions must be candidate-major multiples of four")
    transform = block_transform(b, contrasts=len(cov) // 4)
    return transform @ cov @ transform.T / n


def joint_sample_covariance(contributions):
    """Unbiased per-draw joint covariance; one draw cannot estimate covariance."""
    z = _events(contributions)
    if z.ndim != 3 or not len(z) or not z.shape[1]:
        raise ValueError("Nonempty contributions with shape (draws,contrasts,4) required")
    if len(z) < 2:
        return None
    flat = z.reshape(len(z), -1)
    centered = flat - flat.mean(axis=0)
    result = centered.T @ centered / (len(z) - 1)
    if not np.isfinite(result).all():
        raise FloatingPointError("NONFINITE_SECOND_MOMENT: no primary clipping permitted")
    return result


@dataclass(frozen=True)
class ContributionBatch:
    """Identified finite draws at one prompt from a common sampling experiment.

    ``token_ids`` stores the complete sampled action (an empty action is valid).
    ``sample_ids`` identify RNG draws, not unique token strings: independent
    samples may contain identical tokens. ``policy_pairs`` contain canonical
    baseline/candidate scoring-policy fingerprints. Exact aliases and reversed
    contrasts must reuse identical/opposite contributions, with covariance kept.
    Support and score/sampler normalization are collection-service obligations;
    ``VERIFIED`` means those receipts were checked by that service, not inferred
    from this finite sample. RNG stream identities must name genuine independent
    streams; assigning different strings to reused random draws is not valid.
    """

    contributions: np.ndarray
    sample_ids: tuple
    rng_stream_id: str
    prompt_id: str
    contrast_ids: tuple
    policy_pairs: tuple
    token_ids: tuple
    proposal_kind: str
    support_status: str = "VERIFIED"

    def __post_init__(self):
        z = _events(self.contributions)
        if z.ndim != 3 or not len(z) or not z.shape[1]:
            raise ValueError("contributions must have nonempty (draws,contrasts,4) shape")
        n, c, _ = z.shape
        for field in ("sample_ids", "contrast_ids"):
            values = tuple(getattr(self, field))
            expected = n if field == "sample_ids" else c
            if (
                len(values) != expected
                or any(not isinstance(x, str) or not x for x in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"Unique aligned nonempty {field} required")
            object.__setattr__(self, field, values)
        if (
            not isinstance(self.rng_stream_id, str)
            or not self.rng_stream_id
            or not isinstance(self.prompt_id, str)
            or not self.prompt_id
        ):
            raise ValueError("Explicit prompt and RNG stream identities required")
        pairs = tuple(tuple(pair) for pair in self.policy_pairs)
        if len(pairs) != c or any(
            len(pair) != 2 or any(not isinstance(x, str) or not x for x in pair) for pair in pairs
        ):
            raise ValueError("Canonical baseline/candidate policy identity per contrast required")
        tokens = tuple(tuple(row) for row in self.token_ids)
        if len(tokens) != n or any(
            any(not isinstance(t, (int, np.integer)) or isinstance(t, bool) or t < 0 for t in row)
            for row in tokens
        ):
            raise ValueError("Complete nonnegative integer token IDs required for every draw")
        if self.proposal_kind not in ("origin", "equal_pair_mixture", "equal_three_policy_mixture"):
            raise ValueError("Unknown primary proposal kind")
        if self.support_status != "VERIFIED":
            raise ValueError(
                "SUPPORT_UNVERIFIED: primary observations require collection support receipts"
            )
        for j, (baseline, candidate) in enumerate(pairs):
            if baseline == candidate and np.any(z[:, j] != 0):
                raise ValueError("Exact policy alias must have exactly zero contribution")
            for k in range(j):
                if pairs[k] == pairs[j] and not np.array_equal(z[:, k], z[:, j]):
                    raise ValueError("Reused exact alias contrast has inconsistent contributions")
                if pairs[k] == pairs[j][::-1] and not np.array_equal(z[:, k], -z[:, j]):
                    raise ValueError("Reverse alias contrast has inconsistent signed contributions")
        object.__setattr__(self, "contributions", _readonly(z))
        object.__setattr__(self, "policy_pairs", pairs)
        object.__setattr__(self, "token_ids", tokens)

    @property
    def n(self):
        return len(self.contributions)

    @property
    def contrasts(self):
        return self.contributions.shape[1]

    @property
    def fingerprint(self):
        identity = repr(
            (
                self.prompt_id,
                self.contrast_ids,
                self.policy_pairs,
                self.sample_ids,
                self.rng_stream_id,
                self.token_ids,
                self.proposal_kind,
                self.support_status,
            )
        ).encode()
        return hashlib.sha256(identity + self.contributions.tobytes()).hexdigest()


@dataclass(frozen=True)
class ObservationEstimate:
    method: str
    estimate: np.ndarray
    raw_estimate: np.ndarray
    contributions: np.ndarray
    raw_contributions: np.ndarray
    mass_residual: np.ndarray
    raw_mass_per_draw: np.ndarray
    helmert_estimate: np.ndarray
    raw_helmert_estimate: np.ndarray
    covariance_per_draw: np.ndarray | None
    covariance_of_mean: np.ndarray | None
    raw_covariance_per_draw: np.ndarray | None
    raw_covariance_of_mean: np.ndarray | None
    b: np.ndarray
    n: int
    diagnostics: dict

    @property
    def delta_pX(self):
        return self.estimate[..., 0]

    @property
    def delta_v(self):
        """Primary signed valid-mass target is -delta_pI for EVERY method."""
        return -self.estimate[..., 3]

    @property
    def raw_valid_event_sum(self):
        """Additional RAW diagnostic; differs from -raw_pI by raw mass M."""
        return self.raw_estimate[..., :3].sum(axis=-1)

    def __post_init__(self):
        for name in (
            "estimate",
            "raw_estimate",
            "contributions",
            "raw_contributions",
            "mass_residual",
            "raw_mass_per_draw",
            "helmert_estimate",
            "raw_helmert_estimate",
            "covariance_per_draw",
            "covariance_of_mean",
            "raw_covariance_per_draw",
            "raw_covariance_of_mean",
            "b",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _raw_diagnostics(batch):
    z = batch.contributions
    weights = z.sum(-1)
    mean = weights.mean(0)
    se = weights.std(axis=0, ddof=1) / math.sqrt(batch.n) if batch.n > 1 else None
    absolute = np.abs(weights)
    with np.errstate(over="ignore", invalid="ignore"):
        second = np.mean(weights**2, axis=0)
    if not np.isfinite(second).all():
        raise FloatingPointError("NONFINITE_SECOND_MOMENT: raw weight diagnostics failed")
    return {
        "raw_mass_audited_before_correction": True,
        "raw_mass_mean": mean.tolist(),
        "raw_mass_standard_error": None if se is None else se.tolist(),
        "raw_mass_constant_nonzero": (np.ptp(weights, axis=0) == 0).__and__(mean != 0).tolist(),
        "weight_abs_max": absolute.max(0).tolist(),
        "weight_abs_q95": np.quantile(absolute, 0.95, axis=0).tolist(),
        "weight_second_moment": second.tolist(),
        "event_nonzero_contribution_counts": np.count_nonzero(z, axis=0).tolist(),
        "unseen_event_is_zero_probability": False,
        "weight_clipped": False,
        "self_normalized": False,
        "sampler_score_normalization_inferred_from_mass": False,
    }


def estimate_geometry(batch, method, b=None):
    """Estimate one prompt's joint contrasts, preserving uncorrected observations.

    Fixed, pilot-frozen or evaluator-only oracle coefficients are accepted here.
    Call ``estimate_pilot`` for pilot methods: it enforces actual partition
    identities before obtaining b. Direct b application is a conditional algebra
    operation and is explicitly marked as such in its diagnostics.
    """
    if not isinstance(batch, ContributionBatch):
        raise TypeError("Finite ContributionBatch required; no world or exact table accepted")
    if method not in METHODS or method == "CROSSFIT_COV_ZERO_SUM":
        raise ValueError("Known method required; cross-fit must use crossfit_estimate")
    if method in SIMPLE_METHODS:
        fixed = fixed_coefficient(method)
        if b is not None and not np.array_equal(
            np.broadcast_to(b, (batch.contrasts, 4)), np.broadcast_to(fixed, (batch.contrasts, 4))
        ):
            raise ValueError("Simple observation method cannot override its frozen coefficient")
        b = fixed
    elif b is None:
        raise ValueError("Externally frozen pilot or evaluator-only oracle coefficient required")
    b = _coefficient(b)
    if b.ndim == 1:
        b = np.broadcast_to(b, (batch.contrasts, 4))
    if b.shape != (batch.contrasts, 4):
        raise ValueError("Coefficient shape must align with batch contrasts")
    z = batch.contributions
    fixed = correction(z, b)
    raw_cov = joint_sample_covariance(z)
    # Algebraically A*raw_cov*A.T, calculated directly from the draws so a
    # large joint panel does not need repeated dense PSD decompositions.
    cov = joint_sample_covariance(fixed)
    raw_mean = z.mean(0)
    output = fixed.mean(0)
    diagnostics = _raw_diagnostics(batch)
    diagnostics.update(
        {
            "status": "OBSERVED",
            "batch_fingerprint": batch.fingerprint,
            "event_order": EVENTS,
            "proposal_kind": batch.proposal_kind,
            "covariance_layout": "CANDIDATE_MAJOR_X_S_W_I",
            "covariance_status": "AVAILABLE" if raw_cov is not None else "INSUFFICIENT_DRAWS",
            "covariance_scope": "CONDITIONAL_ON_FIXED_COEFFICIENT",
            "pilot_independence_checked": False,
            "covariance_per_draw_divided_by_n_once": True,
            "independent_draws": batch.n,
            "unique_contrasts": len(
                {tuple(sorted(pair)) for pair in batch.policy_pairs if pair[0] != pair[1]}
            ),
            "alias_contrasts": [
                i for i, pair in enumerate(batch.policy_pairs) if pair[0] == pair[1]
            ],
            "total_draws_cost": batch.n,
            "oracle_diagnostic_only": method == "KNOWN_COV_ORACLE",
            "raw_point_estimate_was_projected": False,
        }
    )
    return ObservationEstimate(
        method=method,
        estimate=output,
        raw_estimate=raw_mean,
        contributions=fixed,
        raw_contributions=z,
        mass_residual=raw_mean.sum(-1),
        raw_mass_per_draw=z.sum(-1),
        helmert_estimate=output @ helmert(),
        raw_helmert_estimate=raw_mean @ helmert(),
        covariance_per_draw=cov,
        covariance_of_mean=None if cov is None else cov / batch.n,
        raw_covariance_per_draw=raw_cov,
        raw_covariance_of_mean=None if raw_cov is None else raw_cov / batch.n,
        b=b,
        n=batch.n,
        diagnostics=diagnostics,
    )


def join_shared_batches(batches):
    """Stack banks only after checking a single physical draw/token ledger.

    Use this before joint GLS. Concatenating per-bank covariance diagonals loses
    the shared-origin dependence; this operation preserves it and rejects
    mismatched sample ordering, token identities or scoring-policy aliases.
    """
    batches = tuple(batches)
    if not batches or any(not isinstance(batch, ContributionBatch) for batch in batches):
        raise ValueError("At least one finite ContributionBatch required")
    first = batches[0]
    for batch in batches[1:]:
        for field in (
            "sample_ids",
            "rng_stream_id",
            "prompt_id",
            "token_ids",
            "proposal_kind",
            "support_status",
        ):
            if getattr(first, field) != getattr(batch, field):
                raise ValueError(f"SHARED_DRAW_IDENTITY_MISMATCH: {field}")
    pairs = tuple(pair for batch in batches for pair in batch.policy_pairs)
    # A common pair/three-policy proposal cannot be relabelled across unrelated
    # endpoint sets. Shared ORIGIN can legitimately score arbitrarily many.
    components = {policy for pair in pairs for policy in pair}
    cap = {"equal_pair_mixture": 2, "equal_three_policy_mixture": 3}.get(first.proposal_kind)
    if cap is not None and len(components) > cap:
        raise ValueError("SHARED_PROPOSAL_POLICY_MISMATCH: endpoint set exceeds mixture")
    return ContributionBatch(
        contributions=np.concatenate([batch.contributions for batch in batches], axis=1),
        sample_ids=first.sample_ids,
        rng_stream_id=first.rng_stream_id,
        prompt_id=first.prompt_id,
        contrast_ids=tuple(name for batch in batches for name in batch.contrast_ids),
        policy_pairs=pairs,
        token_ids=first.token_ids,
        proposal_kind=first.proposal_kind,
        support_status=first.support_status,
    )
