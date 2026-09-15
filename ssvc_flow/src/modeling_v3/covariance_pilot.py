"""Independent-pilot control variates and full-refit two-fold inference.

Coefficients use finite pilot contributions only. Exact covariance coefficients
are evaluator-only diagnostics. No finite-pilot estimator inherits the known-
covariance oracle's variance guarantee; uncertainty is conditional until an
outer repeat or bootstrap reruns the entire pilot/cross-fit procedure.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np

from .observation_geometry import (
    B_EQUAL,
    B_PRESERVE_XI,
    SIMPLE_METHODS,
    ContributionBatch,
    _covariance,
    _events,
    estimate_geometry,
    helmert,
    joint_sample_covariance,
)


def oracle_b(cov):
    """Known per-draw 4x4 covariance coefficient; None means no mass variation.

    The routine consumes a covariance only. Complete probability tables remain
    in the isolated evaluator and cannot be attached to a finite estimator.
    """
    cov = _covariance(cov)
    if cov.shape != (4, 4):
        raise ValueError("One four-event per-draw covariance required")
    c = cov @ np.ones(4)
    denominator = float(c.sum())
    tolerance = 1e-14 * max(float(np.trace(cov)), np.finfo(float).tiny)
    if denominator <= tolerance:
        return None
    b = c / denominator
    if not np.isfinite(b).all():
        raise FloatingPointError("NONFINITE_COVARIANCE_COEFFICIENT")
    return b


def _limit_l1(b, l1_cap):
    """Largest feasible position on the segment equal -> unbounded b."""
    if np.abs(b).sum() <= l1_cap:
        return b.copy(), 1.0
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2
        value = B_EQUAL + middle * (b - B_EQUAL)
        if abs(value).sum() <= l1_cap:
            low = middle
        else:
            high = middle
    return B_EQUAL + low * (b - B_EQUAL), low


def pilot_coefficient(samples, *, shrink=0.1, l1_cap=8.0):
    """Fit and freeze b from pilot-only draws, with reproducible fallback receipts.

    Shrinkage is (1-eta)*Sigma + eta*trace(Sigma)*I/4. A pilot with no
    raw mass variation is uninformative even if artificial shrinkage would
    create positive mass variance. Fewer than two draws cannot estimate a
    covariance and use the separately identified PRESERVE_XI fallback.
    """
    z = _events(samples)
    if z.ndim != 2 or not len(z):
        raise ValueError("Nonempty pilot samples with shape (draws,4) required")
    if (
        not isinstance(shrink, (int, float, np.floating))
        or not math.isfinite(shrink)
        or not 0 <= shrink <= 1
    ):
        raise ValueError("Pilot covariance shrinkage must be in [0,1]")
    if not math.isfinite(l1_cap) or l1_cap < 1:
        raise ValueError("Pilot L1 cap must be finite and at least one")
    diagnostics = {
        "pilot_draws": len(z),
        "shrink": float(shrink),
        "l1_cap": float(l1_cap),
        "fallback": "PRESERVE_XI",
        "fallback_used": False,
        "mix": 0.0,
        "l1_limited": False,
        "unbounded_b": None,
        "pilot_covariance_per_draw": None,
        "shrunk_covariance_per_draw": None,
        "zero_mass_variance": False,
        "no_oracle_optimality_claim": True,
        "heldout_used": False,
    }
    if len(z) < 2:
        diagnostics.update(status="PILOT_INSUFFICIENT_DRAWS", fallback_used=True)
        return B_PRESERVE_XI.copy(), diagnostics
    centered = z - z.mean(0)
    cov = centered.T @ centered / (len(z) - 1)
    if not np.isfinite(cov).all():
        raise FloatingPointError("NONFINITE_SECOND_MOMENT: pilot covariance cannot be clipped")
    mass = z.sum(-1)
    mass_variance = float(np.var(mass, ddof=1))
    trace = float(np.trace(cov))
    tolerance = 1e-14 * max(trace, np.finfo(float).tiny)
    diagnostics.update(pilot_covariance_per_draw=cov.tolist(), raw_mass_variance=mass_variance)
    if mass_variance <= tolerance:
        diagnostics.update(
            status="PILOT_UNINFORMATIVE", fallback_used=True, zero_mass_variance=True
        )
        return B_PRESERVE_XI.copy(), diagnostics
    shrunk = (1 - shrink) * cov + shrink * trace / 4 * np.eye(4)
    diagnostics["shrunk_covariance_per_draw"] = shrunk.tolist()
    b = oracle_b(shrunk)
    if b is None:
        diagnostics.update(
            status="PILOT_UNINFORMATIVE", fallback_used=True, zero_mass_variance=True
        )
        return B_PRESERVE_XI.copy(), diagnostics
    diagnostics["unbounded_b"] = b.tolist()
    bounded, mix = _limit_l1(b, l1_cap)
    diagnostics.update(
        status="PILOT_ESTIMATED",
        mix=float(mix),
        l1_limited=mix < 1,
        fitted_b=bounded.tolist(),
        fitted_l1=float(abs(bounded).sum()),
    )
    return bounded, diagnostics


def validate_independent_batches(first, second):
    """Reject reused sampling/RNG identity before any adaptive coefficient fit.

    Equal token sequences do not imply reused samples. Actual random-draw IDs,
    not action values, define the partition. Both batches must sample the same
    frozen prompt, contrast and proposal experiment for two-fold interchange.
    """
    if not isinstance(first, ContributionBatch) or not isinstance(second, ContributionBatch):
        raise TypeError("Two finite ContributionBatch objects required")
    for field in ("prompt_id", "contrast_ids", "policy_pairs", "proposal_kind", "support_status"):
        if getattr(first, field) != getattr(second, field):
            raise ValueError(f"Pilot/main experiment identity mismatch: {field}")
    if first.rng_stream_id == second.rng_stream_id:
        raise ValueError("PILOT_MAIN_RNG_LEAKAGE: independent RNG streams required")
    if set(first.sample_ids).intersection(second.sample_ids):
        raise ValueError("PILOT_MAIN_SAMPLE_LEAKAGE: disjoint draw sample IDs required")


def _fit_batch(batch, *, shrink, l1_cap):
    rows = []
    diagnostics = []
    for j in range(batch.contrasts):
        coefficient, receipt = pilot_coefficient(
            batch.contributions[:, j], shrink=shrink, l1_cap=l1_cap
        )
        rows.append(coefficient)
        diagnostics.append(receipt)
    return np.asarray(rows), diagnostics


def estimate_pilot(pilot, main, *, shrink=None, l1_cap=8.0, method="PILOT_SHRINK_ZERO_SUM"):
    """Fit on an independent pilot, apply to main, retain raw mass and cost.

    The reported empirical covariance is conditional on the frozen pilot. It
    excludes random-pilot covariance, especially cross-contrast uncertainty
    from shared pilot draws. Outer repeated measurement or full-refit bootstrap
    must include that extra uncertainty before a marginal interval is claimed.
    """
    if method not in ("PILOT_COV_ZERO_SUM", "PILOT_SHRINK_ZERO_SUM"):
        raise ValueError("Independent-pilot observation method required")
    if shrink is None:
        shrink = 0.0 if method == "PILOT_COV_ZERO_SUM" else 0.1
    if method == "PILOT_COV_ZERO_SUM" and shrink != 0:
        raise ValueError("PILOT_COV_ZERO_SUM has zero covariance shrinkage")
    validate_independent_batches(pilot, main)
    b, receipts = _fit_batch(pilot, shrink=shrink, l1_cap=l1_cap)
    result = estimate_geometry(main, method, b=b)
    diag = {
        **result.diagnostics,
        "pilot_independence_checked": True,
        "pilot_batch_fingerprint": pilot.fingerprint,
        "main_batch_fingerprint": main.fingerprint,
        "pilot_draws": pilot.n,
        "main_draws": main.n,
        "total_draws_cost": pilot.n + main.n,
        "pilot_fraction": pilot.n / (pilot.n + main.n),
        "pilot_receipts": receipts,
        "fallback_count": sum(r["fallback_used"] for r in receipts),
        "l1_limit_count": sum(r["l1_limited"] for r in receipts),
        "covariance_scope": "CONDITIONAL_ON_FROZEN_PILOT",
        "pilot_estimation_uncertainty_included": False,
        "pilot_raw_mass_mean": pilot.contributions.sum(-1).mean(0).tolist(),
        "no_oracle_optimality_claim": True,
        "both_policy_aliases_share_observation_noise": True,
    }
    return replace(result, diagnostics=diag)


def concatenate_independent(first, second):
    validate_independent_batches(first, second)
    identity = hashlib.sha256(
        (first.rng_stream_id + "\0" + second.rng_stream_id).encode()
    ).hexdigest()
    return ContributionBatch(
        contributions=np.concatenate([first.contributions, second.contributions], axis=0),
        sample_ids=first.sample_ids + second.sample_ids,
        rng_stream_id="combined:" + identity,
        prompt_id=first.prompt_id,
        contrast_ids=first.contrast_ids,
        policy_pairs=first.policy_pairs,
        token_ids=first.token_ids + second.token_ids,
        proposal_kind=first.proposal_kind,
        support_status=first.support_status,
    )


def cost_matched_estimates(pilot, main, *, shrink=0.1, l1_cap=8.0, include_crossfit=True):
    """Same-cost simple baselines use ALL n_pilot+n_main draws.

    Paired main-only baselines are additional diagnostics, not the primary
    comparator. Cross-fit reuses these independent folds, explicitly recording
    their sizes (possibly unequal) instead of pretending they were equal.
    Oracle coefficients intentionally require a separate evaluator call.
    """
    combined = concatenate_independent(pilot, main)
    all_draws = {method: estimate_geometry(combined, method) for method in SIMPLE_METHODS}
    main_only = {method: estimate_geometry(main, method) for method in SIMPLE_METHODS}
    all_draws["PILOT_COV_ZERO_SUM"] = estimate_pilot(
        pilot, main, shrink=0, l1_cap=l1_cap, method="PILOT_COV_ZERO_SUM"
    )
    all_draws["PILOT_SHRINK_ZERO_SUM"] = estimate_pilot(pilot, main, shrink=shrink, l1_cap=l1_cap)
    if include_crossfit:
        all_draws["CROSSFIT_COV_ZERO_SUM"] = crossfit_estimate(pilot, main, shrink=0, l1_cap=l1_cap)
    return {
        "cost_matched": all_draws,
        "paired_main_only": main_only,
        "total_draws": combined.n,
        "pilot_draws": pilot.n,
        "main_draws": main.n,
        "oracle_is_separate_evaluator_only": True,
    }


def crossfit_estimate(fold_a, fold_b, *, shrink=0.0, l1_cap=8.0):
    """Two-fold adaptive mean with fold-size weighting, no false independent CI.

    Each fold's coefficient is fitted ONLY on the other fold. This keeps the
    mean unbiased under the population zero-mass premise even with unequal
    fold sizes. Fold errors are dependent through their mutual fitted
    coefficients; a concatenated sample covariance is not a variance estimate
    for this adaptive mean, and is intentionally unavailable.
    """
    validate_independent_batches(fold_a, fold_b)
    a_to_b = estimate_pilot(fold_a, fold_b, shrink=shrink, l1_cap=l1_cap)
    b_to_a = estimate_pilot(fold_b, fold_a, shrink=shrink, l1_cap=l1_cap)
    combined = concatenate_independent(fold_a, fold_b)
    raw = estimate_geometry(combined, "RAW4")
    transformed = np.concatenate([b_to_a.contributions, a_to_b.contributions], axis=0)
    estimate = transformed.mean(0)
    diagnostics = {
        **raw.diagnostics,
        "covariance_status": "CROSSFIT_REQUIRES_FULL_REFIT",
        "covariance_scope": "UNAVAILABLE_WITHOUT_FULL_REFIT",
        "pilot_independence_checked": True,
        "pilot_estimation_uncertainty_included": False,
        "crossfit_fold_sizes": [fold_a.n, fold_b.n],
        "crossfit_fold_weights": [fold_a.n / combined.n, fold_b.n / combined.n],
        "folds_are_independent_for_interval": False,
        "fold_coefficients": [b_to_a.b.tolist(), a_to_b.b.tolist()],
        "fold_receipts": [
            b_to_a.diagnostics["pilot_receipts"],
            a_to_b.diagnostics["pilot_receipts"],
        ],
        "covariance_per_draw_divided_by_n_once": False,
        "total_draws_cost": combined.n,
        "shrink": float(shrink),
        "l1_cap": float(l1_cap),
        "fallback_count": a_to_b.diagnostics["fallback_count"]
        + b_to_a.diagnostics["fallback_count"],
    }
    # There is no one common fixed b in cross-fit. The leading axis records
    # the two coefficients applied to folds A and B, never their average.
    return replace(
        raw,
        method="CROSSFIT_COV_ZERO_SUM",
        estimate=estimate,
        contributions=transformed,
        helmert_estimate=estimate @ helmert(),
        covariance_per_draw=None,
        covariance_of_mean=None,
        b=np.stack([b_to_a.b, a_to_b.b]),
        diagnostics=diagnostics,
    )


def _resampled_batch(batch, indices, *, namespace):
    """Bootstrap draws retain action identities but receive bootstrap draw IDs."""
    return ContributionBatch(
        contributions=batch.contributions[indices],
        sample_ids=tuple(f"{namespace}:draw:{j}" for j in range(len(indices))),
        rng_stream_id=namespace,
        prompt_id=batch.prompt_id,
        contrast_ids=batch.contrast_ids,
        policy_pairs=batch.policy_pairs,
        token_ids=tuple(batch.token_ids[j] for j in indices),
        proposal_kind=batch.proposal_kind,
        support_status=batch.support_status,
    )


def bootstrap_crossfit(fold_a, fold_b, *, repetitions=1000, seed=0, shrink=0.0, l1_cap=8.0):
    """Resample each independent fold jointly over contrasts and refit BOTH b's.

    This is a conditional nonparametric bootstrap uncertainty estimate, not a
    finite-sample coverage certificate or a top-level training-seed bootstrap.
    Pipeline-level inference must also refit the selected downstream estimator.
    """
    validate_independent_batches(fold_a, fold_b)
    if type(repetitions) is not int or repetitions < 2 or type(seed) is not int or seed < 0:
        raise ValueError(
            "At least two bootstrap repetitions and a nonnegative integer seed required"
        )
    rng = np.random.default_rng(seed)
    values = []
    for repetition in range(repetitions):
        a = _resampled_batch(
            fold_a,
            rng.integers(0, fold_a.n, size=fold_a.n),
            namespace=f"bootstrap:{seed}:{repetition}:a",
        )
        b = _resampled_batch(
            fold_b,
            rng.integers(0, fold_b.n, size=fold_b.n),
            namespace=f"bootstrap:{seed}:{repetition}:b",
        )
        values.append(crossfit_estimate(a, b, shrink=shrink, l1_cap=l1_cap).estimate)
    values = np.asarray(values)
    return {
        "replicate_estimates": values,
        "covariance_of_mean": joint_sample_covariance(values),
        "status": "BOOTSTRAP_ESTIMATE_NOT_COVERAGE_CERTIFICATE",
        "repetitions": repetitions,
        "seed": seed,
        "refits_per_replicate": 2,
        "sample_unit": "JOINT_DRAW_WITHIN_ORIGINAL_INDEPENDENT_FOLD",
        "raw_draw_ids_retained_in_original_batches": True,
        "training_seed_uncertainty_included": False,
        "downstream_model_refitted": False,
    }
