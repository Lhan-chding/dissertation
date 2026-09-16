"""Shared RAW4 observations and full-refit crossfit uncertainty.

The input is one paid iid packet. Corrections never regenerate actions, and
crossfit never treats its mutually fitted corrected folds as iid observations.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..modeling_v3.covariance_pilot import (
    bootstrap_crossfit,
    crossfit_estimate,
    estimate_pilot,
    validate_independent_batches,
)
from ..modeling_v3.observation_geometry import (
    ContributionBatch,
    estimate_geometry,
    joint_sample_covariance,
)

PRIMARY_METHODS = ("RAW4", "PRESERVE_XI", "CROSSFIT_COV_ZERO_SUM")
DIAGNOSTIC_METHODS = ("EQUAL_ZERO_SUM", "PILOT_SHRINK_ZERO_SUM")


def assert_predictor_independence(
    sample_ids, rng_stream_ids, *, reference_sample_ids=(), reference_rng_stream_ids=()
):
    """Check identities only; this API never receives or reads reference labels."""
    if set(sample_ids) & set(reference_sample_ids):
        raise ValueError("Predictor/reference sample identity leakage")
    if set(rng_stream_ids) & set(reference_rng_stream_ids):
        raise ValueError("Predictor/reference RNG stream leakage")


def _split(batch, *, seed, fraction):
    if type(seed) is not int or seed < 0 or batch.n < 4:
        raise ValueError("At least four iid draws and a nonnegative split seed required")
    indices = np.random.default_rng(seed).permutation(batch.n)
    size = max(2, min(batch.n - 2, int(batch.n * fraction)))
    result = []
    for label, take in (("a", indices[:size]), ("b", indices[size:])):
        result.append(
            ContributionBatch(
                batch.contributions[take],
                tuple(batch.sample_ids[i] for i in take),
                f"{batch.rng_stream_id}/iid_partition/{seed}/{fraction}/{label}",
                batch.prompt_id,
                batch.contrast_ids,
                batch.policy_pairs,
                tuple(batch.token_ids[i] for i in take),
                batch.proposal_kind,
                batch.support_status,
            )
        )
    return tuple(result)


def _result(estimate, uncertainty):
    return {
        "estimate": estimate.estimate,
        "covariance_of_mean": estimate.covariance_of_mean,
        "raw_mass_mean": estimate.raw_mass_per_draw.mean(0),
        "diagnostics": estimate.diagnostics,
        "uncertainty": uncertainty,
    }


def observe_packet(
    batch,
    *,
    methods=PRIMARY_METHODS,
    sampling_design="iid",
    split_seed=0,
    bootstrap_repetitions=1000,
    bootstrap_seed=0,
    shrink=0.0,
    l1_cap=8.0,
    costs=None,
    reference_sample_ids=(),
    reference_rng_stream_ids=(),
):
    """Apply finite V3 estimators once to a shared V4 packet.

    The deterministic split is chosen independently of values. For an iid
    packet, disjoint draw indices form independent folds; their original draw
    IDs remain intact. Bootstrap resamples whole original draws within each
    fold and refits both coefficients in every replicate.
    """
    if not isinstance(batch, ContributionBatch):
        raise TypeError("One identified ContributionBatch required")
    if sampling_design != "iid":
        raise ValueError("This wrapper requires iid sampling, not fixed quota mixture draws")
    methods = tuple(methods)
    if (
        len(set(methods)) != len(methods)
        or not methods
        or set(methods) - set(PRIMARY_METHODS + DIAGNOSTIC_METHODS)
    ):
        raise ValueError("Unknown, empty or duplicated observation methods")
    assert_predictor_independence(
        batch.sample_ids,
        [batch.rng_stream_id],
        reference_sample_ids=reference_sample_ids,
        reference_rng_stream_ids=reference_rng_stream_ids,
    )
    actual_costs = dict(costs or {})
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in actual_costs.values()
    ):
        raise ValueError("Actual measured costs must be finite nonnegative numbers")
    started = time.perf_counter()
    results = {}
    for method in methods:
        uncertainty = {"status": "FIXED_TRANSFORM_EMPIRICAL_COVARIANCE", "model_calls": 0}
        if method == "CROSSFIT_COV_ZERO_SUM":
            a, b = _split(batch, seed=split_seed, fraction=0.5)
            estimate = crossfit_estimate(a, b, shrink=shrink, l1_cap=l1_cap)
            uncertainty = {"status": "UNAVAILABLE_WITHOUT_FULL_REFIT_OR_PACKETS", "model_calls": 0}
            if bootstrap_repetitions is not None:
                uncertainty = {
                    **bootstrap_crossfit(
                        a,
                        b,
                        repetitions=bootstrap_repetitions,
                        seed=bootstrap_seed,
                        shrink=shrink,
                        l1_cap=l1_cap,
                    ),
                    "model_calls": 0,
                }
                uncertainty["percentile_interval"] = np.quantile(
                    uncertainty["replicate_estimates"], [0.025, 0.975], axis=0
                )
        elif method == "PILOT_SHRINK_ZERO_SUM":
            a, b = _split(batch, seed=split_seed, fraction=0.25)
            estimate = estimate_pilot(a, b, shrink=0.1, l1_cap=l1_cap)
            uncertainty = {"status": "CONDITIONAL_ON_PILOT_ONLY", "model_calls": 0}
        else:
            estimate = estimate_geometry(batch, method)
        results[method] = _result(estimate, uncertainty)
        if method == "CROSSFIT_COV_ZERO_SUM" and bootstrap_repetitions is not None:
            results[method]["covariance_of_mean"] = uncertainty["covariance_of_mean"]
    return {
        "methods": results,
        "sample_ids": batch.sample_ids,
        "packet_fingerprint": batch.fingerprint,
        "rng_stream_id": batch.rng_stream_id,
        "raw_contributions": batch.contributions,
        "raw_mass_per_draw": batch.contributions.sum(-1),
        "draws": batch.n,
        "actual_costs": actual_costs,
        "estimation_cost": {"cpu_seconds": time.perf_counter() - started, "model_calls": 0},
        "reference_labels_read": False,
        "sampling_design": sampling_design,
        "all_methods_reuse_one_packet": True,
    }


def independent_packet_uncertainty(packets, *, method, split_seed=0, shrink=0.0, l1_cap=8.0):
    """Refit each independent complete packet before computing between-packet variance."""
    packets = list(packets)
    if len(packets) < 2:
        raise ValueError("At least two complete independent packets required")
    seen_ids, seen_streams = set(), set()
    estimates = []
    for batch in packets:
        if seen_ids & set(batch.sample_ids) or batch.rng_stream_id in seen_streams:
            raise ValueError("Independent packets contain reused draws or RNG streams")
        if estimates:
            validate_independent_batches(packets[0], batch)
        seen_ids.update(batch.sample_ids)
        seen_streams.add(batch.rng_stream_id)
        estimates.append(
            observe_packet(
                batch,
                methods=[method],
                split_seed=split_seed,
                bootstrap_repetitions=None,
                shrink=shrink,
                l1_cap=l1_cap,
            )["methods"][method]["estimate"]
        )
    values = np.asarray(estimates)
    covariance = joint_sample_covariance(values)
    return {
        "status": "INDEPENDENT_COMPLETE_PACKET_EMPIRICAL_UNCERTAINTY",
        "method": method,
        "packet_estimates": values,
        "estimate": values.mean(0),
        "packet_covariance": covariance,
        "covariance_of_packet_mean": covariance / len(packets),
        "independent_packets": len(packets),
        "draws_per_packet": [p.n for p in packets],
        "coefficients_refitted_for_every_packet": True,
        "training_seed_uncertainty_included": False,
        "coverage_certified": False,
        "model_calls": 0,
    }
