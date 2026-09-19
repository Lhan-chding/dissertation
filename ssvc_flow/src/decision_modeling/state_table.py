"""Frequency-preserving sparse semantic states from the same observed draws."""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy

from .semantic_schema import EVENTS, REWARD_CHANNELS, reward_vector, validate_features


def vector_moments(values):
    """Stable full covariance; distinguish behavior variance and mean variance."""
    values = [list(row) for row in values]
    if (
        not values
        or not values[0]
        or any(
            len(row) != len(values[0]) or not all(math.isfinite(v) for v in row) for row in values
        )
    ):
        raise ValueError("nonempty finite equal-width rows required")
    n, width = len(values), len(values[0])
    offsets = [[row[j] - values[0][j] for j in range(width)] for row in values]
    center = [math.fsum(row[j] for row in offsets) / n for j in range(width)]
    means = [values[0][j] + center[j] for j in range(width)]
    cross = [
        [
            math.fsum((row[i] - center[i]) * (row[j] - center[j]) for row in offsets)
            for j in range(width)
        ]
        for i in range(width)
    ]
    population = [[v / n for v in row] for row in cross]
    sample = [[v / (n - 1) for v in row] for row in cross] if n > 1 else None
    return {
        "n": n,
        "mean": means,
        "behavior_covariance": population,
        "sample_covariance": sample,
        "covariance_of_mean": [[v / n for v in row] for row in sample]
        if sample is not None
        else None,
        "zero_empirical_variance_is_certainty": False,
    }


def build_state(samples, *, measurement=None):
    """Zk includes Z(k-1), with no additional observations or token deduplication.

    Call once per fixed prompt/policy/source stream. Aggregate fixed task weights
    outside this routine: pooling completions would implicitly reweight tasks.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("cannot estimate a state from zero observations")
    for row in samples:
        validate_features(row)
    counts = {e: sum(row["event"] == e for row in samples) for e in EVENTS}
    n = len(samples)
    z0 = {"pX": counts["X"] / n, "V": 1 - counts["I"] / n}
    z1 = {**z0, "event_counts": counts, "event_probabilities": {e: counts[e] / n for e in EVENTS}}
    applicable = all(row["relation_score"] is not None for row in samples)
    moments = vector_moments([reward_vector(row) for row in samples]) if applicable else None
    z2 = {**z1, "reward_channels": list(REWARD_CHANNELS), "reward_moments": moments}
    relation = Counter(
        (row["event"], row["relation_numerator"], row["relation_denominator"]) for row in samples
    )
    edits = Counter((row["event"], row["single_edit"]) for row in samples)
    z3 = {
        **z2,
        "event_relation_counts": [
            {"event": e, "numerator": a, "denominator": b, "count": count}
            for (e, a, b), count in sorted(relation.items())
        ],
        "event_single_edit_counts": [
            {"event": e, "single_edit": s, "count": count}
            for (e, s), count in sorted(edits.items())
        ],
    }
    z4 = {
        **z3,
        "measurement": deepcopy(measurement)
        if measurement is not None
        else {
            "evidence_kind": "EMPIRICAL_SAMPLES",
            "known_mass": None,
            "unknown_tail": None,
            "numerical_error_bounded": False,
            "statistical_interval": None,
            "no_X_observed": counts["X"] == 0,
            "no_X_means_absent": False,
        },
    }
    return {"n": n, "Z0": z0, "Z1": z1, "Z2": z2, "Z3": z3, "Z4": z4}


def aggregate_states(states, weights):
    """Explicit fixed task/interface weights, never implicit sample-count weights."""
    if set(states) != set(weights) or not states:
        raise ValueError("weights must cover every state exactly")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(
        math.fsum(weights.values()), 1.0, abs_tol=1e-12
    ):
        raise ValueError("fixed weights must be nonnegative and sum to one")
    return {
        "event_probabilities": {
            e: math.fsum(weights[k] * states[k]["Z1"]["event_probabilities"][e] for k in states)
            for e in EVENTS
        },
        "fixed_weights": dict(weights),
        "scope": "DESCRIPTIVE_FIXED_PANEL",
    }
