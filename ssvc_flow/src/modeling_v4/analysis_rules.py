"""Frozen descriptive reference labels, distinct from model success criteria."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import digest


def load_analysis_rules():
    path = Path(__file__).resolve().parents[2] / "configs/modeling_v4/analysis_rules.json"
    return json.loads(path.read_text())


def endpoint_overlap(log_weights, *, rules=None):
    """Return stable ESS diagnostics without modifying importance contributions.

    The first axis is independent draws; remaining axes may be prompts/policies.
    Normalization here is only for ESS and maximum-weight diagnostics. The raw
    probability estimator must continue to use its original unnormalized ratios.
    """
    rules = load_analysis_rules() if rules is None else rules
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if log_weights.ndim < 1 or len(log_weights) < 2 or not np.isfinite(log_weights).all():
        raise ValueError("Finite actual log importance weights with at least two draws required")
    maximum = log_weights.max(axis=0)
    centered = np.exp(log_weights - maximum)
    totals = centered.sum(axis=0)
    fractions = centered / totals
    ess = totals**2 / (centered**2).sum(axis=0)
    fraction = ess / len(log_weights)
    max_weight = fractions.max(axis=0)
    criteria = rules["origin_overlap"]
    usable = (fraction >= criteria["minimum_endpoint_ess_fraction"]) & (
        max_weight <= criteria["maximum_normalized_endpoint_weight"]
    )
    return {
        "draws": len(log_weights),
        "ess": ess,
        "ess_fraction": fraction,
        "max_normalized_weight": max_weight,
        "usable": usable,
        "all_usable": bool(np.all(usable)),
        "rules_hash": digest(rules),
        "raw_weights_changed": False,
        "accuracy_certified": False,
    }


def reference_precision(standard_error, *, exact_alias=False, overlap_usable=True, rules=None):
    """Label all three registered scales using actual empirical reference SE.

    These are descriptive normal-approximation widths, not an asserted coverage
    guarantee. Nonalias zero empirical variance does not prove a zero response.
    """
    rules = load_analysis_rules() if rules is None else rules
    se = np.asarray(standard_error, dtype=np.float64)
    if np.any(se[np.isfinite(se)] < 0):
        raise ValueError("Reference standard errors cannot be negative")
    alias = np.broadcast_to(np.asarray(exact_alias, dtype=bool), se.shape)
    usable = np.broadcast_to(np.asarray(overlap_usable, dtype=bool), se.shape)
    if np.any(alias & (se != 0)):
        raise ValueError("Exact inference aliases must have exactly zero reference uncertainty")
    spec = rules["reference_precision"]
    available = np.isfinite(se) & (se > 0) & usable
    half_width = spec["normal_approximation_multiplier"] * se
    by_scale = {
        str(scale): alias
        | (available & (half_width <= spec["max_half_width_as_fraction_of_scale"] * scale))
        for scale in spec["probability_scales"]
    }
    return {
        "normal_approximation_half_width": half_width,
        "resolved_at_scale": by_scale,
        "reference_resolved": by_scale[str(spec["primary_resolved_scale"])],
        "primary_scale": spec["primary_resolved_scale"],
        "all_finite_residuals_must_also_be_reported": True,
        "coverage_guarantee": False,
        "rules_hash": digest(rules),
    }
