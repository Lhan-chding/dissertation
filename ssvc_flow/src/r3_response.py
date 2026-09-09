"""Pure NumPy, paired ordinary-IS control responses for R3 banks 0 and 6.

The caller binds model/checkpoint/sampling-lock provenance and writes artifacts.
This module never samples, clips or self-normalizes the primary estimator, and
never certifies safety. Bootstrap intervals cover control-scene uncertainty only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

CATEGORIES = ("X", "S", "W", "I")
INTERFACES = ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")
METRICS = ("pX", "v", "qX_pool", "qS_pool")
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _positive_integer(value, name, *, minimum=1, optional=False):
    if value is None and optional:
        return
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _logprobs(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("Token log probabilities must be finite nonempty one-dimensional arrays")
    return result


def _panel(records, expected_scenes, expected_samples):
    if not records:
        raise ValueError("A nonempty control proposal bank is required")
    keys, prompts, scenes, pairs = set(), {}, {}, {}
    normalized = []
    for row in records:
        for field in ("sample_key", "prompt_id", "base_scene_id", "family", "interface"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Nonempty {field} required")
        key = row["sample_key"]
        if key in keys:
            raise ValueError("Duplicate proposal sample_key")
        keys.add(key)
        if row["interface"] not in INTERFACES or row.get("category") not in CATEGORIES:
            raise ValueError("Unknown interface or category")
        if row.get("split", "control") != "control":
            raise ValueError("Proposal records must belong to the control split")
        if row.get("decode_mode", "sample") not in ("sample", "sampled"):
            raise ValueError("Greedy records cannot be an importance proposal")
        tokens, old = row["token_ids"], _logprobs(row["old_logprobs"])
        if (
            not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or len(tokens) != old.size
        ):
            raise ValueError(
                "Token IDs and old token probabilities must have the same action length"
            )
        identity = (row["family"], row["base_scene_id"], row["interface"])
        prompt = row["prompt_id"]
        if prompt in prompts and prompts[prompt] != identity:
            raise ValueError("Prompt metadata changed within the proposal bank")
        prompts[prompt] = identity
        scene = row["base_scene_id"]
        if scene in scenes and scenes[scene] != row["family"]:
            raise ValueError("A base scene occurs in multiple families")
        scenes[scene] = row["family"]
        if identity in pairs and pairs[identity] != prompt:
            raise ValueError("Multiple prompt IDs for the same scene/interface")
        pairs[identity] = prompt
        normalized.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "sample_key",
                        "prompt_id",
                        "base_scene_id",
                        "family",
                        "interface",
                        "category",
                    )
                },
                "token_ids": list(tokens),
                "old_logprobs": old.tolist(),
            }
        )
    families = sorted(set(scenes.values()))
    if len(families) != 3:
        raise ValueError(
            "Exactly three control families and six family/interface groups are required"
        )
    scene_ids = {f: sorted(s for s, family in scenes.items() if family == f) for f in families}
    for family, ids in scene_ids.items():
        if len(ids) < 2 or (expected_scenes is not None and len(ids) != expected_scenes):
            raise ValueError("Control family scene count differs from the registered panel")
        for scene in ids:
            if any((family, scene, interface) not in pairs for interface in INTERFACES):
                raise ValueError("Both interfaces must be present for every base scene")
    normalized.sort(key=lambda r: r["sample_key"])
    indices = defaultdict(list)
    for index, row in enumerate(normalized):
        indices[row["prompt_id"]].append(index)
    if expected_samples is not None and any(len(v) != expected_samples for v in indices.values()):
        raise ValueError("Per-prompt proposal count differs from the registered panel")
    groups = [f"{f}/{i}" for f in families for i in INTERFACES]
    return (
        normalized,
        {
            "families": families,
            "interfaces": list(INTERFACES),
            "groups": groups,
            "base_scenes": len(scenes),
            "prompts": len(prompts),
            "sequences": len(normalized),
            "base_scene_ids_by_family": scene_ids,
            "prompt_sample_counts": {p: len(v) for p, v in sorted(indices.items())},
            "overall_group_weights": {g: 1 / 6 for g in groups},
            "prompt_weights": "equal prompts within each family/interface group",
            "panel_contract": "STANDARD_R3_24x2x16"
            if expected_scenes == 8 and expected_samples == 16
            else "NONSTANDARD_DIAGNOSTIC_PANEL",
            "proposal_records_sha256": _digest(normalized),
        },
        dict(indices),
        pairs,
    )


def _weights(records, candidate):
    if not isinstance(candidate, dict) or set(candidate) != {r["sample_key"] for r in records}:
        raise ValueError("Candidate sample keys must exactly match the proposal bank")
    log_ratios = []
    normalized = {}
    for row in records:
        key = row["sample_key"]
        new, old = _logprobs(candidate[key]), _logprobs(row["old_logprobs"])
        if new.shape != old.shape:
            raise ValueError("Candidate token log probabilities do not match the proposal action")
        with np.errstate(over="raise", invalid="raise"):
            value = float(np.sum(new - old))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite sequence log-ratio")
        log_ratios.append(value)
        normalized[key] = new.tolist()
    log_ratios = np.asarray(log_ratios)
    # Underflow also fails rather than silently converting positive mass to zero.
    with np.errstate(over="raise", invalid="raise", under="raise"):
        weights = np.exp(log_ratios)
    if not np.isfinite(weights).all() or not np.all(weights > 0):
        raise FloatingPointError("Nonfinite or numerically zero importance weights")
    return weights, log_ratios, _digest(normalized)


def _mean(values, axis=0):
    # Divide first so a representable mean is not lost to a sum overflow.
    with np.errstate(over="raise", invalid="raise"):
        return np.sum(values / values.shape[axis], axis=axis)


def _components(weights, indicators, panel, indices, pairs):
    by_prompt = {
        prompt: _mean(weights[idx, None] * indicators[idx]) for prompt, idx in indices.items()
    }
    return {
        family: np.asarray(
            [
                [by_prompt[pairs[(family, scene, interface)]] for interface in INTERFACES]
                for scene in panel["base_scene_ids_by_family"][family]
            ]
        )
        for family in panel["families"]
    }


def _summaries(components, draws):
    points, bootstrap = {}, {}
    for family, values in components.items():
        point = _mean(values)
        boot = _mean(values[draws[family]], axis=1)
        for index, interface in enumerate(INTERFACES):
            scope = f"{family}/{interface}"
            points[scope] = point[index]
            bootstrap[scope] = boot[:, index]
    points["overall"] = _mean(np.asarray(list(points.values())))
    bootstrap["overall"] = _mean(np.asarray(list(bootstrap.values())))
    return points, bootstrap


def _metrics(components):
    x, s, valid = np.moveaxis(components, -1, 0)
    qx = np.full_like(x, np.nan)
    qs = np.full_like(s, np.nan)
    np.divide(x, valid, out=qx, where=valid > 0)
    np.divide(s, valid, out=qs, where=valid > 0)
    return np.stack((x, valid, qx, qs), axis=-1)


def _deltas(direct_components, target, reference):
    result = target - reference
    # These are paired weight differences, not subtraction of separately summed IS means.
    result[..., 0] = direct_components[..., 0]
    result[..., 1] = direct_components[..., 2]
    return result


def _value(value):
    return float(value) if np.isfinite(value) else None


def _interval(values):
    missing = int((~np.isfinite(values)).sum())
    low, high = (None, None) if missing else map(float, np.quantile(values, [0.025, 0.975]))
    return {
        "status": "NA" if missing else "ESTIMATED",
        "low": low,
        "high": high,
        "half_width": None if missing else (high - low) / 2,
        "undefined_replicates": missing,
        "replicates": len(values),
        "reason": "undefined valid-mass denominator; no replicates discarded" if missing else None,
    }


def _responses(candidate, proposal, baseline, absolute, auxiliary):
    result = {}
    for scope in candidate[0]:
        estimate, theta, zero = [
            _metrics(item[0][scope]) for item in (candidate, proposal, baseline)
        ]
        boot, theta_boot, zero_boot = [
            _metrics(item[1][scope]) for item in (candidate, proposal, baseline)
        ]
        absolute_delta = _deltas(absolute[0][scope], estimate, theta)
        auxiliary_delta = _deltas(auxiliary[0][scope], estimate, zero)
        absolute_boot = _deltas(absolute[1][scope], boot, theta_boot)
        auxiliary_boot = _deltas(auxiliary[1][scope], boot, zero_boot)
        result[scope] = {
            metric: {
                "estimate": _value(estimate[j]),
                "theta_estimate": _value(theta[j]),
                "baseline_estimate": _value(zero[j]),
                "absolute_delta": _value(absolute_delta[j]),
                "auxiliary_delta": _value(auxiliary_delta[j]),
                "predicted_delta": _value(auxiliary_delta[j]),
                "observed_delta": None,
                "prediction_residual": None,
                "ci": {
                    "estimate": _interval(boot[:, j]),
                    "absolute_delta": _interval(absolute_boot[:, j]),
                    "auxiliary_delta": _interval(auxiliary_boot[:, j]),
                },
            }
            for j, metric in enumerate(METRICS)
        }
    return result


def _ess(weights):
    if not weights.size:
        return None, None
    scaled = weights / weights.max()
    normalized = scaled / scaled.sum()
    return float(1 / np.square(normalized).sum()), float(normalized.max())


def _diagnostic(weights, categories, *, scope, scope_id):
    ess, maximum = _ess(weights)
    n = len(weights)
    warnings = []
    if ess / n < 0.5 or maximum > 0.05:
        warnings.append("OVERLAP_WARNING")
    if not np.any(categories == "X"):
        warnings.append("NO_OBSERVED_X_SUPPORT")
    support = {}
    for category in CATEGORIES:
        selected = weights[categories == category]
        class_ess, _ = _ess(selected)
        support[category] = {
            "count": len(selected),
            "ESS": class_ess,
            "ESS_fraction": class_ess / len(selected) if len(selected) else None,
            "support_status": "OBSERVED_EMPIRICAL_SUPPORT"
            if len(selected)
            else "NO_OBSERVED_SUPPORT",
        }
    return {
        "scope": scope,
        "scope_id": scope_id,
        "n": n,
        "ESS": ess,
        "ESS_fraction": ess / n,
        "max_normalized_weight": maximum,
        "mean_weight": float(_mean(weights)),
        "weight_quantiles": {
            str(q): float(v)
            for q, v in zip(QUANTILES, np.quantile(weights, QUANTILES), strict=True)
        },
        "categories": support,
        "warnings": warnings,
        "weight_scope": "RAW_IMPORTANCE_RATIOS_WITHIN_REPORTED_SCOPE",
        "uniform_weight_max_normalized": 1 / n,
        "registered_thresholds": {"min_ESS_fraction": 0.5, "max_normalized_weight": 0.05},
        "can_use_for_safety_decision": False,
    }


def _diagnostics(weights, records, panel, indices):
    categories = np.asarray([row["category"] for row in records])
    by_prompt = {
        p: _diagnostic(weights[idx], categories[idx], scope="prompt", scope_id=p)
        for p, idx in sorted(indices.items())
    }
    by_group = {}
    for group in panel["groups"]:
        idx = [i for i, r in enumerate(records) if f"{r['family']}/{r['interface']}" == group]
        by_group[group] = _diagnostic(
            weights[idx], categories[idx], scope="family_interface", scope_id=group
        )
    return {
        "by_prompt": by_prompt,
        "by_group": by_group,
        "overall": _diagnostic(weights, categories, scope="overall", scope_id="overall"),
        "scope_note": (
            "At n=16 even uniform prompt weights have max_normalized_weight=0.0625 > 0.05. "
            "Thresholds are applied independently at every reported scope; larger-scope "
            "overlap never overrides a prompt warning or certifies safety."
        ),
    }


def analyze_control_responses(
    proposal_records,
    candidate_logprobs,
    *,
    baseline_key,
    bank_index,
    bootstrap_replicates=5000,
    seed=20260909,
    expected_base_scenes_per_family=8,
    expected_samples_per_prompt=16,
):
    """Return JSON-ready per-candidate predicted responses for one scored bank.

    Candidate mappings are candidate_key -> sample_key -> selected-token logps.
    The caller must identify the actual lambda-zero candidate with baseline_key.
    Nondefault panel sizes are explicit diagnostics, never relabeled full R3 data.
    """
    if type(bank_index) is not int or bank_index not in (0, 6):
        raise ValueError("Only preregistered bank indices 0 or 6 have IS responses")
    _positive_integer(bootstrap_replicates, "bootstrap_replicates", minimum=2)
    _positive_integer(seed, "seed", minimum=0)
    _positive_integer(
        expected_base_scenes_per_family, "expected_base_scenes_per_family", minimum=2, optional=True
    )
    _positive_integer(expected_samples_per_prompt, "expected_samples_per_prompt", optional=True)
    if not isinstance(candidate_logprobs, dict) or baseline_key not in candidate_logprobs:
        raise ValueError("Candidate map must include its lambda-zero baseline key")
    if any(not isinstance(key, str) or not key for key in candidate_logprobs):
        raise ValueError("Candidate keys must be nonempty strings")
    records, panel, indices, pairs = _panel(
        proposal_records,
        expected_base_scenes_per_family,
        expected_samples_per_prompt,
    )
    rng = np.random.default_rng(seed)
    draws = {
        family: rng.integers(0, len(ids), size=(bootstrap_replicates, len(ids)))
        for family, ids in panel["base_scene_ids_by_family"].items()
    }
    indicators = np.asarray(
        [[r["category"] == "X", r["category"] == "S", r["category"] != "I"] for r in records],
        dtype=np.float64,
    )
    # Validate every candidate before constructing an apparent successful response.
    weight_banks = {
        key: _weights(records, values) for key, values in sorted(candidate_logprobs.items())
    }

    def summarize(weights):
        return _summaries(_components(weights, indicators, panel, indices, pairs), draws)

    proposal = summarize(np.ones(len(records)))
    baseline_weights = weight_banks[baseline_key][0]
    baseline = summarize(baseline_weights)
    candidates = {}
    for key, (weights, log_ratios, digest) in weight_banks.items():
        candidate = summarize(weights)
        absolute = summarize(weights - 1)
        auxiliary = summarize(weights - baseline_weights)
        diagnostics = _diagnostics(weights, records, panel, indices)
        scopes = [
            *diagnostics["by_prompt"].values(),
            *diagnostics["by_group"].values(),
            diagnostics["overall"],
        ]
        warnings = sorted({w for value in scopes for w in value["warnings"]})
        candidates[key] = {
            "status": "ESTIMATED_WITH_WARNINGS" if warnings else "ESTIMATED",
            "responses": _responses(candidate, proposal, baseline, absolute, auxiliary),
            "diagnostics": diagnostics,
            "warnings": warnings,
            "candidate_logprobs_sha256": digest,
            "sequence_records": [
                {
                    "sample_key": row["sample_key"],
                    "prompt_id": row["prompt_id"],
                    "sequence_log_ratio": float(log_ratios[j]),
                    "weight": float(weights[j]),
                    "paired_weight_difference_from_lambda0": float(
                        weights[j] - baseline_weights[j]
                    ),
                }
                for j, row in enumerate(records)
            ],
            "can_use_for_safety_decision": False,
            "safety_status": "NOT_CERTIFIED",
            "direct_resampling": "NOT_MEASURED",
        }
    return {
        "status": "ESTIMATED_WITH_WARNINGS"
        if any(c["warnings"] for c in candidates.values())
        else "ESTIMATED",
        "method": "ORDINARY_UNCLIPPED_IMPORTANCE_SAMPLING",
        "bank_index": bank_index,
        "baseline_key": baseline_key,
        "panel": panel,
        "candidates": candidates,
        "proposal": {
            scope: {
                metric: _value(value)
                for metric, value in zip(METRICS, _metrics(values), strict=True)
            }
            for scope, values in proposal[0].items()
        },
        "bootstrap_contract": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "unit": "base_scene",
            "stratified_by": "family",
            "interfaces_paired_within_scene": True,
            "candidate_draws_shared": True,
            "within_scene_outputs_resampled": False,
            "all_ratios_recomputed_in_each_replicate": True,
            "undefined_ratio_policy": (
                "NA_IF_ANY_REPLICATE_UNDEFINED; no dropping or epsilon denominator"
            ),
            "draw_indices_sha256": {
                family: hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()
                for family, idx in draws.items()
            },
            "CI_scope": (
                "pointwise evaluation-scene uncertainty conditional on this trained candidate "
                "and proposal bank; excludes training-seed uncertainty"
            ),
            "joint_coverage_scope": "NONE; no simultaneous safety guarantee",
        },
        "primary_weights_clipped": False,
        "primary_self_normalized": False,
        "safety_status": "NOT_CERTIFIED",
        "can_use_for_safety_decision": False,
        "support_contract": (
            "Caller must verify the proposal's full-softmax sampling lock and matching "
            "EOS/action definition; observed support does not certify population support"
        ),
    }
