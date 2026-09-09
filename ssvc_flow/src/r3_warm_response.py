"""Pure CPU comparisons of warm proposal predictions and independent direct draws.

The runner binds the warm checkpoint, candidate identities, full-softmax protocol,
and any random-number coupling. This module pairs base scenes, never individual
outputs or tokens. Its intervals describe evaluation-scene uncertainty conditional
on the realized within-prompt samples and candidates, not safety or training seeds.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .core import canonical_hash
from .r3_response import (
    CATEGORIES,
    INTERFACES,
    METRICS,
    _components,
    _deltas,
    _interval,
    _metrics,
    _panel,
    _summaries,
    _value,
    _weights,
    analyze_control_responses,
)

SMALL_PANEL_SCENES_PER_FAMILY = 8  # Display warning; not a significance or coverage threshold.


def _support(counts):
    warnings = []
    if counts[0] == 0:
        warnings.append("NO_OBSERVED_X_SUPPORT")
    if counts[:3].sum() == 0:
        warnings.append("NO_OBSERVED_VALID_SUPPORT")
    return {
        "n": int(counts.sum()),
        "category_counts": {c: int(v) for c, v in zip(CATEGORIES, counts, strict=True)},
        "warnings": warnings,
        "population_support_certified": False,
    }


def _direct_panels(direct, panel, pairs, keys, bank_index):
    if not isinstance(direct, (list, tuple)) or len(direct) != 1536:
        raise ValueError("Each warm response bank requires 768 direct outputs for each candidate")
    identities = {prompt: identity for identity, prompt in pairs.items()}
    counts = {key: {p: np.zeros(4, dtype=np.float64) for p in identities} for key in keys}
    seen = set()
    for row in direct:
        if not isinstance(row, dict):
            raise ValueError("Direct output record must be a mapping")
        cid, prompt, sample = row.get("candidate_id"), row.get("prompt_id"), row.get("sample_key")
        if (
            not isinstance(cid, str)
            or cid not in keys
            or not isinstance(prompt, str)
            or prompt not in identities
            or not isinstance(sample, str)
            or not sample
            or (cid, sample) in seen
        ):
            raise ValueError(
                "Missing, unknown, or duplicate direct candidate/prompt/sample identity"
            )
        # A sample ID is only unique within its candidate. Matching IDs or seeds
        # across candidates neither creates nor strengthens statistical pairing.
        seen.add((cid, sample))
        if (
            (row.get("family"), row.get("base_scene_id"), row.get("interface"))
            != identities[prompt]
            or row.get("category") not in CATEGORIES
            or row.get("split", "control") != "control"
            or row.get("decode_mode", "sample") not in ("sample", "sampled")
            or (
                "bank_index" in row
                and (type(row["bank_index"]) is not int or row["bank_index"] != bank_index)
            )
            or (
                "checkpoint_step" in row
                and (type(row["checkpoint_step"]) is not int or row["checkpoint_step"] != 64)
            )
        ):
            raise ValueError("Direct scene/category/warm-bank protocol does not match proposal")
        counts[cid][prompt][CATEGORIES.index(row["category"])] += 1
    components, support = {}, {}
    for cid in keys:
        if any(value.sum() != 16 for value in counts[cid].values()):
            raise ValueError("Every direct candidate requires exactly K16 for all 48 prompts")
        by_prompt = {
            p: np.asarray([v[0], v[1], v[:3].sum()]) / v.sum() for p, v in counts[cid].items()
        }
        components[cid] = {
            family: np.asarray(
                [
                    [by_prompt[pairs[(family, scene, interface)]] for interface in INTERFACES]
                    for scene in panel["base_scene_ids_by_family"][family]
                ]
            )
            for family in panel["families"]
        }
        group_counts = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        for prompt, values in counts[cid].items():
            family, _, interface = identities[prompt]
            group_counts[f"{family}/{interface}"] += values
        support[cid] = {
            "by_prompt": {p: _support(v) for p, v in sorted(counts[cid].items())},
            "by_group": {g: _support(v) for g, v in sorted(group_counts.items())},
            "overall": _support(np.sum(list(counts[cid].values()), axis=0)),
            "weighting": "Equal prompts within group; six groups fixed at 1/6",
            "individual_outputs_paired": False,
        }
    return components, support


def _ratio(width, effect):
    if width is None or effect is None or effect == 0:
        return None
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = np.float64(width) / abs(np.float64(effect))
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite CI half-width/effect ratio")
    return float(value)


def _ci(samples, point, predicted, clusters):
    result = _interval(samples)
    warnings = []
    if result["undefined_replicates"]:
        warnings.append("UNDEFINED_VALID_DENOMINATOR")
    elif float(np.ptp(samples)) <= 8 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(samples)))
    ):
        warnings.append("DEGENERATE_BOOTSTRAP")
    unstable = bool(warnings)
    if clusters <= SMALL_PANEL_SCENES_PER_FAMILY:
        warnings.append("SMALL_SCENE_PANEL")
    return {
        **result,
        "status": "UNSTABLE"
        if unstable
        else "ESTIMATED_WITH_WARNINGS"
        if warnings
        else "ESTIMATED",
        "warnings": warnings,
        "confidence_level": 0.95,
        "half_width_to_abs_point_estimate": _ratio(result["half_width"], point),
        "half_width_to_abs_predicted_delta": _ratio(result["half_width"], predicted),
        "minimum_within_family_scene_count": clusters,
        "coverage_scope": "POINTWISE_EVALUATION_SCENE_UNCERTAINTY_ONLY",
    }


def _metric_arrays(summary):
    return tuple({scope: _metrics(value) for scope, value in part.items()} for part in summary)


def _predicted_delta(target, reference, paired_components):
    # Preserve the ordinary-IS paired weight-difference estimator for pX and v;
    # qX/qS are differences of separately recomputed pooled ratios.
    return tuple(
        {scope: _deltas(paired[scope], left[scope], right[scope]) for scope in left}
        for left, right, paired in zip(target, reference, paired_components, strict=True)
    )


def _subtract(left, right):
    with np.errstate(over="raise", invalid="raise"):
        return tuple(
            {scope: a[scope] - b[scope] for scope in a} for a, b in zip(left, right, strict=True)
        )


def _comparison(predicted, observed, panel, *, target, reference):
    residual = _subtract(observed, predicted)
    responses = {}
    for scope in predicted[0]:
        families = panel["families"] if scope == "overall" else [scope.split("/", 1)[0]]
        clusters = min(len(panel["base_scene_ids_by_family"][f]) for f in families)
        responses[scope] = {}
        for index, metric in enumerate(METRICS):
            arrays = {
                "predicted_delta": predicted,
                "observed_delta": observed,
                "residual": residual,
            }
            point = {name: _value(value[0][scope][index]) for name, value in arrays.items()}
            intervals = {
                name: _ci(
                    value[1][scope][:, index], point[name], point["predicted_delta"], clusters
                )
                for name, value in arrays.items()
            }
            warnings = sorted({w for interval in intervals.values() for w in interval["warnings"]})
            resolution = []
            expected = point["predicted_delta"]
            if expected is None:
                resolution.append("UNDEFINED_PREDICTED_EFFECT")
            elif expected == 0:
                resolution.append("ZERO_PREDICTED_EFFECT")
            relative_width = intervals["observed_delta"]["half_width_to_abs_predicted_delta"]
            if relative_width is None or relative_width >= 1:
                resolution.append("DIRECT_CI_DOES_NOT_RESOLVE_PREDICTED_EFFECT_SCALE")
            responses[scope][metric] = {
                **point,
                "ci": intervals,
                "status": "INCONCLUSIVE"
                if resolution or any(ci["status"] == "UNSTABLE" for ci in intervals.values())
                else "ESTIMATED_WITH_WARNINGS"
                if warnings
                else "ESTIMATED",
                "warnings": warnings,
                "resolution_reasons": resolution,
                "residual_zero_is_proof_of_prediction_accuracy": False,
            }
    return {
        "target_candidate_id": target,
        "reference": reference,
        "responses": responses,
        "residual_definition": "observed_delta - predicted_delta",
        "status": "INCONCLUSIVE"
        if any(v["status"] == "INCONCLUSIVE" for s in responses.values() for v in s.values())
        else "ESTIMATED_WITH_WARNINGS",
    }


def analyze_direct_validation(
    proposal_records,
    candidate_logprobs,
    direct_records,
    *,
    bank_index,
    baseline_key,
    candidate_key,
    bootstrap_replicates=5000,
    seed=20260909,
):
    """Compare the predeclared lambda-zero/one direct outputs for bank 0 or 6.

    Candidate maps have exactly the same shape as analyze_control_responses.
    Direct rows bind candidate_id, sample_key, prompt_id, base_scene_id, family,
    interface, category; supplied checkpoint_step/bank_index must match warm64.
    Raises on malformed/nonfinite evidence. INCONCLUSIVE is a completed analysis,
    never a reason to increase training LR or silently add more generations.
    """
    if (
        not isinstance(candidate_key, str)
        or not candidate_key
        or candidate_key == baseline_key
        or not isinstance(candidate_logprobs, dict)
        or candidate_key not in candidate_logprobs
    ):
        raise ValueError("Distinct baseline and candidate keys with complete likelihoods required")
    control = analyze_control_responses(
        proposal_records,
        candidate_logprobs,
        baseline_key=baseline_key,
        bank_index=bank_index,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    records, panel, indices, pairs = _panel(proposal_records, 8, 16)
    direct, support = _direct_panels(
        direct_records, panel, pairs, (baseline_key, candidate_key), bank_index
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

    def summarize(weights):
        return _summaries(_components(weights, indicators, panel, indices, pairs), draws)

    theta = _metric_arrays(summarize(np.ones(len(records))))
    weights = {
        k: _weights(records, candidate_logprobs[k])[0] for k in (baseline_key, candidate_key)
    }
    prediction = {key: _metric_arrays(summarize(value)) for key, value in weights.items()}
    observed = {key: _metric_arrays(_summaries(value, draws)) for key, value in direct.items()}
    comparisons = {}
    for name, key in (("absolute_baseline", baseline_key), ("absolute_candidate", candidate_key)):
        comparisons[name] = _comparison(
            _predicted_delta(prediction[key], theta, summarize(weights[key] - 1)),
            _subtract(observed[key], theta),
            panel,
            target=key,
            reference="WARM_THETA_PROPOSAL",
        )
    comparisons["auxiliary"] = _comparison(
        _predicted_delta(
            prediction[candidate_key],
            prediction[baseline_key],
            summarize(weights[candidate_key] - weights[baseline_key]),
        ),
        _subtract(observed[candidate_key], observed[baseline_key]),
        panel,
        target=candidate_key,
        reference=baseline_key,
    )
    warnings = {w for candidate in control["candidates"].values() for w in candidate["warnings"]}
    for candidate in support.values():
        for item in [
            *candidate["by_prompt"].values(),
            *candidate["by_group"].values(),
            candidate["overall"],
        ]:
            warnings.update(item["warnings"])
    for comparison in comparisons.values():
        for scope in comparison["responses"].values():
            for metric in scope.values():
                warnings.update(metric["warnings"])
    result = {
        "execution_kind": "CPU_MATH",
        "analysis_completed": True,
        "status": "INCONCLUSIVE"
        if any(c["status"] == "INCONCLUSIVE" for c in comparisons.values())
        else "ESTIMATED_WITH_WARNINGS"
        if warnings
        else "ESTIMATED",
        "bank_index": bank_index,
        "baseline_key": baseline_key,
        "candidate_key": candidate_key,
        "proposal_sequences": len(records),
        "direct_sequences": len(direct_records),
        "direct_records_sha256": canonical_hash(
            sorted(direct_records, key=lambda r: (r["candidate_id"], r["sample_key"]))
        ),
        "control_IS": control,
        "comparisons": comparisons,
        "direct_estimates": {
            cid: {
                scope: {m: _value(v) for m, v in zip(METRICS, values, strict=True)}
                for scope, values in summary[0].items()
            }
            for cid, summary in observed.items()
        },
        "direct_support": support,
        "direct_not_measured_candidate_keys": sorted(
            set(candidate_logprobs) - {baseline_key, candidate_key}
        ),
        "bootstrap_contract": {
            **control["bootstrap_contract"],
            "proposal_and_direct_draws_shared": True,
            "individual_outputs_paired": False,
            "individual_tokens_paired": False,
            "coupling": (
                "Caller records CRN method; equal seeds do not imply natural output identity"
            ),
            "CI_scope": (
                "Pointwise scene-cluster uncertainty conditional on realized proposal/direct "
                "samples and fixed candidates; excludes training-seed uncertainty"
            ),
            "simultaneous_group_coverage": "NOT_ESTIMATED",
            "small_panel_display_threshold": {
                "at_most_scenes_per_family": SMALL_PANEL_SCENES_PER_FAMILY,
                "purpose": "Warning only; not a significance or coverage guarantee",
            },
        },
        "warnings": sorted(warnings),
        "safety_status": "NOT_CERTIFIED",
        "can_use_for_safety_decision": False,
        "interpretation": (
            "Only auxiliary compares lambda-one to lambda-zero. Absolute changes use the warm "
            "theta proposal. All residuals are observed minus predicted; matching estimates, "
            "overlap diagnostics, or intervals containing zero do not establish safety or accuracy."
        ),
    }
    canonical_hash(result)
    return result
