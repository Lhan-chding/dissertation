"""Pure CPU R4 engineering KL alarms and paired sampled endpoint statistics.

The caller binds provenance, supplies generated-token masks, and writes artifacts.
No generation, parameter update, smoothing, or safety certification occurs here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .r3_response import _digest, _logprobs, _metrics

CATEGORIES = ("X", "S", "W", "I")
METRICS = ("pX", "v", "qX_pool", "qS_pool")
N_INTERFACES = ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")
MIN_STABLE_SCENES = 8


def _identity(row):
    family = row.get("family", row.get("constraint_family"))
    values = (family, row.get("base_scene_id"), row.get("interface"), row.get("prompt_id"))
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Nonempty family/base_scene_id/interface/prompt_id required")
    return values[:3], values[3]


def _layout(prompts, track):
    scenes, pairs = {}, {}
    for prompt, item in prompts.items():
        family, scene, interface = item["identity"]
        if scene in scenes and scenes[scene] != family:
            raise ValueError("A base scene belongs to multiple families")
        scenes[scene] = family
        identity = (family, scene, interface)
        if identity in pairs:
            raise ValueError("Duplicate scene/interface prompt")
        pairs[identity] = prompt
    families = sorted(set(scenes.values()))
    interfaces = sorted({identity[2] for identity in pairs})
    if track in ("N", "OOD") and tuple(interfaces) != N_INTERFACES:
        raise ValueError("N and graph-OOD require both locked interfaces")
    if track == "N" and len(families) != 3:
        raise ValueError("N requires six family/interface groups")
    if track == "OOD" and families != ["cross_series"]:
        raise ValueError("Graph-OOD only supports cross_series")
    if track == "L" and interfaces != ["collision", "separating"]:
        raise ValueError("L requires its separate collision/separating protocol")
    if not families:
        raise ValueError("Empty endpoint panel")
    ids = {f: sorted(s for s, family in scenes.items() if family == f) for f in families}
    for family, values in ids.items():
        for scene in values:
            if any((family, scene, interface) not in pairs for interface in interfaces):
                raise ValueError("Every scene must retain paired interfaces")
    return {
        "families": families,
        "interfaces": interfaces,
        "scene_ids": ids,
        "pairs": pairs,
        "groups": [f"{f}/{i}" for f in families for i in interfaces],
        "base_scenes": len(scenes),
        "prompts": len(prompts),
    }


def control_kl_diagnostic(proposal_records, candidate_logprobs, *, eos_token_ids):
    """Conditional-token k3 at fixed step0 prefixes; strict engineering stops.

    Within each prompt the denominator is all selected generated tokens across
    its sequences, including terminal EOS. Prompts are then averaged equally in
    each of six groups, and groups receive weight 1/6. Sequence p99 uses the
    unweighted empirical list of absolute sequence log-ratios, separately named.
    """
    if not proposal_records or not isinstance(candidate_logprobs, dict):
        raise ValueError("Nonempty proposal rows and keyed candidate scores required")
    eos = set(eos_token_ids)
    if not eos or any(type(t) is not int or t < 0 for t in eos):
        raise ValueError("Locked EOS token IDs required")
    keys, prompts, sequences = set(), {}, []
    for row in proposal_records:
        key = row.get("sample_key")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Unique nonempty proposal sample keys required")
        keys.add(key)
        if key not in candidate_logprobs:
            raise ValueError("Missing candidate sequence")
        identity, prompt = _identity(row)
        if row.get("split", "control") != "control":
            raise ValueError("KL proposal must use control")
        tokens = row["token_ids"]
        old, new = _logprobs(row["old_logprobs"]), _logprobs(candidate_logprobs[key])
        if not tokens or old.shape != new.shape or len(tokens) != old.size:
            raise ValueError("Old/candidate/action token masks differ")
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid generated token ID")
        if any(t in eos for t in tokens[:-1]):
            raise ValueError("Token mask continues past EOS")
        if row.get("stop_reason") == "eos" and tokens[-1] not in eos:
            raise ValueError("Terminal EOS is missing from the scored action")
        with np.errstate(over="raise", invalid="raise"):
            d = new - old
            token_k3 = np.expm1(d) - d
            sequence_ratio = float(d.sum())
            numerator = float(token_k3.sum())
        if not np.isfinite(token_k3).all() or not np.isfinite(sequence_ratio):
            raise FloatingPointError("Nonfinite KL or sequence log-ratio")
        if prompt in prompts and prompts[prompt]["identity"] != identity:
            raise ValueError("Prompt identity changed within KL proposal")
        p = prompts.setdefault(
            prompt, {"identity": identity, "tokens": 0, "sequences": 0, "numerator": 0.0, "eos": 0}
        )
        p["tokens"] += len(tokens)
        p["sequences"] += 1
        p["numerator"] += numerator
        p["eos"] += int(tokens[-1] in eos)
        sequences.append(
            {
                "sample_key": key,
                "prompt_id": prompt,
                "selected_token_count": len(tokens),
                "includes_terminal_eos": tokens[-1] in eos,
                "sequence_log_ratio_candidate_minus_step0": sequence_ratio,
                "absolute_sequence_log_ratio": abs(sequence_ratio),
                "token_k3_sum": numerator,
                "mean_token_kl": numerator / len(tokens),
            }
        )
    if set(candidate_logprobs) != keys:
        raise ValueError("Extra candidate sample keys")
    layout = _layout(prompts, "N")
    by_prompt = {
        prompt: {
            "family": p["identity"][0],
            "base_scene_id": p["identity"][1],
            "interface": p["identity"][2],
            "selected_token_count": p["tokens"],
            "sequence_count": p["sequences"],
            "included_eos_tokens": p["eos"],
            "mean_token_kl": p["numerator"] / p["tokens"],
        }
        for prompt, p in prompts.items()
    }
    by_group = {}
    for group in layout["groups"]:
        values = [
            p["mean_token_kl"]
            for p in by_prompt.values()
            if f"{p['family']}/{p['interface']}" == group
        ]
        by_group[group] = {
            "mean_token_kl": float(np.mean(values)),
            "prompt_count": len(values),
            "fixed_weight": 1 / 6,
        }
    mean = float(np.mean([p["mean_token_kl"] for p in by_group.values()]))
    p99 = float(np.quantile([s["absolute_sequence_log_ratio"] for s in sequences], 0.99))
    if not np.isfinite(mean) or not np.isfinite(p99):
        raise FloatingPointError("Nonfinite aggregate KL diagnostic")
    alarms = {"mean_token_kl": mean > 0.1, "sequence_log_ratio_p99_abs": p99 > 2.0}
    return {
        "status": "STOP_DIAGNOSE" if any(alarms.values()) else "WITHIN_ENGINEERING_LIMITS",
        "mean_token_kl": mean,
        "sequence_log_ratio_p99_abs": p99,
        "should_stop": any(alarms.values()),
        "alarms": alarms,
        "thresholds": {
            "mean_token_kl": 0.1,
            "sequence_log_ratio_p99_abs": 2.0,
            "comparison": "strict_greater_than",
        },
        "by_prompt": by_prompt,
        "by_group": by_group,
        "sequence_records": sequences,
        "direction": "KL(step0_token_policy || candidate_token_policy) at step0 prefixes",
        "estimator": "exp(d)-d-1 evaluated stably as expm1(d)-d; d=candidate-old",
        "trajectory_kl": False,
        "candidate_state_occupancy_kl": False,
        "values_clipped": False,
        "denominator": {
            "selected_tokens": sum(p["tokens"] for p in prompts.values()),
            "included_eos_tokens": sum(p["eos"] for p in prompts.values()),
            "sequence_count": len(sequences),
            "prompt_count": len(prompts),
            "token_mean": "sum selected-token k3 / all selected tokens within each prompt",
            "aggregate_mean": "equal prompts within group, then six groups equally weighted",
            "sequence_p99": (
                "np.quantile(abs(sequence log-ratio), 0.99), unweighted over recorded sequences"
            ),
        },
        "interpretation": (
            "Finite-sample engineering diagnostic at step0 prefix occupancy; "
            "neither an exact KL nor a safety theorem"
        ),
    }


def _aggregate(rows, track, *, initial=False):
    if not rows:
        raise ValueError("Empty sampled endpoint")
    prompts, keys, steps = {}, set(), set()
    for row in rows:
        if row.get("track") != track:
            raise ValueError("Each endpoint and initial row must match its independent track")
        if row.get("decode_mode", "sample") not in ("sample", "sampled"):
            raise ValueError("Sampled and greedy endpoints cannot be mixed")
        if initial and row.get("arm", "INITIAL") != "INITIAL":
            raise ValueError("Initial records must be the shared INITIAL arm")
        step = row.get("checkpoint_step")
        if step is None and not initial:
            raise ValueError("Endpoint checkpoint_step is required")
        if step is not None:
            if type(step) is not int or step < 0 or (initial and step != 0):
                raise ValueError("Invalid initial or endpoint checkpoint step")
            steps.add(step)
        identity, prompt = _identity(row)
        key = row.get("sample_key")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Unique sampled endpoint keys required within each arm")
        keys.add(key)
        if row.get("category") not in CATEGORIES:
            raise ValueError("Unknown endpoint category")
        if prompt in prompts and prompts[prompt]["identity"] != identity:
            raise ValueError("Prompt identity changed within endpoint")
        item = prompts.setdefault(
            prompt, {"identity": identity, "counts": np.zeros(4, dtype=np.int64)}
        )
        item["counts"][CATEGORIES.index(row["category"])] += 1
    if len(steps) > 1:
        raise ValueError("Multiple checkpoint steps in one endpoint")
    return prompts, next(iter(steps), None)


def _weights(layout, supplied, track):
    groups = layout["groups"]
    if supplied is None and track == "L":
        # The fixed legacy prompt distribution is primary; output sample counts
        # never determine weights. Equal-family weights are a separate sensitivity.
        result = {
            g: len(layout["scene_ids"][g.split("/", 1)[0]]) / layout["prompts"] for g in groups
        }
    else:
        result = {g: 1 / len(groups) for g in groups} if supplied is None else dict(supplied)
    if (
        set(result) != set(groups)
        or any(not np.isfinite(w) or w <= 0 for w in result.values())
        or not np.isclose(sum(result.values()), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Fixed positive weights must cover all protocol groups and sum to one")
    if track in ("N", "OOD") and any(not np.isclose(w, 1 / len(groups)) for w in result.values()):
        raise ValueError("N and OOD require equal locked group weights")
    return result


def _arrays(prompts, layout):
    def components(prompt):
        counts = prompts[prompt]["counts"]
        return np.array([counts[0], counts[1], counts[:3].sum()], dtype=float) / counts.sum()

    return {
        family: np.array(
            [
                [components(layout["pairs"][(family, s, i)]) for i in layout["interfaces"]]
                for s in ids
            ]
        )
        for family, ids in layout["scene_ids"].items()
    }


def _summaries(arrays, layout, draws, weights):
    points, bootstrap = {}, {}
    for family, values in arrays.items():
        means = values.mean(axis=0)
        boot = values[draws[family]].mean(axis=1)
        for j, interface in enumerate(layout["interfaces"]):
            group = f"{family}/{interface}"
            points[group], bootstrap[group] = means[j], boot[:, j]
    points["overall"] = sum(points[g] * w for g, w in weights.items())
    bootstrap["overall"] = sum(bootstrap[g] * w for g, w in weights.items())
    return (
        {g: _metrics(p) for g, p in points.items()},
        {g: _metrics(p) for g, p in bootstrap.items()},
    )


def _value(x):
    return float(x) if np.isfinite(x) else None


def _degenerate(values):
    return np.ptp(values) <= 8 * np.finfo(float).eps * max(1, abs(float(values.mean())))


def _ci(values, min_scenes):
    undefined = int((~np.isfinite(values)).sum())
    reasons = []
    if undefined:
        reasons.append("UNDEFINED_VALID_DENOMINATOR")
    elif _degenerate(values):
        reasons.append("DEGENERATE_BOOTSTRAP")
    if min_scenes < MIN_STABLE_SCENES:
        reasons.append("SMALL_SCENE_PANEL")
    low, high = (None, None) if undefined else map(float, np.quantile(values, [0.025, 0.975]))
    return {
        "status": "UNSTABLE" if reasons else "ESTIMATED",
        "low": low,
        "high": high,
        "half_width": None if undefined else (high - low) / 2,
        "reasons": reasons,
        "undefined_replicates": undefined,
        "method": "PAIRED_SCENE_CLUSTER_PERCENTILE",
        "confidence_level": 0.95,
        "coverage_scope": "ONE_COMPARISON_ONE_GROUP_ONE_METRIC",
    }


def _evidence(ci):
    if ci["status"] != "ESTIMATED":
        return "UNSTABLE_NO_INFERENCE" if ci["status"] == "UNSTABLE" else "NOT_INCLUDED"
    if ci["low"] > 0:
        return "POSITIVE_INTERVAL_EXCLUDES_ZERO"
    if ci["high"] < 0:
        return "NEGATIVE_INTERVAL_EXCLUDES_ZERO"
    return "NO_DIRECTIONAL_INTERVAL_EVIDENCE"


def _joint(points, bootstrap, layout):
    result = {}
    for j, metric in enumerate(METRICS):
        samples = np.stack([bootstrap[g][:, j] for g in layout["groups"]], axis=1)
        effect = np.asarray([points[g][j] for g in layout["groups"]])
        undefined = int((~np.isfinite(samples)).any(axis=1).sum())
        reasons = []
        if undefined or not np.isfinite(effect).all():
            reasons.append("UNDEFINED_VALID_DENOMINATOR")
            critical = None
        else:
            critical = float(np.quantile(np.max(np.abs(samples - effect), axis=1), 0.95))
            if any(_degenerate(samples[:, i]) for i in range(samples.shape[1])):
                reasons.append("DEGENERATE_GROUP_BOOTSTRAP")
        if min(map(len, layout["scene_ids"].values())) < MIN_STABLE_SCENES:
            reasons.append("SMALL_SCENE_PANEL")
        result[metric] = {
            "status": "UNSTABLE" if reasons else "ESTIMATED",
            "method": "BOOTSTRAP_CENTERED_MAX_ABS_DEVIATION",
            "group_count": len(layout["groups"]),
            "groups": list(layout["groups"]),
            "critical_absolute_deviation": critical,
            "confidence_level": 0.95,
            "coverage_scope": "ONE_METRIC_ACROSS_DECLARED_GROUPS_ONLY",
            "undefined_replicates": undefined,
            "reasons": reasons,
        }
    return result


def _comparison(target, reference, *, track, repeats, seed, supplied_weights, role):
    if set(target) != set(reference) or any(
        target[p]["identity"] != reference[p]["identity"] for p in target
    ):
        raise ValueError("Paired comparisons require identical prompt/scene/track identities")
    layout = _layout(target, track)
    weights = _weights(layout, supplied_weights, track)
    rng = np.random.default_rng(seed)
    draws = {
        family: rng.integers(0, len(ids), size=(repeats, len(ids)))
        for family, ids in layout["scene_ids"].items()
    }
    left, left_boot = _summaries(_arrays(target, layout), layout, draws, weights)
    right, right_boot = _summaries(_arrays(reference, layout), layout, draws, weights)
    effects = {g: left[g] - right[g] for g in left}
    bootstrap = {g: left_boot[g] - right_boot[g] for g in left_boot}
    joint = _joint(effects, bootstrap, layout)
    responses = {}
    for group in effects:
        count = (
            min(map(len, layout["scene_ids"].values()))
            if group == "overall"
            else len(layout["scene_ids"][group.split("/", 1)[0]])
        )
        responses[group] = {}
        for j, metric in enumerate(METRICS):
            effect = _value(effects[group][j])
            ci = _ci(bootstrap[group][:, j], count)
            band = {**joint[metric]}
            critical = band["critical_absolute_deviation"]
            band.update(
                low=None if critical is None or effect is None else effect - critical,
                high=None if critical is None or effect is None else effect + critical,
            )
            if group == "overall":
                band = {
                    "status": "NOT_INCLUDED",
                    "low": None,
                    "high": None,
                    "coverage_scope": "GROUP_MAX_STAT_EXCLUDES_OVERALL",
                }
            responses[group][metric] = {
                "estimate": effect,
                "target_estimate": _value(left[group][j]),
                "reference_estimate": _value(right[group][j]),
                "pointwise_ci": ci,
                "simultaneous_ci": band,
                "point_direction": "UNDEFINED"
                if effect is None
                else "NEGATIVE_POINT_TREND"
                if effect < 0
                else "POSITIVE_POINT_TREND"
                if effect > 0
                else "ZERO_POINT_ESTIMATE",
                "pointwise_evidence": _evidence(ci),
                "simultaneous_evidence": _evidence(band),
                "excluded_decline_magnitude_above_pointwise": max(0.0, -ci["low"])
                if ci["status"] == "ESTIMATED"
                else None,
                "excluded_decline_magnitude_above_simultaneous": max(0.0, -band["low"])
                if band["status"] == "ESTIMATED"
                else None,
                "decline_exclusion_interpretation": (
                    "Only decreases larger than the reported magnitude are excluded by "
                    "the named approximate interval; not a no-harm guarantee"
                ),
                "absence_of_significance_implies_no_harm": False,
            }
    reversal = {}
    for group, values in responses.items():
        v = values["v"]["estimate"]
        negative = any(
            values[m]["estimate"] is not None and values[m]["estimate"] < 0
            for m in ("pX", "qX_pool")
        )
        reversal[group] = {
            "point_estimate_reversal_candidate": v is not None and v > 0 and negative,
            "joint_statistical_evidence": False,
            "validity_pointwise_evidence": values["v"]["pointwise_evidence"],
            "truth_pointwise_evidence": {
                m: values[m]["pointwise_evidence"] for m in ("pX", "qX_pool")
            },
            "interpretation": (
                "Joint reversal label is a point trend; separate metric interval evidence "
                "is reported independently, without joint cross-metric coverage"
            ),
        }

    def counts(prompts):
        return {
            p: {
                "n": int(item["counts"].sum()),
                **{f"n_{c}": int(n) for c, n in zip(CATEGORIES, item["counts"], strict=True)},
            }
            for p, item in sorted(prompts.items())
        }

    return {
        "status": "ESTIMATED_WITH_UNSTABLE_COMPONENTS"
        if any(c["status"] == "UNSTABLE" for c in joint.values())
        else "ESTIMATED",
        "panel_role": role,
        "responses": responses,
        "simultaneous_contract": joint,
        "reversal_diagnostics": reversal,
        "panel": {
            "groups": layout["groups"],
            "base_scenes": layout["base_scenes"],
            "prompts": layout["prompts"],
            "base_scene_ids_by_family": layout["scene_ids"],
            "fixed_group_weights": weights,
            "weighting_role": "EXPLICIT_WEIGHT_SENSITIVITY"
            if supplied_weights is not None
            else "PRIMARY_FIXED_PROMPT_DISTRIBUTION"
            if track == "L"
            else "PRIMARY_EQUAL_PROTOCOL_GROUPS",
            "target_counts_by_prompt": counts(target),
            "reference_counts_by_prompt": counts(reference),
        },
        "bootstrap_draws_sha256": {f: _digest(idx.tolist()) for f, idx in draws.items()},
    }


def analyze_sampled_endpoints(
    endpoint_records,
    initial_records=None,
    *,
    track,
    bootstrap_replicates=5000,
    seed=20260909,
    group_weights=None,
):
    """Analyze one protocol/checkpoint, with optional same-protocol initial panel.

    The primary contrast uses every paired endpoint prompt. Initial changes use
    exactly the supplied preexisting initial panel, which must be a subset of
    both endpoint arms. No missing prompts are silently dropped or generated.
    """
    if track not in ("N", "L", "OOD"):
        raise ValueError("Independent track N, L, or OOD required")
    if (
        type(bootstrap_replicates) is not int
        or bootstrap_replicates < 2
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Valid bootstrap count and seed required")
    arms = defaultdict(list)
    for row in endpoint_records:
        if row.get("arm") not in ("X_BASE", "X_VALID"):
            raise ValueError("Endpoint rows need X_BASE or X_VALID arm")
        arms[row["arm"]].append(row)
    if set(arms) != {"X_BASE", "X_VALID"}:
        raise ValueError("Both endpoint arms are required")
    base, base_step = _aggregate(arms["X_BASE"], track)
    valid, valid_step = _aggregate(arms["X_VALID"], track)
    if base_step != valid_step:
        raise ValueError("Endpoint arms use different checkpoint steps")
    kwargs = {
        "track": track,
        "repeats": bootstrap_replicates,
        "seed": seed,
        "supplied_weights": group_weights,
    }
    comparisons = {
        "X_VALID_minus_X_BASE": _comparison(valid, base, role="FULL_PAIRED_ENDPOINT", **kwargs)
    }
    if initial_records is not None:
        initial, _ = _aggregate(initial_records, track, initial=True)
        if not set(initial).issubset(base) or not set(initial).issubset(valid):
            raise ValueError("Initial prompt panel is not fully present in both endpoint arms")
        for name, target in (("X_BASE", base), ("X_VALID", valid)):
            comparisons[f"{name}_minus_initial"] = _comparison(
                {p: target[p] for p in initial}, initial, role="FIXED_INITIAL_PANEL_ONLY", **kwargs
            )
    else:
        for name in ("X_BASE", "X_VALID"):
            comparisons[f"{name}_minus_initial"] = {
                "status": "NOT_MEASURED",
                "reason": "No verified same-protocol initial bank supplied",
            }
    return {
        "status": "ESTIMATED_WITH_UNSTABLE_COMPONENTS"
        if any(c["status"] == "ESTIMATED_WITH_UNSTABLE_COMPONENTS" for c in comparisons.values())
        else "ESTIMATED",
        "track": track,
        "checkpoint_step": base_step,
        "comparisons": comparisons,
        "generalization_scope": "CROSS_SERIES_GRAPH_STRUCTURE_ONLY"
        if track == "OOD"
        else "WITHIN_THIS_LOCKED_PROTOCOL",
        "bootstrap_contract": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "base_scene",
            "stratified_by": "family",
            "interfaces_and_arms_paired_within_scene": True,
            "within_scene_outputs_resampled": False,
            "all_ratios_recomputed_in_each_replicate": True,
            "training_seed_uncertainty_included": False,
            "joint_coverage_across_metrics_protocols_or_comparisons": False,
            "small_panel_display_threshold_scenes_per_family": MIN_STABLE_SCENES,
            "small_panel_threshold_interpretation": (
                "Conservative display warning, not a calibrated significance threshold"
            ),
            "undefined_or_degenerate": "UNSTABLE; no smoothing or replicate deletion",
        },
        "result_scope": (
            "Single-seed exploratory training response; intervals are approximate "
            "scene-bootstrap evidence, not safety certification"
        ),
    }
