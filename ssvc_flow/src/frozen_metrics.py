"""Descriptive frozen metrics with exact, pointwise per-prompt binomial bounds.

Repeated generations estimate conditional prompt probabilities; they are never
presented as independent scenes or as evidence about a training intervention.
"""

from __future__ import annotations

from collections import defaultdict

from scipy.stats import beta

from .statistics import CATEGORIES, summarize

FAMILIES = ("duplicate_encoding", "cross_series", "trend")
INTERFACES = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
PROBABILITY_GROUP_SIZES = (2, 4, 8, 16, 32)
ACTUAL_GROUP_SIZES = (8, 16)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _interval(successes, trials):
    return {
        "low": float(beta.ppf(0.025, successes, trials - successes + 1)) if successes else 0.0,
        "high": float(beta.ppf(0.975, successes + 1, trials - successes))
        if successes < trials
        else 1.0,
        "confidence": 0.95,
        "method": "Clopper-Pearson two-sided exact pointwise conditional-on-prompt",
        "success_count": successes,
        "trial_count": trials,
    }


def _probabilities(p, interval):
    low, high = interval["low"], interval["high"]
    result = {}
    for k in PROBABILITY_GROUP_SIZES:

        def informative(value, size=k):
            return max(0.0, 1 - value**size - (1 - value) ** size)

        candidates = [informative(low), informative(high)]
        if low <= 0.5 <= high:
            candidates.append(informative(0.5))
        result[str(k)] = {
            "no_X": {"estimate": (1 - p) ** k, "low": (1 - high) ** k, "high": (1 - low) ** k},
            "informative_X_reward": {
                "estimate": informative(p),
                "low": min(candidates),
                "high": max(candidates),
            },
            "uncertainty": (
                "image of pointwise pX confidence interval; not simultaneous across prompts"
            ),
        }
    return result


def _actual_groups(rows):
    prompts = defaultdict(list)
    for row in rows:
        prompts[row["prompt_id"]].append(row)
    result = {}
    for k in ACTUAL_GROUP_SIZES:
        groups, unused = [], 0
        for prompt in prompts.values():
            ordered = sorted(prompt, key=lambda row: row["rollout_index"])
            usable = len(ordered) // k * k
            groups.extend(ordered[start : start + k] for start in range(0, usable, k))
            unused += len(ordered) - usable
        counts = {
            "zero_reward_variance_X": sum(
                sum(row["category"] == "X" for row in group) in (0, k) for group in groups
            ),
            **{
                f"contains_{category}": sum(
                    any(row["category"] == category for row in group) for group in groups
                )
                for category in ("X", "S", "I")
            },
        }
        result[str(k)] = {
            "status": "OBSERVED" if groups else "NA",
            "group_count": len(groups),
            "unused_rollout_count": unused,
            "grouping": "nonoverlapping contiguous rollout_index blocks within each prompt",
            **{
                name: {"count": count, "fraction": _ratio(count, len(groups))}
                for name, count in counts.items()
            },
        }
    return result


def _aggregate(rows, *, sampling):
    if not rows:
        return {"status": "NA", "reason": "no fixed prompts in this group", "rollout_count": 0}
    summary = summarize(rows)
    probability_keys = [*(f"p{c}" for c in CATEGORIES), "v", "qX", "qS", "pA", "truth_given_answer"]
    known = [row for row in rows if row["constraint_satisfaction"] is not None]
    satisfied = sum(row["constraint_satisfaction"] is True for row in rows)
    result = {
        "status": "OBSERVED",
        "pooled": {key: summary.pop(key) for key in probability_keys},
        **summary,
        "copy_observation_count": sum(row["copy_observation"] for row in rows),
        "copy_observation_rate": sum(row["copy_observation"] for row in rows) / len(rows),
        "constraint_satisfaction": {
            "satisfied_count": satisfied,
            "known_valid_count": len(known),
            "unknown_count": len(rows) - len(known),
            "invalid_count": sum(row["category"] == "I" for row in rows),
            "rate_among_known_valid": _ratio(satisfied, len(known)),
            "satisfied_fraction_all_outputs": satisfied / len(rows) if known else None,
        },
        "generation": {
            "actual_token_count": sum(row["completion_length"] for row in rows),
            "mean_completion_length": sum(row["completion_length"] for row in rows) / len(rows),
            "completion_lengths": [row["completion_length"] for row in rows],
            "eos_stop_count": sum(row["stop_reason"] == "eos" for row in rows),
            "length_stop_count": sum(row["stop_reason"] == "length" for row in rows),
            "length_stop_complete_valid_count": sum(
                row["stop_reason"] == "length" and row["category"] != "I" for row in rows
            ),
            "length_stop_partial_or_invalid_count": sum(
                row["stop_reason"] == "length" and row["category"] == "I" for row in rows
            ),
            "length_stop_complete_syntax_invalid_action_count": sum(
                row["stop_reason"] == "length" and row["syntax_valid"] and row["category"] == "I"
                for row in rows
            ),
            "length_stop_incomplete_or_invalid_syntax_count": sum(
                row["stop_reason"] == "length" and not row["syntax_valid"] for row in rows
            ),
            "truncation_definition": (
                "length stop split by track-specific complete action validity (category != I); "
                "complete syntax outside the action domain is separately counted"
            ),
        },
    }
    if sampling:
        result["actual_groups"] = _actual_groups(rows)
    else:
        result["estimator"] = "descriptive fixed-prompt greedy outcomes; not sampling probabilities"
    return result


def _validate(rows, expected_rollouts, track):
    if type(expected_rollouts) is not int or expected_rollouts < 1:
        raise ValueError("expected_rollouts must be a positive integer")
    if track not in ("N", "L"):
        raise ValueError("track must be N or L")
    if not rows:
        raise ValueError("no observations")
    required = (
        "prompt_id",
        "base_scene_id",
        "constraint_family",
        "interface",
        "category",
        "decode_mode",
        "rollout_index",
        "completion_length",
        "stop_reason",
        "syntax_valid",
        "copy_observation",
        "constraint_satisfaction",
    )
    sample_keys, identities, prompts = set(), set(), defaultdict(list)
    use_keys = any("sample_key" in row for row in rows)
    for row in rows:
        if any(key not in row for key in required):
            raise ValueError("missing required observation fields")
        if any(
            not isinstance(row[key], str) or not row[key]
            for key in ("prompt_id", "base_scene_id", "constraint_family", "interface")
        ):
            raise ValueError("prompt metadata must contain nonempty strings")
        if track == "N" and (
            row["constraint_family"] not in FAMILIES or row["interface"] not in INTERFACES
        ):
            raise ValueError("unknown N family or interface")
        if track == "L" and row["interface"] not in ("collision", "separating"):
            raise ValueError("unknown L interface")
        if row["category"] not in CATEGORIES or row["decode_mode"] not in ("sample", "greedy"):
            raise ValueError("unknown category or decode_mode")
        if type(row["rollout_index"]) is not int or row["rollout_index"] < 0:
            raise ValueError("rollout_index must be a nonnegative integer")
        if type(row["completion_length"]) is not int or row["completion_length"] < 0:
            raise ValueError("completion_length must be a nonnegative integer")
        if row["stop_reason"] not in ("eos", "length"):
            raise ValueError("unknown stop_reason")
        if type(row["syntax_valid"]) is not bool or type(row["copy_observation"]) is not bool:
            raise ValueError("validity and copy flags must be booleans")
        action_valid = row["category"] != "I"
        if (track == "N" and row["syntax_valid"] != action_valid) or (
            action_valid and not row["syntax_valid"]
        ):
            raise ValueError("category disagrees with strict syntax validity")
        if "semantic_parse_success" in row and (
            type(row["semantic_parse_success"]) is not bool
            or row["semantic_parse_success"] != action_valid
        ):
            raise ValueError("semantic parse validity disagrees with action category")
        if row["constraint_satisfaction"] is not None and (
            type(row["constraint_satisfaction"]) is not bool or not action_valid
        ):
            raise ValueError("constraint satisfaction must be boolean for valid outputs or null")
        if use_keys:
            key = row.get("sample_key")
            if not isinstance(key, str) or not key or key in sample_keys:
                raise ValueError("missing or duplicate sample_key")
            sample_keys.add(key)
        identity = (row["prompt_id"], row["decode_mode"], row["rollout_index"])
        if identity in identities:
            raise ValueError("duplicate prompt/decode/index identity")
        identities.add(identity)
        prompts[row["prompt_id"]].append(row)
    for prompt in prompts.values():
        metadata = {
            (row["base_scene_id"], row["constraint_family"], row["interface"]) for row in prompt
        }
        if len(metadata) != 1:
            raise ValueError("inconsistent prompt metadata")
        samples = [row for row in prompt if row["decode_mode"] == "sample"]
        greedy = [row for row in prompt if row["decode_mode"] == "greedy"]
        if sorted(row["rollout_index"] for row in samples) != list(range(expected_rollouts)):
            raise ValueError("sampling rollout indices are incomplete")
        if len(greedy) != 1:
            raise ValueError("each prompt requires exactly one independent greedy output")
    return prompts


def build_frozen_metrics(rows, expected_rollouts=16, *, track="N"):
    """Summarize complete frozen prompt banks without modifying caller rows.

    N reports the six prescribed groups, including explicit NA for absent groups;
    L reports observed legacy family/condition groups. Dataset completeness and
    fixed scene manifests are checked by the caller before this pure aggregation.
    """
    rows = list(rows)
    prompts = _validate(rows, expected_rollouts, track)
    group_keys = (
        [(f, i) for f in FAMILIES for i in INTERFACES]
        if track == "N"
        else sorted({(row["constraint_family"], row["interface"]) for row in rows})
    )
    modes = {}
    for mode in ("sample", "greedy"):
        selected = sorted(
            (row for row in rows if row["decode_mode"] == mode),
            key=lambda row: (row["prompt_id"], row["rollout_index"]),
        )
        by_prompt = {}
        for prompt_id in sorted(prompts):
            group = [row for row in selected if row["prompt_id"] == prompt_id]
            entry = _aggregate(group, sampling=mode == "sample")
            entry.update(
                {key: group[0][key] for key in ("base_scene_id", "constraint_family", "interface")}
            )
            if mode == "sample":
                entry["category_probability_intervals"] = {
                    c: _interval(entry["category_counts"][c], len(group)) for c in CATEGORIES
                }
                entry["pX_interval"] = entry["category_probability_intervals"]["X"]
                entry["group_probability_estimates"] = _probabilities(
                    entry["pooled"]["pX"], entry["pX_interval"]
                )
            by_prompt[prompt_id] = entry
        modes[mode] = {
            "overall": _aggregate(selected, sampling=mode == "sample"),
            "by_prompt": by_prompt,
            "by_group": {
                f"{family}/{interface}": _aggregate(
                    [
                        row
                        for row in selected
                        if row["constraint_family"] == family and row["interface"] == interface
                    ],
                    sampling=mode == "sample",
                )
                for family, interface in group_keys
            },
        }
    return {
        "schema_version": 1,
        "track": track,
        "expected_rollouts_per_prompt": expected_rollouts,
        "sampling_decoder": "sample only; greedy excluded from all sampling estimands",
        "scope": "frozen support diagnostics; cannot establish training reward safety",
        **modes["sample"],
        "greedy": modes["greedy"],
    }
