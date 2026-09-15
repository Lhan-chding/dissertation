"""Frozen cross-stage RQ comparisons; no missing result becomes a p-value."""

from __future__ import annotations

import copy
import math

import numpy as np

from .io import canonical_hash
from .statistics import holm, seed_bootstrap

COMMON_TEST_SPEC = {
    "target_index": 0,
    "channels": ["group_pX", "group_v"],
    "metric": "mse",
    "channel_aggregation": "EQUAL_CHANNEL_MEAN",
    "test": "PAIRED_SEED_SIGN_FLIP_MEAN",
    "alternative": "two-sided",
    "null_sign_symmetry_assumed": True,
    "randomization_seed": 2026091503,
    "randomization_reps": 5000,
    "minimum_seeds": 6,
}
_RQ4_SHARED = {
    "kind": "MODEL_RULE",
    "estimator": "PRESERVE_XI",
    "selector": "BLOCK_PIVOT_QR",
    "n": 1024,
    "m": 8,
    "alpha": 1e-5,
    "design_seed": 2026091500,
    "regression": "RIDGE",
    "output_policy": "RAW4",
}
RQ4_TEMPLATE = {
    **copy.deepcopy(COMMON_TEST_SPEC),
    "id": "RQ4",
    "stage": "VLM",
    "scope": "COMMON_R_LESS_K_ORIGINS",
    "left": {**_RQ4_SHARED, "method": "RESPONSE_SVD", "rank_cap": 2},
    "right": {**_RQ4_SHARED, "method": "FULL_RIDGE", "rank_cap": "FULL"},
    "primary_arm": "X_BASE",
    "primary_anchors": [32, 96],
    "shift_arm": "X_VALID",
    "shift_anchors": [96],
    "reference": "SHARED_INDEPENDENT_PRECISION_RESOLVED",
    "binding_policy": "VLM_DEVELOPMENT_ONLY_NO_RELATION_CHANGE",
}
DIRECTION_MODELS = {"PCA", "RANDOM_Q", "RESPONSE_SVD", "GROUP_WEIGHTED_RESPONSE"}


def _validate_test(spec):
    for key, expected in COMMON_TEST_SPEC.items():
        if key == "randomization_seed":
            if type(spec.get(key)) is not int or spec[key] < 0:
                raise ValueError("Frozen randomization seed must be a nonnegative integer")
        elif type(spec.get(key)) is not type(expected) or spec.get(key) != expected:
            raise ValueError(f"Frozen primary test {key} must equal {expected}")


def validate_primary_family(family):
    """Validate invariant family structure, independent of CPU/VLM artifacts."""
    if not isinstance(family, dict) or set(family) != {
        "schema",
        "family_id",
        "family_size",
        "alpha",
        "correction",
        "hypotheses",
    }:
        raise ValueError("Complete frozen primary family schema required")
    if (
        family["schema"] != "ssvc-v3-primary-comparison-family-1"
        or family["family_id"] != "SSVC_V3_RQ1_RQ4"
        or family["family_size"] != 4
        or family["alpha"] != 0.05
        or family["correction"] != "HOLM"
    ):
        raise ValueError("Four-RQ Holm family identity or alpha differs")
    hypotheses = family["hypotheses"]
    if not isinstance(hypotheses, list) or [s.get("id") for s in hypotheses] != [
        "RQ1",
        "RQ2",
        "RQ3",
        "RQ4",
    ]:
        raise ValueError("Exactly ordered RQ1, RQ2, RQ3, RQ4 specifications required")
    for spec in hypotheses:
        _validate_test(spec)
        if spec["id"] != "RQ4" and (
            set(spec) != {*COMMON_TEST_SPEC, "id", "stage", "scope", "left", "right"}
            or spec["stage"] != "CPU"
        ):
            raise ValueError("Complete CPU comparison specification required")
    if hypotheses[3] != RQ4_TEMPLATE:
        raise ValueError("RQ4 relation must remain the frozen real-VLM transfer template")
    return copy.deepcopy(family)


def validate_primary_comparison_family(config, selected):
    """Validate endpoint identities against the actual frozen CPU design matrix."""
    family = selected.get("primary_comparison_family")
    if family is None:
        return None
    family = validate_primary_family(family)
    from .cpu_campaign import _digest, development_designs

    designs = {_digest(d)[:20]: d for d in development_designs(config, selected=selected)}
    for spec in family["hypotheses"][:3]:
        left, right = spec["left"], spec["right"]
        if spec["id"] == "RQ1":
            if (
                spec["scope"] != "ALL_CASES"
                or any(
                    set(e) != {"kind", "estimator", "n"}
                    or e["kind"] != "DIRECT_MEASURE"
                    or e["estimator"] not in selected["observation_methods"]
                    or e["n"] not in config["observation"]["total_draws_grid"]
                    for e in [left, right]
                )
                or left["n"] != right["n"]
                or left["estimator"] == right["estimator"]
            ):
                raise ValueError("RQ1 requires distinct direct estimators at the same frozen n")
            continue
        if any(
            set(e) != {"kind", "design_id"} or e["kind"] != "MODEL" or e["design_id"] not in designs
            for e in [left, right]
        ):
            raise ValueError(f"{spec['id']} endpoints must identify real frozen models")
        a, b = designs[left["design_id"]], designs[right["design_id"]]
        if any(a[k] != b[k] for k in ["n", "m", "alpha", "estimator", "design_seed"]):
            raise ValueError(f"{spec['id']} changes unmatched model/measurement settings")
        if spec["id"] == "RQ2":
            if (
                spec["scope"] != "ALL_CASES"
                or a["model"] != b["model"]
                or a["model"] not in {"FULL_RIDGE", "FULL_GLS"}
                or a["rank_cap"] != "FULL"
                or b["rank_cap"] != "FULL"
                or a["selector"] == b["selector"]
            ):
                raise ValueError("RQ2 requires the same FULL model and distinct selectors")
        else:
            models = [a, b]
            if (
                spec["scope"] != "COMMON_R_LESS_K_ORIGINS"
                or a["selector"] != b["selector"]
                or left == right
                or any(d["model"] not in DIRECTION_MODELS | {"FULL_RIDGE"} for d in models)
                or not any(
                    d["model"] in DIRECTION_MODELS and type(d["rank_cap"]) is int for d in models
                )
            ):
                raise ValueError("RQ3 requires matched linear directions and genuine reduction")
    return family


def paired_seed_inference(spec, seed_totals, *, bootstrap_seed, bootstrap_reps=5000):
    """Compare equal-weight channel means, retaining each seed's actual denominator."""
    _validate_test(spec)
    if bootstrap_reps != 5000:
        raise ValueError("Exactly 5000 frozen seed bootstrap replicates required")
    totals = copy.deepcopy(list(seed_totals))
    seeds = [row["seed"] for row in totals]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate independent training seed in paired totals")
    result = {
        "hypothesis_id": spec["id"],
        "spec_hash": canonical_hash(spec),
        "comparison_spec_hash": canonical_hash(spec),
        "stage": spec["stage"],
        "paired_seed_totals": totals,
        "independent_seeds": len(seeds),
        "metric": spec["metric"],
        "channels": spec["channels"],
        "channel_aggregation": "EQUAL_CHANNEL_MEAN",
        "scope": spec["scope"],
        "test": spec["test"],
        "alternative": spec["alternative"],
        "null_sign_symmetry_assumed": True,
        "null_assumption": (
            "Independent training-seed differences are sign-symmetric under the zero-effect null."
        ),
        "effect_interpretation": (
            "Mean across seeds of left minus right equal-channel MSE; negative favors left."
        ),
        "raw_pvalue": None,
        "status": "UNKNOWN",
        "estimate": None,
        "interval": None,
        "reps": 0,
        "channel_effects": {},
        "paired_seed_differences": {},
    }
    differences = {c: [] for c in spec["channels"]}
    reasons = []
    if len(seeds) < spec["minimum_seeds"]:
        reasons.append("TOO_FEW_INDEPENDENT_SEEDS")
    for row in totals:
        if row.get("status", "OBSERVED") != "OBSERVED":
            reasons.append(f"UNRESOLVED_SEED_{row['seed']}")
        if row.get("eligible_origin_count", 0) <= 0:
            reasons.append(f"NO_ELIGIBLE_ORIGIN_SEED_{row['seed']}")
        channels = row.get("channel_totals", {})
        if set(channels) != set(spec["channels"]):
            raise ValueError("Exact frozen channel totals and denominators required")
        for channel in spec["channels"]:
            values = channels[channel]
            count = values.get("count")
            left, right = values.get("left_loss_sum"), values.get("right_loss_sum")
            valid = (
                type(count) is int
                and count > 0
                and all(
                    type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in [left, right]
                )
            )
            if not valid:
                reasons.append(f"UNKNOWN_CHANNEL_{channel}_SEED_{row['seed']}")
            else:
                differences[channel].append((left - right) / count)
    if reasons:
        result["unresolved_reasons"] = sorted(set(reasons))
        return result
    effects = []
    for channel, values in differences.items():
        result["channel_effects"][channel] = seed_bootstrap(
            seeds, values, reps=5000, seed=bootstrap_seed
        )
        effects.append(values)
    delta = np.asarray(effects).mean(axis=0)
    result.update(seed_bootstrap(seeds, delta, reps=5000, seed=bootstrap_seed))
    rng = np.random.default_rng(spec["randomization_seed"])
    reps = spec["randomization_reps"]
    null = np.abs((rng.choice([-1.0, 1.0], size=(reps, len(seeds))) * delta).mean(axis=1))
    result.update(
        status="AVAILABLE",
        raw_pvalue=float((1 + np.count_nonzero(null >= abs(delta.mean()))) / (reps + 1)),
        paired_seed_differences=dict(zip(map(str, seeds), delta.tolist(), strict=True)),
        randomization_reps=reps,
        randomization_seed=spec["randomization_seed"],
    )
    return result


def summarize_primary_family(family, results):
    """Formal Holm requires all four actual RQ results; pending stages stay pending."""
    family = validate_primary_family(family)
    specs = {s["id"]: s for s in family["hypotheses"]}
    seen = {}
    for row in results:
        name = row.get("hypothesis_id")
        if (
            name not in specs
            or name in seen
            or row.get("spec_hash") != canonical_hash(specs[name])
            or row.get("stage") != specs[name]["stage"]
        ):
            raise ValueError("Primary result identity or frozen spec hash mismatch")
        p = row.get("raw_pvalue")
        if row.get("status") == "AVAILABLE" and (
            type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
        ):
            raise ValueError("Available test requires a valid actual p-value")
        if row.get("status") != "AVAILABLE" and p is not None:
            raise ValueError("Unavailable hypothesis cannot carry a p-value")
        seen[name] = row
    missing = [name for name in specs if name not in seen]
    unavailable = [name for name, row in seen.items() if row.get("status") != "AVAILABLE"]
    result = {
        "family_hash": canonical_hash(family),
        "family_size": 4,
        "alpha": 0.05,
        "correction": "HOLM",
        "status": "PENDING" if missing else "UNKNOWN" if unavailable else "COMPLETE",
        "missing_hypotheses": missing,
        "unavailable_hypotheses": unavailable,
        "holm_applied": False,
        "tests": [],
    }
    if not missing and not unavailable:
        adjusted = holm([seen[name]["raw_pvalue"] for name in specs])
        result["holm_applied"] = True
        result["tests"] = [
            {
                "hypothesis_id": name,
                "raw_pvalue": seen[name]["raw_pvalue"],
                "holm_adjusted_pvalue": float(p),
                "reject_at_family_alpha": bool(p <= 0.05),
            }
            for name, p in zip(specs, adjusted, strict=True)
        ]
    return result
