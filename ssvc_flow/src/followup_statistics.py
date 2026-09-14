"""CPU-only follow-up statistics; separate estimands, uncertainty, and support.

All direct estimates use predeclared prompt weights. Scene bootstrap draws are
shared across candidates and interfaces, and never resample completions. Fixed
panel Hoeffding bounds assume independent generation draws within each candidate;
candidate correlations are permitted by the union bound. Neither procedure is a
safety certificate or a model-selection/online-update rule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, fsum, isfinite, log, sqrt
from statistics import mean, stdev

import numpy as np

CATEGORIES = ("X", "S", "W", "I")
METRICS = ("pX", "pS", "pW", "pI", "v", "qX", "qS")
EVENT_METRICS = METRICS[:5]
IDENTITY = ("base_scene_id", "family", "interface", "train_seed", "bank_id", "bank_index")
SEED_FIELDS = (
    "seed_block",
    "sample_seed",
    "sample_index",
    "sample_ordinal",
    "generation_seed",
    "panel_seed",
)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _alpha(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError("alpha must lie strictly between zero and one")


def _panel(rows):
    """Normalize raw samples or per-prompt/per-seed-block count records."""
    rows = list(rows)
    if not rows:
        raise ValueError("A nonempty direct-sampling panel is required")
    mode = "counts" if "counts" in rows[0] else "raw"
    prompts, sample_keys, coverage, scene_families, scene_interfaces = {}, set(), Counter(), {}, {}
    ordinals = Counter()
    for row in rows:
        for key in ("prompt_id", "base_scene_id", "family", "interface"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Nonempty {key} required")
        if ("counts" in row) != (mode == "counts"):
            raise ValueError("Raw samples and aggregate counts cannot be mixed")
        if row.get("alias_of") is not None:
            raise ValueError("Alias records must reference a canonical panel, not add observations")
        if row.get("decode_mode", "sampled") not in ("sampled", "sample"):
            raise ValueError(
                "Direct probability estimates require sampled, not greedy, completions"
            )
        prompt = row["prompt_id"]
        identity = tuple(row.get(k) for k in IDENTITY)
        try:
            hash(identity)
        except TypeError as exc:
            raise ValueError("Panel metadata must be scalar") from exc
        if prompt not in prompts:
            prompts[prompt] = {
                "identity": identity,
                "counts": dict.fromkeys(CATEGORIES, 0),
                "n": 0,
                **{k: row.get(k) for k in IDENTITY},
            }
        elif prompts[prompt]["identity"] != identity:
            raise ValueError("Prompt scene/interface/family/train seed/bank metadata changed")
        scene, family, interface = (row[k] for k in ("base_scene_id", "family", "interface"))
        if scene in scene_families and scene_families[scene] != family:
            raise ValueError("A base scene occurs in multiple families")
        scene_families[scene] = family
        if (scene, interface) in scene_interfaces and scene_interfaces[scene, interface] != prompt:
            raise ValueError("Multiple prompt IDs for one scene/interface")
        scene_interfaces[scene, interface] = prompt
        if "sample_key" in row:
            if (
                not isinstance(row["sample_key"], str)
                or not row["sample_key"]
                or row["sample_key"] in sample_keys
            ):
                raise ValueError("Duplicate or invalid sample_key")
            sample_keys.add(row["sample_key"])
        block = tuple((key, row[key]) for key in SEED_FIELDS if key in row)
        try:
            hash(block)
        except TypeError as exc:
            raise ValueError("Seed block identifiers must be scalar") from exc
        if mode == "counts":
            counts, n = row["counts"], row.get("n")
            _integer(n, "n")
            if not isinstance(counts, dict) or set(counts) != set(CATEGORIES):
                raise ValueError("counts must contain exactly X/S/W/I")
            for count in counts.values():
                _integer(count, "category count", minimum=0)
            if sum(counts.values()) != n:
                raise ValueError("Category counts must sum to n")
            key = (prompt, block)
            if key in coverage:
                raise ValueError("Duplicate count block")
            coverage[key] = n
        else:
            if row.get("category") not in CATEGORIES:
                raise ValueError("Unknown category")
            n, counts = 1, {c: int(row["category"] == c) for c in CATEGORIES}
            # Older ledgers may lack seed fields; they still require the same
            # prompt sample counts. Explicit seed design is matched whenever present.
            key = (prompt, block or (("implicit_ordinal", ordinals[prompt]),))
            if key in coverage:
                raise ValueError("Duplicate sample seed block; provide distinct sample_index")
            coverage[key] = 1
            ordinals[prompt] += 1
        prompts[prompt]["n"] += n
        for category in CATEGORIES:
            prompts[prompt]["counts"][category] += counts[category]
    seeds = {p["train_seed"] for p in prompts.values()}
    banks = {(p["bank_id"], p["bank_index"]) for p in prompts.values()}
    if len(seeds) != 1 or len(banks) != 1:
        raise ValueError("One panel must use one training seed and one bank")
    return {"prompts": prompts, "coverage": coverage, "mode": mode}


def _fixed_weights(panel, fixed_weights):
    if not isinstance(fixed_weights, dict) or set(fixed_weights) != set(panel["prompts"]):
        raise ValueError("Fixed prompt weights must exactly cover the panel")
    for w in fixed_weights.values():
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not isfinite(w) or w < 0:
            raise ValueError("Fixed prompt weights must be finite and nonnegative")
    if not np.isclose(fsum(fixed_weights.values()), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Fixed prompt weights must sum to one")
    return dict(fixed_weights)


def _pair(panels):
    first = next(iter(panels.values()))
    for panel in panels.values():
        if panel["coverage"] != first["coverage"] or panel["mode"] != first["mode"]:
            raise ValueError("Paired candidate prompt/sample seed block coverage differs")
        if {p: r["identity"] for p, r in panel["prompts"].items()} != {
            p: r["identity"] for p, r in first["prompts"].items()
        }:
            raise ValueError("Paired candidate scene/family/interface/train seed/bank differs")


def _metrics(probabilities):
    x, s, w, invalid = probabilities
    valid = x + s + w
    return {
        "pX": x,
        "pS": s,
        "pW": w,
        "pI": invalid,
        "v": valid,
        "qX": x / valid if valid > 0 else None,
        "qS": s / valid if valid > 0 else None,
    }


def _summary(panel, weights):
    prompts = panel["prompts"]
    probs = [
        fsum(weights[p] * r["counts"][c] / r["n"] for p, r in prompts.items()) for c in CATEGORIES
    ]
    counts = {c: sum(r["counts"][c] for r in prompts.values()) for c in CATEGORIES}
    return {
        **_metrics(probs),
        "estimator": "DIRECT_SAMPLING",
        "estimand": "ratio_of_fixed_weight_means_for_qX_and_qS",
        "category_counts": counts,
        "rollout_count": sum(counts.values()),
        "prompt_count": len(prompts),
        "independent_scene_count": len({r["base_scene_id"] for r in prompts.values()}),
        "per_prompt_counts": {
            p: {"n": r["n"], "counts": dict(r["counts"]), "weight": weights[p]}
            for p, r in prompts.items()
        },
        "event_support": {c: "OBSERVED" if counts[c] else "UNOBSERVED" for c in CATEGORIES},
        "prompts_without_X": sum(r["counts"]["X"] == 0 for r in prompts.values()),
        "zero_count_interpretation": "unobserved support, not theoretical zero",
        "safety_status": "NOT_CERTIFIED",
    }


def summarize_counts(rows, *, fixed_weights):
    """Estimate a panel from categories or validated X/S/W/I count blocks."""
    panel = _panel(rows)
    return _summary(panel, _fixed_weights(panel, fixed_weights))


def _candidate_panels(candidate_rows, fixed_weights, baseline_key):
    if not candidate_rows or not isinstance(candidate_rows, dict):
        raise ValueError("Nonempty candidate mapping required")
    if baseline_key is None:
        baseline_key = next(iter(candidate_rows))
    if baseline_key not in candidate_rows:
        raise ValueError("Baseline must be in the canonical candidate panel")
    panels = {name: _panel(rows) for name, rows in candidate_rows.items()}
    _pair(panels)
    weights = _fixed_weights(next(iter(panels.values())), fixed_weights)
    return panels, weights, baseline_key


def _scopes(panel):
    prompts = panel["prompts"]
    scopes = {"overall": set(prompts)}
    for prompt, row in prompts.items():
        scopes.setdefault(f"{row['family']}/{row['interface']}", set()).add(prompt)
    return scopes


def _scope_panel(panel, members):
    return {**panel, "prompts": {p: row for p, row in panel["prompts"].items() if p in members}}


def _scope_weights(weights, members):
    mass = fsum(weights[p] for p in members)
    if mass <= 0:
        raise ValueError(
            "Every reported family/interface scope must have positive fixed weight mass"
        )
    return {p: weights[p] / mass for p in members}


def _interval(values, alpha):
    missing = int(np.count_nonzero(~np.isfinite(values)))
    result = {
        "low": None,
        "high": None,
        "status": "UNINFORMATIVE",
        "replicates": len(values),
        "undefined_replicates": missing,
        "defined_replicates": len(values) - missing,
    }
    if missing:
        return {**result, "reason": "Undefined replicates retained; none discarded"}
    if np.ptp(values) <= 1e-14:
        return {
            **result,
            "reason": "Degenerate scene bootstrap; no safety conclusion",
            "degenerate_value": float(values[0]),
        }
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {**result, "low": float(low), "high": float(high), "status": "ESTIMATED", "reason": None}


def _array_metrics(probabilities):
    valid = probabilities[..., :3].sum(axis=-1)
    ratio = np.full_like(valid, np.nan)
    ratio_s = np.full_like(valid, np.nan)
    np.divide(probabilities[..., 0], valid, out=ratio, where=valid > 0)
    np.divide(probabilities[..., 1], valid, out=ratio_s, where=valid > 0)
    return np.concatenate(
        [probabilities, valid[..., None], ratio[..., None], ratio_s[..., None]], axis=-1
    )


def paired_scene_bootstrap(
    candidate_rows, *, fixed_weights, baseline_key=None, repeats=5000, seed=20260914, alpha=0.05
):
    """Family-stratified scene resampling, shared across interfaces/candidates.

    Each family's original mass is retained. Within a family the resampled
    weighted scene contributions are normalized by their resampled weight mass;
    at the original panel this equals the declared fixed-weight estimand.
    """
    _integer(repeats, "repeats", minimum=2)
    _integer(seed, "seed", minimum=0)
    _alpha(alpha)
    panels, weights, baseline = _candidate_panels(candidate_rows, fixed_weights, baseline_key)
    first = next(iter(panels.values()))["prompts"]
    by_family = defaultdict(set)
    interfaces = set()
    by_scene = defaultdict(set)
    for row in first.values():
        by_family[row["family"]].add(row["base_scene_id"])
        interfaces.add(row["interface"])
        by_scene[row["base_scene_id"]].add(row["interface"])
    if len(interfaces) != 2 or any(found != interfaces for found in by_scene.values()):
        raise ValueError("Scene bootstrap requires both interfaces for every base scene")
    families = {f: sorted(scenes) for f, scenes in sorted(by_family.items())}
    common = {
        "uncertainty": "FAMILY_STRATIFIED_PAIRED_SCENE_BOOTSTRAP",
        "replicates": repeats,
        "seed": seed,
        "alpha": alpha,
        "contract": {
            "candidate_draws_shared": True,
            "interfaces_paired_within_scene": True,
            "family_stratified": True,
            "within_scene_outputs_resampled": False,
            "training_seed_uncertainty_included": False,
        },
        "safety_status": "NOT_CERTIFIED",
    }
    if any(len(scenes) < 2 for scenes in families.values()):
        return {
            **common,
            "status": "UNINFORMATIVE",
            "reason": "Fewer than two independent scenes in a family",
            "candidates": {},
            "deltas": {},
            "scene_draws": {},
        }
    rng = np.random.default_rng(seed)
    draws = {
        f: rng.integers(0, len(scenes), size=(repeats, len(scenes)))
        for f, scenes in families.items()
    }
    scopes = {"overall": set(first)}
    for family in families:
        for interface in sorted(interfaces):
            scopes[f"{family}/{interface}"] = {
                p
                for p, row in first.items()
                if row["family"] == family and row["interface"] == interface
            }
    arrays = {}
    for candidate, panel in panels.items():
        arrays[candidate] = {}
        for scope, members in scopes.items():
            scope_mass = fsum(weights[p] for p in members)
            combined = np.zeros((repeats, 4))
            if scope_mass == 0:
                combined[:] = np.nan
            for family, scenes in families.items():
                family_members = [p for p in members if first[p]["family"] == family]
                family_mass = fsum(weights[p] for p in family_members)
                if not family_mass:
                    continue
                contributions = np.zeros((len(scenes), 4))
                mass = np.zeros(len(scenes))
                for i, scene in enumerate(scenes):
                    for p in family_members:
                        if first[p]["base_scene_id"] == scene:
                            row = panel["prompts"][p]
                            mass[i] += weights[p]
                            contributions[i] += weights[p] * np.asarray(
                                [row["counts"][c] / row["n"] for c in CATEGORIES]
                            )
                sampled_mass = mass[draws[family]].sum(axis=1)
                numerator = contributions[draws[family]].sum(axis=1)
                sampled = np.full_like(numerator, np.nan)
                np.divide(
                    numerator, sampled_mass[:, None], out=sampled, where=sampled_mass[:, None] > 0
                )
                combined += (family_mass / scope_mass) * sampled
            arrays[candidate][scope] = _array_metrics(combined)
    intervals = {
        candidate: {
            scope: {metric: _interval(values[:, i], alpha) for i, metric in enumerate(METRICS)}
            for scope, values in scopes_.items()
        }
        for candidate, scopes_ in arrays.items()
    }
    deltas = {
        candidate: {
            scope: {
                metric: _interval(values[:, i] - arrays[baseline][scope][:, i], alpha)
                for i, metric in enumerate(METRICS)
            }
            for scope, values in scopes_.items()
        }
        for candidate, scopes_ in arrays.items()
    }
    return {
        **common,
        "status": "COMPUTED",
        "reason": None,
        "baseline_key": baseline,
        "direction": "candidate-baseline",
        "unit": "probability",
        "warnings": ["FEW_SCENES_PER_FAMILY"] if any(len(s) < 4 for s in families.values()) else [],
        "candidates": intervals,
        "deltas": deltas,
        "scene_ids": families,
        "scene_draws": {f: d.tolist() for f, d in draws.items()},
    }


def _ratio_bound(numerator, denominator):
    if denominator["low"] <= 0:
        return {
            "low": 0.0,
            "high": 1.0,
            "status": "UNINFORMATIVE",
            "reason": "Valid-mass lower bound is zero",
        }
    return {
        "low": max(0.0, numerator["low"] / denominator["high"]),
        "high": min(1.0, numerator["high"] / denominator["low"]),
        "status": "BOUNDED",
        "reason": None,
    }


def fixed_panel_bounds(
    candidate_rows,
    *,
    fixed_weights,
    baseline_key=None,
    alpha=0.05,
    union_M=None,
    include_groups=False,
):
    """Simultaneous Hoeffding bounds for fixed candidates and prompt weights.

    Five primitive events (X/S/W/I/valid) per canonical candidate/scope enter M.
    Ratios and paired differences propagate this same simultaneous event;
    aliases cannot increase the sample count or reduce the union correction.
    """
    _alpha(alpha)
    panels, weights, baseline = _candidate_panels(candidate_rows, fixed_weights, baseline_key)
    scopes = _scopes(next(iter(panels.values()))) if include_groups else {"overall": set(weights)}
    minimum_M = len(EVENT_METRICS) * len(panels) * len(scopes)
    if union_M is None:
        union_M = minimum_M
    _integer(union_M, "union_M")
    if union_M < minimum_M:
        raise ValueError("union_M must cover every reported primitive event/candidate")
    by_scope = {}
    for scope, members in scopes.items():
        scope_weights = _scope_weights(weights, members)
        result = {}
        for candidate, panel in panels.items():
            scoped = _scope_panel(panel, members)
            point = _summary(scoped, scope_weights)
            squared = fsum(scope_weights[p] ** 2 / row["n"] for p, row in scoped["prompts"].items())
            half = sqrt(0.5 * squared * log(2 * union_M / alpha))
            event_bounds = {
                metric: {
                    "low": max(0.0, point[metric] - half),
                    "high": min(1.0, point[metric] + half),
                    "status": "BOUNDED",
                }
                for metric in EVENT_METRICS
            }
            result[candidate] = {
                **event_bounds,
                "qX": _ratio_bound(event_bounds["pX"], event_bounds["v"]),
                "qS": _ratio_bound(event_bounds["pS"], event_bounds["v"]),
                "half_width": half,
                "sum_squared_sample_weights": squared,
            }
        deltas = {}
        for candidate, bounds in result.items():
            deltas[candidate] = {}
            for metric in METRICS:
                left, right = result[baseline][metric], bounds[metric]
                deltas[candidate][metric] = {
                    "low": right["low"] - left["high"],
                    "high": right["high"] - left["low"],
                    "status": "UNINFORMATIVE"
                    if "UNINFORMATIVE" in (left["status"], right["status"])
                    else "BOUNDED",
                }
        by_scope[scope] = {"candidates": result, "deltas": deltas}
    return {
        "uncertainty": "FIXED_PANEL_GENERATION_SAMPLING_HOEFFDING_UNION_BOUND",
        "alpha": alpha,
        "union_M": union_M,
        "minimum_union_M": minimum_M,
        "primitive_events": list(EVENT_METRICS),
        **by_scope["overall"],
        "by_scope": by_scope,
        "baseline_key": baseline,
        "direction": "candidate-baseline",
        "unit": "probability",
        "assumptions": [
            "Independent j,k generation draws within each candidate",
            "Prompt weights, panel and candidate collection fixed in advance",
        ],
        "excludes": [
            "model selection",
            "training seed variation",
            "scene distribution shift",
            "adaptive repeated selection",
        ],
        "safety_status": "NOT_CERTIFIED",
    }


def summarize_direct_response(
    left_rows,
    right_rows,
    *,
    fixed_weights,
    bootstrap_replicates=5000,
    bootstrap_seed=20260914,
    alpha=0.05,
):
    """Strictly paired right-minus-left direct response from the same bank."""
    candidates = {"left": list(left_rows), "right": list(right_rows)}
    out = summarize_candidate_panel(
        candidates,
        fixed_weights=fixed_weights,
        baseline_key="left",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
    )
    left, right = out["candidates"]["left"], out["candidates"]["right"]
    delta = {
        m: None if left[m] is None or right[m] is None else right[m] - left[m] for m in METRICS
    }
    per_scope = {}
    for scope, points in out["per_scope"].items():
        scoped_delta = {
            m: None
            if points["left"][m] is None or points["right"][m] is None
            else points["right"][m] - points["left"][m]
            for m in METRICS
        }
        per_scope[scope] = {
            "left": points["left"],
            "right": points["right"],
            "delta_probability": scoped_delta,
            "delta_pp": {m: None if v is None else v * 100 for m, v in scoped_delta.items()},
        }
    return {
        "left": left,
        "right": right,
        "direction": "right-left",
        "delta_probability": delta,
        "delta_pp": {m: None if v is None else 100 * v for m, v in delta.items()},
        "fixed_panel": out["fixed_panel"],
        "scene_bootstrap": out["scene_bootstrap"],
        "per_scope": per_scope,
        "safety_status": "NOT_CERTIFIED",
    }


def resolve_aliases(canonical_keys, aliases=None):
    """Resolve a provenance-validated alias graph without creating observations.

    Numerical equality alone must never be used to construct aliases. Runtime
    validates bank, complete inference fingerprint and sampling seed identity
    before passing the graph here.
    """
    canonical, aliases = set(canonical_keys), dict(aliases or {})
    if canonical & set(aliases):
        raise ValueError("A canonical candidate cannot also be an alias")
    resolved = {}
    for alias in aliases:
        seen, target = {alias}, aliases[alias]
        while target not in canonical:
            if target in seen or target not in aliases:
                raise ValueError("Alias graph contains a cycle or unknown target")
            seen.add(target)
            target = aliases[target]
        resolved[alias] = target
    return resolved


def summarize_candidate_panel(
    candidate_rows,
    *,
    fixed_weights,
    baseline_key,
    aliases=None,
    bootstrap_replicates=5000,
    bootstrap_seed=20260914,
    alpha=0.05,
):
    candidate_rows = {key: list(rows) for key, rows in candidate_rows.items()}
    resolved = resolve_aliases(candidate_rows, aliases)
    panels, weights, baseline = _candidate_panels(candidate_rows, fixed_weights, baseline_key)
    points = {key: _summary(panel, weights) for key, panel in panels.items()}
    per_scope = {
        scope: {
            key: _summary(_scope_panel(panel, members), _scope_weights(weights, members))
            for key, panel in panels.items()
        }
        for scope, members in _scopes(next(iter(panels.values()))).items()
    }
    return {
        "candidates": points,
        "per_scope": per_scope,
        "aliases": resolved,
        "baseline_key": baseline,
        "independent_candidate_count": len(panels),
        "unique_rollout_count": sum(p["rollout_count"] for p in points.values()),
        "independent_training_seeds": sorted(
            {
                r["train_seed"]
                for p in panels.values()
                for r in p["prompts"].values()
                if r["train_seed"] is not None
            }
        ),
        "fixed_panel": fixed_panel_bounds(
            candidate_rows,
            fixed_weights=weights,
            baseline_key=baseline,
            alpha=alpha,
            include_groups=True,
        ),
        "scene_bootstrap": paired_scene_bootstrap(
            candidate_rows,
            fixed_weights=weights,
            baseline_key=baseline,
            repeats=bootstrap_replicates,
            seed=bootstrap_seed,
            alpha=alpha,
        ),
        "safety_status": "NOT_CERTIFIED",
    }


def _logsumexp(values):
    if not len(values):
        return None
    maximum = float(np.max(values))
    return maximum + log(fsum(float(exp(v - maximum)) for v in values))


def _exp_diagnostic(log_value):
    if log_value is None:
        return 0.0, "UNOBSERVED"
    if log_value > log(np.finfo(np.float64).max):
        return None, "OVERFLOW"
    # Preserve a finite log estimate even if its linear representation vanishes.
    if log_value < log(float(np.nextafter(0.0, 1.0))):
        return None, "UNDERFLOW"
    return exp(log_value), "FINITE"


def importance_sampling_diagnostics(log_weights, categories):
    """Per-prompt OIS, SNIS and proposal direct estimates without clipping.

    Overflow/underflow in a linear OIS estimate is represented as None with its
    finite log estimate and an explicit status; SNIS and ESS stay in log space.
    Event ESS uses the weights of observed samples for that event only.
    """
    values = np.asarray(log_weights, dtype=np.float64)
    categories = list(categories)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Log weights must be finite nonempty one-dimensional values")
    if len(categories) != values.size or any(c not in CATEGORIES for c in categories):
        raise ValueError("Known categories must match the log-weight length")
    n = len(values)
    maximum = float(values.max())
    shifted = values - maximum
    log_normalizer = _logsumexp(shifted)
    normalized = np.exp(shifted - log_normalizer)
    ess = 1 / float(np.sum(normalized**2))
    log_mean = maximum + log_normalizer - log(n)
    mean_weight, mean_status = _exp_diagnostic(log_mean)
    events, ordinary, self_normalized, direct, logs, statuses = {}, {}, {}, {}, {}, {}
    for category in CATEGORIES:
        mask = np.asarray([c == category for c in categories])
        count = int(mask.sum())
        event_log_sum = _logsumexp(values[mask])
        log_probability = None if event_log_sum is None else event_log_sum - log(n)
        estimate, status = _exp_diagnostic(log_probability)
        ordinary[f"p{category}"] = estimate
        logs[f"p{category}"] = log_probability
        statuses[f"p{category}"] = status
        self_normalized[f"p{category}"] = float(normalized[mask].sum())
        direct[f"p{category}"] = count / n
        event_ess = None
        if count:
            # Renormalize inside the event, even if it has underflowed globally.
            event_shift = values[mask] - float(values[mask].max())
            event_norm = np.exp(event_shift - _logsumexp(event_shift))
            event_ess = 1 / float(np.sum(event_norm**2))
        events[category] = {
            "count": count,
            "support": "OBSERVED" if count else "UNOBSERVED",
            "ESS": event_ess,
            "ESS_fraction_of_observed": event_ess / count if count else None,
        }
    valid_mask = np.asarray([c != "I" for c in categories])
    log_valid = _logsumexp(values[valid_mask])
    log_valid_probability = None if log_valid is None else log_valid - log(n)
    ordinary["v"], statuses["v"] = _exp_diagnostic(log_valid_probability)
    logs["v"] = log_valid_probability
    self_normalized["v"] = float(normalized[valid_mask].sum())
    direct["v"] = int(valid_mask.sum()) / n
    # The ratio cancels the common normalizer. Compute inside valid samples so
    # globally tiny valid mass does not incorrectly make this ratio undefined.
    conditional = {"qX": None, "qS": None}
    if log_valid is not None:
        valid_shift = values[valid_mask] - float(values[valid_mask].max())
        valid_norm = np.exp(valid_shift - _logsumexp(valid_shift))
        for event in ("X", "S"):
            conditional[f"q{event}"] = float(
                valid_norm[np.asarray([c == event for c in categories])[valid_mask]].sum()
            )
    ordinary.update(conditional)
    self_normalized.update(conditional)
    for event in ("X", "S"):
        direct[f"q{event}"] = direct[f"p{event}"] / direct["v"] if direct["v"] else None
    warnings = []
    maxw = float(normalized.max())
    if ess / n < 0.5 or n * maxw > 8:
        warnings.append("OVERLAP_WARNING")
    if not events["X"]["count"]:
        warnings.append("X_EVENT_UNOBSERVED")
    if any(logs[key] is not None and logs[key] > 0 for key in logs):
        warnings.append("OIS_OUTSIDE_PROBABILITY_RANGE")
    if any(status in ("OVERFLOW", "UNDERFLOW") for status in statuses.values()):
        warnings.append("OIS_LINEAR_VALUE_UNREPRESENTABLE")
    return {
        "n": n,
        "mean_weight": mean_weight,
        "log_mean_weight": log_mean,
        "mean_weight_status": mean_status,
        "ESS": ess,
        "ESS_fraction": ess / n,
        "max_normalized_weight": maxw,
        "n_times_max_normalized_weight": n * maxw,
        "events": events,
        "category_counts": {c: events[c]["count"] for c in CATEGORIES},
        "OIS": {
            "estimator": "ORDINARY_IMPORTANCE_SAMPLING",
            **ordinary,
            "log_probabilities": logs,
            "linear_status": statuses,
        },
        "SNIS": {"estimator": "SELF_NORMALIZED_IMPORTANCE_SAMPLING", **self_normalized},
        "DIRECT": {"estimator": "DIRECT_SAMPLING_UNDER_PROPOSAL", **direct},
        "warnings": warnings,
        "warning_thresholds": {"ESS_fraction_below": 0.5, "n_times_maxw_above": 8},
        "role": "ESTIMATOR_DIAGNOSTIC_ONLY_NOT_INDEPENDENT_CANDIDATE_VALIDATION",
        "safety_status": "NOT_CERTIFIED",
        "weights_clipped": False,
    }


def _direction(value):
    return (
        "undefined"
        if value is None
        else "positive"
        if value > 0
        else "negative"
        if value < 0
        else "zero"
    )


def _descriptive(values):
    defined = [v for v in values if v is not None]
    directions = [_direction(v) for v in values]
    return {
        "count": len(values),
        "defined_count": len(defined),
        "mean": mean(defined) if len(defined) == len(values) and defined else None,
        "std": stdev(defined) if len(defined) == len(values) and len(defined) > 1 else None,
        "directions": directions,
        "all_same_direction": bool(directions)
        and "undefined" not in directions
        and len(set(directions)) == 1,
    }


def summarize_seed_results(records, *, baseline_arm):
    """Descriptive seed/arm results first, paired deltas second; no p-values."""
    by_seed, aliases = defaultdict(dict), defaultdict(dict)
    for record in records:
        seed, arm = record.get("seed"), record.get("arm")
        _integer(seed, "seed", minimum=0)
        if not isinstance(arm, str) or not arm:
            raise ValueError("Nonempty arm required")
        if arm in by_seed[seed] or arm in aliases[seed]:
            raise ValueError("Duplicate seed/arm record")
        if record.get("alias_of") is not None:
            aliases[seed][arm] = record["alias_of"]
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Nonempty metrics required")
        if any(
            v is not None
            and (isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v))
            for v in metrics.values()
        ):
            raise ValueError("Seed metrics must be finite or undefined")
        by_seed[seed][arm] = dict(metrics)
    if not by_seed:
        raise ValueError("Nonempty training seed results required")
    arms = set(next(iter(by_seed.values())))
    if baseline_arm not in arms or any(set(results) != arms for results in by_seed.values()):
        raise ValueError("All training seeds must have paired baseline/arm coverage")
    first = next(iter(next(iter(by_seed.values())).values()))
    metric_keys = set(first)
    if any(
        set(metrics) != metric_keys for results in by_seed.values() for metrics in results.values()
    ):
        raise ValueError("Seed and arm metric coverage differs")
    per_seed = []
    for seed, results in sorted(by_seed.items()):
        resolved = resolve_aliases(results, aliases[seed])
        entries = {}
        for arm, metrics in sorted(results.items()):
            delta = {
                key: None
                if value is None or results[baseline_arm][key] is None
                else value - results[baseline_arm][key]
                for key, value in metrics.items()
            }
            entries[arm] = {
                "metrics": metrics,
                "delta_probability": delta,
                "delta_pp": {
                    key: None if value is None else value * 100 for key, value in delta.items()
                },
            }
        role = (
            "EXPLORATORY_ALREADY_OBSERVED"
            if seed == 17
            else "PROSPECTIVE"
            if seed in (29, 41)
            else "UNREGISTERED"
        )
        per_seed.append({"seed": seed, "seed_role": role, "arms": entries, "aliases": resolved})

    def aggregate(entries):
        return {
            arm: {
                key: {
                    "estimate": _descriptive([e["arms"][arm]["metrics"][key] for e in entries]),
                    "paired_delta": _descriptive(
                        [e["arms"][arm]["delta_probability"][key] for e in entries]
                    ),
                }
                for key in sorted(metric_keys)
            }
            for arm in sorted(arms)
        }

    return {
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
        "prospective_aggregate": aggregate(
            [e for e in per_seed if e["seed_role"] == "PROSPECTIVE"]
        ),
        "training_seed_count": len(per_seed),
        "baseline_arm": baseline_arm,
        "direction": "arm-baseline",
        "unit": "probability",
        "uncertainty": "TRAINING_SEED_DESCRIPTIVE_SUMMARY",
        "warnings": ["FEW_TRAINING_SEEDS", "EXPLORATORY_MULTIPLE_COMPARISONS_NOT_JOINT_SAFETY"],
        "safety_status": "NOT_CERTIFIED",
    }
