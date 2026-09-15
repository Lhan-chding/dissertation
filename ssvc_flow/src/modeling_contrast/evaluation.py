"""Pure-array contrast scores with separate targets and seed-cluster inference.

Inputs are differences in probability units, ordered X/S/W/I. Active cases must
come from fit excitation metadata, never from observed test response size.
"""

from __future__ import annotations

import numpy as np

DELTA_GRID = (0.0001, 0.00025, 0.0005, 0.001)


def _arrays(pred_events, truth_events, groups):
    pred, truth = np.asarray(pred_events, dtype=float), np.asarray(truth_events, dtype=float)
    groups = np.asarray(groups)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[1:] != (72, 4):
        raise ValueError("matching [case,72,4] event contrast arrays required")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("finite event contrast arrays required")
    if groups.shape != (72,):
        raise ValueError("72 prompt group identities required")
    if groups.dtype.kind in "fc" and not np.isfinite(groups).all():
        raise ValueError("finite group identities required")
    labels = np.unique(groups)
    if len(labels) != 6:
        raise ValueError("six fixed prompt groups required")
    return pred, truth, groups, labels


def _ratio_norm(error, truth):
    """Scale both vectors together: no artificial floor on true signal energy."""
    scale = float(np.max(np.abs(truth), initial=0))
    if scale == 0:
        return None
    # Long double avoids underflow of tiny but nonzero probability differences.
    t = np.asarray(truth, dtype=np.longdouble) / scale
    e = np.asarray(error, dtype=np.longdouble) / scale
    value = np.sqrt(np.sum(e * e) / np.sum(t * t))
    return float(value) if value <= np.finfo(float).max else None


def _score(pred, truth):
    if not truth.size:
        return {
            "status": "NO_CASES",
            "pooled_nrmse": None,
            "mae": None,
            "rmse": None,
            "error_q95": None,
            "error_ss": 0.0,
            "truth_ss": 0.0,
            "element_count": 0,
        }
    error = pred - truth
    nrmse = _ratio_norm(error, truth)
    signal = bool(np.any(truth != 0))
    return {
        "status": "OK" if nrmse is not None else "NUMERICAL_OVERFLOW" if signal else "NO_SIGNAL",
        "pooled_nrmse": nrmse,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.asarray(error, dtype=np.longdouble) ** 2))),
        "error_q95": float(np.quantile(np.abs(error), 0.95)),
        "error_ss": float(np.sum(np.asarray(error, dtype=np.longdouble) ** 2)),
        "truth_ss": float(np.sum(np.asarray(truth, dtype=np.longdouble) ** 2)),
        "element_count": int(truth.size),
    }


def group_contrasts(events, groups):
    """Group means of pX and validity; validity contrast equals minus I."""
    values, groups = np.asarray(events, dtype=float), np.asarray(groups)
    labels = np.unique(groups)
    means = np.stack([values[:, groups == group].mean(1) for group in labels], axis=1)
    return {"pX": means[..., 0], "v": -means[..., 3]}


def _panel(pred, truth, groups, labels):
    result = {"case_count": len(truth), "events": _score(pred, truth)}
    gp, gt = group_contrasts(pred, groups), group_contrasts(truth, groups)
    for target in ("pX", "v"):
        score = _score(gp[target], gt[target])
        rows = [
            {"group": group.item(), **_score(gp[target][:, i], gt[target][:, i])}
            for i, group in enumerate(labels)
        ]
        eligible = [row for row in rows if row["mae"] is not None]
        # Worst group is defined by absolute-error q95, not by unstable zero energy.
        score["worst_group"] = (
            max(eligible, key=lambda row: (row["error_q95"], row["mae"])) if eligible else None
        )
        score["worst_group_criterion"] = "absolute_error_q95_then_mae"
        score["by_group"] = rows
        result[target] = score
    return result


def contrast_metrics(pred_events, truth_events, groups, *, active=None):
    """Pooled four-event and six-group pX/v scores, before any projection.

    The NRMSE denominator is the unmodified true contrast norm. Empty or exactly
    zero signal is explicit; tiny nonzero signal remains eligible for scoring.
    """
    pred, truth, groups, labels = _arrays(pred_events, truth_events, groups)
    result = {
        "probability_units": True,
        "nrmse_energy_floor": "numerical_zero_only",
        "all": _panel(pred, truth, groups, labels),
    }
    if active is None:
        result["active_only"] = {"status": "ACTIVE_MASK_NOT_SUPPLIED", "case_count": 0}
    else:
        mask = np.asarray(active)
        if mask.shape != (len(pred),) or mask.dtype.kind != "b":
            raise ValueError("active must be a boolean mask from fit excitation, one per case")
        result["active_only"] = _panel(pred[mask], truth[mask], groups, labels)
        result["active_only"]["status"] = "OK" if mask.any() else "NO_ACTIVE_CASES"
    return result


def _identity(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_identity(x) for x in value)
    return value.item() if isinstance(value, np.generic) else value


def direction_grid(
    pred_events,
    truth_events,
    groups,
    training_seeds,
    anchor_bank_units,
    *,
    delta_grid=DELTA_GRID,
    min_units=20,
    min_seeds=5,
):
    """Describe direction only after distinct seed and anchor-bank support gates.

    A unit key must identify arm/anchor/bank where arm is part of the trajectory
    identity. The training seed is prepended automatically. Noise replicas and
    six group outcomes never become new independent units.
    """
    pred, truth, groups, _ = _arrays(pred_events, truth_events, groups)
    seeds = list(training_seeds)
    units = list(anchor_bank_units)
    if len(seeds) != len(pred) or len(units) != len(pred):
        raise ValueError("one seed and anchor-bank unit identity per case required")
    if min_units < 1 or min_seeds < 1:
        raise ValueError("positive support thresholds required")
    gp, gt = group_contrasts(pred, groups), group_contrasts(truth, groups)
    result = {"scope": "descriptive_seed_and_anchor_bank_supported_direction"}
    for target in ("pX", "v"):
        rows = []
        for delta in delta_grid:
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError("positive finite resolution required")
            eligible = np.abs(gt[target]) >= delta
            grouped, eligible_seeds = {}, set()
            for i in range(len(pred)):
                if not eligible[i].any():
                    continue
                training_seed = _identity(seeds[i])
                key = (training_seed, _identity(units[i]))
                eligible_seeds.add(training_seed)
                values = np.sign(gp[target][i, eligible[i]]) == np.sign(gt[target][i, eligible[i]])
                grouped.setdefault(key, []).extend(values.tolist())
            supported = len(grouped) >= min_units and len(eligible_seeds) >= min_seeds
            rows.append(
                {
                    "delta": float(delta),
                    "status": "DESCRIPTIVE_SUPPORTED"
                    if supported
                    else "INSUFFICIENT_EFFECTIVE_CASES",
                    "effective_anchor_bank_units": len(grouped),
                    "training_seed_count": len(eligible_seeds),
                    "eligible_group_case_rows": int(eligible.sum()),
                    "accuracy": float(np.mean([np.mean(values) for values in grouped.values()]))
                    if supported
                    else None,
                    "accuracy_weighting": "equal_anchor_bank_units_then_group_replica_average",
                    "min_anchor_bank_units": min_units,
                    "min_training_seeds": min_seeds,
                }
            )
        result[target] = rows
    return result


def paired_seed_bootstrap(
    method_values,
    baseline_values,
    training_seeds,
    *,
    replicates=5000,
    seed=2026091503,
    statistic="mean",
):
    """Paired resampling of entire training-seed clusters.

    ``mean`` takes one scalar per case and averages seed means. ``nrmse`` takes
    [error_ss, truth_ss] per case and recomputes pooled NRMSE in every resample.
    Rows for both arms, all anchors/banks and replicas travel with their seed.
    """
    method, baseline = (
        np.asarray(method_values, dtype=float),
        np.asarray(baseline_values, dtype=float),
    )
    seeds = np.asarray(training_seeds)
    if method.shape != baseline.shape or seeds.ndim != 1 or len(seeds) != len(method):
        raise ValueError("paired values and one training seed per row required")
    if not np.isfinite(method).all() or not np.isfinite(baseline).all():
        raise ValueError("finite paired scores required")
    if seeds.dtype.kind in "fc" and not np.isfinite(seeds).all():
        raise ValueError("finite training seed identities required")
    if replicates < 1 or int(replicates) != replicates:
        raise ValueError("positive integer bootstrap replicates required")
    if statistic not in ("mean", "nrmse"):
        raise ValueError("statistic must be mean or nrmse")
    if statistic == "mean" and method.ndim != 1:
        raise ValueError("mean bootstrap takes scalar row values")
    if statistic == "nrmse" and (
        method.ndim != 2 or method.shape[1] != 2 or np.any(method < 0) or np.any(baseline < 0)
    ):
        raise ValueError("nrmse bootstrap takes nonnegative [error_ss,truth_ss] rows")
    if statistic == "nrmse" and not np.allclose(method[:, 1], baseline[:, 1], rtol=1e-12, atol=0):
        raise ValueError("paired NRMSE rows must share true contrast energies")
    keys = np.unique(seeds)
    if len(keys) == 0:
        return {"status": "NO_PAIRED_CLUSTERS", "cluster_count": 0}
    aggregate = np.mean if statistic == "mean" else np.sum
    cm = np.array([aggregate(method[seeds == key], axis=0) for key in keys])
    cb = np.array([aggregate(baseline[seeds == key], axis=0) for key in keys])
    indices = np.random.default_rng(seed).integers(0, len(keys), size=(int(replicates), len(keys)))
    if statistic == "mean":
        per_seed = cm - cb
        draws = per_seed[indices].mean(1)
        point = float(per_seed.mean())
    else:
        if np.any(cm[:, 1] <= 0) or np.any(cb[:, 1] <= 0):
            return {"status": "NO_SIGNAL_IN_SEED_CLUSTER", "cluster_count": len(keys)}
        per_seed = np.sqrt(cm[:, 0] / cm[:, 1]) - np.sqrt(cb[:, 0] / cb[:, 1])
        sm, sb = cm[indices].sum(1), cb[indices].sum(1)
        draws = np.sqrt(sm[:, 0] / sm[:, 1]) - np.sqrt(sb[:, 0] / sb[:, 1])
        sm, sb = cm.sum(0), cb.sum(0)
        point = float(np.sqrt(sm[0] / sm[1]) - np.sqrt(sb[0] / sb[1]))
    return {
        "status": "POSTHOC_DEVELOPMENT_SEED_CLUSTER_BOOTSTRAP"
        if len(keys) > 1
        else "INSUFFICIENT_SEED_CLUSTERS",
        "cluster_count": len(keys),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "statistic": statistic,
        "paired_mean": point,
        "ci025": float(np.quantile(draws, 0.025)),
        "ci975": float(np.quantile(draws, 0.975)),
        "interval_type": "paired_training_seed_cluster_percentile_bootstrap",
        "distribution_free_prediction_coverage_claim": False,
        "per_seed": [
            {
                "seed": _identity(key),
                "paired_difference": float(value),
                "row_count": int(np.sum(seeds == key)),
            }
            for key, value in zip(keys, per_seed, strict=True)
        ],
    }


def classify_interval(lower, upper, delta=0.00025):
    """Classify an explicitly supplied interval; unknown never implies safety."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("positive finite delta required")
    if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper):
        return "UNKNOWN"
    if lower > upper:
        raise ValueError("interval lower bound exceeds upper bound")
    if lower > delta:
        return "MEANINGFUL_POSITIVE"
    if upper < -delta:
        return "MEANINGFUL_NEGATIVE"
    if lower >= -delta and upper <= delta:
        return "WITHIN_TOLERANCE"
    return "UNKNOWN"


def calibration_resolution(seed_count, nominal_coverage=0.95):
    """Report the finite seed order-statistic limit, without a coverage promise."""
    if int(seed_count) != seed_count or seed_count < 0 or not 0 < nominal_coverage < 1:
        raise ValueError("nonnegative integer seed count and coverage in (0,1) required")
    supported = seed_count / (seed_count + 1)
    return {
        "calibration_seed_count": int(seed_count),
        "nominal_coverage": float(nominal_coverage),
        "maximum_supported_nominal_coverage": supported,
        "status": "INSUFFICIENT_CALIBRATION_FOR_DISTRIBUTION_FREE_95"
        if supported < nominal_coverage
        else "ORDER_STATISTIC_RESOLUTION_ONLY",
        "distribution_free_95_claim": False,
        "simultaneous_safety_certification": False,
        "interval_type": "independent_empirical_calibration_only",
    }


def resolution_grid(metrics, *, intervals=None, delta_grid=DELTA_GRID, nrmse_max=0.75):
    """Keep prediction error, engineering resolution, and interval evidence apart.

    ``metrics`` is a single all/active panel from ``contrast_metrics``. Optional
    intervals map each target to [lower, upper]; their construction/type must be
    reported by the caller. No interval or safety claim is inferred from q95.
    """
    intervals = intervals or {}
    rows = []
    for delta in delta_grid:
        for target in ("pX", "v"):
            metric = metrics[target]
            nrmse, q95 = metric["pooled_nrmse"], metric["error_q95"]
            bounds = intervals.get(target, (None, None))
            classification = classify_interval(bounds[0], bounds[1], delta)
            rows.append(
                {
                    "target": target,
                    "delta": float(delta),
                    "is_primary_delta": delta == 0.00025,
                    "prediction_status": "PREDICTION_PASS"
                    if nrmse is not None and nrmse <= nrmse_max
                    else "PREDICTION_FAIL",
                    "empirical_error_resolution_status": "RESOLUTION_PASS"
                    if q95 is not None and q95 <= delta / 2
                    else "RESOLUTION_FAIL",
                    "interval_classification": classification,
                    "interval_resolution_status": "RESOLUTION_FAIL"
                    if classification == "UNKNOWN"
                    else "INTERVAL_RESOLVED",
                    "safety_certified": False,
                }
            )
    return rows
