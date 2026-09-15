"""Seed-cluster inference and explicitly empirical measurement summaries."""

from __future__ import annotations

import math

import numpy as np


def error_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if prediction.shape != truth.shape or truth.size == 0:
        raise ValueError("nonempty aligned predictions and truth required")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite estimates must have explicit unresolved status")
    residual = prediction - truth
    absolute = np.abs(residual)
    energy, loss = float(np.sum(truth**2)), float(np.sum(residual**2))
    return {
        "count": int(truth.size),
        "bias": float(residual.mean()),
        "mse": float(np.mean(residual**2)),
        "mae": float(absolute.mean()),
        "error_ss": loss,
        "signal_ss": energy,
        "nrmse": math.sqrt(loss / energy) if energy > 0 else None,
        "nrmse_status": "DEFINED" if energy > 0 else "UNDEFINED_ZERO_SIGNAL",
        **{
            f"q{int(q * 100)}_absolute_residual": float(np.quantile(absolute, q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
        "maximum_absolute_residual": float(absolute.max()),
        "quantiles_are_confidence_intervals": False,
    }


def seed_bootstrap(seeds, paired_values, *, reps=5000, seed=2026091503):
    seeds, values = np.asarray(seeds), np.asarray(paired_values, float)
    if seeds.shape != values.shape or seeds.ndim != 1 or seeds.size == 0:
        raise ValueError("aligned one-dimensional seed/value vectors required")
    if not np.isfinite(values).all() or type(reps) is not int or reps < 1:
        raise ValueError("finite values and positive repetition count required")
    unique = np.unique(seeds)
    means = np.array([values[seeds == item].mean() for item in unique])
    rng = np.random.default_rng(seed)
    replicates = means[rng.integers(0, len(means), size=(reps, len(means)))].mean(1)
    return {
        "estimate": float(means.mean()),
        "interval": np.quantile(replicates, [0.025, 0.975]).tolist(),
        "independent_seeds": len(unique),
        "unit": "training_seed",
        "reps": reps,
        "bootstrap_seed": seed,
        "status": "EMPIRICAL" if len(unique) > 1 else "ONE_SEED_NO_INDEPENDENT_INFERENCE",
    }


def conformal_radius(seed_scores, *, nominal=0.95):
    scores = np.asarray(seed_scores, float)
    if (
        scores.ndim != 1
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or not 0 < nominal < 1
    ):
        raise ValueError("nonnegative seed scores and nominal in (0,1) required")
    n = len(scores)
    index = math.ceil((n + 1) * nominal)
    return {
        "status": "EMPIRICAL_EXCHANGEABILITY_REQUIRED"
        if index <= n
        else "INSUFFICIENT_CALIBRATION_SEEDS",
        "radius": float(np.sort(scores)[index - 1]) if index <= n else None,
        "n_calibration_seeds": n,
        "order_statistic": index,
        "nominal": nominal,
        "coverage_object": "one future exchangeable training-seed score",
        "distribution_shift_covered": False,
    }


def holm(pvalues):
    values = np.asarray(pvalues, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def risk_coverage(loss, accepted):
    loss, accepted = np.asarray(loss, float), np.asarray(accepted)
    if accepted.dtype.kind != "b":
        raise ValueError("acceptance must be an explicit boolean mask, not status strings")
    if loss.shape != accepted.shape or loss.size == 0 or not np.isfinite(loss).all():
        raise ValueError("all cases including UNKNOWN need finite baseline losses")
    return {
        "all_cases": int(loss.size),
        "accepted_cases": int(accepted.sum()),
        "unknown_cases": int((~accepted).sum()),
        "coverage": float(accepted.mean()),
        "all_case_loss": float(loss.mean()),
        "accepted_risk": float(loss[accepted].mean()) if accepted.any() else None,
    }


def reference_corrected_mse(prediction, reference, reference_variance):
    pred, ref, var = map(
        lambda x: np.asarray(x, float), (prediction, reference, reference_variance)
    )
    if pred.shape != ref.shape or var.shape != ref.shape or np.any(var < 0):
        raise ValueError(
            "aligned predictions, independent references and nonnegative variances required"
        )
    if not all(np.isfinite(x).all() for x in (pred, ref, var)):
        raise ValueError("nonfinite reference")
    raw, noise = float(np.mean((pred - ref) ** 2)), float(var.mean())
    return {
        "raw_mse": raw,
        "reference_variance_mean": noise,
        "aggregate_corrected_mse": raw - noise,
        "clipped": False,
        "requires_independent_reference": True,
    }
