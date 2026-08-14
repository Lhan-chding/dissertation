"""Deterministic seed-stratified paired bootstrap for supported OOD metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from compbias.eval.statistics import holm_adjust


def _binary(value: object, *, name: str) -> float:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return float(bool(value))


def _paired_by_seed(
    iid_records: Sequence[Mapping[str, object]],
    ood_records: Sequence[Mapping[str, object]],
    *,
    seeds: tuple[int, ...],
    compensatory_only: bool,
) -> dict[int, np.ndarray] | None:
    iid = {(str(row["sample_id"]), int(row["seed"])): row for row in iid_records}
    ood = {
        (str(row.get("paired_sample_id", row["sample_id"])), int(row["seed"])): row
        for row in ood_records
    }
    if iid.keys() != ood.keys():
        raise ValueError("statistical inference requires exact IID/OOD sample-seed pairs")
    grouped: dict[int, list[float]] = {seed: [] for seed in seeds}
    for key in sorted(iid):
        iid_row = iid[key]
        ood_row = ood[key]
        from compbias.eval.ood import _compensatory

        iid_compensatory = _compensatory(iid_row)
        ood_compensatory = _compensatory(ood_row)
        if iid_compensatory != ood_compensatory:
            raise ValueError("compensatory eligibility must remain paired across the shift")
        if compensatory_only and not iid_compensatory:
            continue
        grouped[key[1]].append(
            _binary(iid_row["answer_correct"], name="IID answer_correct")
            - _binary(ood_row["answer_correct"], name="OOD answer_correct")
        )
    if compensatory_only and any(not values for values in grouped.values()):
        return None
    return {seed: np.asarray(values, dtype=np.float64) for seed, values in grouped.items()}


def _bootstrap_summary(
    groups: Sequence[np.ndarray],
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if not groups or any(group.ndim != 1 or group.size == 0 for group in groups):
        raise ValueError("paired bootstrap groups must be non-empty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    draws = np.zeros(n_resamples, dtype=np.float64)
    null_draws = np.zeros(n_resamples, dtype=np.float64)
    for group in groups:
        indices = rng.integers(0, group.size, size=(n_resamples, group.size))
        draws += group[indices].mean(axis=1)
        centered = group - group.mean()
        null_draws += centered[indices].mean(axis=1)
    draws /= len(groups)
    null_draws /= len(groups)
    estimate = float(np.mean([group.mean() for group in groups]))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    exceedances = int(np.count_nonzero(np.abs(null_draws) >= abs(estimate)))
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_pairs": int(sum(group.size for group in groups)),
        "p_value": (exceedances + 1.0) / (n_resamples + 1.0),
    }


def _metric_inference(
    grouped: Mapping[int, np.ndarray],
    *,
    seeds: tuple[int, ...],
    confidence: float,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    per_seed = {
        str(seed): _bootstrap_summary(
            (grouped[seed],),
            confidence=confidence,
            n_resamples=n_resamples,
            seed=bootstrap_seed + seed,
        )
        for seed in seeds
    }
    aggregate = _bootstrap_summary(
        tuple(grouped[seed] for seed in seeds),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
    )
    return {
        "contrast": "iid_minus_ood",
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


def _apply_holm(metrics: dict[str, dict[str, object]], seeds: tuple[int, ...]) -> None:
    aggregate_adjusted = holm_adjust(
        {
            name: float(metric["aggregate"]["p_value"])  # type: ignore[index]
            for name, metric in metrics.items()
        }
    )
    for name, adjusted in aggregate_adjusted.items():
        metrics[name]["aggregate"]["holm_adjusted_p_value"] = adjusted  # type: ignore[index]
    for seed in seeds:
        seed_key = str(seed)
        adjusted = holm_adjust(
            {
                name: float(metric["per_seed"][seed_key]["p_value"])  # type: ignore[index]
                for name, metric in metrics.items()
            }
        )
        for name, value in adjusted.items():
            metrics[name]["per_seed"][seed_key]["holm_adjusted_p_value"] = value  # type: ignore[index]


def paired_ood_inference(
    iid_records: Sequence[Mapping[str, object]],
    ood_records: Sequence[Mapping[str, object]],
    *,
    seeds: tuple[int, ...],
    metric_names: Sequence[str],
    confidence: float,
    n_resamples: int,
    bootstrap_seed: int = 20260814,
) -> dict[str, object]:
    """Compute the registered paired contrasts, CIs, p-values, and Holm correction."""

    requested = tuple(metric_names)
    supported = {
        "error_mechanism_generalization_gap",
        "compensation_generalization_gap",
    }
    if any(name not in supported for name in requested):
        raise ValueError("paired OOD inference received an unsupported metric")
    all_pairs = _paired_by_seed(iid_records, ood_records, seeds=seeds, compensatory_only=False)
    assert all_pairs is not None
    compensation_pairs = (
        _paired_by_seed(iid_records, ood_records, seeds=seeds, compensatory_only=True)
        if "compensation_generalization_gap" in requested
        else None
    )
    metrics: dict[str, dict[str, object]] = {}
    cache: dict[tuple[bytes, ...], dict[str, object]] = {}
    for name in requested:
        grouped = compensation_pairs if name == "compensation_generalization_gap" else all_pairs
        if grouped is None:
            continue
        fingerprint = tuple(grouped[seed].tobytes() for seed in seeds)
        if fingerprint not in cache:
            cache[fingerprint] = _metric_inference(
                grouped,
                seeds=seeds,
                confidence=confidence,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )
        metrics[name] = {
            "contrast": cache[fingerprint]["contrast"],
            "aggregate": dict(cache[fingerprint]["aggregate"]),  # type: ignore[arg-type]
            "per_seed": {
                seed: dict(summary)
                for seed, summary in cache[fingerprint]["per_seed"].items()  # type: ignore[union-attr]
            },
        }
    _apply_holm(metrics, seeds)
    return {
        "protocol": {
            "bootstrap_resamples": n_resamples,
            "bootstrap_seed": bootstrap_seed,
            "confidence": confidence,
            "multiple_comparisons": "holm",
            "paired_bootstrap": True,
            "resampling_unit": "sample_within_seed",
        },
        "metrics": metrics,
    }
