"""CPU aggregations derived from closed primitive checkpoint-evaluation records."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def _partitioned(iid, ood, function) -> dict[str, object]:
    return {"iid": function(iid), "ood": function(ood)}


def _accuracy(rows: Sequence[Mapping[str, object]]) -> float:
    return float(np.mean([bool(row["answer_correct"]) for row in rows]))


def _numeric_error(rows: Sequence[Mapping[str, object]], *, squared: bool) -> float:
    differences = [
        float(row["numeric_prediction"]) - float(row["numeric_target"])
        for row in rows
        if row["numeric_target"] is not None
    ]
    if not differences:
        raise ValueError("numeric metrics require registered numeric tasks")
    values = np.asarray(differences, dtype=np.float64)
    return float(np.mean(values * values if squared else np.abs(values)))


def _frequency(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    counts = Counter(str(row[field]) for row in rows)
    return {name: count / len(rows) for name, count in sorted(counts.items())}


def _profile_compensabilities(row: Mapping[str, object]) -> tuple[np.ndarray, ...]:
    profile = row["prompt_error_profile"]
    assert isinstance(profile, list)
    severity = np.asarray([float(entry["severity"]) for entry in profile], dtype=np.float64)
    weights = np.asarray([float(entry["base_probability"]) for entry in profile], dtype=np.float64)
    compensability = np.asarray(
        [float(np.mean(entry["rollout_rewards"])) for entry in profile], dtype=np.float64
    )
    return severity, weights, compensability


def _per_prompt_compensability(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    denominators: dict[str, int] = {}
    for row in rows:
        _, weights, values = _profile_compensabilities(row)
        sample_id = str(row["sample_id"])
        grouped[sample_id].append(float(weights @ values))
        profile = row["prompt_error_profile"]
        assert isinstance(profile, list)
        denominators[sample_id] = sum(len(entry["rollout_rewards"]) for entry in profile)
    return {
        sample_id: {
            "estimate": float(np.mean(values)),
            "n_intervention_rollouts_per_seed": denominators[sample_id],
            "n_seeds": len(values),
        }
        for sample_id, values in sorted(grouped.items())
    }


def _per_prompt_covariance(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    n_errors: dict[str, int] = {}
    for row in rows:
        severity, weights, compensability = _profile_compensabilities(row)
        severity_mean = float(weights @ severity)
        compensability_mean = float(weights @ compensability)
        value = float(
            weights @ ((severity - severity_mean) * (compensability - compensability_mean))
        )
        sample_id = str(row["sample_id"])
        grouped[sample_id].append(value)
        n_errors[sample_id] = severity.size
    per_prompt = {
        sample_id: {
            "covariance": float(np.mean(values)),
            "n_errors": n_errors[sample_id],
            "n_seeds": len(values),
        }
        for sample_id, values in sorted(grouped.items())
    }
    return {
        "per_prompt": per_prompt,
        "mean_per_prompt_covariance": float(
            np.mean([entry["covariance"] for entry in per_prompt.values()])
        ),
        "prompt_count": len(per_prompt),
    }


def _relative_gain(rows: Sequence[Mapping[str, object]]) -> float:
    return float(
        np.mean(
            [
                float(row["scaling_probe"]["multiplier_derivative"])
                / float(row["scaling_probe"]["multiplier"])
                for row in rows
            ]
        )
    )


def _pairwise_residual(rows: Sequence[Mapping[str, object]]) -> float:
    residuals: list[float] = []
    for row in rows:
        probe = row["selection_probe"]
        reference = np.asarray(probe["reference_probabilities"], dtype=np.float64)
        selected = np.asarray(probe["selected_probabilities"], dtype=np.float64)
        rewards = np.asarray(probe["rewards"], dtype=np.float64)
        beta = float(probe["beta"])
        for left in range(reference.size):
            for right in range(left + 1, reference.size):
                observed = math.log(selected[left] / selected[right])
                predicted = (
                    math.log(reference[left] / reference[right])
                    + (rewards[left] - rewards[right]) / beta
                )
                residuals.append(observed - predicted)
    return max((abs(value) for value in residuals), default=0.0)


def full_checkpoint_metrics(
    iid_records: Sequence[Mapping[str, object]],
    ood_records: Sequence[Mapping[str, object]],
    *,
    error_mechanism_generalization_gap: float,
) -> dict[str, object]:
    """Compute metrics supported by independently checkable primitive records.

    Perception/reasoning decomposition is intentionally excluded until the producer
    records raw text, parser status, parsed perceived state, and reasoning actions.
    """

    result: dict[str, object] = {
        "exact_answer_accuracy": _partitioned(iid_records, ood_records, _accuracy),
        "numeric_mae": _partitioned(
            iid_records, ood_records, lambda rows: _numeric_error(rows, squared=False)
        ),
        "numeric_mse": _partitioned(
            iid_records, ood_records, lambda rows: _numeric_error(rows, squared=True)
        ),
        "error_mechanism_generalization_gap": error_mechanism_generalization_gap,
    }
    return result


__all__ = ["full_checkpoint_metrics"]
