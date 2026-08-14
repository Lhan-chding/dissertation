"""Preregistered, paired IID-to-OOD evaluation summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

from compbias.io.manifests import canonical_json

_OUTCOME_FIELDS = frozenset(
    {
        "answer_correct",
        "counterfactual_consistent",
        "error_family",
        "severity",
        "perceived_scene",
        "predicted_answer",
        "numeric_prediction",
        "prompt_error_profile",
        "scaling_probe",
        "selection_probe",
    }
)


@dataclass(frozen=True, slots=True)
class OODMetrics:
    iid_accuracy: float
    ood_accuracy: float
    error_mechanism_generalization_gap: float
    iid_counterfactual_consistency: float
    ood_counterfactual_consistency: float
    counterfactual_consistency_gap: float
    iid_compensation_accuracy: float | None
    ood_compensation_accuracy: float | None
    compensation_generalization_gap: float | None
    iid_compensatory_count: int
    ood_compensatory_count: int
    shifted_factors: tuple[str, ...]
    n_pairs: int


def _mapping(record: object) -> Mapping[str, object]:
    if isinstance(record, Mapping):
        return record
    if is_dataclass(record) and not isinstance(record, type):
        return {field.name: getattr(record, field.name) for field in fields(record)}
    raise TypeError("each OOD record must be a mapping or dataclass")


def _boolean(row: Mapping[str, object], field: str) -> bool:
    if field not in row:
        raise ValueError(f"OOD record is missing required field {field!r}")
    value = row[field]
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be boolean")
    return bool(value)


def _compensatory(row: Mapping[str, object]) -> bool:
    profile = row.get("prompt_error_profile")
    mechanism = row.get("error_mechanism")
    if not isinstance(profile, list) or not isinstance(mechanism, str):
        raise ValueError("OOD record lacks primitive compensability profile")
    entry = next(
        (
            candidate
            for candidate in profile
            if isinstance(candidate, Mapping) and candidate.get("error_id") == mechanism
        ),
        None,
    )
    if entry is None:
        raise ValueError("OOD record mechanism is absent from compensability profile")
    rewards = entry.get("rollout_rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError("OOD record compensability profile lacks rollout rewards")
    return any(value == 1 for value in rewards)


def _indexed(records: Iterable[object], label: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for record in records:
        row = _mapping(record)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} record sample_id must be a non-empty string")
        if sample_id in result:
            raise ValueError(f"{label} records contain duplicate sample_id {sample_id!r}")
        _boolean(row, "answer_correct")
        _boolean(row, "counterfactual_consistent")
        result[sample_id] = row
    if not result:
        raise ValueError(f"{label} records must not be empty")
    return result


def _rate(rows: tuple[Mapping[str, object], ...], field: str) -> float:
    return float(np.mean([_boolean(row, field) for row in rows]))


def _compensation_accuracy(rows: tuple[Mapping[str, object], ...]) -> tuple[float | None, int]:
    compensatory = tuple(row for row in rows if _compensatory(row))
    if not compensatory:
        return None, 0
    return _rate(compensatory, "answer_correct"), len(compensatory)


def _factor_value(row: Mapping[str, object], factor: str) -> object:
    split_keys = row.get("split_keys")
    top_level = factor in row
    nested = isinstance(split_keys, Mapping) and factor in split_keys
    if top_level and nested:
        if canonical_json(row[factor]) != canonical_json(split_keys[factor]):
            raise ValueError(
                f"paired OOD record has conflicting factor {factor!r} in two locations"
            )
        raise ValueError(f"paired OOD record has ambiguous duplicate factor {factor!r} locations")
    if top_level:
        return row[factor]
    if nested:
        return split_keys[factor]
    raise ValueError(f"paired OOD record is missing preregistered factor {factor!r}")


def _invariant_payload(
    row: Mapping[str, object], shifted_factors: tuple[str, ...]
) -> dict[str, object]:
    payload = dict(row)
    for field in _OUTCOME_FIELDS:
        payload.pop(field, None)
    for factor in shifted_factors:
        payload.pop(factor, None)
    if "error_mechanism" in shifted_factors:
        payload.pop("error_catalog", None)
        for field in (
            "sample_id",
            "image_path",
            "image_sha256",
            "paired_sample_id",
            "error_family",
            "severity",
            "perceived_scene",
        ):
            payload.pop(field, None)
    split_keys = payload.get("split_keys")
    if isinstance(split_keys, Mapping):
        normalized_split_keys = dict(split_keys)
        for factor in shifted_factors:
            normalized_split_keys.pop(factor, None)
        payload["split_keys"] = normalized_split_keys
    return payload


def _validate_isolated_shift(
    sample_id: str,
    iid: Mapping[str, object],
    ood: Mapping[str, object],
    shifted_factors: tuple[str, ...],
) -> None:
    for factor in shifted_factors:
        if canonical_json(_factor_value(iid, factor)) == canonical_json(_factor_value(ood, factor)):
            raise ValueError(
                f"paired sample {sample_id!r} does not change preregistered factor {factor!r}"
            )
    if canonical_json(_invariant_payload(iid, shifted_factors)) != canonical_json(
        _invariant_payload(ood, shifted_factors)
    ):
        raise ValueError(
            f"paired sample {sample_id!r} changes fields other than the "
            "preregistered shift factors or measured outcomes"
        )


def compute_ood_metrics(
    iid_records: Iterable[object],
    ood_records: Iterable[object],
    *,
    preregistered_shift: Iterable[str],
) -> OODMetrics:
    """Compare paired samples only for explicitly preregistered shift factors."""

    shifted_factors = tuple(preregistered_shift)
    if not shifted_factors:
        raise ValueError("preregistered_shift must name at least one shifted factor")
    if any(not isinstance(factor, str) or not factor for factor in shifted_factors):
        raise ValueError("every preregistered shift factor must be a non-empty string")
    if len(set(shifted_factors)) != len(shifted_factors):
        raise ValueError("preregistered_shift contains duplicate factors")

    iid_by_id = _indexed(iid_records, "IID")
    ood_by_id = _indexed(ood_records, "OOD")
    if iid_by_id.keys() != ood_by_id.keys():
        raise ValueError("IID and OOD records must have paired sample_id values")
    identifiers = tuple(sorted(iid_by_id))
    for sample_id in identifiers:
        _validate_isolated_shift(
            sample_id,
            iid_by_id[sample_id],
            ood_by_id[sample_id],
            shifted_factors,
        )
    iid = tuple(iid_by_id[sample_id] for sample_id in identifiers)
    ood = tuple(ood_by_id[sample_id] for sample_id in identifiers)

    iid_accuracy = _rate(iid, "answer_correct")
    ood_accuracy = _rate(ood, "answer_correct")
    iid_consistency = _rate(iid, "counterfactual_consistent")
    ood_consistency = _rate(ood, "counterfactual_consistent")
    iid_compensation, iid_compensatory_count = _compensation_accuracy(iid)
    ood_compensation, ood_compensatory_count = _compensation_accuracy(ood)
    compensation_gap = (
        None
        if iid_compensation is None or ood_compensation is None
        else iid_compensation - ood_compensation
    )
    return OODMetrics(
        iid_accuracy=iid_accuracy,
        ood_accuracy=ood_accuracy,
        error_mechanism_generalization_gap=iid_accuracy - ood_accuracy,
        iid_counterfactual_consistency=iid_consistency,
        ood_counterfactual_consistency=ood_consistency,
        counterfactual_consistency_gap=iid_consistency - ood_consistency,
        iid_compensation_accuracy=iid_compensation,
        ood_compensation_accuracy=ood_compensation,
        compensation_generalization_gap=compensation_gap,
        iid_compensatory_count=iid_compensatory_count,
        ood_compensatory_count=ood_compensatory_count,
        shifted_factors=shifted_factors,
        n_pairs=len(identifiers),
    )
