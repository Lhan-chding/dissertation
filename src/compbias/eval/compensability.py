"""Rollout-level compensability tables and prompt-local covariances."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

LONG_TABLE_COLUMNS = (
    "sample_id",
    "error_id",
    "severity",
    "base_probability",
    "checkpoint",
    "rollout_seed",
    "reward",
    "c_hat",
    "c_ci_low",
    "c_ci_high",
)


def _record_mapping(record: object) -> dict[str, object]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record) and not isinstance(record, type):
        return {field.name: getattr(record, field.name) for field in fields(record)}
    to_mapping = getattr(record, "to_mapping", None)
    if callable(to_mapping):
        result = to_mapping()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError("each rollout record must be a mapping or dataclass")


def _required(row: Mapping[str, object], name: str) -> object:
    if name not in row:
        raise ValueError(f"rollout record is missing required field {name!r}")
    return row[name]


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _wilson_interval(success_rate: float, count: int, confidence: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / count
    center = (success_rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(success_rate * (1.0 - success_rate) / count + z * z / (4.0 * count**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def build_compensability_long_table(
    records: Iterable[object], *, confidence: float = 0.95
) -> pd.DataFrame:
    """Attach group estimates while retaining exactly one row per rollout."""

    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    raw_rows = tuple(_record_mapping(record) for record in records)
    if not raw_rows:
        raise ValueError("records must not be empty")

    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        view = _required(raw, "view")
        if view != "interventional":
            raise ValueError(
                "view must be interventional; natural and interventional records cannot be mixed"
            )
        seed = _required(raw, "rollout_seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("rollout_seed must be an integer")
        reward = _finite_number(_required(raw, "reward"), "reward")
        if reward not in {0.0, 1.0}:
            raise ValueError("reward must be binary (exactly 0 or 1) for the Wilson interval")
        base_probability = _finite_number(_required(raw, "base_probability"), "base_probability")
        if not 0.0 <= base_probability <= 1.0:
            raise ValueError("base_probability must lie in [0, 1]")
        rows.append(
            {
                "sample_id": _identifier(_required(raw, "sample_id"), "sample_id"),
                "error_id": _identifier(_required(raw, "error_id"), "error_id"),
                "severity": _finite_number(_required(raw, "severity"), "severity"),
                "base_probability": base_probability,
                "checkpoint": _identifier(_required(raw, "checkpoint"), "checkpoint"),
                "rollout_seed": int(seed),
                "reward": reward,
            }
        )

    key_columns = ["sample_id", "error_id", "checkpoint", "rollout_seed"]
    table = pd.DataFrame.from_records(rows)
    if table.duplicated(key_columns).any():
        raise ValueError("duplicate rollout_seed within a sample/error/checkpoint group")

    group_columns = ["sample_id", "error_id", "checkpoint"]
    severity_counts = table.groupby(group_columns, sort=False)["severity"].nunique(dropna=False)
    if (severity_counts != 1).any():
        raise ValueError("severity must be consistent within each sample/error/checkpoint group")

    estimates: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for key, group in table.groupby(group_columns, sort=False):
        c_hat = float(group["reward"].mean())
        low, high = _wilson_interval(c_hat, len(group), confidence)
        estimates[key] = (c_hat, low, high)

    group_keys = zip(table["sample_id"], table["error_id"], table["checkpoint"], strict=True)
    attached = [estimates[key] for key in group_keys]
    table = table.assign(
        c_hat=[item[0] for item in attached],
        c_ci_low=[item[1] for item in attached],
        c_ci_high=[item[2] for item in attached],
    )
    return table.loc[:, list(LONG_TABLE_COLUMNS)].copy()


def per_prompt_covariances(table: pd.DataFrame) -> pd.DataFrame:
    """Compute population covariance across errors separately for each prompt."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    required = {
        "sample_id",
        "error_id",
        "severity",
        "base_probability",
        "checkpoint",
        "c_hat",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing required columns: {', '.join(missing)}")
    if table.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "checkpoint",
                "severity_compensability_covariance",
                "n_errors",
                "base_probability_sum",
            ]
        )

    error_columns = ["sample_id", "checkpoint", "error_id"]
    consistency = table.groupby(error_columns, sort=False)[
        ["severity", "c_hat", "base_probability"]
    ].nunique()
    if (consistency.to_numpy() != 1).any():
        raise ValueError(
            "severity, c_hat, and base_probability must be constant for each prompt/error group"
        )
    per_error = table.drop_duplicates(error_columns).loc[
        :,
        [
            "sample_id",
            "checkpoint",
            "error_id",
            "severity",
            "c_hat",
            "base_probability",
        ],
    ]

    rows: list[dict[str, object]] = []
    for (sample_id, checkpoint), group in per_error.groupby(["sample_id", "checkpoint"], sort=True):
        severity = group["severity"].to_numpy(dtype=np.float64, copy=True)
        compensability = group["c_hat"].to_numpy(dtype=np.float64, copy=True)
        weights = group["base_probability"].to_numpy(dtype=np.float64, copy=True)
        if not (
            np.all(np.isfinite(severity))
            and np.all(np.isfinite(compensability))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError("severity, c_hat, and base_probability must be finite")
        weight_sum = float(np.sum(weights))
        if np.any(weights < 0.0) or not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("base_probability must be nonnegative and sum to one per prompt")
        severity_mean = float(weights @ severity)
        compensability_mean = float(weights @ compensability)
        covariance = float(
            weights @ ((severity - severity_mean) * (compensability - compensability_mean))
        )
        rows.append(
            {
                "sample_id": sample_id,
                "checkpoint": checkpoint,
                "severity_compensability_covariance": covariance,
                "n_errors": len(group),
                "base_probability_sum": weight_sum,
            }
        )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "sample_id",
            "checkpoint",
            "severity_compensability_covariance",
            "n_errors",
            "base_probability_sum",
        ],
    )
