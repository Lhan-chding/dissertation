"""Separate estimators for selection, forked, and synthetic compensability."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass

import pandas as pd

_GROUP = ["sample_id", "checkpoint_id", "interface_id", "error_type"]


def _mapping(record: object) -> dict[str, object]:
    if isinstance(record, Mapping):
        return dict(record)
    to_mapping = getattr(record, "to_mapping", None)
    if callable(to_mapping):
        result = to_mapping()
        if isinstance(result, Mapping):
            return dict(result)
    if is_dataclass(record) and not isinstance(record, type):
        return {field.name: getattr(record, field.name) for field in fields(record)}
    raise TypeError("records must be mappings or dataclasses")


def _rows(records: Iterable[object], name: str) -> list[dict[str, object]]:
    rows = [_mapping(record) for record in records]
    if not rows:
        raise ValueError(f"{name} must not be empty")
    return rows


def estimate_selection_compensability(records: Iterable[object]) -> pd.DataFrame:
    """Estimate c_sel from natural original trajectories, conditional on input."""

    rows = _rows(records, "natural records")
    required = {*_GROUP, "original_reward", "source_kind", "rollout_id"}
    for row in rows:
        if row.get("source_kind") != "natural":
            raise ValueError("selection compensability requires natural trajectory records")
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"natural record is missing fields: {', '.join(missing)}")
        if row["original_reward"] not in {0, 1} or isinstance(row["original_reward"], bool):
            raise ValueError("original_reward must be binary")
    table = pd.DataFrame.from_records(rows)
    if table.duplicated([*_GROUP, "rollout_id"]).any():
        raise ValueError("natural records contain duplicate rollout identifiers")
    result = (
        table.groupby(_GROUP, sort=True, dropna=False)["original_reward"]
        .agg(c_sel="mean", n_natural_rollouts="size")
        .reset_index()
    )
    return result


def _fork_estimate(
    mediator_rows: list[dict[str, object]],
    fork_rows: list[dict[str, object]],
    *,
    source_kind: str,
    error_field: str,
    estimate_name: str,
    count_name: str,
) -> pd.DataFrame:
    metadata: dict[str, dict[str, object]] = {}
    for row in mediator_rows:
        record_id = row.get("mediator_record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("each mediator record requires mediator_record_id")
        if record_id in metadata:
            raise ValueError("mediator_record_id values must be unique")
        error_type = row.get(error_field)
        if not isinstance(error_type, str) or not error_type:
            raise ValueError(f"each mediator record requires {error_field}")
        metadata[record_id] = {
            "sample_id": row.get("sample_id"),
            "checkpoint_id": row.get("checkpoint_id"),
            "interface_id": row.get("interface_id"),
            "error_type": error_type,
        }
    joined: list[dict[str, object]] = []
    seen_forks: set[tuple[str, object]] = set()
    for row in fork_rows:
        if row.get("source_kind") != source_kind:
            raise ValueError(f"{estimate_name} requires {source_kind} fork records")
        record_id = row.get("mediator_record_id")
        if record_id not in metadata:
            raise ValueError("fork record references an unknown mediator_record_id")
        key = (str(record_id), row.get("fork_id"))
        if key in seen_forks:
            raise ValueError("fork records contain duplicate fork identifiers")
        seen_forks.add(key)
        reward = row.get("reward")
        if reward not in {0, 1} or isinstance(reward, bool):
            raise ValueError("fork reward must be binary")
        joined.append(
            {**metadata[str(record_id)], "mediator_record_id": record_id, "reward": reward}
        )
    represented = {str(row["mediator_record_id"]) for row in joined}
    if represented != set(metadata):
        raise ValueError("every mediator record must have at least one fork")

    table = pd.DataFrame.from_records(joined)
    per_mediator = (
        table.groupby([*_GROUP, "mediator_record_id"], sort=True, dropna=False)["reward"]
        .mean()
        .rename("mediator_reward")
        .reset_index()
    )
    grouped = (
        per_mediator.groupby(_GROUP, sort=True, dropna=False)["mediator_reward"]
        .agg(**{estimate_name: "mean", count_name: "size"})
        .reset_index()
    )
    return grouped


def estimate_forked_compensability(
    mediator_records: Iterable[object], fork_records: Iterable[object]
) -> pd.DataFrame:
    """Estimate c_fork after averaging continuations within natural mediator."""

    return _fork_estimate(
        _rows(mediator_records, "natural mediator records"),
        _rows(fork_records, "natural fork records"),
        source_kind="natural",
        error_field="error_type",
        estimate_name="c_fork",
        count_name="n_natural_mediators",
    )


def estimate_synthetic_compensability(
    mediator_records: Iterable[object], fork_records: Iterable[object]
) -> pd.DataFrame:
    """Estimate c_syn separately; never relabel synthetic states as natural."""

    return _fork_estimate(
        _rows(mediator_records, "synthetic mediator records"),
        _rows(fork_records, "synthetic fork records"),
        source_kind="synthetic",
        error_field="target_error_type",
        estimate_name="c_syn",
        count_name="n_synthetic_mediators",
    )


def merge_compensability_estimates(
    selection: pd.DataFrame,
    forked: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> pd.DataFrame:
    """Join, but never pool, the three estimands and attach the two gaps."""

    for table, column in ((selection, "c_sel"), (forked, "c_fork"), (synthetic, "c_syn")):
        if not isinstance(table, pd.DataFrame) or column not in table.columns:
            raise ValueError(f"{column} table is invalid")
        missing = sorted(set(_GROUP).difference(table.columns))
        if missing:
            raise ValueError(f"{column} table is missing keys: {', '.join(missing)}")
    merged = selection.merge(forked, on=_GROUP, how="outer", validate="one_to_one")
    merged = merged.merge(synthetic, on=_GROUP, how="outer", validate="one_to_one")
    merged = merged.assign(
        mediator_gap=merged["c_sel"] - merged["c_fork"],
        transport_gap=merged["c_syn"] - merged["c_fork"],
    )
    return merged.sort_values(_GROUP, kind="stable").reset_index(drop=True)
