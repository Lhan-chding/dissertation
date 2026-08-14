"""Predicted-versus-observed selection-law figures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.patches import Patch
from numpy.typing import ArrayLike, NDArray

from ._common import OutputPath, _figure, _save_png, _title


@dataclass(frozen=True, slots=True)
class _SelectionSeries:
    name: str
    predicted: NDArray[np.float64]
    observed: NDArray[np.float64]


def _series_name(value: object, *, fallback: str) -> str:
    name = fallback if value is None else value
    if not isinstance(name, str) or not name.strip():
        raise ValueError("selection series names must be non-empty strings")
    return name.strip()


def _probability_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite probability vector") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    if not np.isclose(np.sum(array), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must sum to one")
    return array


def _mapping_value(record: Mapping[str, Any], field: str) -> Any:
    if field not in record:
        raise ValueError(f"every selection record must contain {field!r}")
    return record[field]


def _record_series(record: object, *, fallback_name: str) -> _SelectionSeries:
    if isinstance(record, Mapping):
        predicted = _mapping_value(record, "predicted")
        observed = _mapping_value(record, "observed")
        name = _series_name(record.get("name"), fallback=fallback_name)
    elif hasattr(record, "predicted") and hasattr(record, "observed"):
        predicted = record.predicted  # type: ignore[attr-defined]
        observed = record.observed  # type: ignore[attr-defined]
        name = _series_name(getattr(record, "name", None), fallback=fallback_name)
    elif isinstance(record, Sequence) and not isinstance(record, (str, bytes)) and len(record) == 2:
        predicted, observed = record
        name = _series_name(None, fallback=fallback_name)
    else:
        raise TypeError("selection records must expose predicted and observed vectors")
    predicted_array = _probability_vector(predicted, name=f"predicted values for {name!r}")
    observed_array = _probability_vector(observed, name=f"observed values for {name!r}")
    if predicted_array.shape != observed_array.shape:
        raise ValueError(f"predicted and observed values for {name!r} must have the same shape")
    return _SelectionSeries(name, predicted_array, observed_array)


def _is_scalar_mapping(value: Mapping[object, object]) -> bool:
    return bool(value) and all(np.asarray(item).ndim == 0 for item in value.values())


def _paired_inputs(
    predicted: object,
    observed: object,
) -> tuple[tuple[_SelectionSeries, ...], tuple[str, ...] | None]:
    if isinstance(predicted, Mapping) and isinstance(observed, Mapping):
        if not predicted or not observed:
            raise ValueError("predicted and observed mappings must not be empty")
        if set(predicted) != set(observed):
            raise ValueError("predicted and observed mappings must use identical keys")
        if _is_scalar_mapping(predicted) and _is_scalar_mapping(observed):
            keys = tuple(sorted(predicted, key=str))
            labels = tuple(_series_name(key, fallback="") for key in keys)
            series = _record_series(
                {
                    "name": "selection",
                    "predicted": [predicted[key] for key in keys],
                    "observed": [observed[key] for key in keys],
                },
                fallback_name="selection",
            )
            return (series,), labels
        names = tuple(sorted(predicted, key=str))
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("selection mapping keys must be non-empty strings")
        series = tuple(
            _record_series(
                {"name": name, "predicted": predicted[name], "observed": observed[name]},
                fallback_name=name,
            )
            for name in names
        )
        return series, None
    series = _record_series(
        {"name": "selection", "predicted": predicted, "observed": observed},
        fallback_name="selection",
    )
    return (series,), None


def _normalise_series(
    results: object,
    observed: object | None,
) -> tuple[tuple[_SelectionSeries, ...], tuple[str, ...] | None]:
    if observed is not None:
        return _paired_inputs(results, observed)
    if isinstance(results, Mapping):
        if "predicted" in results or "observed" in results:
            return (_record_series(results, fallback_name="selection"),), None
        if not results:
            raise ValueError("selection results must not be empty")
        records = tuple(
            _record_series(record, fallback_name=name)
            for name, record in sorted(results.items(), key=lambda item: str(item[0]))
        )
    elif hasattr(results, "predicted") and hasattr(results, "observed"):
        records = (_record_series(results, fallback_name="selection"),)
    else:
        if isinstance(results, (str, bytes)) or not isinstance(results, Iterable):
            raise TypeError("results must be a selection record, mapping, or iterable")
        materialised = tuple(results)
        if not materialised:
            raise ValueError("selection results must not be empty")
        records = tuple(
            _record_series(record, fallback_name=f"series {index}")
            for index, record in enumerate(materialised)
        )
    names = tuple(record.name for record in records)
    if len(set(names)) != len(names):
        raise ValueError("selection series names must be unique")
    return records, None


def _action_labels(
    labels: Sequence[str] | None,
    *,
    count: int,
    inferred: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if labels is None:
        return inferred if inferred is not None else tuple(str(index) for index in range(count))
    if isinstance(labels, (str, bytes)) or len(labels) != count:
        raise ValueError(f"action_labels must contain exactly {count} labels")
    converted = tuple(labels)
    if any(not isinstance(label, str) or not label.strip() for label in converted):
        raise ValueError("action_labels must contain non-empty strings")
    if len(set(converted)) != len(converted):
        raise ValueError("action_labels must be unique")
    return tuple(label.strip() for label in converted)


def plot_selection_comparison(
    results: object,
    output_path: OutputPath,
    *,
    observed: object | None = None,
    action_labels: Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Plot predicted and observed distributions from results or paired mappings.

    ``results`` may be one result object, an iterable of result objects, or a
    mapping from series names to records.  For plain predicted/observed arrays
    or mappings, pass the observed values through the keyword argument.
    """

    series, inferred_labels = _normalise_series(results, observed)
    action_count = series[0].predicted.size
    if any(item.predicted.size != action_count for item in series):
        raise ValueError("all selection series must use the same number of actions")
    labels = _action_labels(action_labels, count=action_count, inferred=inferred_labels)
    plot_title = _title(title, default="Selection law: predicted versus observed")

    column_count = min(3, len(series))
    row_count = (len(series) + column_count - 1) // column_count
    figure = _figure(width=4.4 * column_count, height=3.2 * row_count + 0.7)
    positions = np.arange(action_count, dtype=np.float64)
    bar_width = 0.36
    for index, item in enumerate(series):
        axis = figure.add_subplot(row_count, column_count, index + 1)
        axis.bar(
            positions - bar_width / 2.0,
            item.predicted,
            width=bar_width,
            color="#0072B2",
            label="predicted",
            zorder=2,
        )
        axis.bar(
            positions + bar_width / 2.0,
            item.observed,
            width=bar_width,
            color="#E69F00",
            label="observed",
            zorder=2,
        )
        axis.set_title(item.name)
        axis.set_xticks(positions, labels)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("action / error type")
        axis.set_ylabel("probability")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)

    figure.suptitle(plot_title, y=0.99)
    figure.legend(
        handles=[
            Patch(facecolor="#0072B2", label="predicted"),
            Patch(facecolor="#E69F00", label="observed"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.9)
    return _save_png(figure, output_path, dpi=dpi)


__all__ = ["plot_selection_comparison"]
