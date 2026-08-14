"""Coordination summary tables and composite basin/bifurcation figures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from compbias.theory.coordination import BasinMap

from ._common import OutputPath, _figure, _save_png, _title
from .phase_diagram import _draw_basin_axis

_MISSING = object()
_COMPOSITE_FIELDS = frozenset(
    {
        "basin",
        "beta_over_a",
        "predicted_positive",
        "observed_positive",
        "observed_negative",
    }
)
_ROW_FIELDS = frozenset(
    {
        "seed",
        "basin",
        "basin_label",
        "equilibrium",
        "equilibrium_mode",
        "endpoint_label",
        "trajectory",
        "history",
    }
)


@dataclass(frozen=True, slots=True)
class _CoordinationRow:
    seed: int
    basin: str
    equilibrium: str
    initial: tuple[float, float] | None
    final: tuple[float, float] | None


def _field(record: object, *names: str) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return _MISSING


def _probability_pair(p: object, q: object, *, name: str) -> tuple[float, float]:
    try:
        pair = np.asarray((p, q), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite probabilities") from error
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite probabilities")
    if np.any((pair < 0.0) | (pair > 1.0)):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return float(pair[0]), float(pair[1])


def _history_trajectory(history: object) -> NDArray[np.float64]:
    if isinstance(history, (str, bytes)) or not isinstance(history, Iterable):
        raise ValueError("history must be a non-empty iterable of coordination checkpoints")
    checkpoints = tuple(history)
    if not checkpoints:
        raise ValueError("history must not be empty")
    pairs = []
    for checkpoint in checkpoints:
        p = _field(checkpoint, "truthful_perception_probability", "p")
        q = _field(checkpoint, "canonical_reasoning_probability", "q")
        if p is _MISSING or q is _MISSING:
            raise ValueError(
                "history checkpoints must expose perception and reasoning probabilities"
            )
        pairs.append(_probability_pair(p, q, name="history checkpoint"))
    return np.asarray(pairs, dtype=np.float64)


def _trajectory(value: object) -> NDArray[np.float64]:
    if isinstance(value, Mapping) and "p" in value and "q" in value:
        try:
            p = np.asarray(value["p"], dtype=np.float64)
            q = np.asarray(value["q"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("trajectory p and q values must be numeric") from error
        if p.ndim != 1 or q.shape != p.shape or p.size == 0:
            raise ValueError("trajectory p and q values must be non-empty aligned vectors")
        array = np.column_stack((p, q))
    else:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return _history_trajectory(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError("trajectory must have shape (time, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError("trajectory must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("trajectory probabilities must lie in [0, 1]")
    return np.array(array, dtype=np.float64, copy=True)


def _seed(value: object, *, fallback: int) -> int:
    if value is _MISSING:
        return fallback
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("coordination seeds must be non-negative integers")
    converted = int(value)
    if converted < 0:
        raise ValueError("coordination seeds must be non-negative integers")
    return converted


def _label(value: object, *, name: str) -> str | None:
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} labels must be non-empty strings")
    return value.strip()


def _endpoint_label(final: tuple[float, float]) -> str:
    score = final[0] + final[1] - 1.0
    if score > 1e-9:
        return "truthful"
    if score < -1e-9:
        return "compensatory"
    return "separatrix"


def _row(record: object, *, fallback_seed: int) -> _CoordinationRow:
    seed = _seed(_field(record, "seed"), fallback=fallback_seed)
    basin = _label(_field(record, "basin", "basin_label"), name="basin")
    equilibrium = _label(
        _field(record, "equilibrium", "equilibrium_mode", "endpoint_label"),
        name="equilibrium",
    )
    trajectory_value = _field(record, "trajectory", "history")
    path: NDArray[np.float64] | None = None
    if trajectory_value is not _MISSING:
        path = _trajectory(trajectory_value)
    elif not isinstance(record, Mapping) and not hasattr(record, "seed"):
        path = _trajectory(record)

    if path is not None:
        initial = (float(path[0, 0]), float(path[0, 1]))
        final = (float(path[-1, 0]), float(path[-1, 1]))
        derived = _endpoint_label(final)
        basin = basin or derived
        equilibrium = equilibrium or derived
    else:
        initial = None
        final = None
    if basin is None and equilibrium is None:
        raise ValueError("each coordination row needs a basin/equilibrium label or trajectory")
    return _CoordinationRow(
        seed=seed,
        basin=basin or equilibrium or "",
        equilibrium=equilibrium or basin or "",
        initial=initial,
        final=final,
    )


def _columnar_records(records: Mapping[str, object]) -> tuple[dict[str, object], ...] | None:
    seed_values = records.get("seed", _MISSING)
    if seed_values is _MISSING or isinstance(seed_values, (str, bytes)):
        return None
    try:
        seeds = tuple(seed_values)  # type: ignore[arg-type]
    except TypeError:
        return None
    if not seeds:
        raise ValueError("coordination records must not be empty")
    columns: dict[str, tuple[object, ...]] = {"seed": seeds}
    for name in _ROW_FIELDS - {"seed"}:
        if name not in records:
            continue
        value = records[name]
        if isinstance(value, (str, bytes)):
            return None
        try:
            column = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            return None
        if len(column) != len(seeds):
            raise ValueError("columnar coordination fields must have matching lengths")
        columns[name] = column
    return tuple(
        {name: column[index] for name, column in columns.items()} for index in range(len(seeds))
    )


def _normalise_rows(records: object) -> tuple[_CoordinationRow, ...]:
    if isinstance(records, Mapping):
        if set(records) & _ROW_FIELDS:
            materialised = _columnar_records(records) or (records,)
        else:
            if not records:
                raise ValueError("coordination records must not be empty")
            materialised = tuple(
                {"seed": seed, "trajectory": trajectory} for seed, trajectory in records.items()
            )
    elif any(hasattr(records, field) for field in _ROW_FIELDS):
        materialised = (records,)
    else:
        if isinstance(records, (str, bytes)) or not isinstance(records, Iterable):
            raise TypeError("records must be a coordination record, mapping, or iterable")
        materialised = tuple(records)
        if not materialised:
            raise ValueError("coordination records must not be empty")
    if len(materialised) > 100:
        raise ValueError("coordination summaries support at most 100 rows")
    rows = tuple(_row(record, fallback_seed=index) for index, record in enumerate(materialised))
    if len({row.seed for row in rows}) != len(rows):
        raise ValueError("coordination seeds must be unique")
    return tuple(sorted(rows, key=lambda row: row.seed))


def _format_probability(value: float) -> str:
    return f"{value:.3f}"


def _composite_array(
    payload: Mapping[str, object],
    name: str,
) -> NDArray[np.float64]:
    try:
        values = np.array(payload[name], dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional array") from error
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array of length at least two")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _composite_data(
    payload: Mapping[str, object],
) -> tuple[
    BasinMap,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    missing = sorted(_COMPOSITE_FIELDS - set(payload))
    if missing:
        raise ValueError(f"coordination composite is missing fields: {', '.join(missing)}")
    basins = payload["basin"]
    if not isinstance(basins, BasinMap):
        raise TypeError("basin must be a BasinMap")
    ratio = _composite_array(payload, "beta_over_a")
    predicted = _composite_array(payload, "predicted_positive")
    observed_positive = _composite_array(payload, "observed_positive")
    observed_negative = _composite_array(payload, "observed_negative")
    if any(
        values.shape != ratio.shape for values in (predicted, observed_positive, observed_negative)
    ):
        raise ValueError("coordination bifurcation arrays must have identical shapes")
    if np.any(ratio < 0.0):
        raise ValueError("beta_over_a must be nonnegative")
    if np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("predicted_positive values must lie in [0, 1]")
    if np.any((observed_positive < 0.0) | (observed_positive > 1.0)):
        raise ValueError("observed_positive values must lie in [0, 1]")
    if np.any((observed_negative < -1.0) | (observed_negative > 0.0)):
        raise ValueError("observed_negative values must lie in [-1, 0]")
    order = np.argsort(ratio, kind="stable")
    ratio = ratio[order]
    if np.any(np.diff(ratio) <= 0.0):
        raise ValueError("beta_over_a values must be unique")
    return (
        basins,
        ratio,
        predicted[order],
        observed_positive[order],
        observed_negative[order],
    )


def _plot_composite_summary(
    payload: Mapping[str, object],
    output_path: OutputPath,
    *,
    title: str | None,
    dpi: int,
) -> Path:
    basins, ratio, predicted, observed_positive, observed_negative = _composite_data(payload)
    plot_title = _title(title, default="Coordination basins and bifurcation")
    figure = _figure(width=11.4, height=4.9)
    basin_axis = figure.add_subplot(1, 2, 1)
    _draw_basin_axis(basin_axis, basins, portrait=None, title="Coordination basins")

    branch_axis = figure.add_subplot(1, 2, 2)
    branch_axis.plot(
        ratio,
        predicted,
        color="#0072B2",
        linewidth=2.1,
        label="predicted branches",
    )
    branch_axis.plot(ratio, -predicted, color="#0072B2", linewidth=2.1)
    branch_axis.scatter(
        ratio,
        observed_positive,
        color="#E69F00",
        edgecolors="white",
        linewidths=0.5,
        s=30,
        zorder=3,
        label="observed branches",
    )
    branch_axis.scatter(
        ratio,
        observed_negative,
        color="#E69F00",
        edgecolors="white",
        linewidths=0.5,
        s=30,
        zorder=3,
    )
    if ratio[0] <= 0.5 <= ratio[-1]:
        branch_axis.axvline(
            0.5,
            color="#D55E00",
            linewidth=1.3,
            linestyle=":",
            label=r"critical $\beta/a=1/2$",
        )
    branch_axis.axhline(
        0.0,
        color="#555555",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )
    branch_axis.set_title("Predicted and observed bifurcation")
    branch_axis.set_xlabel(r"regularization ratio $\beta/a$")
    branch_axis.set_ylabel(r"symmetric branch coordinate $m$")
    branch_axis.set_xlim(float(ratio[0]), float(ratio[-1]))
    branch_axis.set_ylim(-1.05, 1.05)
    branch_axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    branch_axis.legend(frameon=False, loc="best")

    figure.suptitle(plot_title, y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=1.1)
    return _save_png(figure, output_path, dpi=dpi)


def plot_coordination_summary(
    records: object,
    output_path: OutputPath,
    *,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Save a deterministic coordination summary.

    Accepted inputs are explicit record objects/mappings or mappings from seed
    to a ``(time, 2)`` probability trajectory.  A record may provide ``history``
    checkpoints instead of a numeric trajectory.  A mapping containing a
    ``BasinMap`` and predicted/observed branch arrays renders a two-panel
    basin/bifurcation summary instead.
    """

    if isinstance(records, Mapping) and set(records) >= _COMPOSITE_FIELDS:
        return _plot_composite_summary(
            records,
            output_path,
            title=title,
            dpi=dpi,
        )
    rows = _normalise_rows(records)
    plot_title = _title(title, default="Coordination outcome summary")
    has_coordinates = any(row.initial is not None for row in rows)
    headers = ["Seed", "Basin", "Equilibrium"]
    if has_coordinates:
        headers.extend(["p(0)", "q(0)", "p(T)", "q(T)"])
    cells: list[list[str]] = []
    for row in rows:
        values = [str(row.seed), row.basin, row.equilibrium]
        if has_coordinates:
            if row.initial is None or row.final is None:
                values.extend(["—", "—", "—", "—"])
            else:
                values.extend(
                    [
                        _format_probability(row.initial[0]),
                        _format_probability(row.initial[1]),
                        _format_probability(row.final[0]),
                        _format_probability(row.final[1]),
                    ]
                )
        cells.append(values)

    counts = Counter(row.equilibrium for row in rows)
    count_text = ", ".join(f"{label}={counts[label]}" for label in sorted(counts))
    height = min(30.0, max(3.2, 1.8 + 0.34 * len(rows)))
    figure = _figure(width=9.0 if has_coordinates else 6.2, height=height)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_axis_off()
    table = axis.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#E8EEF4")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    figure.suptitle(f"{plot_title}\n{count_text}", y=0.98)
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.91), pad=0.8)
    return _save_png(figure, output_path, dpi=dpi)


__all__ = ["plot_coordination_summary"]
