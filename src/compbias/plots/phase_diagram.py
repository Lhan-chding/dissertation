"""Basin and vector-field phase diagram for coordination dynamics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from numpy.typing import NDArray

from compbias.theory.coordination import BasinMap, PhasePortrait

from ._common import OutputPath, _figure, _save_png, _title

_LABEL_ORDER = ("compensatory", "separatrix", "truthful")
_LABEL_CODE = {label: index for index, label in enumerate(_LABEL_ORDER)}
_LABEL_COLOR = {
    "compensatory": "#D55E00",
    "separatrix": "#B8B8B8",
    "truthful": "#0072B2",
}


def _structured_mesh(
    p: NDArray[np.float64], q: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if p.ndim != 2 or q.ndim != 2 or p.shape != q.shape or p.size == 0:
        raise ValueError("basin coordinates must be non-empty, shape-matched matrices")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("basin coordinates must contain only finite values")
    if np.any((p < 0.0) | (p > 1.0)) or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("basin coordinates must lie in the closed unit square")
    p_values = np.array(p[0], dtype=np.float64, copy=True)
    q_values = np.array(q[:, 0], dtype=np.float64, copy=True)
    if not np.allclose(p, p_values[None, :], rtol=0.0, atol=1e-12):
        raise ValueError("p must be constant down each mesh column")
    if not np.allclose(q, q_values[:, None], rtol=0.0, atol=1e-12):
        raise ValueError("q must be constant across each mesh row")
    if np.unique(p_values).size != p_values.size or np.unique(q_values).size != q_values.size:
        raise ValueError("basin mesh coordinates must be unique along each axis")
    return p_values, q_values


def _coordinate_edges(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.size == 1:
        return np.array([0.0, 1.0], dtype=np.float64)
    midpoint = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (values[1] - values[0]) / 2.0
    last = values[-1] + (values[-1] - values[-2]) / 2.0
    return np.clip(np.concatenate(([first], midpoint, [last])), 0.0, 1.0)


def _validated_data(
    basins: BasinMap,
    portrait: PhasePortrait | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    if not isinstance(basins, BasinMap):
        raise TypeError("basins must be a BasinMap")
    p = np.asarray(basins.p, dtype=np.float64)
    q = np.asarray(basins.q, dtype=np.float64)
    p_values, q_values = _structured_mesh(p, q)
    labels = np.asarray(basins.labels, dtype=str)
    if labels.shape != p.shape:
        raise ValueError("basin labels must match the coordinate mesh shape")
    unknown = sorted(set(np.unique(labels)) - set(_LABEL_ORDER))
    if unknown:
        raise ValueError(f"unknown basin labels: {', '.join(unknown)}")

    p_order = np.argsort(p_values, kind="stable")
    q_order = np.argsort(q_values, kind="stable")
    p_sorted = p_values[p_order]
    q_sorted = q_values[q_order]
    labels_sorted = labels[np.ix_(q_order, p_order)]
    codes = np.empty(labels_sorted.shape, dtype=np.int64)
    for label, code in _LABEL_CODE.items():
        codes[labels_sorted == label] = code

    if portrait is None:
        return p_sorted, q_sorted, codes, None, None
    if not isinstance(portrait, PhasePortrait):
        raise TypeError("phase_portrait must be a PhasePortrait or None")
    portrait_arrays = tuple(
        np.asarray(getattr(portrait, name), dtype=np.float64) for name in ("p", "q", "dp", "dq")
    )
    if any(array.shape != p.shape for array in portrait_arrays):
        raise ValueError("phase portrait arrays must match the basin mesh shape")
    if any(not np.all(np.isfinite(array)) for array in portrait_arrays):
        raise ValueError("phase portrait arrays must contain only finite values")
    if not np.allclose(portrait_arrays[0], p, rtol=0.0, atol=1e-12) or not np.allclose(
        portrait_arrays[1], q, rtol=0.0, atol=1e-12
    ):
        raise ValueError("phase portrait and basin meshes must use the same coordinates")
    dp = portrait_arrays[2][np.ix_(q_order, p_order)]
    dq = portrait_arrays[3][np.ix_(q_order, p_order)]
    return p_sorted, q_sorted, codes, dp, dq


def _draw_basin_axis(
    axis: Axes,
    basins: BasinMap,
    *,
    portrait: PhasePortrait | None,
    title: str,
) -> None:
    p, q, codes, dp, dq = _validated_data(basins, portrait)
    colors = [_LABEL_COLOR[label] for label in _LABEL_ORDER]
    color_map = ListedColormap(colors)
    normalizer = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), color_map.N)
    axis.pcolormesh(
        _coordinate_edges(p),
        _coordinate_edges(q),
        codes,
        cmap=color_map,
        norm=normalizer,
        shading="flat",
        alpha=0.82,
    )
    if dp is not None and dq is not None:
        p_mesh, q_mesh = np.meshgrid(p, q, indexing="xy")
        axis.quiver(
            p_mesh,
            q_mesh,
            dp,
            dq,
            color="#202020",
            angles="xy",
            scale_units="xy",
            scale=1.8,
            width=0.004,
        )

    present = set(np.unique(np.asarray(basins.labels, dtype=str)))
    handles = [
        Patch(facecolor=_LABEL_COLOR[label], edgecolor="none", label=label)
        for label in _LABEL_ORDER
        if label in present
    ]
    axis.legend(handles=handles, frameon=False, loc="upper left")
    axis.set_title(title)
    axis.set_xlabel("truthful perception probability, p")
    axis.set_ylabel("canonical reasoning probability, q")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")


def plot_basin_map(
    basins: BasinMap,
    output_path: OutputPath,
    *,
    phase_portrait: PhasePortrait | None = None,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Save basin labels, optionally overlaid with the aligned vector field."""

    plot_title = _title(title, default="Coordination basins")
    figure = _figure(width=5.8, height=5.0)
    axis = figure.add_subplot(1, 1, 1)
    _draw_basin_axis(axis, basins, portrait=phase_portrait, title=plot_title)
    figure.tight_layout(pad=1.1)
    return _save_png(figure, output_path, dpi=dpi)


__all__ = ["plot_basin_map"]
