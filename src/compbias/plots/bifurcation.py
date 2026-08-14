"""Symmetric KL coordination bifurcation figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from compbias.theory.coordination import BifurcationBranch

from ._common import OutputPath, _figure, _save_png, _title

_CRITICAL_RATIO = 0.5


def _validated_branch(
    branch: BifurcationBranch,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    if not isinstance(branch, BifurcationBranch):
        raise TypeError("branch must be a BifurcationBranch")
    arrays = tuple(
        np.asarray(getattr(branch, name), dtype=np.float64)
        for name in ("beta_over_a", "center", "positive", "negative")
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("bifurcation arrays must be one-dimensional")
    if arrays[0].size < 2 or any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("bifurcation arrays must share a length of at least two")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("bifurcation arrays must contain only finite values")

    ratio, center, positive, negative = arrays
    if np.any(ratio < 0.0):
        raise ValueError("beta_over_a must be nonnegative")
    order = np.argsort(ratio, kind="stable")
    ratio, center, positive, negative = (
        np.array(array[order], dtype=np.float64, copy=True) for array in arrays
    )
    if np.any(np.diff(ratio) <= 0.0):
        raise ValueError("beta_over_a values must be unique")
    if np.any((positive < 0.0) | (positive > 1.0)):
        raise ValueError("positive branch values must lie in [0, 1]")
    if np.any((negative < -1.0) | (negative > 0.0)):
        raise ValueError("negative branch values must lie in [-1, 0]")
    if not np.allclose(center, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("the symmetric center branch must be zero")
    if not np.allclose(negative, -positive, rtol=0.0, atol=1e-12):
        raise ValueError("positive and negative branches must be symmetric")
    return ratio, center, positive, negative


def _nonzero_branch_segment(
    ratio: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    below_critical = ratio < _CRITICAL_RATIO
    segment_x = ratio[below_critical]
    segment_y = values[below_critical]
    if segment_x.size and ratio[0] <= _CRITICAL_RATIO <= ratio[-1]:
        segment_x = np.append(segment_x, _CRITICAL_RATIO)
        segment_y = np.append(segment_y, 0.0)
    return segment_x, segment_y


def plot_bifurcation(
    branch: BifurcationBranch,
    output_path: OutputPath,
    *,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Save a deterministic pitchfork plot and return its :class:`~pathlib.Path`."""

    ratio, center, positive, negative = _validated_branch(branch)
    plot_title = _title(title, default="Symmetric coordination bifurcation")
    figure = _figure(width=6.4, height=4.4)
    axis = figure.add_subplot(1, 1, 1)

    branch_x, positive_segment = _nonzero_branch_segment(ratio, positive)
    _, negative_segment = _nonzero_branch_segment(ratio, negative)
    if branch_x.size:
        axis.plot(
            branch_x,
            positive_segment,
            color="#0072B2",
            linewidth=2.2,
            label="nonzero branches",
        )
        axis.plot(
            branch_x,
            negative_segment,
            color="#0072B2",
            linewidth=2.2,
        )
    axis.plot(
        ratio,
        center,
        color="#3D3D3D",
        linewidth=1.8,
        linestyle="--",
        label="center branch",
    )
    if ratio[0] <= _CRITICAL_RATIO <= ratio[-1]:
        axis.axvline(
            _CRITICAL_RATIO,
            color="#D55E00",
            linewidth=1.4,
            linestyle=":",
            label=r"critical $\beta/a=1/2$",
        )

    axis.set_title(plot_title)
    axis.set_xlabel(r"regularization ratio $\beta/a$")
    axis.set_ylabel(r"symmetric branch coordinate $m$")
    axis.set_xlim(float(ratio[0]), float(ratio[-1]))
    axis.set_ylim(-1.05, 1.05)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout(pad=1.1)
    return _save_png(figure, output_path, dpi=dpi)


__all__ = ["plot_bifurcation"]
