"""Shared deterministic rendering utilities for paper-facing PNG artifacts."""

from __future__ import annotations

from io import BytesIO
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import TypeAlias

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

OutputPath: TypeAlias = str | PathLike[str]

_PNG_METADATA = {"Software": "compbias"}


def _output_path(value: OutputPath) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ValueError("output_path must not be empty")
    if isinstance(value, bytes):
        raise TypeError("output_path must be a string or path-like object")
    try:
        path = Path(value)
    except TypeError as error:
        raise TypeError("output_path must be a string or path-like object") from error
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must use the .png extension")
    if path.exists() and path.is_dir():
        raise ValueError("output_path must identify a PNG file, not a directory")
    return path


def _dpi(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("dpi must be an integer")
    converted = int(value)
    if not 72 <= converted <= 600:
        raise ValueError("dpi must lie between 72 and 600")
    return converted


def _title(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("title must be a non-empty string or None")
    return value.strip()


def _figure(*, width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _save_png(figure: Figure, output_path: OutputPath, *, dpi: int) -> Path:
    """Render through Agg before touching the target, then write stable PNG metadata."""

    path = _output_path(output_path)
    resolution = _dpi(dpi)
    buffer = BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=resolution,
            facecolor="white",
            edgecolor="white",
            metadata=_PNG_METADATA,
        )
        payload = buffer.getvalue()
    finally:
        buffer.close()
        figure.clear()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path
