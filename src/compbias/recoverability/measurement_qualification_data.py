"""Deterministic model-free dataset writer for measurement qualification v1."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .measurement_qualification import (
    MeasurementQualificationConfig,
    QualificationDatasetRecord,
    QualificationScene,
    build_qualification_records,
)

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "status",
        "dataset_id",
        "seed",
        "image_size",
        "render_mode",
        "source_dataset_id",
        "source_dataset_records_sha256",
        "source_stage2_v2_external_evidence_sha256",
        "record_count",
        "split_counts",
        "strata_counts",
        "records_path",
        "records_sha256",
        "images_generated",
        "images_sha256",
        "numeric_table_overlap_with_source",
        "model_calls",
        "hypothesis_tested",
        "confirmatory_execution_authorized",
        "training_invoked",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_bundle_sha256(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(root / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _draw_chart(path: Path, *, record: QualificationDatasetRecord, size: tuple[int, int]) -> None:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = 70, 50, width - 40, height - 55
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
    max_value = max(20, max(record.values))
    for tick in range(max_value + 1):
        y = plot_bottom - tick / max_value * (plot_bottom - plot_top)
        if tick:
            draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.line((plot_left - 4, y, plot_left, y), fill="black", width=1)
        draw.text((plot_left - 28, y - 6), str(tick), fill="#374151")
    x_step = (plot_right - plot_left) / len(record.values)
    points: list[tuple[float, float]] = []
    for index, value in enumerate(record.values):
        x = plot_left + (index + 0.5) * x_step
        y = plot_bottom - value / max_value * (plot_bottom - plot_top)
        points.append((x, y))
        draw.text((x - 4, plot_bottom + 12), chr(ord("A") + index), fill="black")
    if record.chart_type == "grouped_bar":
        colors = ("#1d4ed8", "#ea580c", "#059669", "#7c3aed")
        for (x, y), color in zip(points, colors, strict=True):
            half = x_step * 0.22
            draw.rectangle((x - half, y, x + half, plot_bottom), fill=color)
    elif record.chart_type == "line":
        draw.line(points, fill="#1d4ed8", width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#111827")
    else:  # pragma: no cover - guarded by QualificationDatasetRecord construction
        raise ValueError("chart type is not registered")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _record_mapping(record: QualificationDatasetRecord) -> dict[str, object]:
    return {**asdict(record), "values": list(record.values)}


def write_measurement_qualification_dataset(
    config: MeasurementQualificationConfig,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
    output_dir: Path,
) -> dict[str, object]:
    """Write the registered 300-scene dataset exactly once without model calls."""

    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute path")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"qualification dataset output already exists: {output_dir}")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("qualification dataset output parent must be a regular directory")
    records = build_qualification_records(
        config,
        reserved_numeric_tables=reserved_numeric_tables,
    )
    reserved = frozenset(reserved_numeric_tables)
    output_dir.mkdir()
    try:
        for record in records:
            _draw_chart(output_dir / record.image, record=record, size=config.image_size)
        records_path = output_dir / "records.jsonl"
        with records_path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(_record_mapping(record), sort_keys=True, allow_nan=False) + "\n"
                )
        relative_images = tuple(record.image for record in records)
        strata = Counter(f"{record.chart_type}|{record.operation}" for record in records)
        overlap = len({record.values for record in records}.intersection(reserved))
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "recoverability_measurement_qualification_dataset",
            "status": "FROZEN_DATASET_NOT_EVALUATED",
            "dataset_id": config.dataset_id,
            "seed": config.seed,
            "image_size": list(config.image_size),
            "render_mode": "axis_scale_v0_3",
            "source_dataset_id": config.source_dataset_id,
            "source_dataset_records_sha256": config.source_dataset_records_sha256,
            "source_stage2_v2_external_evidence_sha256": (
                config.source_stage2_v2_external_evidence_sha256
            ),
            "record_count": len(records),
            "split_counts": {"qualification": len(records)},
            "strata_counts": dict(sorted(strata.items())),
            "records_path": "records.jsonl",
            "records_sha256": _sha256(records_path),
            "images_generated": len(relative_images),
            "images_sha256": _image_bundle_sha256(output_dir, relative_images),
            "numeric_table_overlap_with_source": overlap,
            "model_calls": 0,
            "hypothesis_tested": False,
            "confirmatory_execution_authorized": False,
            "training_invoked": False,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


@dataclass(frozen=True, slots=True)
class MeasurementQualificationDatasetVerification:
    verified: bool
    records_sha256: str
    images_sha256: str
    scenes: tuple[QualificationScene, ...]


def _load_manifest(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("qualification manifest must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification manifest must be UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("qualification manifest schema is invalid")
    return payload


def _load_record_rows(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("qualification records must be a regular file")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("qualification record must be JSON") from error
            if not isinstance(row, dict):
                raise ValueError("qualification record must be a JSON object")
            rows.append(row)
    return tuple(rows)


def verify_measurement_qualification_dataset(
    dataset_root: Path,
    *,
    config: MeasurementQualificationConfig,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
) -> MeasurementQualificationDatasetVerification:
    """Replay every record and image hash before any qualification model call."""

    if (
        not isinstance(dataset_root, Path)
        or not dataset_root.is_absolute()
        or dataset_root.is_symlink()
        or not dataset_root.is_dir()
    ):
        raise ValueError("qualification dataset root must be a regular absolute directory")
    manifest = _load_manifest(dataset_root / "manifest.json")
    expected_records = build_qualification_records(
        config,
        reserved_numeric_tables=reserved_numeric_tables,
    )
    expected_strata = Counter(
        f"{record.chart_type}|{record.operation}" for record in expected_records
    )
    required = {
        "schema_version": 1,
        "artifact_type": "recoverability_measurement_qualification_dataset",
        "status": "FROZEN_DATASET_NOT_EVALUATED",
        "dataset_id": config.dataset_id,
        "seed": config.seed,
        "image_size": list(config.image_size),
        "render_mode": "axis_scale_v0_3",
        "source_dataset_id": config.source_dataset_id,
        "source_dataset_records_sha256": config.source_dataset_records_sha256,
        "source_stage2_v2_external_evidence_sha256": (
            config.source_stage2_v2_external_evidence_sha256
        ),
        "record_count": config.scenes,
        "split_counts": {"qualification": config.scenes},
        "strata_counts": dict(sorted(expected_strata.items())),
        "records_path": "records.jsonl",
        "images_generated": config.scenes,
        "numeric_table_overlap_with_source": 0,
        "model_calls": 0,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("qualification manifest differs from the registered contract")
    records_path = dataset_root / "records.jsonl"
    records_sha256 = _sha256(records_path)
    if manifest.get("records_sha256") != records_sha256:
        raise ValueError("qualification records SHA-256 mismatch")
    rows = _load_record_rows(records_path)
    expected_rows = tuple(_record_mapping(record) for record in expected_records)
    if rows != expected_rows:
        raise ValueError("qualification records differ from deterministic replay")

    root = dataset_root.resolve()
    scenes: list[QualificationScene] = []
    relative_images: list[str] = []
    for record in expected_records:
        relative = Path(record.image)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("qualification image path must remain relative")
        image_path = (root / relative).resolve()
        if root not in image_path.parents or image_path.is_symlink() or not image_path.is_file():
            raise ValueError("qualification image must remain a regular file inside the dataset")
        relative_images.append(record.image)
        scenes.append(
            QualificationScene(
                scene_id=record.sample_id,
                image_path=image_path,
                chart_type=record.chart_type,
                operation=record.operation,
                values=record.values,
            )
        )
    images_sha256 = _image_bundle_sha256(root, relative_images)
    if manifest.get("images_sha256") != images_sha256:
        raise ValueError("qualification image bundle SHA-256 mismatch")
    return MeasurementQualificationDatasetVerification(
        verified=True,
        records_sha256=records_sha256,
        images_sha256=images_sha256,
        scenes=tuple(scenes),
    )
