"""Strict, filesystem-backed validation for the frozen CVA-v2 dataset."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from compbias.envs.cva_world.schema import CVASample, SemanticSplit
from compbias.io.manifests import canonical_json, manifest_sha256

FROZEN_CVA_V2_SAMPLE_COUNT = 1820
FROZEN_CVA_V2_ROUNDTRIP_COUNT = 4020
FROZEN_CVA_V2_VISUAL_STYLES = (
    "baseline",
    "font_weight_bold",
    "size_compact",
    "rotation_tilted",
    "contrast_low",
    "background_grid",
    "occlusion_local",
    "blur_mild",
    "distractor_marks",
    "layout_shifted",
)
_TASK_FAMILIES = (
    "digit_offset",
    "count_transform",
    "gauge_calibration",
    "bar_chart_aggregate",
    "relation_rule",
)
_SPLITS = tuple(split.value for split in SemanticSplit)
_FACTORS = ("visual_style", "error_mechanism")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_DATASET_BYTES = 128 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024

FROZEN_CVA_V2_GENERATOR_CONFIG = {
    "seed": 20260814,
    "samples_per_family_per_split": 10,
    "splits": list(_SPLITS),
    "task_families": list(_TASK_FAMILIES),
    "visual_styles": list(FROZEN_CVA_V2_VISUAL_STYLES),
    "train_error_mechanism": "offset_plus_2",
    "ood_error_mechanism": "offset_minus_2",
    "realizations_per_semantic": 2,
    "fully_cross_iid_visual_styles": True,
    "preregistered_ood_factors": list(_FACTORS),
}

_MANIFEST_FIELDS = frozenset(
    {
        "dataset_name",
        "schema_version",
        "sample_count",
        "sample_ids",
        "content_sha256",
        "config_sha256",
        "generator_config",
        "render_config",
        "dataset_file_sha256",
        "image_sha256",
        "jsonl_path",
        "images_dir",
        "rendered_image_count",
        "solver_checks",
        "solver_pass_rate",
        "roundtrip_checks",
        "roundtrip_pass_rate",
        "contact_sheets",
        "contact_sheet_sha256",
        "preregistered_ood_factors",
        "manifest_sha256",
    }
)


def _realization_count(family: str, split: str) -> int:
    if split == SemanticSplit.OOD_TEST.value:
        return 2
    return 9 if family in {"digit_offset", "gauge_calibration", "bar_chart_aggregate"} else 8


FROZEN_CVA_V2_SAMPLE_IDS = tuple(
    sorted(
        f"{family}_{split}_{index:06d}_r{realization:02d}"
        for family in _TASK_FAMILIES
        for split in _SPLITS
        for index in range(10)
        for realization in range(_realization_count(family, split))
    )
)


@dataclass(frozen=True, slots=True)
class FrozenCVADataset:
    """Validated dataset provenance and immutable parsed records."""

    manifest_file_sha256: str
    manifest_self_sha256: str
    content_sha256: str
    dataset_file_sha256: str
    image_sha256: Mapping[str, str]
    records: Mapping[str, CVASample]

    def sample_ids_for_partition(self, partition: str) -> tuple[str, ...]:
        try:
            split = SemanticSplit(partition)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported CVA-v2 partition: {partition!r}") from error
        return tuple(
            sorted(
                sample_id
                for sample_id, sample in self.records.items()
                if sample.split_keys.semantic_split is split
            )
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _validate_json_shape(value: object, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("JSON nesting exceeds 64 levels")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON contains a non-finite number")
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_json_shape(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_json_shape(child, depth=depth + 1)


def _strict_json(raw: bytes, *, name: str) -> object:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (RecursionError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must be strict UTF-8 JSON: {error}") from error
    _validate_json_shape(value)
    return value


def _lower_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _closed_mapping(value: object, fields: frozenset[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{name} must match its closed schema: missing={missing}, unknown={unknown}"
        )
    return value


def _safe_member(root: Path, value: object, *, name: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{name} must be a publishable POSIX path")
    published = PurePosixPath(value)
    if published.is_absolute() or any(part in {"", ".", ".."} for part in published.parts):
        raise ValueError(f"{name} must remain inside the approved artifact root")
    absolute_root = Path(os.path.abspath(root.expanduser()))
    candidate = (
        absolute_root.parent.joinpath(*published.parts)
        if published.parts[0] == absolute_root.name
        else absolute_root.joinpath(*published.parts)
    )
    if candidate != absolute_root and absolute_root not in candidate.parents:
        raise ValueError(f"{name} escapes the approved artifact root")
    cursor = absolute_root
    for part in candidate.relative_to(absolute_root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    if directory and not candidate.is_dir():
        raise ValueError(f"{name} is missing or is not a directory")
    if not directory and not candidate.is_file():
        raise ValueError(f"{name} is missing or is not a file")
    return candidate


def _load_manifest(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen cva_v2 manifest must be an existing non-symlink file")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("frozen cva_v2 manifest exceeds 16 MiB")
    value = _strict_json(path.read_bytes(), name="frozen cva_v2 manifest")
    return _closed_mapping(value, _MANIFEST_FIELDS, name="frozen cva_v2 manifest")


def _validate_manifest_contract(payload: Mapping[str, object]) -> tuple[str, ...]:
    if (
        payload["dataset_name"] != "cva_v2"
        or payload["schema_version"] != "2.0"
        or payload["sample_count"] != FROZEN_CVA_V2_SAMPLE_COUNT
        or payload["rendered_image_count"] != FROZEN_CVA_V2_SAMPLE_COUNT
        or payload["solver_checks"] != FROZEN_CVA_V2_SAMPLE_COUNT
        or payload["solver_pass_rate"] != 1.0
        or payload["roundtrip_checks"] != FROZEN_CVA_V2_ROUNDTRIP_COUNT
        or payload["roundtrip_pass_rate"] != 1.0
    ):
        raise ValueError("manifest is not the frozen cva_v2 1820-sample contract")
    if payload["generator_config"] != FROZEN_CVA_V2_GENERATOR_CONFIG:
        raise ValueError("frozen cva_v2 generator_config does not match")
    if payload["config_sha256"] != manifest_sha256(FROZEN_CVA_V2_GENERATOR_CONFIG):
        raise ValueError("frozen cva_v2 config SHA-256 does not match")
    if payload["render_config"] != {
        "height": 256,
        "samples_per_contact_sheet": 25,
        "width": 256,
    }:
        raise ValueError("frozen cva_v2 render_config does not match")
    if payload["preregistered_ood_factors"] != list(_FACTORS):
        raise ValueError("frozen cva_v2 OOD factors do not match")
    sample_ids = payload["sample_ids"]
    if sample_ids != list(FROZEN_CVA_V2_SAMPLE_IDS):
        raise ValueError("manifest must contain the exact 1820 frozen cva_v2 sample IDs")
    return FROZEN_CVA_V2_SAMPLE_IDS


def _validate_contact_sheet_manifest(payload: Mapping[str, object]) -> Mapping[str, str]:
    sheets = payload["contact_sheets"]
    expected_names = {f"cva_contact_sheet_{index:02d}.png" for index in range(1, 74)}
    if (
        not isinstance(sheets, list)
        or len(sheets) != 73
        or {PurePosixPath(str(path)).name for path in sheets} != expected_names
    ):
        raise ValueError("frozen cva_v2 manifest must list the exact 73 contact sheets")
    hashes = payload["contact_sheet_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != expected_names:
        raise ValueError("contact-sheet hash manifest must cover the exact 73 files")
    for name, digest in hashes.items():
        _lower_sha256(digest, name=f"contact sheet {name} SHA-256")
    return MappingProxyType({str(name): str(digest) for name, digest in hashes.items()})


def _read_dataset(path: Path) -> tuple[Mapping[str, object], ...]:
    if path.stat().st_size == 0 or path.stat().st_size > _MAX_DATASET_BYTES:
        raise ValueError("frozen cva_v2 JSONL must be non-empty and at most 128 MiB")
    records: list[Mapping[str, object]] = []
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip() or len(raw_line) > _MAX_LINE_BYTES:
                raise ValueError(f"dataset JSONL line {line_number} is blank or too large")
            value = _strict_json(raw_line, name=f"dataset JSONL line {line_number}")
            if not isinstance(value, Mapping):
                raise ValueError(f"dataset JSONL line {line_number} must be an object")
            records.append(value)
    return tuple(records)


def _validate_records(
    raw_records: tuple[Mapping[str, object], ...], sample_ids: tuple[str, ...]
) -> Mapping[str, CVASample]:
    parsed = tuple(CVASample.from_mapping(record) for record in raw_records)
    observed_ids = tuple(sample.sample_id for sample in parsed)
    if len(parsed) != FROZEN_CVA_V2_SAMPLE_COUNT or tuple(sorted(observed_ids)) != sample_ids:
        raise ValueError("dataset JSONL must contain the exact 1820 frozen cva_v2 records")
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("dataset JSONL contains duplicate sample IDs")
    for sample in parsed:
        if sample.image_path != f"images/{sample.sample_id}.png":
            raise ValueError(f"dataset image_path is not canonical for {sample.sample_id}")
    for raw, sample in zip(raw_records, parsed, strict=True):
        if canonical_json(raw) != canonical_json(sample.to_mapping()):
            raise ValueError("dataset JSONL contains noncanonical or hidden sample fields")
    return MappingProxyType({sample.sample_id: sample for sample in parsed})


def _validate_images(
    images_dir: Path,
    *,
    sample_ids: tuple[str, ...],
    expected_hashes: object,
) -> None:
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != set(sample_ids):
        raise ValueError("image hash manifest must cover the exact 1820 sample IDs")
    entries = tuple(images_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("frozen cva_v2 image tree may contain only regular PNG files")
    expected_names = {f"{sample_id}.png" for sample_id in sample_ids}
    if {entry.name for entry in entries} != expected_names:
        raise ValueError("image tree must contain exactly one PNG for each frozen sample ID")
    hashes_by_semantic_group: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        path = images_dir / f"{sample_id}.png"
        expected = _lower_sha256(expected_hashes[sample_id], name=f"image {sample_id} SHA-256")
        semantic_group = sample_id.rsplit("_r", maxsplit=1)[0]
        hashes_by_semantic_group.setdefault(semantic_group, []).append(expected)
        with path.open("rb") as stream:
            if stream.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                raise ValueError(f"image {sample_id} is not a PNG file")
        if _sha256_file(path) != expected:
            raise ValueError(f"image SHA-256 mismatch for {sample_id}")
    if len(hashes_by_semantic_group) != 250:
        raise ValueError("frozen cva_v2 must contain exactly 250 semantic render groups")
    for group, hashes in hashes_by_semantic_group.items():
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"render realizations are not byte-distinct for {group}")


def _validate_contact_sheet_paths(
    artifact_root: Path,
    *,
    published_paths: object,
    expected_hashes: Mapping[str, str],
) -> Mapping[str, Path]:
    assert isinstance(published_paths, list)
    paths: dict[str, Path] = {}
    for published in published_paths:
        path = _safe_member(
            artifact_root,
            published,
            name="contact sheet",
            directory=False,
        )
        if path.name in paths or path.name not in expected_hashes:
            raise ValueError("contact-sheet path inventory is duplicated or unexpected")
        if _sha256_file(path) != expected_hashes[path.name]:
            raise ValueError(f"contact-sheet SHA-256 mismatch for {path.name}")
        paths[path.name] = path
    return MappingProxyType(paths)


def _deterministic_replay(
    records: Mapping[str, CVASample],
    *,
    images_dir: Path,
    expected_image_hashes: object,
    contact_sheet_paths: Mapping[str, Path],
    expected_contact_sheet_hashes: Mapping[str, str],
) -> None:
    from compbias.envs.cva_world.canonical_solver import solve, solve_sample
    from compbias.envs.cva_world.corruptions import apply_error, reverse_error
    from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
    from compbias.envs.cva_world.renderer import (
        RenderConfig,
        render_sample,
        sample_render_coordinates,
    )

    expected_samples = generate_dataset(GeneratorConfig(**FROZEN_CVA_V2_GENERATOR_CONFIG))
    if {sample.sample_id for sample in expected_samples} != set(records):
        raise ValueError("frozen cva_v2 sample set differs from deterministic generation")
    if not isinstance(expected_image_hashes, Mapping):
        raise ValueError("image hash manifest is missing")
    rendered_batch: list[tuple[str, object]] = []
    sheet_index = 0
    roundtrip_count = 0
    for expected in expected_samples:
        observed = records[expected.sample_id]
        if canonical_json(observed.to_mapping()) != canonical_json(expected.to_mapping()):
            raise ValueError(
                f"frozen cva_v2 record differs from deterministic generation: {expected.sample_id}"
            )
        solve_sample(observed)
        for error in observed.error_catalog:
            perceived = apply_error(observed.scene, error)
            if reverse_error(perceived, error) != observed.scene:
                raise ValueError(
                    f"error round-trip failed for {observed.sample_id}/{error.error_id}"
                )
            solve(perceived, observed.question, observed.task_family)
            roundtrip_count += 1
        render_seed, realization_index = sample_render_coordinates(
            observed.sample_id,
            base_seed=20260814,
        )
        rendered = render_sample(
            observed,
            RenderConfig(
                width=256,
                height=256,
                seed=render_seed,
                realization_index=realization_index,
            ),
        )
        encoded = io.BytesIO()
        rendered.save(encoded, format="PNG", optimize=True)
        expected_hash = hashlib.sha256(encoded.getvalue()).hexdigest()
        if (
            expected_image_hashes.get(observed.sample_id) != expected_hash
            or _sha256_file(images_dir / f"{observed.sample_id}.png") != expected_hash
        ):
            rendered.close()
            raise ValueError(f"deterministic renderer mismatch for {observed.sample_id}")
        rendered_batch.append((observed.sample_id, rendered))
        if len(rendered_batch) == 25:
            sheet_index += 1
            _verify_replayed_sheet(
                rendered_batch,
                sheet_index=sheet_index,
                paths=contact_sheet_paths,
                expected_hashes=expected_contact_sheet_hashes,
            )
            rendered_batch = []
    if rendered_batch:
        sheet_index += 1
        _verify_replayed_sheet(
            rendered_batch,
            sheet_index=sheet_index,
            paths=contact_sheet_paths,
            expected_hashes=expected_contact_sheet_hashes,
        )
    if roundtrip_count != FROZEN_CVA_V2_ROUNDTRIP_COUNT or sheet_index != 73:
        raise ValueError("deterministic replay counts differ from the frozen contract")


def _verify_replayed_sheet(
    rendered: list[tuple[str, object]],
    *,
    sheet_index: int,
    paths: Mapping[str, Path],
    expected_hashes: Mapping[str, str],
) -> None:
    from compbias.envs.cva_world.renderer import build_contact_sheet

    name = f"cva_contact_sheet_{sheet_index:02d}.png"
    sheet = build_contact_sheet(rendered)
    encoded = io.BytesIO()
    sheet.save(encoded, format="PNG", optimize=True)
    sheet.close()
    for _sample_id, image in rendered:
        image.close()  # type: ignore[union-attr]
    digest = hashlib.sha256(encoded.getvalue()).hexdigest()
    if (
        expected_hashes.get(name) != digest
        or name not in paths
        or _sha256_file(paths[name]) != digest
    ):
        raise ValueError(f"deterministic contact-sheet replay mismatch for {name}")


def validate_frozen_cva_v2_dataset(manifest_path: Path, *, artifact_root: Path) -> FrozenCVADataset:
    """Validate the complete frozen manifest, raw JSONL, and 1820-image tree."""

    payload = _load_manifest(manifest_path)
    sample_ids = _validate_manifest_contract(payload)
    expected_sheet_hashes = _validate_contact_sheet_manifest(payload)
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    manifest_self_hash = manifest_sha256(unsigned)
    if payload["manifest_sha256"] != manifest_self_hash:
        raise ValueError("frozen cva_v2 manifest self SHA-256 does not match")
    dataset_path = _safe_member(
        artifact_root, payload["jsonl_path"], name="dataset JSONL", directory=False
    )
    images_dir = _safe_member(
        artifact_root, payload["images_dir"], name="dataset images", directory=True
    )
    contact_sheet_paths = _validate_contact_sheet_paths(
        artifact_root,
        published_paths=payload["contact_sheets"],
        expected_hashes=expected_sheet_hashes,
    )
    dataset_file_hash = _sha256_file(dataset_path)
    if dataset_file_hash != _lower_sha256(
        payload["dataset_file_sha256"], name="dataset file SHA-256"
    ):
        raise ValueError("frozen cva_v2 raw dataset SHA-256 does not match")
    raw_records = _read_dataset(dataset_path)
    content_hash = manifest_sha256(sorted(raw_records, key=lambda row: str(row["sample_id"])))
    if content_hash != _lower_sha256(payload["content_sha256"], name="content SHA-256"):
        raise ValueError("frozen cva_v2 canonical content SHA-256 does not match")
    records = _validate_records(raw_records, sample_ids)
    _validate_images(
        images_dir,
        sample_ids=sample_ids,
        expected_hashes=payload["image_sha256"],
    )
    _deterministic_replay(
        records,
        images_dir=images_dir,
        expected_image_hashes=payload["image_sha256"],
        contact_sheet_paths=contact_sheet_paths,
        expected_contact_sheet_hashes=expected_sheet_hashes,
    )
    return FrozenCVADataset(
        manifest_file_sha256=_sha256_file(manifest_path),
        manifest_self_sha256=manifest_self_hash,
        content_sha256=content_hash,
        dataset_file_sha256=dataset_file_hash,
        image_sha256=MappingProxyType(dict(payload["image_sha256"])),
        records=records,
    )
