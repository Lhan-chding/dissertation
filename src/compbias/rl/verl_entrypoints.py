"""Pure, audited execution plans for the veRL large-GPU boundary.

The builders in this module validate a proposed run and serialize metadata.
They deliberately do not import a training framework, inspect the filesystem,
download a model, spawn a process, or claim that training has started.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from compbias.models.qwen_vl import (
    DEFAULT_MODEL_NAME,
    PINNED_MODEL_REVISION,
    PINNED_TRANSFORMERS_REVISION,
    PINNED_VERL_DOCKERFILE_SHA256,
    PINNED_VERL_REVISION,
    PINNED_VLLM_REVISION,
    ModelSnapshotEvidence,
    VLMPreflightConfig,
    VLMPreflightReport,
    _build_verl_grpo_config,
    probe_local_cuda_devices,
    require_frozen_qwen_stack,
    revalidate_model_snapshot,
    validate_preflight,
    verify_model_snapshot,
)

AUDITED_GRPO_LEAF_KEYS = (
    "algorithm.adv_estimator",
    "data.image_key",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.kl_loss_coef",
    "actor_rollout_ref.rollout.name",
    "actor_rollout_ref.rollout.n",
    "trainer.project_name",
    "trainer.experiment_name",
    "trainer.nnodes",
    "trainer.n_gpus_per_node",
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.total_epochs",
)

_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PUBLIC_REVIEWER = re.compile(r"reviewer-[a-z0-9][a-z0-9-]{0,30}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_GPU_UUID = re.compile(r"(?:GPU|MIG-GPU)-[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_MINIMUM_ALLOWED_PARSE_RATE = 0.98
_MAX_AUDIT_BYTES = 1024 * 1024
_MAX_DATASET_BYTES = 128 * 1024 * 1024
_MAX_DATASET_LINE_BYTES = 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000

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
_FROZEN_CVA_V2_STYLE_COUNTS = {
    "baseline": 200,
    "font_weight_bold": 120,
    "size_compact": 200,
    "rotation_tilted": 200,
    "contrast_low": 200,
    "background_grid": 200,
    "occlusion_local": 200,
    "blur_mild": 200,
    "distractor_marks": 200,
    "layout_shifted": 100,
}
_FROZEN_CVA_V2_SAMPLE_COUNT = 1820
_FROZEN_CVA_V2_ROUNDTRIP_COUNT = 4020
_FROZEN_CVA_V2_CONTACT_SHEET_COUNT = 73
_FROZEN_CVA_V2_CONTACT_SHEETS = tuple(
    f"figures/cva_v2/cva_contact_sheet_{index:02d}.png"
    for index in range(1, _FROZEN_CVA_V2_CONTACT_SHEET_COUNT + 1)
)
_FROZEN_CVA_V2_FACTORS = ("visual_style", "error_mechanism")
_FROZEN_CVA_V2_SPLITS = ("train", "calibration", "val", "iid_test", "ood_test")
_FROZEN_CVA_V2_TASK_FAMILIES = (
    "digit_offset",
    "count_transform",
    "gauge_calibration",
    "bar_chart_aggregate",
    "relation_rule",
)
_FROZEN_CVA_V2_SAMPLE_IDS = tuple(
    sorted(
        f"{family}_{split}_{index:06d}_r{realization:02d}"
        for family in _FROZEN_CVA_V2_TASK_FAMILIES
        for split in _FROZEN_CVA_V2_SPLITS
        for index in range(10)
        for realization in range(
            2
            if split == "ood_test"
            else (
                9 if family in {"digit_offset", "gauge_calibration", "bar_chart_aggregate"} else 8
            )
        )
    )
)
_FROZEN_CVA_V2_VISUAL_APPLICABILITY = {
    style: (
        ("digit_offset", "gauge_calibration", "bar_chart_aggregate")
        if style == "font_weight_bold"
        else _FROZEN_CVA_V2_TASK_FAMILIES
    )
    for style in FROZEN_CVA_V2_VISUAL_STYLES
}
_FROZEN_CVA_V2_APPLICABLE_SAMPLE_COUNTS = {
    "baseline": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "font_weight_bold": {
        "digit_offset": 40,
        "gauge_calibration": 40,
        "bar_chart_aggregate": 40,
    },
    "size_compact": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "rotation_tilted": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "contrast_low": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "background_grid": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "occlusion_local": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "blur_mild": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "distractor_marks": {family: 40 for family in _FROZEN_CVA_V2_TASK_FAMILIES},
    "layout_shifted": dict.fromkeys(_FROZEN_CVA_V2_TASK_FAMILIES, 20),
}
_FROZEN_CVA_V2_GENERATOR_CONFIG = {
    "seed": 20260814,
    "samples_per_family_per_split": 10,
    "splits": list(_FROZEN_CVA_V2_SPLITS),
    "task_families": list(_FROZEN_CVA_V2_TASK_FAMILIES),
    "visual_styles": list(FROZEN_CVA_V2_VISUAL_STYLES),
    "train_error_mechanism": "offset_plus_2",
    "ood_error_mechanism": "offset_minus_2",
    "realizations_per_semantic": 2,
    "fully_cross_iid_visual_styles": True,
    "preregistered_ood_factors": list(_FROZEN_CVA_V2_FACTORS),
}
_PHASE_D_REPORT_FIELDS = frozenset(
    {
        "audit_report_schema_version",
        "sample_count",
        "split_audit",
        "split_audit_error",
        "split_clean",
        "solver_passes",
        "solver_pass_rate",
        "roundtrip_passes",
        "roundtrip_total",
        "roundtrip_pass_rate",
        "error_solver_passes",
        "error_solver_pass_rate",
        "rendered_image_count",
        "missing_images",
        "extra_images",
        "image_set_matches",
        "rendered_image_count_matches",
        "contact_sheet_sha256_matches",
        "contact_sheet_hash_mismatches",
        "manifest_sample_count_matches",
        "manifest_sample_ids_match",
        "manifest_content_sha256_matches",
        "manifest_config_sha256_matches",
        "manifest_dataset_file_sha256_matches",
        "manifest_image_sha256_matches",
        "manifest_self_sha256_matches",
        "preregistered_ood_factors_match_config",
        "noncanonical_rows",
        "image_path_mismatches",
        "privacy_issues",
        "image_question_answer_collisions",
        "style_counterbalance_violations",
        "evidence_manifest_sha256",
        "evidence_image_set_sha256",
        "visual_review_present",
        "human_reviewer_signoff",
        "human_review_binding_matches",
        "human_review",
        "visual_factor_realization_audit",
        "ood_image_shift",
        "style_semantic_joint_independence",
        "deterministic_replay",
        "answer_balance",
        "dataset",
        "automatic_audit_clean",
        "phase_d_ready",
    }
)
_DATASET_MANIFEST_FIELDS = frozenset(
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


def _freeze(value: Any) -> Any:
    """Create an immutable copy of a JSON-shaped value."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("plan mapping keys must be non-empty strings")
            frozen[key] = _freeze(child)
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported execution-plan value: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    """Return a fresh JSON-shaped copy suitable for serialization."""

    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _mapping_leaf_keys(value: Mapping[str, Any], prefix: str = "") -> set[str]:
    leaves: set[str] = set()
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            leaves.update(_mapping_leaf_keys(child, path))
        else:
            leaves.add(path)
    return leaves


def _validated_devices(
    cuda_available: bool,
    gpu_devices: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if not isinstance(cuda_available, bool):
        raise TypeError("cuda_available must be a boolean")
    if not isinstance(gpu_devices, (tuple, list)):
        raise TypeError("gpu_devices must be a tuple or list of device names")
    devices = tuple(gpu_devices)
    if any(not isinstance(device, str) or not device.strip() for device in devices):
        raise ValueError("GPU device names must be non-empty strings")
    return devices


def _validated_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _validated_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_run_name(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SAFE_RUN_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be 1-128 ASCII letters, digits, dots, hyphens, or underscores "
            "and begin with a letter or digit"
        )
    return value


def _validated_manifest(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("dataset_manifest must be a string")
    if not value.strip() or "\x00" in value:
        raise ValueError("dataset_manifest must be a non-empty path without NUL bytes")
    return value


def _validated_parse_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_parse_rate must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("minimum_parse_rate must be finite")
    if not _MINIMUM_ALLOWED_PARSE_RATE <= converted <= 1.0:
        raise ValueError("minimum_parse_rate must be between 0.98 and 1.0")
    return converted


def _validated_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return value.lower()


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"execution evidence must contain a {key!r} object")
    return value


def _read_hashed_json(path: str | Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    expected = _validated_sha256(expected_sha256, name=f"{name} SHA-256")
    source = Path(path).expanduser()
    if source.is_symlink():
        raise RuntimeError(f"{name} artifact may not be a symlink")
    if not source.is_file():
        raise RuntimeError(f"{name} artifact is missing: {source}")
    if source.stat().st_size > _MAX_AUDIT_BYTES:
        raise RuntimeError(f"{name} artifact exceeds the 1 MiB safety limit")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(f"{name} artifact SHA-256 does not match the reviewed hash")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError(f"{name} artifact must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} artifact must contain a JSON object")
    _validate_json_shape(payload)
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON numbers are forbidden")
    return parsed


def _validate_json_shape(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if depth > _MAX_JSON_DEPTH or visited > _MAX_JSON_NODES:
            raise ValueError("JSON artifact exceeds the permitted depth or complexity")
        if isinstance(current, dict):
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)


def _sha256_file(path: Path, *, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be an existing non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file_sha256(path: object, expected_sha256: object, *, name: str) -> Path:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError(f"{name} path must be a non-empty string")
    expected = _validated_sha256(expected_sha256, name=f"{name} SHA-256")
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not hmac.compare_digest(_sha256_file(resolved, name=name), expected):
        raise RuntimeError(f"{name} SHA-256 does not match its audit")
    return resolved


def _canonical_manifest_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise RuntimeError(
            f"{name} must match the closed schema: missing={missing}, unknown={unknown}"
        )


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _validate_publishable_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RuntimeError(f"{name} must be a non-empty publishable POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RuntimeError(f"{name} must stay within its published artifact root")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_traversal(path: Path, *, boundary: Path, name: str) -> None:
    if path != boundary and boundary not in path.parents:
        raise RuntimeError(f"{name} escapes the bound dataset artifact root")
    current = path
    while True:
        if current.is_symlink():
            raise RuntimeError(f"{name} may not traverse a symlink")
        if current == boundary:
            break
        current = current.parent


def _dataset_artifact_root(
    manifest_path: Path,
    manifest_publishable_path: str,
) -> Path:
    relative = PurePosixPath(
        _validate_publishable_path(
            manifest_publishable_path,
            name="Phase D manifest_path",
        )
    )
    lexical_manifest = _lexical_absolute(manifest_path)
    root = lexical_manifest
    for _part in relative.parts:
        root = root.parent
    expected_manifest = root.joinpath(*relative.parts)
    if expected_manifest != lexical_manifest:
        raise RuntimeError(
            "absolute dataset manifest path is not bound to the Phase D publishable path"
        )
    if not root.is_dir():
        raise RuntimeError("bound dataset artifact root is missing")
    _reject_symlink_traversal(
        lexical_manifest,
        boundary=root,
        name="dataset manifest",
    )
    return root


def _bound_dataset_member(
    root: Path,
    publishable_path: object,
    *,
    name: str,
    directory: bool,
) -> Path:
    relative = PurePosixPath(_validate_publishable_path(publishable_path, name=name))
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_traversal(candidate, boundary=root, name=name)
    if directory:
        if not candidate.is_dir():
            raise RuntimeError(f"{name} must be an existing directory")
    elif not candidate.is_file():
        raise RuntimeError(f"{name} must be an existing file")
    return candidate


def _strict_json_value(raw: bytes, *, name: str) -> object:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise RuntimeError(f"{name} must be valid strict UTF-8 JSON") from error
    _validate_json_shape(value)
    return value


def _validate_dataset_jsonl(
    path: Path,
    *,
    expected_file_sha256: object,
    expected_content_sha256: object,
    expected_sample_ids: list[str],
) -> None:
    expected_file_hash = _validated_sha256(
        expected_file_sha256,
        name="dataset JSONL file SHA-256",
    )
    if path.stat().st_size > _MAX_DATASET_BYTES:
        raise RuntimeError("dataset JSONL exceeds the 128 MiB safety limit")
    if not hmac.compare_digest(_sha256_file(path, name="dataset JSONL"), expected_file_hash):
        raise RuntimeError("dataset JSONL SHA-256 differs from the frozen manifest")
    records: list[Mapping[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if len(raw_line) > _MAX_DATASET_LINE_BYTES:
                raise RuntimeError(f"dataset JSONL line {line_number} exceeds 1 MiB")
            if not raw_line.strip():
                raise RuntimeError(f"dataset JSONL line {line_number} is blank")
            value = _strict_json_value(raw_line, name=f"dataset JSONL line {line_number}")
            if not isinstance(value, Mapping):
                raise RuntimeError(f"dataset JSONL line {line_number} must be an object")
            records.append(value)
    sample_ids = [record.get("sample_id") for record in records]
    if (
        any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or sorted(sample_ids) != expected_sample_ids
    ):
        raise RuntimeError("dataset JSONL does not contain the exact frozen sample IDs")
    ordered_records = sorted(records, key=lambda record: record["sample_id"])
    expected_content_hash = _validated_sha256(
        expected_content_sha256,
        name="dataset content SHA-256",
    )
    if not hmac.compare_digest(
        _canonical_manifest_sha256(ordered_records),
        expected_content_hash,
    ):
        raise RuntimeError("dataset JSONL canonical content differs from the frozen manifest")


def _validate_dataset_image_tree(
    images_dir: Path,
    *,
    sample_ids: list[str],
    expected_hashes: Mapping[str, Any],
) -> None:
    expected_names = {f"{sample_id}.png" for sample_id in sample_ids}
    entries = tuple(images_dir.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise RuntimeError("dataset image tree may not contain symlinks")
        if not entry.is_file():
            raise RuntimeError("dataset image tree may contain only PNG files")
    observed_names = {entry.name for entry in entries}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)[:3]
        extra = sorted(observed_names - expected_names)[:3]
        raise RuntimeError(
            "dataset image tree differs from the exact 1820 PNG inventory: "
            f"missing={missing}, extra={extra}"
        )
    for sample_id in sample_ids:
        path = images_dir / f"{sample_id}.png"
        expected = _validated_sha256(
            expected_hashes.get(sample_id),
            name=f"dataset image {sample_id} SHA-256",
        )
        if not hmac.compare_digest(_sha256_file(path, name="dataset image"), expected):
            raise RuntimeError(f"dataset image SHA-256 mismatch: {sample_id}")


def _validate_contact_sheet_tree(root: Path, payload: Mapping[str, Any]) -> None:
    contact_sheets = payload.get("contact_sheets")
    if contact_sheets != list(_FROZEN_CVA_V2_CONTACT_SHEETS):
        raise RuntimeError("dataset manifest must list the exact 73 contact-sheet paths")
    sheet_dir = _bound_dataset_member(
        root,
        "figures/cva_v2",
        name="contact-sheet directory",
        directory=True,
    )
    expected_names = {PurePosixPath(path).name for path in _FROZEN_CVA_V2_CONTACT_SHEETS}
    entries = tuple(sheet_dir.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise RuntimeError("contact-sheet tree may not contain symlinks")
        if not entry.is_file():
            raise RuntimeError("contact-sheet tree may contain only PNG files")
    observed_names = {entry.name for entry in entries}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)[:3]
        extra = sorted(observed_names - expected_names)[:3]
        raise RuntimeError(
            "contact-sheet tree differs from the exact 73-file inventory: "
            f"missing={missing}, extra={extra}"
        )
    hashes = payload.get("contact_sheet_sha256")
    if not isinstance(hashes, Mapping):
        raise RuntimeError("dataset manifest lacks contact-sheet hashes")
    for name in sorted(expected_names):
        expected = _validated_sha256(
            hashes.get(name),
            name=f"contact sheet {name} SHA-256",
        )
        if not hmac.compare_digest(
            _sha256_file(sheet_dir / name, name="contact sheet"),
            expected,
        ):
            raise RuntimeError(f"contact-sheet SHA-256 mismatch: {name}")


def _validate_cva_v2_data_tree(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_publishable_path: str,
) -> None:
    root = _dataset_artifact_root(manifest_path, manifest_publishable_path)
    dataset_path = _bound_dataset_member(
        root,
        payload.get("jsonl_path"),
        name="dataset jsonl_path",
        directory=False,
    )
    images_dir = _bound_dataset_member(
        root,
        payload.get("images_dir"),
        name="dataset images_dir",
        directory=True,
    )
    sample_ids = payload.get("sample_ids")
    image_hashes = payload.get("image_sha256")
    if not isinstance(sample_ids, list) or not isinstance(image_hashes, Mapping):
        raise RuntimeError("dataset manifest lacks the frozen data-tree inventory")
    _validate_dataset_jsonl(
        dataset_path,
        expected_file_sha256=payload.get("dataset_file_sha256"),
        expected_content_sha256=payload.get("content_sha256"),
        expected_sample_ids=sample_ids,
    )
    _validate_dataset_image_tree(
        images_dir,
        sample_ids=sample_ids,
        expected_hashes=image_hashes,
    )
    _validate_contact_sheet_tree(root, payload)


def _validate_phase_d_split_audit(phase_d: Mapping[str, Any]) -> None:
    split_audit = _required_mapping(phase_d, "split_audit")
    expected = {
        "scene_template_leaks": [],
        "answer_leaks": [],
        "visual_style_leaks": [],
        "error_mechanism_leaks": [],
        "ood_pair_mismatches": [],
        "ood_pair_count": 100,
        "preregistered_ood_factors": list(_FROZEN_CVA_V2_FACTORS),
        "ood_changed_factors": list(_FROZEN_CVA_V2_FACTORS),
    }
    if dict(split_audit) != expected or phase_d.get("split_audit_error") is not None:
        raise RuntimeError("Phase D split audit does not prove the frozen paired OOD contract")


def _validate_phase_d_visual_audit(phase_d: Mapping[str, Any]) -> None:
    visual = _required_mapping(phase_d, "visual_factor_realization_audit")
    _require_exact_fields(
        visual,
        frozenset(
            {
                "complete",
                "catalog",
                "observed_styles",
                "sample_counts",
                "applicability",
                "applicable_sample_counts",
                "applicability_violations",
                "applicable_coverage",
                "nonapplicable_baseline_contract",
            }
        ),
        name="Phase D visual-factor realization audit",
    )
    expected_catalog = list(FROZEN_CVA_V2_VISUAL_STYLES)
    expected_applicability = {
        style: list(families) for style, families in _FROZEN_CVA_V2_VISUAL_APPLICABILITY.items()
    }
    if (
        visual.get("complete") is not True
        or visual.get("catalog") != expected_catalog
        or visual.get("observed_styles") != expected_catalog
        or visual.get("sample_counts") != _FROZEN_CVA_V2_STYLE_COUNTS
        or visual.get("applicability") != expected_applicability
        or visual.get("applicable_sample_counts") != _FROZEN_CVA_V2_APPLICABLE_SAMPLE_COUNTS
        or visual.get("applicability_violations") != []
        or visual.get("applicable_coverage") is not True
        or visual.get("nonapplicable_baseline_contract") is not True
    ):
        raise RuntimeError("Phase D visual-factor realization audit is incomplete or drifted")


def _validate_phase_d_answer_balance(phase_d: Mapping[str, Any]) -> None:
    balance = _required_mapping(phase_d, "answer_balance")
    _require_exact_fields(
        balance,
        frozenset(
            {
                "complete",
                "groups",
                "iid_ood_exact_match",
                "numeric_exact_balance",
                "relation_multiclass_coverage",
                "violations",
            }
        ),
        name="Phase D answer-balance audit",
    )
    from compbias.eval.post_gpu_evidence import _validate_answer_balance

    try:
        _validate_answer_balance(balance)
    except ValueError as error:
        raise RuntimeError(
            "Phase D answer-balance audit differs from deterministic CVA-v2"
        ) from error
    if (
        balance.get("complete") is not True
        or balance.get("iid_ood_exact_match") is not True
        or balance.get("numeric_exact_balance") is not True
        or balance.get("relation_multiclass_coverage") is not True
        or balance.get("violations") != []
    ):
        raise RuntimeError("Phase D answer-balance audit is incomplete")
    groups = _required_mapping(balance, "groups")
    expected_groups = {
        f"{family}/{split}"
        for family in _FROZEN_CVA_V2_TASK_FAMILIES
        for split in _FROZEN_CVA_V2_SPLITS
    }
    if set(groups) != expected_groups:
        raise RuntimeError("Phase D answer-balance audit must cover all 25 family/split groups")
    semantic_frequencies: dict[str, dict[str, int]] = {}
    for name, group_value in groups.items():
        if not isinstance(group_value, Mapping):
            raise RuntimeError(f"Phase D answer-balance group {name} must be an object")
        _require_exact_fields(
            group_value,
            frozenset({"sample_count", "support", "frequencies"}),
            name=f"Phase D answer-balance group {name}",
        )
        family, _split = name.split("/", maxsplit=1)
        support = group_value.get("support")
        frequencies = group_value.get("frequencies")
        split = _split
        multiplier = (
            2
            if split == "ood_test"
            else (
                9 if family in {"digit_offset", "gauge_calibration", "bar_chart_aggregate"} else 8
            )
        )
        expected_sample_count = 10 * multiplier
        if (
            group_value.get("sample_count") != expected_sample_count
            or not isinstance(support, list)
            or not support
            or not isinstance(frequencies, list)
            or not frequencies
            or any(
                not isinstance(entry, Mapping)
                or set(entry) != {"answer", "count"}
                or isinstance(entry.get("count"), bool)
                or not isinstance(entry.get("count"), int)
                or entry["count"] < 1
                for entry in frequencies
            )
            or sum(entry["count"] for entry in frequencies) != expected_sample_count
            or len(support) != len(frequencies)
            or any(
                entry["answer"] != answer
                for answer, entry in zip(support, frequencies, strict=True)
            )
        ):
            raise RuntimeError(f"Phase D answer-balance group {name} is malformed")
        counts = sorted(entry["count"] for entry in frequencies)
        encoded_support = [
            json.dumps(answer, allow_nan=False, sort_keys=True, separators=(",", ":"))
            for answer in support
        ]
        if len(set(encoded_support)) != len(encoded_support):
            raise RuntimeError(f"Phase D answer-balance group {name} repeats an answer")
        semantic_frequencies[name] = {
            encoded: entry["count"] // multiplier
            for encoded, entry in zip(encoded_support, frequencies, strict=True)
        }
        if family == "relation_rule":
            frozen_pattern = (
                len(support) == 6
                and all(isinstance(answer, str) and answer for answer in support)
                and counts
                == sorted(
                    [
                        multiplier,
                        multiplier,
                        2 * multiplier,
                        2 * multiplier,
                        2 * multiplier,
                        2 * multiplier,
                    ]
                )
            )
        else:
            frozen_pattern = (
                len(support) == 10
                and all(
                    not isinstance(answer, bool)
                    and isinstance(answer, (int, float))
                    and math.isfinite(float(answer))
                    for answer in support
                )
                and counts == [multiplier] * 10
            )
        if not frozen_pattern:
            raise RuntimeError(
                f"Phase D answer-balance group {name} differs from the frozen 1820-row pattern"
            )
    for family in _FROZEN_CVA_V2_TASK_FAMILIES:
        if semantic_frequencies[f"{family}/iid_test"] != semantic_frequencies[f"{family}/ood_test"]:
            raise RuntimeError(f"Phase D answer-balance group {family} differs between IID and OOD")


def _validate_phase_d_ood_image_shift(phase_d: Mapping[str, Any]) -> None:
    image_shift = _required_mapping(phase_d, "ood_image_shift")
    _require_exact_fields(
        image_shift,
        frozenset({"complete", "checked_pair_count", "violations"}),
        name="Phase D OOD image-shift audit",
    )
    if (
        image_shift.get("complete") is not True
        or image_shift.get("checked_pair_count") != 100
        or image_shift.get("violations") != []
    ):
        raise RuntimeError(
            "Phase D OOD image-shift audit must prove 100 source/OOD hash-distinct pairs"
        )


def _validate_phase_d_style_semantic_joint_independence(
    phase_d: Mapping[str, Any],
) -> None:
    joint = _required_mapping(phase_d, "style_semantic_joint_independence")
    _require_exact_fields(
        joint,
        frozenset({"complete", "criterion", "groups", "violations"}),
        name="Phase D style/semantic joint-independence audit",
    )
    if (
        joint.get("complete") is not True
        or joint.get("criterion") != "fully_crossed_style_by_semantic_state"
        or joint.get("violations") != []
    ):
        raise RuntimeError("Phase D style/semantic joint-independence audit is incomplete")
    groups = _required_mapping(joint, "groups")
    iid_splits = tuple(split for split in _FROZEN_CVA_V2_SPLITS if split != "ood_test")
    expected_groups = {
        f"{family}/{split}" for family in _FROZEN_CVA_V2_TASK_FAMILIES for split in iid_splits
    }
    if set(groups) != expected_groups:
        raise RuntimeError(
            "Phase D style/semantic joint-independence audit must cover 20 IID groups"
        )
    for group_name, group_value in groups.items():
        if not isinstance(group_value, Mapping):
            raise RuntimeError(f"Phase D joint-independence group {group_name} must be an object")
        _require_exact_fields(
            group_value,
            frozenset(
                {
                    "semantic_state_count",
                    "expected_styles",
                    "fully_crossed_state_count",
                    "sample_count",
                    "style_counts",
                }
            ),
            name=f"Phase D joint-independence group {group_name}",
        )
        family, _split = group_name.split("/", maxsplit=1)
        applicable_styles = [
            style
            for style in FROZEN_CVA_V2_VISUAL_STYLES[:-1]
            if family in _FROZEN_CVA_V2_VISUAL_APPLICABILITY[style]
        ]
        expected_counts = dict.fromkeys(applicable_styles, 10)
        if (
            group_value.get("semantic_state_count") != 10
            or group_value.get("expected_styles") != applicable_styles
            or group_value.get("fully_crossed_state_count") != 10
            or group_value.get("sample_count") != 10 * len(applicable_styles)
            or group_value.get("style_counts") != expected_counts
        ):
            raise RuntimeError(
                f"Phase D joint-independence group {group_name} is not fully crossed"
            )


def _validate_phase_d_deterministic_replay(phase_d: Mapping[str, Any]) -> None:
    replay = _required_mapping(phase_d, "deterministic_replay")
    _require_exact_fields(
        replay,
        frozenset(
            {
                "complete",
                "generator_matches",
                "renderer_matches",
                "contact_sheets_match",
                "generator_mismatches",
                "renderer_mismatches",
                "contact_sheet_mismatches",
            }
        ),
        name="Phase D deterministic replay audit",
    )
    if (
        replay.get("complete") is not True
        or replay.get("generator_matches") is not True
        or replay.get("renderer_matches") is not True
        or replay.get("contact_sheets_match") is not True
        or replay.get("generator_mismatches") != []
        or replay.get("renderer_mismatches") != []
        or replay.get("contact_sheet_mismatches") != []
    ):
        raise RuntimeError("Phase D deterministic generator/renderer replay is incomplete")


def _validate_phase_d_automatic_gates(phase_d: Mapping[str, Any]) -> None:
    exact = {
        "sample_count": _FROZEN_CVA_V2_SAMPLE_COUNT,
        "rendered_image_count": _FROZEN_CVA_V2_SAMPLE_COUNT,
        "solver_passes": _FROZEN_CVA_V2_SAMPLE_COUNT,
        "solver_pass_rate": 1.0,
        "split_clean": True,
        "image_set_matches": True,
        "rendered_image_count_matches": True,
        "contact_sheet_sha256_matches": True,
        "manifest_sample_count_matches": True,
        "manifest_sample_ids_match": True,
        "manifest_content_sha256_matches": True,
        "manifest_config_sha256_matches": True,
        "manifest_dataset_file_sha256_matches": True,
        "manifest_image_sha256_matches": True,
        "manifest_self_sha256_matches": True,
        "preregistered_ood_factors_match_config": True,
        "visual_review_present": True,
        "human_reviewer_signoff": True,
        "human_review_binding_matches": True,
        "automatic_audit_clean": True,
        "phase_d_ready": True,
    }
    empty_lists = (
        "missing_images",
        "extra_images",
        "contact_sheet_hash_mismatches",
        "noncanonical_rows",
        "image_path_mismatches",
        "privacy_issues",
        "image_question_answer_collisions",
        "style_counterbalance_violations",
    )
    for key, expected in exact.items():
        if phase_d.get(key) != expected:
            raise RuntimeError(f"Phase D automated gate failed: {key}")
    for key in empty_lists:
        if phase_d.get(key) != []:
            raise RuntimeError(f"Phase D automated gate failed: {key}")
    roundtrip_total = _require_positive_int(
        phase_d.get("roundtrip_total"),
        name="Phase D roundtrip_total",
    )
    if (
        roundtrip_total != _FROZEN_CVA_V2_ROUNDTRIP_COUNT
        or phase_d.get("roundtrip_passes") != roundtrip_total
        or phase_d.get("error_solver_passes") != roundtrip_total
        or phase_d.get("roundtrip_pass_rate") != 1.0
        or phase_d.get("error_solver_pass_rate") != 1.0
    ):
        raise RuntimeError("Phase D error-catalog execution or round-trip gate failed")


def _validate_phase_d_human_review(
    phase_d: Mapping[str, Any],
) -> tuple[str, list[str]]:
    human = _required_mapping(phase_d, "human_review")
    _require_exact_fields(
        human,
        frozenset(
            {
                "signoff",
                "reviewer",
                "reviewer_type",
                "review_date",
                "review_result",
                "reviewed_image_count",
                "reviewed_sample_ids",
                "contact_sheets_reviewed",
                "binding_matches",
                "manifest_self_sha256",
                "integrity_scope",
            }
        ),
        name="Phase D human review",
    )
    reviewer = human.get("reviewer")
    reviewed = human.get("reviewed_sample_ids")
    if (
        human.get("signoff") is not True
        or human.get("reviewer_type") != "human"
        or human.get("review_result") != "pass"
        or human.get("binding_matches") is not True
        or not isinstance(reviewer, str)
        or _PUBLIC_REVIEWER.fullmatch(reviewer) is None
        or not isinstance(human.get("review_date"), str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", human["review_date"]) is None
        or not isinstance(reviewed, list)
        or any(not isinstance(sample_id, str) for sample_id in reviewed)
        or len(reviewed) < 200
        or len(set(reviewed)) != len(reviewed)
        or human.get("reviewed_image_count") != len(reviewed)
        or human.get("contact_sheets_reviewed") != _FROZEN_CVA_V2_CONTACT_SHEET_COUNT
        or human.get("integrity_scope")
        != "self-reported review record; no external signature verified"
    ):
        raise RuntimeError("Phase D requires a closed, bound human review of at least 200 images")
    return reviewer, reviewed


def _validate_phase_d_report(
    phase_d: Mapping[str, Any],
) -> tuple[str, list[str], str, str, str, str]:
    _require_exact_fields(phase_d, _PHASE_D_REPORT_FIELDS, name="Phase D audit report")
    if phase_d.get("audit_report_schema_version") != 2:
        raise RuntimeError("Phase D audit report must use schema version 2")
    _validate_phase_d_automatic_gates(phase_d)
    _validate_phase_d_split_audit(phase_d)
    _validate_phase_d_visual_audit(phase_d)
    _validate_phase_d_ood_image_shift(phase_d)
    _validate_phase_d_style_semantic_joint_independence(phase_d)
    _validate_phase_d_deterministic_replay(phase_d)
    _validate_phase_d_answer_balance(phase_d)
    reviewer, reviewed_ids = _validate_phase_d_human_review(phase_d)
    dataset = _required_mapping(phase_d, "dataset")
    _require_exact_fields(
        dataset,
        frozenset(
            {
                "manifest_path",
                "manifest_file_sha256",
                "manifest_self_sha256",
                "content_sha256",
                "image_set_sha256",
            }
        ),
        name="Phase D dataset evidence",
    )
    _validate_publishable_path(dataset.get("manifest_path"), name="Phase D manifest_path")
    manifest_file_hash = _validated_sha256(
        dataset.get("manifest_file_sha256"),
        name="Phase D dataset manifest file SHA-256",
    )
    manifest_self_hash = _validated_sha256(
        dataset.get("manifest_self_sha256"),
        name="Phase D dataset manifest self SHA-256",
    )
    content_hash = _validated_sha256(
        dataset.get("content_sha256"),
        name="Phase D dataset content SHA-256",
    )
    image_set_hash = _validated_sha256(
        dataset.get("image_set_sha256"),
        name="Phase D dataset image-set SHA-256",
    )
    if phase_d.get("evidence_manifest_sha256") != manifest_self_hash:
        raise RuntimeError("Phase D evidence manifest hash differs from its dataset binding")
    if phase_d.get("evidence_image_set_sha256") != image_set_hash:
        raise RuntimeError("Phase D evidence image-set hash differs from its dataset binding")
    return (
        reviewer,
        reviewed_ids,
        manifest_file_hash,
        manifest_self_hash,
        content_hash,
        image_set_hash,
    )


def _validate_digest_mapping(
    value: object,
    *,
    expected_keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError(f"{name} must cover the exact frozen key set")
    if any(
        not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
        for digest in value.values()
    ):
        raise RuntimeError(f"{name} must contain lowercase SHA-256 digests")
    return value


def _validate_cva_v2_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_publishable_path: str,
    manifest_self_hash: str,
    content_hash: str,
    image_set_hash: str,
    phase_d: Mapping[str, Any],
) -> list[str]:
    _require_exact_fields(payload, _DATASET_MANIFEST_FIELDS, name="CVA-v2 dataset manifest")
    if (
        payload.get("dataset_name") != "cva_v2"
        or payload.get("schema_version") != "2.0"
        or payload.get("sample_count") != _FROZEN_CVA_V2_SAMPLE_COUNT
        or payload.get("rendered_image_count") != _FROZEN_CVA_V2_SAMPLE_COUNT
        or payload.get("solver_checks") != _FROZEN_CVA_V2_SAMPLE_COUNT
        or payload.get("solver_pass_rate") != 1.0
        or payload.get("roundtrip_pass_rate") != 1.0
        or payload.get("preregistered_ood_factors") != list(_FROZEN_CVA_V2_FACTORS)
    ):
        raise RuntimeError("dataset manifest is not the frozen cva_v2 1820-sample contract")
    generator = payload.get("generator_config")
    if generator != _FROZEN_CVA_V2_GENERATOR_CONFIG:
        raise RuntimeError(
            "dataset manifest generator_config differs from the frozen cva_v2 contract"
        )
    if payload.get("config_sha256") != _canonical_manifest_sha256(_FROZEN_CVA_V2_GENERATOR_CONFIG):
        raise RuntimeError("dataset manifest config SHA-256 is invalid")
    if payload.get("render_config") != {
        "height": 256,
        "samples_per_contact_sheet": 25,
        "width": 256,
    }:
        raise RuntimeError("dataset manifest render_config differs from the frozen contract")
    if payload.get("content_sha256") != content_hash:
        raise RuntimeError("dataset manifest content hash differs from Phase D evidence")
    if payload.get("manifest_sha256") != manifest_self_hash:
        raise RuntimeError("dataset manifest self hash differs from Phase D evidence")
    _validated_sha256(payload.get("dataset_file_sha256"), name="dataset file SHA-256")
    _validate_publishable_path(payload.get("jsonl_path"), name="dataset jsonl_path")
    _validate_publishable_path(payload.get("images_dir"), name="dataset images_dir")
    sample_ids = payload.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != _FROZEN_CVA_V2_SAMPLE_COUNT
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != _FROZEN_CVA_V2_SAMPLE_COUNT
        or tuple(sample_ids) != _FROZEN_CVA_V2_SAMPLE_IDS
    ):
        raise RuntimeError("dataset manifest must list the exact 1820 frozen sample IDs")
    image_hashes = _validate_digest_mapping(
        payload.get("image_sha256"),
        expected_keys=set(sample_ids),
        name="dataset image SHA-256 manifest",
    )
    if _canonical_manifest_sha256(image_hashes) != image_set_hash:
        raise RuntimeError("dataset image-set SHA-256 differs from Phase D evidence")
    roundtrip_checks = _require_positive_int(
        payload.get("roundtrip_checks"),
        name="dataset manifest roundtrip_checks",
    )
    if roundtrip_checks != _FROZEN_CVA_V2_ROUNDTRIP_COUNT or roundtrip_checks != phase_d.get(
        "roundtrip_total"
    ):
        raise RuntimeError("dataset manifest roundtrip count differs from the Phase D audit")
    contact_sheets = payload.get("contact_sheets")
    expected_sheet_names = {
        f"cva_contact_sheet_{index:02d}.png"
        for index in range(1, _FROZEN_CVA_V2_CONTACT_SHEET_COUNT + 1)
    }
    if (
        not isinstance(contact_sheets, list)
        or len(contact_sheets) != _FROZEN_CVA_V2_CONTACT_SHEET_COUNT
        or any(
            not isinstance(path, str)
            or PurePosixPath(_validate_publishable_path(path, name="contact sheet")).name
            not in expected_sheet_names
            for path in contact_sheets
        )
        or {PurePosixPath(path).name for path in contact_sheets} != expected_sheet_names
    ):
        raise RuntimeError("dataset manifest must list the exact 73 frozen contact sheets")
    _validate_digest_mapping(
        payload.get("contact_sheet_sha256"),
        expected_keys=expected_sheet_names,
        name="contact-sheet SHA-256 manifest",
    )
    _validate_cva_v2_data_tree(
        payload,
        manifest_path=manifest_path,
        manifest_publishable_path=manifest_publishable_path,
    )
    return sample_ids


@dataclass(frozen=True, slots=True, init=False)
class VLMExecutionEvidence:
    """Artifact-backed gates required before a VLM plan may be emitted."""

    stage: str
    phase_d_reviewer: str
    phase_d_reviewed_images: int
    target_container_gpu_uuids: tuple[str, ...]
    parser_validity_rate: float | None
    model_snapshot: ModelSnapshotEvidence
    dataset_manifest: str
    dataset_manifest_publishable_path: str
    dataset_manifest_sha256: str
    dataset_manifest_self_sha256: str
    dataset_content_sha256: str
    dataset_image_set_sha256: str
    phase_d_audit_path: str
    phase_d_audit_sha256: str
    execution_audit_path: str
    execution_audit_sha256: str
    sft_checkpoint_sha256: str | None
    state_adapter_sha256: str | None
    h1_gate_passed: bool | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use load_execution_gate_evidence to create execution evidence")

    @classmethod
    def _from_loader(
        cls,
        *,
        stage: str,
        phase_d_reviewer: str,
        phase_d_reviewed_images: int,
        target_container_gpu_uuids: tuple[str, ...],
        parser_validity_rate: float | None,
        model_snapshot: ModelSnapshotEvidence,
        dataset_manifest: str,
        dataset_manifest_publishable_path: str,
        dataset_manifest_sha256: str,
        dataset_manifest_self_sha256: str,
        dataset_content_sha256: str,
        dataset_image_set_sha256: str,
        phase_d_audit_path: str,
        phase_d_audit_sha256: str,
        execution_audit_path: str,
        execution_audit_sha256: str,
        sft_checkpoint_sha256: str | None,
        state_adapter_sha256: str | None,
        h1_gate_passed: bool | None,
    ) -> VLMExecutionEvidence:
        instance = object.__new__(cls)
        values = locals()
        for field_name in (
            "stage",
            "phase_d_reviewer",
            "phase_d_reviewed_images",
            "target_container_gpu_uuids",
            "parser_validity_rate",
            "model_snapshot",
            "dataset_manifest",
            "dataset_manifest_publishable_path",
            "dataset_manifest_sha256",
            "dataset_manifest_self_sha256",
            "dataset_content_sha256",
            "dataset_image_set_sha256",
            "phase_d_audit_path",
            "phase_d_audit_sha256",
            "execution_audit_path",
            "execution_audit_sha256",
            "sft_checkpoint_sha256",
            "state_adapter_sha256",
            "h1_gate_passed",
        ):
            object.__setattr__(instance, field_name, values[field_name])
        instance._validate()
        return instance

    def _validate(self) -> None:
        if self.stage not in {"structured_sft", "joint_outcome_rl"}:
            raise ValueError("execution evidence stage is invalid")
        if (
            not isinstance(self.phase_d_reviewer, str)
            or _PUBLIC_REVIEWER.fullmatch(self.phase_d_reviewer) is None
        ):
            raise ValueError("Phase D reviewer must be a bounded public pseudonym")
        if (
            isinstance(self.phase_d_reviewed_images, bool)
            or not isinstance(self.phase_d_reviewed_images, int)
            or self.phase_d_reviewed_images < 200
        ):
            raise RuntimeError("Phase D human review must cover at least 200 images")
        devices = tuple(self.target_container_gpu_uuids)
        if not devices or any(
            not isinstance(device, str) or _GPU_UUID.fullmatch(device) is None for device in devices
        ):
            raise ValueError("target-container GPU UUIDs must come from nvidia-smi")
        object.__setattr__(self, "target_container_gpu_uuids", devices)
        if self.stage == "structured_sft":
            if any(
                value is not None
                for value in (
                    self.parser_validity_rate,
                    self.sft_checkpoint_sha256,
                    self.state_adapter_sha256,
                    self.h1_gate_passed,
                )
            ):
                raise RuntimeError("SFT evidence may not contain post-SFT RL gates")
        else:
            object.__setattr__(
                self,
                "parser_validity_rate",
                _validated_parse_rate(self.parser_validity_rate),
            )
            object.__setattr__(
                self,
                "sft_checkpoint_sha256",
                _validated_sha256(
                    self.sft_checkpoint_sha256,
                    name="SFT checkpoint SHA-256",
                ),
            )
            object.__setattr__(
                self,
                "state_adapter_sha256",
                _validated_sha256(
                    self.state_adapter_sha256,
                    name="state adapter SHA-256",
                ),
            )
            if self.h1_gate_passed is not True:
                raise RuntimeError("RL evidence requires a passed fixed-reasoner H1 gate")
        if not isinstance(self.model_snapshot, ModelSnapshotEvidence):
            raise TypeError("model_snapshot must be verified from its file manifest")
        if (
            not isinstance(self.dataset_manifest, str)
            or not Path(self.dataset_manifest).is_absolute()
        ):
            raise ValueError("dataset manifest path must be absolute")
        _validate_publishable_path(
            self.dataset_manifest_publishable_path,
            name="dataset manifest publishable path",
        )
        object.__setattr__(
            self,
            "dataset_manifest_sha256",
            _validated_sha256(
                self.dataset_manifest_sha256,
                name="dataset manifest SHA-256",
            ),
        )
        object.__setattr__(
            self,
            "dataset_manifest_self_sha256",
            _validated_sha256(
                self.dataset_manifest_self_sha256,
                name="dataset manifest self SHA-256",
            ),
        )
        object.__setattr__(
            self,
            "dataset_content_sha256",
            _validated_sha256(
                self.dataset_content_sha256,
                name="dataset content SHA-256",
            ),
        )
        object.__setattr__(
            self,
            "dataset_image_set_sha256",
            _validated_sha256(
                self.dataset_image_set_sha256,
                name="dataset image-set SHA-256",
            ),
        )
        for field_name in ("phase_d_audit_path", "execution_audit_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{field_name} must be an absolute path")
        object.__setattr__(
            self,
            "phase_d_audit_sha256",
            _validated_sha256(self.phase_d_audit_sha256, name="Phase D audit SHA-256"),
        )
        object.__setattr__(
            self,
            "execution_audit_sha256",
            _validated_sha256(
                self.execution_audit_sha256,
                name="execution audit SHA-256",
            ),
        )


def load_execution_gate_evidence(
    phase_d_audit: str | Path,
    execution_audit: str | Path,
    *,
    stage: str,
    phase_d_sha256: str,
    execution_audit_sha256: str,
) -> VLMExecutionEvidence:
    """Load and verify prior-gate artifacts without importing a trainer."""

    if stage not in {"structured_sft", "joint_outcome_rl"}:
        raise ValueError("stage must be structured_sft or joint_outcome_rl")

    phase_d_hash = _validated_sha256(phase_d_sha256, name="Phase D audit SHA-256")
    execution_hash = _validated_sha256(
        execution_audit_sha256,
        name="execution audit SHA-256",
    )
    phase_d_path = Path(phase_d_audit).expanduser().resolve(strict=True)
    execution_path = Path(execution_audit).expanduser().resolve(strict=True)
    phase_d = _read_hashed_json(phase_d_path, phase_d_hash, name="Phase D audit")
    (
        reviewer,
        reviewed_sample_ids,
        dataset_manifest_hash,
        dataset_manifest_self_hash,
        dataset_content_hash,
        dataset_image_set_hash,
    ) = _validate_phase_d_report(phase_d)
    reviewed_images = len(reviewed_sample_ids)

    execution = _read_hashed_json(
        execution_path,
        execution_hash,
        name="VLM execution audit",
    )
    if execution.get("schema_version") != 2:
        raise RuntimeError("VLM execution audit must use schema version 2")
    if execution.get("stage") != stage:
        raise RuntimeError("VLM execution audit stage differs from the requested stage")
    if execution.get("training_invoked") is not False:
        raise RuntimeError("VLM execution audit must precede all training")
    if execution.get("large_gpu_started") is not False:
        raise RuntimeError("VLM execution audit cannot describe an already-started run")

    expected_pins = {
        "model_name": DEFAULT_MODEL_NAME,
        "model_revision": PINNED_MODEL_REVISION,
        "transformers_revision": PINNED_TRANSFORMERS_REVISION,
        "verl_revision": PINNED_VERL_REVISION,
        "vllm_revision": PINNED_VLLM_REVISION,
    }
    pins = _required_mapping(execution, "pins")
    if dict(pins) != expected_pins:
        raise RuntimeError("VLM execution audit does not match the frozen stack")

    smoke = _required_mapping(execution, "target_container_smoke")
    if smoke.get("passed") is not True:
        raise RuntimeError("target-container smoke test has not passed")
    if smoke.get("dockerfile_sha256") != PINNED_VERL_DOCKERFILE_SHA256:
        raise RuntimeError("target-container smoke used an unaudited Dockerfile")
    smoke_devices = smoke.get("gpu_uuids")
    if not isinstance(smoke_devices, list):
        raise RuntimeError("target-container smoke must record GPU UUIDs")
    image_digest = smoke.get("container_image_digest")
    if (
        not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
    ):
        raise RuntimeError("target-container smoke must record a pinned image digest")
    if smoke.get("runtime_packages") != {
        "torch": "2.11.0",
        "transformers": PINNED_TRANSFORMERS_REVISION,
        "vllm": PINNED_VLLM_REVISION,
    }:
        raise RuntimeError("target-container smoke runtime packages differ from the frozen stack")
    if smoke.get("verl_revision") != PINNED_VERL_REVISION:
        raise RuntimeError("target-container smoke used a different veRL revision")
    if smoke.get("verl_worktree_clean") is not True:
        raise RuntimeError("target-container veRL worktree must be clean")

    parser_audit: Mapping[str, Any] | None = None
    state_audit: Mapping[str, Any] | None = None
    h1_audit: Mapping[str, Any] | None = None
    parser_rate: float | None = None
    sft_checkpoint_hash: str | None = None
    adapter_hash: str | None = None
    if stage == "structured_sft":
        forbidden_post_sft = {
            "sft_checkpoint",
            "parser_audit",
            "state_injection_audit",
            "fixed_reasoner_h1_audit",
            "verl_api_audit",
        }
        present = sorted(forbidden_post_sft.intersection(execution))
        if present:
            raise RuntimeError(f"SFT preflight may not depend on post-SFT gates: {present}")
    else:
        checkpoint = _required_mapping(execution, "sft_checkpoint")
        _checkpoint_path = _require_file_sha256(
            checkpoint.get("path"),
            checkpoint.get("sha256"),
            name="SFT checkpoint",
        )
        sft_checkpoint_hash = _validated_sha256(
            checkpoint.get("sha256"),
            name="SFT checkpoint SHA-256",
        )
        parser_audit = _required_mapping(execution, "parser_audit")
        if parser_audit.get("measured_on_model") is not True:
            raise RuntimeError("parser validity must be measured on the SFT checkpoint")
        parser_rate = _validated_parse_rate(parser_audit.get("validity_rate"))

        state_audit = _required_mapping(execution, "state_injection_audit")
        if state_audit.get("passed") is not True or state_audit.get("image_hidden") is not True:
            raise RuntimeError("image-hidden state-injection audit has not passed")
        if state_audit.get("isolation_mode") != "separate_text_only_worker":
            raise RuntimeError("state injection must use an isolated text-only worker")
        _adapter_path = _require_file_sha256(
            state_audit.get("adapter_path"),
            state_audit.get("adapter_sha256"),
            name="state-injection adapter",
        )
        adapter_hash = _validated_sha256(
            state_audit.get("adapter_sha256"),
            name="state-injection adapter SHA-256",
        )
        if state_audit.get("reviewed_adapter_sha256") != adapter_hash:
            raise RuntimeError("state-injection adapter hash has not been reviewed")

        h1_audit = _required_mapping(execution, "fixed_reasoner_h1_audit")
        if h1_audit.get("passed") is not True:
            raise RuntimeError("fixed-reasoner H1 audit has not passed")
        if h1_audit.get("sign_prediction_above_chance") is not True:
            raise RuntimeError("fixed-reasoner H1 sign prediction is not above chance")
        coupling_count = h1_audit.get("measurable_coupling_task_count")
        if (
            isinstance(coupling_count, bool)
            or not isinstance(coupling_count, int)
            or coupling_count < 1
        ):
            raise RuntimeError("fixed-reasoner H1 needs measurable coupling on at least one task")

        verl_audit = _required_mapping(execution, "verl_api_audit")
        if verl_audit.get("passed") is not True:
            raise RuntimeError("veRL API audit has not passed")
        if verl_audit.get("revision") != PINNED_VERL_REVISION:
            raise RuntimeError("veRL API audit names a different revision")
        audited_keys = verl_audit.get("audited_leaf_keys")
        if audited_keys != list(AUDITED_GRPO_LEAF_KEYS):
            raise RuntimeError("veRL API audit does not match the strict 16-key whitelist")

    snapshot = _required_mapping(execution, "model_snapshot")
    if snapshot.get("revision") != PINNED_MODEL_REVISION:
        raise RuntimeError("model snapshot evidence names a different revision")
    if snapshot.get("local_files_only") is not True:
        raise RuntimeError("model snapshot evidence must prohibit network fallback")
    model_snapshot = verify_model_snapshot(
        snapshot.get("path"),
        snapshot.get("manifest_path"),
        expected_manifest_sha256=snapshot.get("manifest_sha256"),
    )
    snapshot_hash = model_snapshot.manifest_sha256
    execution_dataset = _required_mapping(execution, "dataset")
    _require_exact_fields(
        execution_dataset,
        frozenset(
            {
                "manifest_path",
                "manifest_file_sha256",
                "manifest_self_sha256",
                "content_sha256",
                "image_set_sha256",
            }
        ),
        name="VLM execution dataset binding",
    )
    if execution_dataset.get("manifest_file_sha256") != dataset_manifest_hash:
        raise RuntimeError("execution evidence is bound to a different dataset manifest")
    if execution_dataset.get("manifest_self_sha256") != dataset_manifest_self_hash:
        raise RuntimeError("execution evidence is bound to a different dataset self hash")
    if execution_dataset.get("content_sha256") != dataset_content_hash:
        raise RuntimeError("execution evidence is bound to different dataset content")
    if execution_dataset.get("image_set_sha256") != dataset_image_set_hash:
        raise RuntimeError("execution evidence is bound to a different dataset image set")
    manifest_value = execution_dataset.get("manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value or "\x00" in manifest_value:
        raise ValueError("execution dataset manifest_path must be a non-empty path")
    dataset_manifest = _lexical_absolute(Path(manifest_value))
    if dataset_manifest.is_symlink():
        raise RuntimeError("dataset manifest may not be a symlink")
    dataset_payload = _read_hashed_json(
        dataset_manifest,
        dataset_manifest_hash,
        name="dataset manifest",
    )
    unsigned_dataset_manifest = {
        key: value for key, value in dataset_payload.items() if key != "manifest_sha256"
    }
    observed_self_hash = _canonical_manifest_sha256(unsigned_dataset_manifest)
    if dataset_payload.get("manifest_sha256") != observed_self_hash:
        raise RuntimeError("dataset manifest self SHA-256 is invalid")
    if observed_self_hash != dataset_manifest_self_hash:
        raise RuntimeError("dataset manifest self SHA-256 differs from Phase D evidence")
    phase_dataset = _required_mapping(phase_d, "dataset")
    manifest_publishable_path = _validate_publishable_path(
        phase_dataset.get("manifest_path"),
        name="Phase D manifest_path",
    )
    manifest_sample_ids = _validate_cva_v2_manifest(
        dataset_payload,
        manifest_path=dataset_manifest,
        manifest_publishable_path=manifest_publishable_path,
        manifest_self_hash=dataset_manifest_self_hash,
        content_hash=dataset_content_hash,
        image_set_hash=dataset_image_set_hash,
        phase_d=phase_d,
    )
    if not set(reviewed_sample_ids).issubset(set(manifest_sample_ids)):
        raise RuntimeError("Phase D human review contains IDs outside the frozen manifest")
    human_review = _required_mapping(phase_d, "human_review")
    if human_review.get("manifest_self_sha256") != dataset_manifest_self_hash:
        raise RuntimeError("Phase D human review is not bound to the frozen manifest")

    bound_audits: list[tuple[str, Mapping[str, Any]]] = [("target-container smoke", smoke)]
    if stage == "joint_outcome_rl":
        assert parser_audit is not None and state_audit is not None and h1_audit is not None
        bound_audits.extend(
            (
                ("parser audit", parser_audit),
                ("state-injection audit", state_audit),
                ("fixed-reasoner H1 audit", h1_audit),
            )
        )
    for audit_name, audit_mapping in bound_audits:
        if audit_mapping.get("model_snapshot_manifest_sha256") != snapshot_hash:
            raise RuntimeError(f"{audit_name} is bound to a different model snapshot")
        if audit_mapping.get("dataset_manifest_file_sha256") != dataset_manifest_hash:
            raise RuntimeError(f"{audit_name} is bound to a different dataset manifest")
    if stage == "joint_outcome_rl":
        assert sft_checkpoint_hash is not None
        for audit_name, audit_mapping in (
            ("parser audit", parser_audit),
            ("state-injection audit", state_audit),
            ("fixed-reasoner H1 audit", h1_audit),
        ):
            assert audit_mapping is not None
            if audit_mapping.get("sft_checkpoint_sha256") != sft_checkpoint_hash:
                raise RuntimeError(f"{audit_name} is bound to a different SFT checkpoint")
        if checkpoint.get("model_snapshot_manifest_sha256") != snapshot_hash:
            raise RuntimeError("SFT checkpoint is bound to a different base model snapshot")
        if checkpoint.get("dataset_manifest_file_sha256") != dataset_manifest_hash:
            raise RuntimeError("SFT checkpoint is bound to a different dataset manifest")
    external_authorization = _required_mapping(execution, "external_authorization")
    if external_authorization.get("status") != "not_granted":
        raise RuntimeError("this metadata-only boundary cannot grant execution authorization")

    return VLMExecutionEvidence._from_loader(
        stage=stage,
        phase_d_reviewer=reviewer,
        phase_d_reviewed_images=reviewed_images,
        target_container_gpu_uuids=tuple(smoke_devices),
        parser_validity_rate=parser_rate,
        model_snapshot=model_snapshot,
        dataset_manifest=str(dataset_manifest),
        dataset_manifest_publishable_path=manifest_publishable_path,
        dataset_manifest_sha256=dataset_manifest_hash,
        dataset_manifest_self_sha256=dataset_manifest_self_hash,
        dataset_content_sha256=dataset_content_hash,
        dataset_image_set_sha256=dataset_image_set_hash,
        phase_d_audit_path=str(phase_d_path),
        phase_d_audit_sha256=phase_d_hash,
        execution_audit_path=str(execution_path),
        execution_audit_sha256=execution_hash,
        sft_checkpoint_sha256=sft_checkpoint_hash,
        state_adapter_sha256=adapter_hash,
        h1_gate_passed=True if stage == "joint_outcome_rl" else None,
    )


def require_machine_verified_cuda(
    *,
    detected_gpu_uuids: tuple[str, ...] | list[str],
    smoke_gpu_uuids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Cross-check local nvidia-smi UUIDs against the target-container audit."""

    detected = _validated_devices(bool(detected_gpu_uuids), detected_gpu_uuids)
    smoke = _validated_devices(bool(smoke_gpu_uuids), smoke_gpu_uuids)
    if not detected:
        raise RuntimeError("no CUDA GPU was detected by local nvidia-smi")
    if any(_GPU_UUID.fullmatch(device) is None for device in (*detected, *smoke)):
        raise ValueError("CUDA evidence must use GPU UUIDs reported by nvidia-smi")
    selected = tuple(device for device in detected if device in set(smoke))
    if not selected:
        raise RuntimeError("detected CUDA GPUs do not match the target-container smoke audit")
    return selected


def _revalidate_execution_dataset(evidence: VLMExecutionEvidence) -> None:
    """Re-hash the bound JSONL and exact image tree immediately before planning."""

    manifest_path = Path(evidence.dataset_manifest)
    payload = _read_hashed_json(
        manifest_path,
        evidence.dataset_manifest_sha256,
        name="dataset manifest",
    )
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    observed_self_hash = _canonical_manifest_sha256(unsigned)
    if (
        payload.get("manifest_sha256") != evidence.dataset_manifest_self_sha256
        or observed_self_hash != evidence.dataset_manifest_self_sha256
        or payload.get("content_sha256") != evidence.dataset_content_sha256
        or _canonical_manifest_sha256(payload.get("image_sha256"))
        != evidence.dataset_image_set_sha256
    ):
        raise RuntimeError("dataset manifest bindings changed before plan construction")
    _validate_cva_v2_data_tree(
        payload,
        manifest_path=manifest_path,
        manifest_publishable_path=evidence.dataset_manifest_publishable_path,
    )


def _validated_preflight(
    config: VLMPreflightConfig,
    *,
    evidence: VLMExecutionEvidence,
) -> tuple[VLMPreflightReport, VLMExecutionEvidence]:
    if not isinstance(evidence, VLMExecutionEvidence):
        raise TypeError("evidence must be artifact-backed VLMExecutionEvidence")
    reloaded = load_execution_gate_evidence(
        evidence.phase_d_audit_path,
        evidence.execution_audit_path,
        stage=evidence.stage,
        phase_d_sha256=evidence.phase_d_audit_sha256,
        execution_audit_sha256=evidence.execution_audit_sha256,
    )
    if reloaded != evidence:
        raise RuntimeError("execution evidence changed after it was loaded")
    revalidate_model_snapshot(reloaded.model_snapshot)
    devices = probe_local_cuda_devices()
    selected_devices = require_machine_verified_cuda(
        detected_gpu_uuids=devices,
        smoke_gpu_uuids=reloaded.target_container_gpu_uuids,
    )
    _revalidate_execution_dataset(reloaded)
    require_frozen_qwen_stack(config)
    return (
        validate_preflight(
            config,
            cuda_available=True,
            gpu_devices=selected_devices,
            require_verl_api_audit=evidence.stage == "joint_outcome_rl",
        ),
        reloaded,
    )


@dataclass(frozen=True, slots=True, init=False)
class VerlExecutionPlan:
    """Immutable description of a validated but deliberately unstarted run."""

    stage: str
    preflight: VLMPreflightReport
    requirements: Mapping[str, Any]
    audited_verl_keys: tuple[str, ...] = ()
    verl_config: Mapping[str, Any] | None = None
    reward_contract: Mapping[str, Any] | None = None
    artifact_type: str = field(init=False, default="execution_plan")
    artifact_visibility: str = field(init=False, default="private_operator_only")
    execution_status: str = field(init=False, default="not_started")
    large_gpu_started: bool = field(init=False, default=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use a stage-specific execution-plan builder")

    @classmethod
    def _from_builder(
        cls,
        *,
        stage: str,
        preflight: VLMPreflightReport,
        requirements: Mapping[str, Any],
        audited_verl_keys: tuple[str, ...] = (),
        verl_config: Mapping[str, Any] | None = None,
        reward_contract: Mapping[str, Any] | None = None,
    ) -> VerlExecutionPlan:
        instance = object.__new__(cls)
        object.__setattr__(instance, "stage", stage)
        object.__setattr__(instance, "preflight", preflight)
        object.__setattr__(instance, "requirements", requirements)
        object.__setattr__(instance, "audited_verl_keys", audited_verl_keys)
        object.__setattr__(instance, "verl_config", verl_config)
        object.__setattr__(instance, "reward_contract", reward_contract)
        object.__setattr__(instance, "artifact_type", "execution_plan")
        object.__setattr__(instance, "artifact_visibility", "private_operator_only")
        object.__setattr__(instance, "execution_status", "not_started")
        object.__setattr__(instance, "large_gpu_started", False)
        instance._validate_and_freeze()
        return instance

    def _validate_and_freeze(self) -> None:
        if self.stage not in {"structured_sft", "joint_outcome_rl"}:
            raise ValueError("stage must be structured_sft or joint_outcome_rl")
        if not isinstance(self.preflight, VLMPreflightReport):
            raise TypeError("preflight must be a VLMPreflightReport")
        expected_preflight = {
            "model_name": DEFAULT_MODEL_NAME,
            "model_revision": PINNED_MODEL_REVISION,
            "transformers_revision": PINNED_TRANSFORMERS_REVISION,
            "verl_revision": PINNED_VERL_REVISION,
            "vllm_revision": PINNED_VLLM_REVISION,
        }
        for field_name, expected in expected_preflight.items():
            if getattr(self.preflight, field_name) != expected:
                raise RuntimeError("execution plan preflight differs from the frozen stack")
        if not self.preflight.gpu_devices or any(
            _GPU_UUID.fullmatch(device) is None for device in self.preflight.gpu_devices
        ):
            raise RuntimeError("execution plan preflight lacks machine-shaped GPU UUID evidence")
        if not isinstance(self.requirements, Mapping):
            raise TypeError("requirements must be a mapping")
        if not isinstance(self.audited_verl_keys, tuple) or any(
            not isinstance(key, str) or not key for key in self.audited_verl_keys
        ):
            raise TypeError("audited_verl_keys must be a tuple of non-empty strings")
        if self.verl_config is not None and not isinstance(self.verl_config, Mapping):
            raise TypeError("verl_config must be a mapping or None")
        if self.reward_contract is not None and not isinstance(self.reward_contract, Mapping):
            raise TypeError("reward_contract must be a mapping or None")
        if self.requirements.get("execution_permitted") is not False:
            raise RuntimeError("execution plans from this boundary must remain blocked")
        if self.requirements.get("external_authorization_status") != "not_granted":
            raise RuntimeError("execution plans cannot self-grant external authorization")
        if self.requirements.get("previous_phase_a_c_artifacts_verified") is not False:
            raise RuntimeError("Phase A-C artifact linkage remains unverified")
        if self.requirements.get("snapshot_authenticity") != "self_consistency_only":
            raise RuntimeError("snapshot authenticity must retain its self-consistency boundary")
        if self.requirements.get("trusted_upstream_snapshot_inventory_status") != "missing":
            raise RuntimeError(
                "trusted upstream snapshot inventory must remain an explicit blocker"
            )
        if self.requirements.get("artifact_hash_assurance") != "integrity_only_not_authentication":
            raise RuntimeError("artifact hashes cannot be promoted to authentication evidence")
        if self.requirements.get("model_snapshot_revalidation_required_at_execution") is not True:
            raise RuntimeError("snapshot revalidation must remain mandatory at execution time")
        if self.requirements.get("local_files_only") is not True:
            raise RuntimeError("execution plans must prohibit model-network fallback")
        if self.requirements.get("trust_remote_code") is not False:
            raise RuntimeError("execution plans must prohibit remote/custom model code")
        if self.requirements.get("use_safetensors") is not True:
            raise RuntimeError("execution plans must require safetensors weights")
        if self.requirements.get("network_access") != "disabled":
            raise RuntimeError("execution plans must keep network access disabled")
        if self.requirements.get("hardened_container_evidence_status") != "pending":
            raise RuntimeError("hardened descendant container evidence remains pending")
        if self.requirements.get("offline_wheelhouse_evidence_status") != "pending":
            raise RuntimeError("offline wheelhouse evidence remains pending")
        if self.requirements.get("container_sbom_status") != "pending":
            raise RuntimeError("container SBOM remains pending")
        if self.requirements.get("container_vulnerability_audit_status") != "pending":
            raise RuntimeError("container vulnerability-policy audit remains pending")
        if self.requirements.get("executor_gpu_uuid_binding_status") != "pending":
            raise RuntimeError("runtime executor GPU UUID binding remains pending")
        if self.stage == "structured_sft":
            if self.preflight.verl_api_audited is not False:
                raise RuntimeError("SFT preflight must not claim an unaudited veRL API surface")
            if (
                self.audited_verl_keys
                or self.verl_config is not None
                or self.reward_contract is not None
            ):
                raise RuntimeError("SFT plans must defer all unaudited veRL configuration")
            if self.requirements.get("verl_configuration_status") != "deferred":
                raise RuntimeError("SFT veRL configuration must remain deferred")
        else:
            if self.preflight.verl_api_audited is not True:
                raise RuntimeError("GRPO preflight lacks the veRL API audit")
            if self.audited_verl_keys != AUDITED_GRPO_LEAF_KEYS:
                raise RuntimeError("GRPO plan keys differ from the strict audited whitelist")
            if self.verl_config is None or _mapping_leaf_keys(self.verl_config) != set(
                AUDITED_GRPO_LEAF_KEYS
            ):
                raise RuntimeError("GRPO veRL mapping differs from the strict audited whitelist")
            if self.reward_contract != {
                "outcome_only": True,
                "perception_reward_weight": 0.0,
                "process_reward_weight": 0.0,
            }:
                raise RuntimeError("GRPO reward contract must remain outcome-only")

        object.__setattr__(self, "requirements", _freeze(self.requirements))
        object.__setattr__(self, "audited_verl_keys", tuple(self.audited_verl_keys))
        if self.verl_config is not None:
            object.__setattr__(self, "verl_config", _freeze(self.verl_config))
        if self.reward_contract is not None:
            object.__setattr__(self, "reward_contract", _freeze(self.reward_contract))

    def to_mapping(self) -> dict[str, Any]:
        """Return a defensive, serialization-ready representation."""

        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "artifact_visibility": self.artifact_visibility,
            "stage": self.stage,
            "execution_status": self.execution_status,
            "large_gpu_started": self.large_gpu_started,
            "preflight": self.preflight.to_mapping(),
            "requirements": _thaw(self.requirements),
            "audited_verl_keys": list(self.audited_verl_keys),
        }
        if self.verl_config is not None:
            payload["verl"] = _thaw(self.verl_config)
        if self.reward_contract is not None:
            payload["reward_contract"] = _thaw(self.reward_contract)
        return payload


def build_sft_execution_plan(
    config: VLMPreflightConfig,
    *,
    evidence: VLMExecutionEvidence,
    dataset_manifest: str,
    minimum_parse_rate: float = 0.98,
) -> VerlExecutionPlan:
    """Validate the SFT boundary without inventing unaudited veRL SFT keys."""

    if evidence.stage != "structured_sft":
        raise RuntimeError("SFT plan requires structured_sft evidence")

    manifest = _validated_manifest(dataset_manifest)
    parse_rate = _validated_parse_rate(minimum_parse_rate)
    report, evidence = _validated_preflight(
        config,
        evidence=evidence,
    )
    if Path(manifest).expanduser().resolve(strict=True) != Path(evidence.dataset_manifest):
        raise RuntimeError("SFT dataset manifest differs from the artifact-backed Phase D dataset")
    return VerlExecutionPlan._from_builder(
        stage="structured_sft",
        preflight=report,
        requirements={
            "dataset_manifest": manifest,
            "minimum_parse_rate": parse_rate,
            "structured_output_required": True,
            "verl_configuration_status": "deferred",
            "defer_reason": "pinned veRL SFT keys have not been audited",
            "execution_permitted": False,
            "phase_d_audit_sha256": evidence.phase_d_audit_sha256,
            "execution_audit_sha256": evidence.execution_audit_sha256,
            "model_snapshot_manifest_sha256": evidence.model_snapshot.manifest_sha256,
            "model_snapshot_revalidation_required_at_execution": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "network_access": "disabled",
            "dataset_manifest_sha256": evidence.dataset_manifest_sha256,
            "dataset_manifest_self_sha256": evidence.dataset_manifest_self_sha256,
            "dataset_content_sha256": evidence.dataset_content_sha256,
            "dataset_image_set_sha256": evidence.dataset_image_set_sha256,
            "snapshot_authenticity": evidence.model_snapshot.authenticity,
            "trusted_upstream_snapshot_inventory_status": "missing",
            "artifact_hash_assurance": "integrity_only_not_authentication",
            "hardened_container_evidence_status": "pending",
            "offline_wheelhouse_evidence_status": "pending",
            "container_sbom_status": "pending",
            "container_vulnerability_audit_status": "pending",
            "executor_gpu_uuid_binding_status": "pending",
            "external_authorization_status": "not_granted",
            "previous_phase_a_c_artifacts_verified": False,
        },
    )


def build_grpo_execution_plan(
    config: VLMPreflightConfig,
    *,
    evidence: VLMExecutionEvidence,
    learning_rate: float = 1.0e-6,
    mini_batch_size: int = 16,
    rollout_samples: int = 8,
    project_name: str = "compbias",
    experiment_name: str = "qwen25_vl_grpo",
) -> VerlExecutionPlan:
    """Build a frozen outcome-only GRPO plan from the audited key whitelist."""

    if evidence.stage != "joint_outcome_rl":
        raise RuntimeError("GRPO plan requires post-SFT joint_outcome_rl evidence")

    rate = _validated_positive_float(learning_rate, name="learning_rate")
    batch_size = _validated_positive_int(mini_batch_size, name="mini_batch_size")
    samples = _validated_positive_int(rollout_samples, name="rollout_samples")
    project = _validated_run_name(project_name, name="project_name")
    experiment = _validated_run_name(experiment_name, name="experiment_name")
    report, evidence = _validated_preflight(
        config,
        evidence=evidence,
    )
    raw_config = _build_verl_grpo_config(
        config,
        model_snapshot=evidence.model_snapshot,
        learning_rate=rate,
        mini_batch_size=batch_size,
        rollout_samples=samples,
        project_name=project,
        experiment_name=experiment,
    )
    verl_config = raw_config
    actual_keys = _mapping_leaf_keys(verl_config)
    expected_keys = set(AUDITED_GRPO_LEAF_KEYS)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            "veRL configuration differs from the audited key whitelist: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return VerlExecutionPlan._from_builder(
        stage="joint_outcome_rl",
        preflight=report,
        requirements={
            "image_hidden_intervention_required": True,
            "target_hardware_smoke_required": True,
            "phase_d_audit_sha256": evidence.phase_d_audit_sha256,
            "execution_audit_sha256": evidence.execution_audit_sha256,
            "model_snapshot_manifest_sha256": evidence.model_snapshot.manifest_sha256,
            "model_snapshot_revalidation_required_at_execution": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "network_access": "disabled",
            "dataset_manifest": evidence.dataset_manifest,
            "dataset_manifest_sha256": evidence.dataset_manifest_sha256,
            "dataset_manifest_self_sha256": evidence.dataset_manifest_self_sha256,
            "dataset_content_sha256": evidence.dataset_content_sha256,
            "dataset_image_set_sha256": evidence.dataset_image_set_sha256,
            "snapshot_authenticity": evidence.model_snapshot.authenticity,
            "trusted_upstream_snapshot_inventory_status": "missing",
            "artifact_hash_assurance": "integrity_only_not_authentication",
            "hardened_container_evidence_status": "pending",
            "offline_wheelhouse_evidence_status": "pending",
            "container_sbom_status": "pending",
            "container_vulnerability_audit_status": "pending",
            "executor_gpu_uuid_binding_status": "pending",
            "external_authorization_status": "not_granted",
            "previous_phase_a_c_artifacts_verified": False,
            "execution_permitted": False,
        },
        audited_verl_keys=AUDITED_GRPO_LEAF_KEYS,
        verl_config=verl_config,
        reward_contract={
            "outcome_only": True,
            "perception_reward_weight": 0.0,
            "process_reward_weight": 0.0,
        },
    )


__all__ = [
    "AUDITED_GRPO_LEAF_KEYS",
    "VLMExecutionEvidence",
    "VerlExecutionPlan",
    "build_grpo_execution_plan",
    "build_sft_execution_plan",
    "load_execution_gate_evidence",
    "require_machine_verified_cuda",
]
