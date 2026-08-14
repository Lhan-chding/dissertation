"""Strict evidence gate that must pass before any GPU pilot training import."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping
from compbias.io.yaml_config import load_yaml_mapping
from compbias.models.structured_parser import parse_trajectory

from .chart_data import generate_dataset
from .config import ACTIVE_PILOT_DATASET_ID, PilotPaths, load_pilot_data_config
from .preflight import model_snapshot_sha256
from .structured_generation import numeric_answer_matches, validate_pilot_trajectory

_PREFLIGHT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "ready",
        "large_gpu_started",
        "hardware",
        "free_disk_gib",
        "model_path",
        "model_snapshot_sha256",
        "storage",
    }
)
_HARDWARE_KEYS = frozenset(
    {
        "cuda_available",
        "device_name",
        "total_vram_gib",
        "bf16_supported",
        "torch_version",
        "torch_cuda_runtime",
    }
)
_SMOKE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "training_invoked",
        "model_path",
        "model_snapshot_sha256",
        "expected_answer",
        "raw_response",
        "parsed",
        "format_attempts",
        "format_retries",
        "format_passed",
        "smoke_passed",
        "answer_correct",
        "latency_seconds",
        "peak_memory_gib",
    }
)
_PARSED_KEYS = frozenset(
    {
        "status",
        "sample_id",
        "raw_text",
        "perceived_scene",
        "reasoning_action",
        "answer",
        "error_code",
    }
)
_ATTEMPT_KEYS = frozenset({"attempt_index", "raw_text", "status", "error_code"})
_CALIBRATION_KEYS = frozenset(
    {
        "schema_version",
        "split",
        "records",
        "answer_accuracy",
        "parse_rate",
        "natural_perception_error_rate",
        "error_counts",
        "output",
        "gate_failures",
        "gate_passed",
        "model_snapshot_sha256",
        "dataset_manifest_sha256",
        "dataset_images_sha256",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "record_count",
        "split_counts",
        "counterfactual_pairs",
        "natural_audit_ids",
        "records_path",
        "records_sha256",
        "counterfactual_path",
        "counterfactual_sha256",
        "images_generated",
        "images_sha256",
    }
)
_EXPECTED_SPLITS = {
    "calibration": 200,
    "smoke_train": 600,
    "pilot_train": 1200,
    "dev": 200,
    "iid_test": 300,
    "mechanism_ood": 300,
}
_DATASET_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "sample_id",
        "split",
        "chart_type",
        "operation",
        "values",
        "question",
        "answer",
        "image",
        "mechanism",
    }
)
_NATURAL_EXTRA_KEYS = frozenset(
    {
        "rollout_id",
        "raw_text",
        "parsed",
        "format_attempts",
        "format_retries",
        "reward",
        "error_type",
    }
)
_COUNTERFACTUAL_KEYS = frozenset(
    {
        "pair_id",
        "source_sample_id",
        "counterfactual_sample_id",
        "image",
        "values",
        "question",
        "answer",
        "operation",
    }
)
_MAX_JSONL_BYTES = 64 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000


def _exact_mapping(value: object, keys: frozenset[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be a string-keyed mapping")
    if set(value) != keys:
        raise RuntimeError(f"{label} does not match its closed schema")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise RuntimeError(f"{label} is outside the supported range") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be a finite number")
    return number


def _regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"missing {label}: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(raw: str, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise RuntimeError(f"{label} contains a non-finite number")
        return value

    def reject_constant(token: str) -> object:
        raise RuntimeError(f"{label} contains non-standard number {token}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeError) as error:
        raise RuntimeError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be a string-keyed JSON object")
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > _MAX_JSON_DEPTH or nodes > _MAX_JSON_NODES:
            raise RuntimeError(f"{label} exceeds the permitted depth or complexity")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
    return value


def _read_strict_jsonl(
    path: Path,
    *,
    label: str,
    expected_rows: int,
) -> tuple[dict[str, object], ...]:
    _regular_file(path, label=label)
    if path.stat().st_size > _MAX_JSONL_BYTES:
        raise RuntimeError(f"{label} exceeds the {_MAX_JSONL_BYTES}-byte limit")
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
                raise RuntimeError(f"{label} line {line_number} exceeds the line limit")
            if line_number > expected_rows:
                raise RuntimeError(f"{label} contains more than {expected_rows} rows")
            records.append(_strict_json_object(line, label=f"{label} line {line_number}"))
    if len(records) != expected_rows:
        raise RuntimeError(f"{label} must contain exactly {expected_rows} rows")
    return tuple(records)


def resolve_project_file(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError(f"{label} must be relative to the project root")
    resolved_root = root.resolve()
    unresolved = resolved_root / relative
    current = resolved_root
    for component in relative.parts:
        current = current / component
        try:
            if current.is_symlink():
                raise RuntimeError(f"{label} path contains a symlink: {current}")
        except OSError as error:
            raise RuntimeError(f"cannot inspect {label} path: {current}") from error
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes the project root") from error
    return candidate


def _validate_preflight(
    report: Mapping[str, object],
    paths: PilotPaths,
    model_hash: str,
) -> None:
    _exact_mapping(report, _PREFLIGHT_KEYS, label="preflight report")
    if (
        report["schema_version"] != 1
        or report["artifact_type"] != "compbias_gpu_pilot_preflight"
        or report["ready"] is not True
        or report["large_gpu_started"] is not False
        or report["model_path"] != str(paths.model_path)
        or report["model_snapshot_sha256"] != model_hash
    ):
        raise RuntimeError("preflight report is not an approved ready snapshot")
    hardware = _exact_mapping(report["hardware"], _HARDWARE_KEYS, label="preflight hardware")
    if (
        hardware["cuda_available"] is not True
        or hardware["bf16_supported"] is not True
        or hardware["torch_version"] != "2.8.0+cu128"
        or hardware["torch_cuda_runtime"] != "12.8"
        or _finite_number(hardware["total_vram_gib"], label="preflight VRAM") < 45.0
    ):
        raise RuntimeError("preflight hardware no longer matches the approved pilot runtime")
    storage = _exact_mapping(
        report["storage"],
        frozenset(paths.to_mapping()),
        label="preflight storage",
    )
    if dict(storage) != paths.to_mapping():
        raise RuntimeError("preflight storage does not match the active paths configuration")


def _validate_smoke(
    report: Mapping[str, object],
    paths: PilotPaths,
    model_hash: str,
) -> None:
    _exact_mapping(report, _SMOKE_KEYS, label="smoke report")
    if (
        report["schema_version"] != 1
        or report["artifact_type"] != "qwen25vl3b_offline_smoke"
        or report["training_invoked"] is not False
        or report["model_path"] != str(paths.model_path)
        or report["model_snapshot_sha256"] != model_hash
        or report["expected_answer"] != 4
        or report["format_passed"] is not True
        or report["answer_correct"] is not True
        or report["smoke_passed"] is not True
    ):
        raise RuntimeError("smoke report is not a successful known-answer inference")
    raw = report["raw_response"]
    if not isinstance(raw, str):
        raise RuntimeError("smoke raw_response must be a string")
    parsed = validate_pilot_trajectory(
        parse_trajectory(raw, sample_id="smoke-000001"),
        operation="max_minus_min",
        expected_value_count=3,
    )
    parsed_report = _exact_mapping(report["parsed"], _PARSED_KEYS, label="smoke parsed result")
    if parsed.to_mapping() != dict(parsed_report) or not numeric_answer_matches(parsed.answer, 4):
        raise RuntimeError("smoke parsed result does not replay to the known answer")
    attempts = report["format_attempts"]
    retries = report["format_retries"]
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= 3
        or isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries != len(attempts) - 1
    ):
        raise RuntimeError("smoke format attempt accounting is invalid")
    for index, attempt in enumerate(attempts):
        mapped = _exact_mapping(attempt, _ATTEMPT_KEYS, label="smoke format attempt")
        if mapped["attempt_index"] != index or not isinstance(mapped["raw_text"], str):
            raise RuntimeError("smoke format attempts are not ordered raw responses")
    if attempts[-1]["raw_text"] != raw or attempts[-1]["status"] != "ok":
        raise RuntimeError("smoke final attempt does not match the successful parse")
    _finite_number(report["latency_seconds"], label="smoke latency")
    _finite_number(report["peak_memory_gib"], label="smoke peak memory")


def _validate_calibration(
    report: Mapping[str, object],
    paths: PilotPaths,
    derived: Mapping[str, object],
    *,
    model_hash: str,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> None:
    _exact_mapping(report, _CALIBRATION_KEYS, label="calibration report")
    expected_output = paths.trajectories / "natural" / "calibration_records.jsonl"
    if (
        report["schema_version"] != 1
        or report["split"] != "calibration"
        or report["records"] != 200
        or report["gate_passed"] is not True
        or report["gate_failures"] != []
        or report["output"] != str(expected_output)
        or report["model_snapshot_sha256"] != model_hash
        or report["dataset_manifest_sha256"] != _sha256(manifest_path)
        or report["dataset_images_sha256"] != manifest["images_sha256"]
    ):
        raise RuntimeError("calibration report is not the approved 200-record gate")
    accuracy = _finite_number(report["answer_accuracy"], label="calibration accuracy")
    parse_rate = _finite_number(report["parse_rate"], label="calibration parse rate")
    error_rate = _finite_number(
        report["natural_perception_error_rate"],
        label="calibration natural perception error rate",
    )
    if not 0.30 <= accuracy <= 0.75 or parse_rate < 0.95 or not 0.15 <= error_rate <= 0.50:
        raise RuntimeError("calibration metrics do not satisfy the registered gates")
    counts = report["error_counts"]
    if (
        not isinstance(counts, Mapping)
        or any(
            not isinstance(name, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in counts.items()
        )
        or sum(counts.values()) != 200
        or sum(
            int(counts.get(name, 0)) >= 10
            for name in ("visual_error", "compensated_visual_error", "reasoning_error")
        )
        < 3
    ):
        raise RuntimeError("calibration error-family support is invalid")
    for key in (
        "records",
        "answer_accuracy",
        "parse_rate",
        "natural_perception_error_rate",
        "error_counts",
    ):
        if report[key] != derived[key]:
            raise RuntimeError(f"calibration {key} does not replay from the trajectory")
    _regular_file(expected_output, label="calibration trajectory")


def _validate_pilot_a_summary(
    report: Mapping[str, object],
    paths: PilotPaths,
    *,
    model_hash: str,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> None:
    _exact_mapping(report, _CALIBRATION_KEYS, label="Pilot A natural-record summary")
    expected_output = paths.trajectories / "natural" / "pilot_train_records.jsonl"
    if (
        report["schema_version"] != 1
        or report["split"] != "pilot_train"
        or report["records"] != 1_200
        or report["gate_passed"] is not True
        or report["gate_failures"] != []
        or report["output"] != str(expected_output)
        or report["model_snapshot_sha256"] != model_hash
        or report["dataset_manifest_sha256"] != _sha256(manifest_path)
        or report["dataset_images_sha256"] != manifest["images_sha256"]
    ):
        raise RuntimeError("Pilot A natural records are not bound to this model and dataset")


def _validate_manifest(report: Mapping[str, object]) -> None:
    _exact_mapping(report, _MANIFEST_KEYS, label="pilot dataset manifest")
    if (
        report["schema_version"] != 1
        or report["dataset_id"] != ACTIVE_PILOT_DATASET_ID
        or report["record_count"] != 2_800
        or report["split_counts"] != _EXPECTED_SPLITS
        or report["counterfactual_pairs"] != 150
        or report["images_generated"] != 2_950
        or report["records_path"] != "records.jsonl"
        or report["counterfactual_path"] != "counterfactual_pairs.jsonl"
    ):
        raise RuntimeError("pilot dataset manifest does not match the registered contract")
    audit_ids = report["natural_audit_ids"]
    if not isinstance(audit_ids, list) or len(audit_ids) != 150 or len(set(audit_ids)) != 150:
        raise RuntimeError("pilot dataset natural audit IDs are invalid")
    for field in ("records_sha256", "counterfactual_sha256", "images_sha256"):
        value = report[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"pilot dataset {field} is invalid")


def _expected_answer(values: list[int], operation: str) -> int:
    if operation == "sum":
        return values[0] + values[1]
    if operation == "difference":
        return values[0] - values[1]
    if operation == "max_minus_min":
        return max(values) - min(values)
    raise RuntimeError(f"unsupported dataset operation: {operation!r}")


def _expected_question(operation: str) -> str:
    return {
        "sum": "What is the sum of the first two values?",
        "difference": "What is the first value minus the second value?",
        "max_minus_min": "What is the maximum value minus the minimum value?",
    }[operation]


def _validate_dataset_record(
    record: Mapping[str, object],
    *,
    split: str,
    index: int,
    global_index: int,
    dataset_root: Path,
) -> None:
    _exact_mapping(record, _DATASET_RECORD_KEYS, label="pilot dataset record")
    sample_id = f"{split}-{index:06d}"
    values = record["values"]
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool) or not 2 <= value <= 18
            for value in values
        )
    ):
        raise RuntimeError(f"pilot dataset values are invalid for {sample_id}")
    operation = record["operation"]
    if not isinstance(operation, str):
        raise RuntimeError(f"pilot dataset operation is invalid for {sample_id}")
    expected_image = f"images/{sample_id}.png"
    if (
        record["schema_version"] != 1
        or record["dataset_id"] != ACTIVE_PILOT_DATASET_ID
        or record["sample_id"] != sample_id
        or record["split"] != split
        or record["chart_type"] != ("grouped_bar", "line")[global_index % 2]
        or operation != ("difference", "sum", "max_minus_min")[global_index % 3]
        or record["question"] != _expected_question(operation)
        or record["answer"] != _expected_answer(values, operation)
        or record["image"] != expected_image
        or record["mechanism"] != ("shifted_style" if split == "mechanism_ood" else "iid")
    ):
        raise RuntimeError(f"pilot dataset record contract failed for {sample_id}")
    _regular_file(dataset_root / expected_image, label=f"pilot image {sample_id}")


def _image_bundle_sha256(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(root / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_dataset_bundle(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    dataset_root = manifest_path.parent
    records_path = dataset_root / "records.jsonl"
    counterfactual_path = dataset_root / "counterfactual_pairs.jsonl"
    _regular_file(records_path, label="pilot records")
    _regular_file(counterfactual_path, label="pilot counterfactual pairs")
    if _sha256(records_path) != manifest["records_sha256"]:
        raise RuntimeError("pilot records hash does not match the manifest")
    if _sha256(counterfactual_path) != manifest["counterfactual_sha256"]:
        raise RuntimeError("pilot counterfactual hash does not match the manifest")
    records = _read_strict_jsonl(records_path, label="pilot records", expected_rows=2_800)
    lookup: dict[str, Mapping[str, object]] = {}
    cursor = 0
    image_paths: list[str] = []
    for split, count in _EXPECTED_SPLITS.items():
        for index in range(count):
            record = records[cursor]
            global_index = cursor
            cursor += 1
            _validate_dataset_record(
                record,
                split=split,
                index=index,
                global_index=global_index,
                dataset_root=dataset_root,
            )
            sample_id = str(record["sample_id"])
            if sample_id in lookup:
                raise RuntimeError(f"duplicate pilot sample_id: {sample_id}")
            lookup[sample_id] = record
            image_paths.append(str(record["image"]))
    counterfactuals = _read_strict_jsonl(
        counterfactual_path,
        label="pilot counterfactual pairs",
        expected_rows=150,
    )
    iid = [record for record in records if record["split"] == "iid_test"]
    for index, pair in enumerate(counterfactuals):
        _exact_mapping(pair, _COUNTERFACTUAL_KEYS, label="pilot counterfactual pair")
        source = iid[index]
        source_values = source["values"]
        assert isinstance(source_values, list)
        values = [source_values[0] + 3, *source_values[1:]]
        operation = str(source["operation"])
        counterfactual_id = f"counterfactual-{index:06d}"
        relative = f"counterfactual/{counterfactual_id}.png"
        if dict(pair) != {
            "pair_id": f"pair-{index:06d}",
            "source_sample_id": source["sample_id"],
            "counterfactual_sample_id": counterfactual_id,
            "image": relative,
            "values": values,
            "question": source["question"],
            "answer": _expected_answer(values, operation),
            "operation": operation,
        }:
            raise RuntimeError(f"counterfactual pair {index} does not replay")
        _regular_file(dataset_root / relative, label=f"counterfactual image {index}")
        image_paths.append(relative)
    if manifest["natural_audit_ids"] != [f"calibration-{index:06d}" for index in range(150)]:
        raise RuntimeError("pilot natural audit IDs do not match the frozen calibration prefix")
    if _image_bundle_sha256(dataset_root, image_paths) != manifest["images_sha256"]:
        raise RuntimeError("pilot image bundle hash does not match the manifest")
    actual_images = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*.png")
        if path.is_file() and not path.is_symlink()
    }
    if actual_images != set(image_paths):
        raise RuntimeError("pilot image tree does not exactly match the registered image set")
    return lookup


def _regular_tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise RuntimeError(f"dataset tree contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"dataset tree contains a non-regular file: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


def _validate_canonical_dataset(
    manifest_path: Path,
    data_config_path: Path,
    cache_root: Path,
) -> None:
    """Regenerate the committed dataset and require an exact byte-for-byte match."""

    config = load_pilot_data_config(data_config_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-replay-", dir=cache_root) as temporary:
        replay_root = Path(temporary) / config.output_slug
        generate_dataset(config, replay_root)
        expected = _regular_tree_hashes(replay_root)
    actual = _regular_tree_hashes(manifest_path.parent)
    if actual != expected:
        raise RuntimeError(
            "pilot dataset bytes do not replay from the committed seed, config, and renderer"
        )


def _validate_registered_data_config(path: Path) -> None:
    config = load_pilot_data_config(path)
    if (
        config.seed != 20260814
        or config.dataset_id != ACTIVE_PILOT_DATASET_ID
        or config.image_size != (512, 384)
        or config.chart_types != ("grouped_bar", "line")
        or config.operations != ("difference", "sum", "max_minus_min")
        or dict(config.split_counts) != _EXPECTED_SPLITS
        or config.counterfactual_pairs != 150
        or config.natural_audit != 150
    ):
        raise RuntimeError("pilot data config does not equal the registered v0.2 design")


def _validate_model_config(path: Path, paths: PilotPaths) -> None:
    config = load_yaml_mapping(path, label="GPU pilot model configuration")
    expected = {
        "schema_version": 1,
        "model_id": "Qwen2.5-VL-3B-Instruct",
        "path": str(paths.model_path),
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bf16",
        "device": "cuda:0",
        "max_new_tokens": 512,
    }
    if config != expected:
        raise RuntimeError("model config does not equal the active registered offline model")


def _derived_error_type(record: Mapping[str, object], parsed: object) -> str:
    if parsed.status.value != "ok":  # type: ignore[attr-defined]
        return "parse_failure"
    perceived = parsed.perceived_scene  # type: ignore[attr-defined]
    perceived_values = perceived.get("values") if isinstance(perceived, Mapping) else None
    perception_correct = perceived_values == tuple(record["values"])  # type: ignore[arg-type]
    answer_correct = numeric_answer_matches(parsed.answer, record["answer"])  # type: ignore[attr-defined]
    if perception_correct and answer_correct:
        return "none"
    if not perception_correct and answer_correct:
        return "compensated_visual_error"
    if not perception_correct:
        return "visual_error"
    return "reasoning_error"


def _validate_natural_records(
    path: Path,
    *,
    split: str,
    dataset_records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    expected_count = _EXPECTED_SPLITS[split]
    records = _read_strict_jsonl(
        path,
        label=f"{split} natural records",
        expected_rows=expected_count,
    )
    counts: Counter[str] = Counter()
    correct = 0
    parsed_count = 0
    for index, record in enumerate(records):
        _exact_mapping(
            record,
            _DATASET_RECORD_KEYS | _NATURAL_EXTRA_KEYS,
            label=f"{split} natural record",
        )
        sample_id = f"{split}-{index:06d}"
        source = dataset_records.get(sample_id)
        if source is None or any(record[key] != value for key, value in source.items()):
            raise RuntimeError(f"natural record source fields do not match {sample_id}")
        raw = record["raw_text"]
        if not isinstance(raw, str):
            raise RuntimeError(f"natural raw_text must be a string for {sample_id}")
        values = source["values"]
        assert isinstance(values, list)
        parsed = validate_pilot_trajectory(
            parse_trajectory(raw, sample_id=sample_id),
            operation=str(source["operation"]),
            expected_value_count=len(values),
        )
        parsed_report = _exact_mapping(
            record["parsed"],
            _PARSED_KEYS,
            label=f"natural parsed result for {sample_id}",
        )
        if parsed.to_mapping() != dict(parsed_report):
            raise RuntimeError(f"natural parsed result does not replay for {sample_id}")
        attempts = record["format_attempts"]
        if record["format_retries"] != 0 or not isinstance(attempts, list) or len(attempts) != 1:
            raise RuntimeError(f"natural response was resampled for {sample_id}")
        attempt = _exact_mapping(
            attempts[0],
            _ATTEMPT_KEYS,
            label=f"natural format attempt for {sample_id}",
        )
        if (
            attempt["attempt_index"] != 0
            or attempt["raw_text"] != raw
            or attempt["status"] != parsed.status.value
            or attempt["error_code"] != parsed.error_code
        ):
            raise RuntimeError(f"natural attempt evidence is inconsistent for {sample_id}")
        error_type = _derived_error_type(source, parsed)
        answer_correct = parsed.status.value == "ok" and numeric_answer_matches(
            parsed.answer, source["answer"]
        )
        if (
            record["rollout_id"] != f"{split}-rollout-{index:06d}"
            or record["error_type"] != error_type
            or record["reward"] != int(answer_correct)
        ):
            raise RuntimeError(f"natural derived fields are inconsistent for {sample_id}")
        counts[error_type] += 1
        correct += int(answer_correct)
        parsed_count += int(parsed.status.value == "ok")
    visual = counts["visual_error"] + counts["compensated_visual_error"]
    return {
        "records": expected_count,
        "answer_accuracy": correct / expected_count,
        "parse_rate": parsed_count / expected_count,
        "natural_perception_error_rate": visual / expected_count,
        "error_counts": dict(counts),
    }


def validate_execution_evidence(
    config: Mapping[str, object],
    paths: PilotPaths,
    *,
    stage_config_path: Path,
    paths_config_path: Path,
) -> dict[str, str]:
    """Validate and hash the exact local artifacts that authorize one training run."""

    project_root = paths.project_root
    preflight_path = Path(
        str(config.get("validated_preflight_path", paths.outputs / "preflight" / "report.json"))
    ).resolve()
    smoke_path = Path(
        str(config.get("validated_smoke_path", paths.outputs / "smoke" / "smoke_report.json"))
    ).resolve()
    for label, path in (("preflight", preflight_path), ("smoke", smoke_path)):
        try:
            path.relative_to(paths.outputs)
        except ValueError as error:
            raise RuntimeError(f"{label} evidence must remain inside outputs") from error
    evidence_paths = {
        "preflight": preflight_path,
        "smoke": smoke_path,
        "calibration": paths.trajectories / "natural" / "calibration_records.summary.json",
        "dataset_manifest": resolve_project_file(
            project_root,
            config.get("dataset_manifest"),
            label="dataset_manifest",
        ),
        "stage_config": stage_config_path.resolve(),
        "paths_config": paths_config_path.resolve(),
        "model_config": resolve_project_file(
            project_root,
            config.get("model_config"),
            label="model_config",
        ),
        "data_config": resolve_project_file(
            project_root,
            config.get("data_config"),
            label="data_config",
        ),
    }
    for label, path in evidence_paths.items():
        _regular_file(path, label=label)

    preflight = load_strict_json_mapping(evidence_paths["preflight"], label="preflight report")
    smoke = load_strict_json_mapping(evidence_paths["smoke"], label="smoke report")
    calibration = load_strict_json_mapping(
        evidence_paths["calibration"],
        label="calibration report",
    )
    manifest = load_strict_json_mapping(
        evidence_paths["dataset_manifest"],
        label="pilot dataset manifest",
    )
    model_hash = model_snapshot_sha256(paths.model_path)
    _validate_preflight(preflight, paths, model_hash)
    _validate_smoke(smoke, paths, model_hash)
    _validate_manifest(manifest)
    _validate_registered_data_config(evidence_paths["data_config"])
    _validate_model_config(evidence_paths["model_config"], paths)
    _validate_canonical_dataset(
        evidence_paths["dataset_manifest"],
        evidence_paths["data_config"],
        paths.cache,
    )
    dataset_records = _validate_dataset_bundle(evidence_paths["dataset_manifest"], manifest)
    calibration_trajectory = paths.trajectories / "natural" / "calibration_records.jsonl"
    calibration_derived = _validate_natural_records(
        calibration_trajectory,
        split="calibration",
        dataset_records=dataset_records,
    )
    _validate_calibration(
        calibration,
        paths,
        calibration_derived,
        model_hash=model_hash,
        manifest_path=evidence_paths["dataset_manifest"],
        manifest=manifest,
    )
    evidence_paths = {
        **evidence_paths,
        "dataset_records": evidence_paths["dataset_manifest"].parent / "records.jsonl",
        "dataset_counterfactuals": (
            evidence_paths["dataset_manifest"].parent / "counterfactual_pairs.jsonl"
        ),
        "calibration_records": calibration_trajectory,
    }

    if config.get("stage") == "pilot_a":
        natural_records = resolve_project_file(
            project_root,
            config.get("natural_records"),
            label="natural_records",
        )
        _regular_file(natural_records, label="Pilot A natural records")
        _validate_natural_records(
            natural_records,
            split="pilot_train",
            dataset_records=dataset_records,
        )
        summary_path = natural_records.with_suffix(".summary.json")
        _regular_file(summary_path, label="Pilot A natural-record summary")
        summary = load_strict_json_mapping(summary_path, label="Pilot A natural-record summary")
        _validate_pilot_a_summary(
            summary,
            paths,
            model_hash=model_hash,
            manifest_path=evidence_paths["dataset_manifest"],
            manifest=manifest,
        )
        evidence_paths = {
            **evidence_paths,
            "natural_records": natural_records,
            "natural_records_summary": summary_path,
        }
    return {
        **{label: _sha256(path) for label, path in evidence_paths.items()},
        "model_snapshot": model_hash,
    }
