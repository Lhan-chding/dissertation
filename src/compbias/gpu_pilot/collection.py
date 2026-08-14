"""Natural structured-evidence collection from the local Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping
from compbias.models.structured_parser import ParseStatus

from .config import (
    ACTIVE_CALIBRATION_RECORDS_NAME,
    ACTIVE_PILOT_DATA_CONFIG,
    ACTIVE_PILOT_OUTPUT_SLUG,
    load_pilot_paths,
)
from .preflight import model_snapshot_sha256
from .qwen_smoke import decode_qwen_once, load_local_qwen
from .safe_io import atomic_write_json_text, prepare_output_path
from .structured_generation import generate_with_format_retries, numeric_answer_matches
from .taxonomy import PERCEPTION_ERROR_TYPES, natural_error_type


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} in {path} is not a JSON object")
            records.append(value)
    return tuple(records)


def _error_type(record: Mapping[str, object], parsed: object) -> str:
    return natural_error_type(record, parsed)


def collect_split(
    dataset_dir: Path,
    model_path: Path,
    output_path: Path,
    *,
    split: str,
    data_config_path: Path,
    output_root: Path | None = None,
    replace_incomplete: bool = False,
) -> dict[str, object]:
    approved_root = output_path.parent if output_root is None else output_root
    prepare_output_path(approved_root, output_path, allow_existing=replace_incomplete)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest_path = dataset_dir / "manifest.json"
    manifest_before = load_strict_json_mapping(manifest_path, label="pilot dataset manifest")
    dataset_manifest_hash_before = _sha256(manifest_path)
    dataset_images_hash = manifest_before.get("images_sha256")
    if not isinstance(dataset_images_hash, str) or len(dataset_images_hash) != 64:
        raise ValueError("pilot dataset manifest lacks images_sha256")
    from .execution_gate import _validate_canonical_dataset

    _validate_canonical_dataset(manifest_path, data_config_path, output_path.parent)
    model_hash_before = model_snapshot_sha256(model_path)
    source = tuple(
        record
        for record in _read_jsonl(dataset_dir / "records.jsonl")
        if record.get("split") == split
    )
    if not source:
        raise ValueError(f"no records found for split {split}")
    model, processor = load_local_qwen(model_path)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    counts: dict[str, int] = {}
    correct = 0
    parsed_count = 0
    try:
        with staging_path.open("w", encoding="utf-8") as stream:
            for index, record in enumerate(source):
                image_path = dataset_dir / str(record["image"])
                operation = record.get("operation")
                question = record.get("question")
                sample_id = record.get("sample_id")
                if not isinstance(operation, str):
                    raise ValueError("dataset operation must be a string")
                if not isinstance(question, str):
                    raise ValueError("dataset question must be a string")
                if not isinstance(sample_id, str):
                    raise ValueError("dataset sample_id must be a string")
                expected_values = record.get("values")
                if not isinstance(expected_values, list) or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in expected_values
                ):
                    raise ValueError("dataset values must be an integer list")
                generation = generate_with_format_retries(
                    partial(decode_qwen_once, model, processor, image_path),
                    question=question,
                    operation=operation,
                    sample_id=sample_id,
                    expected_value_count=len(expected_values),
                    max_format_retries=0,
                )
                parsed = generation.parsed
                error_type = _error_type(record, parsed)
                counts[error_type] = counts.get(error_type, 0) + 1
                answer_correct = parsed.status is ParseStatus.OK and numeric_answer_matches(
                    parsed.answer, record["answer"]
                )
                correct += int(answer_correct)
                parsed_count += int(parsed.status is ParseStatus.OK)
                result = {
                    **record,
                    "rollout_id": f"{split}-rollout-{index:06d}",
                    "raw_text": generation.raw_text,
                    "parsed": parsed.to_mapping(),
                    "format_attempts": list(generation.attempts),
                    "format_retries": len(generation.attempts) - 1,
                    "reward": int(answer_correct),
                    "error_type": error_type,
                }
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        total = len(source)
        model_hash_after = model_snapshot_sha256(model_path)
        manifest_after = load_strict_json_mapping(manifest_path, label="pilot dataset manifest")
        _validate_canonical_dataset(manifest_path, data_config_path, output_path.parent)
        if (
            model_hash_after != model_hash_before
            or _sha256(manifest_path) != dataset_manifest_hash_before
            or manifest_after.get("images_sha256") != dataset_images_hash
        ):
            raise RuntimeError("model or dataset changed during natural trajectory collection")
        if replace_incomplete:
            os.replace(staging_path, output_path)
        else:
            os.link(staging_path, output_path)
            staging_path.unlink()
    finally:
        staging_path.unlink(missing_ok=True)
    visual_errors = sum(counts.get(error_type, 0) for error_type in PERCEPTION_ERROR_TYPES)
    return {
        "schema_version": 1,
        "split": split,
        "records": total,
        "answer_accuracy": correct / total,
        "parse_rate": parsed_count / total,
        "natural_perception_error_rate": visual_errors / total,
        "error_counts": counts,
        "output": str(output_path),
        "model_snapshot_sha256": model_hash_before,
        "dataset_manifest_sha256": dataset_manifest_hash_before,
        "dataset_images_sha256": dataset_images_hash,
    }


def calibration_gate(report: Mapping[str, object]) -> tuple[str, ...]:
    failures: list[str] = []
    accuracy = float(report["answer_accuracy"])
    parse_rate = float(report["parse_rate"])
    error_rate = float(report["natural_perception_error_rate"])
    counts = report["error_counts"]
    assert isinstance(counts, Mapping)
    supported = sum(
        int(counts.get(name, 0)) >= 10
        for name in ("visual_error", "compensated_visual_error", "reasoning_error")
    )
    if not 0.30 <= accuracy <= 0.75:
        failures.append("base_answer_accuracy_outside_30_75_percent")
    if not 0.15 <= error_rate <= 0.50:
        failures.append("natural_perception_error_outside_15_50_percent")
    if parse_rate < 0.95:
        failures.append("evidence_parse_rate_below_95_percent")
    if supported < 3:
        failures.append("fewer_than_three_supported_natural_error_families")
    return tuple(failures)


def main(argv: Sequence[str] | None = None, *, calibration: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--split", default="calibration" if calibration else "pilot_train")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("BLOCKED: natural collection requires explicit --execute on the reviewed GPU server")
        return 2
    try:
        paths = load_pilot_paths(args.paths)
        dataset = paths.data / "generated" / ACTIVE_PILOT_OUTPUT_SLUG
        target_name = (
            ACTIVE_CALIBRATION_RECORDS_NAME
            if calibration and args.split == "calibration"
            else f"{args.split}_records.jsonl"
        )
        target = paths.trajectories / "natural" / target_name
        report_path = target.with_suffix(".summary.json")
        if calibration and any(
            path.exists() or path.is_symlink() for path in (target, report_path)
        ):
            raise FileExistsError(
                "completed or ambiguous v0.3 calibration evidence already exists; rerun refused"
            )
        report = collect_split(
            dataset,
            paths.model_path,
            target,
            split=args.split,
            data_config_path=paths.project_root / ACTIVE_PILOT_DATA_CONFIG,
            output_root=paths.trajectories,
            replace_incomplete=not calibration and target.exists(),
        )
        failures = calibration_gate(report) if calibration else ()
        report = {**report, "gate_failures": list(failures), "gate_passed": not failures}
        atomic_write_json_text(
            paths.trajectories,
            report_path,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if not failures else 3
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3


def calibration_main(argv: Sequence[str] | None = None) -> int:
    return main(argv, calibration=True)
