#!/usr/bin/env python3
"""Run the one-seed, budget-matched Study-B LoRA pilot on one offline 4090."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v4.qwen.model_loader import MODEL_PATH  # noqa: E402
from compensability_v5.qwen.study_b_runtime import (  # noqa: E402
    ARMS,
    MODEL_SNAPSHOT_SHA256,
    PILOT_SEED,
    QwenStudyBBackend,
    StudyBError,
    evaluation_rows_from_study_a,
    require_offline_environment,
    run_study_b,
    sha256_file,
    validate_evaluation_rows,
    validate_support_package,
    verify_runtime_package_lock,
)

CONFIG = ROOT / "configs/v5/budget_matched_lora.yaml"
PACKAGE_LOCK = ROOT / "configs/v5/server_package_lock.yaml"
OUTPUT_ROOT = ROOT / "artifacts/v5/study_b"
ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_STARTS_V5_STUDY_B_ON_ONE_4090"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--support-package", type=Path)
    parser.add_argument("--support-sha256")
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--parent-manifest-sha256")
    parser.add_argument("--child-manifest", type=Path)
    parser.add_argument("--child-manifest-sha256")
    parser.add_argument("--frozen-scenes", type=Path)
    parser.add_argument("--frozen-scenes-sha256")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--evaluation-sha256")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--config-sha256")
    parser.add_argument("--package-lock", type=Path, default=PACKAGE_LOCK)
    parser.add_argument("--package-lock-sha256")
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--model-sha256")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser


def _required_sha(value: str, label: str) -> str:
    if SHA256.fullmatch(value) is None:
        raise StudyBError(f"{label} must be a lowercase SHA-256")
    return value


def _verify_file(path: Path, expected: str, label: str) -> str:
    expected = _required_sha(expected, f"{label} expected hash")
    actual = sha256_file(path)
    if actual != expected:
        raise StudyBError(f"{label} SHA-256 mismatch")
    return actual


def _canonical_file(path: Path, canonical: Path, expected: str, label: str) -> str:
    if path.is_symlink() or not path.is_file() or path.resolve() != canonical.resolve():
        raise StudyBError(f"{label} must be canonical repository file {canonical}")
    return _verify_file(path, expected, label)


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudyBError(f"cannot read {label}: {error}") from error


def _bind_study_a_manifest_provenance(
    path: Path,
    values: tuple[object, ...],
) -> tuple[object, ...]:
    """Bind Study A rows to the atomically published sibling manifest."""

    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StudyBError("Study A manifest must be a regular non-symlink file")
    manifest = _load_json(manifest_path, "Study A manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise StudyBError("Study A manifest must be a schema_version 1 mapping")
    if manifest.get("status") != "V5_STUDY_A_ATOMICALLY_PUBLISHED":
        raise StudyBError("Study A manifest is not atomically published")
    files = manifest.get("files")
    file_record = files.get(path.name) if isinstance(files, Mapping) else None
    if not isinstance(file_record, Mapping):
        raise StudyBError("Study A manifest does not bind the evaluation artifact")
    if file_record.get("sha256") != sha256_file(path):
        raise StudyBError("Study A manifest evaluation SHA-256 mismatch")
    recorded_size = file_record.get("size_bytes")
    if (
        not isinstance(recorded_size, int)
        or isinstance(recorded_size, bool)
        or recorded_size != path.stat().st_size
    ):
        raise StudyBError("Study A manifest evaluation size mismatch")
    source_sha256 = manifest.get("source_sha256")
    if not isinstance(source_sha256, Mapping):
        raise StudyBError("Study A manifest lacks source_sha256")
    return tuple(
        {**dict(value), "source_sha256": dict(source_sha256)}
        if isinstance(value, Mapping)
        else value
        for value in values
    )


def _load_evaluation(path: Path) -> tuple[dict[str, object], ...]:
    if path.suffix == ".jsonl":
        try:
            values = tuple(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise StudyBError(f"cannot read evaluation JSONL: {error}") from error
    else:
        payload = _load_json(path, "evaluation package")
        if isinstance(payload, Mapping):
            payload = payload.get("rows")
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise StudyBError("evaluation JSON must be a row list or a mapping with rows")
        values = tuple(payload)
    try:
        return validate_evaluation_rows(values)  # type: ignore[arg-type]
    except StudyBError:
        try:
            return evaluation_rows_from_study_a(values)  # type: ignore[arg-type]
        except StudyBError:
            manifest_path = path.parent / "manifest.json"
            if not manifest_path.exists() and not manifest_path.is_symlink():
                raise
            bound_values = _bind_study_a_manifest_provenance(path, values)
            return evaluation_rows_from_study_a(bound_values)  # type: ignore[arg-type]


def _validate_support_source_provenance(
    *,
    support: Mapping[str, object],
    parent_manifest: Path,
    child_manifest: Path,
    frozen_scenes: Path,
) -> dict[str, str]:
    """Rehash the Phase2a source chain and verify the child/package links."""

    provenance = support.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise StudyBError("support package lacks source_provenance")
    observed = {
        "parent_manifest_sha256": sha256_file(parent_manifest),
        "child_manifest_sha256": sha256_file(child_manifest),
        "frozen_scenes_sha256": sha256_file(frozen_scenes),
    }
    for field, digest in observed.items():
        if provenance.get(field) != digest:
            raise StudyBError(f"support source_provenance {field} mismatch")
    child = _load_json(child_manifest, "Phase2a child manifest")
    if not isinstance(child, Mapping):
        raise StudyBError("Phase2a child manifest must be a mapping")
    if child.get("status") != "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN":
        raise StudyBError("Phase2a child manifest is not frozen")
    if child.get("parent_manifest_modified") is not False:
        raise StudyBError("Phase2a child manifest reports a modified parent")
    if child.get("semantic_scene_count") != 96:
        raise StudyBError("Phase2a child manifest must freeze exactly 96 semantic scenes")
    if child.get("base_sha256") != MODEL_SNAPSHOT_SHA256:
        raise StudyBError("Phase2a child manifest has the wrong Base SHA-256")
    for field in ("parent_manifest_sha256", "frozen_scenes_sha256"):
        if child.get(field) != observed[field]:
            raise StudyBError(f"Phase2a child manifest {field} mismatch")
    return observed


def _validate_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise StudyBError("Study B config must be a schema_version 1 mapping")
    if payload.get("phase") != "phase4_budget_matched_lora":
        raise StudyBError("Study B config phase is not registered")
    if payload.get("arms") != list(ARMS) or payload.get("seeds") != [PILOT_SEED]:
        raise StudyBError("Study B config must register B0-B3 and exactly the pilot seed")
    canonical_scalars = {
        "unique_source_scenes": 96,
        "rows_per_source": 6,
        "rows_per_arm": 576,
        "steps": 72,
        "per_device_train_batch_size": 1,
        "gradient_accumulation": 8,
        "num_train_epochs": 1,
        "target_token_relative_tolerance": 0.01,
    }
    for field, expected in canonical_scalars.items():
        if payload.get(field) != expected:
            raise StudyBError(f"Study B config {field} must be exactly {expected}")
    if payload.get("vision_frozen") is not True or payload.get("merger_frozen") is not True:
        raise StudyBError("Study B config must freeze vision and merger")
    authorization = payload.get("authorization")
    if authorization != {
        "inference_allowed": True,
        "training_allowed": True,
        "rl_allowed": False,
        "downloads_allowed": False,
    }:
        raise StudyBError("Study B authorization must allow only offline SFT and inference")
    offline = payload.get("offline")
    if offline != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }:
        raise StudyBError("Study B config offline contract drifted")
    return payload


def _validate_config_against_support(
    config: Mapping[str, object], support: Mapping[str, object]
) -> None:
    validated = validate_support_package(support)
    reference = validated["budgets"]["B0"]
    optimizer, lora = config.get("optimizer"), config.get("lora")
    if not isinstance(optimizer, Mapping) or not isinstance(lora, Mapping):
        raise StudyBError("Study B config lacks optimizer or LoRA mappings")
    if dict(optimizer) != reference["optimizer"]:
        raise StudyBError("config optimizer differs from the frozen support budget")
    expected_lora = {
        "rank": reference["lora_rank"],
        "alpha": 2 * int(reference["lora_rank"]),
        "dropout": 0.0,
        "targets": reference["lora_targets"],
    }
    if dict(lora) != expected_lora:
        raise StudyBError("config LoRA parameters differ from the frozen support budget")
    if (
        config.get("target_token_relative_tolerance")
        != validated["target_token_relative_tolerance"]
    ):
        raise StudyBError("config target-token tolerance differs from support freeze")
    expected_budget_fields = {
        "unique_source_scenes": config["unique_source_scenes"],
        "rows": config["rows_per_arm"],
        "steps": config["steps"],
        "gradient_accumulation": config["gradient_accumulation"],
    }
    for field, expected in expected_budget_fields.items():
        if reference[field] != expected:
            raise StudyBError(f"support budget {field} differs from canonical config")


def _validate_output(output: Path, root: Path, *, resume: bool) -> Path:
    if root.is_symlink():
        raise StudyBError("output root must not be a symlink")
    target = output.resolve(strict=False)
    try:
        target.relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise StudyBError("Study B output must remain below --output-root") from error
    if target.exists() and not resume:
        raise FileExistsError("Study B output exists; immutable outputs cannot be overwritten")
    if not target.exists() and resume:
        raise StudyBError("--resume requires an existing Study B output")
    return target


def main() -> int:
    arguments = _parser().parse_args()
    if not arguments.execute:
        print("BLOCKED: explicit --execute is required before any GPU import or model load")
        return 2
    if arguments.ack != ACKNOWLEDGEMENT:
        print(f"BLOCKED: exact --ack {ACKNOWLEDGEMENT} is required")
        return 2
    try:
        required = {
            "support-package": arguments.support_package,
            "support-sha256": arguments.support_sha256,
            "parent-manifest": arguments.parent_manifest,
            "parent-manifest-sha256": arguments.parent_manifest_sha256,
            "child-manifest": arguments.child_manifest,
            "child-manifest-sha256": arguments.child_manifest_sha256,
            "frozen-scenes": arguments.frozen_scenes,
            "frozen-scenes-sha256": arguments.frozen_scenes_sha256,
            "evaluation": arguments.evaluation,
            "evaluation-sha256": arguments.evaluation_sha256,
            "config-sha256": arguments.config_sha256,
            "package-lock-sha256": arguments.package_lock_sha256,
            "model-sha256": arguments.model_sha256,
            "output": arguments.output,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise StudyBError("missing required execution arguments: " + ", ".join(missing))
        assert arguments.support_package is not None
        assert arguments.support_sha256 is not None
        assert arguments.parent_manifest is not None
        assert arguments.parent_manifest_sha256 is not None
        assert arguments.child_manifest is not None
        assert arguments.child_manifest_sha256 is not None
        assert arguments.frozen_scenes is not None
        assert arguments.frozen_scenes_sha256 is not None
        assert arguments.evaluation is not None
        assert arguments.evaluation_sha256 is not None
        assert arguments.config_sha256 is not None
        assert arguments.package_lock_sha256 is not None
        assert arguments.model_sha256 is not None
        assert arguments.output is not None
        require_offline_environment()
        config_hash = _canonical_file(
            arguments.config, CONFIG, arguments.config_sha256, "Study B config"
        )
        lock_hash = _canonical_file(
            arguments.package_lock,
            PACKAGE_LOCK,
            arguments.package_lock_sha256,
            "Study B package lock",
        )
        support_hash = _verify_file(
            arguments.support_package, arguments.support_sha256, "support package"
        )
        parent_manifest_hash = _verify_file(
            arguments.parent_manifest,
            arguments.parent_manifest_sha256,
            "Phase2a parent manifest",
        )
        child_manifest_hash = _verify_file(
            arguments.child_manifest,
            arguments.child_manifest_sha256,
            "Phase2a child manifest",
        )
        frozen_scenes_hash = _verify_file(
            arguments.frozen_scenes,
            arguments.frozen_scenes_sha256,
            "Phase2a frozen scenes",
        )
        evaluation_hash = _verify_file(
            arguments.evaluation, arguments.evaluation_sha256, "evaluation package"
        )
        if arguments.model_path != Path(MODEL_PATH):
            raise StudyBError(f"model path must be the frozen v4 path {MODEL_PATH}")
        if _required_sha(arguments.model_sha256, "model SHA") != MODEL_SNAPSHOT_SHA256:
            raise StudyBError("model SHA differs from the frozen v4 Qwen snapshot")
        output = _validate_output(arguments.output, arguments.output_root, resume=arguments.resume)
        config = _validate_config(arguments.config)
        support = _load_json(arguments.support_package, "support package")
        if not isinstance(support, Mapping):
            raise StudyBError("support package must be a JSON mapping")
        _validate_config_against_support(config, support)
        source_provenance = _validate_support_source_provenance(
            support=support,
            parent_manifest=arguments.parent_manifest,
            child_manifest=arguments.child_manifest,
            frozen_scenes=arguments.frozen_scenes,
        )
        evaluation = _load_evaluation(arguments.evaluation)
        verify_runtime_package_lock(arguments.package_lock)
        backend = QwenStudyBBackend(
            model_path=arguments.model_path,
            max_sequence_length=arguments.max_sequence_length,
        )
        result = run_study_b(
            support_package=support,
            evaluation_rows=evaluation,
            output=output,
            backend=backend,
            expected_model_sha256=arguments.model_sha256,
            seed=PILOT_SEED,
            resume=arguments.resume,
            provenance={
                "config_sha256": config_hash,
                "package_lock_sha256": lock_hash,
                "support_file_sha256": support_hash,
                "parent_manifest_sha256": parent_manifest_hash,
                "child_manifest_sha256": child_manifest_hash,
                "frozen_scenes_sha256": frozen_scenes_hash,
                "support_source_provenance": source_provenance,
                "evaluation_file_sha256": evaluation_hash,
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError, StudyBError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
