#!/usr/bin/env python3
"""Run the fixed 300-scene bridge on the reviewed offline GPU server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

_BOOTSTRAP_SERVER_PATHS = frozenset(
    {
        "configs/data/cva_chart_pilot_v0_3.yaml",
        "configs/recoverability/recoverability_v1.yaml",
        "configs/recoverability/power_plan_v1.json",
        "configs/recoverability/server_runtime_v1.yaml",
        "configs/recoverability/v0_3_negative_pilot.yaml",
        "experiments/recoverability_v1/00_preflight.py",
        "experiments/recoverability_v1/02_capture_v03_evidence.py",
        "experiments/recoverability_v1/03_bridge.py",
        "requirements-gpu.lock.txt",
        "src/compbias/gpu_pilot/chart_data.py",
        "src/compbias/gpu_pilot/config.py",
        "src/compbias/gpu_pilot/execution_gate.py",
        "src/compbias/gpu_pilot/preflight.py",
        "src/compbias/gpu_pilot/qwen_smoke.py",
        "src/compbias/gpu_pilot/safe_io.py",
        "src/compbias/gpu_pilot/structured_generation.py",
        "src/compbias/gpu_pilot/taxonomy.py",
        "src/compbias/io/strict_json.py",
        "src/compbias/io/yaml_config.py",
        "src/compbias/models/structured_parser.py",
        "src/compbias/recoverability/bridge.py",
        "src/compbias/recoverability/config.py",
        "src/compbias/recoverability/dsl/executor.py",
        "src/compbias/recoverability/dsl/parser.py",
        "src/compbias/recoverability/dsl/schema.py",
        "src/compbias/recoverability/evidence.py",
        "src/compbias/recoverability/evidence_capture.py",
        "src/compbias/recoverability/preflight.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# This stdlib-only check runs before any project module can be imported or a model loaded.
def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in os.sys.argv:
        return
    try:
        raw = os.sys.argv[os.sys.argv.index("--server-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical server package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    if lock_path != root / "configs/recoverability/server_package_lock_v1.yaml":
        raise SystemExit("BLOCKED: server package lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: server package lock is malformed")
    if frozenset(paths) != _BOOTSTRAP_SERVER_PATHS or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: server package lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: server package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import ACTIVE_PILOT_OUTPUT_SLUG, load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.execution_gate import (
    _validate_dataset_bundle,
    _validate_manifest,
    _validate_natural_records,
)  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import decode_qwen_once, load_local_qwen  # noqa: E402
from compbias.gpu_pilot.structured_generation import build_structured_messages  # noqa: E402
from compbias.recoverability.bridge import (
    BridgeScene,
    decode_text_qwen_once,
    run_bridge_protocol,
)  # noqa: E402
from compbias.recoverability.config import load_recoverability_protocol  # noqa: E402
from compbias.recoverability.evidence import (
    load_negative_pilot_record,
    verify_server_package_lock,
)  # noqa: E402


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_preflight(path: Path, *, lock_path: Path) -> None:
    payload = _load_json(path, label="preflight report")
    if (
        payload.get("artifact_type") != "recoverability_v1_metadata_preflight"
        or payload.get("ready") is not True
        or payload.get("large_gpu_started") is not False
        or payload.get("model_loaded") is not False
        or payload.get("training_authorized") is not False
        or payload.get("server_package_lock_verified") is not True
        or payload.get("server_package_lock_sha256") != _sha256(lock_path)
    ):
        raise ValueError("preflight report does not authorize the fixed bridge")


def _validate_external_evidence(
    path: Path,
    *,
    records_path: Path,
    negative: object,
) -> None:
    payload = _load_json(path, label="v0.3 external evidence")
    expected = {
        "status": "FROZEN_FAILED_NOT_TO_BE_RERUN",
        "records": negative.records,
        "gate_passed": False,
        "calibration_exit": 3,
        "model_snapshot_sha256": negative.model_snapshot_sha256,
        "dataset_manifest_sha256": negative.dataset_manifest_sha256,
        "dataset_records_sha256": negative.dataset_records_sha256,
        "dataset_images_sha256": negative.dataset_images_sha256,
        "counterfactual_sha256": negative.counterfactual_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("v0.3 external evidence differs from the frozen negative pilot")
    sources = payload.get("source_files")
    if not isinstance(sources, list):
        raise ValueError("v0.3 external evidence source registry is invalid")
    record_source = next(
        (
            item
            for item in sources
            if isinstance(item, dict) and item.get("basename") == "calibration_records_v0_3.jsonl"
        ),
        None,
    )
    if record_source is None or record_source.get("sha256") != _sha256(records_path):
        raise ValueError("v0.3 calibration records differ from captured evidence")


def _validate_canonical_dataset(dataset: Path, *, negative: object) -> dict[str, object]:
    manifest_path = dataset / "manifest.json"
    manifest = _load_json(manifest_path, label="v0.3 dataset manifest")
    _validate_manifest(manifest)
    if (
        _sha256(manifest_path) != negative.dataset_manifest_sha256
        or manifest.get("records_sha256") != negative.dataset_records_sha256
        or manifest.get("images_sha256") != negative.dataset_images_sha256
        or manifest.get("counterfactual_sha256") != negative.counterfactual_sha256
    ):
        raise ValueError("v0.3 dataset hashes differ from the frozen negative pilot")
    return _validate_dataset_bundle(manifest_path, manifest)


def _read_scenes(dataset: Path) -> tuple[BridgeScene, ...]:
    records_path = dataset / "records.jsonl"
    scenes: list[BridgeScene] = []
    with records_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("split") != "iid_test":
                continue
            values = record.get("values")
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError("bridge values must contain four integers")
            scenes.append(
                BridgeScene(
                    scene_id=str(record["sample_id"]),
                    image_path=(dataset / str(record["image"])).resolve(),
                    question=str(record["question"]),
                    operation=str(record["operation"]),
                    values=tuple(values),  # type: ignore[arg-type]
                    answer=int(record["answer"]),
                )
            )
    if len(scenes) != 300:
        raise ValueError(f"bridge requires exactly 300 iid_test scenes, observed {len(scenes)}")
    if any(not scene.image_path.is_file() or scene.image_path.is_symlink() for scene in scenes):
        raise ValueError("bridge image is missing or is a symlink")
    return tuple(scenes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--external-evidence", type=Path)
    parser.add_argument("--v03-records", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: bridge inference requires explicit --execute on the reviewed GPU server")
        return 2
    if any(
        value is None
        for value in (
            args.server_package_lock,
            args.preflight_report,
            args.external_evidence,
            args.v03_records,
        )
    ):
        print("BLOCKED: bridge inference requires lock, preflight, and v0.3 evidence")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repository_root = Path(__file__).resolve().parents[2]
    verify_server_package_lock(args.server_package_lock, repository_root=repository_root)
    _validate_preflight(args.preflight_report, lock_path=args.server_package_lock)
    negative_path = repository_root / "configs/recoverability/v0_3_negative_pilot.yaml"
    negative = load_negative_pilot_record(negative_path)
    _validate_external_evidence(
        args.external_evidence,
        records_path=args.v03_records,
        negative=negative,
    )
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("bridge forbids COMPBIAS path overrides")
    if args.paths.resolve() != repository_root / "configs/paths.yaml":
        raise ValueError("bridge paths configuration must use configs/paths.yaml")
    if args.protocol.resolve() != repository_root / "configs/recoverability/recoverability_v1.yaml":
        raise ValueError("bridge protocol configuration path is not canonical")
    paths = load_pilot_paths(args.paths)
    protocol = load_recoverability_protocol(args.protocol)
    dataset = paths.data / "generated" / ACTIVE_PILOT_OUTPUT_SLUG
    dataset_records = _validate_canonical_dataset(dataset, negative=negative)
    derived_v03 = _validate_natural_records(
        args.v03_records,
        split="calibration",
        dataset_records=dataset_records,
    )
    if derived_v03["error_counts"] != dict(negative.error_counts):
        raise ValueError("v0.3 calibration taxonomy does not replay")
    output = paths.outputs / "recoverability_v1" / protocol.bridge.output_subdirectory
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite bridge output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt_marker = output.parent / f"{protocol.bridge.output_subdirectory}.attempted.json"
    with attempt_marker.open("x", encoding="utf-8") as stream:
        json.dump({"status": "BRIDGE_ATTEMPT_STARTED", "schema_version": 1}, stream)
        stream.write("\n")
    scenes = _read_scenes(dataset)
    model_hash_before = model_snapshot_sha256(paths.model_path)
    if model_hash_before != negative.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen v0.3 pilot")
    model, processor = load_local_qwen(paths.model_path)

    def legacy(scene: BridgeScene) -> str:
        messages = build_structured_messages(
            question=scene.question,
            operation=scene.operation,
            retry_index=0,
            expected_value_count=4,
        )
        return decode_qwen_once(model, processor, scene.image_path, messages)

    report, records = run_bridge_protocol(
        scenes,
        legacy_generate=legacy,
        stage1_generate=lambda scene, messages: decode_qwen_once(
            model, processor, scene.image_path, messages
        ),
        stage2_generate=lambda _scene, messages: decode_text_qwen_once(model, processor, messages),
        equivalence_margin=0.03,
    )
    if model_snapshot_sha256(paths.model_path) != model_hash_before:
        raise RuntimeError("model snapshot changed during bridge inference")
    payload = {
        **asdict(report),
        "schema_version": 1,
        "artifact_type": "recoverability_v1_bridge_report",
        "dataset_id": protocol.bridge.dataset_id,
        "model_snapshot_sha256": model_hash_before,
        "format_retries": 0,
        "training_invoked": False,
    }
    with tempfile.TemporaryDirectory(prefix=".bridge-stage-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        with (staging / "bridge_records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
        (staging / "bridge_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.protocols_mergeable else 3


if __name__ == "__main__":
    raise SystemExit(main())
