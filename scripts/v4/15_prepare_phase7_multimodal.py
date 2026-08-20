"""Prepare the hash-bound Phase 7 support-dev multimodal execution manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

from compensability_v4.qwen.model_loader import MODEL_SNAPSHOT_SHA256
from compensability_v4.qwen.phase5_runtime import checkpoint_tree_hashes, tree_sha256
from compensability_v4.qwen.phase7_runtime import (
    PHASE7_LOCKED_PATHS,
    build_phase7_execution_manifest,
    load_phase7_config,
    verify_phase7_package_lock,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_7.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_7.yaml"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
PHASE6_RUN_ROOT = ROOT / "artifacts/v4/rl/runs/phase6-r1"
DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"
PROMPTS = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
OUTPUT = ROOT / "artifacts/v4/phase7/execution_manifest.json"
_SOURCE_NAMES = (
    "dataset_records",
    "support_dev",
    "phase4_summary",
    "phase5_summary",
    "phase6_evaluation",
)
_PHASE6_CHECKPOINTS = (
    "Base_AnswerOnly_RL",
    "Recovery_LoRA_RecoveryOutcome_RL",
    "Recovery_LoRA_AnswerOnly_RL",
)
_EXECUTION_PARAMETERS = {
    "resized_height": 280,
    "resized_width": 280,
    "stage1_max_new_tokens": 32,
    "recovery_max_new_tokens": 32,
    "operation_max_new_tokens": 8,
    "answer_max_new_tokens": 8,
    "greedy_seed": 2026082102,
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_named(values: list[str] | None, *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        name, separator, item = value.partition("=")
        if not separator or name not in _SOURCE_NAMES or not item or name in parsed:
            raise ValueError(
                f"Phase 7 {option} entries must be unique NAME=VALUE pairs for "
                f"{','.join(_SOURCE_NAMES)}"
            )
        parsed[name] = item
    if set(parsed) != set(_SOURCE_NAMES):
        raise ValueError(f"Phase 7 {option} must bind exactly {','.join(_SOURCE_NAMES)}")
    return parsed


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 7 {label} is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Phase 7 {label} must contain one JSON object")
    return payload


def _jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 7 {label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 7 {label} is empty or malformed")
    return rows  # type: ignore[return-value]


def _support_dev_image_bundle_sha256(
    *, support_dev: Path, dataset_records: Path, dataset_root: Path
) -> str:
    import hashlib

    if not dataset_root.is_absolute() or dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RuntimeError("Phase 7 dataset root must be an absolute regular directory")
    root = dataset_root.resolve()
    records = {str(row.get("scene_id")): row for row in _jsonl(dataset_records, "dataset records")}
    errors = _jsonl(support_dev, "support-dev errors")
    if len(errors) != 32 or len({row.get("scene_id") for row in errors}) != 32:
        raise RuntimeError("Phase 7 support-dev image bundle requires exactly 32 unique scenes")
    bundle: list[tuple[str, str, Path]] = []
    for error in errors:
        scene_id, relative = error.get("scene_id"), error.get("image_path")
        record = records.get(str(scene_id))
        if (
            not isinstance(scene_id, str)
            or not isinstance(relative, str)
            or not isinstance(record, dict)
            or record.get("image") != relative
        ):
            raise RuntimeError("Phase 7 support-dev image mapping drifted")
        posix = PurePosixPath(relative)
        image = (root / Path(*posix.parts)).resolve()
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or posix.suffix.lower() != ".png"
            or root not in image.parents
            or image.is_symlink()
            or not image.is_file()
        ):
            raise RuntimeError("Phase 7 support-dev image is missing or unsafe")
        bundle.append((scene_id, relative, image))
    digest = hashlib.sha256()
    for scene_id, relative, image in sorted(bundle):
        digest.update(scene_id.encode())
        digest.update(b"\0")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_sha256(image).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_hashes(
    phase4_root: Path,
    phase6_root: Path,
    *,
    phase5_summary: dict[str, object],
    phase6_evaluation: dict[str, object],
) -> dict[str, str]:
    phase4 = checkpoint_tree_hashes(phase4_root)
    expected_phase4 = phase5_summary.get("source_sha256")
    if not isinstance(expected_phase4, dict):
        raise RuntimeError("Phase 7 Phase-5 checkpoint provenance is missing")
    hashes = {"Base": MODEL_SNAPSHOT_SHA256, **phase4}
    for checkpoint in ("Base", "C0", "C1", "T"):
        if hashes.get(checkpoint) != expected_phase4.get(checkpoint):
            raise RuntimeError(f"Phase 7 {checkpoint} hash differs from Phase 5 evidence")
    phase6_hashes = phase6_evaluation.get("checkpoint_sha256")
    if not isinstance(phase6_hashes, dict):
        raise RuntimeError("Phase 7 Phase-6 checkpoint provenance is missing")
    for checkpoint in _PHASE6_CHECKPOINTS:
        adapter = phase6_root / checkpoint / "final_adapter"
        observed = tree_sha256(adapter)
        if observed != phase6_hashes.get(checkpoint):
            raise RuntimeError(f"Phase 7 {checkpoint} hash differs from Phase 6 evidence")
        evidence = _json(phase6_root / checkpoint / "execution_evidence.json", checkpoint)
        if (
            evidence.get("status") != "PHASE_6_VARIANT_TRAINED"
            or evidence.get("variant") != checkpoint
            or evidence.get("final_adapter_tree_sha256") != observed
        ):
            raise RuntimeError(f"Phase 7 {checkpoint} training evidence drifted")
        hashes[checkpoint] = observed
    return hashes


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 7 execution manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".phase7-manifest-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite Phase 7 execution manifest") from error
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument(
        "--input",
        action="append",
        help="hash-bound NAME=PATH; required names are fixed by the Phase 7 contract",
    )
    parser.add_argument(
        "--input-sha256",
        action="append",
        help="matching NAME=SHA256 for every --input",
    )
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--phase6-run-root", type=Path, default=PHASE6_RUN_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--prompt-config", type=Path, default=PROMPTS)
    parser.add_argument("--output-path", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 7 manifest preparation requires explicit --execute.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("Phase 7 requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
        paths = {
            name: Path(value).resolve()
            for name, value in _parse_named(arguments.input, option="--input").items()
        }
        expected = _parse_named(arguments.input_sha256, option="--input-sha256")
        observed = {name: _sha256(path) for name, path in paths.items()}
        if observed != expected:
            raise RuntimeError("Phase 7 input SHA-256 bindings do not match")
        config = load_phase7_config(arguments.config)
        lock_hash = verify_phase7_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE7_LOCKED_PATHS,
        )
        phase4_summary = _json(paths["phase4_summary"], "Phase-4 summary")
        phase5_summary = _json(paths["phase5_summary"], "Phase-5 summary")
        phase6_evaluation = _json(paths["phase6_evaluation"], "Phase-6 evaluation")
        if (
            phase4_summary.get("artifact_type") != "phase_4_support_data"
            or phase4_summary.get("contains_confirmatory_data") is not False
        ):
            raise RuntimeError("Phase 7 requires the frozen non-confirmatory Phase 4 corpus")
        if phase5_summary.get("status") != "PHASE_5_POLICY_SUPPORT_EXECUTED":
            raise RuntimeError("Phase 7 requires completed Phase 5 policy support")
        if phase6_evaluation.get("status") != "PHASE_6_RL_EVALUATED":
            raise RuntimeError("Phase 7 requires completed Phase 6 evaluation")
        if any(
            payload.get("confirmatory_data_used") is not False
            for payload in (phase5_summary, phase6_evaluation)
        ):
            raise RuntimeError("Phase 7 support-dev preparation cannot consume confirmatory data")
        phase5_sources = phase5_summary.get("source_sha256")
        phase6_inputs = phase6_evaluation.get("input_sha256")
        if (
            not isinstance(phase5_sources, dict)
            or phase5_sources.get("support_dev") != observed["support_dev"]
            or not isinstance(phase6_inputs, dict)
            or phase6_inputs.get("held_out_natural_errors") != observed["support_dev"]
            or phase6_inputs.get("dataset_records") != observed["dataset_records"]
        ):
            raise RuntimeError("Phase 7 source hashes do not close Phase 5/6 provenance")
        checkpoint_hashes = _checkpoint_hashes(
            arguments.phase4_run_root,
            arguments.phase6_run_root,
            phase5_summary=phase5_summary,
            phase6_evaluation=phase6_evaluation,
        )
        manifest = build_phase7_execution_manifest(
            config=config,
            source_sha256=observed,
            checkpoint_sha256=checkpoint_hashes,
            config_sha256=_sha256(arguments.config),
            package_lock_sha256=lock_hash,
        )
        image_bundle_hash = _support_dev_image_bundle_sha256(
            support_dev=paths["support_dev"],
            dataset_records=paths["dataset_records"],
            dataset_root=arguments.dataset_root,
        )
        if arguments.prompt_config.is_symlink() or not arguments.prompt_config.is_file():
            raise RuntimeError("Phase 7 Stage-1 prompt config is missing or unsafe")
        manifest = {
            **manifest,
            "execution_parameters": _EXECUTION_PARAMETERS,
            "support_dev_image_bundle_sha256": image_bundle_hash,
            "stage1_prompt_config_sha256": _sha256(arguments.prompt_config),
        }
        _write_manifest(arguments.output_path, manifest)
    except Exception as error:
        print(f"BLOCKED: Phase 7 {error}")
        return 2
    print(f"READY: Phase 7 multimodal execution manifest written to {arguments.output_path}")
    print(f"SHA256 {_sha256(arguments.output_path)}  {arguments.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
