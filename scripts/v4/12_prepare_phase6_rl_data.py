"""Build the split-isolated Phase 6 GRPO prompt/reward datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _guards import (  # noqa: E402
    PHASE_C_DATASET_MANIFEST_SHA256,
    PHASE_C_DATASET_RECORDS_SHA256,
    ROOT,
    sha256,
)

from compensability_v4.qwen.phase6_runtime import (  # noqa: E402
    PHASE6_LOCKED_PATHS,
    load_phase6_execution_manifest,
    verify_phase6_package_lock,
)
from compensability_v4.training.phase4_sources import (  # noqa: E402
    PreparedSourcePaths,
    validate_prepared_source_summary,
)
from compensability_v4.training.phase6 import (  # noqa: E402
    RewardKind,
    build_phase6_examples,
    load_phase6_config,
    validate_phase5_policy_support,
)

CONFIG = ROOT / "configs/recoverability/v4_phase_6.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml"
PHASE4_SOURCE_ROOT = ROOT / "artifacts/v4/training/sources"
PHASE5_ROOT = ROOT / "artifacts/v4/support"
DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"
OUTPUT_ROOT = ROOT / "artifacts/v4/rl/data"
EXECUTION_MANIFEST = ROOT / "artifacts/v4/phase6/execution_manifest.json"


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 6 {label} is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Phase 6 {label} must contain one object")
    return value


def _jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 6 {label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 6 {label} is empty or malformed")
    return rows  # type: ignore[return-value]


def _publish(
    output_root: Path,
    *,
    recovery_rows: tuple[dict[str, object], ...],
    answer_rows: tuple[dict[str, object], ...],
    provenance: dict[str, object],
) -> dict[str, Path]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 6 RL data")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase6-data-", dir=str(output_root.parent)))
    paths = {
        "recovery": temporary / "recovery_outcome.jsonl",
        "answer": temporary / "answer_only.jsonl",
        "summary": temporary / "summary.json",
    }
    try:
        for path, rows in ((paths["recovery"], recovery_rows), (paths["answer"], answer_rows)):
            with path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        summary = {
            "schema_version": 1,
            "status": "PHASE_6_RL_DATA_FROZEN",
            "recovery_outcome_count": len(recovery_rows),
            "answer_only_count": len(answer_rows),
            "scene_count": len(recovery_rows),
            "recovery_outcome_sha256": sha256(paths["recovery"]),
            "answer_only_sha256": sha256(paths["answer"]),
            **provenance,
            "confirmatory_data_used": False,
            "subjective_success_threshold_applied": False,
            "training_invoked": False,
            "rl_invoked": False,
        }
        with paths["summary"].open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {key: output_root / value.name for key, value in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--execution-manifest", type=Path, default=EXECUTION_MANIFEST)
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--phase4-source-root", type=Path, default=PHASE4_SOURCE_ROOT)
    parser.add_argument("--phase5-root", type=Path, default=PHASE5_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 6 data preparation requires explicit --execute.")
        return 2
    if not arguments.execution_manifest_sha256:
        print("BLOCKED: Phase 6 data preparation requires --execution-manifest-sha256.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        _payload, _training = load_phase6_config(arguments.config)
        lock_hash = verify_phase6_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE6_LOCKED_PATHS,
        )
        manifest = load_phase6_execution_manifest(
            arguments.execution_manifest,
            expected_sha256=arguments.execution_manifest_sha256,
            expected_config_sha256=sha256(arguments.config),
            expected_package_lock_sha256=lock_hash,
        )
        source_paths = PreparedSourcePaths(
            symbolic_scenes=arguments.phase4_source_root / "symbolic_scenes.jsonl",
            natural_scenes=arguments.phase4_source_root / "natural_scenes.jsonl",
            natural_observations=arguments.phase4_source_root / "natural_observations.jsonl",
            selection_trace=arguments.phase4_source_root / "selection_trace.jsonl",
            summary=arguments.phase4_source_root / "source_summary.json",
        )
        source_hashes = validate_prepared_source_summary(source_paths.summary, paths=source_paths)
        dataset_manifest = arguments.dataset_root / "manifest.json"
        dataset_records = arguments.dataset_root / "records.jsonl"
        if (
            sha256(dataset_manifest) != PHASE_C_DATASET_MANIFEST_SHA256
            or sha256(dataset_records) != PHASE_C_DATASET_RECORDS_SHA256
        ):
            raise RuntimeError("Phase 6 dataset hashes drifted")
        phase5_summary_path = arguments.phase5_root / "informative_group_rate.json"
        phase5_summary = _json(phase5_summary_path, "Phase 5 summary")
        validate_phase5_policy_support(phase5_summary)
        if sha256(phase5_summary_path) != manifest["phase5_policy_support_summary_sha256"]:
            raise RuntimeError("Phase 6 Phase 5 summary no longer matches the execution manifest")
        if phase5_summary.get("source_sha256") != manifest["source_sha256"]:
            raise RuntimeError("Phase 6 Phase 5 source hashes drifted from the execution manifest")
        phase5_files = (
            arguments.phase5_root / "policy_support_by_scene.parquet",
            phase5_summary_path,
            arguments.phase5_root / "pass_at_k.csv",
        )
        if any(path.is_symlink() or not path.is_file() for path in phase5_files):
            raise RuntimeError("Phase 6 requires all three Phase 5 formal artifacts")
        examples = build_phase6_examples(
            natural_scenes=_jsonl(source_paths.natural_scenes, "natural scenes"),
            natural_observations=_jsonl(source_paths.natural_observations, "natural observations"),
            dataset_records=_jsonl(dataset_records, "dataset records"),
        )
        recovery = tuple(
            row.to_mapping() for row in examples if row.reward_kind is RewardKind.RECOVERY_OUTCOME
        )
        answer = tuple(
            row.to_mapping() for row in examples if row.reward_kind is RewardKind.ANSWER_ONLY
        )
        if len(recovery) != len(answer):
            raise RuntimeError("Phase 6 reward-view scene closure drifted")
        paths = _publish(
            arguments.output_root,
            recovery_rows=recovery,
            answer_rows=answer,
            provenance={
                "config_sha256": sha256(arguments.config),
                "package_lock_sha256": lock_hash,
                "execution_manifest_sha256": arguments.execution_manifest_sha256,
                "phase5_policy_support_summary_sha256": manifest[
                    "phase5_policy_support_summary_sha256"
                ],
                "phase4_source_sha256": source_hashes,
                "phase5_artifact_sha256": {path.name: sha256(path) for path in phase5_files},
                "dataset_manifest_sha256": PHASE_C_DATASET_MANIFEST_SHA256,
                "dataset_records_sha256": PHASE_C_DATASET_RECORDS_SHA256,
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 6 RL data frozen; scenes={len(recovery)}")
    for path in paths.values():
        print(f"SHA256 {sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
