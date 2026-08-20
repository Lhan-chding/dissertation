"""Prepare Phase 4 support sources directly from completed, frozen S6 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compensability_v4.qwen.model_loader import MODEL_SNAPSHOT_SHA256  # noqa: E402
from compensability_v4.training.phase4 import (  # noqa: E402
    load_phase4_config,
    sha256_path,
    verify_phase4_package_lock,
)
from compensability_v4.training.phase4_sources import (  # noqa: E402
    prepare_legacy_s6_support_sources,
    write_prepared_support_sources,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_4.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_4.yaml"
S6_PER_SCENE = ROOT / "artifacts/v4/interface_ladder/per_scene.jsonl"
S6_SUMMARY = ROOT / "artifacts/v4/interface_ladder/summary.json"
DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"
OUTPUT_ROOT = ROOT / "artifacts/v4/training/sources"
DATASET_MANIFEST_SHA256 = "bc57389dc3164b6aeba8d4565aecfaea3fa7ba171b4df4843c8ec86cbee8a19f"
DATASET_RECORDS_SHA256 = "36e09f7e15107057fd1b942875d12259b1f281e0354b87c82ed17f420693c766"
EXPECTED_S6_SCENES = 579
SYMBOLIC_SCENE_COUNT = 579
SYMBOLIC_SEED = 2026081804
VALUE_DOMAIN = range(2, 19)
IMAGE_GRID_THW = (1, 20, 20)
_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_4.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/07_prepare_phase4_support_sources.py",
    "scripts/v4/07_build_support_data.py",
    "scripts/v4/08_train_phase4_lora.py",
    "src/compensability_v4/training/__init__.py",
    "src/compensability_v4/training/phase4.py",
    "src/compensability_v4/training/phase4_sources.py",
)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 4 {label} must be a regular JSON file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Phase 4 {label} must contain one JSON object")
    return payload


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 4 {label} must be a regular JSONL file")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 4 {label} must contain nonempty JSON objects")
    return rows  # type: ignore[return-value]


def _load_dataset(root: Path) -> tuple[Path, Path, tuple[dict[str, object], ...]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("Phase 4 dataset root must be an absolute regular directory")
    manifest, records = root / "manifest.json", root / "records.jsonl"
    if (
        manifest.is_symlink()
        or records.is_symlink()
        or not manifest.is_file()
        or not records.is_file()
        or sha256_path(manifest) != DATASET_MANIFEST_SHA256
        or sha256_path(records) != DATASET_RECORDS_SHA256
    ):
        raise RuntimeError("Phase 4 dataset manifest/records SHA-256 drifted")
    metadata, rows = (
        _read_json(manifest, "dataset manifest"),
        _read_jsonl(records, "dataset records"),
    )
    if (
        metadata.get("record_count") != 8000
        or metadata.get("records_sha256") != DATASET_RECORDS_SHA256
        or len(rows) != 8000
    ):
        raise RuntimeError("Phase 4 dataset structure drifted")
    return manifest, records, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--s6-per-scene", type=Path, default=S6_PER_SCENE)
    parser.add_argument("--s6-summary", type=Path, default=S6_SUMMARY)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 4 source preparation requires explicit --execute.")
        return 2
    try:
        load_phase4_config(arguments.config)
        verify_phase4_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=_LOCKED_PATHS,
        )
        interface_records = _read_jsonl(arguments.s6_per_scene, "S6 per-scene source")
        interface_summary = _read_json(arguments.s6_summary, "S6 summary source")
        dataset_manifest, dataset_records, dataset_rows = _load_dataset(arguments.dataset_root)
        prepared = prepare_legacy_s6_support_sources(
            interface_records=interface_records,
            interface_summary=interface_summary,
            dataset_records=dataset_rows,
            model_snapshot_sha256=MODEL_SNAPSHOT_SHA256,
            expected_scenes=EXPECTED_S6_SCENES,
            symbolic_scene_count=SYMBOLIC_SCENE_COUNT,
            symbolic_seed=SYMBOLIC_SEED,
            value_domain=VALUE_DOMAIN,
            image_grid_thw=IMAGE_GRID_THW,
        )
        paths = write_prepared_support_sources(
            output_root=arguments.output_root,
            prepared=prepared,
            source_hashes={
                "s6_per_scene": sha256_path(arguments.s6_per_scene),
                "s6_summary": sha256_path(arguments.s6_summary),
                "dataset_manifest": sha256_path(dataset_manifest),
                "dataset_records": sha256_path(dataset_records),
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(
        "READY: Phase 4 prepared support sources written; "
        f"symbolic={len(prepared.symbolic_scenes)}, natural={len(prepared.natural_scenes)}"
    )
    for path in (
        paths.symbolic_scenes,
        paths.natural_scenes,
        paths.natural_observations,
        paths.selection_trace,
        paths.summary,
    ):
        print(f"SHA256 {sha256_path(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
