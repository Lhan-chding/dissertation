"""Study C3 config, environment, and frozen input validators."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from compensability_v4.qwen.model_loader import MODEL_SNAPSHOT_SHA256, require_server_model
from compensability_v5.qwen.study_b_backend import (
    require_offline_environment,
    verify_runtime_package_lock,
)

from .factorial_design import validate_study_c3_config
from .io import read_json, sha256_file
from .paths import (
    C2_EVALUATION_MANIFEST,
    C2_EVALUATION_RAW,
    C2_EVALUATION_SUMMARY,
    C2_FIBER_ROWS,
    C2_TRAINING_MANIFEST,
    PACKAGE_LOCK,
)


def load_config(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Study C3 config is missing or unsafe: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Study C3 YAML root must be a mapping")
    return validate_study_c3_config(payload)


def require_offline_snapshot() -> dict[str, object]:  # pragma: no cover - server snapshot
    require_offline_environment()
    package = verify_runtime_package_lock(PACKAGE_LOCK)
    require_server_model()
    return {
        "model_snapshot_sha256": MODEL_SNAPSHOT_SHA256,
        "package_lock_sha256": sha256_file(PACKAGE_LOCK),
        "package_lock": package,
        "gpu_invoked": False,
    }


def require_gpu_runtime() -> dict[str, object]:  # pragma: no cover - server CUDA
    require_offline_environment()
    package = verify_runtime_package_lock(PACKAGE_LOCK)
    require_server_model()
    return {
        "model_snapshot_sha256": MODEL_SNAPSHOT_SHA256,
        "package_lock_sha256": sha256_file(PACKAGE_LOCK),
        "package_lock": package,
    }


def validate_c2_evaluation_sources() -> dict[str, object]:
    manifest = read_json(C2_EVALUATION_MANIFEST)
    summary = read_json(C2_EVALUATION_SUMMARY)
    if (
        manifest.get("status") != "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE"
        or summary.get("status") != "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE"
        or manifest.get("raw_row_count") != 5632
        or summary.get("raw_row_count") != 5632
        or manifest.get("evaluation_scene_count") != 176
        or manifest.get("evaluation_pair_count") != 88
        or manifest.get("sampled_rollouts") != 16
        or manifest.get("raw_rows_sha256") != sha256_file(C2_EVALUATION_RAW)
        or manifest.get("summary_sha256") != sha256_file(C2_EVALUATION_SUMMARY)
        or manifest.get("fiber_rows_sha256") != sha256_file(C2_FIBER_ROWS)
        or manifest.get("training_pair_manifest_sha256") != sha256_file(C2_TRAINING_MANIFEST)
        or manifest.get("model_snapshot_sha256") != MODEL_SNAPSHOT_SHA256
    ):
        raise ValueError("frozen Study C2 evaluation evidence drifted")
    return {
        "raw_rows_sha256": manifest["raw_rows_sha256"],
        "summary_sha256": manifest["summary_sha256"],
        "manifest_sha256": sha256_file(C2_EVALUATION_MANIFEST),
        "fiber_rows_sha256": manifest["fiber_rows_sha256"],
        "training_pair_manifest_sha256": manifest["training_pair_manifest_sha256"],
        "rollout_seed_base": summary["rollout_seed_base"],
        "rollout_seed_algorithm": summary["rollout_seed_algorithm"],
    }


def select_evaluation_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    selected = tuple(
        dict(row) for row in rows if row.get("split") in {"dev", "test", "positive_control"}
    )
    if len(selected) != 176:
        raise ValueError(f"Study C3 requires 176 evaluation scenes, observed {len(selected)}")
    pairs: dict[str, set[str]] = {}
    scenes: set[str] = set()
    for row in selected:
        pair_id = row.get("pair_id")
        scene_id = row.get("scene_id")
        condition = row.get("condition")
        if (
            not isinstance(pair_id, str)
            or not isinstance(scene_id, str)
            or scene_id in scenes
            or condition not in {"collision", "separating"}
        ):
            raise ValueError("Study C3 evaluation rows are malformed or duplicated")
        scenes.add(scene_id)
        pairs.setdefault(pair_id, set()).add(str(condition))
    if len(pairs) != 88 or any(values != {"collision", "separating"} for values in pairs.values()):
        raise ValueError("Study C3 evaluation rows are not 88 complete matched pairs")
    return selected


def b3_defaults() -> tuple[Path, str]:
    root = Path(os.environ.get("V5_B_ROOT", "artifacts/v5/study_b/pilot-2026082201"))
    adapter = Path(os.environ.get("V5_B3_ADAPTER", str(root / "arms/B3/final_adapter")))
    expected = os.environ.get(
        "V5_EXPECTED_B3_SHA",
        "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
    )
    return adapter, expected


__all__ = [
    "b3_defaults",
    "load_config",
    "require_gpu_runtime",
    "require_offline_snapshot",
    "select_evaluation_rows",
    "validate_c2_evaluation_sources",
]
