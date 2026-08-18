"""Build hash-bound Phase 4 C0/C1/T support corpora without loading the model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compensability_v4.training.phase4 import (  # noqa: E402
    build_support_sets,
    load_phase4_config,
    load_support_sources,
    sha256_path,
    verify_phase4_package_lock,
    write_support_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_4.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_4.yaml"
_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_4.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/07_build_support_data.py",
    "scripts/v4/08_train_phase4_lora.py",
    "src/compensability_v4/training/__init__.py",
    "src/compensability_v4/training/phase4.py",
)


def _require_hash(path: Path, expected: str, label: str) -> str:
    if (
        path.is_symlink()
        or not path.is_file()
        or len(expected) != 64
        or sha256_path(path) != expected
    ):
        raise RuntimeError(f"Phase 4 {label} SHA-256 mismatch")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--symbolic-scenes", type=Path, required=True)
    parser.add_argument("--symbolic-scenes-sha256", required=True)
    parser.add_argument("--natural-scenes", type=Path, required=True)
    parser.add_argument("--natural-scenes-sha256", required=True)
    parser.add_argument("--natural-observations", type=Path, required=True)
    parser.add_argument("--natural-observations-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/v4/training/support.jsonl")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts/v4/training/support_summary.json"
    )
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 4 support-data build requires explicit --execute.")
        return 2
    try:
        load_phase4_config(arguments.config)
        verify_phase4_package_lock(
            lock_path=arguments.package_lock, repository_root=ROOT, expected_paths=_LOCKED_PATHS
        )
        source_hashes = {
            "symbolic_scenes": _require_hash(
                arguments.symbolic_scenes, arguments.symbolic_scenes_sha256, "symbolic scenes"
            ),
            "natural_scenes": _require_hash(
                arguments.natural_scenes, arguments.natural_scenes_sha256, "natural scenes"
            ),
            "natural_observations": _require_hash(
                arguments.natural_observations,
                arguments.natural_observations_sha256,
                "natural observations",
            ),
        }
        symbolic, natural, errors = load_support_sources(
            symbolic_scenes_path=arguments.symbolic_scenes,
            natural_scenes_path=arguments.natural_scenes,
            natural_observations_path=arguments.natural_observations,
        )
        write_support_artifact(
            output_path=arguments.output,
            summary_path=arguments.summary,
            support_sets=build_support_sets(
                symbolic_scenes=symbolic, natural_scenes=natural, natural_errors=errors
            ),
            source_hashes=source_hashes,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 4 support data written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
