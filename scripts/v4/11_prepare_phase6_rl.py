"""Prepare the frozen Phase 6 RL execution manifest from Phase 4/5 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compensability_v4.qwen.phase6_runtime import (  # noqa: E402
    PHASE6_LOCKED_PATHS,
    build_phase6_execution_manifest,
    load_phase5_policy_support_summary,
    load_phase6_config,
    verify_phase6_package_lock,
    write_phase6_execution_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CONFIG = ROOT / "configs/recoverability/v4_phase_6.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml"
POLICY_SUPPORT = ROOT / "artifacts/v4/support/informative_group_rate.json"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
OUTPUT = ROOT / "artifacts/v4/phase6/execution_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--policy-support-summary", type=Path, default=POLICY_SUPPORT)
    parser.add_argument("--policy-support-summary-sha256")
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--output-path", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 6 RL manifest preparation requires explicit --execute.")
        return 2
    if not arguments.policy_support_summary_sha256:
        print("BLOCKED: Phase 6 requires --policy-support-summary-sha256.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        config = load_phase6_config(arguments.config)
        lock_hash = verify_phase6_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE6_LOCKED_PATHS,
        )
        summary = load_phase5_policy_support_summary(
            arguments.policy_support_summary,
            expected_sha256=arguments.policy_support_summary_sha256,
        )
        manifest = build_phase6_execution_manifest(
            config=config,
            phase5_summary=summary,
            phase5_summary_sha256=arguments.policy_support_summary_sha256,
            phase4_run_root=arguments.phase4_run_root,
            config_sha256=_sha256(arguments.config),
            package_lock_sha256=lock_hash,
        )
        write_phase6_execution_manifest(arguments.output_path, manifest)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 6 RL execution manifest written to {arguments.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
