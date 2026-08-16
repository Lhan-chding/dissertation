"""Closed server-package boundary for Phase-C v3 six-arm execution."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .phase_c_screen_execution import PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS

PHASE_C_ARM_EXECUTION_LOCK_PATH = (
    "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml"
)
PHASE_C_ARM_EXECUTION_PACKAGE_PATHS = PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml",
        "configs/recoverability/server_package_lock_phase_c_screen_v2.yaml",
        "experiments/recoverability_v1/16_phase_c_arm_preflight.py",
        "experiments/recoverability_v1/17_run_phase_c_arms.py",
        "src/compbias/recoverability/paired_effects.py",
        "src/compbias/recoverability/phase_c_arm_execution.py",
        "src/compbias/recoverability/phase_c_arms.py",
        "src/compbias/recoverability/phase_c_postscreen_amendment.py",
        "src/compbias/recoverability/phase_c_screen_result.py",
        "src/compbias/recoverability/power.py",
    }
)


def verify_phase_c_arm_execution_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    root = repository_root.resolve()
    canonical = root / PHASE_C_ARM_EXECUTION_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Phase C arm execution lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != PHASE_C_ARM_EXECUTION_PACKAGE_PATHS:
        missing = sorted(PHASE_C_ARM_EXECUTION_PACKAGE_PATHS - observed)
        extra = sorted(observed - PHASE_C_ARM_EXECUTION_PACKAGE_PATHS)
        raise ValueError(
            f"Phase C arm execution closure mismatch; missing={missing}, extra={extra}"
        )
    return result
