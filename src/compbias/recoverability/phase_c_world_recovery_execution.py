"""Closed package boundary for the 12-call Phase-C world-recovery audit."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .phase_c_arm_execution import verify_phase_c_arm_execution_package_lock

PHASE_C_WORLD_RECOVERY_LOCK_PATH = (
    "configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml"
)
PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS = frozenset(
    {
        "configs/recoverability/phase_c_world_recovery_v1.yaml",
        "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        "experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py",
        "experiments/recoverability_v1/21_run_phase_c_world_recovery.py",
        "prompts/no_cue.user.template.txt",
        "prompts/valid_cue.user.template.txt",
        "prompts/world_recovery_v1_ablation_no_examples.system.txt",
        "prompts/world_recovery_v1_main.system.txt",
        "src/compbias/recoverability/phase_c_world_recovery.py",
        "src/compbias/recoverability/phase_c_world_recovery_execution.py",
    }
)


def verify_phase_c_world_recovery_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    root = repository_root.resolve()
    canonical = root / PHASE_C_WORLD_RECOVERY_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Phase C world recovery lock path is not canonical")
    verify_phase_c_arm_execution_package_lock(
        root / "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        repository_root=root,
    )
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS:
        missing = sorted(PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS - observed)
        extra = sorted(observed - PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS)
        raise ValueError(
            f"Phase C world recovery closure mismatch; missing={missing}, extra={extra}"
        )
    return result


__all__ = [
    "PHASE_C_WORLD_RECOVERY_LOCK_PATH",
    "PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS",
    "verify_phase_c_world_recovery_package_lock",
]
