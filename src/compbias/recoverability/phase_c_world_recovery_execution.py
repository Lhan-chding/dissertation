"""Closed package boundaries for Phase-C world-recovery audits."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .phase_c_arm_execution import verify_phase_c_arm_execution_package_lock

PHASE_C_WORLD_RECOVERY_LOCK_PATH = (
    "configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml"
)
PHASE_C_WORLD_RECOVERY_100_LOCK_PATH = (
    "configs/recoverability/server_package_lock_phase_c_world_recovery_100_v1.yaml"
)
_COMMON_PACKAGE_PATHS = frozenset(
    {
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
PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS = _COMMON_PACKAGE_PATHS | {
    "configs/recoverability/phase_c_world_recovery_v1.yaml"
}
PHASE_C_WORLD_RECOVERY_100_PACKAGE_PATHS = _COMMON_PACKAGE_PATHS | {
    "configs/recoverability/phase_c_world_recovery_100_v1.yaml"
}
_LOCK_PACKAGES = {
    PHASE_C_WORLD_RECOVERY_LOCK_PATH: PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS,
    PHASE_C_WORLD_RECOVERY_100_LOCK_PATH: PHASE_C_WORLD_RECOVERY_100_PACKAGE_PATHS,
}


def verify_phase_c_world_recovery_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    root = repository_root.resolve()
    matched = next(
        (
            (relative, package_paths)
            for relative, package_paths in _LOCK_PACKAGES.items()
            if path.resolve() == root / relative
        ),
        None,
    )
    if matched is None or path.is_symlink():
        raise ValueError("Phase C world recovery lock path is not canonical")
    verify_phase_c_arm_execution_package_lock(
        root / "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        repository_root=root,
    )
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    expected = matched[1]
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Phase C world recovery closure mismatch; missing={missing}, extra={extra}"
        )
    return result


__all__ = [
    "PHASE_C_WORLD_RECOVERY_100_LOCK_PATH",
    "PHASE_C_WORLD_RECOVERY_100_PACKAGE_PATHS",
    "PHASE_C_WORLD_RECOVERY_LOCK_PATH",
    "PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS",
    "verify_phase_c_world_recovery_package_lock",
]
