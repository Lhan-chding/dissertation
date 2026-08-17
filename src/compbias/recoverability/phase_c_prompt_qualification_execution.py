"""Closed package boundary for the 36-call Phase-C prompt qualification."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .phase_c_arm_execution import verify_phase_c_arm_execution_package_lock

PHASE_C_PROMPT_QUALIFICATION_LOCK_PATH = (
    "configs/recoverability/server_package_lock_phase_c_prompt_qualification_v1.yaml"
)
PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS = frozenset(
    {
        "configs/recoverability/phase_c_prompt_qualification_v1.yaml",
        "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        "experiments/recoverability_v1/18_phase_c_prompt_qualification_preflight.py",
        "experiments/recoverability_v1/19_run_phase_c_prompt_qualification.py",
        "src/compbias/recoverability/phase_c_prompt_qualification.py",
        "src/compbias/recoverability/phase_c_prompt_qualification_execution.py",
    }
)


def verify_phase_c_prompt_qualification_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    root = repository_root.resolve()
    canonical = root / PHASE_C_PROMPT_QUALIFICATION_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Phase C prompt qualification lock path is not canonical")
    verify_phase_c_arm_execution_package_lock(
        root / "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        repository_root=root,
    )
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS:
        missing = sorted(PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS - observed)
        extra = sorted(observed - PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS)
        raise ValueError(
            "Phase C prompt qualification closure mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return result


__all__ = [
    "PHASE_C_PROMPT_QUALIFICATION_LOCK_PATH",
    "PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS",
    "verify_phase_c_prompt_qualification_package_lock",
]
