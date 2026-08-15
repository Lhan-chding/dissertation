"""Closed server-package boundary for the one-shot Phase-C v2 screen."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .phase_n_execution import PHASE_N_EXECUTION_PACKAGE_PATHS

PHASE_C_SCREEN_EXECUTION_LOCK_PATH = (
    "configs/recoverability/server_package_lock_phase_c_screen_v2.yaml"
)
PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS = PHASE_N_EXECUTION_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/phase_n_frozen_result.yaml",
        "configs/recoverability/recoverability_phase_c_v2_amendment.yaml",
        "configs/recoverability/server_package_lock_phase_n.yaml",
        "experiments/recoverability_v1/14_phase_c_screen_preflight.py",
        "experiments/recoverability_v1/15_run_phase_c_screen.py",
        "src/compbias/recoverability/compatibility.py",
        "src/compbias/recoverability/operators.py",
        "src/compbias/recoverability/phase_c_amendment.py",
        "src/compbias/recoverability/phase_c_screen.py",
        "src/compbias/recoverability/phase_c_screen_execution.py",
        "src/compbias/recoverability/phase_n_result.py",
        "src/compbias/recoverability/selection.py",
    }
)


def verify_phase_c_screen_execution_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the exact prospective Phase-C screen code and configuration closure."""

    root = repository_root.resolve()
    canonical = root / PHASE_C_SCREEN_EXECUTION_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Phase C screen execution lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS:
        missing = sorted(PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS - observed)
        extra = sorted(observed - PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS)
        raise ValueError(
            f"Phase C screen execution closure mismatch; missing={missing}, extra={extra}"
        )
    return result
