"""Closed server package boundary for one-shot Phase-N execution."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .measurement_qualification_execution import (
    MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS,
)

PHASE_N_EXECUTION_LOCK_PATH = "configs/recoverability/server_package_lock_phase_n.yaml"
PHASE_N_EXECUTION_PACKAGE_PATHS = MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/measurement_qualification_frozen_result.yaml",
        "configs/recoverability/server_package_lock_measurement_qualification.yaml",
        "experiments/recoverability_v1/12_phase_n_preflight.py",
        "experiments/recoverability_v1/13_run_phase_n.py",
        "src/compbias/recoverability/measurement_qualification_result.py",
        "src/compbias/recoverability/natural_inference.py",
        "src/compbias/recoverability/phase_n.py",
        "src/compbias/recoverability/phase_n_execution.py",
    }
)


def verify_phase_n_execution_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the canonical Phase-N code and configuration closure."""

    root = repository_root.resolve()
    canonical = root / PHASE_N_EXECUTION_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Phase N execution lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != PHASE_N_EXECUTION_PACKAGE_PATHS:
        missing = sorted(PHASE_N_EXECUTION_PACKAGE_PATHS - observed)
        extra = sorted(observed - PHASE_N_EXECUTION_PACKAGE_PATHS)
        raise ValueError(f"Phase N execution closure mismatch; missing={missing}, extra={extra}")
    return result
