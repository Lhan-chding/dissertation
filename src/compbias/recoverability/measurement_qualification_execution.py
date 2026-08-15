"""Closed server package boundary for measurement qualification execution."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .measurement_qualification_server import MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS

MEASUREMENT_QUALIFICATION_EXECUTION_LOCK_PATH = (
    "configs/recoverability/server_package_lock_measurement_qualification.yaml"
)
MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS = (
    MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS
    | frozenset(
        {
            "configs/recoverability/measurement_qualification_data_anchor.yaml",
            "configs/recoverability/server_package_lock_measurement_qualification_data.yaml",
            "experiments/recoverability_v1/10_measurement_qualification_preflight.py",
            "experiments/recoverability_v1/11_run_measurement_qualification.py",
            "src/compbias/recoverability/measurement_qualification_anchor.py",
            "src/compbias/recoverability/measurement_qualification_execution.py",
        }
    )
)


def verify_measurement_qualification_execution_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the exact preflight and one-shot execution code closure."""

    root = repository_root.resolve()
    canonical = root / MEASUREMENT_QUALIFICATION_EXECUTION_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("measurement qualification execution lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS:
        missing = sorted(MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS - observed)
        extra = sorted(observed - MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS)
        raise ValueError(
            "measurement qualification execution closure mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return result
