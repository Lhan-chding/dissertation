"""Fail-closed server package boundary for qualification dataset generation."""

from __future__ import annotations

from pathlib import Path

from .evidence import ProtocolLockResult, verify_protocol_lock
from .stage2_v2_evidence import STAGE2_V2_EVIDENCE_PACKAGE_PATHS

MEASUREMENT_QUALIFICATION_DATA_LOCK_PATH = (
    "configs/recoverability/server_package_lock_measurement_qualification_data.yaml"
)
MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS = STAGE2_V2_EVIDENCE_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/measurement_qualification_v1.yaml",
        "configs/recoverability/server_package_lock_stage2_v2_evidence.yaml",
        "configs/recoverability/stage2_v2_external_evidence_anchor.yaml",
        "experiments/recoverability_v1/09_generate_measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification.py",
        "src/compbias/recoverability/measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification_server.py",
        "src/compbias/recoverability/stage2_v2_anchor.py",
    }
)


def verify_measurement_qualification_data_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the canonical model-free qualification generation closure."""

    root = repository_root.resolve()
    canonical = root / MEASUREMENT_QUALIFICATION_DATA_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("measurement qualification data package lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS:
        missing = sorted(MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS - observed)
        extra = sorted(observed - MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS)
        raise ValueError(
            "measurement qualification data package lock closure mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return result
