"""Closed pilot phase routing; a profile never grants model access.

V1 keeps its historical measurement chain. V3 consumes a separately verified
confirmation and immutable public-input overlay. Actual source, runtime and
ledger checks remain the responsibility of their existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PilotContract:
    phase: str
    registration_directory: str
    recovery_directory: str
    evaluation_directory: str
    jobs_directory: str
    audit_directory: str
    admission_path: str
    admission_scope: str
    spec_file: str = "PILOT_SPEC_FROZEN.json"


def for_phase(phase):
    if type(phase) is not str or phase not in ("PILOT_V1", "PILOT_V3"):
        raise PermissionError("Only an explicit registered pilot phase is supported")
    prefix = f"PHASE_RUNS/{phase}"
    return PilotContract(
        phase=phase,
        registration_directory=f"{prefix}/PILOT_FROZEN",
        recovery_directory=f"{prefix}/RECOVERY/b0/r0/U11",
        evaluation_directory=f"{prefix}/EVALUATION",
        jobs_directory=f"{prefix}/JOBS",
        audit_directory=f"{prefix}/RUNTIME_PILOT_AUDIT",
        admission_path=f"PHASES/{phase}/ADMISSION.json",
        admission_scope=f"{phase}_REGISTERED_ONLY",
    )


def verify_registration(root, *, phase_context, pilot_spec_sha256):
    contract = for_phase(phase_context.get("phase"))
    if contract.phase == "PILOT_V1":
        from .registration import verify_pilot_registration

        return verify_pilot_registration(
            root, phase_context=phase_context, expected_spec_sha256=pilot_spec_sha256
        )
    from .pilot_v3_registration import verify_pilot_v3_registration

    return verify_pilot_v3_registration(
        root, phase_context=phase_context, expected_spec_sha256=pilot_spec_sha256
    )


def require_admission(root, *, phase, phase_identity_sha256, pilot_spec_sha256, admission_ref):
    contract = for_phase(phase)
    if contract.phase == "PILOT_V1":
        from .admission import require_pilot_admission

        return require_pilot_admission(
            root,
            phase_identity_sha256=phase_identity_sha256,
            pilot_spec_sha256=pilot_spec_sha256,
            admission_ref=admission_ref,
        )
    from .pilot_v3_admission import require_pilot_v3_admission

    return require_pilot_v3_admission(
        root,
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
    )
