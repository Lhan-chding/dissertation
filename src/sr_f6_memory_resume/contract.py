"""Exact v8 execution envelope; metadata validation never imports tensor runtimes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from sr_f6_memory_repair import execution as legacy
from sr_f6_memory_repair import full_contract as full

SCHEMA = "srf6-memory-resume-contract-v8"
PROJECT = legacy.PROJECT
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_RESUME_V8_20261030"
FULL_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V7_20261030"
FULL_REVISION = "4be18ca1c9bd2db0b905b0d3c3f22873f97cc081"
FULL_DEPLOYMENT = PROJECT / "sr_f6_memory_full_code_4be18ca"
FULL_BINDING = "797bc640bea90b03f21a4952196df1d18ef196459aaae481d0b740482787fe02"
FULL_CONTRACT_SHA = "e5e14a0ce0036b9b640c930d01fc9f34ce68a7f48f0303eebb44d323b0e28cd9"
FULL_RESULT_SHA = "2f6c641d2dd84b1691459eccca766e02ffbd4bd1c6af6047986e843edd2e34e7"
FULL_COMPLETE_SHA = "29a4204f3e7a3d02de0b1486cddaa4d95768d90f7f9b95f1eb51287afba76f4e"
FULL_COLD_SHA = "376448405f87ef45c09ab3a91e15e871da756ffe07b91c24c18758c7a1bd7108"
QUEUE = legacy.QUEUE
MEMBERS = (
    "src/sr_f6_memory_resume/__init__.py",
    "src/sr_f6_memory_resume/contract.py",
    "src/sr_f6_memory_resume/execution.py",
    "scripts/sr_f6_memory_resume_ops/operations.py",
    "scripts/sr_f6_memory_resume_ops/launch.sbatch",
)
require = legacy.require


def predecessor_refs():
    return {
        "full_contract_ref": dict(path=str(FULL_OPS / "CONTRACT.json"), sha256=FULL_CONTRACT_SHA),
        "full_result_ref": dict(
            path=str(FULL_OPS / "VALIDATION/RESULT.json"), sha256=FULL_RESULT_SHA
        ),
        "full_complete_ref": dict(
            path=str(FULL_OPS / "validate_COMPLETE.json"), sha256=FULL_COMPLETE_SHA
        ),
        "full_cold_ref": dict(
            path=str(FULL_OPS / "VALIDATION/COLD_VERIFICATION.json"), sha256=FULL_COLD_SHA
        ),
    }


def _read(ref):
    path = legacy.ref_path(PROJECT, ref)
    require(path.stat().st_size <= 16 * 1024**2, "Bounded resume metadata required")
    return legacy._json(legacy.strict_ref(path, ref["sha256"]))


def _envelope(envelope):
    require(isinstance(envelope, dict), "Resume envelope must be an object")
    require(
        set(envelope) == {"schema", "execution_source", "queue"} | set(predecessor_refs()),
        "Exact resume envelope fields required; scientific overrides forbidden",
    )
    require(envelope["schema"] == SCHEMA, "Resume schema differs")
    require(envelope["queue"] == list(QUEUE), "Original fixed b1 queue required")
    for key, ref in predecessor_refs().items():
        require(envelope[key] == ref, "Exact completed v7 reference required: " + key)


def _sources(envelope):
    source = envelope["execution_source"]
    require(
        isinstance(source, dict) and set(source) == {"revision", "deployment", "files"},
        "Exact execution source identity required",
    )
    revision = source["revision"]
    require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        and revision not in {FULL_REVISION, legacy.PRODUCER_REVISION},
        "Separate committed recovery source required",
    )
    deployment = legacy.plain_path(source["deployment"])
    require(
        deployment == PROJECT / ("sr_f6_memory_resume_code_" + revision[:7]),
        "Separate exact recovery deployment required",
    )
    require(isinstance(source["files"], list), "Execution source files must be a list")
    listed = {}
    for ref in source["files"]:
        path = legacy.ref_path(deployment, ref)
        name = ref["path"]
        require(
            name in MEMBERS and name not in listed and path == deployment / name,
            "Exact unique recovery source paths required",
        )
        listed[name] = ref["sha256"]
        legacy.strict_ref(path, ref["sha256"])
    require(set(listed) == set(MEMBERS), "Complete independent source closure required")
    manifest_path = legacy.plain_path(deployment / "RESUME_SOURCE.json")
    require(manifest_path.stat().st_size <= 1024**2, "Bounded source manifest required")
    manifest = legacy._json(manifest_path.read_bytes())
    require(
        manifest.get("schema") == "srf6-memory-resume-source-v1"
        and manifest.get("revision") == revision
        and manifest.get("deployment") == str(deployment)
        and manifest.get("producer_revision") == legacy.PRODUCER_REVISION
        and set(manifest.get("files", {})) == set(MEMBERS),
        "Recovery manifest differs",
    )
    for name, pin in manifest["files"].items():
        require(
            set(pin) == {"bytes", "sha256"}
            and type(pin["bytes"]) is int
            and pin["bytes"] == (deployment / name).stat().st_size
            and pin["sha256"] == listed[name],
            "Recovery manifest/source binding differs",
        )
    import sr_f6_memory_resume

    from . import execution

    require(
        Path(__file__).resolve() == deployment / MEMBERS[1]
        and Path(execution.__file__).resolve() == deployment / MEMBERS[2]
        and Path(sr_f6_memory_resume.__file__).resolve() == deployment / MEMBERS[0],
        "Recovery helpers loaded from another deployment",
    )
    require(
        Path(full.__file__).resolve()
        == FULL_DEPLOYMENT / "src/sr_f6_memory_repair/full_contract.py"
        and Path(legacy.__file__).resolve()
        == FULL_DEPLOYMENT / "src/sr_f6_memory_repair/execution.py",
        "Frozen full-validation helpers must be imported from actual v7 deployment",
    )
    require(
        hashlib.sha256((FULL_DEPLOYMENT / "FULL_SOURCE.json").read_bytes()).hexdigest()
        == FULL_BINDING,
        "Actual v7 deployment manifest changed",
    )


def _completed(envelope):
    checked = full.verify_validation_metadata(
        envelope["full_contract_ref"]["path"],
        envelope["full_contract_ref"]["sha256"],
        envelope["full_result_ref"],
    )
    complete = _read(envelope["full_complete_ref"])
    require(
        complete.get("status") == "COMPLETED_FULL_VALIDATION_NO_TRAINING"
        and complete.get("job_id") == "235864"
        and complete.get("contract_ref") == envelope["full_contract_ref"]
        and complete.get("result_ref") == envelope["full_result_ref"]
        and complete.get("cold_verification_ref") == envelope["full_cold_ref"]
        and complete.get("source")
        == dict(
            revision=FULL_REVISION, deployment=str(FULL_DEPLOYMENT), binding_sha256=FULL_BINDING
        ),
        "Actual v7 outer completion binding differs",
    )
    require(
        checked.get("status") == full.STATUS
        and checked.get("cold_verification_ref") == envelope["full_cold_ref"]
        and checked["contract"]["repair_source"]["revision"] == FULL_REVISION
        and checked["contract"]["repair_source"]["deployment"] == str(FULL_DEPLOYMENT),
        "Actual completed v7 storage validation required",
    )
    return checked["contract"]


def validate_metadata(path, expected_sha):
    path = legacy.plain_path(path)
    require(
        path in (OPS / "CONTRACT.json", OPS / "PREFLIGHT_CONTRACT.json"),
        "Unique v8 contract or metadata preflight required",
    )
    envelope = _read(dict(path=str(path), sha256=expected_sha))
    _envelope(envelope)
    _sources(envelope)
    _completed(envelope)
    return envelope


def full_contract(envelope):
    """Return authentic v7 scientific pins, never a v8 override or copied recipe."""
    _envelope(envelope)
    return _completed(envelope)


def verify_contract(path, expected_sha):
    require(
        legacy.plain_path(path) == OPS / "CONTRACT.json", "Execution requires final v8 contract"
    )
    envelope = validate_metadata(path, expected_sha)
    ref = envelope["full_contract_ref"]
    # Original step-000/READY step-001 and the failed-audit prefix are immutable.
    # Later producer checkpoints/audit appends are checked by pilot_jobs, not
    # mistaken for corruption of these historical inputs on queue tasks 2 and 3.
    full.verify_contract(ref["path"], ref["sha256"])
    return envelope


def verify_validation(envelope):
    """Cold-read complete original tensors in the allocated GPU child process."""
    from sr_f6_memory_repair import validation_full

    _envelope(envelope)
    _completed(envelope)
    ref = envelope["full_contract_ref"]
    return validation_full.verify_validation(
        ref["path"], ref["sha256"], envelope["full_result_ref"]
    )
