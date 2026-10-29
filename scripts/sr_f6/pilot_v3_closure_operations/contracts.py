"""Exact local/server activation facts for the delivered corrected closure.

These checks bind existing receipts. They do not recompute scientific evidence,
replace the isolated production verifier, or grant pilot/model permission.
"""

import hashlib
import json
import re
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
CLOSURE = "sr_f6_confirmation_closure_v1_20261029"
CLOSURE_OPS = "sr_f6_operations_20261027/CLOSURE_REPAIR_V1_20261029"
REPAIR_REVISION = "1879f8c8f0eb14a69aa65f4ee46557dd446957f5"
REGISTRATION_PIN = "aee0e4d7fc416d9ae8ab0fca4524af8c986187feec49bf87a5867411f21ab39b"
RECEIPT_PIN = "5da462e35256fbfd7b062da118c09099a2c59f06ca859b2ada074459b60fafa3"
PERMISSIONS = ("training_allowed", "generation_allowed", "evaluation_allowed")
EXPECTED_FILES = frozenset(
    [
        f"{CLOSURE}/{name}.json"
        for name in (
            "ASSESSMENT",
            "ATTEMPT",
            "CONTRACT",
            "NUMERICAL",
            "ORIGINAL_RULE_ASSESSMENT",
            "RECEIPT",
            "REGISTRATION",
            "SOURCE_MANIFEST",
        )
    ]
    + [f"{CLOSURE_OPS}/{name}.json" for name in ("CPU_VERIFY_ATTEMPT", "CPU_VERIFY_SUBMISSION")]
)


def require(value, message):
    if not value:
        raise PermissionError(message)


def producer(record):
    revision = record.get("revision")
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "An actual full committed producer revision is required",
    )
    code = PROJECT / ("sr_f6_code_" + revision[:7])
    require(record.get("deployment") == str(code), "Actual canonical producer deployment required")
    require(
        revision != "763172582c4e16a0d9583deecbbe72b0817a4a37",
        "The old producer cannot consume corrected closure",
    )
    return revision, code


def original_references(inputs):
    """Require complete original S0 references before creating any request."""

    def reference(value, path):
        require(
            isinstance(value, dict)
            and set(value) == {"path", "sha256"}
            and value["path"] == path
            and isinstance(value["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
            "Complete original S0 reference required: " + path,
        )

    origin = inputs.get("origin", {})
    require(
        origin.get("phase") == "CENSUS_S0"
        and origin.get("deployment") == str(PROJECT / "sr_f6_code_017b60a"),
        "Original S0 identity required",
    )
    reference(origin.get("identity"), "IDENTITY.json")
    reference(origin.get("source_manifest"), "SOURCE_MANIFEST.json")
    upstream = inputs.get("upstream")
    require(
        isinstance(upstream, list) and len(upstream) == 1, "Complete original S0 upstream required"
    )
    require(
        upstream[0].get("producer_phase") == "CENSUS_S0"
        and upstream[0].get("identity_sha256") == origin["identity"]["sha256"]
        and isinstance(upstream[0].get("evidence"), list)
        and len(upstream[0]["evidence"]) == 1,
        "Original upstream identity differs",
    )
    reference(upstream[0]["evidence"][0], "PREFLIGHT/CENSUS_READY.json")
    parents = inputs.get("parent_references")
    require(
        isinstance(parents, dict) and set(parents) == {"b0", "b1", "b2"},
        "All original parent references required",
    )
    for parent, value in parents.items():
        reference(value, f"runtime_identity/CENSUS-{parent}-0-of-2.json")


def closure_facts(blobs):
    """Validate all ten original delivery members, without evaluating evidence."""
    require(set(blobs) == EXPECTED_FILES, "Exact ten-member corrected closure delivery required")
    pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}
    require(
        pins[f"{CLOSURE}/REGISTRATION.json"] == REGISTRATION_PIN,
        "Actual independently registered repair source required",
    )
    require(
        pins[f"{CLOSURE}/RECEIPT.json"] == RECEIPT_PIN,
        "Actual corrected receipt required; old UNKNOWN cannot authorize",
    )
    values = {name: json.loads(raw) for name, raw in blobs.items()}

    def value(name):
        return values[f"{CLOSURE}/{name}.json"]

    def bound(ref):
        require(
            isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact closure ref required"
        )
        path = Path(ref["path"])
        require(
            path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
            "Closure reference escaped canonical project",
        )
        relative = str(path.relative_to(PROJECT))
        require(relative in pins and pins[relative] == ref["sha256"], "Closure reference changed")
        return values[relative]

    receipt, registration, assessment = value("RECEIPT"), value("REGISTRATION"), value("ASSESSMENT")
    require(
        receipt["schema"] == "srf6-independent-technical-closure-v1"
        and receipt["status"] == "CONFIRMATION_GATES_PASS"
        and receipt["gate_status"] == "CORRECTED_CONFIRMATION_FACTS_SEALED"
        and all(receipt[k] is False for k in PERMISSIONS),
        "Corrected sealed facts required",
    )
    require(
        bound(receipt["assessment"]) == assessment
        and assessment["identity"] == receipt["identity"]
        and assessment["status"] == "CONFIRMATION_GATES_PASS"
        and assessment["old_admission_status"] == "UNKNOWN_NOT_MODIFIED"
        and assessment["new_model_calls"] == assessment["optimizer_updates"] == 0
        and all(assessment[k] is False for k in PERMISSIONS),
        "Complete corrected assessment required",
    )
    identity = receipt["identity"]["repair_verifier"]
    require(
        identity["revision"] == registration["revision"] == REPAIR_REVISION
        and identity["deployment"]
        == registration["deployment"]
        == str(PROJECT / "sr_f6_closure_code_1879f8c")
        and bound(identity["registration"]) == registration
        and identity["source_manifest"] == registration["source_manifest"]
        and identity["contract"] == registration["contract"],
        "Distinct actual repair identity required",
    )
    bound(registration["source_manifest"])
    bound(registration["contract"])
    require(
        registration["status"] == "BOUND_NOT_AUTHORIZED"
        and all(registration[k] is False for k in PERMISSIONS),
        "Registration is not permission",
    )
    original = bound(assessment["original_rule_assessment"])
    numerical = bound(assessment["numerical"])
    canonical = json.dumps(
        numerical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    require(
        original["status"] == "CONFIRMATION_GATES_PASS"
        and original["numerical_sha256"] == hashlib.sha256(canonical).hexdigest()
        and numerical["scope"] == "SERVER_ACTUAL"
        and numerical["status"] == "PASS"
        and set(numerical["parents"]) == {"b0", "b1", "b2"}
        and all(row["status"] == "PASS" for row in numerical["parents"].values())
        and numerical["generated_responses"] == numerical["optimizer_updates"] == 0,
        "Complete original rules and three numerical parents required",
    )
    submission = values[f"{CLOSURE_OPS}/CPU_VERIFY_SUBMISSION.json"]
    require(
        submission["status"] == "SUBMITTED"
        and submission["job_id"] == "233920"
        and submission["requested_gpus"] == 0,
        "Unique actual closure submission required",
    )
    return dict(
        closure_registration_ref=dict(
            path=str(PROJECT / CLOSURE / "REGISTRATION.json"), sha256=REGISTRATION_PIN
        ),
        corrected_confirmation_receipt_ref=dict(
            path=str(PROJECT / CLOSURE / "RECEIPT.json"), sha256=RECEIPT_PIN
        ),
        closure_job_id="233920",
        closure_files={str(PROJECT / name): pin for name, pin in sorted(pins.items())},
    )
