"""Round B source and completed A bindings; bounded metadata never loads tensors."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sr_f6_memory_resume import contract as resume

PROJECT = resume.PROJECT
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_ROUND_B_20261031"
V8_DEPLOYMENT = PROJECT / "sr_f6_memory_resume_code_e4d2b0c"
V8_REVISION = "e4d2b0ca4155b489a21f0b3cd2696b6860df7626"
V8_BINDING = "0b3df1de667a6b172e40e03646eeec329f4c9e1d34862d5ee4bc4e335458618a"
V8_CONTRACT_SHA = "6ae70ea969803febcb8f91c4e2ad6da78c9e901abb30769bd109ddda8e608957"
V8_COMPLETE_SHA = "11ebc9105631ad5836fd1c0c8933bec48e46ed9f7d3fb21b5fcc473482e670cb"
A_AUDIT_SHA = "5d59cde59c8dbca4c0144ab622876a2008b40ca097ab6f8ba97dc3305802fd9d"
SCHEMA = "srf6-round-b-contract-v1"
SOURCE_SCHEMA = "srf6-round-b-source-v1"
MEMBERS = (
    "src/sr_f6_round_b/__init__.py",
    "src/sr_f6_round_b/contract.py",
    "src/sr_f6_round_b/execution.py",
    "scripts/sr_f6_round_b_ops/operations.py",
    "scripts/sr_f6_round_b_ops/launch.sbatch",
)
QUEUES = [["PRIMARY-b2-U10", "PRIMARY-b2-U11", "PRIMARY-b2-C_SHUF"], ["RECOVERY-b0-U11"]]
TASKS = tuple(task for queue in QUEUES for task in queue)
A_TASKS = {f"PRIMARY-{parent}-{arm}" for parent in ("b0", "b1") for arm in ("U10", "U11", "C_SHUF")}
require = resume.require
read_ref = resume._read


def require_untouched_primary(directory):
    directory = resume.legacy.plain_path(directory)
    require(directory.is_dir(), "Original registered primary directory required")
    names = {p.name for p in directory.iterdir()}
    require(
        {"RUN_IDENTITY.json", "ROLLOUT_REGISTRATION.json"} <= names
        and names <= {"RUN_IDENTITY.json", "ROLLOUT_REGISTRATION.json", ".run.lock"},
        "Prior primary output forbids retry",
    )
    for name in names:
        path = resume.legacy.plain_path(directory / name)
        require(
            path.is_file() and (name != ".run.lock" or path.stat().st_size == 0),
            "Unexpected primary artifact",
        )


def require_untouched_recovery(target):
    target = resume.legacy.plain_path(target)
    require(
        not target.exists() and not any(target.parent.glob(".recovery-staging-*")),
        "Prior recovery output or staging forbids retry",
    )


def predecessor_refs():
    return dict(
        resume_contract_ref=dict(path=str(resume.OPS / "CONTRACT.json"), sha256=V8_CONTRACT_SHA),
        resume_complete_ref=dict(
            path=str(resume.OPS / "resume_COMPLETE.json"), sha256=V8_COMPLETE_SHA
        ),
        a_delivery_ref=dict(
            path=str(OPS / "A_EVIDENCE/A_DELIVERY_AUDIT_20261031.json"), sha256=A_AUDIT_SHA
        ),
    )


def envelope(source):
    """Source must be the actual subsequently committed and deployed source closure."""
    return dict(schema=SCHEMA, execution_source=source, queues=QUEUES, **predecessor_refs())


def shape(value):
    require(
        isinstance(value, dict)
        and set(value) == {"schema", "execution_source", "queues"} | set(predecessor_refs()),
        "Exact B envelope required; scientific overrides forbidden",
    )
    require(
        value["schema"] == SCHEMA and value["queues"] == QUEUES, "Only frozen B queues authorized"
    )
    for key, ref in predecessor_refs().items():
        require(value[key] == ref, "Changed completed A predecessor: " + key)


def sources(source):
    require(
        isinstance(source, dict) and set(source) == {"revision", "deployment", "files"},
        "Exact new source identity required",
    )
    revision = source["revision"]
    require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        and revision not in (V8_REVISION, resume.FULL_REVISION, resume.legacy.PRODUCER_REVISION),
        "Separate committed B revision required",
    )
    deployment = resume.legacy.plain_path(source["deployment"])
    require(
        deployment == PROJECT / ("sr_f6_round_b_code_" + revision[:7]),
        "Exact independent B deployment required",
    )
    require(isinstance(source["files"], list), "Source references must be a list")
    listed = {}
    for ref in source["files"]:
        path = resume.legacy.ref_path(deployment, ref)
        require(
            ref["path"] in MEMBERS
            and ref["path"] not in listed
            and path == deployment / ref["path"],
            "Unique B source closure required",
        )
        require(path.stat().st_size <= 2**20, "Bounded source required")
        resume.legacy.strict_ref(path, ref["sha256"])
        listed[ref["path"]] = ref["sha256"]
    require(set(listed) == set(MEMBERS), "Complete B source closure required")
    manifest_path = resume.legacy.plain_path(deployment / "ROUND_B_SOURCE.json")
    require(manifest_path.stat().st_size <= 2**20, "Bounded B manifest required")
    manifest = json.loads(manifest_path.read_bytes())
    require(
        set(manifest) == {"schema", "revision", "deployment", "producer_revision", "files"}
        and manifest["schema"] == SOURCE_SCHEMA
        and manifest["revision"] == revision
        and manifest["deployment"] == str(deployment)
        and manifest["producer_revision"] == resume.legacy.PRODUCER_REVISION
        and set(manifest["files"]) == set(MEMBERS),
        "B manifest identity differs",
    )
    for name, pin in manifest["files"].items():
        require(
            set(pin) == {"bytes", "sha256"}
            and type(pin["bytes"]) is int
            and pin["bytes"] == (deployment / name).stat().st_size
            and pin["sha256"] == listed[name],
            "B source byte binding differs",
        )
    import sr_f6_round_b

    from . import execution

    for module, name in ((sr_f6_round_b, MEMBERS[0]), (execution, MEMBERS[2])):
        require(
            Path(module.__file__).resolve() == deployment / name,
            "B helper loaded from another deployment",
        )
    require(Path(__file__).resolve() == deployment / MEMBERS[1], "Wrong B verifier origin")


def completed_a(value):
    """Use byte-pinned existing delivery reports, never reclassify them as a scientific gate."""
    shape(value)
    require(
        Path(resume.__file__).resolve() == V8_DEPLOYMENT / "src/sr_f6_memory_resume/contract.py",
        "Actual frozen v8 verifier required",
    )
    require(
        hashlib.sha256((V8_DEPLOYMENT / "RESUME_SOURCE.json").read_bytes()).hexdigest()
        == V8_BINDING,
        "Frozen v8 source manifest changed",
    )
    ref = value["resume_contract_ref"]
    previous = resume.validate_metadata(ref["path"], ref["sha256"])
    scientific = resume.full_contract(previous)
    require(
        scientific["experiment_root"] == str(ROOT)
        and scientific["storage_backend"] == "arena_v3"
        and scientific["max_device_bytes"] == 44 * 2**30
        and scientific["max_host_bytes"] == 8 * 2**30
        and scientific["max_disk_bytes"] == 96 * 2**30,
        "Actual validated storage contract required",
    )
    complete = read_ref(value["resume_complete_ref"])
    require(
        complete.get("status") == "QUEUE_COMPLETE_NOT_GATE_SIGNED"
        and complete.get("job_id") == "235953"
        and complete.get("contract_ref") == ref
        and complete.get("queue") == list(resume.QUEUE)
        and complete.get("scientific_gate_status") == "NOT_SIGNED",
        "Actual completed v8 queue required",
    )
    require(not (resume.OPS / "resume_FAILURE.json").exists(), "Failed A cannot admit B")
    audit = read_ref(value["a_delivery_ref"])
    require(
        audit.get("status") == "VERIFIED_A_SIX_TASK_DELIVERY_NOT_SCIENTIFIC_GATE"
        and audit.get("resume_complete_ref") == value["resume_complete_ref"]
        and audit.get("verified_checkpoint_files") == 78
        and audit.get("verified_updates") == 72
        and audit.get("audit_generated_responses") == 0
        and audit.get("audit_optimizer_updates") == 0
        and audit.get("scientific_gate") == "NOT_SIGNED",
        "Actual six task delivery audit required",
    )
    require(
        len(audit["tasks"]) == 6 and {t["task_id"] for t in audit["tasks"]} == A_TASKS,
        "Exactly six original A task audits required",
    )
    for item in audit["tasks"]:
        original = item["audit_ref"]
        audit_ref = dict(
            path=str(OPS / "A_EVIDENCE" / Path(original["path"]).name), sha256=original["sha256"]
        )
        wrapper = read_ref(audit_ref)
        require(
            wrapper.get("returncode") == 0 and wrapper.get("task") == item["task_id"],
            "Failed A audit cannot admit B",
        )
        checked = json.loads(wrapper["stdout"])
        require(
            checked.get("status") == "VERIFIED_A_TASK_DELIVERY_NOT_SCIENTIFIC_GATE"
            and checked["task"]["task_id"] == item["task_id"],
            "Wrong A task evidence",
        )
        require(
            checked.get("torch_threads") == 1
            and len(checked.get("steps", [])) == 12
            and len(checked.get("checkpoint_files", [])) == 13
            and checked.get("generated_responses") == 0
            and checked.get("optimizer_updates_performed") == 0
            and checked.get("checkpoint_tensor_deserialization") is False,
            "Complete original single-thread A audit required",
        )
        receipt = checked["receipt"]
        path = resume.legacy.plain_path(
            ROOT / resume.legacy.JOBS / "TASKS" / item["task_id"] / "COMPLETE.json"
        )
        require(
            path.stat().st_size <= 2**20
            and json.loads(path.read_bytes()) == receipt
            and receipt["plan_ref"] == scientific["plan_ref"],
            "Actual A task closure changed",
        )
        read_ref(
            dict(path=str(ROOT / receipt["producer"]["path"]), sha256=receipt["producer"]["sha256"])
        )
    require(len(complete.get("completed", [])) == 3, "All v8 executions required")
    for item in complete["completed"]:
        require(item["task_id"] in resume.QUEUE, "Unexpected A execution")
        extension = resume.OPS / "EXECUTIONS" / item["task_id"]
        require(not (extension / "FAILURE.json").exists(), "Failed A execution")
        for key in ("task_complete_ref", "execution_complete_ref", "storage_facts_ref"):
            read_ref(item[key])
    return previous, scientific


def validate_metadata(path, expected_sha):
    path = resume.legacy.plain_path(path)
    require(
        path in (OPS / "CONTRACT.json", OPS / "PREFLIGHT_CONTRACT.json"),
        "Unique B contract required",
    )
    value = read_ref(dict(path=str(path), sha256=expected_sha))
    shape(value)
    sources(value["execution_source"])
    completed_a(value)
    return value


def verify_contract(path, expected_sha):
    require(
        resume.legacy.plain_path(path) == OPS / "CONTRACT.json",
        "Execution requires final B contract",
    )
    value = validate_metadata(path, expected_sha)
    previous, scientific = completed_a(value)
    ref = previous["full_contract_ref"]
    resume.full.verify_contract(ref["path"], ref["sha256"])
    checked = resume.verify_validation(previous)
    require(
        checked.get("status") == resume.full.STATUS
        and checked.get("contract_sha256") == ref["sha256"]
        and checked.get("generated_responses") == 0
        and checked.get("optimizer_updates") == 0,
        "Actual complete cold storage verification required",
    )
    return value, scientific, previous["full_result_ref"]
