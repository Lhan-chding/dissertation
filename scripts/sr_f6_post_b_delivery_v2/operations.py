"""Immutable CPU primary-audit deployment and one-shot scheduler operations.

No research-source mutation, cleanup, training, recovery or DEV operation exists.
The deploy bootstrap must authenticate this operator against the supplied source
manifest before executing it; subsequent actions require its deployed file origin.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_POST_B_DELIVERY_V2_20261101"
CODE = PROJECT / "sr_f6_code_b7e7b06"
PYTHON = PROJECT / "envs/ssvc-py312/bin/python3.12"
PREFIX = "scripts/sr_f6_post_b_delivery_v2/"
SCRIPT, DRIVER, LAUNCHER, RECOVERY, PRIMARY = (
    PREFIX + name
    for name in (
        "operations.py",
        "audit.py",
        "launch.sbatch",
        "recovery_readonly.py",
        "primary_readonly.py",
    )
)
MEMBERS = {SCRIPT, DRIVER, LAUNCHER, RECOVERY, PRIMARY}
MANIFEST = "DELIVERY_SOURCE.json"
SOURCE_SCHEMA = "srf6-post-b-delivery-source-v2"
JOB_NAME = "LH-SRF6-PostB-DeliveryV2"
USER = "varun024"
QOS = "soujanya-poria-startfund-2026-03"
B_COMPLETE_SHA = "0047dabc1db4484e9949d1f367896e02df07b3d4ae6e99d9636d25b7c87bd77f"
PRIOR_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_POST_B_DELIVERY_20261101"
PRIOR_CONTRACT_SHA = "c2806275b851afa7e0dd684cec1a7005f8de7da330f450b10dca79ca4902dfb5"
PRIOR_FAILURE_SHA = "bdb76777e45b3708c5c915dc07e4a5a7184bd2bd9ce52fc297a202dff1db0a3e"
PRIOR_ERROR = "Failed numerical attempt remains in inventory; cannot assert runtime gate"
HISTORICAL_OOM_PROOF_SHA = "5d235c3885217559562c63cea42123029b1177da4394b7739ed678ea4616c81c"
LIMIT = 32 * 1024 * 1024


def require(value, message):
    if not value:
        raise PermissionError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "Absolute canonical path required")
    for part in (*reversed(path.parents), path):
        require(not part.is_symlink(), f"Symlink forbidden: {part}")
    return path


def child(base, value):
    path = Path(value)
    path = canonical(path if path.is_absolute() else canonical(base) / path)
    require(path != base and path.is_relative_to(base), "Path escapes authorized root")
    return path


def data(path):
    fd = os.open(canonical(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= LIMIT, "Bounded regular file required"
        )
        return stream.read()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def read(path):
    return json.loads(data(path))


def reference(path):
    return dict(path=str(path), sha256=digest(data(path)))


def write_bytes(path, payload, *, deployment=None):
    path = canonical(path)
    allowed = path.is_relative_to(OPS) and path != OPS
    if deployment is not None:
        allowed |= path.is_relative_to(deployment) and path != deployment
    require(allowed, "Writes limited to the new audit deployment and operations root")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return reference(path)


def write(path, value):
    return write_bytes(path, encode(value))


def deployment_for(revision):
    require(
        isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision),
        "40-hex revision required",
    )
    return canonical(PROJECT / ("sr_f6_post_b_delivery_v2_code_" + revision[:7]))


def validate_manifest(manifest, revision, binding_sha256):
    require(re.fullmatch("[0-9a-f]{64}", binding_sha256), "Explicit manifest hash required")
    require(digest(encode(manifest)) == binding_sha256, "Manifest hash differs")
    target = deployment_for(revision)
    require(
        set(manifest) == {"schema", "revision", "deployment", "files"}
        and manifest["schema"] == SOURCE_SCHEMA
        and manifest["revision"] == revision
        and manifest["deployment"] == str(target)
        and set(manifest["files"]) == MEMBERS,
        "Exact independent five-file source manifest required",
    )
    for item in manifest["files"].values():
        require(
            set(item) == {"bytes", "sha256"}
            and type(item["bytes"]) is int
            and 0 < item["bytes"] <= 1024 * 1024
            and re.fullmatch("[0-9a-f]{64}", item["sha256"]),
            "Invalid source member metadata",
        )
    return target


def make_package(revision, contents):
    """Build the transport payload from git-extracted immutable member bytes."""
    require(
        set(contents) == MEMBERS and all(isinstance(v, bytes) for v in contents.values()),
        "Exact five byte payloads required",
    )
    manifest = dict(
        schema=SOURCE_SCHEMA,
        revision=revision,
        deployment=str(deployment_for(revision)),
        files={
            name: dict(bytes=len(payload), sha256=digest(payload))
            for name, payload in contents.items()
        },
    )
    validate_manifest(manifest, revision, digest(encode(manifest)))
    return dict(
        manifest=manifest,
        contents={
            name: base64.b64encode(payload).decode("ascii") for name, payload in contents.items()
        },
    )


def deploy(package, revision, binding_sha256):
    """Publish authenticated package once; partial publication is never repaired here."""
    require(set(package) == {"manifest", "contents", "evidence"}, "Exact package fields required")
    require(set(package["evidence"]) == {"historical_oom_proof"}, "Exact private evidence required")
    encoded_proof = package["evidence"]["historical_oom_proof"]
    require(
        isinstance(encoded_proof, str) and len(encoded_proof) <= 1024 * 1024,
        "Bounded proof required",
    )
    proof = base64.b64decode(encoded_proof, validate=True)
    require(digest(proof) == HISTORICAL_OOM_PROOF_SHA, "Historical OOM proof differs")
    require(isinstance(json.loads(proof), dict), "Historical OOM proof object required")
    manifest = package["manifest"]
    target = validate_manifest(manifest, revision, binding_sha256)
    require(set(package["contents"]) == MEMBERS, "Unexpected source payload member")
    payloads = {}
    for name, encoded in package["contents"].items():
        require(
            isinstance(encoded, str) and len(encoded) < 2 * 1024 * 1024, "Bounded base64 required"
        )
        payload = base64.b64decode(encoded, validate=True)
        item = manifest["files"][name]
        require(
            len(payload) == item["bytes"] and digest(payload) == item["sha256"],
            "Source bytes differ",
        )
        if name.endswith(".py"):
            ast.parse(payload, filename=name)
        payloads[name] = payload
    require(
        canonical(PROJECT).is_dir() and canonical(OPS.parent).is_dir(),
        "Existing authorized roots required",
    )
    require(
        not canonical(OPS).exists() and not target.exists(),
        "Prior deployment/ops forbids deploy retry",
    )
    OPS.mkdir(mode=0o700)
    write(
        OPS / "DEPLOY_INTENT.json",
        dict(at=now(), revision=revision, deployment=str(target), binding_sha256=binding_sha256),
    )
    try:
        proof_ref = write_bytes(OPS / "HISTORICAL_OOM_PROOF.json", proof)
        target.mkdir(mode=0o700)
        for name in sorted(MEMBERS):
            path = child(target, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(path, payloads[name], deployment=target)
            path.chmod(0o400)
        write_bytes(target / MANIFEST, encode(manifest), deployment=target)
        (target / MANIFEST).chmod(0o400)
        result = dict(
            status="DEPLOYED_NOT_SUBMITTED",
            at=now(),
            revision=revision,
            deployment=str(target),
            binding_sha256=binding_sha256,
            historical_oom_proof_ref=proof_ref,
            files=len(MEMBERS) + 1,
            bytes=sum(len(v) for v in payloads.values()) + len(encode(manifest)),
        )
        write(OPS / "DEPLOYED.json", result)
        return result
    except BaseException as exc:
        write(
            OPS / "DEPLOY_FAILURE.json",
            dict(at=now(), status="PARTIAL_DEPLOYMENT_PRESERVED", error=str(exc)),
        )
        raise


def identity(revision, binding_sha256):
    target = deployment_for(revision)
    require(
        canonical(__file__) == target / SCRIPT,
        "Only the actual immutable deployed operator may act",
    )
    manifest_bytes = data(target / MANIFEST)
    manifest = json.loads(manifest_bytes)
    require(manifest_bytes == encode(manifest), "Canonical manifest bytes required")
    validate_manifest(manifest, revision, binding_sha256)
    for name, item in manifest["files"].items():
        payload = data(child(target, name))
        require(
            len(payload) == item["bytes"] and digest(payload) == item["sha256"],
            "Deployed source changed",
        )
    inventory = {
        p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file() or p.is_symlink()
    }
    require(inventory == MEMBERS | {MANIFEST}, "Unexpected deployment inventory")
    actual = dict(revision=revision, deployment=str(target), binding_sha256=binding_sha256)
    receipt = read(OPS / "DEPLOYED.json")
    require(
        receipt.get("status") == "DEPLOYED_NOT_SUBMITTED"
        and all(receipt.get(k) == v for k, v in actual.items()),
        "Missing matching deployment receipt",
    )
    return actual


@contextmanager
def lock():
    require(canonical(OPS).is_dir(), "Deployed operations root required")
    fd = os.open(canonical(OPS / ".operation.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def driver(actual):
    # The checked source deployment must never acquire import-generated files,
    # including if a caller omitted the documented interpreter -B flag.
    sys.dont_write_bytecode = True
    path = Path(actual["deployment"]) / DRIVER
    spec = importlib.util.spec_from_file_location("post_b_pinned_audit_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(
        module.ROOT == PROJECT / "sr_f6_semantic_factorial_20261027"
        and module.NEW_OPS == OPS
        and module.DEPLOYMENT == CODE,
        "Wrong audit boundaries",
    )
    return module


def query(argv):
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "TZ": "UTC"},
    ).stdout


def queue_rows():
    text = query(["squeue", "--array", "-h", "-u", USER, "-o", "%i|%j|%T|%k"])
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    require(
        all(
            len(row) == 4 and re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", row[0]) and row[1] and row[2]
            for row in rows
        ),
        "Unknown queue shape; no submission",
    )
    require(len({row[0] for row in rows}) == len(rows), "Duplicate queue rows")
    return rows


def predecessor():
    text = query(
        ["sacct", "-X", "-n", "-P", "-j", "237289,238732", "-o", "JobIDRaw,State,ExitCode"]
    )
    rows = [line.rstrip("|").split("|") for line in text.splitlines() if line.strip()]
    require(
        sorted(rows) == [["237289", "COMPLETED", "0:0"], ["238732", "FAILED", "1:0"]],
        "B must be COMPLETED 0:0 and prior CPU must be FAILED 1:0",
    )
    rows = queue_rows()
    require(not any(row[0] in {"237289", "238732"} for row in rows), "Predecessor still queued")
    return dict(at=now(), job_ids=["237289", "238732"], accounting=text, queue_gone=True), rows


def prior_evidence():
    contract_path = canonical(PRIOR_OPS / "CONTRACT.json")
    failure_path = canonical(PRIOR_OPS / "RESULT/FAILURE.json")
    require(digest(data(contract_path)) == PRIOR_CONTRACT_SHA, "Prior delivery contract changed")
    require(digest(data(failure_path)) == PRIOR_FAILURE_SHA, "Prior delivery failure changed")
    failure = read(failure_path)
    require(
        failure.get("status") == "DELIVERY_VERIFICATION_INCOMPLETE"
        and failure.get("scientific_gate_status") == "NOT_EVALUATED"
        and failure.get("error_type") == "PermissionError"
        and failure.get("error") == PRIOR_ERROR
        and failure.get("new_responses") == 0
        and failure.get("new_updates") == 0,
        "Unexpected prior delivery failure; no automatic successor",
    )
    return dict(
        prior_delivery_contract_ref=dict(path=str(contract_path), sha256=PRIOR_CONTRACT_SHA),
        prior_delivery_failure_ref=dict(path=str(failure_path), sha256=PRIOR_FAILURE_SHA),
    )


def historical_oom_proof_ref():
    path = canonical(OPS / "HISTORICAL_OOM_PROOF.json")
    require(digest(data(path)) == HISTORICAL_OOM_PROOF_SHA, "Historical OOM proof differs")
    return dict(path=str(path), sha256=HISTORICAL_OOM_PROOF_SHA)


def check_evidence(actual):
    previous = prior_evidence()
    verifier = driver(actual)
    plan, _, _, _ = verifier.source_and_pins()
    closure, recovery = verifier.verify_b(plan)
    return dict(
        **previous,
        b_closure=closure,
        recovery_saved_receipts=recovery,
        full_delivery_verified=False,
        evaluation_authorized=False,
    )


def contract_value(actual):
    return dict(
        schema="srf6-post-b-delivery-v2",
        output_directory=str(OPS / "RESULT"),
        driver_sha256=digest(data(Path(actual["deployment"]) / DRIVER)),
        source_deployment=str(CODE),
        b_job_id="237289",
        round_b_complete_sha256=B_COMPLETE_SHA,
        **prior_evidence(),
        historical_oom_proof_ref=historical_oom_proof_ref(),
    )


def metadata(actual, contract_sha256):
    path = OPS / "CONTRACT.json"
    require(
        digest(data(path)) == contract_sha256 and read(path) == contract_value(actual),
        "Contract differs",
    )
    expected = dict(source=actual, contract_ref=dict(path=str(path), sha256=contract_sha256))
    require(read(OPS / "SOURCE_BINDING.json") == expected, "Source/contract binding differs")
    return expected


def freeze(actual):
    with lock():
        require(
            not (OPS / "CONTRACT.json").exists() and not (OPS / "SOURCE_BINDING.json").exists(),
            "Prior/partial freeze forbids retry",
        )
        prior, _ = predecessor()
        evidence = check_evidence(actual)
        contract_ref = write(OPS / "CONTRACT.json", contract_value(actual))
        write(OPS / "SOURCE_BINDING.json", dict(source=actual, contract_ref=contract_ref))
        result = dict(
            status="FROZEN_NOT_SUBMITTED",
            at=now(),
            source=actual,
            contract_ref=contract_ref,
            predecessor=prior,
            evidence=evidence,
        )
        write(OPS / "FROZEN.json", result)
        return result


def argv_for(actual, contract_sha256, comment):
    require(re.fullmatch(r"SRF6PDV2-[0-9a-f]{32}", comment), "Unique submission comment required")
    exports = dict(
        PATH="/usr/bin:/bin",
        SRF6_DELIVERY_CONTRACT=str(OPS / "CONTRACT.json"),
        SRF6_DELIVERY_CONTRACT_SHA256=contract_sha256,
        SRF6_DELIVERY_DRIVER=str(Path(actual["deployment"]) / DRIVER),
    )
    require(
        all("," not in v and "\n" not in v for v in exports.values()), "Unsafe scheduler export"
    )
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=" + JOB_NAME,
        "--account=rose",
        "--partition=cluster02",
        "--qos=" + QOS,
        "--cpus-per-task=4",
        "--mem=12G",
        "--time=02:00:00",
        "--nodes=1",
        "--ntasks=1",
        "--comment=" + comment,
        "--chdir=" + str(OPS),
        "--output=" + str(OPS / "audit-%j.log"),
        "--export=" + ",".join(k + "=" + v for k, v in exports.items()),
        str(Path(actual["deployment"]) / LAUNCHER),
    ]


def unused():
    allowed = {
        "HISTORICAL_OOM_PROOF.json",
        "DEPLOY_INTENT.json",
        "DEPLOYED.json",
        "CONTRACT.json",
        "SOURCE_BINDING.json",
        "FROZEN.json",
        ".operation.lock",
    }
    require(
        {p.name for p in OPS.iterdir()} <= allowed,
        "Prior/partial/unknown audit intent or execution; reconcile, never resubmit",
    )
    require((OPS / "FROZEN.json").is_file(), "Complete frozen receipt required")


def requested(job, comment):
    raw = query(["scontrol", "show", "job", "-o", job])
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", raw))
    require(
        fields.get("JobId") == job
        and fields.get("JobName") == JOB_NAME
        and fields.get("Account") == "rose"
        and fields.get("Partition") == "cluster02"
        and fields.get("QOS") == QOS
        and fields.get("Requeue") == "0"
        and fields.get("TimeLimit") == "02:00:00"
        and fields.get("Comment") == comment
        and fields.get("UserId", "").startswith(USER + "("),
        "Submitted job identity differs",
    )
    tres = dict(item.split("=", 1) for item in fields.get("ReqTRES", "").split(",") if "=" in item)
    require(
        tres.get("cpu") == "4"
        and tres.get("mem") in ("12G", "12288M")
        and not any("gpu" in key and value != "0" for key, value in tres.items()),
        "Submitted CPU resources differ",
    )
    return dict(at=now(), status="REQUEST_VERIFIED", job_id=job, raw=raw)


def submit(actual, contract_sha256):
    require(getpass.getuser() == USER, "Wrong scheduler user")
    with lock():
        metadata(actual, contract_sha256)
        unused()
        prior, rows = predecessor()
        require(
            not any(row[1] == JOB_NAME or row[3].startswith("SRF6PDV2-") for row in rows),
            "Matching live audit requires reconciliation",
        )
        if len(rows) >= 5:
            return dict(
                status="NO_LEGAL_CAPACITY",
                submitted_by_user=len(rows),
                submit_limit=5,
                requested_gpus=0,
            )
        evidence = check_evidence(actual)
        # Recheck capacity immediately before persisting the unique intent.
        rows = queue_rows()
        require(
            not any(row[1] == JOB_NAME or row[3].startswith("SRF6PDV2-") for row in rows),
            "Matching live audit requires reconciliation",
        )
        if len(rows) >= 5:
            return dict(
                status="NO_LEGAL_CAPACITY",
                submitted_by_user=len(rows),
                submit_limit=5,
                requested_gpus=0,
            )
        comment = "SRF6PDV2-" + uuid.uuid4().hex
        argv = argv_for(actual, contract_sha256, comment)
        intent = dict(
            status="SUBMISSION_INTENT_NOT_COMPLETION",
            at=now(),
            source=actual,
            contract_ref=dict(path=str(OPS / "CONTRACT.json"), sha256=contract_sha256),
            comment=comment,
            argv=argv,
            predecessor=prior,
            evidence=evidence,
            submitted_by_user=len(rows),
            submit_limit=5,
            requested_gpus=0,
        )
        intent_ref = write(OPS / "SUBMIT_INTENT.json", intent)
        try:
            process = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "TZ": "UTC"},
            )
            match = re.fullmatch(r"([1-9][0-9]*)(?:;cluster02)?", process.stdout.strip())
            ok = process.returncode == 0 and match is not None
            result = dict(
                status="SUBMITTED" if ok else "SUBMISSION_UNKNOWN",
                job_id=match[1] if ok else None,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = dict(
                status="SUBMISSION_UNKNOWN",
                job_id=None,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        result.update(at=now(), source=actual, intent_ref=intent_ref)
        write(OPS / "SUBMISSION.json", result)
        if result["job_id"]:
            try:
                check = requested(result["job_id"], comment)
            except Exception as exc:
                check = dict(
                    status="SUBMITTED_REQUEST_UNVERIFIED_NO_RETRY",
                    job_id=result["job_id"],
                    at=now(),
                    error=str(exc),
                )
            write(OPS / "SUBMISSION_CHECK.json", check)
            result = dict(result, request_check=check)
        return result


def reconcile(actual, contract_sha256):
    """Only resolve a saved intent; never calls sbatch or changes original receipts."""
    with lock():
        metadata(actual, contract_sha256)
        intent_path = OPS / "SUBMIT_INTENT.json"
        intent = read(intent_path)
        require(
            intent["source"] == actual
            and intent["contract_ref"]["sha256"] == contract_sha256
            and intent["argv"] == argv_for(actual, contract_sha256, intent["comment"]),
            "Unknown intent provenance",
        )
        if (OPS / "SUBMISSION.json").exists():
            previous = read(OPS / "SUBMISSION.json")
            require(previous["intent_ref"] == reference(intent_path), "Submission intent changed")
            if previous["status"] == "SUBMITTED":
                return dict(status="ALREADY_SUBMITTED_NO_RETRY", job_id=previous["job_id"])
        start = (
            dt.datetime.fromisoformat(intent["at"])
            .astimezone(dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S")
        )
        history = query(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-u",
                USER,
                "-S",
                start,
                "-o",
                "JobIDRaw,JobName%64,State,ExitCode,Comment%128",
            ]
        )
        history_rows = [row.rstrip("|").split("|") for row in history.splitlines() if row.strip()]
        require(all(len(row) == 5 for row in history_rows), "Unknown accounting shape")
        matches = {
            row[0] for row in history_rows if row[1] == JOB_NAME and row[4] == intent["comment"]
        }
        matches.update(
            row[0] for row in queue_rows() if row[1] == JOB_NAME and row[3] == intent["comment"]
        )
        require(
            all(re.fullmatch("[1-9][0-9]*", job) for job in matches), "Unexpected reconciled job ID"
        )
        result = dict(
            status="RECONCILED_SUBMITTED_NO_RETRY"
            if len(matches) == 1
            else "SUBMISSION_STILL_UNKNOWN_NO_RETRY",
            at=now(),
            job_id=next(iter(matches)) if len(matches) == 1 else None,
            matches=sorted(matches),
            accounting=history,
            intent_ref=reference(intent_path),
        )
        write(OPS / ("RECONCILIATION_" + uuid.uuid4().hex + ".json"), result)
        return result


def status(actual):
    """Read existing small receipts and scheduler state without creating an artifact."""
    receipts = {}
    contract_state = "NOT_FROZEN"
    if (OPS / "CONTRACT.json").exists():
        contract_state = "PARTIAL_FREEZE_PRESERVED"
        if (OPS / "SOURCE_BINDING.json").exists():
            binding = read(OPS / "SOURCE_BINDING.json")
            metadata(actual, binding["contract_ref"]["sha256"])
            contract_state = "CONTRACT_AND_SOURCE_BINDING_VERIFIED"
    for name in ("DEPLOYED", "FROZEN", "SUBMIT_INTENT", "SUBMISSION", "SUBMISSION_CHECK"):
        path = OPS / (name + ".json")
        if path.exists():
            receipts[name] = dict(reference=reference(path), value=read(path))
    for path in sorted(OPS.glob("RECONCILIATION_*.json")):
        receipts[path.name] = dict(reference=reference(path), value=read(path))
    result = OPS / "RESULT"
    execution = {}
    if result.exists():
        canonical(result)
        if (result / "COMPLETE.json").exists():
            complete = read(result / "COMPLETE.json")
            require(
                complete.get("result_ref") == reference(result / "RESULT.json"),
                "Completed primary audit result hash differs",
            )
        for path in sorted(result.glob("*.json")):
            value = read(path)
            execution[path.name] = dict(
                reference=reference(path),
                summary={
                    k: value[k]
                    for k in (
                        "status",
                        "at",
                        "error_type",
                        "error",
                        "full_delivery_verified",
                        "recovery_recheck_status",
                        "measurement",
                        "measurements",
                        "exposure",
                        "result_ref",
                    )
                    if k in value
                },
            )
    jobs = sorted(
        {record["value"]["job_id"] for record in receipts.values() if record["value"].get("job_id")}
    )
    require(
        all(isinstance(job, str) and re.fullmatch("[1-9][0-9]*", job) for job in jobs),
        "Invalid saved scheduler job ID",
    )
    accounting = (
        query(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(jobs),
                "-o",
                "JobIDRaw,State,ExitCode,Elapsed,Start,End",
            ]
        )
        if jobs
        else ""
    )
    return dict(
        at=now(),
        source=actual,
        receipts=receipts,
        contract_state=contract_state,
        execution=execution,
        queue=queue_rows(),
        accounting=accounting,
        full_delivery_verified=False,
        evaluation_authorized=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "freeze", "submit", "reconcile", "read"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--contract-sha256")
    args = parser.parse_args()
    if args.action == "deploy":
        payload = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
        require(len(payload) <= 4 * 1024 * 1024, "Deployment package too large")
        result = deploy(json.loads(payload), args.revision, args.binding_sha256)
    else:
        actual = identity(args.revision, args.binding_sha256)
        if args.action in ("submit", "reconcile"):
            require(args.contract_sha256 is not None, "Pinned contract hash required")
            result = (submit if args.action == "submit" else reconcile)(
                actual, args.contract_sha256
            )
        else:
            result = freeze(actual) if args.action == "freeze" else status(actual)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
