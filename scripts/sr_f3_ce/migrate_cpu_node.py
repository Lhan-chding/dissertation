#!/usr/bin/env python3
"""One-job, receipt-preserving CPU node exception; never submits or requeues.

The old dispatcher and worker remain unchanged. Their historical allocation
checks do not certify this exception. Complete delivery requires BOTH their
scientific delivery receipt and this module's independent final receipt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

TARGET_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2"
JOB_ID = "215520"
JOB_KEY = "job-164520339dd44ae6"
CLAIM_ID = "20225757473db3656141775e6e3071850f4ba14e1514c545a80af0368e00e4cd"
OLD_OPERATION = "35800b02b1ef1eff2378e4c933393b4ea2668e4d"
TEACHER_QOS = "soujanya-poria-startfund-2026-03"
ACCOUNT_FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "Start",
    "ElapsedRaw",
    "AllocTRES",
    "NodeList",
    "User",
    "Account",
    "QOS",
    "JobName",
)
ACCOUNT_FORMAT = ",".join(
    "JobName%100" if field == "JobName" else field for field in ACCOUNT_FIELDS
)
JOB_REL = f"orchestration/jobs/{JOB_KEY}"
REQUIRED_FILES = {
    *(
        f"{JOB_REL}/{name}"
        for name in (
            "INTENT.json",
            "run.sbatch",
            "ALLOCATION_VERIFIED.json",
            "RELEASE_INTENT.json",
            "RELEASE.json",
        )
    ),
    "orchestration/CONFIG.json",
    "EXECUTION_FREEZE.json",
    "RUN_IDENTITY.json",
    f"operations/{OLD_OPERATION}/OPERATION_IDENTITY.json",
}
DELIVERY_CHECKS = {
    "crc",
    "safe_unique_names",
    "payload_hashes",
    "manifest_hashes",
    "entry_links",
    "gzip_read_to_eof",
    "raw_rescoring_chain",
}
# Only scheduler-computed fields may differ across a hold/update/release.
# Everything else, including unrecognized site-specific request fields, stays
# byte-for-byte identical in the parsed scontrol record.
VOLATILE_FIELDS = {
    "JobState",
    "Reason",
    "Priority",
    "StartTime",
    "EndTime",
    "LastSchedEval",
    "EligibleTime",
    "AccrueTime",
    "SchedNodeList",
    "Scheduler",
    "RunTime",
    "SuspendTime",
    "SecsPreSuspend",
    "PreemptEligibleTime",
    "PreemptTime",
}
ALLOCATED_FIELDS = {
    "NodeList",
    "BatchHost",
    "AllocTRES",
    "CPU_IDs",
    "Mem",
    "GRES",
    "NodeAddr",
    "NodeHostName",
    "AllocNode:Sid",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def save(path, value):
    """Durable exclusive receipt: never overwrites a predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bounded(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe relative evidence path")
    result = (root / path).resolve(strict=True)
    require(result.is_relative_to(root), "Evidence path escapes run root")
    return result


def parse_fields(text):
    """Parse one scontrol -o record, retaining values containing spaces."""
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", text.strip()))
    result = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        require(key not in result, "Repeated scontrol field: " + key)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text.strip())
        result[key] = text.strip()[match.end() : end].strip()
    require(result, "Empty scontrol response")
    return result


def parse_accounting(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("|").split("|")
        require(len(parts) == len(ACCOUNT_FIELDS), "Malformed sacct response")
        rows.append(dict(zip(ACCOUNT_FIELDS, parts, strict=True)))
    exact = [row for row in rows if row["JobIDRaw"] == JOB_ID]
    require(len(exact) == 1, "Accounting must contain exactly the original job")
    return exact[0]


def tres(text):
    result = {}
    for part in text.split(","):
        if not part:
            continue
        key, sep, value = part.partition("=")
        require(sep and key not in result, "Malformed or duplicate TRES")
        result[key] = value
    return result


def memory_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value)
    require(match is not None, "Unknown memory unit")
    number, unit = match.groups()
    return int(number) * {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1048576}[unit]


def verify_job_fields(fields, intent, node, *, held=False):
    expected = {
        "JobId": JOB_ID,
        "JobName": intent["job_name"],
        "Account": "rose",
        "Partition": "cluster02",
        "QOS": TEACHER_QOS,
        "Command": intent["script"],
        "Comment": intent["comment"],
        "NumCPUs": "4",
        "NumTasks": "1",
        "CPUs/Task": "4",
        "ReqNodeList": node,
        "TimeLimit": "3-00:00:00",
        "Requeue": "0",
    }
    for key, value in expected.items():
        require(fields.get(key) == value, "Job request differs: " + key)
    require(fields.get("UserId", "").split("(")[0] == intent["user"], "Job owner differs")
    require(fields.get("NumNodes") in {"1", "1-1"}, "Job node count differs")
    requested = tres(fields.get("ReqTRES", ""))
    require(
        requested.get("cpu") == "4" and memory_mib(requested.get("mem", "")) == 12288,
        "CPU or memory request differs",
    )
    require(
        all(not k.startswith("gres/gpu") or v == "0" for k, v in requested.items()),
        "GPU request forbidden",
    )
    if held:
        require(
            (fields.get("JobState"), fields.get("Reason")) == ("PENDING", "JobHeldUser"),
            "Job is not user-held and pending",
        )


def pending_never_started(fields, accounting):
    return (
        fields.get("JobState") == "PENDING"
        and fields.get("RunTime") == "00:00:00"
        and fields.get("AllocTRES", "") in {"", "(null)"}
        and fields.get("NodeList", "") in {"", "(null)", "None"}
        and accounting.get("State") == "PENDING"
        and accounting.get("Start") in {"Unknown", "None"}
        and accounting.get("ElapsedRaw") == "0"
        and accounting.get("AllocTRES") == ""
    )


def verify_node(fields):
    require(fields.get("NodeName") == "cpu-1", "Wrong destination node")
    require("cluster02" in fields.get("Partitions", "").split(","), "Destination partition differs")
    features = fields.get("ActiveFeatures", fields.get("AvailableFeatures", ""))
    require("cpu_ok" in re.split(r"[,&]", features), "Destination lacks cpu_ok")
    state = fields.get("State", "")
    require(state.split("+")[0] in {"IDLE", "MIXED"}, "Destination not available")
    require(
        not any(flag in state for flag in ("DRAIN", "DOWN", "FAIL", "MAINT", "NOT_RESPONDING")),
        "Destination has an unavailable flag",
    )
    require(
        int(fields.get("CPUEfctv", fields["CPUTot"])) - int(fields["CPUAlloc"]) >= 4,
        "Insufficient destination CPUs",
    )
    require(
        int(fields["RealMemory"]) - int(fields["AllocMem"]) - int(fields.get("MemSpecLimit", "0"))
        >= 12288,
        "Insufficient scheduler memory",
    )


def same_request(before, after, *, running=False):
    ignored = VOLATILE_FIELDS | {"ReqNodeList"}
    if running:
        ignored |= ALLOCATED_FIELDS
    left = {k: v for k, v in before.items() if k not in ignored}
    right = {k: v for k, v in after.items() if k not in ignored}
    for fields in (left, right):
        if fields.get("NumNodes") == "1-1":
            fields["NumNodes"] = "1"
    require(left == right, "Fields other than the authorized node request changed")


def verify_paused_process(process):
    """Recheck the local process; a retained PAUSED receipt alone is insufficient."""
    pid = process.get("pid")
    require(type(pid) is int and pid > 1, "Invalid paused dispatcher PID")
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    require(
        stat[0] == "T" and str(stat[19]) == str(process.get("startticks")),
        "Dispatcher is not still the exact stopped process",
    )
    argv = (proc / "cmdline").read_bytes().split(b"\0")
    argv = [part.decode() for part in argv if part]
    require(argv == process.get("argv"), "Paused dispatcher argv changed")
    env = (proc / "environ").read_bytes().split(b"\0")
    require(b"SLURM_JOB_ID=211333" in env, "Paused process does not belong to dispatcher")


class Migration:
    def __init__(self, root, authorization, *, runner=subprocess.run):
        self.root = Path(root).resolve(strict=True)
        self.authorization = Path(authorization).resolve(strict=True)
        self.runner = runner
        self.job = self.root / JOB_REL
        self.journal = self.root / "operations/cpu_node_migration" / JOB_KEY
        self.work = self.journal / "migrate"
        self.serial = 0

    @contextmanager
    def lock(self):
        path = self.root / "orchestration/SUBMIT.lock"
        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def validate(self):
        require(str(self.root) == TARGET_ROOT, "Only the exact selected_v2 run is authorized")
        require(
            self.authorization.is_relative_to(self.root / "operations"),
            "Authorization not retained",
        )
        value = read(self.authorization)
        expected = dict(
            schema_version=1,
            scope="SRF3CE_EXACT_CPU_NODE_MIGRATION",
            root=str(self.root),
            job_id=JOB_ID,
            job_key=JOB_KEY,
            claim_id=CLAIM_ID,
            from_node="cpu-2",
            to_node="cpu-1",
            scientific_settings_changed=False,
            preserve_history=True,
            no_resubmit=True,
        )
        for key, expected_value in expected.items():
            require(
                type(value.get(key)) is type(expected_value) and value[key] == expected_value,
                "Authorization differs: " + key,
            )
        require(
            re.fullmatch(r"[0-9a-f]{40}", value.get("operation_commit", "")) is not None,
            "New operational commit is required",
        )
        deployment = self.root / "operations" / value["operation_commit"]
        require(
            self.authorization.parent == deployment
            and Path(__file__).resolve(strict=True) == deployment / "migrate_cpu_node.py",
            "Migration source and authorization must be deployed under their new commit",
        )
        request = value.get("user_request")
        require(
            isinstance(request, str) and request.strip(), "Explicit user authorization required"
        )
        require(
            hashlib.sha256(request.encode()).hexdigest() == value.get("user_request_sha256"),
            "User authorization hash differs",
        )
        require(value.get("source_sha256") == sha(Path(__file__)), "Migration source hash differs")
        require(
            value.get("guard_source_sha256")
            == sha(self.authorization.parent / "cpu_dispatcher_pause.py"),
            "Dispatcher guard source hash differs",
        )
        files = value.get("files", {})
        require(
            set(files) == REQUIRED_FILES, "Authorization must bind all original immutable receipts"
        )
        for name, digest in files.items():
            require(sha(bounded(self.root, name)) == digest, "Immutable evidence changed: " + name)
        self.auth = value
        self.intent = read(self.job / "INTENT.json")
        intent_expected = dict(
            root=str(self.root),
            key=JOB_KEY,
            kind="release",
            claim_id=CLAIM_ID,
            task_ids=["release"],
            cpu_node="cpu-2",
            cpus=4,
            memory_gib=12,
            gpu_count=0,
            qos=TEACHER_QOS,
            partition="cluster02",
            execution_entry_schema=2,
            script=str(self.job / "run.sbatch"),
        )
        require(
            all(self.intent.get(k) == v for k, v in intent_expected.items()),
            "Original intent differs",
        )
        require(
            self.intent["script_sha256"] == files[f"{JOB_REL}/run.sbatch"], "Script binding differs"
        )
        require(
            self.intent["run_identity_sha256"] == files["RUN_IDENTITY.json"], "Run identity differs"
        )
        require(
            self.intent["execution_freeze_sha256"] == files["EXECUTION_FREEZE.json"],
            "Freeze differs",
        )
        require(
            self.intent["pool_operation_identity_sha256"]
            == files[f"operations/{OLD_OPERATION}/OPERATION_IDENTITY.json"],
            "Old operation identity differs",
        )
        require(
            read(self.job / "RELEASE_INTENT.json").get("job_id") == JOB_ID, "Release intent differs"
        )
        require(
            read(self.job / "ALLOCATION_VERIFIED.json").get("job_id") == JOB_ID,
            "Original allocation job differs",
        )
        return value

    def command(self, label, argv, *, mutation=False):
        self.serial += 1
        stem = f"{self.serial:03d}_{label}"
        request = dict(at=now(), argv=argv, mutation=mutation, may_have_been_called=mutation)
        save(self.work / (stem + ".INTENT.json"), request)
        try:
            result = self.runner(argv, text=True, capture_output=True, check=False, timeout=15)
        except BaseException as error:
            save(self.work / (stem + ".UNKNOWN.json"), dict(at=now(), error=repr(error)))
            raise
        response = dict(
            at=now(), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
        save(self.work / (stem + ".RESULT.json"), response)
        require(result.returncode == 0, "Slurm command failed; do not retry mutations: " + label)
        return result.stdout

    def snapshot(self, label):
        fields = parse_fields(
            self.command(label + "_job", ["scontrol", "show", "job", "-o", JOB_ID])
        )
        accounting = parse_accounting(
            self.command(
                label + "_accounting",
                [
                    "sacct",
                    "-n",
                    "-P",
                    "-j",
                    JOB_ID,
                    "--format=" + ACCOUNT_FORMAT,
                ],
            )
        )
        require(
            accounting["User"] == self.intent["user"]
            and accounting["Account"] == "rose"
            and accounting["QOS"] == TEACHER_QOS
            and accounting["JobName"] == self.intent["job_name"],
            "Accounting identity differs",
        )
        return fields, accounting

    def no_execution_files(self):
        names = (
            "STARTED.json",
            "POOL_END.json",
            "COMPLETE.json",
            "release.LAUNCH.json",
            "release.EXIT.json",
        )
        forbidden = [self.job / name for name in names]
        forbidden += [
            self.root / "tasks/release/COMPLETE.json",
            self.root / "accounting/observability" / JOB_KEY,
        ]
        return not any(p.exists() or p.is_symlink() for p in forbidden)

    def check_unstarted(self, fields, accounting):
        if pending_never_started(fields, accounting) and self.no_execution_files():
            return True
        save(
            self.work / "ABORTED_ALREADY_STARTED.json",
            dict(
                at=now(),
                status="ABORTED_ALREADY_STARTED_OR_UNKNOWN",
                job_id=JOB_ID,
                fields=fields,
                accounting=accounting,
                action="No further scheduler mutations; never cancel or requeue",
            ),
        )
        return False

    def batch_script(self, label):
        body = self.command(
            label + "_batch_script", ["scontrol", "write", "batch_script", JOB_ID, "-"]
        )
        require(
            hashlib.sha256(body.encode()).hexdigest() == self.intent["script_sha256"],
            "Scheduler-saved batch script differs",
        )

    def migrate(self, *, pause_receipt=None):
        self.validate()
        if (self.journal / "MIGRATED.json").exists():
            return self.verify_chain()
        require(pause_receipt is not None, "A verified live dispatcher pause is required")
        pause_path = Path(pause_receipt).resolve(strict=True)
        require(pause_path.is_relative_to(self.root / "operations"), "Pause receipt not retained")
        pause = read(pause_path)
        require(
            pause.get("root") == str(self.root)
            and pause.get("job_id") == "211333"
            and pause.get("source_sha256") == self.auth["guard_source_sha256"]
            and pause.get("process", {}).get("state") == "T",
            "Dispatcher pause receipt differs",
        )
        verify_paused_process(pause["process"])
        with self.lock():
            self.validate()
            require(
                not self.work.exists(), "Existing attempt: possible prior mutation; fail closed"
            )
            self.work.mkdir(parents=True)
            save(
                self.work / "DISPATCHER_PAUSE_BINDING.json",
                dict(
                    path=str(pause_path.relative_to(self.root)),
                    sha256=sha(pause_path),
                    guard_source_sha256=self.auth["guard_source_sha256"],
                ),
            )
            state = read(self.job / "STATE.json")
            require(
                state.get("job_id") == JOB_ID
                and state.get("key") == JOB_KEY
                and state.get("status") == "SUBMITTED"
                and state.get("intent_sha256") == sha(self.job / "INTENT.json"),
                "Current original claim state differs",
            )
            before, account = self.snapshot("before")
            if not self.check_unstarted(before, account):
                return dict(status="ABORTED_ALREADY_STARTED", job_id=JOB_ID)
            verify_job_fields(before, self.intent, "cpu-2")
            self.batch_script("before")
            verify_node(
                parse_fields(
                    self.command("destination", ["scontrol", "show", "node", "-o", "cpu-1"])
                )
            )
            self.command("queue", ["squeue", "-h", "-j", JOB_ID, "-o", "%i|%T|%R"])
            self.command("hold", ["scontrol", "hold", JOB_ID], mutation=True)
            held, account = self.snapshot("held")
            if not self.check_unstarted(held, account):
                return dict(status="ABORTED_ALREADY_STARTED", job_id=JOB_ID)
            verify_job_fields(held, self.intent, "cpu-2", held=True)
            same_request(before, held)
            self.validate()
            self.command(
                "update",
                ["scontrol", "update", f"JobId={JOB_ID}", "ReqNodeList=cpu-1"],
                mutation=True,
            )
            updated, account = self.snapshot("updated")
            require(
                pending_never_started(updated, account) and self.no_execution_files(),
                "Updated job no longer known unstarted; stop without release",
            )
            verify_job_fields(updated, self.intent, "cpu-1", held=True)
            same_request(held, updated)
            self.batch_script("updated")
            self.validate()
            save(
                self.work / "RELEASE_GATE.json",
                dict(
                    at=now(),
                    status="PASS",
                    job_id=JOB_ID,
                    claim_id=CLAIM_ID,
                    authorization_sha256=sha(self.authorization),
                    source_sha256=self.auth["source_sha256"],
                    before=before,
                    held=held,
                    updated=updated,
                    immutable_files=self.auth["files"],
                    no_execution_records=True,
                    independent_final_verification_required=True,
                ),
            )
            self.command("release", ["scontrol", "release", JOB_ID], mutation=True)
            after, account = self.snapshot("released")
            require(
                after.get("JobState") in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"},
                "Released job status unknown",
            )
            require(after.get("Reason") != "JobHeldUser", "Release not confirmed")
            verify_job_fields(after, self.intent, "cpu-1")
            same_request(updated, after, running=after.get("JobState") != "PENDING")
            if after.get("JobState") != "PENDING":
                require(after.get("NodeList") == "cpu-1", "Actual allocation is not cpu-1")
            self.validate()
            index = {
                str(p.relative_to(self.journal)): sha(p)
                for p in sorted(self.work.rglob("*"))
                if p.is_file()
            }
            record = dict(
                status="MIGRATED",
                schema_version=1,
                at=now(),
                job_id=JOB_ID,
                job_key=JOB_KEY,
                claim_id=CLAIM_ID,
                from_node="cpu-2",
                to_node="cpu-1",
                authorization_path=str(self.authorization.relative_to(self.root)),
                authorization_sha256=sha(self.authorization),
                source_sha256=self.auth["source_sha256"],
                operation_commit=self.auth["operation_commit"],
                immutable_files=self.auth["files"],
                journal_files=index,
                released_fields=after,
                scientific_settings_changed=False,
                no_resubmit=True,
                final_migration_verification_required=True,
            )
            save(self.journal / "MIGRATED.json", record)
            return record

    def verify_chain(self):
        self.validate()
        work = self.journal / "migrate"
        record = read(self.journal / "MIGRATED.json")
        expected = dict(
            status="MIGRATED",
            schema_version=1,
            job_id=JOB_ID,
            job_key=JOB_KEY,
            claim_id=CLAIM_ID,
            from_node="cpu-2",
            to_node="cpu-1",
            authorization_path=str(self.authorization.relative_to(self.root)),
            authorization_sha256=sha(self.authorization),
            source_sha256=self.auth["source_sha256"],
            operation_commit=self.auth["operation_commit"],
            immutable_files=self.auth["files"],
            scientific_settings_changed=False,
            no_resubmit=True,
            final_migration_verification_required=True,
        )
        require(
            all(record.get(k) == v for k, v in expected.items()), "Migration identity chain differs"
        )
        index = record.get("journal_files", {})
        actual = {str(p.relative_to(self.journal)): sha(p) for p in work.rglob("*") if p.is_file()}
        require(index and actual == index, "Migration journal missing, changed, or extended")
        gate = read(work / "RELEASE_GATE.json")
        require(
            gate.get("status") == "PASS"
            and gate.get("authorization_sha256") == sha(self.authorization)
            and gate.get("source_sha256") == self.auth["source_sha256"]
            and gate.get("immutable_files") == self.auth["files"]
            and gate.get("no_execution_records") is True,
            "Release gate differs",
        )
        verify_job_fields(gate["before"], self.intent, "cpu-2")
        verify_job_fields(gate["held"], self.intent, "cpu-2", held=True)
        verify_job_fields(gate["updated"], self.intent, "cpu-1", held=True)
        same_request(gate["before"], gate["held"])
        same_request(gate["held"], gate["updated"])

        def query(label):
            found = list(work.glob("*_" + label + ".RESULT.json"))
            require(len(found) == 1, "Missing or duplicate query: " + label)
            return read(found[0])["stdout"]

        for label in ("before", "held", "updated"):
            fields = parse_fields(query(label + "_job"))
            account = parse_accounting(query(label + "_accounting"))
            require(
                fields == gate[label] and pending_never_started(fields, account),
                "Never-started scheduler evidence differs: " + label,
            )
            require(
                account["User"] == self.intent["user"]
                and account["Account"] == "rose"
                and account["QOS"] == TEACHER_QOS
                and account["JobName"] == self.intent["job_name"],
                "Accounting identity changed in migration chain",
            )
        for label in ("before", "updated"):
            require(
                hashlib.sha256(query(label + "_batch_script").encode()).hexdigest()
                == self.intent["script_sha256"],
                "Scheduler batch script changed in migration chain",
            )
        verify_node(parse_fields(query("destination")))
        released = parse_fields(query("released_job"))
        require(released == record.get("released_fields"), "Released snapshot differs")
        verify_job_fields(released, self.intent, "cpu-1")
        require(
            released.get("JobState") in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
            and released.get("Reason") != "JobHeldUser",
            "Release not proven",
        )
        same_request(gate["updated"], released, running=released.get("JobState") != "PENDING")
        if released.get("JobState") != "PENDING":
            require(released.get("NodeList") == "cpu-1", "Actual released node differs")
        pause = read(work / "DISPATCHER_PAUSE_BINDING.json")
        require(
            pause.get("guard_source_sha256") == self.auth["guard_source_sha256"]
            and sha(bounded(self.root, pause["path"])) == pause["sha256"],
            "Pause binding differs",
        )
        mutations = []
        commands = []
        for path in sorted(work.glob("*.INTENT.json")):
            request = read(path)
            require(
                not path.with_name(path.name.replace(".INTENT.", ".UNKNOWN.")).exists(),
                "Unknown scheduler result in completed migration",
            )
            response = read(path.with_name(path.name.replace(".INTENT.", ".RESULT.")))
            require(response.get("returncode") == 0, "Failed command in migration chain")
            commands.append(request.get("argv"))
            if request.get("mutation"):
                mutations.append(request.get("argv"))
        require(
            mutations
            == [
                ["scontrol", "hold", JOB_ID],
                ["scontrol", "update", f"JobId={JOB_ID}", "ReqNodeList=cpu-1"],
                ["scontrol", "release", JOB_ID],
            ],
            "Migration mutation sequence differs",
        )
        job_query = ["scontrol", "show", "job", "-o", JOB_ID]
        account_query = ["sacct", "-n", "-P", "-j", JOB_ID, "--format=" + ACCOUNT_FORMAT]
        script_query = ["scontrol", "write", "batch_script", JOB_ID, "-"]
        require(
            commands
            == [
                job_query,
                account_query,
                script_query,
                ["scontrol", "show", "node", "-o", "cpu-1"],
                ["squeue", "-h", "-j", JOB_ID, "-o", "%i|%T|%R"],
                ["scontrol", "hold", JOB_ID],
                job_query,
                account_query,
                ["scontrol", "update", f"JobId={JOB_ID}", "ReqNodeList=cpu-1"],
                job_query,
                account_query,
                script_query,
                ["scontrol", "release", JOB_ID],
                job_query,
                account_query,
            ],
            "Migration command sequence differs",
        )
        return record

    def verify_final(self):
        # This read-only gate must not contend with the live dispatcher's
        # nonblocking SUBMIT.lock while hashing a potentially large archive.
        with nullcontext():
            self.verify_chain()
            dispatcher_directory = self.root / "operations/dispatcher/attempts/0007"
            dispatcher_end_path = dispatcher_directory / "END.json"
            require(
                read(dispatcher_end_path).get("status") == "DONE",
                "Original dispatcher has not completed final delivery",
            )
            # Append final queries outside the archived migration snapshot.
            self.work = self.root / "delivery/migration_final_checks" / str(time.time_ns())
            self.serial = 0
            accounting = parse_accounting(
                self.command(
                    "terminal_accounting",
                    [
                        "sacct",
                        "-n",
                        "-P",
                        "-j",
                        JOB_ID,
                        "--format=" + ACCOUNT_FORMAT,
                    ],
                )
            )
            require(
                accounting["State"] == "COMPLETED"
                and accounting["ExitCode"] == "0:0"
                and accounting["NodeList"] == "cpu-1"
                and accounting["Start"] not in {"Unknown", "None", ""}
                and int(accounting["ElapsedRaw"]) > 0
                and accounting["User"] == self.intent["user"]
                and accounting["QOS"] == TEACHER_QOS
                and accounting["Account"] == "rose"
                and accounting["JobName"] == self.intent["job_name"],
                "Actual terminal execution differs",
            )
            allocated = tres(accounting["AllocTRES"])
            require(
                allocated.get("cpu") == "4"
                and memory_mib(allocated.get("mem", "")) == 12288
                and all(not k.startswith("gres/gpu") or v == "0" for k, v in allocated.items()),
                "Actual terminal resources differ",
            )
            completed = read(self.job / "COMPLETE.json")
            require(
                completed
                == dict(status="COMPLETE", job_id=JOB_ID, task_ids=["release"], claim_id=CLAIM_ID),
                "Job completion binding differs",
            )
            pool = read(self.job / "POOL_END.json")
            require(
                pool.get("status") == "COMPLETE"
                and pool.get("job_id") == JOB_ID
                and pool.get("verified_tasks") == ["release"]
                and pool.get("exits") == {"release": 0},
                "Pool completion differs",
            )
            require(
                read(self.job / "STARTED.json").get("job_id") == JOB_ID, "Start receipt differs"
            )
            launch = read(self.job / "release.LAUNCH.json")
            require(
                launch.get("claim_id") == CLAIM_ID
                and launch.get("device") == ""
                and launch.get("command") == self.intent["worker_entry_commands"]["release"],
                "Release launch binding differs",
            )
            require(
                read(self.job / "release.EXIT.json").get("returncode") == 0, "Release exit differs"
            )
            task_path = self.root / "tasks/release/COMPLETE.json"
            task = read(task_path)
            require(
                task.get("job_id") == JOB_ID and task.get("claim_id") == CLAIM_ID,
                "Release task claim differs",
            )
            require(
                task.get("status") == "COMPLETE"
                and task.get("run_identity_sha256") == self.auth["files"]["RUN_IDENTITY.json"]
                and task.get("execution_freeze_sha256")
                == self.auth["files"]["EXECUTION_FREEZE.json"],
                "Release task scientific identity differs",
            )
            observer_path = self.root / "accounting/observability" / JOB_KEY / "release/FINAL.json"
            observer = read(observer_path)
            require(
                observer.get("status") == "COMPLETE"
                and observer.get("task_receipt_sha256") == sha(task_path),
                "Observer completion binding differs",
            )
            archive_path = self.root / "delivery/SR_F3_CE_20261016.zip"
            builder_path = Path(str(archive_path) + ".verification.json")
            builder = read(builder_path)
            require(
                builder.get("status") == "VERIFIED_DELIVERY"
                and all(builder.get("checks", {}).get(k) == "PASS" for k in DELIVERY_CHECKS),
                "Original scientific delivery verification incomplete",
            )
            require(
                builder.get("run_identity_sha256") == sha(self.root / "RUN_IDENTITY.json"),
                "Builder run identity differs",
            )
            archive = builder.get("archive", {})
            require(
                archive.get("path") == str(archive_path)
                and archive.get("sha256") == sha(archive_path)
                and archive.get("bytes") == archive_path.stat().st_size,
                "Archive identity differs",
            )
            dispatcher_path = dispatcher_directory / "DELIVERY_VERIFIED.json"
            dispatcher = read(dispatcher_path)
            require(
                dispatcher.get("receipt") == str(builder_path)
                and dispatcher.get("sha256") == sha(builder_path),
                "Dispatcher delivery binding differs",
            )
            migration_files = [
                self.journal / "MIGRATED.json",
                self.authorization,
                *[
                    self.journal / name
                    for name in read(self.journal / "MIGRATED.json")["journal_files"]
                ],
                *[bounded(self.root, name) for name in self.auth["files"]],
            ]
            source = Path(__file__).resolve(strict=True)
            require(
                source.is_relative_to(self.root / "operations"),
                "Final verifier must be retained deployed source",
            )
            migration_files.append(source)
            migration_files.append(self.authorization.parent / "cpu_dispatcher_pause.py")
            pause_binding = read(self.journal / "migrate/DISPATCHER_PAUSE_BINDING.json")
            pause_path = bounded(self.root, pause_binding["path"])
            require(sha(pause_path) == pause_binding["sha256"], "Dispatcher pause evidence changed")
            migration_files.append(pause_path)
            resume_path = pause_path.parent / "RESUMED.json"
            resume = read(resume_path)
            paused = read(pause_path)
            resumed_process = resume.get("process", {})
            require(
                resume.get("job_id") == "211333"
                and resume.get("worker_reaped_before_resume") is True
                and resumed_process.get("state") not in {None, "", "T", "t", "Z", "X"}
                and all(
                    resumed_process.get(k) == paused.get("process", {}).get(k)
                    for k in ("pid", "startticks", "argv", "uids", "slurm_job_id", "exe")
                ),
                "Dispatcher resume evidence differs",
            )
            migration_files.append(resume_path)
            action_path = pause_path.parent / "ACTION_RESULT.json"
            action = read(action_path)
            require(
                action.get("status") == "RETURNED"
                and action.get("result") == read(self.journal / "MIGRATED.json"),
                "Guard action did not return the exact successful migration",
            )
            # The supervisor writes its remaining records before returning;
            # retain every guard record, including the child outcome.
            migration_files.extend(path for path in pause_path.parent.rglob("*") if path.is_file())
            archived = {}
            with zipfile.ZipFile(archive_path) as package:
                names = package.namelist()
                require(len(names) == len(set(names)), "Duplicate archive names")
                require(
                    type(archive.get("entries")) is int and archive["entries"] == len(names),
                    "Archive entry count differs",
                )
                for path in sorted(set(migration_files)):
                    name = "evidence/" + str(path.relative_to(self.root))
                    require(name in names, "Migration evidence absent from delivery: " + name)
                    with package.open(name) as stream:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    require(
                        digest.hexdigest() == sha(path),
                        "Archived migration evidence differs: " + name,
                    )
                    archived[name] = digest.hexdigest()
            result = dict(
                schema_version=1,
                status="FINAL_MIGRATION_VERIFIED",
                at=now(),
                job_id=JOB_ID,
                claim_id=CLAIM_ID,
                actual_node="cpu-1",
                accounting=accounting,
                migration_sha256=sha(self.journal / "MIGRATED.json"),
                authorization_sha256=sha(self.authorization),
                source_sha256=sha(source),
                archive=archive,
                builder_receipt_sha256=sha(builder_path),
                dispatcher_delivery_sha256=sha(dispatcher_path),
                dispatcher_end_sha256=sha(dispatcher_end_path),
                task_receipt_sha256=sha(task_path),
                observer_final_sha256=sha(observer_path),
                archived_migration_files=archived,
                acceptance=(
                    "Original scientific delivery verification "
                    "AND independent migration verification"
                ),
            )
            final_path = self.root / "delivery/FINAL_MIGRATION_VERIFIED.json"
            if final_path.exists():
                prior = read(final_path)
                require(
                    all(prior.get(k) == v for k, v in result.items() if k != "at"),
                    "Existing final migration verification differs",
                )
                return prior
            save(final_path, result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--mode", choices=("migrate", "verify-final"), required=True)
    parser.add_argument("--pause-receipt")
    args = parser.parse_args()
    operation = Migration(args.root, args.authorization)
    result = (
        operation.migrate(pause_receipt=args.pause_receipt)
        if args.mode == "migrate"
        else operation.verify_final()
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
