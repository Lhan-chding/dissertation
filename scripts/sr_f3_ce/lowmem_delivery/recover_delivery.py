#!/usr/bin/env python3
"""One authorized measured low-memory delivery attempt; never schedules scientific work."""

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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

TARGET_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2"
OLD_OPERATION = "35800b02b1ef1eff2378e4c933393b4ea2668e4d"
MIGRATION_OPERATION = "3a7f5d120e02dcd042397408c6031a98713d1e7d"
SUPERSEDED_JOB = "216128"
SUPERSEDED_OPERATION = "83e116d1f55c3656d88d8f5a8ca965c4dd168e45"
SUPERSEDED_DIRECTORY = "operations/delivery_recovery/20261021_delivery_oom"
FAILED_ATTEMPT = "operations/dispatcher/attempts/0007"
RELEASE_KEY = "job-164520339dd44ae6"
REQUIRED_FILES = {
    f"{FAILED_ATTEMPT}/{name}"
    for name in ("END.json", "DELIVERY_PROCESS.json", "DELIVERY_INTENT.json")
} | {
    "EXECUTION_FREEZE.json",
    "RUN_IDENTITY.json",
    "orchestration/CONFIG.json",
    "tasks/release/COMPLETE.json",
    f"accounting/observability/{RELEASE_KEY}/release/FINAL.json",
    *[
        f"operations/{OLD_OPERATION}/{n}"
        for n in ("OPERATION_IDENTITY.json", "OPERATION_SOURCES.json", "build_delivery.py")
    ],
    *[
        f"operations/{MIGRATION_OPERATION}/{n}"
        for n in ("MIGRATION_AUTHORIZATION.json", "migrate_cpu_node.py", "cpu_dispatcher_pause.py")
    ],
}
CHECKS = {
    "crc",
    "safe_unique_names",
    "payload_hashes",
    "manifest_hashes",
    "entry_links",
    "gzip_read_to_eof",
    "raw_rescoring_chain",
}
COVERAGE = dict(
    verified_tasks=33,
    training_paths=12,
    evaluated_models=15,
    logical_updates=768,
    training_records=98304,
    support_examples=2304,
    monitor_records=26112,
    engine_diagnostic_records=2880,
    engine_new_rollouts=512,
    engine_replayed_rollouts=256,
    engine_physical_updates=6,
    records=233340,
)


def require(value, message):
    if not value:
        raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def bounded(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe evidence path")
    path = (root / rel).resolve(strict=True)
    require(path.is_relative_to(root), "Evidence escaped root")
    return path


def fields(text):
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", text.strip()))
    result = {}
    for i, m in enumerate(matches):
        key = m[1]
        require(key not in result, "Duplicate scheduler field")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text.strip())
        result[key] = text.strip()[m.end() : end].strip()
    require(result, "Empty scheduler response")
    return result


def memory_mib(value):
    match = re.fullmatch(r"(\d+)([MGT]?)", value)
    require(match is not None, "Unrecognized memory allocation")
    return int(match[1]) * {"": 1, "M": 1, "G": 1024, "T": 1024**2}[match[2]]


def resources(value):
    result = {}
    for part in value.split(","):
        key, val = part.split("=", 1)
        require(key not in result, "Repeated resource")
        result[key] = val
    require(
        result.get("cpu") == "4" and memory_mib(result.get("mem", "")) == 12288,
        "Recovery CPU or memory allocation differs",
    )
    require(
        result.get("node") == "1"
        and all(not k.startswith("gres/gpu") or v == "0" for k, v in result.items()),
        "Recovery node or GPU allocation differs",
    )


class Recovery:
    def __init__(self, root, authorization, runner=subprocess.run, require_submission=True):
        self.root = Path(root).resolve(strict=True)
        require(str(self.root) == TARGET_ROOT, "Recovery root differs")
        self.authorization = Path(authorization).resolve(strict=True)
        require(
            self.authorization.is_relative_to(self.root / "operations"),
            "Authorization outside operations",
        )
        self.runner = runner
        self.auth = read(self.authorization)
        a = self.auth
        require(
            a.get("schema_version") == 1
            and a.get("status") == "AUTHORIZED_DELIVERY_ONLY_RECOVERY"
            and a.get("root") == str(self.root)
            and a.get("failed_job_id") == "211333"
            and a.get("failed_attempt") == FAILED_ATTEMPT,
            "Recovery authorization identity differs",
        )
        require(
            a.get("allowed_nodes") == ["cpu-1", "cpu-2"]
            and a.get("qos") == "soujanya-poria-startfund-2026-03"
            and a.get("account") == "rose"
            and a.get("cpus") == 4
            and a.get("memory_mib") == 12288
            and a.get("time_limit") == "3-00:00:00",
            "Recovery resource authorization differs",
        )
        self.source = Path(__file__).resolve(strict=True)
        require(
            self.source.is_relative_to(self.root / "operations")
            and sha(self.source) == a.get("source_sha256"),
            "Recovery source is not retained authorized source",
        )
        self.directory = bounded(self.root, a["recovery_directory"])
        require(
            self.directory.is_relative_to(self.root / "operations/delivery_recovery"),
            "Recovery journal outside namespace",
        )
        self.script = bounded(self.root, a["script"])
        require(
            self.script.parent == self.directory and sha(self.script) == a.get("script_sha256"),
            "Recovery script differs",
        )
        require(
            set(a.get("files", {})) >= REQUIRED_FILES,
            "Authorization omitted required original evidence",
        )
        self.check_originals()
        self.verify_supersession()
        extras = a.get("extra_sources", {})
        require(
            isinstance(extras, dict)
            and {Path(n).name for n in extras}
            >= {
                "verify_recovered_migration.py",
                "submit_delivery_recovery.py",
                "lowmem_history.py",
                "build_delivery.py",
                "profile_delivery_memory.py",
            },
            "Recovery auxiliary sources omitted",
        )
        for name, digest in extras.items():
            path = bounded(self.root, name)
            require(
                path.parent == self.source.parent and sha(path) == digest,
                "Recovery auxiliary source differs",
            )
        require(
            a.get("scope") == "DELIVERY_PACKAGING_AND_VERIFICATION_ONLY", "Recovery scope differs"
        )
        self.submission_path = self.directory / "SUBMITTED.json"
        self.archive = self.root / "delivery/SR_F3_CE_20261016.zip"
        self.builder_path = Path(str(self.archive) + ".verification.json")
        self.command = read(self.root / FAILED_ATTEMPT / "DELIVERY_INTENT.json")["command"]
        old = self.root / "operations" / OLD_OPERATION
        require(
            isinstance(self.command, list)
            and len(self.command) == 12
            and self.command[1] == str(old / "build_delivery.py")
            and self.command[2:6] == ["--root", str(self.root), "--output", str(self.archive)]
            and self.command[6:8] == ["--helper-manifest", str(old / "OPERATION_SOURCES.json")]
            and self.command[8] == "--extension-manifest"
            and self.command[10:12]
            == ["--operation-identity", str(old / "OPERATION_IDENTITY.json")],
            "Only original delivery builder command is allowed",
        )
        identity = read(old / "OPERATION_IDENTITY.json")
        require(
            self.command[0] == identity["python"]
            and self.command[9] == identity["extension_manifest_path"],
            "Builder interpreter or extension changed",
        )
        require(
            str(Path(self.command[0]).resolve(strict=True))
            == str(Path(identity["python"]).resolve(strict=True)),
            "Interpreter differs",
        )
        self.original_command = list(self.command)
        self.configure_revision()
        require(
            self.auth.get("expected_builder_identity") == self.expected_builder_identity(),
            "Authorized builder identity differs from frozen sources",
        )
        if require_submission:
            self.bind_submission()

    def revision_identity(self):
        binding = self.auth.get("delivery_revision_manifest", {})
        require(set(binding) == {"path", "sha256"}, "Revision manifest binding incomplete")
        path = bounded(self.root, binding["path"])
        require(
            path == self.source.parent / "DELIVERY_REVISION.json"
            and sha(path) == binding["sha256"],
            "Revision manifest differs",
        )
        builder = self.source.parent / "build_delivery.py"
        require(
            self.auth["extra_sources"].get(str(builder.relative_to(self.root))) == sha(builder),
            "Actual delivery builder source differs",
        )
        module = ModuleType("bound_lowmem_delivery_builder")
        module.__file__ = str(builder)
        exec(compile(builder.read_bytes(), str(builder), "exec"), module.__dict__)
        old = self.root / "operations" / OLD_OPERATION
        checked = module.load_delivery_revision(
            self.root,
            path,
            helper_manifest=old / "OPERATION_SOURCES.json",
            operation_identity=old / "OPERATION_IDENTITY.json",
        )
        return checked.identity

    def configure_revision(self):
        self.revision_identity()
        self.profile_path = self.directory / "MEMORY_PROFILE.json"
        require(
            self.auth.get("profile_receipt") == str(self.profile_path.relative_to(self.root)),
            "Profile receipt path differs",
        )
        old = self.original_command
        self.command = [
            old[0],
            "-B",
            str(self.source.parent / "profile_delivery_memory.py"),
            "--mode",
            "real-build",
            "--builder",
            str(self.source.parent / "build_delivery.py"),
            *old[2:],
            "--delivery-revision-manifest",
            str(self.source.parent / "DELIVERY_REVISION.json"),
            "--receipt",
            str(self.profile_path),
        ]

    @staticmethod
    def own_cgroup_memory():
        """Optional current-worker cgroup-v2 diagnostics, never another job's processes."""
        try:
            lines = Path("/proc/self/cgroup").read_text().splitlines()
            candidates = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
            require(len(candidates) == 1, "No unique cgroup v2 membership")
            base = Path("/sys/fs/cgroup").resolve(strict=True)
            path = (base / candidates[0].lstrip("/")).resolve(strict=True)
            require(path.is_relative_to(base), "Cgroup path escaped mount")
            peak = int((path / "memory.peak").read_text().strip())
            limit = (path / "memory.max").read_text().strip()
            return dict(
                status="OBSERVED",
                cgroup_path=str(path),
                peak_bytes=peak,
                memory_max=limit,
                scope="current worker cgroup; may be a step, not full Slurm job",
            )
        except (OSError, ValueError, PermissionError, IndexError) as exc:
            return dict(
                status="UNKNOWN",
                reason=type(exc).__name__,
                scope="current worker cgroup; not measured",
            )

    def verify_profile(self, builder_receipt):
        profile = read(self.profile_path)
        canonical = json.dumps(
            builder_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        require(
            profile.get("status") == "REAL_BUILD_RETURNED"
            and profile.get("exitcode") == 0
            and profile.get("science_execution_requested") is False
            and profile.get("output") == str(self.archive)
            and profile.get("builder") == str(self.source.parent / "build_delivery.py")
            and profile.get("builder_sha256") == sha(self.source.parent / "build_delivery.py")
            and profile.get("profile_source_sha256")
            == sha(self.source.parent / "profile_delivery_memory.py")
            and profile.get("result_sha256") == hashlib.sha256(canonical).hexdigest()
            and type(profile.get("peak_rss_bytes")) is int
            and profile["peak_rss_bytes"] > 0
            and profile.get("measurement_scope")
            == "builder child only; not Slurm/cgroup total or resource minimum",
            "Real builder memory measurement or archive result binding differs",
        )
        return profile

    def validate(self):
        require(read(self.authorization) == self.auth, "Recovery authorization changed")
        require(
            sha(self.source) == self.auth["source_sha256"]
            and sha(self.script) == self.auth["script_sha256"],
            "Recovery source or script changed",
        )
        for name, digest in self.auth["extra_sources"].items():
            require(sha(bounded(self.root, name)) == digest, "Recovery auxiliary source changed")
        self.check_originals()
        self.verify_supersession()
        require(
            self.auth.get("expected_builder_identity") == self.expected_builder_identity(),
            "Frozen builder identity differs",
        )
        return self.auth

    @staticmethod
    def check_superseded_accounting(row):
        require(
            row.get("JobIDRaw") == SUPERSEDED_JOB
            and row.get("State", "").split(" ", 1)[0] == "CANCELLED"
            and row.get("ExitCode") == "0:0"
            and row.get("Start") in {"Unknown", "None", ""}
            and row.get("ElapsedRaw") == "0"
            and row.get("AllocTRES") == ""
            and row.get("NodeList") in {"Unknown", "None", "None assigned", "", "(null)"},
            "Superseded recovery was not cancelled unstarted and unallocated",
        )

    def verify_supersession(self):
        binding = self.auth.get("supersession", {})
        require(
            set(binding) == {"path", "sha256"}, "Exact recovery supersession authorization required"
        )
        proof_path = bounded(self.root, binding["path"])
        require(sha(proof_path) == binding["sha256"], "Recovery supersession proof changed")
        proof = read(proof_path)
        require(
            proof.get("status") == "CANCELLED_UNSTARTED_RECOVERY"
            and proof.get("job_id") == SUPERSEDED_JOB
            and proof.get("prior_recovery_directory") == SUPERSEDED_DIRECTORY,
            "Recovery supersession scope differs",
        )
        expected = {
            "prior_authorization": f"operations/{SUPERSEDED_OPERATION}/RECOVERY_AUTHORIZATION.json",
            "prior_source": f"operations/{SUPERSEDED_OPERATION}/recover_delivery.py",
            "prior_script": SUPERSEDED_DIRECTORY + "/run.sbatch",
            "prior_submitted": SUPERSEDED_DIRECTORY + "/SUBMITTED.json",
        }
        retained = [proof_path]
        for key, relative in expected.items():
            require(
                proof.get(key + "_path") == relative, "Superseded recovery evidence path differs"
            )
            path = bounded(self.root, relative)
            require(sha(path) == proof.get(key + "_sha256"), "Superseded recovery evidence changed")
            retained.append(path)
        prior_auth = read(self.root / expected["prior_authorization"])
        require(
            prior_auth.get("root") == str(self.root)
            and prior_auth.get("recovery_directory") == SUPERSEDED_DIRECTORY
            and prior_auth.get("source_sha256") == proof["prior_source_sha256"]
            and prior_auth.get("script_sha256") == proof["prior_script_sha256"]
            and prior_auth.get("cpus") == 4
            and prior_auth.get("memory_mib") == 65536,
            "Superseded authorization identity differs",
        )
        submitted = read(self.root / expected["prior_submitted"])
        require(
            submitted.get("status") == "SUBMITTED"
            and submitted.get("job_id") == SUPERSEDED_JOB
            and submitted.get("authorization_sha256") == proof["prior_authorization_sha256"]
            and submitted.get("script_sha256") == proof["prior_script_sha256"],
            "Superseded submission binding differs",
        )
        prior = self.root / SUPERSEDED_DIRECTORY
        require(
            not any(
                (prior / name).exists()
                for name in (
                    "STARTED.json",
                    "PROCESS.json",
                    "END.json",
                    "DELIVERY_VERIFIED.json",
                    "RELEASE.json",
                    "RELEASE_INTENT.json",
                    "ALLOCATION_VERIFIED.json",
                )
            ),
            "Superseded recovery has launch, release, or completion evidence",
        )
        self.check_superseded_accounting(proof.get("terminal_accounting", {}))
        commands = proof.get("command_receipts", {})
        require(
            isinstance(commands, dict) and bool(commands), "Supersession command evidence missing"
        )
        cancellations = 0
        for name, digest in commands.items():
            path = bounded(self.root, name)
            require(sha(path) == digest, "Supersession command evidence changed")
            result = read(path)
            argv = result.get("argv", [])
            require(
                isinstance(argv, list)
                and all(isinstance(item, str) for item in argv)
                and result.get("returncode") == 0,
                "Supersession command outcome unknown or failed",
            )
            if argv == ["scancel", "--state=PENDING", SUPERSEDED_JOB]:
                cancellations += 1
            else:
                read_only = (
                    argv[:2] == ["scontrol", "show"]
                    or argv == ["scontrol", "write", "batch_script", SUPERSEDED_JOB, "-"]
                    or argv[:1] in [["sacct"], ["squeue"]]
                )
                require(read_only, "Supersession contains an unauthorized command")
            retained.append(path)
        require(
            cancellations == 1, "Supersession must prove exactly one precise pending cancellation"
        )
        return proof, retained

    def superseded_terminal(self):
        keys = ("JobIDRaw", "State", "ExitCode", "Start", "ElapsedRaw", "AllocTRES", "NodeList")
        command = [
            "sacct",
            "-n",
            "-P",
            "-j",
            SUPERSEDED_JOB,
            "--format=" + ",".join("State%100" if key == "State" else key for key in keys),
        ]
        raw = self.query(command)
        rows = [line.split("|") for line in raw.splitlines() if line.strip()]
        rows = [row[: len(keys)] for row in rows if row[0] == SUPERSEDED_JOB]
        require(len(rows) == 1 and len(rows[0]) == len(keys), "Superseded accounting ambiguous")
        row = dict(zip(keys, rows[0], strict=True))
        self.check_superseded_accounting(row)
        return dict(argv=command, stdout=raw, accounting=row)

    def bind_submission(self):
        self.teacher_history(require_retired=True)
        self.submission = read(self.submission_path)
        self.job_id = self.submission.get("job_id", "")
        require(
            isinstance(self.job_id, str)
            and re.fullmatch(r"[1-9]\d*", self.job_id)
            and self.job_id not in {"211333", "215520", "216128", "216142"},
            "Recovery needs distinct registered job",
        )
        require(
            self.submission.get("status") == "SUBMITTED"
            and self.submission.get("authorization_sha256") == sha(self.authorization)
            and self.submission.get("script_sha256") == sha(self.script),
            "Submission binding differs",
        )
        for name, status in [
            ("ALLOCATION_VERIFIED.json", "VERIFIED"),
            ("RELEASE.json", "RELEASED"),
        ]:
            record = read(self.directory / name)
            require(
                record.get("status") == status
                and record.get("job_id") == self.job_id
                and record.get("authorization_sha256") == sha(self.authorization)
                and record.get("script_sha256") == sha(self.script),
                "Recovery scheduling receipt differs: " + name,
            )
        require(
            read(self.directory / "RELEASE.json").get("returncode") == 0,
            "Recovery release did not succeed",
        )

    def submission_chain(self):
        source = self.source.parent / "submit_delivery_recovery.py"
        require(
            self.auth["extra_sources"].get(str(source.relative_to(self.root))) == sha(source),
            "Submission verifier source differs",
        )
        module = ModuleType("retained_delivery_submission_verifier")
        module.__file__ = str(source)
        source_bytes = source.read_bytes()
        require(
            hashlib.sha256(source_bytes).hexdigest()
            == self.auth["extra_sources"][str(source.relative_to(self.root))],
            "Submission verifier bytes changed",
        )
        exec(compile(source_bytes, str(source), "exec"), module.__dict__)
        submission = module.Submission(self.root, self.authorization, validator=lambda *_: None)
        submission.validate()
        release = submission.prior_release()
        require(release["job_id"] == self.job_id, "Submission chain binds another job")
        return release

    def expected_builder_identity(self):
        old = self.root / "operations" / OLD_OPERATION
        operation = read(old / "OPERATION_IDENTITY.json")
        extension_path = Path(operation["extension_manifest_path"])
        require(extension_path.is_relative_to(self.root), "Extension outside frozen root")
        extension = read(extension_path)
        ep = extension_path.parent
        required = [
            extension_path,
            ep / "observe_runtime.py",
            ep / "AMENDMENT.json",
            ep / "CPU_VALIDATION.json",
            self.root / "deployment/SOURCE_MANIFEST.json",
        ]
        for path in required:
            require(
                self.auth["files"].get(str(path.relative_to(self.root))) == sha(path),
                "Authorization omitted extension/source evidence",
            )
        observed = dict(
            manifest_path=str(extension_path.relative_to(self.root)),
            manifest_sha256=sha(extension_path),
            extension_commit=extension["extension_commit"],
            base_code_commit=extension["base_code_commit"],
            observer_source_sha256=sha(ep / "observe_runtime.py"),
            amendment_sha256=sha(ep / "AMENDMENT.json"),
            cpu_validation_sha256=sha(ep / "CPU_VALIDATION.json"),
        )
        if "protocol_source_guard" in extension:
            observed["protocol_source_guard"] = extension["protocol_source_guard"]
        require(
            operation["run_identity_sha256"] == sha(self.root / "RUN_IDENTITY.json")
            and operation["source_manifest_sha256"]
            == sha(self.root / "deployment/SOURCE_MANIFEST.json")
            and operation["extension_manifest_sha256"] == sha(extension_path),
            "Frozen operation source identity differs",
        )
        return dict(
            schema_version=1,
            status="VERIFIED_DELIVERY",
            run_identity_sha256=operation["run_identity_sha256"],
            scientific_source_manifest_sha256=operation["source_manifest_sha256"],
            helper_manifest_sha256=sha(old / "OPERATION_SOURCES.json"),
            observability_extension=observed,
            delivery_revision=self.revision_identity(),
            task_pool_operation=dict(
                operation_identity_path=str(
                    (old / "OPERATION_IDENTITY.json").relative_to(self.root)
                ),
                operation_identity_sha256=sha(old / "OPERATION_IDENTITY.json"),
                operation_commit=OLD_OPERATION,
                task_pool=operation["task_pool"],
            ),
        )

    def teacher_history(self, *, require_retired=False):
        path = self.authorization.parent / "lowmem_history.py"
        require(
            self.auth["extra_sources"].get(str(path.relative_to(self.root))) == sha(path),
            "Low-memory history source changed",
        )
        module = ModuleType("bound_teacher_history")
        module.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        return module.verify_history(self.root, self.auth, require_retired=require_retired)

    def check_originals(self):
        self.teacher_history()
        for name, digest in self.auth["files"].items():
            require(sha(bounded(self.root, name)) == digest, "Original evidence differs: " + name)
        failed = self.root / FAILED_ATTEMPT
        require(
            read(failed / "END.json").get("status") == "FAILED"
            and read(failed / "DELIVERY_PROCESS.json").get("returncode") == -9,
            "Failure is not bound builder OOM",
        )
        require(
            not (failed / "DELIVERY_VERIFIED.json").exists(),
            "Original dispatcher already verified delivery",
        )
        task_path = self.root / "tasks/release/COMPLETE.json"
        task = read(task_path)
        observer = read(self.root / f"accounting/observability/{RELEASE_KEY}/release/FINAL.json")
        require(
            task.get("status") == "COMPLETE"
            and task.get("job_id") == "215520"
            and task.get("run_identity_sha256") == sha(self.root / "RUN_IDENTITY.json")
            and task.get("execution_freeze_sha256") == sha(self.root / "EXECUTION_FREEZE.json")
            and observer.get("status") == "COMPLETE"
            and observer.get("task_receipt_sha256") == sha(task_path),
            "Frozen release or observer binding differs",
        )

    @contextmanager
    def lock(self):
        with (self.directory / "RECOVERY.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def query(self, argv):
        result = self.runner(argv, text=True, capture_output=True, check=False, timeout=30)
        require(result.returncode == 0, "Scheduler query failed")
        return result.stdout

    def failed_dispatcher_terminal(self):
        keys = ("JobIDRaw", "State", "ExitCode", "NodeList", "AllocTRES", "ElapsedRaw")
        command = ["sacct", "-n", "-P", "-j", "211333", "--format=" + ",".join(keys)]
        raw = self.query(command)
        rows = [line.rstrip("|").split("|") for line in raw.splitlines() if line.strip()]
        exact = [row for row in rows if row[0] == "211333"]
        require(
            len(exact) == 1 and len(exact[0]) == len(keys), "Failed dispatcher accounting ambiguous"
        )
        row = dict(zip(keys, exact[0], strict=True))
        require(
            row["State"] == "OUT_OF_MEMORY"
            and row["ExitCode"] == "0:125"
            and row["NodeList"] == "cpu-2"
            and int(row["ElapsedRaw"]) > 0,
            "Failed dispatcher is not the authorized terminal OOM",
        )
        allocation = dict(part.split("=", 1) for part in row["AllocTRES"].split(","))
        require(
            allocation.get("cpu") == "4"
            and memory_mib(allocation.get("mem", "")) == 12288
            and allocation.get("node") == "1"
            and all(not k.startswith("gres/gpu") or v == "0" for k, v in allocation.items()),
            "Failed dispatcher resources differ",
        )
        return dict(argv=command, stdout=raw, accounting=row)

    def live_allocation(self):
        require(os.environ.get("SLURM_JOB_ID") == self.job_id, "Wrong Slurm allocation")
        f = fields(self.query(["scontrol", "show", "job", "-o", self.job_id]))
        require(
            f.get("JobId") == self.job_id
            and f.get("JobState") == "RUNNING"
            and f.get("NodeList") in self.auth["allowed_nodes"]
            and f.get("ReqNodeList") in {"(null)", "", "None"}
            and f.get("Features") == "cpu_ok"
            and f.get("QOS") == "soujanya-poria-startfund-2026-03"
            and f.get("Account") == "rose"
            and f.get("UserId", "").split("(")[0] == self.auth["user"]
            and f.get("Comment") == self.auth["comment"]
            and f.get("JobName") == self.auth["job_name"]
            and f.get("Command") == str(self.script)
            and f.get("Partition") == "cluster02"
            and f.get("TimeLimit") == "3-00:00:00"
            and f.get("Requeue") == "0"
            and f.get("NumCPUs") == "4"
            and f.get("NumNodes") == "1",
            "Recovery live allocation differs",
        )
        resources(f.get("AllocTRES", ""))
        node = fields(self.query(["scontrol", "show", "node", f["NodeList"], "-o"]))
        require(
            node.get("NodeName") == f["NodeList"]
            and node.get("Arch") == "x86_64"
            and "cpu_ok" in node.get("ActiveFeatures", "").split(","),
            "Allocated CPU node architecture or feature differs",
        )
        actual = self.query(["scontrol", "write", "batch_script", self.job_id, "-"])
        require(
            hashlib.sha256(actual.encode()).hexdigest() == sha(self.script),
            "Scheduler script differs",
        )
        return f

    def verify_builder(self):
        self.check_originals()
        receipt = read(self.builder_path)
        expected = self.auth.get("expected_builder_identity", {})
        identity_keys = {
            "schema_version",
            "status",
            "run_identity_sha256",
            "scientific_source_manifest_sha256",
            "helper_manifest_sha256",
            "observability_extension",
            "task_pool_operation",
            "delivery_revision",
        }
        require(
            set(expected) == identity_keys
            and expected["schema_version"] == 1
            and expected["status"] == "VERIFIED_DELIVERY"
            and expected["run_identity_sha256"] == sha(self.root / "RUN_IDENTITY.json")
            and expected["helper_manifest_sha256"]
            == sha(self.root / f"operations/{OLD_OPERATION}/OPERATION_SOURCES.json"),
            "Expected builder identity incomplete",
        )
        require(all(receipt.get(k) == v for k, v in expected.items()), "Builder identity differs")
        require(
            all(receipt.get("checks", {}).get(k) == "PASS" for k in CHECKS),
            "Builder checks incomplete",
        )
        require(
            all(
                type(receipt.get("coverage", {}).get(k)) is int and receipt["coverage"][k] == v
                for k, v in COVERAGE.items()
            ),
            "Builder coverage incomplete",
        )
        archive = receipt.get("archive", {})
        require(
            archive.get("path") == str(self.archive)
            and archive.get("sha256") == sha(self.archive)
            and archive.get("bytes") == self.archive.stat().st_size
            and type(archive.get("entries")) is int
            and archive["entries"] > 0,
            "Archive identity differs",
        )
        self.verify_profile(receipt)
        return receipt

    def run(self):
        with self.lock():
            self.validate()
            self.bind_submission()
            self.submission_chain()
            require(
                not any(
                    (self.directory / n).exists()
                    for n in ("STARTED.json", "PROCESS.json", "END.json", "MEMORY_PROFILE.json")
                ),
                "Recovery already attempted; no blind retries",
            )
            require(
                not self.archive.exists() and not self.builder_path.exists(),
                "Delivery output already exists",
            )
            allocation = self.live_allocation()
            failed_terminal = self.failed_dispatcher_terminal()
            superseded_terminal = self.superseded_terminal()
            save(
                self.directory / "STARTED.json",
                dict(
                    status="RUNNING",
                    at=now(),
                    job_id=self.job_id,
                    authorization_sha256=sha(self.authorization),
                    source_sha256=sha(self.source),
                    submission_sha256=sha(self.submission_path),
                    script_sha256=sha(self.script),
                    allocation=allocation,
                    failed_dispatcher_terminal=failed_terminal,
                    superseded_recovery_terminal=superseded_terminal,
                    command=self.command,
                ),
            )
            start = time.monotonic()
            try:
                with (
                    (self.directory / "builder.stdout.log").open("xb") as out,
                    (self.directory / "builder.stderr.log").open("xb") as err,
                ):
                    result = self.runner(self.command, stdout=out, stderr=err, check=False)
                    out.flush()
                    err.flush()
                    os.fsync(out.fileno())
                    os.fsync(err.fileno())
                save(
                    self.directory / "PROCESS.json",
                    dict(
                        status="RETURNED",
                        at=now(),
                        job_id=self.job_id,
                        command=self.command,
                        returncode=result.returncode,
                        elapsed_seconds=time.monotonic() - start,
                        cgroup_memory=self.own_cgroup_memory(),
                        stdout_sha256=sha(self.directory / "builder.stdout.log"),
                        stderr_sha256=sha(self.directory / "builder.stderr.log"),
                    ),
                )
                require(result.returncode == 0, "Delivery builder failed")
                self.verify_builder()
                save(
                    self.directory / "DELIVERY_VERIFIED.json",
                    dict(receipt=str(self.builder_path), sha256=sha(self.builder_path)),
                )
                save(
                    self.directory / "END.json",
                    dict(
                        status="DONE",
                        at=now(),
                        job_id=self.job_id,
                        authorization_sha256=sha(self.authorization),
                        delivery_verified_sha256=sha(self.directory / "DELIVERY_VERIFIED.json"),
                    ),
                )
                return read(self.directory / "END.json")
            except BaseException as exc:
                if not (self.directory / "END.json").exists():
                    save(
                        self.directory / "END.json",
                        dict(status="FAILED", at=now(), job_id=self.job_id, error=repr(exc)),
                    )
                raise

    def terminal(self):
        keys = (
            "JobIDRaw",
            "State",
            "ExitCode",
            "NodeList",
            "AllocTRES",
            "User",
            "Account",
            "QOS",
            "JobName",
            "ElapsedRaw",
            "Timelimit",
        )
        raw = self.query(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                self.job_id,
                "--format=" + ",".join(k + "%100" if k == "JobName" else k for k in keys),
            ]
        )
        rows = [line.rstrip("|").split("|") for line in raw.splitlines() if line.strip()]
        rows = [row for row in rows if row[0] == self.job_id]
        require(len(rows) == 1 and len(rows[0]) == len(keys), "Ambiguous terminal accounting")
        f = dict(zip(keys, rows[0], strict=True))
        require(
            f["State"] == "COMPLETED"
            and f["ExitCode"] == "0:0"
            and f["NodeList"] in self.auth["allowed_nodes"]
            and f["User"] == self.auth["user"]
            and f["Account"] == "rose"
            and f["QOS"] == "soujanya-poria-startfund-2026-03"
            and f["JobName"] == self.auth["job_name"]
            and int(f["ElapsedRaw"]) > 0
            and f["Timelimit"] == "3-00:00:00",
            "Terminal recovery execution differs",
        )
        resources(f["AllocTRES"])
        return f

    def verify_final(self):
        # Independent lock; never takes the old dispatcher's nonblocking SUBMIT.lock.
        with self.lock():
            self.validate()
            self.bind_submission()
            release_chain = self.submission_chain()
            end = read(self.directory / "END.json")
            require(
                end.get("status") == "DONE"
                and end.get("job_id") == self.job_id
                and end.get("authorization_sha256") == sha(self.authorization)
                and end.get("delivery_verified_sha256")
                == sha(self.directory / "DELIVERY_VERIFIED.json"),
                "Recovery not DONE",
            )
            started = read(self.directory / "STARTED.json")
            require(
                started.get("job_id") == self.job_id
                and started.get("command") == self.command
                and started.get("authorization_sha256") == sha(self.authorization)
                and started.get("source_sha256") == sha(self.source)
                and started.get("script_sha256") == sha(self.script)
                and started.get("submission_sha256") == sha(self.submission_path),
                "Recovery start binding differs",
            )
            process = read(self.directory / "PROCESS.json")
            require(
                process.get("status") == "RETURNED"
                and process.get("job_id") == self.job_id
                and process.get("command") == self.command
                and process.get("returncode") == 0
                and process.get("stdout_sha256") == sha(self.directory / "builder.stdout.log")
                and process.get("stderr_sha256") == sha(self.directory / "builder.stderr.log"),
                "Recovery process differs",
            )
            failed_terminal = self.failed_dispatcher_terminal()
            require(
                started.get("failed_dispatcher_terminal") == failed_terminal,
                "Failed dispatcher terminal evidence changed",
            )
            superseded_terminal = self.superseded_terminal()
            require(
                started.get("superseded_recovery_terminal", {}).get("accounting")
                == superseded_terminal["accounting"],
                "Superseded terminal evidence changed",
            )
            account = self.terminal()
            require(
                started.get("allocation", {}).get("NodeList") == account["NodeList"],
                "Recovery terminal node differs from started allocation",
            )
            builder = self.verify_builder()
            proof = read(self.directory / "DELIVERY_VERIFIED.json")
            require(
                proof == dict(receipt=str(self.builder_path), sha256=sha(self.builder_path)),
                "Recovery delivery binding differs",
            )
            pre_run = [
                self.authorization,
                bounded(self.root, self.auth["delivery_revision_manifest"]["path"]),
                self.source,
                self.script,
                self.submission_path,
                self.directory / "STARTED.json",
                self.directory / "ALLOCATION_VERIFIED.json",
                self.directory / "RELEASE.json",
                *[bounded(self.root, n) for n in self.auth["extra_sources"]],
                *[bounded(self.root, n) for n in self.auth["files"]],
            ]
            pre_run.extend(self.directory / name for name in release_chain["chain"])
            pre_run.extend(self.directory / name for name in release_chain["command_files"])
            _, superseded_files = self.verify_supersession()
            pre_run.extend(superseded_files)
            pre_run.extend(self.teacher_history(require_retired=True))
            pre_run.extend(
                self.directory / name for name in release_chain.get("teacher_audit_files", {})
            )
            archived = {}
            with zipfile.ZipFile(self.archive) as package:
                names = package.namelist()
                require(
                    len(names) == len(set(names)) == builder["archive"]["entries"],
                    "Archive entry count differs",
                )
                for path in sorted(set(pre_run)):
                    name = "evidence/" + str(path.relative_to(self.root))
                    require(name in names, "Recovery evidence absent from ZIP: " + name)
                    h = hashlib.sha256()
                    with package.open(name) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(chunk)
                    require(
                        h.hexdigest() == sha(path), "Archived recovery evidence differs: " + name
                    )
                    archived[name] = h.hexdigest()
            result = dict(
                schema_version=1,
                status="FINAL_RECOVERY_VERIFIED",
                at=now(),
                job_id=self.job_id,
                failed_job_id="211333",
                failed_attempt=FAILED_ATTEMPT,
                recovery_directory=str(self.directory.relative_to(self.root)),
                authorization_sha256=sha(self.authorization),
                source_sha256=sha(self.source),
                archive=builder["archive"],
                builder_receipt_sha256=sha(self.builder_path),
                delivery_revision=builder["delivery_revision"],
                memory_profile_sha256=sha(self.profile_path),
                memory_profile=self.verify_profile(builder),
                resource_measurement_scope="builder child RSS; not cgroup total or proven minimum",
                cgroup_memory=process.get("cgroup_memory", {"status": "UNKNOWN"}),
                accounting=account,
                failed_dispatcher_terminal=failed_terminal,
                superseded_recovery_terminal=superseded_terminal,
                recovery_end_sha256=sha(self.directory / "END.json"),
                recovery_process_sha256=sha(self.directory / "PROCESS.json"),
                recovery_delivery_sha256=sha(self.directory / "DELIVERY_VERIFIED.json"),
                archived_recovery_files=archived,
                acceptance=(
                    "Revised streaming builder with unchanged scientific acceptance AND "
                    "independent delivery-only recovery "
                    "verification; migration verification still required"
                ),
            )
            target = self.root / "delivery/FINAL_RECOVERY_VERIFIED.json"
            if target.exists():
                old = read(target)
                require(
                    all(old.get(k) == v for k, v in result.items() if k != "at"),
                    "Existing final recovery receipt differs",
                )
                return old
            save(target, result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--mode", choices=("run", "verify-final"), required=True)
    args = parser.parse_args()
    if args.mode == "run":
        prepared = Recovery(args.root, args.authorization, require_submission=False)
        deadline = time.monotonic() + 60
        while not (prepared.directory / "RELEASE.json").exists() and time.monotonic() < deadline:
            time.sleep(1)
        require(
            (prepared.directory / "RELEASE.json").exists(),
            "Recovery release receipt unavailable after 60 seconds",
        )
    operation = Recovery(args.root, args.authorization)
    print(
        json.dumps(
            operation.run() if args.mode == "run" else operation.verify_final(), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
