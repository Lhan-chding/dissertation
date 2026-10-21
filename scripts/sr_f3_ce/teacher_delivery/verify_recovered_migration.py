#!/usr/bin/env python3
"""Certify the original CPU migration after an explicitly authorized delivery recovery.

The original migration helper, failed dispatcher END, and scientific builder stay
unchanged. This gate retains the original migration checks and uses the independent
recovery terminal acceptance in place of the failed dispatcher's impossible DONE.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
import zipfile
from pathlib import Path

TARGET_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2"
MIGRATION_COMMIT = "3a7f5d120e02dcd042397408c6031a98713d1e7d"


def load_bound(path, expected, name):
    path = Path(path).resolve(strict=True)
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise PermissionError("Bound verifier source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


def verify(root, authorization, *, runner=subprocess.run):
    root = Path(root).resolve(strict=True)
    authorization = Path(authorization).resolve(strict=True)
    if str(root) != TARGET_ROOT or not authorization.is_relative_to(root / "operations"):
        raise PermissionError("Only the exact registered recovery is allowed")
    auth = json.loads(authorization.read_text())
    source = Path(__file__).resolve(strict=True)
    if (
        source.parent != authorization.parent
        or auth.get("extra_sources", {}).get(str(source.relative_to(root)))
        != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        raise PermissionError("Recovery migration verifier source differs")
    recovery_module = load_bound(
        authorization.parent / "recover_delivery.py",
        auth["source_sha256"],
        "srf3ce_bound_delivery_recovery",
    )
    recovery = recovery_module.Recovery(root, authorization, runner=runner)
    recovery_result = recovery.verify_final()
    old = root / "operations" / MIGRATION_COMMIT
    migration_source = old / "migrate_cpu_node.py"
    m = load_bound(
        migration_source,
        auth["files"][str(migration_source.relative_to(root))],
        "srf3ce_original_cpu_migration",
    )
    self = m.Migration(root, old / "MIGRATION_AUTHORIZATION.json", runner=runner)
    self.verify_chain()
    read, sha, require, bounded = m.read, m.sha, m.require, m.bounded
    parse_accounting, tres, memory_mib = m.parse_accounting, m.tres, m.memory_mib
    now, save = m.now, m.save
    JOB_ID, JOB_KEY, CLAIM_ID = m.JOB_ID, m.JOB_KEY, m.CLAIM_ID
    ACCOUNT_FORMAT, TEACHER_QOS, DELIVERY_CHECKS = (
        m.ACCOUNT_FORMAT,
        m.TEACHER_QOS,
        m.DELIVERY_CHECKS,
    )
    dispatcher_directory = root / "operations/dispatcher/attempts/0007"
    dispatcher_end_path = dispatcher_directory / "END.json"
    require(read(dispatcher_end_path).get("status") == "FAILED", "Original failure was rewritten")

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
    require(read(self.job / "STARTED.json").get("job_id") == JOB_ID, "Start receipt differs")
    launch = read(self.job / "release.LAUNCH.json")
    require(
        launch.get("claim_id") == CLAIM_ID
        and launch.get("device") == ""
        and launch.get("command") == self.intent["worker_entry_commands"]["release"],
        "Release launch binding differs",
    )
    require(read(self.job / "release.EXIT.json").get("returncode") == 0, "Release exit differs")
    task_path = self.root / "tasks/release/COMPLETE.json"
    task = read(task_path)
    require(
        task.get("job_id") == JOB_ID and task.get("claim_id") == CLAIM_ID,
        "Release task claim differs",
    )
    require(
        task.get("status") == "COMPLETE"
        and task.get("run_identity_sha256") == self.auth["files"]["RUN_IDENTITY.json"]
        and task.get("execution_freeze_sha256") == self.auth["files"]["EXECUTION_FREEZE.json"],
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
    recovery_path = root / "delivery/FINAL_RECOVERY_VERIFIED.json"
    require(read(recovery_path) == recovery_result, "Recovery final receipt changed")
    require(recovery_result.get("archive") == archive, "Recovery archive binding differs")
    migration_files = [
        self.journal / "MIGRATED.json",
        self.authorization,
        *[self.journal / name for name in read(self.journal / "MIGRATED.json")["journal_files"]],
        *[bounded(self.root, name) for name in self.auth["files"]],
    ]
    source = Path(m.__file__).resolve(strict=True)
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
        original_failed_dispatcher_end_sha256=sha(dispatcher_end_path),
        recovery_final_sha256=sha(recovery_path),
        recovery_authorization_sha256=sha(authorization),
        recovery_verifier_source_sha256=sha(Path(__file__)),
        recovery_job_id=recovery_result["job_id"],
        task_receipt_sha256=sha(task_path),
        observer_final_sha256=sha(observer_path),
        archived_migration_files=archived,
        acceptance=(
            "Original scientific delivery verification "
            "AND authorized delivery recovery verification "
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
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.authorization), sort_keys=True))


if __name__ == "__main__":
    main()
