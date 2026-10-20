#!/usr/bin/env python3
"""Continue only the registered 216142 capacity stop; preserve both prior journals."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

TARGET_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2"
OLD_COMMIT = "249539afb0e3d8b25e750a0076207d7338deb06d"
OLD_AUTH_SHA = "9bd7d76cb4b81dbd4df3dab1de5d70e3d2b0a04886f15f5fece142b102cf5e57"
JOB = "216142"
WORK = "operations/delivery_recovery/20261021_delivery_oom_cpu22"
OLD_COMMANDS = WORK + "/commands/1792520258958164529"
AUDIT = "operations/delivery_recovery_resume/216142_20261021"
INVOCATION = "resume_216142_20261021"
SOURCES = ("recover_delivery.py", "submit_delivery_recovery.py", "verify_recovered_migration.py")
ORIGINALS = (
    "run.sbatch",
    "SUBMITTED.json",
    "MAY_HAVE_SUBMITTED.json",
    "CAPACITY_BEFORE_SUBMIT.json",
)
CHAIN = (
    "MAY_HAVE_SUBMITTED.json",
    "SUBMITTED.json",
    "ALLOCATION_VERIFIED.json",
    "CAPACITY_BEFORE_SUBMIT.json",
    "CAPACITY_BEFORE_RELEASE.json",
    "RELEASE_INTENT.json",
)


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def save_bytes(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="." + path.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save(path, value):
    save_bytes(path, encoded(value))


def bounded(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe evidence path")
    target = root / relative
    require(
        target.resolve().is_relative_to(root) and not target.is_symlink(), "Evidence path escapes"
    )
    return target


def load_bound(path, digest, name):
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == digest, "Bound source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


HELD_COMMIT = "ab7c9d45065d1af58f49d4d62385a5bd0eba14dc"
JOURNAL = "operations/delivery_recovery_continue/216142_20261021"
NEW_INVOCATION = "continue_216142_20261021"
WAIT_REASONS = {
    "Rose capacity unavailable",
    "Association capacity unavailable",
    "Node unavailable",
    "Insufficient CPUs",
    "Insufficient memory",
}


class Continuation:
    def __init__(self, root, authorization, *, runner=subprocess.run):
        self.root = Path(root).resolve(strict=True)
        self.authorization = Path(authorization).resolve(strict=True)
        self.auth = read(self.authorization)
        self.auth_sha = sha(self.authorization)
        self.source = Path(__file__).resolve(strict=True)
        self.runner = runner
        self.work = self.root / WORK
        self.journal = self.root / JOURNAL
        self.prior_auth = self.root / "operations" / HELD_COMMIT / "HELD_RESUME_AUTHORIZATION.json"

    def validate(self):
        a = self.auth
        require(
            str(self.root) == TARGET_ROOT and sha(self.authorization) == self.auth_sha,
            "Continuation root or authorization differs",
        )
        fixed = dict(
            schema_version=1,
            status="AUTHORIZED_HELD_CAPACITY_CONTINUATION",
            scope="EXACT_216142_AB7_CAPACITY_STOP_ONLY",
            root=str(self.root),
            job_id=JOB,
            original_operation_commit=OLD_COMMIT,
            held_operation_commit=HELD_COMMIT,
            recovery_directory=WORK,
            journal_directory=JOURNAL,
            resumed_invocation=NEW_INVOCATION,
        )
        require(
            all(type(a.get(k)) is type(v) and a.get(k) == v for k, v in fixed.items()),
            "Continuation scope differs",
        )
        commit = a.get("operation_commit", "")
        require(
            bool(re.fullmatch("[0-9a-f]{40}", commit))
            and self.authorization.parent
            == self.source.parent
            == self.root / "operations" / commit,
            "Continuation deployment differs",
        )
        require(sha(self.source) == a.get("source_sha256"), "Continuation source changed")
        if "user_request_sha256" in a:
            require(
                isinstance(a.get("user_request"), str)
                and hashlib.sha256(a["user_request"].encode()).hexdigest()
                == a["user_request_sha256"],
                "User request binding differs",
            )
        files = a["files"]
        for name, digest in files.items():
            require(
                sha(bounded(self.root, name)) == digest, "Pinned partial evidence changed: " + name
            )
        require(
            a.get("prior_authorization")
            == dict(path=str(self.prior_auth.relative_to(self.root)), sha256=sha(self.prior_auth)),
            "Prior authorization differs",
        )
        prior_source = self.prior_auth.parent / "resume_held_delivery.py"
        self.p = load_bound(
            prior_source,
            files[str(prior_source.relative_to(self.root))],
            "capacity_continuation_prior_resume",
        )
        self.prior = self.p.Resume(self.root, self.prior_auth, runner=self.runner).validate()
        self.sub, self.m = self.prior.sub, self.prior.m
        expected = set(self.prior.auth["files"]) | {
            str(self.prior_auth.relative_to(self.root)),
            str(prior_source.relative_to(self.root)),
            WORK + "/ALLOCATION_VERIFIED.json",
        }
        expected.update(
            WORK + f"/commands/{INVOCATION}/{i:03d}_{label}.{kind}.json"
            for i, (label, _) in enumerate(self.sub.expected_commands(JOB)[:12], 1)
            for kind in ("intent", "result")
        )
        audit = self.root / AUDIT
        audit_names = {"RESUME_INTENT.json", "PRECHECK_VERIFIED.json", "COPY_LINEAGE.json"}
        audit_names.update(
            f"commands/precheck/{i:03d}_{label}.{kind}.json"
            for i, (label, _) in enumerate(self.sub.expected_commands(JOB)[5:8], 1)
            for kind in ("intent", "result")
        )
        expected.update(AUDIT + "/" + name for name in audit_names)
        require(set(files) == expected, "Exact partial inventory differs")
        require(
            {str(p.relative_to(audit)) for p in audit.rglob("*") if p.is_file()} == audit_names,
            "Prior resume audit advanced",
        )
        partial = self.work / "commands" / INVOCATION
        names = {str(p.relative_to(self.root)) for p in partial.iterdir()}
        require(
            names
            == {n for n in expected if n.startswith(str(partial.relative_to(self.root)) + "/")},
            "Prior resume command inventory advanced",
        )
        self.verify_partial()
        return self

    def pairs(self, directory, commands, *, first=1):
        result = []
        for i, (label, argv) in enumerate(commands, first):
            pair = [
                read(directory / f"{i:03d}_{label}.{kind}.json") for kind in ("intent", "result")
            ]
            require(
                all(r.get("argv") == argv for r in pair) and pair[1].get("returncode") == 0,
                "Read-only command differs",
            )
            require(
                pair[0]["at"] <= pair[1]["at"]
                and (not result or result[-1][1]["at"] <= pair[0]["at"]),
                "Command chronology differs",
            )
            result.append(pair)
        return result

    def verify_partial(self):
        commands = self.sub.expected_commands(JOB)
        directory = self.work / "commands" / INVOCATION
        pairs = self.pairs(directory, commands[:12])
        old = self.root / OLD_COMMANDS
        expected_lineage = {}
        for i, (label, _) in enumerate(commands[:7], 1):
            for kind in ("intent", "result"):
                name = f"{i:03d}_{label}.{kind}.json"
                require(
                    (directory / name).read_bytes() == (old / name).read_bytes(),
                    "Prior copied history differs",
                )
                expected_lineage[str((directory / name).relative_to(self.root))] = dict(
                    original=str((old / name).relative_to(self.root)), sha256=sha(old / name)
                )
        require(
            read(self.root / AUDIT / "COPY_LINEAGE.json")["files"] == expected_lineage,
            "Prior copy lineage differs",
        )
        require(
            pairs[7][1]["stdout"].strip().splitlines()
            in ([JOB + "|PENDING|Unknown|0|"], [JOB + "|PENDING|None|0|"]),
            "Prior accounting did not prove never started",
        )
        allocation = read(self.work / "ALLOCATION_VERIFIED.json")
        require(
            all(
                allocation.get(k) == v
                for k, v in dict(
                    job_id=JOB,
                    status="VERIFIED",
                    fields=self.m.fields(pairs[5][1]["stdout"]),
                    authorization_sha256=self.p.OLD_AUTH_SHA,
                    script_sha256=sha(self.sub.script),
                ).items()
            ),
            "Original allocation proof differs",
        )
        audit = self.root / AUDIT
        intent, precheck, lineage = (
            read(audit / n)
            for n in ("RESUME_INTENT.json", "PRECHECK_VERIFIED.json", "COPY_LINEAGE.json")
        )
        require(
            intent.get("job_id") == JOB
            and intent.get("authorization_sha256") == sha(self.prior_auth)
            and intent.get("original_authorization_sha256") == self.p.OLD_AUTH_SHA,
            "Prior resume intent differs",
        )
        pre_pairs = self.pairs(audit / "commands/precheck", commands[5:8])
        self.check_held_results(pre_pairs)
        require(
            precheck["fields"] == self.m.fields(pre_pairs[0][1]["stdout"]),
            "Prior held proof differs",
        )
        require(
            precheck["command_files"]
            == {str(p.relative_to(audit)): sha(p) for p in (audit / "commands/precheck").iterdir()},
            "Prior precheck binding differs",
        )
        require(
            self.prior.pairs[7][1]["at"]
            <= intent["at"]
            <= pre_pairs[0][0]["at"]
            <= pre_pairs[-1][1]["at"]
            <= precheck["at"]
            <= lineage["at"]
            <= pairs[7][0]["at"]
            <= allocation["at"]
            <= pairs[8][0]["at"],
            "Prior partial chronology differs",
        )
        node = self.m.fields(pairs[11][1]["stdout"])
        require(
            node.get("NodeName") == "cpu-1"
            and (
                node.get("State") not in ("IDLE", "MIXED")
                or int(node["CPUEfctv"]) - int(node["CPUAlloc"]) < 22
                or int(node["RealMemory"])
                - int(node["AllocMem"])
                - int(node.get("MemSpecLimit", "0"))
                < 67584
            ),
            "Prior node result is not the authorized capacity stop",
        )

    def check_held_results(self, pairs):
        self.sub.check_held_fields(self.m.fields(pairs[0][1]["stdout"]), JOB)
        require(
            hashlib.sha256(pairs[1][1]["stdout"].encode()).hexdigest() == sha(self.sub.script),
            "Held script differs",
        )
        require(
            pairs[2][1]["stdout"].strip().splitlines()
            in ([JOB + "|PENDING|Unknown|0|"], [JOB + "|PENDING|None|0|"]),
            "Held accounting differs",
        )

    def is_capacity_wait(self, reason, directory):
        if reason not in WAIT_REASONS:
            return False
        if reason != "Node unavailable":
            return True
        node = self.m.fields(read(directory / "commands/probe/007_node.result.json")["stdout"])
        return node.get("State") in {
            "IDLE+COMPLETING",
            "MIXED+COMPLETING",
            "ALLOCATED",
            "ALLOCATED+COMPLETING",
        } and "cpu_ok" in node.get("ActiveFeatures", "").split(",")

    def poll(self, directory):
        self.sub.work, self.sub.invocation, self.sub.serial = directory, "probe", 0
        self.sub.verify_held(JOB)
        try:
            cap = self.sub.capacity(own_job=JOB)
        except PermissionError as exc:
            if not self.is_capacity_wait(str(exc), directory):
                raise
            result = dict(status="WAIT_CAPACITY", reason=str(exc), at=time.time())
        else:
            result = dict(status="READY", capacity=cap, at=time.time())
        result["files"] = {
            str(p.relative_to(directory)): sha(p)
            for p in sorted((directory / "commands/probe").iterdir())
        }
        save(directory / "RESULT.json", result)
        self.sub.work = self.work
        return result

    def replay_poll(self, directory):
        result = read(directory / "RESULT.json")
        require(result.get("status") in ("WAIT_CAPACITY", "READY"), "Poll failure requires review")
        files = result["files"]
        require(
            {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
            == set(files) | {"RESULT.json"},
            "Poll evidence inventory differs",
        )
        for name, digest in files.items():
            require(sha(bounded(directory, name)) == digest, "Poll evidence changed")
        index, previous = 0, 0
        old_command = self.sub.command

        def replay(label, argv):
            nonlocal index, previous
            index += 1
            names = [
                f"commands/probe/{index:03d}_{label}.{kind}.json" for kind in ("intent", "result")
            ]
            require(all(n in files for n in names), "Poll command missing")
            pair = [read(directory / n) for n in names]
            require(
                all(r.get("argv") == argv for r in pair)
                and pair[1].get("returncode") == 0
                and previous <= pair[0]["at"] <= pair[1]["at"],
                "Poll command differs",
            )
            previous = pair[1]["at"]
            return pair[1]["stdout"]

        self.sub.command = replay
        try:
            self.sub.verify_held(JOB)
            try:
                cap = self.sub.capacity(own_job=JOB)
            except PermissionError as exc:
                require(
                    self.is_capacity_wait(str(exc), directory)
                    and result.get("status") == "WAIT_CAPACITY"
                    and result.get("reason") == str(exc),
                    "Unrecognized capacity wait",
                )
            else:
                require(
                    result.get("status") == "READY"
                    and {k: v for k, v in cap.items() if k != "at"}
                    == {k: v for k, v in result["capacity"].items() if k != "at"},
                    "Ready capacity proof differs",
                )
                require(
                    previous <= result["capacity"]["at"] <= result["at"],
                    "Capacity chronology differs",
                )
            require(len(files) == 2 * index and previous <= result["at"], "Extra poll commands")
        finally:
            self.sub.command = old_command
        return result

    def run(self):
        self.validate()
        with (self.work / "SUBMIT.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.work / "RELEASE.json").exists():
                return self.verify_audit()
            require(
                not any(
                    (self.work / n).exists()
                    for n in (
                        "CAPACITY_BEFORE_RELEASE.json",
                        "RELEASE_INTENT.json",
                        "STARTED.json",
                        "PROCESS.json",
                        "END.json",
                    )
                )
                and not (self.journal / "CONTINUE_INTENT.json").exists()
                and not (self.work / "commands" / NEW_INVOCATION).exists(),
                "Continuation has advanced or outcome unknown; no retry",
            )
            polls = self.journal / "polls"
            for directory in sorted(polls.glob("*")):
                require(
                    self.replay_poll(directory)["status"] == "WAIT_CAPACITY",
                    "Prior ready attempt exists",
                )
            directory = polls / str(time.time_ns())
            result = self.poll(directory)
            if result["status"] == "WAIT_CAPACITY":
                return result
            self.replay_poll(directory)
            cap = result["capacity"]
            require(
                0 <= time.time() - cap["at"] <= 60, "Ready capacity stale; inspect before retry"
            )
            save(
                self.journal / "CONTINUE_INTENT.json",
                dict(
                    at=time.time(),
                    job_id=JOB,
                    authorization_sha256=self.auth_sha,
                    selected_poll=str(directory.relative_to(self.journal)),
                ),
            )
            new = self.work / "commands" / NEW_INVOCATION
            lineage = {}
            commands = self.sub.expected_commands(JOB)
            for i, (label, _) in enumerate(commands[:12], 1):
                for kind in ("intent", "result"):
                    name = f"{i:03d}_{label}.{kind}.json"
                    original = (
                        self.work / "commands" / INVOCATION / name
                        if i <= 8
                        else directory / "commands/probe" / f"{i - 5:03d}_{label}.{kind}.json"
                    )
                    save_bytes(new / name, original.read_bytes())
                    lineage[str((new / name).relative_to(self.root))] = dict(
                        original=str(original.relative_to(self.root)), sha256=sha(original)
                    )
            save(self.journal / "COPY_LINEAGE.json", dict(at=time.time(), files=lineage))
            save(self.work / "CAPACITY_BEFORE_RELEASE.json", cap)
            self.validate()
            self.sub.work, self.sub.invocation, self.sub.serial = self.work, NEW_INVOCATION, 12
            self.sub.verify_held(JOB)
            require(0 <= time.time() - cap["at"] <= 60, "Release capacity snapshot stale")
            save(
                self.work / "RELEASE_INTENT.json",
                dict(
                    job_id=JOB,
                    at=time.time(),
                    capacity_sha256=sha(self.work / "CAPACITY_BEFORE_RELEASE.json"),
                ),
            )
            output = self.sub.command("release", ["scontrol", "release", JOB])
            release = dict(
                job_id=JOB,
                status="RELEASED",
                stdout=output,
                stderr="",
                returncode=0,
                at=time.time(),
                authorization_sha256=self.p.OLD_AUTH_SHA,
                script_sha256=sha(self.sub.script),
                chain={n: sha(self.work / n) for n in CHAIN},
                command_files={
                    str(p.relative_to(self.work)): sha(p) for p in sorted(new.iterdir())
                },
            )
            self.sub.check_command_chain(release)
            proof = dict(
                status="CONTINUED",
                job_id=JOB,
                at=time.time(),
                authorization_sha256=self.auth_sha,
                release_sha256=hashlib.sha256(encoded(release)).hexdigest(),
                files={
                    str(p.relative_to(self.journal)): sha(p)
                    for p in sorted(self.journal.rglob("*"))
                    if p.is_file()
                },
            )
            save(self.journal / "CONTINUED.json", proof)
            save(self.work / "RELEASE.json", release)
            return self.verify_audit()

    def verify_audit(self):
        self.validate()
        release = self.sub.prior_release()
        require(release["job_id"] == JOB, "Continuation job differs")
        proof = read(self.journal / "CONTINUED.json")
        require(
            proof.get("status") == "CONTINUED"
            and proof.get("job_id") == JOB
            and proof.get("authorization_sha256") == self.auth_sha
            and proof.get("release_sha256") == sha(self.work / "RELEASE.json"),
            "Continuation binding differs",
        )
        require(
            {str(p.relative_to(self.journal)) for p in self.journal.rglob("*") if p.is_file()}
            == set(proof["files"]) | {"CONTINUED.json"},
            "Continuation inventory differs",
        )
        for name, digest in proof["files"].items():
            require(sha(bounded(self.journal, name)) == digest, "Continuation audit changed")
        intent = read(self.journal / "CONTINUE_INTENT.json")
        require(
            intent.get("job_id") == JOB and intent.get("authorization_sha256") == self.auth_sha,
            "Continuation intent differs",
        )
        selected = bounded(self.journal, intent["selected_poll"])
        polls = sorted((self.journal / "polls").iterdir())
        require(selected in polls and selected == polls[-1], "Selected poll differs")
        for directory in polls:
            result = self.replay_poll(directory)
            require(
                result["status"] == ("READY" if directory == selected else "WAIT_CAPACITY"),
                "Poll selection differs",
            )
        result = read(selected / "RESULT.json")
        require(
            read(self.work / "CAPACITY_BEFORE_RELEASE.json") == result["capacity"],
            "Selected capacity differs",
        )
        lineage = read(self.journal / "COPY_LINEAGE.json")
        expected = {}
        new = self.work / "commands" / NEW_INVOCATION
        for i, (label, _) in enumerate(self.sub.expected_commands(JOB)[:12], 1):
            for kind in ("intent", "result"):
                name = f"{i:03d}_{label}.{kind}.json"
                original = (
                    self.work / "commands" / INVOCATION / name
                    if i <= 8
                    else selected / "commands/probe" / f"{i - 5:03d}_{label}.{kind}.json"
                )
                require(
                    (new / name).read_bytes() == original.read_bytes(),
                    "Continuation copied history differs",
                )
                expected[str((new / name).relative_to(self.root))] = dict(
                    original=str(original.relative_to(self.root)), sha256=sha(original)
                )
        require(lineage["files"] == expected, "Continuation copy lineage differs")
        require(
            set(release["command_files"])
            == {
                f"commands/{NEW_INVOCATION}/{i:03d}_{label}.{kind}.json"
                for i, (label, _) in enumerate(self.sub.expected_commands(JOB), 1)
                for kind in ("intent", "result")
            },
            "Continuation command invocation differs",
        )
        fresh = read(new / "013_held_job.intent.json")
        require(
            result["at"]
            <= intent["at"]
            <= lineage["at"]
            <= fresh["at"]
            <= release["at"]
            <= proof["at"],
            "Continuation chronology differs",
        )
        require(
            0 <= read(new / "016_release.intent.json")["at"] - result["capacity"]["at"] <= 60,
            "Released from stale capacity",
        )
        started = self.work / "STARTED.json"
        if started.exists():
            at = read(started)["at"]
            require(
                proof["at"]
                <= (datetime.fromisoformat(at).timestamp() if isinstance(at, str) else at),
                "Audit finalized after worker start",
            )
        return proof

    def verify_final(self):
        proof = self.verify_audit()
        old_auth = self.prior.old_auth
        migration = load_bound(
            old_auth.parent / SOURCES[2],
            self.auth["files"]["operations/" + OLD_COMMIT + "/" + SOURCES[2]],
            "continuation_bound_migration",
        )
        migrated = migration.verify(self.root, old_auth, runner=self.runner)
        recovered = read(self.root / "delivery/FINAL_RECOVERY_VERIFIED.json")
        require(
            recovered.get("job_id") == JOB
            and recovered.get("status") == "FINAL_RECOVERY_VERIFIED"
            and migrated.get("recovery_job_id") == JOB
            and migrated.get("status") == "FINAL_MIGRATION_VERIFIED",
            "Original acceptance differs",
        )
        archive = recovered["archive"]
        package_path = Path(archive["path"])
        require(
            package_path == self.root / "delivery/SR_F3_CE_20261016.zip"
            and sha(package_path) == archive["sha256"]
            and migrated["archive"] == archive,
            "Final archive binding differs",
        )
        required = {self.source, self.authorization, self.journal / "CONTINUED.json"}
        required.update(bounded(self.root, n) for n in self.auth["files"])
        required.update(bounded(self.journal, n) for n in proof["files"])
        archived = {}
        with zipfile.ZipFile(package_path) as package:
            names = package.namelist()
            require(len(names) == len(set(names)), "Duplicate archive entries")
            for path in sorted(required):
                name = "evidence/" + str(path.relative_to(self.root))
                require(name in names, "Continuation evidence missing from archive: " + name)
                h = hashlib.sha256()
                with package.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                require(
                    h.hexdigest() == sha(path), "Archived continuation evidence differs: " + name
                )
                archived[name] = h.hexdigest()
        receipts = [
            Path(str(package_path) + ".verification.json"),
            self.root / "delivery/FINAL_RECOVERY_VERIFIED.json",
            self.root / "delivery/FINAL_MIGRATION_VERIFIED.json",
        ]
        result = dict(
            schema_version=1,
            status="FINAL_HELD_CAPACITY_VERIFIED",
            job_id=JOB,
            at=time.time(),
            authorization_sha256=self.auth_sha,
            source_sha256=sha(self.source),
            archive=archive,
            prior_authorization_sha256=sha(self.prior_auth),
            acceptance_receipts={str(p.relative_to(self.root)): sha(p) for p in receipts},
            continued_receipt_sha256=sha(self.journal / "CONTINUED.json"),
            archived_files=archived,
            supersedes_final_gate="FINAL_HELD_RECOVERY_VERIFIED (ab7 partial never released)",
        )
        target = self.root / "delivery/FINAL_HELD_CAPACITY_VERIFIED.json"
        if target.exists():
            prior = read(target)
            require(
                all(prior.get(k) == v for k, v in result.items() if k != "at"),
                "Existing final differs",
            )
            return prior
        save(target, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--mode", choices=("continue", "verify-final"), required=True)
    args = parser.parse_args()
    operation = Continuation(args.root, args.authorization)
    print(
        json.dumps(
            operation.run() if args.mode == "continue" else operation.verify_final(), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
