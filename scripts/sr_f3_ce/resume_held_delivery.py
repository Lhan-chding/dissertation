#!/usr/bin/env python3
"""Resume only held recovery 216142 after its recorded empty accounting query."""

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


class Resume:
    def __init__(self, root, authorization, *, runner=subprocess.run):
        self.root = Path(root).resolve(strict=True)
        self.authorization = Path(authorization).resolve(strict=True)
        self.auth = read(self.authorization)
        self.auth_sha = sha(self.authorization)
        self.runner = runner
        self.source = Path(__file__).resolve(strict=True)
        self.work = self.root / WORK
        self.audit = self.root / AUDIT
        self.old_auth = self.root / "operations" / OLD_COMMIT / "RECOVERY_AUTHORIZATION.json"

    def validate(self):
        a = self.auth
        require(str(self.root) == TARGET_ROOT, "Only the exact registered root is allowed")
        require(sha(self.authorization) == self.auth_sha, "Resume authorization changed")
        fixed = dict(
            schema_version=1,
            status="AUTHORIZED_HELD_DELIVERY_RESUME",
            scope="EXACT_216142_ACCOUNTING_LAG_ONLY",
            root=str(self.root),
            job_id=JOB,
            operation_commit=OLD_COMMIT,
            recovery_directory=WORK,
            original_command_directory=OLD_COMMANDS,
            audit_directory=AUDIT,
            resumed_invocation=INVOCATION,
        )
        require(
            all(type(a.get(k)) is type(v) and a.get(k) == v for k, v in fixed.items()),
            "Resume authorization scope differs",
        )
        commit = a.get("resume_operation_commit", "")
        require(bool(re.fullmatch("[0-9a-f]{40}", commit)), "Invalid resume operation")
        require(
            self.source.parent == self.authorization.parent == self.root / "operations" / commit,
            "Resume deployment differs",
        )
        require(sha(self.source) == a.get("source_sha256"), "Resume source changed")
        if "user_request_sha256" in a:
            require(
                isinstance(a.get("user_request"), str)
                and hashlib.sha256(a["user_request"].encode()).hexdigest()
                == a["user_request_sha256"],
                "User request binding differs",
            )
        require(
            a.get("original_authorization")
            == dict(path=str(self.old_auth.relative_to(self.root)), sha256=OLD_AUTH_SHA),
            "Original authorization identity differs",
        )
        require(sha(self.old_auth) == OLD_AUTH_SHA, "Original authorization changed")
        files = a.get("files", {})
        expected = {str(self.old_auth.relative_to(self.root))}
        expected.update("operations/" + OLD_COMMIT + "/" + name for name in SOURCES)
        expected.update(WORK + "/" + name for name in ORIGINALS)
        for name, digest in files.items():
            require(sha(bounded(self.root, name)) == digest, "Pinned evidence changed: " + name)
        self.old = read(self.old_auth)
        self.m = load_bound(
            self.old_auth.parent / "submit_delivery_recovery.py",
            files.get("operations/" + OLD_COMMIT + "/submit_delivery_recovery.py"),
            "held_resume_original_submission",
        )
        self.sub = self.m.Submission(self.root, self.old_auth, runner=self.runner)
        self.sub.validate()
        require(
            self.sub.work == self.work and self.sub.script == self.work / "run.sbatch",
            "Original recovery paths differ",
        )
        commands = self.sub.expected_commands(JOB)[:8]
        expected.update(
            OLD_COMMANDS + f"/{i:03d}_{label}.{kind}.json"
            for i, (label, _) in enumerate(commands, 1)
            for kind in ("intent", "result")
        )
        require(set(files) == expected, "Original evidence inventory differs")
        actual = {str(p.relative_to(self.root)) for p in (self.root / OLD_COMMANDS).iterdir()}
        require(
            actual == {n for n in expected if n.startswith(OLD_COMMANDS + "/")},
            "Original command directory changed",
        )
        self.pairs = self.original_pairs(commands)
        submitted = read(self.work / "SUBMITTED.json")
        pending = read(self.work / "MAY_HAVE_SUBMITTED.json")
        for record in (submitted, pending):
            require(
                record.get("authorization_sha256") == OLD_AUTH_SHA
                and record.get("script_sha256") == sha(self.sub.script),
                "Submission binding differs",
            )
        require(
            submitted.get("job_id") == JOB
            and submitted.get("status") == "SUBMITTED"
            and submitted.get("stdout") == self.pairs[4][1]["stdout"],
            "Original job differs",
        )
        require(pending.get("argv") == commands[4][1], "Original submission argv differs")
        require(
            pending["at"]
            <= self.pairs[4][0]["at"]
            <= self.pairs[4][1]["at"]
            <= submitted["at"]
            <= self.pairs[5][0]["at"],
            "Submission chronology differs",
        )
        return self

    def original_pairs(self, commands):
        pairs = []
        for i, (label, argv) in enumerate(commands, 1):
            pair = [
                read(self.root / OLD_COMMANDS / f"{i:03d}_{label}.{kind}.json")
                for kind in ("intent", "result")
            ]
            require(
                all(r.get("argv") == argv for r in pair) and pair[1].get("returncode") == 0,
                "Original command differs",
            )
            require(
                pair[0]["at"] <= pair[1]["at"]
                and (not pairs or pairs[-1][1]["at"] <= pair[0]["at"]),
                "Original chronology differs",
            )
            if i == 5:
                require(
                    bool(
                        re.fullmatch(re.escape(JOB) + r"(?:;[A-Za-z0-9_-]+)?\n?", pair[1]["stdout"])
                    ),
                    "Original sbatch job differs",
                )
            if i == 6:
                self.sub.check_held_fields(self.m.fields(pair[1]["stdout"]), JOB)
            if i == 7:
                require(
                    hashlib.sha256(pair[1]["stdout"].encode()).hexdigest() == sha(self.sub.script),
                    "Original stored script differs",
                )
            if i == 8:
                require(
                    pair[1].get("stdout") == "" and pair[1].get("stderr") == "",
                    "Only the recorded empty accounting result is authorized",
                )
            pairs.append(pair)
        return pairs

    def resume(self):
        self.validate()
        with (self.work / "SUBMIT.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.work / "RELEASE.json").exists():
                return self.verify_audit()
            require(
                not any(
                    (self.work / n).exists()
                    for n in (
                        "ALLOCATION_VERIFIED.json",
                        "CAPACITY_BEFORE_RELEASE.json",
                        "RELEASE_INTENT.json",
                        "STARTED.json",
                        "PROCESS.json",
                        "END.json",
                    )
                ),
                "Original recovery advanced; stop",
            )
            require(
                not self.audit.exists() and not (self.work / "commands" / INVOCATION).exists(),
                "Prior resume exists; inspect it, never automatically retry",
            )
            save(
                self.audit / "RESUME_INTENT.json",
                dict(
                    at=time.time(),
                    job_id=JOB,
                    authorization_sha256=self.auth_sha,
                    original_authorization_sha256=OLD_AUTH_SHA,
                ),
            )
            # Current preflight stays separate from the reconstructed submission chain.
            self.sub.work, self.sub.invocation, self.sub.serial = self.audit, "precheck", 0
            current = self.sub.verify_held(JOB)
            self.sub.work = self.work
            save(
                self.audit / "PRECHECK_VERIFIED.json",
                dict(
                    at=time.time(),
                    fields=current,
                    command_files={
                        str(p.relative_to(self.audit)): sha(p)
                        for p in sorted((self.audit / "commands/precheck").glob("*.json"))
                    },
                ),
            )
            lineage = {}
            for i, (label, _) in enumerate(self.sub.expected_commands(JOB)[:7], 1):
                for kind in ("intent", "result"):
                    name = f"{i:03d}_{label}.{kind}.json"
                    old = self.root / OLD_COMMANDS / name
                    new = self.work / "commands" / INVOCATION / name
                    save_bytes(new, old.read_bytes())
                    lineage[str(new.relative_to(self.root))] = dict(
                        original=str(old.relative_to(self.root)), sha256=sha(old)
                    )
            save(self.audit / "COPY_LINEAGE.json", dict(at=time.time(), files=lineage))
            self.sub.invocation, self.sub.serial = INVOCATION, 7
            label, argv = self.sub.expected_commands(JOB)[7]
            accounting = self.sub.command(label, argv)
            require(
                accounting.strip().splitlines()
                in ([JOB + "|PENDING|Unknown|0|"], [JOB + "|PENDING|None|0|"]),
                "Fresh accounting must prove the original never started",
            )
            save(
                self.work / "ALLOCATION_VERIFIED.json",
                dict(
                    status="VERIFIED",
                    job_id=JOB,
                    fields=self.m.fields(self.pairs[5][1]["stdout"]),
                    at=time.time(),
                    authorization_sha256=OLD_AUTH_SHA,
                    script_sha256=sha(self.sub.script),
                ),
            )
            cap = self.sub.capacity(own_job=JOB)
            save(self.work / "CAPACITY_BEFORE_RELEASE.json", cap)
            self.validate()
            self.sub.invocation, self.sub.serial = INVOCATION, 12
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
            result = dict(
                job_id=JOB,
                status="RELEASED",
                stdout=output,
                stderr="",
                returncode=0,
                at=time.time(),
                authorization_sha256=OLD_AUTH_SHA,
                script_sha256=sha(self.sub.script),
                chain={n: sha(self.work / n) for n in CHAIN},
                command_files={
                    str(p.relative_to(self.work)): sha(p)
                    for p in sorted((self.work / "commands" / INVOCATION).glob("*.json"))
                },
            )
            self.sub.check_command_chain(result)
            proof = dict(
                status="RESUMED",
                job_id=JOB,
                at=time.time(),
                authorization_sha256=self.auth_sha,
                release_sha256=hashlib.sha256(encoded(result)).hexdigest(),
                audit_files={
                    str(p.relative_to(self.audit)): sha(p)
                    for p in sorted(self.audit.rglob("*.json"))
                },
            )
            save(self.audit / "RESUMED.json", proof)
            # Worker waits for this atomic publication. All supplemental evidence exists first.
            save(self.work / "RELEASE.json", result)
            return self.verify_audit()

    def verify_audit(self):
        self.validate()
        release = self.sub.prior_release()
        require(release["job_id"] == JOB, "Resumed job differs")
        proof = read(self.audit / "RESUMED.json")
        require(
            proof.get("status") == "RESUMED"
            and proof.get("job_id") == JOB
            and proof.get("authorization_sha256") == self.auth_sha
            and proof.get("release_sha256") == sha(self.work / "RELEASE.json"),
            "Resume release binding differs",
        )
        required = {"RESUME_INTENT.json", "PRECHECK_VERIFIED.json", "COPY_LINEAGE.json"}
        required.update(
            f"commands/precheck/{i:03d}_{label}.{kind}.json"
            for i, (label, _) in enumerate(self.sub.expected_commands(JOB)[5:8], 1)
            for kind in ("intent", "result")
        )
        require(set(proof.get("audit_files", {})) == required, "Resume audit inventory differs")
        require(
            {str(p.relative_to(self.audit)) for p in self.audit.rglob("*.json")}
            == required | {"RESUMED.json"},
            "Unexpected resume audit files",
        )
        for name, digest in proof["audit_files"].items():
            require(sha(bounded(self.audit, name)) == digest, "Resume audit changed")
        intent = read(self.audit / "RESUME_INTENT.json")
        require(
            intent.get("job_id") == JOB
            and intent.get("authorization_sha256") == self.auth_sha
            and intent.get("original_authorization_sha256") == OLD_AUTH_SHA,
            "Resume intent differs",
        )
        require(self.pairs[-1][1]["at"] <= intent["at"], "Resume predates original interruption")
        previous = intent["at"]
        responses = []
        for i, (label, argv) in enumerate(self.sub.expected_commands(JOB)[5:8], 1):
            pair = [
                read(self.audit / f"commands/precheck/{i:03d}_{label}.{kind}.json")
                for kind in ("intent", "result")
            ]
            require(
                all(r.get("argv") == argv for r in pair)
                and pair[1].get("returncode") == 0
                and previous <= pair[0]["at"] <= pair[1]["at"],
                "Resume precheck differs",
            )
            previous = pair[1]["at"]
            responses.append(pair[1]["stdout"])
        self.sub.check_held_fields(self.m.fields(responses[0]), JOB)
        require(
            hashlib.sha256(responses[1].encode()).hexdigest() == sha(self.sub.script),
            "Precheck stored script differs",
        )
        require(
            responses[2].strip().splitlines()
            in ([JOB + "|PENDING|Unknown|0|"], [JOB + "|PENDING|None|0|"]),
            "Precheck accounting differs",
        )
        precheck = read(self.audit / "PRECHECK_VERIFIED.json")
        require(
            precheck["fields"] == self.m.fields(responses[0]) and previous <= precheck["at"],
            "Precheck proof differs",
        )
        require(
            precheck["command_files"]
            == {n: proof["audit_files"][n] for n in required if n.startswith("commands/")},
            "Precheck binding differs",
        )
        lineage = read(self.audit / "COPY_LINEAGE.json")
        expected = {}
        for i, (label, _) in enumerate(self.sub.expected_commands(JOB)[:7], 1):
            for kind in ("intent", "result"):
                name = f"{i:03d}_{label}.{kind}.json"
                old, new = (
                    self.root / OLD_COMMANDS / name,
                    self.work / "commands" / INVOCATION / name,
                )
                require(new.read_bytes() == old.read_bytes(), "Copied history differs")
                expected[str(new.relative_to(self.root))] = dict(
                    original=str(old.relative_to(self.root)), sha256=sha(old)
                )
        require(lineage["files"] == expected, "Copy lineage differs")
        fresh = read(self.work / "commands" / INVOCATION / "008_held_accounting.intent.json")
        require(
            precheck["at"] <= lineage["at"] <= fresh["at"] <= release["at"] <= proof["at"],
            "Resume chronology differs",
        )
        require(
            set(release["command_files"])
            == {
                f"commands/{INVOCATION}/{i:03d}_{label}.{kind}.json"
                for i, (label, _) in enumerate(self.sub.expected_commands(JOB), 1)
                for kind in ("intent", "result")
            },
            "Resumed invocation differs",
        )
        started = self.work / "STARTED.json"
        if started.exists():
            started_at = read(started)["at"]
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at).timestamp()
            require(proof["at"] <= started_at, "Resume audit finalized after worker start")
        return proof

    def verify_final(self):
        proof = self.verify_audit()
        migration = load_bound(
            self.old_auth.parent / SOURCES[2],
            self.auth["files"]["operations/" + OLD_COMMIT + "/" + SOURCES[2]],
            "held_resume_bound_migration",
        )
        migrated = migration.verify(self.root, self.old_auth, runner=self.runner)
        # This bound migration gate already calls the bound Recovery.verify_final.
        recovered = read(self.root / "delivery/FINAL_RECOVERY_VERIFIED.json")
        require(
            recovered.get("job_id") == JOB and recovered.get("status") == "FINAL_RECOVERY_VERIFIED",
            "Original recovery verification differs",
        )
        require(
            migrated.get("status") == "FINAL_MIGRATION_VERIFIED"
            and migrated.get("recovery_job_id") == JOB,
            "Migration verification differs",
        )
        archive = recovered["archive"]
        package_path = Path(archive["path"])
        require(
            package_path == self.root / "delivery/SR_F3_CE_20261016.zip"
            and sha(package_path) == archive["sha256"]
            and migrated["archive"] == archive,
            "Final archive binding differs",
        )
        required = {self.authorization, self.source}
        required.update(bounded(self.root, n) for n in self.auth["files"])
        required.update(bounded(self.audit, n) for n in proof["audit_files"])
        required.add(self.audit / "RESUMED.json")
        archived = {}
        with zipfile.ZipFile(package_path) as package:
            names = package.namelist()
            require(len(names) == len(set(names)), "Duplicate archive entries")
            for path in sorted(required):
                name = "evidence/" + str(path.relative_to(self.root))
                require(name in names, "Held-resume evidence missing from archive: " + name)
                h = hashlib.sha256()
                with package.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                digest = h.hexdigest()
                require(digest == sha(path), "Archived held-resume evidence differs: " + name)
                archived[name] = digest
        receipts = [
            Path(str(package_path) + ".verification.json"),
            self.root / "delivery/FINAL_RECOVERY_VERIFIED.json",
            self.root / "delivery/FINAL_MIGRATION_VERIFIED.json",
        ]
        result = dict(
            schema_version=1,
            status="FINAL_HELD_RECOVERY_VERIFIED",
            job_id=JOB,
            at=time.time(),
            authorization_sha256=self.auth_sha,
            source_sha256=sha(self.source),
            original_authorization_sha256=OLD_AUTH_SHA,
            archive=archive,
            acceptance_receipts={str(p.relative_to(self.root)): sha(p) for p in receipts},
            resumed_receipt_sha256=sha(self.audit / "RESUMED.json"),
            archived_files=archived,
        )
        target = self.root / "delivery/FINAL_HELD_RECOVERY_VERIFIED.json"
        if target.exists():
            prior = read(target)
            require(
                all(prior.get(k) == v for k, v in result.items() if k != "at"),
                "Existing held-resume final receipt differs",
            )
            return prior
        save(target, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--mode", required=True, choices=("resume", "verify-final"))
    args = parser.parse_args()
    operation = Resume(args.root, args.authorization)
    print(
        json.dumps(
            operation.resume() if args.mode == "resume" else operation.verify_final(),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
