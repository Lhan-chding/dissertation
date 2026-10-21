"""Exact, once-only scheduler constraint normalization for held recovery 216983.

The bound 538 worker, script, authority, scientific outputs and old receipts remain
unchanged. Only an actual scheduler cpu_ok&cpu_ok -> cpu_ok transition is accepted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from types import ModuleType

COMMIT = "538aa3dd36e9d5a53f5aa0040b24d3999eab76e9"
JOB = "216983"
WORK = "operations/delivery_recovery/20261021_delivery_lowmem"
AUDIT = "commands/poll_constraint_normalization_216983"


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe bound path")
    path = (root / p).resolve(strict=True)
    require(path.is_relative_to(root), "Bound path escapes")
    return path


class Sidecar:
    def __init__(self, root, authorization, *, runner=subprocess.run, validator=None):
        self.root = Path(root).resolve(strict=True)
        self.authorization = Path(authorization).resolve(strict=True)
        self.auth = read(self.authorization)
        self.authorization_sha256 = sha(self.authorization)
        require(
            self.auth.get("root") == str(self.root) and self.auth.get("job_id") == JOB,
            "Exact normalization scope differs",
        )
        source = bounded(self.root, self.auth["source"])
        require(
            source == Path(__file__).resolve() and sha(source) == self.auth["source_sha256"],
            "Normalization source differs",
        )
        binding = self.auth["original_authorization"]
        original = bounded(self.root, binding["path"])
        require(
            original == self.root / "operations" / COMMIT / "RECOVERY_AUTHORIZATION.json"
            and sha(original) == binding["sha256"],
            "Original authority differs",
        )
        original_auth = read(original)
        source = original.parent / "submit_delivery_recovery.py"
        require(
            original_auth["extra_sources"].get(str(source.relative_to(self.root))) == sha(source),
            "Original submit source differs",
        )
        self.module = ModuleType("bound_538_submission")
        self.module.__file__ = str(source)
        exec(compile(source.read_bytes(), str(source), "exec"), self.module.__dict__)
        self.op = self.module.Submission(self.root, original, runner=runner, validator=validator)
        self.op.validate()
        self.work = self.root / WORK
        self.audit = self.work / AUDIT
        self.partial = bounded(self.root, self.auth["partial_poll"]["path"])
        require(
            self.partial.parent == self.work / "commands"
            and re.fullmatch(r"poll_continue_[0-9]+", self.partial.name),
            "Partial poll scope differs",
        )
        self.validate()

    def validate(self):
        require(
            sha(self.authorization) == self.authorization_sha256, "Normalization authority changed"
        )
        require(sha(__file__) == self.auth["source_sha256"], "Normalization source changed")
        self.op.validate()
        self.op.history(require_retired=True)
        submitted = read(self.work / "SUBMITTED.json")
        require(
            submitted.get("job_id") == JOB
            and submitted.get("authorization_sha256") == sha(self.op.authorization)
            and submitted.get("script_sha256") == sha(self.op.script),
            "Original submission differs",
        )
        expected = {"001_held_job.intent.json", "001_held_job.result.json"}
        require(
            {p.name for p in self.partial.iterdir()} == expected,
            "Partial poll advanced or incomplete",
        )
        pins = self.auth["partial_poll"]["files"]
        require(
            set(pins) == expected and all(sha(self.partial / n) == h for n, h in pins.items()),
            "Partial poll bytes changed",
        )
        intent, result = [
            read(self.partial / f"001_held_job.{k}.json") for k in ("intent", "result")
        ]
        argv = ["scontrol", "show", "job", JOB, "-o"]
        require(
            intent.get("argv") == result.get("argv") == argv
            and result.get("returncode") == 0
            and intent["at"] <= result["at"],
            "Partial query unknown",
        )
        self.check_duplicate(self.module.fields(result["stdout"]))

    def check_duplicate(self, fields):
        require(fields.get("Features") == "cpu_ok&cpu_ok", "Not the exact duplicate constraint")
        normalized = dict(fields, Features="cpu_ok")
        self.op.check_held_fields(normalized, JOB)

    def pristine(self):
        for name in (
            "ALLOCATION_VERIFIED.json",
            "CAPACITY_BEFORE_RELEASE.json",
            "RELEASE_INTENT.json",
            "RELEASE.json",
            "STARTED.json",
            "PROCESS.json",
            "END.json",
            "MEMORY_PROFILE.json",
        ):
            require(not (self.work / name).exists(), "Original recovery advanced: " + name)

    def verify_three(self, *, duplicate):
        f = self.module.fields(self.op.command("held_job", ["scontrol", "show", "job", JOB, "-o"]))
        self.check_duplicate(f) if duplicate else self.op.check_held_fields(f, JOB)
        script = self.op.command("stored_script", ["scontrol", "write", "batch_script", JOB, "-"])
        require(
            hashlib.sha256(script.encode()).hexdigest() == sha(self.op.script),
            "Scheduler script changed",
        )
        raw = self.op.command(
            "held_accounting",
            ["sacct", "-n", "-P", "-j", JOB, "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES"],
        )
        rows = [line.split("|") for line in raw.splitlines() if line.strip()]
        rows = [row for row in rows if row[0] == JOB]
        require(
            len(rows) == 1
            and rows[0][1:] in (["PENDING", "None", "0", ""], ["PENDING", "Unknown", "0", ""]),
            "No exact zero-allocation proof",
        )
        return f

    def normalize(self):
        self.validate()
        self.pristine()
        if (self.audit / "RESULT.json").exists():
            return self.verify_normalization()
        require(not self.audit.exists(), "Normalization already attempted; inspect only")
        self.audit.mkdir()
        save(self.audit / "AUTHORIZATION.json", self.auth)
        save(
            self.audit / "SOURCE.json",
            dict(
                path=self.auth["source"],
                sha256=self.auth["source_sha256"],
                utf8=Path(__file__).read_text(),
            ),
        )
        self.op.invocation = Path(AUDIT).name
        self.op.serial = 0
        try:
            before = self.verify_three(duplicate=True)
            self.op.command("normalize", ["scontrol", "update", "JobId=" + JOB, "Features=cpu_ok"])
            after = self.verify_three(duplicate=False)
            # The independent after check proves all original held invariants. Do
            # not require volatile scheduler Priority/LastSchedEval fields stable.
            result = dict(
                status="NORMALIZED",
                job_id=JOB,
                before=before,
                after=after,
                authorization_sha256=self.authorization_sha256,
                source_sha256=self.auth["source_sha256"],
                at=time.time(),
                files={p.name: sha(p) for p in self.audit.glob("*.json")},
            )
            save(self.audit / "RESULT.json", result)
            return self.verify_normalization()
        except BaseException as exc:
            save(
                self.audit / "STOP.json",
                dict(status="NORMALIZATION_STOPPED", error=repr(exc), at=time.time()),
            )
            raise

    def verify_normalization(self):
        self.validate()
        record = read(self.audit / "RESULT.json")
        require(
            record.get("status") == "NORMALIZED"
            and record.get("job_id") == JOB
            and record.get("authorization_sha256") == self.authorization_sha256
            and record.get("source_sha256") == self.auth["source_sha256"],
            "Normalization receipt differs",
        )
        actual = {p.name: sha(p) for p in self.audit.glob("*.json") if p.name != "RESULT.json"}
        require(
            actual == record.get("files") and "STOP.json" not in actual,
            "Normalization audit changed",
        )
        require(
            read(self.audit / "AUTHORIZATION.json") == self.auth,
            "Normalization authority copy differs",
        )
        source_record = read(self.audit / "SOURCE.json")
        require(
            source_record
            == dict(
                path=self.auth["source"],
                sha256=self.auth["source_sha256"],
                utf8=Path(__file__).read_text(),
            ),
            "Embedded normalization source differs",
        )
        queries = self.op.expected_commands(JOB)[5:8]
        commands = [
            *queries,
            ("normalize", ["scontrol", "update", "JobId=" + JOB, "Features=cpu_ok"]),
            *queries,
        ]
        expected_names = {"AUTHORIZATION.json", "SOURCE.json"} | {
            f"{i:03d}_{label}.{kind}.json"
            for i, (label, _) in enumerate(commands, 1)
            for kind in ("intent", "result")
        }
        require(set(actual) == expected_names, "Normalization commands incomplete")
        last = 0
        held = []
        for i, (label, argv) in enumerate(commands, 1):
            intent, result = [
                read(self.audit / f"{i:03d}_{label}.{kind}.json") for kind in ("intent", "result")
            ]
            require(
                intent.get("argv") == result.get("argv") == argv
                and result.get("returncode") == 0
                and last <= intent["at"] <= result["at"],
                "Normalization command unknown",
            )
            last = result["at"]
            if label == "held_job":
                f = self.module.fields(result["stdout"])
                self.check_duplicate(f) if i == 1 else self.op.check_held_fields(f, JOB)
                held.append(f)
            elif label == "stored_script":
                require(
                    hashlib.sha256(result["stdout"].encode()).hexdigest() == sha(self.op.script),
                    "Archived script differs",
                )
            elif label == "held_accounting":
                rows = [line.split("|") for line in result["stdout"].splitlines() if line.strip()]
                rows = [row for row in rows if row[0] == JOB]
                require(
                    len(rows) == 1
                    and rows[0][1:]
                    in (["PENDING", "None", "0", ""], ["PENDING", "Unknown", "0", ""]),
                    "Archived zero allocation differs",
                )
        require(
            record["before"] == held[0] and record["after"] == held[1],
            "Normalization fields differ",
        )
        return record

    def safe_polls(self, *, final=False):
        self.verify_normalization()
        submitted = read(self.work / "SUBMITTED.json")
        for directory in sorted((self.work / "commands").glob("poll_*")):
            if directory in (self.partial, self.audit) or directory.name == submitted["invocation"]:
                continue
            require(
                re.fullmatch(r"poll_continue_[0-9]+", directory.name), "Unknown polling namespace"
            )
            marker = read(directory / "RESULT.json")
            require(
                marker.get("status")
                in (
                    ("WAIT_CAPACITY", "WAIT_ACCOUNTING", "READY")
                    if final
                    else ("WAIT_CAPACITY", "WAIT_ACCOUNTING")
                ),
                "Unsafe previous continuation",
            )
            require(
                marker.get("files")
                == {p.name: sha(p) for p in directory.glob("*.json") if p.name != "RESULT.json"},
                "Wait bytes changed",
            )
            commands = self.op.expected_commands(JOB)[5:12]
            pairs = len(marker["files"]) // 2
            expected = {
                f"{i:03d}_{label}.{kind}.json"
                for i, (label, _) in enumerate(commands[:pairs], 1)
                for kind in ("intent", "result")
            }
            require(
                0 < pairs <= 7 and set(marker["files"]) == expected, "Wait command coverage differs"
            )
            if marker["status"] == "READY":
                lineage = read(self.work / "COPY_LINEAGE.json")
                require(
                    pairs == 7
                    and any(
                        Path(v["source"]).parent == directory.relative_to(self.work)
                        for v in lineage.values()
                    ),
                    "READY is not the released poll",
                )
            last = 0
            for i, (label, argv) in enumerate(commands[:pairs], 1):
                intent, result = [
                    read(directory / f"{i:03d}_{label}.{kind}.json")
                    for kind in ("intent", "result")
                ]
                require(
                    intent.get("argv") == result.get("argv") == argv
                    and result.get("returncode") == 0
                    and last <= intent["at"] <= result["at"],
                    "Unknown wait command",
                )
                last = result["at"]

    def run(self):
        lock = self.root / "operations/delivery_recovery/20261021_delivery_teacher/SUBMIT.lock"
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (self.work / "RELEASE.json").exists():
                self.normalize()
            else:
                self.verify_normalization()
                return self.op.prior_release()
        return continue_bound(self.op, self)

    def verify_final(self):
        self.verify_normalization()
        original = self.op.verify_final()
        release = self.op.prior_release()
        require(release["job_id"] == JOB, "Final job differs")
        self.safe_polls(final=True)
        archive = self.root / "delivery/SR_F3_CE_20261016.zip"
        paths = [*self.audit.glob("*.json"), *self.partial.glob("*.json")]
        archived = {}
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            require(len(names) == len(set(names)), "Duplicate archive members")
            for path in paths:
                name = "evidence/" + str(path.relative_to(self.root))
                h = hashlib.sha256()
                with package.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                require(h.hexdigest() == sha(path), "Archived normalization evidence differs")
                archived[name] = h.hexdigest()
        require(sha(archive) == original["archive"]["sha256"], "Final archive changed")
        result = dict(
            status="FINAL_CONSTRAINT_NORMALIZATION_VERIFIED",
            job_id=JOB,
            archive=original["archive"],
            archived_normalization=archived,
            authorization_sha256=self.authorization_sha256,
            original_final_sha256=sha(self.root / "delivery/FINAL_LOWMEM_RECOVERY_VERIFIED.json"),
            at=time.time(),
        )
        target = self.root / "delivery/FINAL_CONSTRAINT_NORMALIZATION_VERIFIED.json"
        if target.exists():
            previous = read(target)
            require(
                all(previous.get(k) == v for k, v in result.items() if k != "at"),
                "Final normalization changed",
            )
            return previous
        save(target, result)
        return result


# Keep immutable atomic receipt publication identical to the bound submission helper.
def save(path, value):
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix="." + path.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
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


def continue_bound(self, sidecar):
    self.validate()
    self.history(require_retired=True)
    with (self.root / "operations/delivery_recovery/20261021_delivery_teacher/SUBMIT.lock").open(
        "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.work / "RELEASE.json").exists():
            return self.prior_release()
        require(
            not (self.work / "RELEASE_INTENT.json").exists(),
            "Release already attempted; inspect only",
        )
        require(
            not (self.work / "ALLOCATION_VERIFIED.json").exists(),
            "Final release chain already started; inspect only",
        )
        submitted = read(self.work / "SUBMITTED.json")
        require(
            submitted.get("authorization_sha256") == sha(self.authorization)
            and submitted.get("script_sha256") == sha(self.script),
            "Submission changed",
        )
        job = submitted["job_id"]
        require(
            isinstance(job, str)
            and re.fullmatch(r"[1-9][0-9]*", job)
            and job not in {"211333", "215520", "216128", "216142", "216897"},
            "Invalid replacement job",
        )
        require(
            re.fullmatch(r"poll_submit_[0-9]+", submitted.get("invocation", "")),
            "Invalid submit invocation",
        )
        sidecar.verify_normalization()
        sidecar.safe_polls()
        self.invocation = "poll_continue_" + str(time.time_ns())
        self.serial = 0
        try:
            verified = self.verify_held(job)
            cap = self.capacity(own_job=job)
        except (sidecar.module.WaitAccounting, sidecar.module.WaitCapacity) as exc:
            return self.poll_result(
                "WAIT_ACCOUNTING"
                if isinstance(exc, sidecar.module.WaitAccounting)
                else "WAIT_CAPACITY",
                reason=str(exc),
                job_id=job,
            )
        self.poll_result("READY", capacity=cap, job_id=job)
        poll = self.work / "commands" / self.invocation
        initial = self.work / "commands" / submitted["invocation"]
        self.invocation = "release_" + str(time.time_ns())
        final = self.work / "commands" / self.invocation
        final.mkdir()
        lineage = {}

        def copy_pair(source, old_index, label, index):
            for kind in ("intent", "result"):
                src = source / f"{old_index:03d}_{label}.{kind}.json"
                dest = final / f"{index:03d}_{label}.{kind}.json"
                require(not dest.exists(), "Copy target already exists")
                shutil.copyfile(src, dest)
                require(sha(dest) == sha(src), "Command copy differs")
                lineage[str(dest.relative_to(self.work))] = dict(
                    source=str(src.relative_to(self.work)), sha256=sha(src)
                )

        for i, label in enumerate(("qos", "assoc", "queue", "node", "sbatch"), 1):
            copy_pair(initial, i, label, i)
        for i, label in enumerate(
            ("held_job", "stored_script", "held_accounting", "qos", "assoc", "queue", "node"), 1
        ):
            copy_pair(poll, i, label, i + 5)
        save(self.work / "COPY_LINEAGE.json", lineage)
        save(
            self.work / "ALLOCATION_VERIFIED.json",
            dict(
                status="VERIFIED",
                job_id=job,
                fields=verified,
                at=time.time(),
                authorization_sha256=sha(self.authorization),
                script_sha256=sha(self.script),
            ),
        )
        save(self.work / "CAPACITY_BEFORE_RELEASE.json", cap)
        self.validate()
        self.history(require_retired=True)
        self.serial = 12
        self.verify_held(job)
        require(0 <= time.time() - cap["at"] <= 60, "Release capacity stale")
        save(
            self.work / "RELEASE_INTENT.json",
            dict(
                job_id=job,
                at=time.time(),
                capacity_sha256=sha(self.work / "CAPACITY_BEFORE_RELEASE.json"),
            ),
        )
        output = self.command("release", ["scontrol", "release", job])
        result = dict(
            job_id=job,
            status="RELEASED",
            stdout=output,
            stderr="",
            returncode=0,
            at=time.time(),
            authorization_sha256=sha(self.authorization),
            script_sha256=sha(self.script),
            chain={
                name: sha(self.work / name)
                for name in (
                    "MAY_HAVE_SUBMITTED.json",
                    "SUBMITTED.json",
                    "ALLOCATION_VERIFIED.json",
                    "CAPACITY_BEFORE_SUBMIT.json",
                    "CAPACITY_BEFORE_RELEASE.json",
                    "RELEASE_INTENT.json",
                )
            },
            command_files={str(p.relative_to(self.work)): sha(p) for p in final.glob("*.json")},
            teacher_audit_files={
                str(p.relative_to(self.work)): sha(p)
                for p in self.work.glob("commands/poll_*/*.json")
            },
        )
        result["teacher_audit_files"]["COPY_LINEAGE.json"] = sha(self.work / "COPY_LINEAGE.json")
        # Original worker only sees the release marker after all audit is durable.
        save(self.work / "RELEASE.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("continue", "verify-final"))
    args = parser.parse_args()
    sidecar = Sidecar(args.root, args.authorization)
    result = sidecar.run() if args.mode == "continue" else sidecar.verify_final()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
