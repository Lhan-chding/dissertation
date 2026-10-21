"""One fail-closed held submission for the separately authorized delivery recovery."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from types import ModuleType


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


def save(path, value):
    """Publish complete immutable bytes atomically, including the worker release gate."""
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


def fields(text):
    return dict(re.findall(r"(\S+?)=(\S*)", text))


def table(text):
    rows = [x.split("|") for x in text.splitlines() if x.strip()]
    require(len(rows) >= 2 and len(set(rows[0])) == len(rows[0]), "Missing capacity table")
    require(all(len(x) == len(rows[0]) for x in rows), "Truncated capacity table")
    return [dict(zip(rows[0], row, strict=True)) for row in rows[1:]]


def limit(value):
    if value in ("", "-1", "UNLIMITED", "NONE"):
        return None
    require(bool(re.fullmatch(r"[0-9]+", value)), "Unknown capacity limit")
    return int(value)


def memory(value):
    match = re.fullmatch(r"([0-9]+)([MG])", value)
    require(match is not None, "Unknown memory unit")
    return int(match[1]) * (1024 if match[2] == "G" else 1)


TEACHER_QOS = "soujanya-poria-startfund-2026-03"


class WaitCapacity(PermissionError):
    pass


class WaitAccounting(PermissionError):
    pass


class Submission:
    """Once MAY_HAVE_SUBMITTED exists, no path can issue another sbatch."""

    def __init__(self, root, authorization, *, runner=subprocess.run, validator=None):
        self.root = Path(root).resolve(strict=True)
        self.authorization = Path(authorization).resolve(strict=True)
        self.auth = read(self.authorization)
        self.authorization_sha256 = sha(self.authorization)
        self.runner = runner
        self.validator = validator
        self.work = (self.root / self.auth["recovery_directory"]).resolve()
        require(
            self.work.is_relative_to(self.root / "operations/delivery_recovery"),
            "Recovery path escapes",
        )
        self.serial = 0

    def validate(self):
        require(sha(self.authorization) == self.authorization_sha256, "Authorization changed")
        if self.validator:
            self.validator(self.root, self.authorization)
        else:
            path = self.authorization.parent / "recover_delivery.py"
            spec = importlib.util.spec_from_file_location(
                "delivery_recovery_submit_validation", path
            )
            module = importlib.util.module_from_spec(spec)
            body = path.read_bytes()
            require(
                hashlib.sha256(body).hexdigest() == self.auth["source_sha256"],
                "Recovery helper source differs",
            )
            exec(compile(body, str(path), "exec"), module.__dict__)
            module.Recovery(self.root, self.authorization, require_submission=False).validate()
        a = self.auth
        expected = dict(
            allowed_nodes=["cpu-1", "cpu-2"],
            qos=TEACHER_QOS,
            account="rose",
            cpus=4,
            memory_mib=12288,
            time_limit="3-00:00:00",
        )
        require(
            all(type(a.get(k)) is type(v) and a.get(k) == v for k, v in expected.items()),
            "Fixed recovery resources differ",
        )
        require(a["root"] == str(self.root), "Root differs")
        require(
            a["recovery_directory"] == "operations/delivery_recovery/20261021_delivery_lowmem",
            "Teacher recovery namespace differs",
        )
        require(
            a["extra_sources"].get(str(Path(__file__).resolve().relative_to(self.root)))
            == sha(__file__),
            "Submit source differs",
        )
        self.script = (self.root / a["script"]).resolve(strict=True)
        require(self.script.is_relative_to(self.root / "operations"), "Script outside operations")
        require(sha(self.script) == a["script_sha256"], "Script hash differs")
        self.user = a["user"]
        self.name = a["job_name"]
        self.comment = a["comment"]
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", self.user)), "Invalid user")

    def command(self, label, argv):
        self.serial += 1
        directory = self.work / "commands" / self.invocation
        stem = f"{self.serial:03d}_{label}"
        save(directory / (stem + ".intent.json"), dict(argv=argv, at=time.time()))
        try:
            result = self.runner(argv, capture_output=True, text=True, timeout=30, check=False)
        except BaseException as exc:
            save(
                directory / (stem + ".result.json"),
                dict(status="UNKNOWN", error=repr(exc), at=time.time()),
            )
            raise
        record = dict(
            argv=argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            at=time.time(),
        )
        save(directory / (stem + ".result.json"), record)
        require(result.returncode == 0, "Command failed; inspect preserved receipt: " + label)
        return result.stdout

    def capacity(self, *, own_job=None, pre_submit=False):
        qos = table(
            self.command(
                "qos",
                [
                    "sacctmgr",
                    "-P",
                    "show",
                    "qos",
                    "where",
                    "name=" + TEACHER_QOS,
                    "format=Name,GrpSubmitJobs,MaxSubmitJobsPerUser,GrpJobs,MaxJobsPerUser,Flags",
                ],
            )
        )
        qos = [q for q in qos if q.get("Name") == TEACHER_QOS]
        require(len(qos) == 1, "Ambiguous teacher QoS")
        q = qos[0]
        require(
            limit(q["GrpSubmit"]) is None and limit(q["GrpJobs"]) is None,
            "Cannot prove private group capacity",
        )
        require("relative" not in q["Flags"].lower(), "Relative QoS unsupported")
        require(
            limit(q["MaxSubmitPU"]) == 5
            and (limit(q["MaxJobsPU"]) is None or limit(q["MaxJobsPU"]) > 0),
            "Teacher limits changed; review required",
        )
        assoc = table(
            self.command(
                "assoc",
                [
                    "sacctmgr",
                    "-P",
                    "show",
                    "assoc",
                    "where",
                    "user=" + self.user,
                    "account=rose",
                    "format=Cluster,Account,User,Partition,QOS%1000,GrpSubmitJobs,MaxSubmitJobs,GrpJobs,MaxJobs",
                ],
            )
        )
        assoc = [
            x
            for x in assoc
            if x["User"] == self.user
            and x["Account"] == "rose"
            and x["Partition"] in ("", "cluster02")
        ]
        require(
            bool(assoc)
            and len({x["Cluster"] for x in assoc}) == 1
            and len({x["Partition"] for x in assoc}) == len(assoc),
            "Association unavailable or ambiguous",
        )
        require(
            all(TEACHER_QOS in x["QOS"].split(",") for x in assoc), "Teacher membership missing"
        )
        require(
            all(limit(x[k]) is None for x in assoc for k in ("GrpSubmit", "GrpJobs")),
            "Private association group cap cannot be proven",
        )
        raw = self.command("queue", ["squeue", "-h", "-u", self.user, "-o", "%i|%u|%a|%q|%P|%T"])
        queue = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = line.split("|")
            require(
                len(row) == 6 and row[0] not in queue and row[1] == self.user, "Ambiguous queue"
            )
            require(
                all(v and v.lower() not in ("unknown", "(null)", "n/a") for v in row),
                "Unknown queue reservation",
            )
            queue[row[0]] = row
        if own_job:
            require(
                own_job in queue
                and queue[own_job][3] == TEACHER_QOS
                and queue[own_job][5] == "PENDING",
                "Held self missing from queue",
            )
        submitted = sum(row[3] == TEACHER_QOS for row in queue.values())
        running = sum(row[3] == TEACHER_QOS and row[5] != "PENDING" for row in queue.values())
        if submitted + (0 if own_job else 1) > 5 or (
            not pre_submit
            and limit(q["MaxJobsPU"]) is not None
            and running >= limit(q["MaxJobsPU"])
        ):
            raise WaitCapacity("Teacher capacity unavailable")
        for x in assoc:
            scoped = [
                r
                for r in queue.values()
                if r[2] == "rose" and (not x["Partition"] or x["Partition"] == r[4])
            ]
            for key, count in [
                ("MaxSubmit", len(scoped) + (0 if own_job else 1)),
                ("MaxJobs", sum(r[5] != "PENDING" for r in scoped) + 1),
            ]:
                cap = limit(x[key])
                if not (pre_submit and key == "MaxJobs") and cap is not None and count > cap:
                    raise WaitCapacity("Association capacity unavailable")
        raw_nodes = self.command("node", ["scontrol", "show", "node", "cpu-1,cpu-2", "-o"])
        node_rows = [fields(line) for line in raw_nodes.splitlines() if line.strip()]
        require(
            len(node_rows) == 2 and {n.get("NodeName") for n in node_rows} == {"cpu-1", "cpu-2"},
            "CPU candidate inventory differs",
        )
        nodes = {n["NodeName"]: n for n in node_rows}
        eligible = []
        for name, node in sorted(nodes.items()):
            require(
                "cluster02" in node.get("Partitions", "").split(",")
                and node.get("Arch") == "x86_64"
                and "cpu_ok" in node.get("ActiveFeatures", "").split(","),
                "Node partition/architecture/CPU feature changed",
            )
            require(
                int(node["CPUEfctv"]) >= 4
                and int(node["RealMemory"]) - int(node.get("MemSpecLimit", "0")) >= 12288,
                "Configured node cannot fit recovery",
            )
            state = node.get("State", "")
            require(bool(re.fullmatch(r"[A-Z_]+(?:\+[A-Z_]+)*", state)), "Malformed node state")
            if (
                state in ("IDLE", "MIXED")
                and int(node["CPUEfctv"]) - int(node["CPUAlloc"]) >= 4
                and int(node["RealMemory"])
                - int(node["AllocMem"])
                - int(node.get("MemSpecLimit", "0"))
                >= 12288
            ):
                eligible.append(name)
        if not pre_submit and not eligible:
            raise WaitCapacity(
                "No eligible CPU node with 4 CPUs and 12 GiB: "
                + ", ".join(name + "=" + node["State"] for name, node in sorted(nodes.items()))
            )
        return dict(
            at=time.time(),
            queue=queue,
            qos=q,
            associations=assoc,
            nodes=nodes,
            eligible_nodes=eligible,
            self_job=own_job,
        )

    def check_held_fields(self, f, job):
        expected = dict(
            JobId=job,
            JobName=self.name,
            Account="rose",
            QOS=TEACHER_QOS,
            Partition="cluster02",
            Command=str(self.script),
            Comment=self.comment,
            NumCPUs="4",
            NumTasks="1",
            **{"CPUs/Task": "4"},
            TimeLimit="3-00:00:00",
            Requeue="0",
            JobState="PENDING",
            Reason="JobHeldUser",
            RunTime="00:00:00",
        )
        require(
            all(f.get(k) == v for k, v in expected.items()), "Held job request differs or started"
        )
        require(
            f.get("ReqNodeList", "") in ("", "(null)", "None")
            and f.get("Features") in {"cpu_ok", "cpu_ok&cpu_ok"},
            "CPU-only candidate constraint differs",
        )
        require(
            f.get("UserId", "").split("(")[0] == self.user and f.get("NumNodes") in ("1", "1-1"),
            "Owner/node count differs",
        )
        require(
            f.get("NodeList", "") in ("", "(null)", "None")
            and f.get("AllocTRES", "") in ("", "(null)"),
            "Already allocated",
        )
        requested = dict(x.split("=", 1) for x in f["ReqTRES"].split(","))
        require(
            requested.get("cpu") == "4" and memory(requested.get("mem", "")) == 12288,
            "Resource request differs",
        )
        require(
            all(not k.startswith("gres/gpu") or v == "0" for k, v in requested.items()),
            "GPU forbidden",
        )
        return f

    def verify_held(self, job):
        f = fields(self.command("held_job", ["scontrol", "show", "job", job, "-o"]))
        self.check_held_fields(f, job)
        script = self.command("stored_script", ["scontrol", "write", "batch_script", job, "-"])
        require(
            hashlib.sha256(script.encode()).hexdigest() == self.auth["script_sha256"],
            "Stored script differs",
        )
        accounting = self.command(
            "held_accounting",
            ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES"],
        )
        if not accounting.strip():
            raise WaitAccounting("Accounting not yet visible")
        rows = [line.split("|") for line in accounting.splitlines() if line.strip()]
        rows = [r for r in rows if r[0] == job]
        require(
            (len(rows) == 1 and rows[0][1:] == ["PENDING", "Unknown", "0", ""])
            or (len(rows) == 1 and rows[0][1:] == ["PENDING", "None", "0", ""]),
            "Accounting does not prove never started",
        )
        return f

    def expected_commands(self, job):
        capacity = [
            (
                "qos",
                [
                    "sacctmgr",
                    "-P",
                    "show",
                    "qos",
                    "where",
                    "name=" + TEACHER_QOS,
                    "format=Name,GrpSubmitJobs,MaxSubmitJobsPerUser,GrpJobs,MaxJobsPerUser,Flags",
                ],
            ),
            (
                "assoc",
                [
                    "sacctmgr",
                    "-P",
                    "show",
                    "assoc",
                    "where",
                    "user=" + self.user,
                    "account=rose",
                    "format=Cluster,Account,User,Partition,QOS%1000,GrpSubmitJobs,MaxSubmitJobs,GrpJobs,MaxJobs",
                ],
            ),
            ("queue", ["squeue", "-h", "-u", self.user, "-o", "%i|%u|%a|%q|%P|%T"]),
            ("node", ["scontrol", "show", "node", "cpu-1,cpu-2", "-o"]),
        ]
        held = [
            ("held_job", ["scontrol", "show", "job", job, "-o"]),
            ("stored_script", ["scontrol", "write", "batch_script", job, "-"]),
            (
                "held_accounting",
                [
                    "sacct",
                    "-n",
                    "-P",
                    "-j",
                    job,
                    "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES",
                ],
            ),
        ]
        return [
            *capacity,
            ("sbatch", ["sbatch", "--hold", "--parsable", str(self.script)]),
            *held,
            *capacity,
            *held,
            ("release", ["scontrol", "release", job]),
        ]

    def check_command_chain(self, result):
        commands = result.get("command_files", {})
        require(bool(commands), "Prior command evidence missing")
        job = result.get("job_id")
        require(
            isinstance(job, str) and bool(re.fullmatch(r"[1-9][0-9]*", job)),
            "Prior release job unknown",
        )
        parents = {str(Path(name).parent) for name in commands}
        require(len(parents) == 1, "Command invocation differs")
        directory = Path(next(iter(parents)))
        require(
            not directory.is_absolute()
            and ".." not in directory.parts
            and directory.parts[0] == "commands",
            "Unsafe command evidence",
        )
        expected = self.expected_commands(job)
        names = {
            str(directory / f"{i:03d}_{label}.{kind}.json")
            for i, (label, _) in enumerate(expected, 1)
            for kind in ("intent", "result")
        }
        require(set(commands) == names, "Command evidence inventory differs")
        responses = []
        for i, (label, argv) in enumerate(expected, 1):
            pair = []
            for kind in ("intent", "result"):
                name = str(directory / f"{i:03d}_{label}.{kind}.json")
                path = (self.work / name).resolve(strict=True)
                require(
                    path.is_relative_to(self.work / "commands") and sha(path) == commands[name],
                    "Prior command evidence changed",
                )
                record = read(path)
                require(record.get("argv") == argv, "Command argv differs")
                pair.append(record)
            require(pair[1].get("returncode") == 0, "Prior command outcome differs")
            require(
                isinstance(pair[0].get("at"), (int, float))
                and isinstance(pair[1].get("at"), (int, float))
                and pair[0]["at"] <= pair[1]["at"],
                "Command chronology differs",
            )
            if responses:
                require(responses[-1]["at"] <= pair[0]["at"], "Commands overlap or reorder")
            response = pair[1]
            if label == "sbatch":
                require(
                    bool(
                        re.fullmatch(
                            re.escape(job) + r"(?:;[A-Za-z0-9_-]+)?\n?", response.get("stdout", "")
                        )
                    ),
                    "Submitted job differs",
                )
            elif label == "held_job":
                self.check_held_fields(fields(response["stdout"]), job)
            elif label == "stored_script":
                require(
                    hashlib.sha256(response["stdout"].encode()).hexdigest()
                    == self.auth["script_sha256"],
                    "Stored script differs",
                )
            elif label == "held_accounting":
                rows = [line.split("|") for line in response["stdout"].splitlines() if line.strip()]
                rows = [r for r in rows if r[0] == job]
                require(
                    len(rows) == 1
                    and rows[0][1:]
                    in (["PENDING", "Unknown", "0", ""], ["PENDING", "None", "0", ""]),
                    "Held accounting differs",
                )
            responses.append(response)
        # Replay the exact accepted capacity queries, including the deliberate
        # distinction between held submission and runnable-node release.
        command = self.command
        try:
            for start, pre_submit, own_job, receipt in (
                (0, True, None, "CAPACITY_BEFORE_SUBMIT.json"),
                (8, False, job, "CAPACITY_BEFORE_RELEASE.json"),
            ):
                replay = iter(responses[start : start + 4])

                def replay_command(label, argv, replay=replay):
                    record = next(replay)
                    require(record["argv"] == argv, "Capacity replay argv differs")
                    return record["stdout"]

                self.command = replay_command
                verified = self.capacity(own_job=own_job, pre_submit=pre_submit)
                stored = read(self.work / receipt)
                require(
                    all(stored.get(k) == v for k, v in verified.items() if k != "at"),
                    "Capacity receipt differs from queries",
                )
                require(
                    responses[start + 3]["at"] <= stored["at"] <= responses[start + 4]["at"],
                    "Capacity receipt chronology differs",
                )
        finally:
            self.command = command
        allocation = read(self.work / "ALLOCATION_VERIFIED.json")
        require(
            allocation.get("fields") == fields(responses[5]["stdout"]),
            "Allocation receipt differs from command",
        )
        require(result.get("stdout") == responses[-1]["stdout"], "Release stdout differs")
        return responses

    def prior_release(self):
        result = read(self.work / "RELEASE.json")
        require(
            result.get("status") == "RELEASED" and result.get("returncode") == 0,
            "Prior release unknown",
        )
        require(
            result.get("authorization_sha256") == sha(self.authorization)
            and result.get("script_sha256") == sha(self.script),
            "Prior release binding differs",
        )
        expected = {
            "MAY_HAVE_SUBMITTED.json",
            "SUBMITTED.json",
            "ALLOCATION_VERIFIED.json",
            "CAPACITY_BEFORE_SUBMIT.json",
            "CAPACITY_BEFORE_RELEASE.json",
            "RELEASE_INTENT.json",
        }
        require(set(result.get("chain", {})) == expected, "Prior release chain incomplete")
        require(
            all(sha(self.work / name) == digest for name, digest in result["chain"].items()),
            "Prior release chain changed",
        )
        self.check_command_chain(result)
        audit = result.get("teacher_audit_files", {})
        require(bool(audit) and "COPY_LINEAGE.json" in audit, "Teacher scheduling audit missing")
        actual = {
            str(p.relative_to(self.work)): sha(p) for p in self.work.glob("commands/poll_*/*.json")
        }
        actual["COPY_LINEAGE.json"] = sha(self.work / "COPY_LINEAGE.json")
        require(actual == audit, "Teacher polling inventory changed")
        lineage = read(self.work / "COPY_LINEAGE.json")
        require(
            set(lineage)
            == {name for name in result["command_files"] if int(Path(name).name[:3]) <= 12},
            "Copy lineage inventory differs",
        )
        for name, item in lineage.items():
            require(
                name in result["command_files"]
                and item["source"] in audit
                and sha(self.work / name) == sha(self.work / item["source"]) == item["sha256"],
                "Copied evidence lineage differs",
            )
        self.history(require_retired=True)
        job = result.get("job_id")
        require(
            isinstance(job, str) and bool(re.fullmatch(r"[0-9]+", job)), "Prior release job unknown"
        )
        require(
            all(
                read(self.work / name).get("job_id") == job
                for name in ("SUBMITTED.json", "ALLOCATION_VERIFIED.json", "RELEASE_INTENT.json")
            ),
            "Prior release job differs",
        )
        # Reporting an already-released original never performs a scheduler mutation.
        return result

    def history(self, *, require_retired=False):
        source = self.authorization.parent / "lowmem_history.py"
        require(
            self.auth["extra_sources"].get(str(source.relative_to(self.root))) == sha(source),
            "History source differs",
        )
        module = ModuleType("bound_lowmem_history")
        module.__file__ = str(source)
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        paths = module.verify_history(self.root, self.auth, require_retired=require_retired)
        return module, paths

    def check_polls(self):
        for directory in sorted((self.work / "commands").glob("poll_*")):
            result = directory / "RESULT.json"
            require(
                result.exists(), "Incomplete poll; inspect preserved commands, no automatic retry"
            )
            record = read(result)
            require(
                record.get("status") in ("WAIT_CAPACITY", "WAIT_ACCOUNTING"),
                "Previous poll reached READY; no automatic retry",
            )
            actual = {p.name: sha(p) for p in directory.glob("*.json") if p.name != "RESULT.json"}
            require(record.get("files") == actual, "Prior poll changed")
            for path in directory.glob("*.result.json"):
                require(read(path).get("returncode") == 0, "Prior query outcome unknown")

    def poll_result(self, status, **extra):
        directory = self.work / "commands" / self.invocation
        value = dict(
            status=status,
            at=time.time(),
            files={p.name: sha(p) for p in directory.glob("*.json")},
            **extra,
        )
        save(directory / "RESULT.json", value)
        return value

    def retire(self):
        self.validate()
        module, _ = self.history()
        journal = self.root / self.auth["retirement_directory"]
        journal.mkdir(parents=True, exist_ok=True)
        with (
            self.root / "operations/delivery_recovery/20261021_delivery_teacher/SUBMIT.lock"
        ).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (journal / "RETIREMENT.json").exists():
                self.history(require_retired=True)
                return read(journal / "RETIREMENT.json")
            old_auth_path = (
                self.root / "operations" / module.COMMITS[0] / "RECOVERY_AUTHORIZATION.json"
            )
            old_source = old_auth_path.parent / "submit_delivery_recovery.py"
            old_module = ModuleType("old_teacher_submit")
            old_module.__file__ = str(old_source)
            exec(compile(old_source.read_bytes(), str(old_source), "exec"), old_module.__dict__)
            old = old_module.Submission(
                self.root, old_auth_path, runner=self.runner, validator=lambda *_: None
            )
            old.validate()
            self.serial = 0

            def command(directory, label, argv):
                self.serial += 1
                stem = f"{self.serial:03d}_{label}"
                save(directory / (stem + ".intent.json"), dict(argv=argv, at=time.time()))
                try:
                    response = self.runner(
                        argv, capture_output=True, text=True, timeout=30, check=False
                    )
                except BaseException as exc:
                    save(
                        directory / (stem + ".result.json"),
                        dict(status="UNKNOWN", error=repr(exc), at=time.time()),
                    )
                    raise
                record = dict(
                    argv=argv,
                    returncode=response.returncode,
                    stdout=response.stdout,
                    stderr=response.stderr,
                    at=time.time(),
                )
                save(directory / (stem + ".result.json"), record)
                require(response.returncode == 0, "Retirement command failed")
                return response.stdout

            if not (journal / "CANCEL_INTENT.json").exists():
                require(
                    not list(journal.glob("*.json")), "Incomplete retirement precheck; inspect only"
                )
                old.command = lambda label, argv: command(journal, label, argv)
                old.verify_held("216897")
                self.history()
                save(
                    journal / "CANCEL_INTENT.json",
                    dict(
                        job_id="216897",
                        at=time.time(),
                        authorization_sha256=sha(self.authorization),
                        authorization=self.auth,
                    ),
                )
                command(journal, "cancel", ["scancel", "--state=PENDING", "216897"])
            else:
                cancel = journal / "004_cancel.result.json"
                require(
                    cancel.exists()
                    and read(cancel).get("returncode") == 0
                    and read(cancel).get("argv") == ["scancel", "--state=PENDING", "216897"],
                    "Cancellation outcome unknown; inspect only",
                )
                require(
                    read(journal / "CANCEL_INTENT.json").get("authorization_sha256")
                    == sha(self.authorization),
                    "Cancellation authority changed",
                )
            # A successful cancellation may precede accounting propagation. Only reads repeat.
            for prior in (journal / "polls").glob("*"):
                require(
                    (prior / "RESULT.json").exists(), "Incomplete retirement poll; inspect only"
                )
                wait = read(prior / "RESULT.json")
                require(
                    wait.get("status") == "WAIT_TERMINAL"
                    and wait.get("files")
                    == {p.name: sha(p) for p in prior.glob("*.json") if p.name != "RESULT.json"},
                    "Retirement wait differs",
                )
            poll = journal / "polls" / str(time.time_ns())
            self.serial = 0
            raw = command(
                poll,
                "terminal",
                [
                    "sacct",
                    "-n",
                    "-P",
                    "-j",
                    "216897",
                    "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES,ExitCode",
                ],
            )
            queue = command(poll, "queue", ["squeue", "-h", "-j", "216897", "-o", "%i|%T"])
            rows = [line.split("|") for line in raw.splitlines() if line.strip()]
            rows = [r for r in rows if r[0] == "216897"]
            queue_rows = [line.split("|") for line in queue.splitlines() if line.strip()]
            require(
                not queue_rows
                or queue_rows in ([["216897", "PENDING"]], [["216897", "COMPLETING"]]),
                "Unexpected retired queue state",
            )
            zero_pending = (
                len(rows) == 1
                and len(rows[0]) == 6
                and rows[0][1] == "PENDING"
                and rows[0][2] in ("Unknown", "None")
                and rows[0][3:] == ["0", "", "0:0"]
            )
            waiting = not rows or zero_pending
            if rows and not zero_pending:
                module.check_terminal(raw)
                waiting = bool(queue_rows)
            if waiting:
                result = dict(
                    status="WAIT_TERMINAL",
                    job_id="216897",
                    files={p.name: sha(p) for p in poll.glob("*.json")},
                )
                save(poll / "RESULT.json", result)
                return result
            module.check_terminal(raw)
            require(not queue.strip(), "Retired recovery still queued")
            save(
                poll / "RESULT.json",
                dict(status="TERMINAL", files={p.name: sha(p) for p in poll.glob("*.json")}),
            )
            result = dict(
                status="CANCELLED_UNSTARTED_RECOVERY",
                job_id="216897",
                at=time.time(),
                historical_files=self.auth["retired_files"],
                terminal_poll=str(poll.relative_to(journal)),
                command_files={
                    str(p.relative_to(journal)): sha(p) for p in journal.rglob("*.json")
                },
            )
            save(journal / "RETIREMENT.json", result)
            self.history(require_retired=True)
            return result

    def submit(self):
        self.validate()
        self.history(require_retired=True)
        self.work.mkdir(parents=True, exist_ok=True)
        with (
            self.root / "operations/delivery_recovery/20261021_delivery_teacher/SUBMIT.lock"
        ).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.work / "SUBMITTED.json").exists():
                result = read(self.work / "SUBMITTED.json")
                require(
                    result.get("authorization_sha256") == sha(self.authorization)
                    and result.get("script_sha256") == sha(self.script),
                    "Submission binding changed",
                )
                return result
            require(
                not (self.work / "MAY_HAVE_SUBMITTED.json").exists(),
                "Prior submission exists; no automatic retry",
            )
            self.check_polls()
            self.invocation = "poll_submit_" + str(time.time_ns())
            self.serial = 0
            try:
                cap = self.capacity(pre_submit=True)
            except WaitCapacity as exc:
                return self.poll_result("WAIT_CAPACITY", reason=str(exc))
            self.poll_result("READY", capacity=cap)
            save(self.work / "CAPACITY_BEFORE_SUBMIT.json", cap)
            self.validate()
            self.history(require_retired=True)
            require(0 <= time.time() - cap["at"] <= 60, "Submit capacity stale")
            save(
                self.work / "MAY_HAVE_SUBMITTED.json",
                dict(
                    at=time.time(),
                    argv=["sbatch", "--hold", "--parsable", str(self.script)],
                    authorization_sha256=sha(self.authorization),
                    script_sha256=sha(self.script),
                ),
            )
            response = self.command("sbatch", ["sbatch", "--hold", "--parsable", str(self.script)])
            require(
                bool(re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_-]+)?\n?", response)),
                "Unknown sbatch identity; never retry",
            )
            job = response.strip().split(";")[0]
            require(
                job not in {"211333", "215520", "216128", "216142", "216897"},
                "Reused historical job id",
            )
            result = dict(
                status="SUBMITTED",
                job_id=job,
                stdout=response,
                at=time.time(),
                authorization_sha256=sha(self.authorization),
                script_sha256=sha(self.script),
                invocation=self.invocation,
            )
            save(self.work / "SUBMITTED.json", result)
            return result

    def continue_job(self):
        self.validate()
        self.history(require_retired=True)
        with (
            self.root / "operations/delivery_recovery/20261021_delivery_teacher/SUBMIT.lock"
        ).open("a") as lock:
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
            # Successful initial submit poll is sealed separately, not a retryable wait.
            for d in (self.work / "commands").glob("poll_continue_*"):
                require((d / "RESULT.json").exists(), "Incomplete poll; inspect only")
                record = read(d / "RESULT.json")
                require(
                    record.get("status") in ("WAIT_ACCOUNTING", "WAIT_CAPACITY"),
                    "Previous release poll not a safe wait",
                )
                require(
                    record.get("files")
                    == {p.name: sha(p) for p in d.glob("*.json") if p.name != "RESULT.json"},
                    "Wait poll changed",
                )
            self.invocation = "poll_continue_" + str(time.time_ns())
            self.serial = 0
            try:
                verified = self.verify_held(job)
                cap = self.capacity(own_job=job)
            except (WaitAccounting, WaitCapacity) as exc:
                return self.poll_result(
                    "WAIT_ACCOUNTING" if isinstance(exc, WaitAccounting) else "WAIT_CAPACITY",
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
            result["teacher_audit_files"]["COPY_LINEAGE.json"] = sha(
                self.work / "COPY_LINEAGE.json"
            )
            # Original worker only sees the release marker after all audit is durable.
            save(self.work / "RELEASE.json", result)
            return result

    def verify_final(self):
        self.validate()
        self.history(require_retired=True)
        release = self.prior_release()
        source = self.authorization.parent / "verify_recovered_migration.py"
        require(
            self.auth["extra_sources"].get(str(source.relative_to(self.root))) == sha(source),
            "Migration verifier changed",
        )
        module = ModuleType("bound_teacher_migration")
        module.__file__ = str(source)
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        migration = module.verify(self.root, self.authorization, runner=self.runner)
        recovery_receipt = read(self.root / "delivery/FINAL_RECOVERY_VERIFIED.json")
        profile = self.root / self.auth["profile_receipt"]
        require(
            recovery_receipt.get("status") == "FINAL_RECOVERY_VERIFIED"
            and recovery_receipt.get("archive") == migration.get("archive")
            and recovery_receipt.get("memory_profile_sha256") == sha(profile)
            and isinstance(recovery_receipt.get("delivery_revision"), dict),
            "Low-memory acceptance binding differs",
        )
        archive = self.root / "delivery/SR_F3_CE_20261016.zip"
        paths = self.history(require_retired=True)[1]
        paths.extend(self.work / name for name in release["teacher_audit_files"])
        archived = {}
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            require(len(names) == len(set(names)), "Duplicate archive entries")
            for path in sorted(set(paths)):
                name = "evidence/" + str(path.relative_to(self.root))
                require(name in names, "Teacher history missing from archive: " + name)
                h = hashlib.sha256()
                with package.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                require(h.hexdigest() == sha(path), "Teacher archived history differs")
                archived[name] = h.hexdigest()
        result = dict(
            status="FINAL_LOWMEM_RECOVERY_VERIFIED",
            job_id=release["job_id"],
            retired_job_id="216897",
            archive=migration["archive"],
            authorization_sha256=sha(self.authorization),
            archived_history=archived,
            delivery_revision=recovery_receipt["delivery_revision"],
            memory_profile_sha256=sha(profile),
            recovery_receipt_sha256=sha(self.root / "delivery/FINAL_RECOVERY_VERIFIED.json"),
            migration_receipt_sha256=sha(self.root / "delivery/FINAL_MIGRATION_VERIFIED.json"),
            supersedes=[
                "FINAL_HELD_RECOVERY_VERIFIED",
                "FINAL_HELD_CAPACITY_VERIFIED",
                "FINAL_TEACHER_RECOVERY_VERIFIED",
            ],
            at=time.time(),
        )
        target = self.root / "delivery/FINAL_LOWMEM_RECOVERY_VERIFIED.json"
        if target.exists():
            old = read(target)
            require(
                all(old.get(k) == v for k, v in result.items() if k != "at"),
                "Existing teacher verification differs",
            )
            return old
        save(target, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument(
        "--mode", required=True, choices=("retire", "submit", "continue", "verify-final")
    )
    args = parser.parse_args()
    operation = Submission(args.root, args.authorization)
    if args.mode == "verify-final":
        result = operation.verify_final()
    else:
        result = {
            "retire": operation.retire,
            "submit": operation.submit,
            "continue": operation.continue_job,
        }[args.mode]()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
