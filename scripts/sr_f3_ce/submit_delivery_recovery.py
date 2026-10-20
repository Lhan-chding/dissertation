"""One fail-closed held submission for the separately authorized delivery recovery."""

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
from pathlib import Path


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            node="cpu-1",
            qos="rose",
            account="rose",
            cpus=22,
            memory_mib=67584,
            time_limit="3-00:00:00",
        )
        require(
            all(type(a.get(k)) is type(v) and a.get(k) == v for k, v in expected.items()),
            "Fixed recovery resources differ",
        )
        require(a["root"] == str(self.root), "Root differs")
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

    def capacity(self, *, own_job=None):
        qos = table(
            self.command(
                "qos",
                [
                    "sacctmgr",
                    "-P",
                    "show",
                    "qos",
                    "where",
                    "name=rose",
                    "format=Name,GrpSubmitJobs,MaxSubmitJobsPerUser,GrpJobs,MaxJobsPerUser,Flags",
                ],
            )
        )
        qos = [q for q in qos if q.get("Name") == "rose"]
        require(len(qos) == 1, "Ambiguous rose QoS")
        q = qos[0]
        require(
            limit(q["GrpSubmit"]) is None and limit(q["GrpJobs"]) is None,
            "Cannot prove private group capacity",
        )
        require("relative" not in q["Flags"].lower(), "Relative QoS unsupported")
        require(
            limit(q["MaxSubmitPU"]) == 5 and limit(q["MaxJobsPU"]) == 2,
            "Rose limits changed; review required",
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
        require(all("rose" in x["QOS"].split(",") for x in assoc), "Rose membership missing")
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
                own_job in queue and queue[own_job][3] == "rose" and queue[own_job][5] == "PENDING",
                "Held self missing from queue",
            )
        submitted = sum(row[3] == "rose" for row in queue.values())
        running = sum(row[3] == "rose" and row[5] != "PENDING" for row in queue.values())
        require(submitted + (0 if own_job else 1) <= 5 and running < 2, "Rose capacity unavailable")
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
                require(cap is None or count <= cap, "Association capacity unavailable")
        node = fields(self.command("node", ["scontrol", "show", "node", "cpu-1", "-o"]))
        require(
            node.get("NodeName") == "cpu-1"
            and "cluster02" in node.get("Partitions", "").split(","),
            "Node/partition differs",
        )
        require(
            node.get("State") in ("IDLE", "MIXED")
            and "cpu_ok" in node.get("ActiveFeatures", "").split(","),
            "Node unavailable",
        )
        require(int(node["CPUEfctv"]) - int(node["CPUAlloc"]) >= 22, "Insufficient CPUs")
        require(
            int(node["RealMemory"]) - int(node["AllocMem"]) - int(node.get("MemSpecLimit", "0"))
            >= 67584,
            "Insufficient memory",
        )
        return dict(
            at=time.time(), queue=queue, qos=q, associations=assoc, node=node, self_job=own_job
        )

    def check_held_fields(self, f, job):
        expected = dict(
            JobId=job,
            JobName=self.name,
            Account="rose",
            QOS="rose",
            Partition="cluster02",
            Command=str(self.script),
            Comment=self.comment,
            NumCPUs="22",
            NumTasks="1",
            **{"CPUs/Task": "22"},
            ReqNodeList="cpu-1",
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
            requested.get("cpu") == "22" and memory(requested.get("mem", "")) == 67584,
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
                    "name=rose",
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
            ("node", ["scontrol", "show", "node", "cpu-1", "-o"]),
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

    def submit(self):
        self.validate()
        self.work.mkdir(parents=True, exist_ok=True)
        with (self.work / "SUBMIT.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.invocation = str(time.time_ns())
            if (self.work / "RELEASE.json").exists():
                return self.prior_release()
            require(
                not (self.work / "MAY_HAVE_SUBMITTED.json").exists(),
                "Prior submission exists; no automatic retry",
            )
            cap = self.capacity()
            save(self.work / "CAPACITY_BEFORE_SUBMIT.json", cap)
            self.validate()
            require(0 <= time.time() - cap["at"] <= 60, "Capacity snapshot stale")
            argv = ["sbatch", "--hold", "--parsable", str(self.script)]
            save(
                self.work / "MAY_HAVE_SUBMITTED.json",
                dict(
                    at=time.time(),
                    argv=argv,
                    authorization_sha256=sha(self.authorization),
                    script_sha256=sha(self.script),
                ),
            )
            response = self.command("sbatch", argv)
            require(
                bool(re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_-]+)?\n?", response)),
                "Unknown sbatch identity; never retry",
            )
            job = response.strip().split(";")[0]
            save(
                self.work / "SUBMITTED.json",
                dict(
                    status="SUBMITTED",
                    job_id=job,
                    stdout=response,
                    at=time.time(),
                    authorization_sha256=sha(self.authorization),
                    script_sha256=sha(self.script),
                ),
            )
            verified = self.verify_held(job)
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
            cap = self.capacity(own_job=job)
            save(self.work / "CAPACITY_BEFORE_RELEASE.json", cap)
            self.validate()
            self.verify_held(job)
            require(0 <= time.time() - cap["at"] <= 60, "Release capacity snapshot stale")
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
            )
            result["command_files"] = {
                str(path.relative_to(self.work)): sha(path)
                for path in sorted((self.work / "commands" / self.invocation).glob("*.json"))
            }
            save(self.work / "RELEASE.json", result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    print(json.dumps(Submission(args.root, args.authorization).submit(), sort_keys=True))


if __name__ == "__main__":
    main()
