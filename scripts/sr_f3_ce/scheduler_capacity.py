"""Light Slurm capacity discovery for operational scheduling, never scientific state.

The scheduler is authoritative. Current limits are read at allocation boundaries;
historical claims retain their original small capacity snapshot. Model files and
checkpoints are never read here. No numerical GPU ceiling is configured in code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

POLICY = "LIVE_SLURM_QOS_AND_ASSOCIATION"
LIMIT_KEYS = ("gpu_limit", "submitted_limit", "max_tasks_per_job", "gpu_per_user_limit")
QOS = "soujanya-poria-startfund-2026-03"
QOS_FORMAT = (
    "Name%80,GrpTRES%300,MaxTRESPerUser%300,MaxTRESPerJob%300,"
    "GrpJobs,GrpSubmitJobs,MaxJobsPerUser,MaxSubmitJobsPerUser,Flags%100"
)
ASSOC_FORMAT = (
    "Cluster,Account,User,Partition,GrpTRES%300,MaxTRESPerJob%300,"
    "MaxTRESPerNode%300,GrpJobs,GrpSubmitJobs,MaxJobs,MaxSubmitJobs"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(raw):
    lines = [line.split("|") for line in raw.splitlines() if line.strip()]
    if len(lines) < 2 or len(set(lines[0])) != len(lines[0]):
        raise PermissionError("Scheduler capacity output is empty or ambiguous")
    header = lines[0]
    if any(len(row) != len(header) for row in lines[1:]):
        raise PermissionError("Scheduler capacity output is truncated")
    return [dict(zip(header, row, strict=True)) for row in lines[1:]]


def _number(value):
    if value in (None, "", "-1", "UNLIMITED", "NONE"):
        return None
    if not re.fullmatch(r"[0-9]+", str(value)):
        raise PermissionError("Unparseable scheduler capacity: " + str(value))
    return int(value)


def _minimum(*values):
    bounded = [v for v in values if v is not None]
    return min(bounded) if bounded else None


def _tres(value):
    result = {}
    if not value:
        return result
    for entry in value.split(","):
        if "=" not in entry:
            raise PermissionError("Malformed scheduler TRES limit")
        key, number = entry.split("=", 1)
        if key in result:
            raise PermissionError("Duplicate scheduler TRES limit")
        result[key] = number
    return result


def _gpu(value):
    tres = _tres(value)
    return _minimum(_number(tres.get("gres/gpu")), _number(tres.get("gres/gpu:pro6000")))


def _memory_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value, re.IGNORECASE)
    if not match:
        raise PermissionError("Unparseable scheduler memory limit")
    return int(match[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[match[2].upper()]


def _job_tasks(value):
    tres = _tres(value)
    cpu = _number(tres.get("cpu"))
    memory = tres.get("mem")
    return _minimum(
        _gpu(value),
        None if cpu is None else cpu // 4,
        None if memory is None else int(_memory_mib(memory) // (90 * 1024)),
    )


def _node_tasks(raw):
    eligible = []
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 6:
            raise PermissionError("Malformed Slurm node capacity output")
        name, gres, memory, cpus, features, state = fields
        if name in seen:
            raise PermissionError("Duplicate node capacity record")
        seen.add(name)
        if not memory.isdigit() or not cpus.isdigit() or int(memory) < 1 or int(cpus) < 1:
            raise PermissionError("Invalid node memory or CPU capacity")
        if "highmem" not in features.split(",") or not re.search(r"(?:^|,)gpu:pro6000:", gres):
            continue
        if any(flag in state.lower() for flag in ("down", "drain", "fail", "unk")):
            continue
        matches = re.findall(r"(?:^|,)gpu:pro6000:([0-9]+)(?:\([^)]*\))?(?=,|$)", gres)
        if len(matches) != 1:
            raise PermissionError("Ambiguous PRO6000 node capacity")
        count = min(int(matches[0]), int(memory) // (90 * 1024), int(cpus) // 4)
        if count:
            eligible.append(dict(node=name, max_tasks=count, state=state))
    return max((row["max_tasks"] for row in eligible), default=0), eligible


def derive(qos_text, assoc_text, nodes_text, *, user, account="rose", partition="cluster02"):
    qos_rows = [row for row in _rows(qos_text) if row.get("Name") == QOS]
    if len(qos_rows) != 1:
        raise PermissionError("The actual experiment QoS must have exactly one record")
    qos = qos_rows[0]
    if "relative" in qos.get("Flags", "").lower().split(","):
        raise PermissionError("Relative QoS limits require partition-relative capacity accounting")
    required = {"GrpTRES", "MaxTRESPU", "MaxTRES", "GrpSubmit", "MaxSubmitPU"}
    if not required <= qos.keys():
        raise PermissionError("QoS capacity fields are missing")
    assoc = [
        row
        for row in _rows(assoc_text)
        if row.get("User") == user
        and row.get("Account") == account
        and row.get("Partition") in ("", partition)
    ]
    if not assoc:
        raise PermissionError("The experiment's scheduler association is unavailable")
    # These are user-association records. Account ancestors can be private; Slurm
    # enforces inaccessible ancestor and running/billing limits at scheduling.
    if any(
        not {"GrpTRES", "MaxTRES", "MaxTRESPerNode", "GrpSubmit", "MaxSubmit"} <= row.keys()
        for row in assoc
    ):
        raise PermissionError("Association capacity fields are missing")
    if _number(qos["GrpSubmit"]) is not None:
        raise PermissionError("QoS group submission limit needs group-job accounting")
    for value in [qos["GrpTRES"], qos["MaxTRESPU"], *(row["GrpTRES"] for row in assoc)]:
        if any(key in _tres(value) for key in ("cpu", "mem", "node")):
            raise PermissionError(
                "Additional aggregate CPU/memory/node limits require capacity accounting"
            )
    gpu_limit = _gpu(qos["GrpTRES"])
    user_limit = _minimum(_gpu(qos["MaxTRESPU"]), *(_gpu(row["GrpTRES"]) for row in assoc))
    submitted = _minimum(
        _number(qos["MaxSubmitPU"]),
        *(_number(row["MaxSubmit"]) for row in assoc),
        *(_number(row["GrpSubmit"]) for row in assoc),
    )
    node_limit, nodes = _node_tasks(nodes_text)
    per_job = _minimum(
        node_limit,
        gpu_limit,
        user_limit,
        _job_tasks(qos["MaxTRES"]),
        *(_job_tasks(row["MaxTRES"]) for row in assoc),
        *(_job_tasks(row["MaxTRESPerNode"]) for row in assoc),
    )
    return dict(
        gpu_limit=gpu_limit,
        submitted_limit=submitted,
        max_tasks_per_job=per_job,
        gpu_per_user_limit=user_limit,
    ), nodes


def derive_dispatcher(qos_text, assoc_text, *, qos, user, account="rose", partition="cluster02"):
    """CPU submission limits have separate scopes from science GPU capacity."""
    if qos != "rose":
        raise PermissionError("Only ordinary rose dispatcher QoS is authorized")
    records = [row for row in _rows(qos_text) if row.get("Name") == qos]
    if len(records) != 1:
        raise PermissionError("Dispatcher QoS record is unavailable or ambiguous")
    record = records[0]
    required = {"GrpSubmit", "MaxSubmitPU", "GrpJobs", "MaxJobsPU", "Flags"}
    if not required <= record.keys():
        raise PermissionError("Dispatcher QoS capacity fields are missing")
    if "relative" in record["Flags"].lower().split(","):
        raise PermissionError("Relative dispatcher QoS is unsupported")
    assoc = [
        row
        for row in _rows(assoc_text)
        if row.get("User") == user
        and row.get("Account") == account
        and row.get("Partition") in ("", partition)
    ]
    if not assoc or len({row.get("Cluster") for row in assoc}) != 1:
        raise PermissionError("Dispatcher association is unavailable or ambiguous")
    fields = {"Cluster", "Partition", "QOS", "GrpSubmit", "MaxSubmit", "GrpJobs", "MaxJobs"}
    if any(not fields <= row.keys() for row in assoc):
        raise PermissionError("Dispatcher association fields are missing")
    if len({row["Partition"] for row in assoc}) != len(assoc):
        raise PermissionError("Duplicate dispatcher association")
    associations = []
    for row in assoc:
        membership = row["QOS"].split(",")
        if qos not in membership or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in membership):
            raise PermissionError("Ordinary dispatcher QoS membership is not explicit")
        associations.append(
            dict(
                partition=row["Partition"],
                submitted_limit=_minimum(_number(row["GrpSubmit"]), _number(row["MaxSubmit"])),
                running_limit=_minimum(_number(row["GrpJobs"]), _number(row["MaxJobs"])),
            )
        )
    return dict(
        qos=qos,
        user=user,
        account=account,
        partition=partition,
        qos_group_submitted_limit=_number(record["GrpSubmit"]),
        qos_user_submitted_limit=_number(record["MaxSubmitPU"]),
        qos_group_running_limit=_number(record["GrpJobs"]),
        qos_user_running_limit=_number(record["MaxJobsPU"]),
        associations=associations,
    )


class Capacity:
    def __init__(self, value, *, execute=subprocess.run, clock=time.time):
        self.value = copy.deepcopy(value)
        self.root = Path(value["root"]).resolve(strict=True)
        self.execute, self.clock = execute, clock
        self._cached = None

    def dispatcher_limits(self, qos="rose", *, fresh=True):
        """Always query before the caller obtains its fresh queue; never cache."""
        del fresh  # Compatibility with capacity callers; ordinary limits are always fresh.
        if qos != "rose":
            raise PermissionError("Only ordinary rose dispatcher QoS is authorized")
        user = self.value["user"]
        account = self.value.get("account", "rose")
        partition = self.value.get("partition", "cluster02")
        commands = [
            ["sacctmgr", "-P", "show", "qos", "where", "name=" + qos, "format=" + QOS_FORMAT],
            [
                "sacctmgr",
                "-P",
                "show",
                "assoc",
                "where",
                "user=" + user,
                "account=" + account,
                "format=" + ASSOC_FORMAT + ",QOS%1000",
            ],
        ]
        observations = []
        record = dict(
            schema_version=1,
            capacity_policy="LIVE_SLURM_DISPATCHER_QOS_AND_ASSOCIATION",
            root=str(self.root),
            qos=qos,
            user=user,
            account=account,
            partition=partition,
            observed_at=self.clock(),
            observations=observations,
        )
        try:
            for command in commands:
                result = self.execute(
                    command, capture_output=True, text=True, timeout=20, check=False
                )
                observations.append(
                    dict(
                        command=command,
                        returncode=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                )
                if result.returncode:
                    raise RuntimeError("Dispatcher scheduler capacity unavailable")
            limits = derive_dispatcher(
                *(item["stdout"] for item in observations),
                qos=qos,
                user=user,
                account=account,
                partition=partition,
            )
            record.update(status="PASS", limits=limits)
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):

                def decoded(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else value

                observations.append(
                    dict(
                        command=exc.cmd,
                        timeout=True,
                        stdout=decoded(exc.stdout),
                        stderr=decoded(exc.stderr),
                    )
                )
            record.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
            self._save_dispatcher_capacity(record)
            raise
        path, digest = self._save_dispatcher_capacity(record)
        return dict(limits, capacity_snapshot_path=str(path), capacity_snapshot_sha256=digest)

    def _save_dispatcher_capacity(self, record):
        directory = self.root / "operations/capacity/dispatcher_snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        body = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(body).hexdigest()
        path = directory / (str(time.time_ns()) + "-" + digest[:16] + ".json")
        with path.open("xb") as stream:
            stream.write(body)
        return path, digest

    def dispatcher_headroom(self, queue, limits, *, queue_complete=False):
        """Count pending, held and unknown jobs as submissions, regardless of GPUs.

        The normal queue contains all of this user plus shared teacher jobs. A
        finite ordinary QoS group limit requires an explicitly complete queue.
        Running caps are diagnostics: Slurm may queue jobs at those caps.
        """
        if any(
            limits.get(key) != expected
            for key, expected in (
                ("qos", "rose"),
                ("user", self.value["user"]),
                ("account", self.value.get("account", "rose")),
                ("partition", self.value.get("partition", "cluster02")),
            )
        ):
            raise PermissionError("Dispatcher capacity scope differs")
        if limits["qos_group_submitted_limit"] is not None and not queue_complete:
            raise PermissionError("Dispatcher QoS group submission limit requires complete queue")
        if not isinstance(queue, dict) or any(not isinstance(row, dict) for row in queue.values()):
            raise PermissionError("Dispatcher queue is unavailable or malformed")
        usage = dict(qos_group_submitted=0, qos_user_submitted=0, association_submitted=[])
        assoc_counts = [0] * len(limits["associations"])

        def identity(row, key, alias):
            value = row.get(key, row.get(alias))
            if value is None or str(value).lower() in ("", "unknown", "(null)", "n/a", "none"):
                return None
            return str(value)

        for row in queue.values():
            # No state exclusion: even a terminal-looking row still in squeue
            # conservatively occupies a slot until it leaves the live queue.
            username = identity(row, "UserId", "user")
            if username:
                username = str(username).split("(", 1)[0]
            own = not username or username == limits["user"]
            row_qos = identity(row, "QOS", "qos")
            target = not row_qos or row_qos == limits["qos"]
            if target:
                usage["qos_group_submitted"] += 1
                usage["qos_user_submitted"] += int(own)
            row_account = identity(row, "Account", "account")
            if own and (not row_account or row_account == limits["account"]):
                for i, association in enumerate(limits["associations"]):
                    row_partition = identity(row, "Partition", "partition")
                    if (
                        not association["partition"]
                        or not row_partition
                        or row_partition == association["partition"]
                    ):
                        assoc_counts[i] += 1
        usage["association_submitted"] = assoc_counts
        rooms = []
        for key in ("qos_group_submitted", "qos_user_submitted"):
            limit = limits[key + "_limit"]
            if limit is not None:
                rooms.append(limit - usage[key])
        for count, association in zip(assoc_counts, limits["associations"], strict=True):
            if association["submitted_limit"] is not None:
                rooms.append(association["submitted_limit"] - count)
        if not rooms:
            raise PermissionError("Dispatcher submission capacity has no finite serializable bound")
        return dict(
            submitted=min(rooms),
            usage=usage,
            group_usage_complete=queue_complete,
            running_limits=dict(
                group=limits["qos_group_running_limit"],
                user=limits["qos_user_running_limit"],
                associations=[a["running_limit"] for a in limits["associations"]],
            ),
        )

    def limits(self, fresh=False):
        now = self.clock()
        if not fresh and self._cached is not None and 0 <= now - self._cached[0] < 60:
            return copy.deepcopy(self._cached[1])
        user, account, partition = (
            self.value["user"],
            self.value.get("account", "rose"),
            self.value.get("partition", "cluster02"),
        )
        commands = [
            ["sacctmgr", "-P", "show", "qos", "where", "name=" + QOS, "format=" + QOS_FORMAT],
            [
                "sacctmgr",
                "-P",
                "show",
                "assoc",
                "where",
                "user=" + user,
                "account=" + account,
                "format=" + ASSOC_FORMAT,
            ],
            ["sinfo", "-h", "-N", "-p", partition, "-o", "%N|%G|%m|%c|%f|%t"],
        ]
        observations = []
        for command in commands:
            result = self.execute(command, capture_output=True, text=True, timeout=20, check=False)
            if result.returncode:
                raise RuntimeError("Live scheduler capacity unavailable: " + result.stderr.strip())
            observations.append(dict(command=command, stdout=result.stdout, stderr=result.stderr))
        limits, nodes = derive(
            *(row["stdout"] for row in observations),
            user=user,
            account=account,
            partition=partition,
        )
        record = dict(
            schema_version=1,
            capacity_policy=POLICY,
            root=str(self.root),
            qos=QOS,
            user=user,
            account=account,
            partition=partition,
            observed_at=self.clock(),
            limits=limits,
            eligible_nodes=nodes,
            observations=observations,
        )
        directory = self.root / "operations/capacity/snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        body = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(body).hexdigest()
        path = directory / (str(time.time_ns()) + "-" + digest[:16] + ".json")
        with path.open("xb") as stream:
            stream.write(body)
        answer = dict(limits, capacity_snapshot_path=str(path), capacity_snapshot_sha256=digest)
        self._cached = (now, answer)
        return copy.deepcopy(answer)

    def verify_snapshot(self, intent):
        path = Path(intent["capacity_snapshot_path"]).resolve(strict=True)
        if not path.is_relative_to(self.root / "operations/capacity/snapshots") or _sha(
            path
        ) != intent.get("capacity_snapshot_sha256"):
            raise PermissionError("Retained scheduler capacity snapshot differs")
        record = _read(path)
        expected = dict(
            schema_version=1,
            capacity_policy=POLICY,
            root=str(self.root),
            qos=QOS,
            user=self.value["user"],
            account=self.value.get("account", "rose"),
            partition=self.value.get("partition", "cluster02"),
        )
        if any(record.get(k) != v for k, v in expected.items()):
            raise PermissionError("Capacity snapshot belongs to a different scheduler scope")
        observations = record.get("observations", [])
        if len(observations) != 3:
            raise PermissionError("Capacity snapshot lacks scheduler evidence")
        limits, nodes = derive(
            *(row["stdout"] for row in observations),
            user=expected["user"],
            account=expected["account"],
            partition=expected["partition"],
        )
        if (
            record.get("limits") != limits
            or record.get("eligible_nodes") != nodes
            or intent.get("capacity_limits") != limits
        ):
            raise PermissionError("Capacity snapshot was not derived from its scheduler evidence")
        if (
            type(intent.get("max_tasks_per_job")) is not int
            or intent["max_tasks_per_job"] != limits["max_tasks_per_job"]
        ):
            raise PermissionError("Pool size differs from the retained scheduler capacity")
        return dict(
            limits,
            capacity_snapshot_path=str(path),
            capacity_snapshot_sha256=intent["capacity_snapshot_sha256"],
        )

    def headroom(self, usage, limits=None):
        limits = self.limits() if limits is None else limits

        def remaining(key, current):
            return math.inf if limits.get(key) is None else limits[key] - current

        gpu_room = remaining("gpu_limit", usage["gpus"])
        if limits.get("gpu_per_user_limit") is not None:
            if "user_gpus" not in usage:
                raise PermissionError("Per-user GPU accounting is required by the live QoS")
            gpu_room = min(gpu_room, remaining("gpu_per_user_limit", usage["user_gpus"]))
        return dict(gpus=gpu_room, submitted=remaining("submitted_limit", usage["submitted"]))

    def assert_within(self, usage, limits=None):
        room = self.headroom(usage, limits)
        if room["gpus"] < 0 or room["submitted"] < 0:
            raise PermissionError("Actual reservations exceed the live scheduler limits")
        return room
