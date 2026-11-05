"""Read-only placement planning. No submission, cancellation, or model imports."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys


TARGET_QOS = "soujanya-poria-startfund-2026-03"
PRODUCTION_COMMIT = "f594e32705581d8bcfdd609560eda87fd11cbc31"
SQUEUE_FORMAT = "%i|%u|%q|%j|%T|%b|%D"
MAX_LEDGER_AGE_SECONDS = 900
PLANNED_ARGS = (
    "--nodes=1", "--gpus-per-node=pro6000:2", "--cpus-per-task=8",
    "--mem=66G", "--time=48:00:00", "--exclude=gpu-pro6000-12",
)
LIVE_STATES = frozenset((
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING",
    "SPECIAL_EXIT", "STAGE_OUT",
))


class UnsafeInput(ValueError):
    """Observation is ambiguous; no capacity authorization is possible."""


def positive_integer(value, field):
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise UnsafeInput("invalid " + field)
    return int(value)


def identifier(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise UnsafeInput("invalid " + field)
    return value


@dataclass(frozen=True)
class Request:
    job_id: str
    owner: str
    qos: str
    name: str
    gpu_type: str
    gpus_per_node: int
    nodes: int

    @property
    def gpus(self):
        return self.gpus_per_node * self.nodes


def request_from_fields(job_id, owner, qos, name, gpu_type, count, nodes):
    # Array/heterogeneous IDs need explicit expansion; never guess their cardinality.
    positive_integer(job_id, "job_id (only expanded numeric IDs supported)")
    count = positive_integer(count, "gpus_per_node")
    if gpu_type == "none":
        raise UnsafeInput("GPU absence requires independent ReqTRES evidence, unsupported here")
    return Request(str(job_id), identifier(owner, "owner"), identifier(qos, "qos"),
                   identifier(name, "name"), identifier(gpu_type, "gpu_type"),
                   count, positive_integer(nodes, "nodes"))


def insert_unique(target, request):
    if request.job_id in target and target[request.job_id] != request:
        raise UnsafeInput("conflicting resource/identity rows for job " + request.job_id)
    target[request.job_id] = request


def parse_queue(output, current_owner=None, target_qos=TARGET_QOS):
    if not isinstance(output, str):
        raise UnsafeInput("squeue output must be text")
    requests, states, observed_rows = {}, {}, {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 7:
            raise UnsafeInput("squeue row must have seven fields")
        job, owner, qos, name, state, tres, nodes = fields
        if not all(fields):
            raise UnsafeInput("empty field in squeue row")
        if job in observed_rows and observed_rows[job] != fields:
            raise UnsafeInput("conflicting squeue rows for job " + job)
        observed_rows[job] = fields
        identifier(owner, "owner")
        identifier(qos, "qos")
        # Only these rows can affect the three limits. Unrelated arrays/typed
        # resources in other QoS must not prevent a useful read-only decision.
        if (current_owner is not None and owner != current_owner and qos != target_qos
                and not name.startswith("LH-ORD-")):
            continue
        if state not in LIVE_STATES:
            raise UnsafeInput("unknown/non-live squeue state: " + state)
        match = re.fullmatch(r"gres/gpu:([A-Za-z0-9_.-]+):([1-9][0-9]*)", tres)
        if match:
            gpu_type, count = match[1], match[2]
        else:
            raise UnsafeInput("unsupported or ambiguous GPU TRES: " + tres)
        request = request_from_fields(job, owner, qos, name, gpu_type, count, nodes)
        insert_unique(requests, request)
        if job in states and states[job] != state:
            raise UnsafeInput("conflicting queue states for job " + job)
        states[job] = state
    return requests, states


def parse_ledger(ledger, now=None):
    if not isinstance(ledger, dict):
        raise UnsafeInput("ledger must be an object")
    required = {"at_utc", "basis", "unresolved_requests"}
    if set(ledger) != required:
        raise UnsafeInput("ledger fields must be at_utc, basis, unresolved_requests")
    at = ledger["at_utc"]
    if not isinstance(at, str) or not at.endswith("Z"):
        raise UnsafeInput("ledger at_utc must be a UTC ISO timestamp ending Z")
    try:
        audited_at = datetime.fromisoformat(at[:-1] + "+00:00")
    except ValueError as exc:
        raise UnsafeInput("invalid ledger at_utc") from exc
    observed_at = now if now is not None else datetime.now(timezone.utc)
    age_seconds = (observed_at - audited_at).total_seconds()
    if age_seconds < 0 or age_seconds > MAX_LEDGER_AGE_SECONDS:
        raise UnsafeInput("ledger audit is future-dated or older than 15 minutes")
    if not isinstance(ledger["basis"], str) or not ledger["basis"].strip():
        raise UnsafeInput("ledger basis must document the audit, including empty ledgers")
    if not isinstance(ledger["unresolved_requests"], list):
        raise UnsafeInput("unresolved_requests must be a list")
    requests = {}
    expected = {"job_id", "owner", "qos", "name", "gpu_type", "gpus_per_node", "nodes"}
    for entry in ledger["unresolved_requests"]:
        if not isinstance(entry, dict) or set(entry) != expected:
            raise UnsafeInput("unresolved request fields do not match documented schema")
        request = request_from_fields(entry["job_id"], entry["owner"], entry["qos"],
                                      entry["name"], entry["gpu_type"],
                                      entry["gpus_per_node"], entry["nodes"])
        insert_unique(requests, request)
    return requests


def plan(queue_output, ledger, owner, target_qos=TARGET_QOS, now=None):
    identifier(owner, "owner")
    identifier(target_qos, "target_qos")
    queued, states = parse_queue(queue_output, owner, target_qos)
    unresolved = parse_ledger(ledger, now=now)
    combined = dict(queued)
    for request in unresolved.values():
        insert_unique(combined, request)
    if any(r.qos == target_qos and r.gpu_type != "pro6000" for r in combined.values()):
        raise UnsafeInput("target QoS GPU type is not pro6000")
    experiment = [r for r in combined.values() if r.name.startswith("LH-ORD-")]
    target = [r for r in combined.values() if r.qos == target_qos]
    before = {
        "experiment_gpus": sum(r.gpus for r in experiment),
        "shared_target_qos_pro6000": sum(r.gpus for r in target),
        "owner_target_qos_jobs": sum(r.owner == owner for r in target),
    }
    increments = {"experiment_gpus": 2, "shared_target_qos_pro6000": 2,
                  "owner_target_qos_jobs": 1}
    limits = {"experiment_gpus": 6, "shared_target_qos_pro6000": 8,
              "owner_target_qos_jobs": 5}
    after = {key: value + increments[key] for key, value in before.items()}
    blocked = [key for key in limits if after[key] > limits[key]]
    rows = []
    for request in sorted(combined.values(), key=lambda r: int(r.job_id)):
        row = asdict(request)
        row.update(total_gpus=request.gpus, state=states.get(request.job_id, "UNKNOWN"),
                   in_unresolved_ledger=request.job_id in unresolved)
        rows.append(row)
    return {
        "schema": "SR_ORD_PLACEMENT_V1", "status": "BLOCKED_CAPACITY" if blocked else "READY",
        "read_only": True, "submission_performed": False,
        "planned_resource_args": list(PLANNED_ARGS) + ["--qos=" + target_qos],
        "production_commit": PRODUCTION_COMMIT, "owner": owner, "target_qos": target_qos,
        "before": before, "hypothetical_after_one_new_job": after,
        "limits": limits, "blocking_limits": blocked,
        "requests": rows, "other_qos_job_ids": [r["job_id"] for r in rows if r["qos"] != target_qos],
        "ledger_audit": {key: ledger[key] for key in ("at_utc", "basis")},
        "warning": "Planning snapshot only; re-audit immediately before any separately authorized submission. Queue absence is not evidence of release.",
    }


def verify_job_detail(output, request, expected_state=None):
    """Require unambiguous scheduler totals, not just minimum per-node TRES."""
    values = {}
    for field in ("JobId", "JobName", "UserId", "QOS", "JobState", "NumNodes", "ReqTRES"):
        matches = re.findall(r"(?:^|\s)" + field + r"=([^\s]+)", output)
        if len(matches) != 1:
            raise UnsafeInput("missing/ambiguous scontrol field " + field)
        values[field] = matches[0]
    if values["JobId"] != request.job_id or values["QOS"] != request.qos:
        raise UnsafeInput("scontrol job/QoS identity differs from squeue")
    if values["JobName"] != request.name:
        raise UnsafeInput("scontrol job name differs from squeue")
    if values["JobState"] not in LIVE_STATES or (expected_state is not None and values["JobState"] != expected_state):
        raise UnsafeInput("scontrol job state is terminal, unknown, or changed during observation")
    if not re.fullmatch(re.escape(request.owner) + r"\([0-9]+\)", values["UserId"]):
        raise UnsafeInput("scontrol owner differs from squeue")
    if positive_integer(values["NumNodes"], "scontrol NumNodes (ranges unsupported)") != request.nodes:
        raise UnsafeInput("scontrol node count differs from squeue")
    tres = {}
    for component in values["ReqTRES"].split(","):
        if component.count("=") != 1:
            raise UnsafeInput("malformed ReqTRES")
        key, value = component.split("=")
        if key in tres or not key or not value:
            raise UnsafeInput("duplicate/empty ReqTRES")
        tres[key] = value
    gpu_key = "gres/gpu:" + request.gpu_type
    if gpu_key not in tres or positive_integer(tres[gpu_key], "ReqTRES GPU total") != request.gpus:
        raise UnsafeInput("ReqTRES typed GPU total differs from per-node estimate")
    if any(key.startswith("gres/gpu:") and key != gpu_key for key in tres):
        raise UnsafeInput("multiple GPU types in ReqTRES")
    if "gres/gpu" in tres and positive_integer(tres["gres/gpu"], "ReqTRES generic GPU total") != request.gpus:
        raise UnsafeInput("ReqTRES generic GPU total differs")
    if "node" not in tres or positive_integer(tres["node"], "ReqTRES nodes") != request.nodes:
        raise UnsafeInput("ReqTRES node count differs")
    allocations = re.findall(r"(?:^|\s)AllocTRES=([^\s]+)", output)
    if len(allocations) > 1:
        raise UnsafeInput("ambiguous AllocTRES")
    if not allocations and values["JobState"] != "PENDING":
        raise UnsafeInput("non-pending job lacks AllocTRES")
    if allocations and allocations[0] in ("(null)", "N/A") and values["JobState"] == "PENDING":
        # No allocation yet; the complete verified ReqTRES still counts in full.
        return
    if allocations:
        allocated = {}
        for component in allocations[0].split(","):
            if component.count("=") != 1:
                raise UnsafeInput("malformed AllocTRES")
            key, value = component.split("=")
            if key in allocated or not key or not value:
                raise UnsafeInput("duplicate/empty AllocTRES")
            allocated[key] = value
        if gpu_key not in allocated:
            raise UnsafeInput("AllocTRES lacks typed GPU total")
        for key, value in allocated.items():
            if key.startswith("gres/gpu:") and key != gpu_key:
                raise UnsafeInput("unexpected GPU type in AllocTRES")
            if key in ("gres/gpu", gpu_key) and positive_integer(value, "allocated GPU total") > request.gpus:
                raise UnsafeInput("allocated GPU total exceeds counted request")


def read_scheduler(command, timeout, environment):
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            timeout=timeout, env=environment)
    if result.stderr.strip():
        raise UnsafeInput("scheduler command emitted diagnostic stderr; observation not accepted")
    return result.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unresolved-ledger", required=True, type=Path)
    parser.add_argument("--owner", default=getpass.getuser())
    parser.add_argument("--target-qos", default=TARGET_QOS)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.timeout <= 60:
            raise UnsafeInput("timeout must be in (0, 60] seconds")
        ledger = json.loads(args.unresolved_ledger.read_text(encoding="utf-8"))
        parse_ledger(ledger)
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SQUEUE_")}
        output = read_scheduler(["squeue", "--all", "--states=all", "--noheader",
                                 "--format=" + SQUEUE_FORMAT], args.timeout, environment)
        queued, states = parse_queue(output, args.owner, args.target_qos)
        for request in queued.values():
            detail = read_scheduler(["scontrol", "show", "job", "-o", request.job_id],
                                    args.timeout, environment)
            verify_job_detail(detail, request, states[request.job_id])
        report = plan(output, ledger, args.owner, args.target_qos)
        report["scontrol_verified_job_ids"] = sorted(queued, key=int)
        report["observed_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        exit_code = 0 if report["status"] == "READY" else 1
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report = {"schema": "SR_ORD_PLACEMENT_V1", "status": "BLOCKED_INPUT",
                  "read_only": True, "submission_performed": False,
                  "planned_resource_args": list(PLANNED_ARGS), "error": str(exc)}
        exit_code = 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
