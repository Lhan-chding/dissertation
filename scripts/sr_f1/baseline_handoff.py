#!/usr/bin/env python3
"""Observe the existing ENGINE, then activate the authenticated parallel baseline.

This CPU-only bridge never ticks the scheduler, submits a GPU job, or cancels a
GPU. The old source stays installed until ENGINE's complete scientific gate and
Slurm terminal state agree. Source promotion is recoverable from its fsynced
intent, including interruption between the two directory renames.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

INCIDENT = "technical_incidents/baseline_parallel_20261010"
CANDIDATE = "code_baseline_parallel_candidate_20261010"
PRESERVED = "code_before_baseline_parallel_20261010"
REPAIR = "BASELINE_PARALLEL_REPAIR.json"
CPU_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
SACCT_FIELDS = "JobIDRaw,JobName,State,User,Account,QOS,AllocTRES,ExitCode,Comment"
EVIDENCE = (
    "orchestration/STATE.json",
    "orchestration/journal.jsonl",
    "orchestration/REGISTRATION.json",
    "orchestration/completions/ENGINE.json",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sync_folder(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, value):
    """Exclusive, durable creation; identical reentry is permitted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, "IMMUTABLE_HANDOFF_RECEIPT_CHANGED:" + str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        sync_folder(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lease(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def scheduler_locks(root):
    with (
        lease(root.parent / ".sr_f1_project_controller.lock"),
        lease(root / "orchestration/controller.lock"),
    ):
        yield


def command(arguments, *, timeout=45, environment=None, cwd=None):
    result = subprocess.run(
        list(map(str, arguments)),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment or {**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "standard"},
        cwd=cwd,
    )
    return dict(
        command=list(map(str, arguments)),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        time=now(),
    )


def gpu_count(tres):
    fields = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    values = {
        key: value
        for key, value in fields.items()
        if key == "gres/gpu" or key.startswith("gres/gpu:")
    }
    require(all(value.isdigit() for value in values.values()), "UNKNOWN_GPU_TRES")
    return int(values["gres/gpu"]) if "gres/gpu" in values else sum(map(int, values.values()))


class ObservationUnknown(RuntimeError):
    """Scheduler visibility is incomplete; never infer failure or activate."""


class Slurm:
    def __init__(self, runner=command, owner=None):
        self.run = runner
        self.owner = owner or os.environ.get("USER", "")

    def snapshot(self, job_id):
        require(re.fullmatch(r"[0-9]+", str(job_id)), "INVALID_JOB_ID")
        try:
            queue = self.run(["squeue", "--noheader", "--user=" + self.owner, "--format=%i|%T"])
            accounting = self.run(
                [
                    "sacct",
                    "--allocations",
                    "--noheader",
                    "--parsable2",
                    "--jobs=" + str(job_id),
                    "--format=" + SACCT_FIELDS,
                ]
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ObservationUnknown("SLURM_QUERY_UNAVAILABLE") from exc
        if queue["returncode"] != 0 or accounting["returncode"] != 0:
            raise ObservationUnknown("SLURM_STATE_UNKNOWN")
        all_queued = [
            line.strip().split("|") for line in queue["stdout"].splitlines() if line.strip()
        ]
        require(all(len(row) == 2 for row in all_queued), "INVALID_QUEUE_ROW")
        queued = [row for row in all_queued if row[0] == str(job_id)]
        rows = []
        for line in accounting["stdout"].splitlines():
            if not line.strip():
                continue
            values = line.strip().split("|")
            if len(values) == len(SACCT_FIELDS.split(",")) + 1 and values[-1] == "":
                values.pop()
            require(len(values) == len(SACCT_FIELDS.split(",")), "INVALID_ACCOUNTING_ROW")
            rows.append(dict(zip(SACCT_FIELDS.split(","), values, strict=True)))
        require(all(len(row) == 2 and row[0] == str(job_id) for row in queued), "QUEUE_ID_CHANGED")
        require(
            all(row["JobIDRaw"] == str(job_id) for row in rows) and len(rows) <= 1,
            "ACCOUNTING_ID_CHANGED",
        )
        return dict(
            job_id=str(job_id),
            queue=queued,
            accounting=rows,
            receipts={"queue": queue, "accounting": accounting},
        )

    def detail(self, job_id):
        try:
            result = self.run(["scontrol", "show", "job", str(job_id), "--oneliner"])
        except (OSError, subprocess.SubprocessError) as exc:
            raise ObservationUnknown("CPU_CONTROLLER_QUERY_UNAVAILABLE") from exc
        if result["returncode"] != 0:
            raise ObservationUnknown("CPU_CONTROLLER_IDENTITY_UNKNOWN")
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result["stdout"]))
        return fields, result


def verify_cpu(fields, authorization, engine_job_id):
    expected = authorization["old_controller_identity"]
    job = authorization["old_controller_job_id"]
    require(str(job) != str(engine_job_id), "REFUSE_ENGINE_CANCELLATION")
    require(fields.get("JobId") == str(job), "CPU_CONTROLLER_ID_CHANGED")
    require(
        fields.get("JobState") in {"RUNNING", "PENDING", "COMPLETING", *CPU_TERMINAL},
        "CPU_STATE_UNKNOWN",
    )
    for key in ("JobName", "UserId", "Account", "QOS", "Command", "Comment", "StdOut"):
        observed = fields.get(key)
        if key == "Comment" and observed in {None, "(null)"}:
            observed = ""
        if key == "UserId" and observed is not None:
            observed = observed.split("(")[0]
        require(observed == expected[key], "CPU_CONTROLLER_IDENTITY_CHANGED:" + key)
    require(
        bool(fields.get("ReqTRES")) and gpu_count(fields["ReqTRES"]) == 0,
        "REFUSE_GPU_CONTROLLER_CANCELLATION",
    )
    if fields.get("JobState") in {"RUNNING", "COMPLETING"}:
        require(bool(fields.get("AllocTRES")), "CPU_ALLOCATION_UNKNOWN")
    require(gpu_count(fields.get("AllocTRES", "")) == 0, "REFUSE_GPU_CONTROLLER_CANCELLATION")


def cpu_terminal(snapshot, authorization):
    if snapshot["queue"] or len(snapshot["accounting"]) != 1:
        return False
    row = snapshot["accounting"][0]
    expected = authorization["old_controller_identity"]
    for key, reference in (
        ("JobIDRaw", authorization["old_controller_job_id"]),
        ("JobName", expected["JobName"]),
        ("User", expected["UserId"]),
        ("Account", expected["Account"]),
        ("QOS", expected["QOS"]),
    ):
        require(row[key] == str(reference), "TERMINAL_CPU_IDENTITY_CHANGED:" + key)
    require(
        bool(row["AllocTRES"]) and gpu_count(row["AllocTRES"]) == 0,
        "TERMINAL_CONTROLLER_GPU_IDENTITY",
    )
    return row["State"].split()[0].rstrip("+") in CPU_TERMINAL


def stop_cpu(root, authorization, engine_job_id, *, slurm, sleep=time.sleep, poll_seconds=15):
    """Caller holds both scheduler locks until this returns; no GPU cancellation API."""
    incident = root / INCIDENT
    intent_path = incident / "CPU_CANCEL_INTENT.json"
    terminal_path = incident / "CPU_TERMINAL.json"
    while True:
        try:
            snapshot = slurm.snapshot(authorization["old_controller_job_id"])
        except ObservationUnknown as exc:
            print(
                json.dumps({"time": now(), "status": "CPU_STATE_UNKNOWN_WAIT", "error": str(exc)}),
                flush=True,
            )
            sleep(poll_seconds)
            continue
        if cpu_terminal(snapshot, authorization):
            if not terminal_path.exists():
                save(terminal_path, {"time": now(), "snapshot": snapshot})
            return snapshot
        if not intent_path.exists():
            if not snapshot["queue"]:
                sleep(poll_seconds)
                continue
            try:
                fields, detail = slurm.detail(authorization["old_controller_job_id"])
            except ObservationUnknown:
                sleep(poll_seconds)
                continue
            verify_cpu(fields, authorization, engine_job_id)
            if fields["JobState"] == "COMPLETING" or fields["JobState"] in CPU_TERMINAL:
                sleep(poll_seconds)
                continue
            cancel = ["scancel", str(authorization["old_controller_job_id"])]
            intent = dict(
                time=now(),
                command=cancel,
                cpu_identity=fields,
                before=snapshot,
                detail=detail,
                authorization_sha256=sha(incident / "HANDOFF_AUTHORIZATION.json"),
            )
            save(intent_path, intent)
            # An ambiguous response consumes this intent. Restarts only observe.
            try:
                result = slurm.run(cancel)
            except Exception as exc:
                result = {"returncode": None, "error": repr(exc), "time": now()}
            save(incident / "CPU_CANCEL_RESULT.json", result)
        else:
            intent = read(intent_path)
            require(
                intent["command"] == ["scancel", str(authorization["old_controller_job_id"])]
                and intent["authorization_sha256"] == sha(incident / "HANDOFF_AUTHORIZATION.json"),
                "CANCEL_INTENT_CHANGED",
            )
            verify_cpu(intent["cpu_identity"], authorization, engine_job_id)
        sleep(poll_seconds)


def prebaseline(root, state):
    require(not (root / "STOP").exists(), "STOP_REQUESTED")
    require(state.get("test_sealed") is True, "TEST_ALREADY_RELEASED")
    require(state["tasks"]["COMMON_START"]["status"] == "COMPLETE", "COMMON_START_NOT_COMPLETE")
    for key, task in state["tasks"].items():
        if key not in {"COMMON_START", "ENGINE"}:
            require(
                task.get("status") == "WAITING" and not task.get("attempts"),
                "DOWNSTREAM_ALREADY_STARTED:" + key,
            )
    raw = root / "raw/evaluation"
    require(
        not raw.exists() or not any(path.is_file() for path in raw.rglob("*")),
        "BASELINE_RAW_ALREADY_EXISTS",
    )


def observe_engine(root, scheduler, verify_receipt, slurm):
    """Load/observe one existing task only. Never call tick or _submit."""
    scheduler._load()
    prebaseline(root, scheduler.state)
    task = scheduler.state["tasks"]["ENGINE"]
    if task["status"] in {"SUBMITTING", "REGISTERED", "ACTIVE", "UNKNOWN"}:
        scheduler._observe("ENGINE")
        task = scheduler.state["tasks"]["ENGINE"]
    if task["status"] != "COMPLETE":
        require(
            task["status"] in {"SUBMITTING", "REGISTERED", "ACTIVE", "UNKNOWN"},
            "ENGINE_REQUIRES_TECHNICAL_REPAIR:" + task["status"],
        )
        return None
    require(task.get("attempts"), "ENGINE_HAS_NO_ATTEMPT")
    attempt = task["attempts"][-1]
    snapshot = slurm.snapshot(attempt["job_id"])
    if snapshot["queue"] or len(snapshot["accounting"]) != 1:
        return None
    row = snapshot["accounting"][0]
    account = attempt.get("accounting", {})
    require(
        row["State"] == "COMPLETED"
        and row["ExitCode"] == "0:0"
        and account.get("terminal_state") == "COMPLETED"
        and account.get("exit_code") == "0:0",
        "ENGINE_TERMINAL_NOT_SUCCESSFUL",
    )
    # Marker verification also checks complete artifact hashes and worker identity.
    require(scheduler._marker_valid("ENGINE", attempt), "ENGINE_COMPLETION_MARKER_INVALID")
    verify_receipt(root)
    prebaseline(root, scheduler.state)
    return snapshot


def source_inventory(folder):
    require(folder.is_dir() and not folder.is_symlink(), "SOURCE_DIRECTORY_INVALID")
    deployment = read(folder / "SOURCE_DEPLOYMENT.json")
    require(
        re.fullmatch(r"[0-9a-f]{40}", deployment.get("source_commit", "")), "SOURCE_COMMIT_UNKNOWN"
    )
    require(deployment.get("source_dirty_files") == [], "UNCOMMITTED_SOURCE")
    for name, expected in deployment["source_file_hashes"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "UNSAFE_SOURCE_PATH")
        path = folder / relative
        require(
            not any(
                parent.is_symlink()
                for parent in (path, *path.parents)
                if parent.is_relative_to(folder)
            ),
            "SOURCE_SYMLINK",
        )
        require(sha(path) == expected, "SOURCE_BYTES_CHANGED:" + name)
    actual_names = set()
    for relative in ("src/sr_f1", "scripts/sr_f1", "src/mm_core", "src/mm_dev"):
        actual_names.update(
            str(path.relative_to(folder))
            for path in (folder / relative).rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}
        )
    actual_names.update(name for name in ("pyproject.toml", "uv.lock") if (folder / name).is_file())
    actual_names.update(
        str(path.relative_to(folder))
        for path in (folder / "docs/sr_f1/amendments").rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    )
    require(actual_names == set(deployment["source_file_hashes"]), "SOURCE_INVENTORY_CHANGED")
    encoded = json.dumps(
        deployment["source_file_hashes"],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    require(
        hashlib.sha256(encoded).hexdigest() == deployment["source_tree_sha256"],
        "SOURCE_TREE_HASH_CHANGED",
    )
    return deployment


def authorization_for(root, candidate, old_controller):
    incident = root / INCIDENT
    auth = read(incident / "HANDOFF_AUTHORIZATION.json")
    require(
        auth.get("status") == "AUTHORIZED_BASELINE_PARALLEL_HANDOFF"
        and auth.get("run_root") == str(root)
        and auth.get("old_controller_job_id") == str(old_controller),
        "HANDOFF_AUTHORIZATION_CHANGED",
    )
    require(
        auth["original_execution_freeze_sha256"] == sha(root / "EXECUTION_FREEZE.json")
        and auth["previous_engine_multigpu_repair_sha256"]
        == sha(root / "ENGINE_MULTIGPU_REPAIR.json"),
        "HANDOFF_FROZEN_IDENTITY_CHANGED",
    )
    permission = read(root / "manifests/ALLOCATION_PERMISSION.json")
    expected = auth["old_controller_identity"]
    require(
        expected["UserId"] == permission["owner"]
        and expected["Account"] == permission["account"]
        and expected["JobName"] == "srf11-controller-multigpu"
        and expected["Command"] == str(root / "code/scripts/sr_f1/controller.sbatch")
        and expected["StdOut"]
        == str(
            root
            / "technical_incidents/multigpu_20261010"
            / ("controller_" + str(old_controller) + ".log")
        ),
        "HANDOFF_TARGET_NOT_OWN_PROJECT_CONTROLLER",
    )
    require(candidate == root / CANDIDATE and root.resolve() == root, "HANDOFF_PATH_CHANGED")
    validation = read(incident / "CANDIDATE_VALIDATION.json")
    require(
        sha(incident / "CANDIDATE_VALIDATION.json") == auth["candidate_validation_sha256"]
        and validation.get("status") == "PASS",
        "CANDIDATE_NOT_VALIDATED",
    )
    actual_path = candidate if candidate.exists() else root / "code"
    deployment = source_inventory(actual_path)
    require(
        sha(actual_path / "SOURCE_DEPLOYMENT.json") == auth["candidate_source_deployment_sha256"]
        and all(
            validation.get(key) == deployment[key]
            for key in ("source_commit", "source_tree_sha256")
        ),
        "VALIDATED_CANDIDATE_CHANGED",
    )
    return auth


def preserve_activation(root, authorization, scheduler, engine_snapshot, cpu_snapshot):
    folder = root / INCIDENT / "activation"
    for name in EVIDENCE:
        source, target = root / name, folder / "evidence" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(source) == sha(target), "ACTIVATION_EVIDENCE_CHANGED:" + name)
        else:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(source.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
                sync_folder(target.parent)
            finally:
                temporary.unlink(missing_ok=True)
    cpu = cpu_snapshot["accounting"][0]
    engine = engine_snapshot["accounting"][0]
    terminal = dict(
        controller_job_id=authorization["old_controller_job_id"],
        controller_terminal=True,
        controller_terminal_state=cpu["State"].split()[0].rstrip("+"),
        controller_queue_empty=True,
        engine_job_id=engine["JobIDRaw"],
        engine_terminal=True,
        engine_terminal_state="COMPLETED",
        engine_exit_code="0:0",
        engine_queue_empty=True,
        controller_snapshot=cpu_snapshot,
        engine_snapshot=engine_snapshot,
    )
    terminal_path = folder / "TERMINAL_JOBS.json"
    if terminal_path.exists():
        saved = read(terminal_path)
        require(
            all(
                saved[key] == value
                for key, value in terminal.items()
                if key not in {"controller_snapshot", "engine_snapshot"}
            ),
            "TERMINAL_ACTIVATION_IDENTITY_CHANGED",
        )
    else:
        save(terminal_path, terminal)
    before, after = source_inventory(root / "code"), source_inventory(root / CANDIDATE)
    require(
        before["source_commit"] == read(root / "ENGINE_MULTIGPU_REPAIR.json")["source_commit"],
        "OLD_ENGINE_SOURCE_CHANGED",
    )
    intent = dict(
        status="ENGINE_PASSED_SOURCE_ACTIVATION_INTENT",
        time=now(),
        run_root=str(root),
        authorization_sha256=sha(root / INCIDENT / "HANDOFF_AUTHORIZATION.json"),
        original_execution_freeze_sha256=sha(root / "EXECUTION_FREEZE.json"),
        before=before,
        after=after,
        historical_artifact_hashes={
            str(path.relative_to(root)): sha(path)
            for path in [folder / "evidence" / name for name in EVIDENCE]
            + [folder / "TERMINAL_JOBS.json"]
        },
    )
    save(folder / "ACTIVATION_INTENT.json", intent)
    return intent


def recover_moves(root, intent, *, rename=None):
    """Three permitted layouts; every transition checks both source inventories."""
    rename = rename or (lambda source, target: source.rename(target))
    code, candidate, preserved = root / "code", root / CANDIDATE, root / PRESERVED
    require(
        intent["run_root"] == str(root)
        and intent["original_execution_freeze_sha256"] == sha(root / "EXECUTION_FREEZE.json"),
        "ACTIVATION_INTENT_ROOT_CHANGED",
    )
    require(
        intent["authorization_sha256"] == sha(root / INCIDENT / "HANDOFF_AUTHORIZATION.json"),
        "ACTIVATION_AUTHORIZATION_CHANGED",
    )
    for relative, expected in intent["historical_artifact_hashes"].items():
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "UNSAFE_ACTIVATION_EVIDENCE",
        )
        require(sha(root / relative) == expected, "ACTIVATION_EVIDENCE_CHANGED")
    if not preserved.exists():
        require(code.exists() and candidate.exists(), "UNKNOWN_PRE_ACTIVATION_LAYOUT")
        require(
            source_inventory(code) == intent["before"]
            and source_inventory(candidate) == intent["after"],
            "ACTIVATION_SOURCE_CHANGED",
        )
        rename(code, preserved)
        sync_folder(root)
    require(source_inventory(preserved) == intent["before"], "PRESERVED_SOURCE_CHANGED")
    if not code.exists():
        require(
            candidate.exists() and source_inventory(candidate) == intent["after"],
            "CANDIDATE_MISSING_DURING_ACTIVATION",
        )
        rename(candidate, code)
        sync_folder(root)
    require(
        not candidate.exists() and source_inventory(code) == intent["after"],
        "UNKNOWN_POST_ACTIVATION_LAYOUT",
    )


def candidate_python(root, python, program, arguments=(), *, runner=command):
    code = root / "code"
    environment = {**os.environ, "PYTHONPATH": str(code / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = runner(
        [str(python), "-c", program, *map(str, arguments)],
        timeout=1800,
        environment=environment,
        cwd=str(code),
    )
    require(
        result["returncode"] == 0, "ACTIVATION_PREFLIGHT_FAILED:" + result.get("stderr", "")[-2000:]
    )
    return result


def activate(root, plan, python, authorization, *, runner=command):
    folder = root / INCIDENT / "activation"
    intent = read(folder / "ACTIVATION_INTENT.json")
    recover_moves(root, intent)
    if not (root / REPAIR).exists():
        built = candidate_python(
            root,
            python,
            "import json,sys; from sr_f1.freeze import build_baseline_parallel_repair; "
            "a=json.load(open(sys.argv[2])); "
            "s=json.load(open(sys.argv[3])); "
            "print(json.dumps(build_baseline_parallel_repair(sys.argv[1],actual_source=s,"
            "authorized_at=a['authorized_at'],authorized_user_message=a['authorized_user_message'])))",
            (
                root,
                root / INCIDENT / "HANDOFF_AUTHORIZATION.json",
                root / "code/SOURCE_DEPLOYMENT.json",
            ),
            runner=runner,
        )
        save(root / REPAIR, json.loads(built["stdout"]))
    active_path = folder / "SOURCE_ACTIVATED.json"
    # New imports in a fresh process, never stale old modules whose SOURCE_ROOT
    # now denotes the promoted directory. _load validates registration, not tick.
    result = candidate_python(
        root,
        python,
        "import json,sys; "
        "from sr_f1.freeze import verify_execution,verify_baseline_parallel_repair; "
        "from sr_f1.orchestration import Scheduler; "
        "root,plan,py=sys.argv[1:]; r=verify_baseline_parallel_repair(root); "
        "verify_execution(plan,root,require_engine=True); "
        "s=Scheduler(plan,root,code_root=root+'/code',python=py); s._load(); "
        "print(json.dumps({'status':'PASS','repair':r,'test_sealed':s.state['test_sealed']}))",
        (root, plan, python),
        runner=runner,
    )
    checked = json.loads(result["stdout"])
    require(checked["status"] == "PASS", "ACTIVATION_NOT_VERIFIED")
    if not active_path.exists():
        preflight_path = folder / "PREFLIGHT.json"
        if preflight_path.exists():
            prior = read(preflight_path)
            require(
                prior.get("returncode") == 0 and json.loads(prior["stdout"]) == checked,
                "PRIOR_PREFLIGHT_IDENTITY_CHANGED",
            )
        else:
            save(preflight_path, result)
        save(
            active_path,
            dict(
                status="VERIFIED_BASELINE_PARALLEL_SOURCE_ACTIVE",
                intent_sha256=sha(folder / "ACTIVATION_INTENT.json"),
                repair_sha256=sha(root / REPAIR),
                source=source_inventory(root / "code"),
                preflight_sha256=sha(folder / "PREFLIGHT.json"),
                time=now(),
            ),
        )
    else:
        active = read(active_path)
        require(
            active["intent_sha256"] == sha(folder / "ACTIVATION_INTENT.json")
            and active["repair_sha256"] == sha(root / REPAIR)
            and active["source"] == source_inventory(root / "code")
            and active["preflight_sha256"] == sha(folder / "PREFLIGHT.json"),
            "ACTIVATED_SOURCE_RECEIPT_CHANGED",
        )
    return checked


def old_runtime(root, plan, python):
    sys.path.insert(0, str(root / "code/src"))
    from sr_f1.freeze import verify_engine_receipt
    from sr_f1.orchestration import Scheduler, SlurmBackend

    class ObservationOnlyBackend(SlurmBackend):
        def _run(self, arguments):
            require(
                arguments[0] in {"squeue", "sacct"} or arguments[:3] == ["scontrol", "show", "job"],
                "WATCHER_REFUSES_SLURM_MUTATION",
            )
            return super()._run(arguments)

    scheduler = Scheduler(
        plan, root, code_root=root / "code", python=python, backend=ObservationOnlyBackend()
    )
    return scheduler, verify_engine_receipt


def controller_command(root, plan, python):
    return [
        str(python),
        str(root / "code/scripts/sr_f1/submit_matrix.py"),
        "--plan",
        str(plan),
        "--run-root",
        str(root),
        "--code-root",
        str(root / "code"),
        "--python",
        str(python),
        "--lease-minutes",
        "4320",
        "--engine-lease-minutes",
        "4320",
        "--watch",
        "--controller-requeue-on-lease",
        "--poll-seconds",
        "30",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--old-controller-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    root, plan, python = args.run_root, args.plan, args.python
    require(
        root == root.resolve()
        and plan.resolve().is_relative_to(root)
        and 5 <= args.poll_seconds <= 60,
        "INVALID_HANDOFF_ARGUMENTS",
    )
    require(
        str(os.environ.get("SLURM_JOB_ID", "")).isdigit()
        and os.environ["SLURM_JOB_ID"] != args.old_controller_id,
        "WATCHER_REQUIRES_SEPARATE_CPU_JOB",
    )
    authorization = authorization_for(root, args.candidate_root, args.old_controller_id)
    incident = root / INCIDENT
    slurm = Slurm(owner=authorization["old_controller_identity"]["UserId"])
    with lease(incident / "handoff_process.lock", blocking=False):
        intent_path = incident / "activation/ACTIVATION_INTENT.json"
        if not intent_path.exists():
            scheduler, verify_receipt = old_runtime(root, plan, python)
            from sr_f1.orchestration import controller_incarnation, request_controller_requeue

            identity = controller_incarnation(root)
            # Hold scheduler locks before terminating only the authenticated CPU.
            # A TERM requeue handler cannot tick/submit while these locks are held.
            with scheduler_locks(root):
                scheduler._load()
                prebaseline(root, scheduler.state)
                engine_job = scheduler.state["tasks"]["ENGINE"]["attempts"][-1]["job_id"]
                cpu_snapshot = stop_cpu(root, authorization, engine_job, slurm=slurm)
            with lease(root / "orchestration/controller_process.lock", blocking=False):
                boundary = {"requested": False}

                def mark_boundary(signum, frame):
                    boundary["requested"] = True

                for signum in (signal.SIGTERM, signal.SIGUSR1):
                    signal.signal(signum, mark_boundary)
                while True:
                    with scheduler_locks(root):
                        try:
                            snapshot = observe_engine(root, scheduler, verify_receipt, slurm)
                        except ObservationUnknown as exc:
                            snapshot = None
                            print(
                                json.dumps(
                                    {
                                        "time": now(),
                                        "status": "ENGINE_SLURM_UNKNOWN_WAIT",
                                        "error": str(exc),
                                    }
                                ),
                                flush=True,
                            )
                        if snapshot is not None:
                            preserve_activation(
                                root, authorization, scheduler, snapshot, cpu_snapshot
                            )
                            activate(root, plan, python, authorization)
                            break
                    print(
                        json.dumps(
                            {
                                "time": now(),
                                "status": "WAITING_FOR_ENGINE_ACCEPTANCE",
                                "engine_status": scheduler.state["tasks"]["ENGINE"]["status"],
                            }
                        ),
                        flush=True,
                    )
                    if boundary["requested"]:
                        result = request_controller_requeue(root, identity)
                        print(json.dumps(result), flush=True)
                        return 0 if result["status"] == "REQUEST_ACCEPTED" else 3
                    time.sleep(args.poll_seconds)
        else:
            # Recovery after either rename or after verified activation. Never
            # import/verify old runtime against the newly promoted source.
            with (
                lease(root / "orchestration/controller_process.lock", blocking=False),
                scheduler_locks(root),
            ):
                activate(root, plan, python, authorization)
        arguments = controller_command(root, plan, python)
        if not (incident / "CONTROLLER_EXEC_INTENT.json").exists():
            save(
                incident / "CONTROLLER_EXEC_INTENT.json",
                dict(
                    time=now(),
                    command=arguments,
                    slurm_job_id=os.environ["SLURM_JOB_ID"],
                    activation_sha256=sha(incident / "activation/SOURCE_ACTIVATED.json"),
                ),
            )
        os.environ["PYTHONPATH"] = str(root / "code/src")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execv(str(python), arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
