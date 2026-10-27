"""Fixed A/B/EVAL two-worker PILOT queues; never a scientific gate or Slurm submitter.

A runs the registered b0/b1 queues. B runs b2 alongside the unique recovery.
EVAL starts only after all nine primaries and verified exact recovery. Each model
task uses a fresh child process and its original production API/locks. A failed
lane stops; the other independent worker is never killed by this module.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from . import evaluation, recovery
from .admission import require_pilot_admission
from .common import now, write_once
from .registration import DIRECTORY as REGISTRATION_DIRECTORY
from .transactions import RunTransactions, _plain_path, _read_bytes, _require

DIRECTORY = "PHASE_RUNS/PILOT_V1/JOBS"
PLAN = DIRECTORY + "/PLAN.json"
SCHEMA = "srf6-fixed-pilot-jobs-v1"
ROUNDS = ("A", "B", "EVAL")


def _read(path):
    return recovery._json(_read_bytes(path))


def _ref(root, path):
    return dict(
        path=_plain_path(path).relative_to(root).as_posix(),
        sha256=hashlib.sha256(_read_bytes(path)).hexdigest(),
    )


def _write(path, value):
    path = _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, value)


def _safe_id(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value),
        "Safe unique attempt ID of at most64 characters required",
    )
    return value


def _admitted(root, pins):
    admitted = require_pilot_admission(root, **pins)
    _require(
        admitted["status"] == "VERIFIED_PILOT_ADMISSION"
        and admitted["scope"] == "PILOT_V1_REGISTERED_ONLY",
        "Actual PILOT admission required",
    )
    return admitted


def _plan(admitted, pins):
    spec = admitted["pilot_spec"]
    tasks, queues = [], {r: [[], []] for r in ROUNDS}
    for item in spec["paths"]:
        parent = item["parent"]
        round_id, worker = ("A", int(parent == "b1")) if parent != "b2" else ("B", 0)
        task_id = f"PRIMARY-{parent}-{item['arm']}"
        queues[round_id][worker].append(task_id)
        tasks.append(
            dict(
                task_id=task_id,
                kind="PRIMARY",
                parent=parent,
                arm=item["arm"],
                round=round_id,
                worker=worker,
                registered_order=item["order"],
            )
        )
    _require(
        len(tasks) == 9
        and [len(queues["A"][i]) for i in (0, 1)] == [3, 3]
        and len(queues["B"][0]) == 3,
        "Exactly the frozen three parent queues required",
    )
    queues["B"][1] = ["RECOVERY-b0-U11"]
    tasks.append(
        dict(
            task_id="RECOVERY-b0-U11", kind="RECOVERY", parent="b0", arm="U11", round="B", worker=1
        )
    )
    for index, endpoint in enumerate(spec["evaluation"]["endpoints"]):
        task_id = "EVAL-" + endpoint["endpoint_id"].replace("/", "-")
        queues["EVAL"][index % 2].append(task_id)
        tasks.append(
            dict(
                task_id=task_id,
                kind="EVALUATION",
                endpoint_id=endpoint["endpoint_id"],
                round="EVAL",
                worker=index % 2,
                registered_order=index + 1,
            )
        )
    _require(
        len(queues["EVAL"][0]) == len(queues["EVAL"][1]) == 6,
        "Exactly12 registered endpoints required",
    )
    return dict(
        schema=SCHEMA,
        status="FROZEN_OPERATIONS_NOT_AUTHORIZED",
        phase="PILOT_V1",
        pins=copy.deepcopy(pins),
        phase_identity=spec["phase_identity"],
        source_manifest=spec["actual_source_manifest"],
        queues=queues,
        tasks=tasks,
        round_dependencies={"A": [], "B": ["A"], "EVAL": ["A", "B"]},
        budgets=copy.deepcopy(spec["budgets"]),
        workers=2,
        gpus_per_worker=1,
        cpus_per_worker=4,
        memory_gib_per_worker=33,
        peer_verification_wait_seconds=3600,
        job_prefix="LH-SRF6-",
        failure_policy="Stop failed lane; retain all attempts; never auto-retry or kill peer",
        recovery_requirement="Actual continuous b0/U11 baseline and exact independent recovery",
        final_status="WORKLOAD_COMPLETE_NOT_GATE_SIGNED",
        gate_status="NOT_SIGNED",
    )


def build_plan(root, *, phase_identity_sha256, pilot_spec_sha256, admission_ref):
    root = _plain_path(root)
    pins = dict(
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
    )
    return _plan(_admitted(root, pins), pins)


def freeze_plan(root, **pins):
    root = _plain_path(root)
    plan = build_plan(root, **pins)
    if not (root / PLAN).exists():
        admitted = _admitted(root, pins)
        for item in admitted["pilot_spec"]["paths"]:
            directory = root / item["audit_root"]
            _require(
                not any(
                    (directory / name).exists()
                    for name in ("checkpoints", "TRAINER_ATTEMPTS", "EXECUTIONS")
                ),
                "Freeze operations before first pilot execution; never retrofit task order",
            )
        _require(
            not (root / recovery.DIRECTORY).exists() and not (root / evaluation.DIRECTORY).exists(),
            "Freeze operations before recovery or evaluation output",
        )
    _write(root / PLAN, plan)
    return _ref(root, root / PLAN)


def verify_plan(root, plan_ref):
    root = _plain_path(root)
    _require(plan_ref.get("path") == PLAN, "Only fixed PILOT jobs plan is allowed")
    plan = recovery._bound(root, plan_ref)
    admitted = _admitted(root, plan["pins"])
    _require(
        plan == _plan(admitted, plan["pins"]), "Operations plan differs from frozen registration"
    )
    return plan, admitted


@contextmanager
def _lock(path, *, wait=False):
    path = _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        yield
    finally:
        os.close(fd)


def _registered(root, admitted, task):
    matches = [
        p
        for p in admitted["pilot_spec"]["paths"]
        if p["parent"] == task["parent"] and p["arm"] == task["arm"]
    ]
    _require(len(matches) == 1, "Task is outside registered primary matrix")
    item = matches[0]
    identity = recovery._bound(root / REGISTRATION_DIRECTORY, item["transaction_identity"])
    return _plain_path(root / item["audit_root"]), identity


def _training_done(root, admitted, task, report, *, recovered=None):
    directory, identity = _registered(root, admitted, task)
    lane = "PRIMARY"
    if recovered is not None:
        directory, lane = Path(recovered["directory"]), "RECOVERY"
        _require(
            recovered["identity"] == identity and recovered["scope"] == "SERVER_ACTUAL",
            "Actual recovery binding required",
        )
    with RunTransactions(directory, identity) as tx:
        markers = tx.published()
        _require(
            len(markers) == 13 and markers[-1]["step"] == 12,
            "Complete actual committed step0..12 required",
        )
        _require(
            report.get("status") == "COMPLETED_NOT_GATE_SIGNED"
            and report.get("execution_origin") == "SERVER_ACTUAL"
            and report.get("storage_lane") == lane
            and report.get("run_id") == identity["run_id"]
            and report.get("committed_step") == 12
            and report.get("final_policy_sha256") == markers[-1]["policy_sha256"],
            "Producer completion differs from committed state",
        )
        final = _ref(root, directory / "checkpoints/step-012/COMMIT.json")
        _require(report["final_commit"] == final, "Actual terminal commit reference differs")
        executed = recovery._execution(root, report["execution"], directory, identity)
        _require(
            executed["end"]["scope"] == "SERVER_ACTUAL",
            "Synthetic execution cannot complete production task",
        )
    return report


def _result_path(root, task):
    return root / DIRECTORY / "TASKS" / task["task_id"] / "COMPLETE.json"


def _verify_task(root, plan_ref, admitted, task, *, receipt=None):
    """Scheduling completion only; the final scientific consumer must audit all facts."""
    path = _result_path(root, task)
    if receipt is None:
        if not path.exists():
            return None
        receipt = _read(path)
    _require(
        receipt.get("schema") == SCHEMA
        and receipt.get("status") == "TASK_COMPLETE_NOT_GATE_SIGNED"
        and receipt.get("task") == task
        and receipt.get("plan_ref") == plan_ref,
        "Task completion identity changed",
    )
    producer = recovery._bound(root, receipt["producer"])
    if task["kind"] == "PRIMARY":
        _training_done(root, admitted, task, producer)
    elif task["kind"] == "RECOVERY":
        bound = recovery.verify_recovery(
            root,
            phase_context=admitted["phase_context"],
            pilot_spec_sha256=admitted["pilot_spec_ref"]["sha256"],
            binding_ref=producer["binding_ref"],
        )
        _training_done(
            root, admitted, task, recovery._bound(root, producer["training_ref"]), recovered=bound
        )
        comparison = recovery._bound(root, producer["comparison_ref"])
        _require(
            comparison.get("status") == "PASS"
            and comparison.get("scope") == "SERVER_ACTUAL"
            and comparison.get("exact") is True
            and comparison.get("independent_process_verified") is True
            and comparison.get("binding_ref") == producer["binding_ref"],
            "Recovery comparison is incomplete or failed; expansion is stopped",
        )
    else:
        spec, plan = admitted["pilot_spec"], admitted["pilot_spec"]["evaluation"]
        endpoint = evaluation._endpoint(plan, task["endpoint_id"])
        public, tasks = evaluation._inputs(root, plan)
        with evaluation._source(root, spec, endpoint) as source:
            identity = evaluation._identity(
                root,
                admitted,
                endpoint,
                source,
                admitted["pilot_spec_ref"]["sha256"],
                admitted["admission_ref"],
            )
            with evaluation.EvaluationLedger(root, identity, plan, public, tasks) as ledger:
                groups = ledger.completed()
                _require(
                    groups is not None and producer == evaluation._endpoint_report(ledger, groups),
                    "Evaluation completion differs from original raw evidence",
                )
    return receipt


def _completed_attempt(root, plan_ref, admitted, task):
    """Recover only derived operations receipts; never redo a producer's work."""
    history = _result_path(root, task).parent / "ATTEMPTS"
    if not history.exists():
        return None
    for attempt in sorted(history.iterdir()):
        attempt = _plain_path(attempt)
        start = _read(attempt / "STARTED.json")
        _require(
            start.get("task") == task and start.get("plan_ref") == plan_ref,
            "Prior attempt belongs to another task/plan",
        )
        if task["kind"] == "PRIMARY":
            directory, _ = _registered(root, admitted, task)
            producer = directory / "TRAINER_ATTEMPTS" / start["execution_id"] / "COMPLETE.json"
        elif task["kind"] == "RECOVERY":
            producer = attempt / "RECOVERY_RESULT.json"
            trained = (
                root
                / recovery.DIRECTORY
                / "TRAINER_ATTEMPTS"
                / start["execution_id"]
                / "COMPLETE.json"
            )
            if not producer.exists() and trained.exists():
                _finish_recovery(
                    root,
                    admitted,
                    task,
                    _ref(root, trained),
                    _ref(root, root / recovery.BINDING),
                    attempt,
                )
        else:
            producer = root / evaluation.DIRECTORY / task["endpoint_id"] / "COMPLETE.json"
        if not producer.exists():
            continue
        receipt = dict(
            schema=SCHEMA,
            status="TASK_COMPLETE_NOT_GATE_SIGNED",
            task=task,
            plan_ref=plan_ref,
            producer=_ref(root, producer),
            gate_status="NOT_SIGNED",
        )
        _verify_task(root, plan_ref, admitted, task, receipt=receipt)
        _write(_result_path(root, task), receipt)
        _write(attempt / "COMPLETE.json", receipt)
        return receipt
    return None


def _required_tasks(plan, round_id, worker):
    completed_rounds = plan["round_dependencies"][round_id]
    ids = {
        task for previous in completed_rounds for lane in plan["queues"][previous] for task in lane
    }
    return [task for task in plan["tasks"] if task["task_id"] in ids]


def _dependency_check(root, plan_ref, plan, admitted, round_id, worker):
    for task in _required_tasks(plan, round_id, worker):
        _require(
            _verify_task(root, plan_ref, admitted, task) is not None,
            "Prior fixed round is not complete: " + task["task_id"],
        )


def _comparison(root, plan_ref, admitted, plan):
    task = next(t for t in plan["tasks"] if t["kind"] == "RECOVERY")
    receipt = _verify_task(root, plan_ref, admitted, task)
    _require(receipt is not None, "Recovery task has no actual terminal evidence")
    producer = recovery._bound(root, receipt["producer"])
    binding = recovery._bound(root, producer["binding_ref"])
    directory, identity = _registered(root, admitted, task)
    with (
        RunTransactions(directory, identity) as primary,
        RunTransactions(root / recovery.DIRECTORY, identity) as restored,
    ):
        result = recovery.verify_comparison(
            primary,
            restored,
            root=root,
            phase_context=admitted["phase_context"],
            pilot_spec_sha256=admitted["pilot_spec_ref"]["sha256"],
            binding_ref=producer["binding_ref"],
            primary_execution_ref=binding["primary_execution_ref"],
            recovery_execution_ref=producer["recovery_execution_ref"],
            comparison_ref=producer["comparison_ref"],
        )
    _require(
        result["status"] == "PASS"
        and result["scope"] == "SERVER_ACTUAL"
        and result["exact"]
        and result["independent_process_verified"],
        "Actual recovery exactness failed; no evaluation expansion",
    )
    return (directory, root / recovery.DIRECTORY)


def _stamps(directories):
    """No model/payload hashes; fail if verified recovery evidence changes mid-worker."""
    result = []
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            path = _plain_path(path)
            if path.is_file():
                st = path.stat()
                result.append((str(path), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    return result


@contextmanager
def _interruptions():
    def stop(signum, _frame):
        raise InterruptedError(f"Worker interrupted by signal{signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _stop_child_group(child):
    # The process group may still have grandchildren after its leader exits.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with suppress(subprocess.TimeoutExpired):
        child.wait(timeout=20)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def _evaluation_barrier(root, plan_ref, worker, attempt_id, *, timeout):
    """Both actual cold verifications finish before either endpoint queue starts."""
    base = root / DIRECTORY / "WORKERS" / "EVAL"
    own = base / str(worker) / "ATTEMPTS" / attempt_id
    peer = base / str(1 - worker) / "ATTEMPTS" / attempt_id
    value = dict(status="PREREQUISITES_VERIFIED", plan_ref=plan_ref, attempt_id=attempt_id)
    _write(own / "PREREQUISITES.json", value)
    deadline = time.monotonic() + timeout
    while True:
        _require(
            not (peer / "FAILURE.json").exists(),
            "Peer prerequisite verification failed; evaluation stopped",
        )
        if (peer / "PREREQUISITES.json").exists():
            _require(
                _read(peer / "PREREQUISITES.json") == value, "Peer verified another operations plan"
            )
            return
        _require(
            time.monotonic() < deadline, "Peer prerequisite barrier timed out; no endpoint started"
        )
        time.sleep(0.2)


def _invoke(command, log_path):
    """Only this worker's child is stopped on interruption, never the peer worker."""
    with _interruptions(), log_path.open("xb") as stream:
        deployment = Path(__file__).resolve(strict=True).parents[2]
        environment = dict(os.environ, PYTHONPATH=str(deployment / "src"))
        child = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=deployment,
            env=environment,
        )
        try:
            code = child.wait()
        finally:
            _stop_child_group(child)
    _require(code == 0, f"Production child failed with exit code{code}; attempt preserved")


def _arguments(root, pins):
    return [
        "--phase-identity-sha256",
        pins["phase_identity_sha256"],
        "--pilot-spec-sha256",
        pins["pilot_spec_sha256"],
    ]


def _finish_recovery(root, admitted, task, trained_ref, binding_ref, attempt):
    """Finish the original recovery producer; never create a step12 no-op journal."""
    trained = recovery._bound(root, trained_ref)
    binding = recovery._bound(root, binding_ref)
    common = dict(
        root=root,
        phase_context=admitted["phase_context"],
        pilot_spec_sha256=admitted["pilot_spec_ref"]["sha256"],
        binding_ref=binding_ref,
        primary_execution_ref=binding["primary_execution_ref"],
        recovery_execution_ref=trained["execution"],
    )
    bound = recovery.verify_recovery(
        root,
        phase_context=common["phase_context"],
        pilot_spec_sha256=common["pilot_spec_sha256"],
        binding_ref=binding_ref,
    )
    _training_done(root, admitted, task, trained, recovered=bound)
    primary_directory, identity = _registered(root, admitted, task)
    directory = root / recovery.DIRECTORY
    with (
        RunTransactions(primary_directory, identity) as primary,
        RunTransactions(directory, identity) as restored,
    ):
        matches = []
        for path in sorted((directory / "COMPARISONS").glob("*.json")):
            report = _read(path)
            if all(
                report.get(k) == common[k]
                for k in ("binding_ref", "primary_execution_ref", "recovery_execution_ref")
            ):
                _require(
                    report.get("status") == "PASS",
                    "Existing recovery comparison failed; no replacement comparison",
                )
                matches.append(_ref(root, path))
        _require(
            len(matches) <= 1,
            "Multiple recovery comparisons require review; never select a preferred outcome",
        )
        comparison = (
            recovery.verify_comparison(primary, restored, **common, comparison_ref=matches[0])
            if matches
            else recovery.compare_recovery(primary, restored, **common)
        )
    combined = dict(
        training_ref=trained_ref,
        binding_ref=binding_ref,
        comparison_ref=comparison["receipt_reference"],
        recovery_execution_ref=trained["execution"],
    )
    _write(attempt / "RECOVERY_RESULT.json", combined)
    _require(
        comparison["status"] == "PASS"
        and comparison["scope"] == "SERVER_ACTUAL"
        and comparison["exact"]
        and comparison["independent_process_verified"],
        "Exact recovery failed; preserve evidence and stop expansion",
    )
    return _ref(root, attempt / "RECOVERY_RESULT.json")


def _execute(root, plan_ref, plan, admitted, task, execution_id, attempt):
    pins = plan["pins"]
    if task["kind"] == "EVALUATION":
        command = [
            sys.executable,
            "-u",
            "-m",
            "sr_f6.evaluation",
            "run",
            str(root),
            *_arguments(root, pins),
            "--admission-sha256",
            pins["admission_ref"]["sha256"],
            "--endpoint-id",
            task["endpoint_id"],
            "--task-id",
            execution_id,
        ]
        _invoke(command, attempt / "process.log")
        return _ref(root, root / evaluation.DIRECTORY / task["endpoint_id"] / "COMPLETE.json")
    directory, identity = _registered(root, admitted, task)
    binding_ref = None
    if task["kind"] == "RECOVERY":
        primary_task = next(t for t in plan["tasks"] if t["task_id"] == "PRIMARY-b0-U11")
        primary = _verify_task(root, plan_ref, admitted, primary_task)
        _require(primary is not None, "Actual b0/U11 primary completion required")
        primary_report = recovery._bound(root, primary["producer"])
        with RunTransactions(directory, identity) as tx:
            if (root / recovery.BINDING).exists():
                binding_ref = _ref(root, root / recovery.BINDING)
                recovery.verify_recovery(
                    root,
                    phase_context=admitted["phase_context"],
                    pilot_spec_sha256=pins["pilot_spec_sha256"],
                    binding_ref=binding_ref,
                )
            else:
                imported = recovery.prepare_recovery(
                    tx,
                    experiment_root=root,
                    phase_context=admitted["phase_context"],
                    pilot_spec_sha256=pins["pilot_spec_sha256"],
                    source_commit_sha256=tx.published()[4]["commit_sha256"],
                    primary_execution_ref=primary_report["execution"],
                )
                binding_ref = imported["binding_ref"]
        directory = root / recovery.DIRECTORY
    command = [
        sys.executable,
        "-u",
        "-m",
        "sr_f6.pilot",
        "--root",
        str(root),
        "--parent",
        task["parent"],
        "--arm",
        task["arm"],
        "--task-id",
        execution_id,
        *_arguments(root, pins),
        "--admission",
        pins["admission_ref"]["path"],
        pins["admission_ref"]["sha256"],
    ]
    if binding_ref is not None:
        command.extend(["--recovery-binding", binding_ref["path"], binding_ref["sha256"]])
    _invoke(command, attempt / "process.log")
    trained_ref = _ref(root, directory / "TRAINER_ATTEMPTS" / execution_id / "COMPLETE.json")
    if binding_ref is None:
        return trained_ref
    return _finish_recovery(root, admitted, task, trained_ref, binding_ref, attempt)


def _worker_environment(root):
    _require(
        bool(os.environ.get("SLURM_JOB_ID"))
        and os.environ.get("SLURM_JOB_NAME", "").startswith("LH-SRF6-"),
        "Existing LH-SRF6- Slurm allocation required; this entry never submits jobs",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(
        bool(visible) and "," not in visible and visible != "-1",
        "Worker must see exactly one assigned GPU",
    )
    config = _read(root / "CONFIG.json")
    _require(
        Path(sys.executable).resolve(strict=True)
        == Path(config["deployment"]["python"]).resolve(strict=True),
        "Use the frozen production Python environment",
    )


def run_worker(root, *, plan_ref, round_id, worker, attempt_id, resume_tasks=()):
    root, attempt_id = _plain_path(root), _safe_id(attempt_id)
    _require(
        round_id in ROUNDS and type(worker) is int and worker in (0, 1),
        "Fixed round and worker0/1 required",
    )
    plan, admitted = verify_plan(root, plan_ref)
    _worker_environment(root)
    queue = plan["queues"][round_id][worker]
    _require(set(resume_tasks) <= set(queue), "Resume may only name tasks in this fixed lane")
    folder = root / DIRECTORY / "WORKERS" / round_id / str(worker)
    with _lock(folder / ".worker.lock"), _interruptions():
        attempt = folder / "ATTEMPTS" / attempt_id
        _require(not attempt.exists(), "Worker attempt IDs are immutable")
        attempt.mkdir(parents=True)
        started = dict(
            schema=SCHEMA,
            status="STARTED",
            at=now(),
            plan_ref=plan_ref,
            round=round_id,
            worker=worker,
            attempt_id=attempt_id,
            queue=queue,
            resume_tasks=list(resume_tasks),
            visible_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            pid=os.getpid(),
        )
        _write(attempt / "STARTED.json", started)
        completed = []
        try:
            with _lock(root / DIRECTORY / ".prerequisites.lock", wait=True):
                _dependency_check(root, plan_ref, plan, admitted, round_id, worker)
                guard_dirs = (
                    _comparison(root, plan_ref, admitted, plan) if round_id == "EVAL" else ()
                )
            guard = _stamps(guard_dirs)
            if round_id == "EVAL":
                _evaluation_barrier(
                    root,
                    plan_ref,
                    worker,
                    attempt_id,
                    timeout=plan["peer_verification_wait_seconds"],
                )
            for task_id in queue:
                task = next(t for t in plan["tasks"] if t["task_id"] == task_id)
                _require(
                    _stamps(guard_dirs) == guard,
                    "Verified recovery evidence changed; stop this worker",
                )
                existing = _verify_task(root, plan_ref, admitted, task)
                if existing is not None:
                    completed.append(_ref(root, _result_path(root, task)))
                    continue
                task_dir = root / DIRECTORY / "TASKS" / task_id
                with _lock(task_dir / ".task.lock"):
                    if _completed_attempt(root, plan_ref, admitted, task) is not None:
                        completed.append(_ref(root, _result_path(root, task)))
                        continue
                    history = task_dir / "ATTEMPTS"
                    if history.exists() and any(history.iterdir()):
                        _require(
                            task_id in resume_tasks,
                            "Prior failed/unknown task requires explicit resume: " + task_id,
                        )
                        _require(
                            not any(
                                (p / "RECOVERY_RESULT.json").exists() for p in history.iterdir()
                            ),
                            "A completed recovery comparison cannot be retried to obtain PASS",
                        )
                    execution_id = f"LH-SRF6-{round_id}-{worker}-{attempt_id}-{task_id}"
                    task_attempt = history / execution_id
                    _require(not task_attempt.exists(), "Physical task attempt is immutable")
                    task_attempt.mkdir(parents=True)
                    _write(
                        task_attempt / "STARTED.json",
                        dict(started, task=task, execution_id=execution_id),
                    )
                    try:
                        producer = _execute(
                            root, plan_ref, plan, admitted, task, execution_id, task_attempt
                        )
                        receipt = dict(
                            schema=SCHEMA,
                            status="TASK_COMPLETE_NOT_GATE_SIGNED",
                            task=task,
                            plan_ref=plan_ref,
                            producer=producer,
                            gate_status="NOT_SIGNED",
                        )
                        # Never publish an operations COMPLETE before actual producer verification.
                        _verify_task(root, plan_ref, admitted, task, receipt=receipt)
                        _write(_result_path(root, task), receipt)
                        _write(task_attempt / "COMPLETE.json", receipt)
                        completed.append(_ref(root, _result_path(root, task)))
                    except BaseException as exc:
                        _write(
                            task_attempt / "FAILURE.json",
                            dict(
                                status="FAILED_PRESERVED",
                                error_type=type(exc).__name__,
                                error=str(exc),
                                at=now(),
                                task=task,
                                sampling_retry_authorized=False,
                            ),
                        )
                        raise
            result = dict(
                status="QUEUE_COMPLETE_NOT_GATE_SIGNED",
                plan_ref=plan_ref,
                round=round_id,
                worker=worker,
                completed=completed,
                gate_status="NOT_SIGNED",
            )
            _write(attempt / "COMPLETE.json", result)
            return result
        except BaseException as exc:
            _write(
                attempt / "FAILURE.json",
                dict(
                    status="FAILED_PRESERVED",
                    at=now(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    completed=completed,
                    peer_cancellation_requested=False,
                    sampling_retry_authorized=False,
                ),
            )
            raise


def aggregate(root, *, plan_ref):
    root = _plain_path(root)
    plan, admitted = verify_plan(root, plan_ref)
    refs = []
    for task in plan["tasks"]:
        _require(
            _verify_task(root, plan_ref, admitted, task) is not None,
            "Workload is incomplete: " + task["task_id"],
        )
        refs.append(_ref(root, _result_path(root, task)))
    _comparison(root, plan_ref, admitted, plan)
    evaluation.aggregate(root, **plan["pins"])
    result = dict(
        schema=SCHEMA,
        status="WORKLOAD_COMPLETE_NOT_GATE_SIGNED",
        plan_ref=plan_ref,
        inputs=[*refs, _ref(root, root / evaluation.DIRECTORY / "SUMMARY.json")],
        scientific_gate="REQUIRES_SEPARATE_COMPLETE_FACT_AUDIT",
        gate_status="NOT_SIGNED",
        formal_authorized=False,
    )
    _write(root / DIRECTORY / "SUMMARY.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "worker", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase-identity-sha256")
    parser.add_argument("--pilot-spec-sha256")
    parser.add_argument("--admission-sha256")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--round", choices=ROUNDS)
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--attempt-id")
    parser.add_argument("--resume-task", action="append", default=[])
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze_plan(
            args.root,
            phase_identity_sha256=args.phase_identity_sha256,
            pilot_spec_sha256=args.pilot_spec_sha256,
            admission_ref=dict(path="PHASES/PILOT_V1/ADMISSION.json", sha256=args.admission_sha256),
        )
    else:
        plan_ref = dict(path=PLAN, sha256=args.plan_sha256)
        result = (
            aggregate(args.root, plan_ref=plan_ref)
            if args.action == "aggregate"
            else run_worker(
                args.root,
                plan_ref=plan_ref,
                round_id=args.round,
                worker=args.worker,
                attempt_id=args.attempt_id,
                resume_tasks=args.resume_task,
            )
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
