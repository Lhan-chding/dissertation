#!/usr/bin/env python3
"""Independent CPU-only dispatcher for the already frozen nine-task ablation.

Install this helper and its test under operations/ablation_driver/COMMIT, with
a SOURCE_MANIFEST.json containing driver_commit, scientific_source_commit and
files [{path, sha256, bytes}]. Register, submit one numbered CPU lease, then run.
This tool never edits ROOT/code, retries a model task, or starts SCIENCE.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

SOURCE_COMMIT = "652e52d530520a276b6420ff0f4ee479dcde6db1"
QOS = "soujanya-poria-startfund-2026-03"
HELPER = "dispatch_protocol_ablation.py"
HELPER_TEST = "test_protocol_ablation_dispatch.py"
UNKNOWN = {"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN", "HELD"}
FINAL = {"COMPLETE", "FAILED"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def safe(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Unsafe operational artifact path")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise PermissionError("Operational artifact escaped its registered root")
    return result


def save(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read(path) != value:
                    raise PermissionError("Immutable operation evidence changed") from None
        else:
            os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another ablation dispatcher owns this run") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_sources(root, source_commit):
    if source_commit != SOURCE_COMMIT or (root / "STOP").exists():
        raise PermissionError("Only the registered, non-stopped ablation source is allowed")
    identity = read(root / "RUN_IDENTITY.json")
    manifest = read(root / "deployment/SOURCE_MANIFEST.json")
    if (
        identity.get("code_commit") != source_commit
        or manifest.get("code_commit") != source_commit
        or identity.get("run_stage") != "PROTOCOL_ABLATION"
        or Path(identity.get("root", "")).resolve() != root
        or identity.get("bound_files", {}).get("deployment/SOURCE_MANIFEST.json")
        != sha(root / "deployment/SOURCE_MANIFEST.json")
    ):
        raise PermissionError("Frozen ablation source/run identity differs")
    code = root / "code"
    rows = manifest.get("files", [])
    names = {row["path"] for row in rows}
    required = {
        "src/sr_f3_ce/ablation.py",
        "src/sr_f3_ce/protocol.py",
        "src/sr_f3_ce/orchestration.py",
        "scripts/sr_f3_ce/run_protocol_ablation.py",
    }
    if len(names) != len(rows) or not required <= names:
        raise PermissionError("Scientific source inventory is incomplete or ambiguous")
    actual = set()
    for path in code.rglob("*"):
        if path.is_symlink() or path.suffix in {".pyc", ".pyo"}:
            raise PermissionError("Unmanifested source links/bytecode cannot be imported")
        if path.is_file():
            actual.add(path.relative_to(code).as_posix())
    if actual != names:
        raise PermissionError("Scientific deployment file inventory is not complete")
    for row in rows:
        path = safe(code, row["path"])
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise PermissionError("Frozen source bytes changed before import")
    return identity


def load_api(root, source_commit):
    sys.dont_write_bytecode = True
    verify_sources(root, source_commit)
    code = root / "code"
    sys.path.insert(0, str(code / "src"))
    modules = {}
    for name in ("protocol", "ablation", "orchestration"):
        module = importlib.import_module("sr_f3_ce." + name)
        if Path(module.__file__).resolve() != code / "src/sr_f3_ce" / (name + ".py"):
            raise PermissionError("Scientific module was loaded from another deployment")
        modules[name] = module
    modules["ablation"].verify_manifest(root)
    return SimpleNamespace(**modules, slurm=modules["orchestration"].Slurm())


def operation_directory(root):
    return root / "operations/ablation_driver"


def helper_sources(root, source_commit, driver_commit):
    if not re.fullmatch(r"[a-f0-9]{40}", driver_commit):
        raise PermissionError("A committed helper identity is required")
    directory = operation_directory(root) / driver_commit
    if Path(__file__).resolve() != directory / HELPER:
        raise PermissionError("Run the explicitly deployed helper entrypoint")
    manifest = read(directory / "SOURCE_MANIFEST.json")
    if manifest.get("driver_commit") != driver_commit or (
        manifest.get("scientific_source_commit") != source_commit
    ):
        raise PermissionError("Auxiliary source commit identity differs")
    rows = manifest.get("files", [])
    if len(rows) != 2 or {row["path"] for row in rows} != {HELPER, HELPER_TEST}:
        raise PermissionError("Both helper source and its tested fixture must be bound")
    for row in rows:
        path = safe(directory, row["path"])
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise PermissionError("Auxiliary source bytes changed")
    return directory


def register(root, source_commit, driver_commit, python, *, api=None):
    root = Path(root).resolve(strict=True)
    verify_sources(root, source_commit)
    directory = helper_sources(root, source_commit, driver_commit)
    api = api or load_api(root, source_commit)
    api.ablation.verify_manifest(root)
    python = str(Path(python).absolute())
    if not Path(python).is_file():
        raise PermissionError("Explicit scientific Python executable is missing")
    value = dict(
        schema_version=1,
        status="REGISTERED_ABLATION_DISPATCH",
        root=str(root),
        source_commit=source_commit,
        driver_commit=driver_commit,
        python=python,
        user=getpass.getuser(),
        poll_seconds=60,
        run_identity_sha256=sha(root / "RUN_IDENTITY.json"),
        ablation_manifest_sha256=sha(root / "ABLATION_MANIFEST.json"),
        scientific_source_manifest_sha256=sha(root / "deployment/SOURCE_MANIFEST.json"),
        helper_source_manifest_sha256=sha(directory / "SOURCE_MANIFEST.json"),
        tasks=api.ablation.task_matrix(),
        max_gpus=5,
        max_submitted=5,
        scientific_retry_authorized=False,
        scientific_execution_authorized=False,
    )
    save(operation_directory(root) / "OPERATION_IDENTITY.json", value, immutable=True)
    return value


def verify_operation(root, source_commit, driver_commit):
    root = Path(root).resolve(strict=True)
    verify_sources(root, source_commit)
    directory = helper_sources(root, source_commit, driver_commit)
    value = read(operation_directory(root) / "OPERATION_IDENTITY.json")
    expected = dict(
        root=str(root),
        source_commit=source_commit,
        driver_commit=driver_commit,
        run_identity_sha256=sha(root / "RUN_IDENTITY.json"),
        ablation_manifest_sha256=sha(root / "ABLATION_MANIFEST.json"),
        scientific_source_manifest_sha256=sha(root / "deployment/SOURCE_MANIFEST.json"),
        helper_source_manifest_sha256=sha(directory / "SOURCE_MANIFEST.json"),
        user=getpass.getuser(),
        max_gpus=5,
        max_submitted=5,
        poll_seconds=60,
        scientific_retry_authorized=False,
        scientific_execution_authorized=False,
    )
    if any(value.get(k) != v for k, v in expected.items()):
        raise PermissionError("Registered operation identity changed")
    return value


def event(registration, value):
    root = operation_directory(Path(registration["root"]))
    previous = root / "PROGRESS.json"
    if previous.exists() and read(previous).get("event") == value:
        return
    record = dict(
        at=time.time(), event=value, operation_identity_sha256=sha(root / "OPERATION_IDENTITY.json")
    )
    with (root / "EVENTS.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    save(previous, record)


def _command(registration, *, task=None, driver_key=None, select=False):
    root = Path(registration["root"])
    if driver_key is not None:
        return [
            registration["python"],
            str(operation_directory(root) / registration["driver_commit"] / HELPER),
            "--mode",
            "run",
            "--root",
            str(root),
            "--source-commit",
            registration["source_commit"],
            "--driver-commit",
            registration["driver_commit"],
            "--driver-key",
            driver_key,
        ]
    result = [
        registration["python"],
        str(root / "code/scripts/sr_f3_ce/run_protocol_ablation.py"),
        "--mode",
        "select" if select else "run",
        "--root",
        str(root),
    ]
    if not select:
        result += ["--block", str(task["block"]), "--variant", task["variant"]]
    return result


def _body(registration, *, task=None, driver_key=None):
    root = Path(registration["root"])
    body = "#!/bin/bash\nset -euo pipefail\n"
    body += "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
    body += "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
    body += "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
    if driver_key is not None:
        body += "export CUDA_VISIBLE_DEVICES=''\n"
    body += "export PYTHONPATH=" + shlex.quote(str(root / "code/src")) + "\n"
    body += "cd " + shlex.quote(str(root / "code")) + "\n"
    body += "test ! -e " + shlex.quote(str(root / "STOP")) + "\n"
    return (
        body + "exec " + shlex.join(_command(registration, task=task, driver_key=driver_key)) + "\n"
    )


def _sbatch(intent, directory):
    args = [
        "sbatch",
        "--hold",
        "--parsable",
        "--no-requeue",
        "--open-mode=append",
        "--account=rose",
        "--partition=cluster02",
        "--qos=" + QOS,
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        f"--mem={intent['memory_gib']}G",
        "--time=4320",
    ]
    args += (
        ["--gres=gpu:pro6000:1", "--constraint=highmem"]
        if intent["gpu_count"]
        else ["--nodelist=cpu-2"]
    )
    args += [
        "--job-name=" + intent["job_name"],
        "--comment=" + intent["comment"],
        "--output=" + str(directory / "slurm-%j.log"),
    ]
    return [*args, intent["script"]]


def submissions(root):
    return root / "ablation/submissions"


def drivers(root):
    return operation_directory(root) / "drivers"


def _states(root):
    result = []
    for base in (submissions(root), drivers(root)):
        if not base.exists():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_dir():
                continue
            if path.is_symlink() or not (path / "INTENT.json").is_file():
                raise PermissionError("Unregistered submission directory")
            intent = read(path / "INTENT.json")
            state = (
                read(path / "STATE.json")
                if (path / "STATE.json").exists()
                else dict(
                    status="SUBMISSION_UNKNOWN",
                    key=intent.get("key"),
                    gpu_count=intent.get("gpu_count"),
                    qos=intent.get("qos"),
                    intent_sha256=sha(path / "INTENT.json"),
                )
            )
            result.append((path, intent, state))
    return result


def _verify_attempt(registration, api, directory, intent, state):
    root = Path(registration["root"])
    is_driver = directory.parent == drivers(root)
    key = directory.name
    task = None if is_driver else registration["tasks"].get(key)
    if not is_driver and task is None:
        raise PermissionError("Unregistered ablation task submission")
    expected = dict(
        key=key,
        gpu_count=0 if is_driver else 1,
        qos=QOS,
        partition="cluster02",
        cpus=4,
        memory_gib=12 if is_driver else 90,
        script=str(directory / "run.sbatch"),
        code_commit=registration["source_commit"],
        run_identity_sha256=registration["run_identity_sha256"],
        user=registration["user"],
    )
    if is_driver:
        if not re.fullmatch(r"driver-[0-9]{4}", key):
            raise PermissionError("Invalid CPU driver sequence")
        expected.update(
            cpu_node="cpu-2",
            job_name="SRF3CE-ABLDRIVER-N" + key[7:] + "-LH",
            operation_identity_sha256=sha(operation_directory(root) / "OPERATION_IDENTITY.json"),
            comment="srf3ce-abldriver-" + key,
        )
    else:
        expected.update(
            job_name=task["job_name"],
            task=task,
            cpu_node=None,
            comment="srf3ce-fix-" + key,
            ablation_manifest_sha256=registration["ablation_manifest_sha256"],
        )
    if any(intent.get(k) != v for k, v in expected.items()) or (
        state.get("key") != key
        or state.get("gpu_count") != expected["gpu_count"]
        or state.get("qos") != QOS
        or state.get("intent_sha256") != sha(directory / "INTENT.json")
        or state.get("status") not in FINAL | UNKNOWN | {"SUBMITTED", "ACCOUNTING_UNKNOWN"}
        or ("job_id" in state and not str(state["job_id"]).isdigit())
    ):
        raise PermissionError("Submission intent/state identity differs")
    body = _body(registration, task=task, driver_key=key if is_driver else None)
    if Path(intent["script"]).read_text() != body or sha(intent["script"]) != intent.get(
        "script_sha256"
    ):
        raise PermissionError("Submission script differs from the registered exact command")
    if intent.get("sbatch") != _sbatch(intent, directory):
        raise PermissionError("Submission resource arguments differ")
    if state["status"] not in UNKNOWN:
        reply = read(directory / "SUBMISSION.json")
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", reply.get("stdout", ""))
        allocation = read(directory / "ALLOCATION_VERIFIED.json")
        release_intent = read(directory / "RELEASE_INTENT.json")
        release = read(directory / "RELEASE.json")
        if (
            reply.get("returncode") != 0
            or not match
            or match[1] != state.get("job_id")
            or allocation.get("job_id") != state["job_id"]
            or allocation.get("script_sha256") != intent["script_sha256"]
            or release_intent.get("job_id") != state["job_id"]
            or release.get("returncode") != 0
            or release.get("job_id", state["job_id"]) != state["job_id"]
        ):
            raise PermissionError("Held/submission/release proof is incomplete or differs")
        api.orchestration.verify_allocation(
            allocation["fields"],
            dict(intent, job_id=state["job_id"]),
            registration["user"],
            held=True,
        )
    return is_driver


def _record_state(directory, state, new):
    if state == new:
        return
    save(directory / "state_history" / (digest(state) + ".json"), state, immutable=True)
    save(directory / "STATE.json", new)


def _receipt(registration, task_id, state, intent):
    root = Path(registration["root"])
    directory = root / "ablation/raw" / task_id
    receipt = read(directory / "COMPLETE.json")
    expected = dict(
        status="COMPLETE",
        task_id=task_id,
        job_id=state["job_id"],
        job_name=intent["job_name"],
        user=registration["user"],
        records=256,
        ablation_manifest_sha256=registration["ablation_manifest_sha256"],
        optimizer_steps=0,
        teacher_forward_calls=0,
        monitor_calls=0,
        allocation_sha256=sha(directory / "ALLOCATION.json"),
    )
    if any(receipt.get(k) != v for k, v in expected.items()) or (
        len(receipt.get("raw_groups", [])) != 64 or len(receipt.get("raw_artifacts", [])) != 128
    ):
        raise PermissionError("Completed task receipt is missing or has another identity")
    return dict(
        path=str((directory / "COMPLETE.json").relative_to(root)),
        sha256=sha(directory / "COMPLETE.json"),
    )


def _terminal(registration, intent, state, terminal):
    expected = dict(
        job_id=state["job_id"],
        user=registration["user"],
        account="rose",
        qos=QOS,
        name=intent["job_name"],
    )
    if not terminal or any(terminal.get(k) != v for k, v in expected.items()):
        raise PermissionError("Terminal accounting belongs to another allocation")


def refresh(registration, api, queue, rows):
    for directory, intent, state in rows:
        is_driver = _verify_attempt(registration, api, directory, intent, state)
        if state["status"] in UNKNOWN:
            continue
        job = state.get("job_id")
        if not job:
            raise PermissionError("Known submission has no Slurm job identity")
        if job in queue:
            if state["status"] in FINAL:
                raise PermissionError("Terminal job unexpectedly reappeared in the queue")
            api.orchestration.verify_allocation(
                queue[job], dict(intent, job_id=job), registration["user"]
            )
            if state["status"] == "ACCOUNTING_UNKNOWN":
                _record_state(directory, state, dict(state, status="SUBMITTED"))
            continue
        if state["status"] in FINAL:
            terminal = read(directory / "TERMINAL.json")
        else:
            terminal = api.slurm.terminal(job)
            if terminal is None or terminal.get("state") not in api.orchestration.TERMINAL:
                _record_state(directory, state, dict(state, status="ACCOUNTING_UNKNOWN"))
                continue
            _terminal(registration, intent, state, terminal)
            save(directory / "TERMINAL.json", terminal, immutable=True)
        _terminal(registration, intent, state, terminal)
        success = terminal["state"] == "COMPLETED" and terminal.get("exit_code") == "0:0"
        if state["status"] == "COMPLETE" and not success:
            raise PermissionError("Completed ledger contradicts actual terminal evidence")
        if not is_driver and success:
            try:
                receipt = _receipt(registration, directory.name, state, intent)
            except Exception as error:
                save(
                    directory / "RECEIPT_FAILURE.json",
                    dict(
                        error=repr(error),
                        receipt_only_reconciliation_required=(
                            Path(registration["root"])
                            / "ablation/raw"
                            / directory.name
                            / "COMPLETE.json"
                        ).exists(),
                    ),
                    immutable=True,
                )
                success = False
            else:
                save(directory / "COMPLETION_VERIFIED.json", receipt, immutable=True)
        if not is_driver:
            _record_state(directory, state, dict(state, status="COMPLETE" if success else "FAILED"))
        elif state["status"] not in FINAL:
            # CPU lease expiration is an operational handoff; it never authorizes
            # a repeat of any failed model allocation.
            _record_state(directory, state, dict(state, status="COMPLETE" if success else "FAILED"))


def _persist_submit(registration, api, directory, intent, queue, states):
    _verified_registration(registration, api)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "INTENT.json").exists():
        raise PermissionError("Submission was already attempted; never repeat sbatch")
    task = registration["tasks"].get(intent["key"])
    body = _body(
        registration, task=task, driver_key=intent["key"] if intent["gpu_count"] == 0 else None
    )
    with (directory / "run.sbatch").open("x") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    intent.update(script=str(directory / "run.sbatch"), script_sha256=sha(directory / "run.sbatch"))
    intent["sbatch"] = _sbatch(intent, directory)
    save(
        directory / "QUEUE_BEFORE_SUBMIT.json",
        dict(
            jobs=queue,
            teacher_usage=api.orchestration.teacher_usage(queue, states, registration["user"]),
        ),
        immutable=True,
    )
    save(directory / "INTENT.json", intent, immutable=True)
    state = dict(
        key=intent["key"],
        status="SUBMISSION_UNKNOWN",
        gpu_count=intent["gpu_count"],
        qos=QOS,
        intent_sha256=sha(directory / "INTENT.json"),
    )
    save(directory / "STATE.json", state)
    try:
        result = api.slurm.run(intent["sbatch"])
    except BaseException as error:
        save(directory / "SUBMISSION_EXCEPTION.json", dict(error=repr(error)), immutable=True)
        raise
    save(
        directory / "SUBMISSION.json",
        dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
        immutable=True,
    )
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
    if result.returncode or not match:
        raise RuntimeError("Submission outcome is unknown; no automatic retry")
    state.update(status="HELD", job_id=match[1])
    save(directory / "STATE.json", state)
    fields = api.slurm.show(state["job_id"])
    api.orchestration.verify_allocation(
        fields, dict(intent, job_id=state["job_id"]), registration["user"], held=True
    )
    if (
        hashlib.sha256(api.slurm.batch_script(state["job_id"]).encode()).hexdigest()
        != intent["script_sha256"]
    ):
        raise PermissionError("Actual Slurm batch script differs; allocation remains held")
    release_queue = api.slurm.queue(registration["user"])
    usage = api.orchestration.teacher_usage(
        release_queue, [row[2] for row in _states(Path(registration["root"]))], registration["user"]
    )
    save(
        directory / "QUEUE_BEFORE_RELEASE.json",
        dict(jobs=release_queue, teacher_usage=usage),
        immutable=True,
    )
    if usage["gpus"] > 5 or usage["submitted"] > 5:
        raise PermissionError("Capacity changed; allocation remains held")
    _verified_registration(registration, api)
    save(
        directory / "ALLOCATION_VERIFIED.json",
        dict(job_id=state["job_id"], fields=fields, script_sha256=intent["script_sha256"]),
        immutable=True,
    )
    save(directory / "RELEASE_INTENT.json", dict(job_id=state["job_id"]), immutable=True)
    state["status"] = "RELEASE_UNKNOWN"
    save(directory / "STATE.json", state)
    result = api.slurm.run(["scontrol", "release", state["job_id"]])
    save(
        directory / "RELEASE.json",
        dict(
            job_id=state["job_id"],
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ),
        immutable=True,
    )
    if result.returncode:
        raise RuntimeError("Release outcome is unknown; no automatic retry")
    state["status"] = "SUBMITTED"
    save(directory / "STATE.json", state)
    return state


def _verified_registration(registration, api):
    current = verify_operation(
        Path(registration["root"]), registration["source_commit"], registration["driver_commit"]
    )
    if current != registration or registration["tasks"] != api.ablation.task_matrix():
        raise PermissionError("Operational task matrix or identity changed")


def _intent(registration, key, *, driver=False):
    root = Path(registration["root"])
    task = None if driver else registration["tasks"][key]
    directory = (drivers(root) if driver else submissions(root)) / key
    value = dict(
        key=key,
        created=time.time(),
        user=registration["user"],
        code_commit=registration["source_commit"],
        run_identity_sha256=registration["run_identity_sha256"],
        qos=QOS,
        partition="cluster02",
        cpus=4,
        gpu_count=0 if driver else 1,
        memory_gib=12 if driver else 90,
        cpu_node="cpu-2" if driver else None,
        job_name="SRF3CE-ABLDRIVER-N" + key[7:] + "-LH" if driver else task["job_name"],
        comment=("srf3ce-abldriver-" if driver else "srf3ce-fix-") + key,
        script=str(directory / "run.sbatch"),
    )
    if driver:
        value["operation_identity_sha256"] = sha(
            operation_directory(root) / "OPERATION_IDENTITY.json"
        )
    else:
        value.update(task=task, ablation_manifest_sha256=registration["ablation_manifest_sha256"])
    return value


def _verified_rows(registration, api):
    rows = _states(Path(registration["root"]))
    jobs = []
    for directory, intent, state in rows:
        _verify_attempt(registration, api, directory, intent, state)
        if state.get("job_id"):
            jobs.append(state["job_id"])
    if len(jobs) != len(set(jobs)):
        raise PermissionError("One Slurm allocation appears in multiple submission records")
    return rows


def _blockers(rows):
    return {
        directory.name: state["status"]
        for directory, _, state in rows
        if state["status"] in UNKNOWN | {"FAILED"}
    }


def submit_driver(registration, api, key="driver-0001"):
    """Submit one explicit CPU lease; never requeue or renew it automatically."""
    root = Path(registration["root"])
    if key != "driver-0001":
        raise PermissionError("This helper supports one CPU lease; further handoff needs review")
    with api.orchestration.lease(submissions(root)):
        _verified_registration(registration, api)
        rows = _verified_rows(registration, api)
        if any(directory == drivers(root) / key for directory, _, _ in rows):
            raise PermissionError("CPU submission already attempted; never repeat sbatch")
        queue = api.slurm.queue(registration["user"])
        refresh(registration, api, queue, rows)
        rows = _verified_rows(registration, api)
        if _blockers(rows) or any(state["status"] == "ACCOUNTING_UNKNOWN" for _, _, state in rows):
            raise PermissionError("Unresolved or failed allocation blocks a new CPU lease")
        usage = api.orchestration.teacher_usage(
            queue, [row[2] for row in rows], registration["user"]
        )
        if usage["gpus"] > 5 or usage["submitted"] >= 5:
            return dict(status="WAITING_FOR_CPU_SLOT", teacher_usage=usage)
        return _persist_submit(
            registration,
            api,
            drivers(root) / key,
            _intent(registration, key, driver=True),
            queue,
            [row[2] for row in rows],
        )


def verify_cpu_allocation(registration, api, key):
    root = Path(registration["root"])
    job = os.environ.get("SLURM_JOB_ID", "")
    directory = drivers(root) / key
    if not job.isdigit() or not re.fullmatch(r"driver-[0-9]{4}", key):
        raise PermissionError("An explicitly submitted CPU Slurm allocation is required")
    intent, state = read(directory / "INTENT.json"), read(directory / "STATE.json")
    _verify_attempt(registration, api, directory, intent, state)
    if state.get("job_id") != job or state["status"] != "SUBMITTED":
        raise PermissionError("CPU lease identity or release acknowledgment differs")
    fields = api.slurm.show(job)
    api.orchestration.verify_allocation(fields, dict(intent, job_id=job), registration["user"])
    if (
        fields.get("JobState") != "RUNNING"
        or fields.get("NodeList") != "cpu-2"
        or api.orchestration.physical_gpu_count(fields.get("AllocTRES", "")) != 0
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or Path(sys.executable).absolute() != Path(registration["python"]).absolute()
        or hashlib.sha256(api.slurm.batch_script(job).encode()).hexdigest()
        != intent["script_sha256"]
    ):
        raise PermissionError("Actual CPU lease, Python, or launch script differs")
    return dict(job_id=job, fields=fields, script_sha256=intent["script_sha256"])


def _snapshot(registration, queue, rows, usage):
    keys = ("JobId", "QOS", "JobState", "ReqTRES", "AllocTRES", "UserId")
    return dict(
        teacher_usage=usage,
        queue={
            job: {key: fields.get(key) for key in keys}
            for job, fields in sorted(queue.items())
            if fields.get("QOS") == QOS
        },
        tasks={
            directory.name: dict(status=state["status"], job_id=state.get("job_id"))
            for directory, _, state in rows
        },
    )


def _guard_unregistered_raw(registration, model_rows):
    root = Path(registration["root"]) / "ablation/raw"
    if not root.exists():
        return
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir() or path.name not in registration["tasks"]:
            raise PermissionError("Raw evidence exists outside the registered task matrix")
        if path.name not in model_rows and any(path.iterdir()):
            raise PermissionError("Unsubmitted task has raw/partial evidence; never regenerate")


def select_once(registration, api, *, execute=subprocess.run):
    root = Path(registration["root"])
    directory = operation_directory(root)
    intent_path = directory / "SELECT_INTENT.json"
    selection_path = root / "PROTOCOL_SELECTION.json"
    command = _command(registration, select=True)
    expected = dict(
        command=command,
        operation_identity_sha256=sha(directory / "OPERATION_IDENTITY.json"),
        ablation_manifest_sha256=registration["ablation_manifest_sha256"],
        scientific_source_manifest_sha256=registration["scientific_source_manifest_sha256"],
    )
    if intent_path.exists():
        if read(intent_path) != expected:
            raise PermissionError("Selection intent identity changed")
        if not selection_path.exists():
            raise PermissionError("Selection outcome is unknown; no automatic invocation retry")
    elif selection_path.exists():
        raise PermissionError("Selection exists without this operation's invocation evidence")
    else:
        _verified_registration(registration, api)
        save(intent_path, expected, immutable=True)
        environment = dict(
            os.environ,
            PYTHONPATH=str(root / "code/src"),
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="",
        )
        try:
            result = execute(
                command,
                cwd=root / "code",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        except BaseException as error:
            save(directory / "SELECT_EXCEPTION.json", dict(error=repr(error)), immutable=True)
            raise
        save(
            directory / "SELECT_RESULT.json",
            dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
            immutable=True,
        )
        if result.returncode:
            raise RuntimeError("Frozen selection CLI failed; preserve evidence for review")
    value = api.ablation.verify_selection(root)
    if value.get("ablation_root") != str(root) or value.get("status") not in {
        "PROTOCOL_SELECTED",
        "STACK_REPRODUCTION_FAILURE",
        "STOP_PROTOCOL_FAILURE",
    }:
        raise PermissionError("Frozen selection receipt identity/status differs")
    result = dict(
        status="SELECTION_COMPLETE",
        selection_status=value["status"],
        selected_revision=value.get("selected_revision"),
        selection_sha256=sha(selection_path),
        scientific_execution_authorized=False,
    )
    save(directory / "RESULT.json", result, immutable=True)
    return result


def tick(registration, api, driver_key="driver-0001", *, execute=subprocess.run):
    root = Path(registration["root"])
    with api.orchestration.lease(submissions(root)):
        _verified_registration(registration, api)
        verify_cpu_allocation(registration, api, driver_key)
        queue = api.slurm.queue(registration["user"])
        rows = _verified_rows(registration, api)
        refresh(registration, api, queue, rows)
        rows = _verified_rows(registration, api)
        usage = api.orchestration.teacher_usage(
            queue, [row[2] for row in rows], registration["user"]
        )
        snapshot = _snapshot(registration, queue, rows, usage)
        blockers = _blockers(rows)
        if blockers:
            event(registration, dict(status="BLOCKED", blockers=blockers, **snapshot))
            raise PermissionError(
                "Unknown, held, or failed submission requires review; never retry"
            )
        if usage["gpus"] > 5 or usage["submitted"] > 5:
            event(registration, dict(status="BLOCKED_CAPACITY", **snapshot))
            raise PermissionError("Observed teacher usage exceeds the registered resource limits")
        if any(state["status"] == "ACCOUNTING_UNKNOWN" for _, _, state in rows):
            result = dict(status="WAITING_FOR_ACCOUNTING", **snapshot)
            event(registration, result)
            return result
        model_rows = {
            directory.name: state
            for directory, _, state in rows
            if directory.parent == submissions(root)
        }
        _guard_unregistered_raw(registration, model_rows)
        if len(model_rows) == 9 and all(
            state["status"] == "COMPLETE" for state in model_rows.values()
        ):
            result = select_once(registration, api, execute=execute)
            event(registration, result)
            return result
        # At most one new task per poll. Every invocation gets a fresh queue
        # snapshot and retains a durable pre-sbatch intent before touching Slurm.
        missing = next((key for key in api.ablation.task_matrix() if key not in model_rows), None)
        if missing and usage["gpus"] < 5 and usage["submitted"] < 5:
            state = _persist_submit(
                registration,
                api,
                submissions(root) / missing,
                _intent(registration, missing),
                queue,
                [row[2] for row in rows],
            )
            result = dict(status="SUBMITTED_FIXED_TASK", task_id=missing, job_id=state["job_id"])
        else:
            result = dict(status="WAITING", **snapshot)
        event(registration, result)
        return result


def run(registration, api, driver_key):
    root = Path(registration["root"])
    directory = drivers(root) / driver_key
    with lock(operation_directory(root) / "DRIVER.lock"):
        # sbatch --hold was checked before release; the process may start a
        # moment before the submitter durably records successful release.
        for _ in range(15):
            if read(directory / "STATE.json").get("status") != "RELEASE_UNKNOWN":
                break
            time.sleep(1)
        _verified_registration(registration, api)
        allocation = verify_cpu_allocation(registration, api, driver_key)
        if (directory / "STARTED.json").exists():
            raise PermissionError("This CPU lease was already started; no automatic process retry")
        save(directory / "STARTED.json", dict(at=time.time(), **allocation), immutable=True)
        try:
            while True:
                result = tick(registration, api, driver_key)
                if result["status"] == "SELECTION_COMPLETE":
                    save(
                        directory / "END.json",
                        dict(status="COMPLETE", result=result),
                        immutable=True,
                    )
                    return result
                time.sleep(60)
        except BaseException as error:
            try:
                save(
                    directory / "END.json",
                    dict(status="FAILED", error=repr(error), scientific_retry_authorized=False),
                    immutable=True,
                )
            except BaseException as persistence_error:
                error.add_note(
                    "CPU failure receipt could not be persisted: " + repr(persistence_error)
                )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("register", "submit", "run", "tick"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--driver-commit", required=True)
    parser.add_argument("--python", help="Required for register; frozen worker Python path")
    parser.add_argument("--driver-key", default="driver-0001")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    helper_sources(root, args.source_commit, args.driver_commit)
    if args.mode in {"run", "tick"} and not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise PermissionError("Polling and selection must run inside the registered CPU Slurm job")
    if args.mode == "register":
        if not args.python:
            parser.error("--python is required for register")
        result = register(root, args.source_commit, args.driver_commit, args.python)
    else:
        registration = verify_operation(root, args.source_commit, args.driver_commit)
        api = load_api(root, args.source_commit)
        if args.mode == "submit":
            result = submit_driver(registration, api, args.driver_key)
        elif args.mode == "tick":
            with lock(operation_directory(root) / "DRIVER.lock"):
                result = tick(registration, api, args.driver_key)
        else:
            result = run(registration, api, args.driver_key)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
