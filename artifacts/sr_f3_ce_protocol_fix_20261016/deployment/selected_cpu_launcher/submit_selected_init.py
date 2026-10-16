#!/usr/bin/env python3
"""Single CPU initialization lease after a verified protocol selection.

This operational helper reuses the audited 755 dispatcher and frozen 652 APIs.
It never initializes a selected directory itself or retries an uncertain job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

SOURCE_COMMIT = "652e52d530520a276b6420ff0f4ee479dcde6db1"
DISPATCH_COMMIT = "7551617b2c3fa6c958be87059119d78a604580f6"
DISPATCH_SHA256 = "617e1809399ba86758883e132bc18657088ca10a788e3e7318214acb44d61fea"
TEST_SHA256 = "1aa06c0bcc2ac8ec36d772fe1874b9fc5d2cabd41ca4b23d80516825de09a98c"
QOS = "soujanya-poria-startfund-2026-03"
JOB_NAME = "SRF3CE-SELECTED-INIT-LH"


def support(root):
    """Execute the same bytes authenticated by the immutable audited manifest."""
    path = root / "operations/ablation_driver" / DISPATCH_COMMIT / "dispatch_protocol_ablation.py"
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != DISPATCH_SHA256:
        raise PermissionError("Audited dispatcher bytes differ")
    spec = importlib.util.spec_from_file_location("selected_init_audited_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    module.helper_sources(root, SOURCE_COMMIT, DISPATCH_COMMIT)
    if module.sha(path.parent / "test_protocol_ablation_dispatch.py") != TEST_SHA256:
        raise PermissionError("Audited dispatcher validation source differs")
    # This verifies the full 652 file inventory before any scientific import.
    api = module.load_api(root, SOURCE_COMMIT)
    registration = module.verify_operation(root, SOURCE_COMMIT, DISPATCH_COMMIT)
    return module, api, registration


def directory(root):
    return root.parent / "deployment/selected_init"


def selected(root, api):
    receipt = api.ablation.verify_selection(root)
    revision = receipt.get("selected_revision")
    if (
        receipt.get("status") != "PROTOCOL_SELECTED"
        or receipt.get("ablation_root") != str(root)
        or revision not in {"v2", "c2"}
    ):
        raise PermissionError("A complete, successful frozen protocol selection is required")
    target = root.parent / ("selected_" + revision)
    return receipt, target


def payload_check(h, root, path, expected_hash):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or not path.is_file()
        or not path.is_relative_to(root.parent / "deployment")
        or path.stat().st_mode & 0o222
        or not re.fullmatch(r"[a-f0-9]{64}", expected_hash)
        or h.sha(path) != expected_hash
    ):
        raise PermissionError("Payload must be a registered, read-only local deployment file")
    data = path.read_bytes()
    if not data.startswith((b"#!/bin/bash\n", b"#!/usr/bin/env bash\n")) or b"\x00" in data:
        raise PermissionError("The registered payload must be an explicit Bash script")
    return path


def register(root, h, api, ablation_registration, payload, payload_sha256, python):
    _, target = selected(root, api)
    if target.exists() or target.is_symlink():
        raise PermissionError("Fresh selected root already exists; never overwrite or resume")
    payload = payload_check(h, root, payload, payload_sha256)
    python = str(Path(python).absolute())
    if python != ablation_registration["python"] or not Path(python).is_file():
        raise PermissionError("Use the already registered scientific Python executable")
    value = dict(
        schema_version=1,
        status="REGISTERED_SINGLE_SELECTED_CPU_INITIALIZATION",
        ablation_root=str(root),
        selected_root=str(target),
        selected_revision=target.name.removeprefix("selected_"),
        source_commit=SOURCE_COMMIT,
        dispatcher_commit=DISPATCH_COMMIT,
        dispatcher_sha256=DISPATCH_SHA256,
        run_identity_sha256=h.sha(root / "RUN_IDENTITY.json"),
        selection_sha256=h.sha(root / "PROTOCOL_SELECTION.json"),
        source_manifest_sha256=h.sha(root / "deployment/SOURCE_MANIFEST.json"),
        ablation_operation_sha256=h.sha(h.operation_directory(root) / "OPERATION_IDENTITY.json"),
        helper_path=str(Path(__file__).resolve(strict=True)),
        helper_sha256=h.sha(__file__),
        payload_path=str(payload),
        payload_sha256=payload_sha256,
        payload_bytes=payload.stat().st_size,
        python=python,
        user=ablation_registration["user"],
        gpu_count=0,
        max_gpus=5,
        max_submitted=5,
        automatic_retry_authorized=False,
    )
    h.save(directory(root) / "OPERATION_IDENTITY.json", value, immutable=True)
    return value


def verify(root, h, api, ablation_registration, *, fresh=False):
    value = h.read(directory(root) / "OPERATION_IDENTITY.json")
    _, target = selected(root, api)
    expected = dict(
        ablation_root=str(root),
        selected_root=str(target),
        selected_revision=target.name.removeprefix("selected_"),
        source_commit=SOURCE_COMMIT,
        dispatcher_commit=DISPATCH_COMMIT,
        dispatcher_sha256=DISPATCH_SHA256,
        run_identity_sha256=h.sha(root / "RUN_IDENTITY.json"),
        selection_sha256=h.sha(root / "PROTOCOL_SELECTION.json"),
        source_manifest_sha256=h.sha(root / "deployment/SOURCE_MANIFEST.json"),
        ablation_operation_sha256=h.sha(h.operation_directory(root) / "OPERATION_IDENTITY.json"),
        helper_path=str(Path(__file__).resolve(strict=True)),
        helper_sha256=h.sha(__file__),
        python=ablation_registration["python"],
        user=ablation_registration["user"],
        gpu_count=0,
        max_gpus=5,
        max_submitted=5,
        automatic_retry_authorized=False,
    )
    if any(value.get(key) != item for key, item in expected.items()):
        raise PermissionError("Selected CPU operation identity changed")
    payload = payload_check(h, root, value["payload_path"], value["payload_sha256"])
    if payload.stat().st_size != value["payload_bytes"]:
        raise PermissionError("Payload length changed")
    if fresh and (target.exists() or target.is_symlink()):
        raise PermissionError("Fresh selected root already exists; never retry partial preparation")
    return value


def script(registration):
    command = [
        registration["python"],
        registration["helper_path"],
        "--mode",
        "execute",
        "--ablation-root",
        registration["ablation_root"],
    ]
    return (
        "#!/bin/bash\nset -euo pipefail\n"
        "export CUDA_VISIBLE_DEVICES=''\n"
        "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
        "exec " + shlex.join(command) + "\n"
    )


def intent_for(root, h, registration):
    target = directory(root)
    intent = dict(
        key="selected-init-0001",
        job_name=JOB_NAME,
        qos=QOS,
        partition="cluster02",
        cpus=4,
        memory_gib=12,
        gpu_count=0,
        cpu_node="cpu-2",
        user=registration["user"],
        comment="srf3ce-selected-init-0001",
        script=str(target / "run.sbatch"),
        operation_identity_sha256=h.sha(target / "OPERATION_IDENTITY.json"),
        payload_command=["/bin/bash", registration["payload_path"]],
        payload_sha256=registration["payload_sha256"],
    )
    intent["script_sha256"] = hashlib.sha256(script(registration).encode()).hexdigest()
    intent["sbatch"] = h._sbatch(intent, target)
    return intent


def verify_intent(root, h, registration):
    target = directory(root)
    intent, state = h.read(target / "INTENT.json"), h.read(target / "STATE.json")
    if intent != intent_for(root, h, registration) or (
        state.get("key") != intent["key"]
        or state.get("gpu_count") != 0
        or state.get("qos") != QOS
        or state.get("intent_sha256") != h.sha(target / "INTENT.json")
        or state.get("status") not in {"SUBMISSION_UNKNOWN", "HELD", "RELEASE_UNKNOWN", "SUBMITTED"}
        or ("job_id" in state and not str(state["job_id"]).isdigit())
        or Path(intent["script"]).read_text() != script(registration)
    ):
        raise PermissionError("Selected initialization intent/state/script differs")
    return intent, state


def capacity(root, h, api, ablation_registration, *, own_state=None):
    """Resolve old terminal reservations read-only; retain every uncertain slot."""
    queue = api.slurm.queue(ablation_registration["user"])
    rows = h._verified_rows(ablation_registration, api)
    states, accounting = [], {}
    for _, intent, state in rows:
        if state["status"] in h.UNKNOWN:
            raise PermissionError("Uncertain earlier submission blocks selected initialization")
        job = state.get("job_id")
        if not job:
            raise PermissionError("Earlier submission has no allocation identity")
        state = dict(state)
        if job in queue:
            api.orchestration.verify_allocation(
                queue[job], dict(intent, job_id=job), ablation_registration["user"]
            )
            if state["status"] in h.FINAL:
                raise PermissionError("Terminal allocation reappeared in queue")
        else:
            terminal = api.slurm.terminal(job)
            if not terminal or terminal.get("state") not in api.orchestration.TERMINAL:
                raise PermissionError("Earlier accounting is unknown; keep its reservation")
            h._terminal(ablation_registration, intent, state, terminal)
            success = terminal["state"] == "COMPLETED" and terminal.get("exit_code") == "0:0"
            if intent["gpu_count"] and not success:
                raise PermissionError("An ablation model allocation did not successfully complete")
            state["status"] = "COMPLETE" if success else "FAILED"
            accounting[job] = terminal
        states.append(state)
    if own_state is not None:
        states.append(own_state)
    return dict(
        jobs=queue,
        accounting=accounting,
        teacher_usage=api.orchestration.teacher_usage(queue, states, ablation_registration["user"]),
    )


def submit(root, h, api, ablation_registration):
    target = directory(root)
    with api.orchestration.lease(h.submissions(root)):
        registration = verify(root, h, api, ablation_registration, fresh=True)
        if (target / "INTENT.json").exists() or (target / "STATE.json").exists():
            raise PermissionError(
                "This single initialization was already attempted; never resubmit"
            )
        before = capacity(root, h, api, ablation_registration)
        usage = before["teacher_usage"]
        if usage["gpus"] > 5 or usage["submitted"] >= 5:
            return dict(status="WAITING_FOR_CAPACITY", teacher_usage=usage)
        intent = intent_for(root, h, registration)
        with (target / "run.sbatch").open("x") as stream:
            stream.write(script(registration))
            stream.flush()
            os.fsync(stream.fileno())
        h.save(target / "QUEUE_BEFORE_SUBMIT.json", before, immutable=True)
        h.save(target / "INTENT.json", intent, immutable=True)
        state = dict(
            key=intent["key"],
            status="SUBMISSION_UNKNOWN",
            gpu_count=0,
            qos=QOS,
            intent_sha256=h.sha(target / "INTENT.json"),
        )
        h.save(target / "STATE.json", state)
        try:
            reply = api.slurm.run(intent["sbatch"])
        except BaseException as error:
            h.save(target / "SUBMISSION_EXCEPTION.json", dict(error=repr(error)), immutable=True)
            raise
        h.save(target / "SUBMISSION.json", vars_reply(reply), immutable=True)
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", reply.stdout)
        if reply.returncode or not match:
            raise RuntimeError("Submission outcome is unknown; never repeat sbatch")
        state.update(job_id=match[1], status="HELD")
        h.save(target / "STATE.json", state)
        fields = api.slurm.show(state["job_id"])
        api.orchestration.verify_allocation(
            fields, dict(intent, job_id=state["job_id"]), registration["user"], held=True
        )
        actual_script = api.slurm.batch_script(state["job_id"])
        if hashlib.sha256(actual_script.encode()).hexdigest() != intent["script_sha256"]:
            raise PermissionError("Actual batch script differs; allocation remains held")
        h.save(
            target / "ALLOCATION_VERIFIED.json",
            dict(job_id=state["job_id"], fields=fields, script_sha256=intent["script_sha256"]),
            immutable=True,
        )
        verify(root, h, api, ablation_registration, fresh=True)
        after = capacity(root, h, api, ablation_registration, own_state=state)
        h.save(target / "QUEUE_BEFORE_RELEASE.json", after, immutable=True)
        if after["teacher_usage"]["gpus"] > 5 or after["teacher_usage"]["submitted"] > 5:
            raise PermissionError("Capacity changed; allocation remains held")
        h.save(target / "RELEASE_INTENT.json", dict(job_id=state["job_id"]), immutable=True)
        state["status"] = "RELEASE_UNKNOWN"
        h.save(target / "STATE.json", state)
        try:
            reply = api.slurm.run(["scontrol", "release", state["job_id"]])
        except BaseException as error:
            h.save(target / "RELEASE_EXCEPTION.json", dict(error=repr(error)), immutable=True)
            raise
        h.save(target / "RELEASE.json", vars_reply(reply), immutable=True)
        if reply.returncode:
            raise RuntimeError("Release outcome is unknown; never repeat release")
        state["status"] = "SUBMITTED"
        h.save(target / "STATE.json", state)
        return state


def vars_reply(reply):
    return dict(returncode=reply.returncode, stdout=reply.stdout, stderr=reply.stderr)


def released(root, h, api, registration):
    intent, state = verify_intent(root, h, registration)
    target = directory(root)
    allocation = h.read(target / "ALLOCATION_VERIFIED.json")
    reply = h.read(target / "SUBMISSION.json")
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", reply.get("stdout", ""))
    if (
        state["status"] != "SUBMITTED"
        or reply.get("returncode") != 0
        or not match
        or match[1] != state.get("job_id")
        or allocation.get("job_id") != state["job_id"]
        or allocation.get("script_sha256") != intent["script_sha256"]
        or h.read(target / "RELEASE_INTENT.json").get("job_id") != state["job_id"]
        or h.read(target / "RELEASE.json").get("returncode") != 0
    ):
        raise PermissionError("Successful held/release acknowledgment is missing")
    api.orchestration.verify_allocation(
        allocation["fields"], dict(intent, job_id=state["job_id"]), registration["user"], held=True
    )
    return intent, state


def execute(root, h, api, ablation_registration, *, run_payload=subprocess.run):
    target = directory(root)
    with h.lock(target / "EXECUTION.lock"):
        for _ in range(15):
            if h.read(target / "STATE.json").get("status") != "RELEASE_UNKNOWN":
                break
            time.sleep(1)
        registration = verify(root, h, api, ablation_registration, fresh=True)
        intent, state = released(root, h, api, registration)
        if state["job_id"] != os.environ.get("SLURM_JOB_ID"):
            raise PermissionError("Execute only inside this released CPU allocation")
        fields = api.slurm.show(state["job_id"])
        api.orchestration.verify_allocation(
            fields, dict(intent, job_id=state["job_id"]), registration["user"]
        )
        if (
            fields.get("JobState") != "RUNNING"
            or fields.get("NodeList") != "cpu-2"
            or socket.gethostname().split(".", 1)[0] != "cpu-2"
            or api.orchestration.physical_gpu_count(fields.get("AllocTRES", "")) != 0
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.executable).absolute() != Path(registration["python"]).absolute()
            or hashlib.sha256(api.slurm.batch_script(state["job_id"]).encode()).hexdigest()
            != intent["script_sha256"]
            or (target / "STARTED.json").exists()
        ):
            raise PermissionError("Actual CPU launch differs or initialization was already started")
        h.save(
            target / "STARTED.json",
            dict(
                job_id=state["job_id"],
                fields=fields,
                command=intent["payload_command"],
                payload_sha256=intent["payload_sha256"],
                at=time.time(),
            ),
            immutable=True,
        )
        try:
            result = run_payload(
                intent["payload_command"],
                check=False,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
            )
            value = dict(
                status="PAYLOAD_PROCESS_COMPLETED" if result.returncode == 0 else "FAILED",
                returncode=result.returncode,
                job_id=state["job_id"],
            )
            h.save(target / "PAYLOAD_RESULT.json", value, immutable=True)
            if result.returncode:
                raise RuntimeError("Selected initialization payload failed; no automatic retry")
            return value
        except BaseException as error:
            try:
                h.save(target / "EXECUTION_FAILURE.json", dict(error=repr(error)), immutable=True)
            except BaseException as persistence_error:
                error.add_note("Failure receipt could not be persisted: " + repr(persistence_error))
            raise


def terminal(root, h, api, ablation_registration):
    registration = verify(root, h, api, ablation_registration)
    intent, state = verify_intent(root, h, registration)
    job = state.get("job_id")
    if not job:
        return dict(status="SUBMISSION_UNKNOWN", automatic_retry_authorized=False)
    queue = api.slurm.queue(registration["user"])
    if job in queue:
        api.orchestration.verify_allocation(
            queue[job], dict(intent, job_id=job), registration["user"]
        )
        return dict(status="ACCOUNTING_UNKNOWN_OR_ACTIVE", job_id=job)
    record = api.slurm.terminal(job)
    if not record or record.get("state") not in api.orchestration.TERMINAL:
        return dict(status="ACCOUNTING_UNKNOWN_OR_ACTIVE", job_id=job)
    h._terminal(ablation_registration, intent, state, record)
    h.save(directory(root) / "TERMINAL.json", record, immutable=True)
    return dict(status="TERMINAL_RECORDED", terminal=record, scientific_acceptance_claimed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("register", "submit", "execute", "terminal")
    )
    parser.add_argument("--ablation-root", required=True, type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--payload-sha256")
    parser.add_argument("--python")
    args = parser.parse_args(argv)
    root = args.ablation_root.resolve(strict=True)
    h, api, ablation_registration = support(root)
    if args.mode == "register":
        if not args.payload or not args.payload_sha256 or not args.python:
            parser.error("register requires --payload, --payload-sha256 and --python")
        result = register(
            root, h, api, ablation_registration, args.payload, args.payload_sha256, args.python
        )
    elif args.mode == "submit":
        result = submit(root, h, api, ablation_registration)
    elif args.mode == "execute":
        result = execute(root, h, api, ablation_registration)
    else:
        result = terminal(root, h, api, ablation_registration)
    print(h.json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
