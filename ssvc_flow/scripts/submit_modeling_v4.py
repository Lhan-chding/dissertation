#!/usr/bin/env python3
"""Print or submit one V4 campaign. No models, data arrays, or SSH are opened.

Default and --print-only are side-effect-free. --submit records CPU jobs;
the first GPU submission also requires --execute-gpu, retained for this config
and campaign. --resume consults Slurm and never resubmits an active job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
RECORD = "SLURM_SUBMISSIONS.json"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        ).encode()
    ).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--campaign-root", required=True)
    result.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    result.add_argument(
        "--python", default="/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
    )
    result.add_argument("--tasks", help="Frozen gpu_collect task list; phases come from this file")
    result.add_argument("--reuse", help="Existing CPU raw collection or input mapping JSON")
    result.add_argument("--origin-subset", help="Comma-separated CPU origin IDs, passed verbatim")
    result.add_argument(
        "--phase",
        choices=("bridge", "firstmap", "development", "calibration", "test", "tracking"),
        default="bridge",
        help="Submission/log label only; does not filter the task table",
    )
    result.add_argument("--stage", choices=("cpu", "gpu", "both"), default="both")
    result.add_argument("--workers", type=int, choices=(2, 3))
    result.add_argument("--cpu-cpus", type=int, default=4)
    result.add_argument("--cpu-mem", default="32G")
    result.add_argument("--gpu-cpus", type=int, default=4)
    result.add_argument("--gpu-mem", default="48G")
    result.add_argument("--submit", action="store_true")
    result.add_argument("--print-only", action="store_true")
    result.add_argument("--execute-gpu", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument(
        "--extend-phase",
        action="store_true",
        help="With --resume, accept an immutable child task table after GPU jobs finish",
    )
    return result


def build_plan(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    operations = config["operations"]
    maximum = operations["max_concurrent_gpus"]
    workers = args.workers or operations["default_workers"]
    if type(maximum) is not int or not 1 <= maximum <= 3 or not 2 <= workers <= maximum:
        raise ValueError("workers exceed the campaign limit of at most three GPUs")
    for cpus, memory in ((args.cpu_cpus, args.cpu_mem), (args.gpu_cpus, args.gpu_mem)):
        if cpus < 1 or re.fullmatch(r"[1-9][0-9]*[KMGT]?", memory) is None:
            raise ValueError("Resource requests need positive CPUs and a Slurm memory quantity")
    root, project = Path(args.campaign_root).resolve(), Path(args.project_root).resolve()
    python = str(Path(args.python).expanduser())
    if not Path(python).is_absolute():
        raise ValueError("Use an absolute production Python path")
    identity = {
        "campaign_root": str(root),
        "config": binding(config_path),
        "config_hash": digest(config),
        "project_root": str(project),
        "python": python,
    }
    tasks, task_binding = None, None
    if args.stage in ("gpu", "both"):
        if not args.tasks:
            raise ValueError("GPU submission requires --tasks")
        task_binding = binding(args.tasks)
        tasks = json.loads(Path(args.tasks).read_text())
        if tasks.get("schema") != "ssvc-v4-task-list-1":
            raise ValueError("Expected the frozen V4 task list schema")
        if tasks.get("workers") != workers or Path(tasks.get("root", "")).resolve() != root:
            raise ValueError("Task-list workers/root must match this campaign")
        if (
            tasks.get("task_list_hash")
            != digest({key: value for key, value in tasks.items() if key != "task_list_hash"})
            or tasks.get("config") != config
        ):
            raise ValueError("Task-list content/config hash changed")
    commands = []

    def command(key, kind, script, arguments, job_identity):
        cpus = args.cpu_cpus if kind == "cpu" else args.gpu_cpus
        memory = args.cpu_mem if kind == "cpu" else args.gpu_mem
        script_path = project / "scripts" / script
        argv = [
            "sbatch",
            "--parsable",
            "--partition=cluster02",
            "--account=rose",
            "--qos=soujanya-poria-startfund-2026-03",
            "--nodes=1",
            "--ntasks=1",
            "--no-requeue",
            "--time=3-00:00:00",
            f"--cpus-per-task={cpus}",
            f"--mem={memory}",
            "--export=ALL",
            "--constraint=cpu_ok" if kind == "cpu" else "--gres=gpu:pro6000:1",
            f"--job-name=ssvc-v4-{args.phase}-{key}",
            f"--output={root}/slurm/%x-%j.out",
            f"--error={root}/slurm/%x-%j.err",
            str(script_path),
            python,
            str(project),
            str(root),
            *arguments,
        ]
        if args.resume:
            argv.append("--resume")
        commands.append(
            {
                "key": key,
                "kind": kind,
                "identity": job_identity,
                "script": binding(script_path),
                "argv": argv,
                "shell": shlex.join(argv),
            }
        )

    if args.stage in ("cpu", "both"):
        if not args.reuse or not Path(args.reuse).exists():
            raise ValueError("CPU submission requires an existing --reuse root or mapping")
        reuse = Path(args.reuse).resolve()
        reuse_binding = binding(reuse) if reuse.is_file() else {"path": str(reuse)}
        extra = [str(config_path), str(reuse)]
        if args.origin_subset:
            extra += ["--origin-subset", args.origin_subset]
        command(
            "cpu",
            "cpu",
            "modeling_v4_cpu.sbatch",
            extra,
            {"reuse": reuse_binding, "origin_subset": args.origin_subset},
        )
    if tasks is not None:
        for worker in range(workers):
            command(
                f"gpu_worker_{worker}",
                "gpu",
                "modeling_v4_gpu.sbatch",
                [task_binding["path"], "--worker-id", str(worker), "--workers", str(workers)],
                {
                    "task_list": task_binding,
                    "campaign_id": tasks["campaign_id"],
                    "task_list_hash": tasks["task_list_hash"],
                    "worker_id": worker,
                    "workers": workers,
                },
            )
    return {
        "status": "DRY_RUN",
        "identity": identity,
        "max_concurrent_gpus": maximum,
        "phase_label_only": args.phase,
        "commands": commands,
        "requested_resources_are_not_measured_allocations": True,
    }


def clean_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SBATCH_", "PYTHON"))
        and key not in {"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "PYTEST_ADDOPTS"}
    }


def job_states(jobs, run, env):
    ids = []
    for entry in jobs.values():
        attempt = entry["attempts"][-1]
        if attempt["status"] in {"INTENT", "UNCERTAIN_SUBMISSION"}:
            raise RuntimeError("Uncertain submission: reconcile the retained receipt before retry")
        if attempt.get("job_id"):
            ids.append(attempt["job_id"])
    if not ids:
        return {}
    result = run(
        ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    states = {}
    for line in result.stdout.splitlines():
        pieces = line.strip().split("|")
        if len(pieces) == 2 and pieces[0] in ids:
            states[pieces[0]] = pieces[1].split()[0].split("+")[0]
    for job_id in ids:
        if job_id in states:
            continue
        result = run(
            ["sacct", "--noheader", "--parsable2", "--jobs", job_id, "--format=JobIDRaw,State"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        for line in result.stdout.splitlines():
            pieces = line.strip().split("|")
            if len(pieces) >= 2 and pieces[0] == job_id and pieces[1]:
                states[job_id] = pieces[1].split()[0].split("+")[0]
        if job_id not in states:
            raise RuntimeError(
                f"Unknown scheduler state for {job_id}; no duplicate will be submitted"
            )
    return states


def worker_finished(root, command):
    worker = command["identity"]["worker_id"]
    path = root / "workers" / f"worker_{worker}" / "LATEST.json"
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    expected = command["identity"]
    if any(
        receipt.get(k) != expected[k]
        for k in ("campaign_id", "task_list_hash", "worker_id", "workers")
    ):
        raise ValueError("Worker receipt identity changed")
    return receipt.get("status") == "COMPLETED_REQUESTED_STAGES"


def _verify_extension(previous_binding, current_binding):
    """Keep the old table byte-bound; phase extensions cannot rewrite a campaign."""
    if binding(previous_binding["path"]) != previous_binding:
        raise ValueError("The old immutable task manifest changed or was overwritten")
    previous = json.loads(Path(previous_binding["path"]).read_text())
    current = json.loads(Path(current_binding["path"]).read_text())
    if current.get("parent_task_list_hash") != previous["task_list_hash"]:
        raise ValueError("Phase extension must name the previous task-list hash")
    changing = {"task_list_hash", "parent_task_list_hash", "phase_extensions", "tasks", "stages"}
    if {k: v for k, v in previous.items() if k not in changing} != {
        k: v for k, v in current.items() if k not in changing
    }:
        raise ValueError("Phase extension changed the existing config/campaign/runtime inputs")
    previous_extensions = previous.get("phase_extensions", [])
    extensions = current.get("phase_extensions", [])
    if extensions[: len(previous_extensions)] != previous_extensions or len(extensions) <= len(
        previous_extensions
    ):
        raise ValueError("Phase metadata must be append-only")
    seen = {row["id"]: row for row in previous["tasks"]}
    appended = []
    phases = list(previous.get("stages", []))
    for extension in extensions[len(previous_extensions) :]:
        if (
            extension.get("schema") != "ssvc-v4-phase-extension-1"
            or extension.get("extension_hash")
            != digest({k: v for k, v in extension.items() if k != "extension_hash"})
            or extension.get("campaign_id") != current["campaign_id"]
            or extension.get("config_hash") != digest(current["config"])
            or extension.get("workers") != current["workers"]
            or extension.get("phase") not in ("D", "E")
        ):
            raise ValueError("Invalid immutable phase extension metadata")
        phases.append(extension["phase"])
        for task in extension["tasks"]:
            old = seen.get(task["id"])
            if old:
                if (
                    old.get("kind") != "source"
                    or task.get("kind") != "source"
                    or not task.get("reuse_completed_receipt")
                    or any(old.get(k) != task.get(k) for k in ("seed", "arm"))
                ):
                    raise ValueError("Phase extension rewrites an existing task")
            else:
                seen[task["id"]] = task
                appended.append(task)
    if current["tasks"] != previous["tasks"] + appended or current.get("stages") != list(
        dict.fromkeys(phases)
    ):
        raise ValueError("Task manifest is not the declared append-only extension")


def _accept_extension(record, plan, args, states):
    changed = []
    for command in plan["commands"]:
        old = record["jobs"].get(command["key"])
        if not old:
            continue
        if old["script"] != command["script"]:
            raise ValueError("Existing script identity changed")
        if old["identity"] != command["identity"]:
            if command["kind"] != "gpu" or not args.resume or not args.extend_phase:
                raise ValueError(
                    "Existing task/input identity changed; phase extension must be explicit"
                )
            changed.append((command, old))
    if not changed:
        return
    if any(
        entry["kind"] == "gpu" and states.get(entry["attempts"][-1].get("job_id")) not in TERMINAL
        for entry in record["jobs"].values()
    ):
        raise ValueError("Cannot extend while any registered campaign GPU job is active")
    old_bindings = {digest(old["identity"]["task_list"]) for _, old in changed}
    new_bindings = {digest(command["identity"]["task_list"]) for command, _ in changed}
    if len(old_bindings) != 1 or len(new_bindings) != 1:
        raise ValueError("All campaign workers must share one immutable task-table version")
    previous = changed[0][1]["identity"]["task_list"]
    current = changed[0][0]["identity"]["task_list"]
    _verify_extension(previous, current)
    record.setdefault("phase_transitions", []).append(
        {
            "previous": previous,
            "current": current,
            "recorded_utc": now(),
            "all_previous_gpu_jobs_terminal": True,
        }
    )
    # Preserve entire old job entries instead of editing historical attempts.
    for command, old in changed:
        archived_key = command["key"] + "@" + old["identity"]["task_list_hash"]
        if archived_key in record["jobs"]:
            raise ValueError("Duplicate historical task-table version")
        record["jobs"][archived_key] = old
        del record["jobs"][command["key"]]


def execute(args, *, run=None):
    plan = build_plan(args)
    if not args.submit or args.print_only:
        return plan
    run = subprocess.run if run is None else run
    root = Path(plan["identity"]["campaign_root"])
    root.mkdir(parents=True, exist_ok=True)
    env = clean_environment()
    with (root / ".slurm_submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = root / RECORD
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (
                record.get("schema") != "ssvc-v4-slurm-submissions-1"
                or record.get("identity") != plan["identity"]
            ):
                raise ValueError("Campaign/config submission identity changed")
            if not args.resume:
                raise ValueError("Existing campaign submission requires explicit --resume")
        else:
            record = {
                "schema": "ssvc-v4-slurm-submissions-1",
                "identity": plan["identity"],
                "created_utc": now(),
                "gpu_execution_enabled": False,
                "jobs": {},
            }
        wants_gpu = any(c["kind"] == "gpu" for c in plan["commands"])
        if wants_gpu and not (args.execute_gpu or record["gpu_execution_enabled"]):
            raise ValueError("First campaign GPU submission requires --execute-gpu")
        states = job_states(record["jobs"], run, env)
        _accept_extension(record, plan, args, states)
        active = sum(
            entry["kind"] == "gpu"
            and states.get(entry["attempts"][-1].get("job_id")) not in TERMINAL
            for entry in record["jobs"].values()
            if entry["attempts"][-1].get("job_id")
        )
        outcomes, pending = [], []
        for command in plan["commands"]:
            old = record["jobs"].get(command["key"])
            previous = old["attempts"][-1] if old else {}
            state = states.get(previous.get("job_id"))
            action = None
            if state and state not in TERMINAL:
                action = "REUSED_ACTIVE"
            elif state == "COMPLETED" and (
                command["kind"] == "cpu" or worker_finished(root, command)
            ):
                action = "REUSED_COMPLETED"
            outcome = {
                "key": command["key"],
                "action": action,
                "job_id": previous.get("job_id"),
                "observed_scheduler_state": state,
            }
            outcomes.append(outcome)
            if action is None:
                pending.append((command, outcome))
        if active + sum(c["kind"] == "gpu" for c, _ in pending) > plan["max_concurrent_gpus"]:
            raise ValueError("Active plus requested campaign GPU jobs exceed the GPU limit")
        if wants_gpu and args.execute_gpu and not record["gpu_execution_enabled"]:
            record.update(
                gpu_execution_enabled=True,
                gpu_enabled_utc=now(),
                gpu_authorization_source="operator --execute-gpu for this campaign/config",
            )
        atomic_json(record_path, record)
        (root / "slurm").mkdir(exist_ok=True)
        for command, outcome in pending:
            entry = record["jobs"].setdefault(
                command["key"],
                {
                    "kind": command["kind"],
                    "identity": command["identity"],
                    "script": command["script"],
                    "attempts": [],
                },
            )
            attempt = {
                "status": "INTENT",
                "created_utc": now(),
                "phase_label": args.phase,
                "argv": command["argv"],
                "shell": command["shell"],
            }
            entry["attempts"].append(attempt)
            atomic_json(record_path, record)
            try:
                result = run(
                    command["argv"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )
                attempt.update(
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
                )
                match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
                if result.returncode != 0 or match is None:
                    raise RuntimeError(
                        "Uncertain sbatch submission; inspect retained stdout/stderr"
                    )
                attempt.update(status="SUBMITTED", job_id=match[1])
                outcome.update(action="SUBMITTED", job_id=match[1])
            except BaseException as exc:
                attempt.update(status="UNCERTAIN_SUBMISSION", error=f"{type(exc).__name__}: {exc}")
                atomic_json(record_path, record)
                raise
            atomic_json(record_path, record)
        return {**plan, "status": "SUBMITTED", "jobs": outcomes, "record": str(record_path)}


def main():
    args = parser().parse_args()
    print(json.dumps(execute(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
