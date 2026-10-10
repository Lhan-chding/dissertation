"""CPU-only evidence capture around externally authorized ENGINE maintenance.

This script never cancels, submits, activates source, or consumes a resume token.
Run snapshot before cancellation; run preserve-only after both jobs are terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

INCIDENT = "technical_incidents/compute_parallel_20261010"
TRACK = "engineering/engine/natural/continuous"
TERMINAL = {
    "CANCELLED",
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "PREEMPTED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
}
TEACHER_QOS = "soujanya-poria-startfund-2026-03"
COPY_PATHS = (
    "engineering/engine",
    "engineering/ENGINE_CONFIG.json",
    "manifests/ENGINE_SCHEDULE.json",
    "accounting/ENGINE.jsonl",
    "orchestration",
    "EXECUTION_FREEZE.json",
    "AMENDMENT.json",
    "QOS_SCOPE_REPAIR.json",
    "ENGINE_MEMORY_REPAIR.json",
    "ENGINE_STORAGE_REPAIR.json",
    "ENGINE_IO_REPAIR.json",
    "ENGINE_MULTIGPU_REPAIR.json",
    "COMMON_START.json",
    "COMMON_ZERO_LORA.json",
    "FORMAT_AND_BRIDGE_RECEIPT.json",
    "technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_ACTIVATED.json",
    "technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_PROCESS.json",
    "technical_incidents/baseline_parallel_20261010",
)


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bounded(root, name):
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "UNSAFE_PATH")
    path = root / name
    require(
        not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
        "PATH_ESCAPED_OR_SYMLINK",
    )
    return path


def save(path, value):
    """Exclusive durable publication; repeats may only assert identical content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            not path.is_symlink() and read(path) == value, "IMMUTABLE_RECEIPT_CHANGED:" + str(path)
        )
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return dict(
        command=args, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


def inventory(root):
    deployment = read(root / "SOURCE_DEPLOYMENT.json")
    for name, checksum in deployment["source_file_hashes"].items():
        require(sha(bounded(root, name)) == checksum, "SOURCE_CHANGED:" + name)
    return deployment


def job_targets(root):
    state = read(root / "orchestration/STATE.json")
    require(
        state["test_sealed"] is True
        and not any(
            t.get("attempts")
            for k, t in state["tasks"].items()
            if k not in {"COMMON_START", "ENGINE"}
        ),
        "DOWNSTREAM_ALREADY_STARTED",
    )
    require(
        not (root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json").exists(), "ENGINE_ALREADY_ACCEPTED"
    )
    require(
        not (root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists(), "COMPUTE_ALREADY_ACTIVATED"
    )
    attempt = state["tasks"]["ENGINE"]["attempts"][-1]
    handoff = read(root / "technical_incidents/baseline_parallel_20261010/HANDOFF_SUBMISSION.json")
    require(handoff["returncode"] == 0, "HANDOFF_SUBMISSION_UNKNOWN")
    cpu_id = handoff["stdout"].strip().split(";")[0]
    gpu_id = str(attempt["job_id"])
    require(cpu_id.isdigit() and gpu_id.isdigit() and cpu_id != gpu_id, "JOB_IDS_UNKNOWN")
    permission = read(root / "manifests/ALLOCATION_PERMISSION.json")
    return state, attempt, permission, cpu_id, gpu_id


def fields(receipt):
    require(receipt["returncode"] == 0, "SLURM_IDENTITY_UNKNOWN")
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", receipt["stdout"]))


def identities(root, *, run=command):
    _, attempt, permission, cpu, gpu = job_targets(root)
    result = {}
    for role, job in (("controller", cpu), ("gpu", gpu)):
        receipt = run(["scontrol", "show", "job", job, "--oneliner"])
        actual = fields(receipt)
        require(
            actual.get("JobId") == job
            and actual.get("UserId", "").split("(")[0] == permission["owner"]
            and actual.get("Account") == permission["account"],
            "JOB_OWNER_OR_ACCOUNT_CHANGED",
        )
        if role == "gpu":
            require(
                actual.get("JobName") == attempt["job_name"]
                and actual.get("Comment") == attempt["comment"]
                and actual.get("QOS") == TEACHER_QOS
                and actual.get("Command")
                == str(root / "orchestration/attempts" / (attempt["attempt_id"] + ".sh")),
                "GPU_NOT_REGISTERED_PROJECT_JOB",
            )
        else:
            require(
                actual.get("JobName") == "srf11-baseline-handoff"
                and actual.get("Comment") == "srf11-baseline-parallel-20261010"
                and actual.get("Command")
                == str(root / "technical_incidents/baseline_parallel_20261010/handoff.sbatch"),
                "CPU_NOT_BASELINE_HANDOFF",
            )
            require("ReqTRES" in actual and "AllocTRES" in actual, "CPU_RESOURCES_UNKNOWN")
            for tres in (actual["ReqTRES"], actual["AllocTRES"]):
                require(
                    all(
                        not key.startswith("gres/gpu") or value == "0"
                        for key, value in (
                            part.split("=", 1) for part in tres.split(",") if "=" in part
                        )
                    ),
                    "HANDOFF_CPU_HAS_GPU",
                )
        result[role] = dict(job_id=job, fields=actual, receipt=receipt)
    return result


def copy_files(root, destination, relatives, *, stable):
    result = {}
    for name in relatives:
        base = bounded(root, name)
        require(base.exists(), "PRESERVATION_SOURCE_MISSING:" + name)
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for source in paths:
            require(not source.is_symlink(), "PRESERVATION_SYMLINK")
            if source.is_dir():
                continue
            require(source.is_file(), "PRESERVATION_NOT_REGULAR")
            relative = str(source.relative_to(root))
            target = bounded(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and stable:
                require(sha(target) == sha(source), "PRESERVED_BYTES_CHANGED:" + relative)
            elif not target.exists():
                before = sha(source) if stable else None
                with (
                    source.open("rb") as incoming,
                    tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output,
                ):
                    temporary = Path(output.name)
                    for block in iter(lambda: incoming.read(8 << 20), b""):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    if stable:
                        require(
                            before == sha(source) == sha(temporary),
                            "LIVE_FILE_CHANGED_DURING_PRESERVATION:" + relative,
                        )
                    os.link(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            result[relative] = dict(sha256=sha(target), bytes=target.stat().st_size)
    return result


def verify_preservation(root, record, *, destination="evidence"):
    evidence = root / INCIDENT / destination
    for name, entry in record["artifact_hashes"].items():
        path = bounded(evidence, name)
        require(
            sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"],
            "PRESERVATION_CHANGED:" + name,
        )
    require(
        record["files"] == len(record["artifact_hashes"])
        and record["bytes"] == sum(v["bytes"] for v in record["artifact_hashes"].values()),
        "PRESERVATION_COUNTS_CHANGED",
    )


def snapshot(root, *, run=command):
    incident = root / INCIDENT
    existing = incident / "PRE_MAINTENANCE_SNAPSHOT.json"
    if existing.exists():
        receipt = read(existing)
        verify_preservation(root, receipt, destination="pre_maintenance")
        return receipt
    identity = identities(root, run=run)
    source = inventory(root / "code")
    require(
        source["source_commit"] == read(root / "ENGINE_MULTIGPU_REPAIR.json")["source_commit"],
        "OLD_PRODUCTION_CHANGED",
    )
    files = copy_files(root, incident / "pre_maintenance", COPY_PATHS, stable=False)
    result = dict(
        status="PRE_MAINTENANCE_SNAPSHOT",
        identities=identity,
        source=source,
        artifact_hashes=files,
        files=len(files),
        bytes=sum(v["bytes"] for v in files.values()),
        dynamic_files_may_span_observation_times=True,
        unrelated_jobs_modified=0,
    )
    save(existing, result)
    return result


def terminal_jobs(root, prior, *, run=command):
    _, _, permission, cpu, gpu = job_targets(root)
    require(
        prior["identities"]["controller"]["job_id"] == cpu
        and prior["identities"]["gpu"]["job_id"] == gpu,
        "MAINTENANCE_TARGET_CHANGED",
    )
    queue = run(["squeue", "--noheader", "--user", permission["owner"], "--format=%i"])
    require(
        queue["returncode"] == 0 and not {cpu, gpu}.intersection(queue["stdout"].split()),
        "JOBS_STILL_QUEUED_OR_UNKNOWN",
    )
    accounting = run(
        [
            "sacct",
            "-j",
            cpu + "," + gpu,
            "--format=JobIDRaw,State,ExitCode,User%100,Account%100,JobName%100,QOS%100",
            "-n",
            "-P",
        ]
    )
    require(accounting["returncode"] == 0, "ACCOUNTING_UNKNOWN")
    rows = {}
    for line in accounting["stdout"].splitlines():
        values = line.split("|")
        if values[0] in {cpu, gpu}:
            require(len(values) >= 7 and values[0] not in rows, "ACCOUNTING_ROW_INVALID")
            rows[values[0]] = values
    require(set(rows) == {cpu, gpu}, "TERMINAL_ACCOUNTING_MISSING")
    for role, job in (("controller", cpu), ("gpu", gpu)):
        row, expected = rows[job], prior["identities"][role]["fields"]
        require(
            row[1].split()[0].rstrip("+") in TERMINAL
            and row[3] == permission["owner"]
            and row[4] == permission["account"]
            and row[5] == expected["JobName"]
            and row[6] == expected["QOS"],
            "TERMINAL_IDENTITY_OR_STATE_CHANGED",
        )
    return dict(
        controller_terminal=True,
        gpu_terminal=True,
        controller_queue_empty=True,
        gpu_queue_empty=True,
        controller_job_id=cpu,
        gpu_job_id=gpu,
        controller_terminal_state=rows[cpu][1].split()[0].rstrip("+"),
        gpu_terminal_state=rows[gpu][1].split()[0].rstrip("+"),
        accounting=accounting,
        queue=queue,
        unrelated_jobs_modified=0,
    )


def verify_journal(evidence):
    from sr_f1.orchestration import digest

    previous, last = None, None
    for index, line in enumerate(
        (evidence / "orchestration/journal.jsonl").read_bytes().splitlines(keepends=True), 1
    ):
        require(line.endswith(b"\n"), "JOURNAL_TRUNCATED")
        row = json.loads(line)
        checksum = row.pop("sha256")
        require(
            digest(row) == checksum and row["previous"] == previous and row["sequence"] == index,
            "JOURNAL_CHAIN_CHANGED",
        )
        previous, last = checksum, row
    require(
        last is not None and last["state"] == read(evidence / "orchestration/STATE.json"),
        "JOURNAL_STATE_CHANGED",
    )
    return last["sequence"]


def cpu_review(root):
    import torch

    from mm_core.training import state_hash
    from sr_f1.contract import digest
    from sr_f1.engine import _read_state
    from sr_f1.json_protocol import AMENDMENT_ID, validate_record
    from sr_f1.training import CHECKPOINT_FIELDS, token_path_record

    require(not torch.cuda.is_initialized(), "CPU_AUDIT_CUDA_INITIALIZED")
    torch.set_num_threads(1)
    incident, track = root / INCIDENT, root / INCIDENT / "evidence" / TRACK
    latest = read(track / "checkpoints/LATEST.json")
    count = latest["step"]
    require(type(count) is int and 0 <= count <= 4, "CHECKPOINT_STEP_INVALID")
    raw = sorted((track / "rollouts").glob("*.json"))
    for path in raw:
        record = read(path)
        require(
            record["record_hash"] == digest({k: v for k, v in record.items() if k != "record_hash"})
            and validate_record(record, AMENDMENT_ID) is True,
            "RAW_RECORD_CHANGED",
        )
    previous = None
    for step in range(count + 1):
        state = _read_state(track, step)
        require(
            set(state) == CHECKPOINT_FIELDS and state["committed_logical_step"] == step,
            "CHECKPOINT_FIELDS_INVALID",
        )
        require(
            len(state["rng"]["cuda"]) == read(root / "ENGINE_MULTIGPU_REPAIR.json")["gpu_count"],
            "RNG_TOPOLOGY_CHANGED",
        )
        require(
            state["run_identity"]["training_identity"]["learning_rate"] == 1e-4,
            "LEARNING_RATE_CHANGED",
        )
        require(
            all(group["lr"] == 1e-4 for group in state["optimizer"]["param_groups"])
            and state["scheduler"]["_last_lr"] == [1e-4] * len(state["optimizer"]["param_groups"]),
            "OPTIMIZER_LR_CHANGED",
        )
        if step == 0:
            require(
                not state["optimizer"]["state"]
                and state_hash(state["parameters"])
                == read(root / "COMMON_START.json")["trainable_state_hash"],
                "COMMON_START_CHANGED",
            )
        else:
            metrics = read(track / "steps" / f"{step:02d}.json")
            records = [read(p) for p in sorted((track / "rollouts").glob(f"{step:02d}-*.json"))]
            require(
                len(records) == 128
                and digest([token_path_record(r) for r in records])
                == metrics["token_path_hash"]
                == state["token_path_hash"],
                "COMPLETED_RAW_TOKEN_PATH_CHANGED",
            )
            require(
                digest(metrics) == state["diagnostics_hash"]
                and state_hash(state["optimizer"]) == metrics["optimizer_state_hash"]
                and state_hash(previous["parameters"]) == metrics["parameter_hash_before"]
                and state_hash(state["parameters"]) == metrics["parameter_hash_after"],
                "UPDATE_STATE_BINDING_CHANGED",
            )
            for field in (
                "plan_id",
                "run_identity",
                "reference",
                "reference_hash",
                "input_stream_hash",
                "sampling_hash",
            ):
                require(
                    state_hash(previous[field]) == state_hash(state[field]),
                    "TRAINING_IDENTITY_CHANGED:" + field,
                )
            require(state["optimizer"]["state"], "ADAM_STATE_EMPTY")
            for value in state["optimizer"]["state"].values():
                require(
                    float(value["step"]) == step
                    and bool(torch.isfinite(value["exp_avg"]).all())
                    and bool(torch.isfinite(value["exp_avg_sq"]).all()),
                    "ADAM_INVALID",
                )
        previous = state
    require(not torch.cuda.is_initialized(), "CPU_AUDIT_CUDA_INITIALIZED")
    journal_lines = verify_journal(incident / "evidence")
    return dict(
        status="PASS_PRESERVED_FULL_STATE",
        committed_logical_step=count,
        checkpoint_state_hash=latest["state_hash"],
        checkpoint_file_sha256=latest["sha256"],
        raw_record_hashes_verified=True,
        optimizer_state_verified=True,
        rng_state_verified=True,
        learning_rate=1e-4,
        controller_identity_verified=True,
        journal_integrity=True,
        journal_lines=journal_lines,
        raw_file_hashes={p.name: sha(p) for p in raw},
        raw_rollouts=len(raw),
        model_calls=0,
        cuda_initialized=False,
    )


def preserve(root, *, run=command, review=cpu_review):
    incident = root / INCIDENT
    prior = read(incident / "PRE_MAINTENANCE_SNAPSHOT.json")
    verify_preservation(root, prior, destination="pre_maintenance")
    terminal = terminal_jobs(root, prior, run=run)
    source = inventory(root / "code")
    require(source == prior["source"], "PRODUCTION_CHANGED_AFTER_SNAPSHOT")
    from sr_f1.orchestration import Scheduler, process_lease

    with (
        process_lease(root.parent / ".sr_f1_project_controller.lock"),
        process_lease(root / "orchestration/controller.lock"),
    ):
        if not (incident / "RECONCILED.json").exists():
            scheduler = Scheduler(
                root / "config/SR_F1_1.json",
                root,
                code_root=root / "code",
                python=Path(sys.executable),
            )
            scheduler._load()
            scheduler._observe("ENGINE")
            require(
                scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
                .get("accounting", {})
                .get("terminal_state")
                in TERMINAL,
                "ENGINE_RECONCILIATION_NOT_TERMINAL",
            )
            scheduler._phase()
            scheduler._save("COMPUTE_PARALLEL_MAINTENANCE_RECONCILED_NO_SUBMISSION")
            save(
                incident / "RECONCILED.json",
                dict(
                    journal_sha256=sha(root / "orchestration/journal.jsonl"),
                    state_sha256=sha(root / "orchestration/STATE.json"),
                ),
            )
        reconciled = read(incident / "RECONCILED.json")
        require(
            sha(root / "orchestration/journal.jsonl") == reconciled["journal_sha256"]
            and sha(root / "orchestration/STATE.json") == reconciled["state_sha256"],
            "STATE_CHANGED_AFTER_RECONCILIATION",
        )
        if not (incident / "PRESERVATION.json").exists():
            files = copy_files(root, incident / "evidence", COPY_PATHS, stable=True)
            save(
                incident / "PRESERVATION.json",
                dict(
                    status="BYTE_VERIFIED_PRESERVED",
                    artifact_hashes=files,
                    files=len(files),
                    bytes=sum(v["bytes"] for v in files.values()),
                    source_commit=source["source_commit"],
                    unrelated_jobs_modified=0,
                ),
            )
        preservation = read(incident / "PRESERVATION.json")
        verify_preservation(root, preservation)
        saved_source = root / "code_before_engine_compute_parallel_20261010"
        copy_files(
            root / "code",
            saved_source,
            [*source["source_file_hashes"], "SOURCE_DEPLOYMENT.json"],
            stable=True,
        )
        require(inventory(saved_source) == source, "SOURCE_SNAPSHOT_CHANGED")
        if not (incident / "TERMINAL_JOBS.json").exists():
            save(incident / "TERMINAL_JOBS.json", terminal)
        else:
            old = read(incident / "TERMINAL_JOBS.json")
            require(
                all(old[k] == terminal[k] for k in terminal if k not in {"accounting", "queue"}),
                "TERMINAL_RECEIPT_CHANGED",
            )
        if not (incident / "CPU_STATE_REVIEW.json").exists():
            save(incident / "CPU_STATE_REVIEW.json", review(root))
        else:
            require(
                read(incident / "CPU_STATE_REVIEW.json")["status"] == "PASS_PRESERVED_FULL_STATE",
                "CPU_REVIEW_NOT_PASS",
            )
        candidate = root / "code_baseline_parallel_candidate_20261010"
        auth = read(
            root / "technical_incidents/baseline_parallel_20261010/HANDOFF_AUTHORIZATION.json"
        )
        require(
            sha(candidate / "SOURCE_DEPLOYMENT.json") == auth["candidate_source_deployment_sha256"],
            "OLD_BASELINE_CANDIDATE_CHANGED",
        )
        inventory(candidate)
        save(
            incident / "RECOVERY_REVIEW.json",
            dict(
                status="PASS_FULL_ENGINE_RESTART",
                science_attempts=0,
                test_sealed=True,
                full_engine_restart=True,
                no_partial_gradient_reuse=True,
                no_old_once_reuse=True,
                old_handoff_superseded=True,
                old_candidate_preserved=True,
                unrelated_jobs_modified=0,
                preserved_source_sha256=sha(saved_source / "SOURCE_DEPLOYMENT.json"),
            ),
        )
    return dict(
        status="COMPUTE_MAINTENANCE_PRESERVED_NOT_ACTIVATED",
        files=preservation["files"],
        checkpoint_step=read(incident / "CPU_STATE_REVIEW.json")["committed_logical_step"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("snapshot", "preserve-only"), required=True)
    args = parser.parse_args(argv)
    root = args.run_root
    require(root.is_absolute() and root.resolve(strict=True) == root, "CANONICAL_ROOT_REQUIRED")
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "CPU_SLURM_JOB_REQUIRED")
    own = fields(command(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "--oneliner"]))
    require("AllocTRES" in own, "CPU_AUDIT_RESOURCES_UNKNOWN")
    require(
        not any(
            k.startswith("gres/gpu") and v != "0"
            for k, v in (
                part.split("=", 1) for part in own.get("AllocTRES", "").split(",") if "=" in part
            )
        ),
        "CPU_AUDIT_REQUIRES_ZERO_GPU",
    )
    sys.path.insert(0, str(root / "code/src"))
    action = snapshot if args.phase == "snapshot" else preserve
    print(json.dumps(action(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
