"""Explicit recovery release gate; scientific reporting uses the frozen implementation.

Original cancelled allocations and claims remain unchanged.  This is a separate
entry point, not a replacement or monkeypatch of the original release gate.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from sr_f2 import reporting as frozen
from sr_f2.diagnostics import EXPERIMENT_ID
from sr_f2.orchestration import global_task_matrix, pool_specs, task_id, verify_task
from sr_f12.evaluation import durable_json, shard_lease
from sr_f12.orchestration import file_hash, read_json
from sr_f12.protocol import TEACHER_QOS, object_hash

from .orchestration import POOL_SPECS, RUN_GPU_LIMIT, Recovery, RecoveryQueue, Slurm, within


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def _command(recovery, task):
    return [
        recovery.plan["python"],
        str(recovery.code / "scripts/sr_f2/run_worker.py"),
        "--root",
        str(recovery.root),
        "--kind",
        task["kind"],
        "--model-id",
        task["model_id"],
    ]


def _original_allocations(recovery, slurm, queue, user):
    """Derive the five old identities from their original durable submission records."""
    expected = {}
    config = read_json(recovery.root / "orchestration/CONFIG.json")
    require(
        config.get("user") == user
        and config.get("code_root") == str(recovery.code)
        and config.get("root") == str(recovery.root)
        and config.get("global_tasks") == global_task_matrix()
        and config.get("global_task_matrix_sha256") == object_hash(global_task_matrix())
        and config.get("pools") == pool_specs()
        and config.get("strict_seed_waves") is False,
        "Original orchestration identity or matrix changed",
    )
    for spec in pool_specs():
        directory = recovery.root / "orchestration/pools" / spec["pool_id"]
        state, intent, plan = [
            read_json(directory / (name + ".json")) for name in ("STATE", "INTENT", "PLAN")
        ]
        require(
            state.get("pool_id") == spec["pool_id"]
            and state.get("gpu_count") == spec["gpu_count"]
            and state.get("registration_sha256") == recovery.plan["registration_sha256"]
            and intent.get("plan_sha256") == object_hash(plan)
            and intent.get("registration_sha256") == recovery.plan["registration_sha256"]
            and all(plan.get(k) == v for k, v in spec.items()),
            "Original GPU submission identity changed",
        )
        job = state.get("job_id")
        require(
            isinstance(job, str) and job.isdigit() and job not in expected,
            "Original GPU job identifier is missing or duplicated",
        )
        expected[job] = dict(
            job_id=job, user=user, account="rose", qos=TEACHER_QOS, name=intent["job_name"]
        )
    directory = recovery.root / "deployment/cpu_jobs/controller"
    state, intent = [read_json(directory / (name + ".json")) for name in ("STATE", "INTENT")]
    job = state.get("job_id")
    require(
        isinstance(job, str)
        and job.isdigit()
        and job not in expected
        and intent.get("kind") == "controller"
        and intent.get("user") == user
        and intent.get("root") == str(recovery.root)
        and intent.get("code_root") == str(recovery.code),
        "Original CPU controller identity changed",
    )
    expected[job] = dict(
        job_id=job,
        user=user,
        account="rose",
        qos="override-limits-but-killable",
        name=intent["job_name"],
    )
    declared = recovery.plan["old_cancelled_jobs"]
    require(
        len(declared) == 5
        and {r.get("job_id") for r in declared} == set(expected)
        and all(all(r.get(k) == v for k, v in expected[r["job_id"]].items()) for r in declared),
        "Recovery cancellation list differs from original jobs",
    )
    terminals = []
    for job, identity in expected.items():
        require(job not in queue, "Original experiment allocation remains in squeue")
        actual = slurm.terminal(job)
        require(
            isinstance(actual, dict)
            and actual.get("state") == "CANCELLED"
            and all(actual.get(k) == v for k, v in identity.items()),
            "Original cancellation accounting is missing or identity differs",
        )
        terminals.append(actual)
    return terminals


def _inherited_task(recovery, task, result, old_jobs):
    key = task_id(task)
    directory = recovery.root / "orchestration/queue/tasks" / key
    claim, state = [read_json(directory / (name + ".json")) for name in ("CLAIM", "STATE")]
    require(
        claim.get("task") == task
        and claim.get("job_id") in old_jobs
        and claim.get("registration_sha256") == recovery.plan["registration_sha256"]
        and state.get("status") == "COMPLETE"
        and state.get("result") == result
        and state.get("claim_sha256") == object_hash(claim)
        and all(
            state.get(k) == claim.get(k)
            for k in ("task", "pool_id", "device", "job_id", "registration_sha256", "claim_id")
        ),
        "Inherited task claim, success or receipt differs",
    )
    require(
        claim.get("pool_id") in {s["pool_id"] for s in pool_specs()},
        "Inherited task belongs to an unknown original pool",
    )
    pool = recovery.root / "orchestration/pools" / claim["pool_id"]
    pool_state, started = [read_json(pool / (name + ".json")) for name in ("STATE", "STARTED")]
    worker = pool / "workers" / key
    launch, exited = [
        read_json(worker / (name + ".json")) for name in ("LAUNCH_INTENT", "PROCESS_EXIT")
    ]
    require(
        pool_state.get("job_id") == started.get("job_id") == claim["job_id"]
        and claim["device"] in started.get("devices", [])
        and launch.get("command") == _command(recovery, task)
        and all(launch.get(k) == claim.get(k) for k in ("task", "device", "claim_id"))
        and exited.get("returncode") == 0
        and exited.get("pool_aborted") is not True,
        "Inherited task launch or successful process exit is unverified",
    )
    return dict(
        task=task,
        original_job_id=claim["job_id"],
        claim_sha256=object_hash(claim),
        launch_sha256=object_hash(launch),
        exit_sha256=object_hash(exited),
        receipt=result,
    )


def _recovery_pools(recovery, slurm, queue, user):
    base = recovery.directory / "pools"
    actual = {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()
    require(actual == set(POOL_SPECS), "Recovery pool coverage is missing or unregistered")
    pools, jobs = {}, set()
    for pool_id, count in POOL_SPECS.items():
        directory = base / pool_id
        state, intent, started, done = [
            read_json(directory / (name + ".json"))
            for name in ("STATE", "INTENT", "STARTED", "COMPLETE")
        ]
        job = state.get("job_id")
        require(
            isinstance(job, str) and job.isdigit() and job not in jobs,
            "Recovery job is unknown or duplicated",
        )
        jobs.add(job)
        require(
            state.get("status") == done.get("status") == "DRAINED"
            and state.get("gpu_count") == intent.get("gpu_count") == count
            and all(
                row.get("plan_sha256") == recovery.sha256 for row in (state, intent, started, done)
            )
            and state.get("pool_id") == intent.get("pool_id") == done.get("pool_id") == pool_id
            and done.get("job_id") == started.get("job_id") == job
            and len(started.get("devices", [])) == count
            and len(set(started["devices"])) == count
            and intent.get("user") == user
            and not (directory / "FAILURE.json").exists(),
            "Recovery pool is incomplete, failed, unknown or changed",
        )
        script = within(
            recovery.directory, str(Path(intent["script"]).relative_to(recovery.directory))
        )
        require(
            script == directory / "run.sbatch" and file_hash(script) == intent["script_sha256"],
            "Recovery batch script differs",
        )
        allocation = read_json(directory / "ALLOCATION_VERIFIED.json")["fields"]
        from .orchestration import verify_allocation

        verify_allocation(allocation, dict(intent, job_id=job), user, held=True)
        release_intent = read_json(directory / "RELEASE_INTENT.json")
        release = read_json(directory / "RELEASE.json")
        require(
            release_intent.get("job_id") == job and release.get("returncode") == 0,
            "Recovery allocation release is not verified",
        )
        require(job not in queue, "Recovery experiment allocation remains in squeue")
        terminal = slurm.terminal(job)
        expected = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user=user,
            account="rose",
            qos=TEACHER_QOS,
            name=intent["job_name"],
        )
        require(
            isinstance(terminal, dict) and all(terminal.get(k) == v for k, v in expected.items()),
            "Recovery sacct is missing, unsuccessful or has a different identity",
        )
        pools[pool_id] = dict(
            job_id=job,
            devices=started["devices"],
            gpu_count=count,
            accounting=terminal,
            intent_sha256=object_hash(intent),
            completion_sha256=object_hash(done),
        )
    require(
        sum(p["gpu_count"] for p in pools.values()) == RUN_GPU_LIMIT == 5,
        "Recovery five-GPU bound changed",
    )
    return pools


def _recovered_task(recovery, task, state, result, pools):
    key, pool_id = task_id(task), state["pool_id"]
    require(
        state.get("status") == "COMPLETE"
        and state.get("result") == result
        and state["job_id"] == pools[pool_id]["job_id"]
        and state["device"] in pools[pool_id]["devices"],
        "Recovery task is incomplete, failed or bound to a different allocation",
    )
    directory = recovery.directory / "pools" / pool_id / "workers" / key
    launch, exited = [
        read_json(directory / (name + ".json")) for name in ("LAUNCH_INTENT", "PROCESS_EXIT")
    ]
    require(
        launch.get("command") == _command(recovery, task)
        and launch.get("plan_sha256") == recovery.sha256
        and launch.get("audit_entry_sha256") == object_hash(recovery.allowed[key])
        and all(launch.get(k) == state.get(k) for k in ("task", "device", "claim_id"))
        and exited.get("returncode") == 0
        and exited.get("pool_aborted") is not True,
        "Recovery task launch or successful process exit is unverified",
    )
    return dict(
        task=task,
        job_id=state["job_id"],
        claim_sha256=state["claim_sha256"],
        launch_sha256=object_hash(launch),
        exit_sha256=object_hash(exited),
        receipt=result,
    )


def verify_accounting(recovery, task):
    """Match attempts and generated-token rows to raw metadata without scoring TEST."""
    kind, model = task["kind"], task["model_id"]
    training = kind in {"train", "pilot"}
    accounting_id = model if training else kind + "-" + model
    ledger = recovery.root / "accounting" / (accounting_id + ".jsonl")
    before = file_hash(ledger)
    attempts, tokens = Counter(), Counter()
    with ledger.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(
                row.get("experiment_id") == EXPERIMENT_ID
                and row.get("task_id") == accounting_id
                and type(row.get("count")) is int
                and row["count"] >= 0,
                "Accounting row identity or count differs",
            )
            meta = row.get("metadata", {})
            if row.get("kind") == "completion_attempts":
                require(
                    type(meta.get("group_seed")) is int and 1 <= row["count"] <= 8,
                    "Unknown accounting generation group",
                )
                attempts[(meta["group_seed"], row["count"])] += 1
            elif row.get("kind") == "generated_tokens":
                require(
                    type(meta.get("group_seed")) is int and type(meta.get("row_index")) is int,
                    "Unknown accounting token group",
                )
                tokens[(meta["group_seed"], meta["row_index"], row["count"])] += 1
    if training:
        base = recovery.root / ("runs" if kind == "train" else "technical") / model
        paths = sorted(
            p
            for p in (base / "rollouts").glob("*.json")
            if re.fullmatch(r"\d{2}-\d{2}\.json", p.name)
        )
        require(
            len(paths) == (96 if kind == "train" else 8) * 16,
            "Final training raw rollout coverage differs",
        )
        paths += sorted(base.glob("monitor/step-*/groups/*.json"))
    else:
        base = (
            recovery.root
            / {"history": "history", "diagnostic": "diagnostics", "endpoint": "sealed/evaluation"}[
                kind
            ]
            / model
        )
        paths = sorted(
            base.glob("step-*/groups/*.json") if kind == "history" else base.glob("groups/*.json")
        )
    raw_attempts, raw_tokens = Counter(), Counter()
    require(bool(paths), "Final accounting has no matching raw groups")
    for path in paths:
        rows = read_json(path)["records"]
        require(1 <= len(rows) <= 8, "Invalid raw accounting group size")
        seeds = {row.get("group_seed") for row in rows}
        require(
            len(seeds) == 1 and type(next(iter(seeds))) is int,
            "Raw group seed is missing or inconsistent",
        )
        seed = next(iter(seeds))
        raw_attempts[(seed, len(rows))] += 1
        for index, row in enumerate(rows):
            require(
                row.get("group_row_index") == index
                and row.get("group_size") == len(rows)
                and isinstance(row.get("tokens"), list)
                and len(row["tokens"]) > 0,
                "Raw accounting row index, size or tokens differ",
            )
            raw_tokens[(seed, index, len(row["tokens"]))] += 1
    require(
        attempts == raw_attempts and tokens == raw_tokens,
        "Unresolved accounting gap or repeated generation attempt; release blocked",
    )
    require(file_hash(ledger) == before, "Accounting changed during release verification")
    return dict(
        task=task,
        ledger_path=str(ledger.relative_to(recovery.root)),
        ledger_sha256=before,
        groups=sum(raw_attempts.values()),
        records=sum(raw_tokens.values()),
        unresolved_groups=0,
        repeated_attempts=0,
    )


def verify_recovery_release_gate(plan_path, *, slurm=None):
    recovery = Recovery(plan_path)
    recovery.verify()
    required_source = {"src/sr_f2_recovery/reporting.py", "scripts/sr_f2/release_recovery.py"}
    require(
        required_source <= set(recovery.plan["extension_manifest"]),
        "Recovery manifest omits release executable source",
    )
    require(
        Path(__file__).resolve() == recovery.extension / "src/sr_f2_recovery/reporting.py"
        and Path(frozen.__file__).resolve() == recovery.code / "src/sr_f2/reporting.py",
        "Loaded release modules differ from bound source roots",
    )
    require(
        not any(
            row["status"] == "BLOCKED_INCOMPLETE_GENERATION" for row in recovery.allowed.values()
        ),
        "Unresolved generation gaps block scientific release",
    )
    matrix = global_task_matrix()
    require(len(matrix) == 53 and recovery.matrix == matrix, "Original 53-task matrix differs")
    inherited = {
        key for key, row in recovery.allowed.items() if row["status"] == "INHERITED_COMPLETE"
    }
    require(len(inherited) == 35, "This recovery must preserve exactly35 inherited completions")
    states = RecoveryQueue(recovery).states()
    require(
        set(states) == set(recovery.tasks) - inherited
        and all(row["status"] == "COMPLETE" for row in states.values()),
        "Recovery task coverage is missing, failed or unknown",
    )
    user = read_json(recovery.root / "orchestration/CONFIG.json")["user"]
    slurm = slurm or Slurm()
    live = slurm.queue(user)
    old = _original_allocations(recovery, slurm, live, user)
    pools = _recovery_pools(recovery, slurm, live, user)
    all_jobs = {row["job_id"] for row in old} | {row["job_id"] for row in pools.values()}
    require(len(all_jobs) == 8, "Original and recovery allocation identities overlap")
    for fields in live.values():
        command = fields.get("Command", "")
        require(
            not command.startswith(str(recovery.root) + "/")
            and not command.startswith(str(recovery.directory) + "/"),
            "An unregistered live experiment allocation blocks release",
        )
    inherited_rows, recovered_rows, accounting, receipts = [], [], [], []
    for task in matrix:
        key = task_id(task)
        result = verify_task(recovery.root, task, recovery.plan["registration_sha256"])
        receipts.append(result)
        if key in inherited:
            inherited_rows.append(_inherited_task(recovery, task, result, all_jobs))
        else:
            recovered_rows.append(_recovered_task(recovery, task, states[key], result, pools))
        accounting.append(verify_accounting(recovery, task))
    expected_workers = {
        pool: {key for key, state in states.items() if state["pool_id"] == pool}
        for pool in POOL_SPECS
    }
    for pool, expected in expected_workers.items():
        actual = {
            p.name
            for p in (recovery.directory / "pools" / pool / "workers").glob("*")
            if p.is_dir()
        }
        require(actual == expected, "Unclaimed or duplicate recovery worker attempt exists")
    new_models = {row["model_id"] for row in frozen.scientific_matrix()}
    for kind in ("train", "endpoint"):
        require(
            {t["model_id"] for t in matrix if t["kind"] == kind} == new_models
            and len(new_models) == 12,
            "All twelve training and endpoint paths are required",
        )
    recovery.verify()
    require(not all_jobs.intersection(slurm.queue(user)), "Experiment job reappeared in squeue")
    return dict(
        status="AUTHORIZED_AFTER_ALL_GPU_TASKS_VERIFIED",
        experiment_id=frozen.AMENDMENT_ID,
        registration_sha256=recovery.plan["registration_sha256"],
        tasks=receipts,
        new_endpoint_records=183600,
        live_queue_and_accounting_verified=True,
        scientific_confirmation_separate_from_delivery=True,
        recovery=dict(
            recovery_id=recovery.plan["recovery_id"],
            plan_sha256=recovery.sha256,
            audit_sha256=recovery.plan["audit_sha256"],
            extension_manifest_sha256=object_hash(recovery.plan["extension_manifest"]),
            max_run_gpus=5,
            cancelled_allocations=old,
            pools=pools,
            inherited_tasks=inherited_rows,
            recovered_tasks=recovered_rows,
            generation_accounting=accounting,
        ),
    )


def release_and_report(plan_path, *, slurm=None):
    """Original scientific functions and publication order, with an explicit recovery gate."""
    recovery = Recovery(plan_path)
    root = recovery.root
    with shard_lease(root / "released"):
        cohort = frozen.cohort_verification(root)
        require(
            cohort["status"] == "PASS"
            and cohort["raw_evidence_status"] == "RAW_AND_GOLD_REVERIFIED",
            "Requested122 identity and complete raw/gold delivery are required",
        )
        authorization = verify_recovery_release_gate(plan_path, slurm=slurm)
        authorization = dict(
            authorization, requested_chartqa_cohort_sha256=cohort["receipt_sha256"]
        )
        durable_json(root / "released/AUTHORIZATION.json", authorization)
        registration = read_json(root / "REGISTRATION.json")
        tasks, chart = frozen.load_tasks(root), frozen.load_verified_manifest(root)
        new_rows = frozen.score_new_endpoints(root, registration, tasks, chart)
        old_rows, old_receipt = frozen.load_old_endpoints(root, tasks, chart)
        new_hash = frozen._write_jsonl_once(root / "released/SCORES.jsonl", new_rows)
        old_hash = frozen._write_jsonl_once(root / "released/OLD_SCORES.jsonl", old_rows)
        trajectories, states, usage, steps = frozen.load_trajectories(root)
        diag_rows, diagnostics = frozen.diagnostic_results(root, registration, tasks, old_rows)
        diagnostic_hash = frozen._write_jsonl_once(
            root / "released/DIAGNOSTIC_SCORES.jsonl", diag_rows
        )
        main = frozen.report_main_contrasts(
            new_rows, old_rows, trigger_states=states, semantic_group_usage=usage
        )
        incidents = [
            dict(path=str(p.resolve(strict=True).relative_to(root)), sha256=file_hash(p))
            for p in sorted((root / "technical_incidents").glob("*.json"))
        ]
        results = dict(
            status="COMPLETE_COVERAGE_VERIFIED",
            experiment_id=frozen.AMENDMENT_ID,
            experiment_complete=True,
            authorization_sha256=object_hash(authorization),
            source_manifest_sha256=registration["source_manifest_sha256"],
            coverage=dict(
                new_endpoint_records=len(new_rows),
                old_endpoint_records=len(old_rows),
                diagnostics=len(diag_rows),
                trajectory_cells=len(trajectories),
                historical_monitor_records=79872,
                new_monitor_records=79872,
            ),
            score_files=dict(
                new_sha256=new_hash, old_sha256=old_hash, diagnostic_sha256=diagnostic_hash
            ),
            prior_rescore=old_receipt,
            requested_chartqa_cohort=cohort,
            main=main,
            endpoint_metrics=frozen.summarize_rows([*new_rows, *old_rows]),
            diagnostics=diagnostics,
            triggers=frozen.trigger_summary(states, old_rows),
            secondary=frozen.secondary_predictions(new_rows, old_rows, states, steps),
            semantic_group_usage=usage,
            advantage_energy=frozen.advantage_energy_report(root, steps, registration),
            trajectories=trajectories,
            technical_incidents=incidents,
            scientific_confirmation_separate_from_delivery=True,
            technical_recovery=authorization["recovery"],
        )
        require(
            frozen.cohort_verification(root) == cohort,
            "Requested122 verification or delivery changed during report construction",
        )
        durable_json(root / "released/results.json", results)
        report = frozen.render_report(results)
        report += (
            "\n## 技术恢复记录\n\n原五个作业的取消记录完整保留。本次恢复最多使用5张GPU。"
            "恢复任务与继承任务均经独立解封验收。完整来源见AUTHORIZATION.json的recovery字段。\n"
        )
        frozen.publish_bytes(root / "released/REPORT.md", report.encode())
        figure = root / "released/trajectories.png"
        fd, temporary = tempfile.mkstemp(prefix=".trajectories.", suffix=".png", dir=figure.parent)
        os.close(fd)
        try:
            frozen.plot_trajectories(trajectories, temporary)
            frozen.publish_bytes(figure, Path(temporary).read_bytes())
        finally:
            Path(temporary).unlink()
        delivery = dict(
            status="COMPLETE_COVERAGE_VERIFIED",
            experiment_complete=True,
            authorization_sha256=object_hash(authorization),
            coverage=results["coverage"],
            recovery_plan_sha256=recovery.sha256,
            files=[
                dict(
                    path=str(p.resolve(strict=True).relative_to(root)),
                    sha256=file_hash(p),
                    bytes=p.stat().st_size,
                )
                for p in (
                    root / "released" / name
                    for name in (
                        "AUTHORIZATION.json",
                        "SCORES.jsonl",
                        "OLD_SCORES.jsonl",
                        "DIAGNOSTIC_SCORES.jsonl",
                        "results.json",
                        "REPORT.md",
                        "trajectories.png",
                        "../prior_audit/chartqa122/COHORT_VERIFICATION.json",
                        "../prior_audit/chartqa122/COHORT_ROWS.jsonl",
                        "../prior_audit/chartqa122/SOURCE_MANIFEST.json",
                    )
                )
            ],
        )
        durable_json(root / "released/COMPLETE.json", delivery)
        return delivery
