"""Explicit composite release for original, checkpoint-resumed and replayed work.

The original and first-recovery gates stay unchanged. Scientific calculations
are delegated to the original frozen reporting functions only after this gate.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from sr_f2 import reporting as frozen
from sr_f2.diagnostics import EXPERIMENT_ID
from sr_f2.orchestration import global_task_matrix, task_id, verify_task
from sr_f2_recovery import reporting as parent_reporting
from sr_f2_recovery.orchestration import RecoveryQueue, Slurm
from sr_f12.evaluation import durable_json, shard_lease
from sr_f12.orchestration import file_hash, gpu_count, memory_gib, read_json
from sr_f12.protocol import TEACHER_QOS, object_hash

from .replay import ReplayPlan, verify_endpoint_task, verify_replay_receipt, within


def require(value, message):
    if not value:
        raise PermissionError(message)


def _workers_exact(directory, states, pools):
    for pool in pools:
        expected = {key for key, row in states.items() if row["pool_id"] == pool}
        actual = {p.name for p in (directory / "pools" / pool / "workers").glob("*") if p.is_dir()}
        require(actual == expected, "Unclaimed or duplicate worker attempt blocks release")


def _no_live_experiment_jobs(queue, jobs, root):
    require(not jobs.intersection(queue), "Experiment allocations remain in squeue")
    for fields in queue.values():
        command = fields.get("Command", "")
        require(
            isinstance(command, str) and not command.startswith(str(root) + "/"),
            "Unregistered live experiment allocation blocks release",
        )


def _replay_pools(plan, slurm, queue, user, parent_pools):
    from .orchestration import POOL_SPECS, Controller, ReplayQueue, verify_allocation

    base = plan.directory / "pools"
    actual = {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()
    require(actual == set(POOL_SPECS), "Replacement pool coverage is missing or unregistered")
    states = ReplayQueue(plan).states()
    controller = Controller(plan, slurm=slurm, user=user)
    pools, jobs = {}, set()
    for pool_id, spec in POOL_SPECS.items():
        directory, count = base / pool_id, spec["gpu_count"]
        if (directory / "NOT_NEEDED.json").exists():
            skip = read_json(directory / "NOT_NEEDED.json")
            require(
                skip.get("pool_id") == pool_id
                and skip.get("plan_sha256") == plan.sha256
                and skip.get("reason") == "ALL_ENDPOINTS_ALREADY_CLAIMED"
                and set(skip.get("task_ids", [])) == set(plan.tasks)
                and len(skip["task_ids"]) == 5
                and skip.get("task_claims_sha256")
                == object_hash({key: row["claim_sha256"] for key, row in states.items()})
                and not any(row["pool_id"] == pool_id for row in states.values())
                and {p.name for p in directory.iterdir()} == {"NOT_NEEDED.json"},
                "Unused replacement pool has an unknown submission or invalid claim proof",
            )
            continue
        state, intent, started, done = [
            read_json(directory / (name + ".json"))
            for name in ("STATE", "INTENT", "STARTED", "COMPLETE")
        ]
        job = state.get("job_id")
        require(
            isinstance(job, str) and job.isdigit() and job not in jobs,
            "Replacement allocation is unknown or duplicated",
        )
        jobs.add(job)
        require(
            state.get("status") == done.get("status") == "DRAINED"
            and state.get("gpu_count") == intent.get("gpu_count") == count
            and all(row.get("plan_sha256") == plan.sha256 for row in (state, intent, started, done))
            and state.get("pool_id") == intent.get("pool_id") == done.get("pool_id") == pool_id
            and started.get("job_id") == done.get("job_id") == job
            and len(started.get("devices", [])) == len(set(started.get("devices", []))) == count
            and intent.get("user") == user
            and not (directory / "FAILURE.json").exists(),
            "Replacement pool is incomplete, failed or changed",
        )
        parent = parent_pools[spec["parent_pool_id"]]
        _, fresh_parent = controller._parent(pool_id, queue)
        proof = read_json(directory / "PARENT_TERMINAL.json")
        require(
            fresh_parent is not None
            and proof == fresh_parent
            and intent.get("parent_pool_id") == spec["parent_pool_id"]
            and intent.get("parent_job_id") == parent["job_id"]
            and intent.get("parent_terminal_sha256")
            == done.get("parent_terminal_sha256")
            == object_hash(proof),
            "Replacement did not inherit its verified drained parent",
        )
        capacity_hashes = {}
        for name in ("CAPACITY_BEFORE_SUBMIT", "CAPACITY_BEFORE_RELEASE", "CAPACITY_AT_START"):
            record = read_json(directory / (name + ".json"))
            proposed = count if name == "CAPACITY_BEFORE_SUBMIT" else 0
            values = record.get("capacity", {})
            require(
                record.get("plan_sha256") == plan.sha256
                and record.get("proposed_gpus") == proposed
                and all(
                    type(values.get(k)) is int and values[k] >= 0
                    for k in ("teacher_gpus", "user_submissions", "run_gpus")
                )
                and values["teacher_gpus"] + proposed <= 8
                and values["run_gpus"] + proposed <= 5
                and values["user_submissions"] + (1 if proposed else 0) <= 5
                and record.get("parent_terminal_sha256", {}).get(pool_id) == object_hash(proof),
                "Replacement capacity snapshot exceeds limits or lacks its drained parent",
            )
            for other, digest in record["parent_terminal_sha256"].items():
                require(other in POOL_SPECS, "Capacity snapshot contains an unknown parent")
                _, evidence = controller._parent(other, queue)
                require(
                    evidence is not None and object_hash(evidence) == digest,
                    "Capacity parent terminal evidence changed",
                )
            capacity_hashes[name] = object_hash(record)
        script = within(plan.directory, str(Path(intent["script"]).relative_to(plan.directory)))
        require(
            script == directory / "run.sbatch" and file_hash(script) == intent["script_sha256"],
            "Replacement batch script differs",
        )
        verify_allocation(
            read_json(directory / "ALLOCATION_VERIFIED.json")["fields"],
            intent,
            job,
            user,
            held=True,
        )
        require(
            read_json(directory / "RELEASE_INTENT.json").get("job_id") == job
            and read_json(directory / "RELEASE.json").get("returncode") == 0,
            "Replacement allocation release is unverified",
        )
        require(job not in queue, "Replacement allocation remains in squeue")
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
            "Replacement sacct is missing, unsuccessful or has a different identity",
        )
        pools[pool_id] = dict(
            job_id=job,
            devices=started["devices"],
            gpu_count=count,
            accounting=terminal,
            parent_pool_id=spec["parent_pool_id"],
            parent_terminal_sha256=object_hash(proof),
            capacity_receipts=capacity_hashes,
            intent_sha256=object_hash(intent),
            completion_sha256=object_hash(done),
        )
    require(
        1 <= len(pools) <= 3 and sum(row["gpu_count"] for row in pools.values()) <= 5,
        "Replacement allocations exceed the five-GPU cap",
    )
    return pools


def _replayed_task(plan, task, state, result, pools):
    key, pool = task_id(task), state["pool_id"]
    require(
        pool in pools
        and state.get("status") == "COMPLETE"
        and state.get("result") == result
        and state.get("parent_plan_sha256") == plan.parent.sha256
        and state["job_id"] == pools[pool]["job_id"]
        and state["device"] in pools[pool]["devices"],
        "Replay task is incomplete or belongs to another allocation",
    )
    directory = plan.directory / "pools" / pool / "workers" / key
    launch, exited = [
        read_json(directory / (name + ".json")) for name in ("LAUNCH_INTENT", "PROCESS_EXIT")
    ]
    command = [
        plan.parent.plan["python"],
        str(plan.extension / "scripts/sr_f2/replay_endpoint.py"),
        "--plan",
        str(plan.path),
        "--task-id",
        key,
    ]
    require(
        launch.get("command") == command
        and launch.get("plan_sha256") == plan.sha256
        and launch.get("task_definition_sha256") == object_hash(plan.tasks[key])
        and all(launch.get(k) == state.get(k) for k in ("task", "device", "claim_id"))
        and exited.get("returncode") == 0
        and exited.get("pool_aborted") is not True,
        "Replay endpoint launch or successful process exit is unverified",
    )
    return dict(
        task=task,
        job_id=state["job_id"],
        claim_sha256=state["claim_sha256"],
        launch_sha256=object_hash(launch),
        exit_sha256=object_hash(exited),
        receipt=result,
    )


def _replay_accounting(plan, task):
    """Reconcile physical replay separately from canonical scientific output, without scoring."""
    proof = verify_replay_receipt(plan, task)
    spec = plan.tasks[task_id(task)]
    ledger = within(plan.root, spec["original_ledger_path"])
    before = file_hash(ledger)
    attempts, tokens = Counter(), Counter()
    with ledger.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(
                row.get("experiment_id") == EXPERIMENT_ID
                and row.get("task_id") == "endpoint-" + task["model_id"]
                and type(row.get("count")) is int
                and row["count"] >= 0,
                "Original endpoint accounting identity or count differs",
            )
            meta = row.get("metadata", {})
            require(
                type(meta.get("group_seed")) is int, "Original endpoint accounting group is unknown"
            )
            if row.get("kind") == "completion_attempts":
                require(1 <= row["count"] <= 8, "Original generation reservation is invalid")
                attempts[(meta["group_seed"], row["count"])] += 1
            elif row.get("kind") == "generated_tokens":
                require(type(meta.get("row_index")) is int, "Original token row index is unknown")
                tokens[(meta["group_seed"], meta["row_index"], row["count"])] += 1
            else:
                raise PermissionError("Unexpected endpoint cost event")
    raw_attempts, raw_tokens = Counter(), Counter()
    paths = sorted((plan.root / "sealed/evaluation" / task["model_id"] / "groups").glob("*.json"))
    require(bool(paths), "Replay endpoint has no canonical groups")
    for path in paths:
        rows = read_json(path)["records"]
        seeds = {row.get("group_seed") for row in rows}
        require(
            1 <= len(rows) <= 8 and len(seeds) == 1 and type(next(iter(seeds))) is int,
            "Canonical accounting group identity differs",
        )
        seed = next(iter(seeds))
        raw_attempts[(seed, len(rows))] += 1
        for index, row in enumerate(rows):
            require(
                row.get("group_row_index") == index
                and row.get("group_size") == len(rows)
                and isinstance(row.get("tokens"), list)
                and bool(row["tokens"]),
                "Canonical accounting token metadata differs",
            )
            raw_tokens[(seed, index, len(row["tokens"]))] += 1
    staged = within(plan.directory, proof["staged_path"])
    replay_rows = read_json(staged)["records"]
    preserved = spec["preserved_count"]
    require(len(replay_rows) == 4 and preserved in (0, 3), "Authorized replay row budget differs")
    suffix = Counter(
        (spec["group_seed"], i, len(row["tokens"]))
        for i, row in enumerate(replay_rows)
        if i >= preserved
    )
    require(
        attempts == raw_attempts and tokens + suffix == raw_tokens,
        "Unresolved original/replay accounting gap or repeated generation",
    )
    require(file_hash(ledger) == before, "Original accounting changed during release")
    cost = proof["accounting"]
    require(
        cost["physical_attempt_groups"] == 1
        and cost["physical_attempt_rows"] == 4
        and cost["canonical_new_rows"] == 4 - preserved
        and cost["duplicate_verification_rows"] == preserved,
        "Authorized replay accounting budget differs",
    )
    return dict(
        task=task,
        authorized_replay=True,
        original_ledger_sha256=before,
        replay_provenance_sha256=object_hash(proof),
        replay_accounting=cost,
        replay_timing=proof["timing"],
        groups=sum(raw_attempts.values()),
        records=sum(raw_tokens.values()),
        physical_replay_records=4,
        canonical_added_records=4 - preserved,
        overlap_verification_records=preserved,
        unresolved_groups=0,
    )


def _dispatcher(plan, slurm, queue, user, states):
    from .orchestration import verify_dispatcher_allocation

    directory = plan.directory / "dispatcher"
    intent, state, started, done = [
        read_json(directory / (name + ".json"))
        for name in ("INTENT", "STATE", "STARTED", "COMPLETE")
    ]
    job = state.get("job_id")
    require(
        isinstance(job, str)
        and job.isdigit()
        and job not in queue
        and state.get("status") == done.get("status") == "COMPLETE"
        and done.get("reason") == "ALL_ENDPOINTS_CLAIMED"
        and intent.get("plan_sha256")
        == started.get("plan_sha256")
        == done.get("plan_sha256")
        == plan.sha256
        and started.get("job_id") == done.get("job_id") == job
        and started.get("intent_sha256") == object_hash(intent)
        and intent.get("user") == user
        and intent.get("qos") == "override-limits-but-killable"
        and done.get("task_claims_sha256")
        == object_hash({key: row["claim_sha256"] for key, row in states.items()})
        and not (directory / "FAILURE.json").exists(),
        "CPU dispatcher is missing, active, failed or has unknown claim coverage",
    )
    script = within(plan.directory, str(Path(intent["script"]).relative_to(plan.directory)))
    require(
        script.parent == directory and file_hash(script) == intent["script_sha256"],
        "CPU dispatcher script changed",
    )
    allocation = read_json(directory / "ALLOCATION_AT_START.json")
    verify_dispatcher_allocation(allocation["fields"], intent, job, user)
    held = read_json(directory / "ALLOCATION_VERIFIED.json")["fields"]
    held_identity = dict(
        JobId=job,
        Account="rose",
        Partition="cluster02",
        QOS="override-limits-but-killable",
        ReqNodeList="cpu-2",
        NumCPUs="4",
        TimeLimit="3-00:00:00",
        Requeue="0",
        JobState="PENDING",
        Reason="JobHeldUser",
        Command=intent["script"],
        Comment=intent["comment"],
        JobName=intent["job_name"],
    )
    require(
        all(held.get(k) == v for k, v in held_identity.items())
        and held.get("UserId", "").split("(")[0] == user
        and held.get("NumNodes") in {"1", "1-1"}
        and bool(held.get("ReqTRES"))
        and gpu_count(held["ReqTRES"]) == 0
        and memory_gib(held["ReqTRES"]) == 12,
        "CPU held allocation resource or submission identity differs",
    )
    submission = read_json(directory / "SUBMISSION.json")
    require(
        submission.get("returncode") == 0
        and submission.get("stdout", "").strip().split(";")[0] == job,
        "CPU dispatcher submission identity is unverified",
    )
    require(
        read_json(directory / "RELEASE_INTENT.json").get("job_id") == job
        and read_json(directory / "RELEASE.json").get("returncode") == 0,
        "CPU dispatcher release is unverified",
    )
    terminal = slurm.terminal(job)
    expected = dict(
        job_id=job,
        state="COMPLETED",
        exit_code="0:0",
        user=user,
        account="rose",
        qos=intent["qos"],
        name=intent["job_name"],
    )
    require(
        isinstance(terminal, dict) and all(terminal.get(k) == v for k, v in expected.items()),
        "CPU dispatcher accounting is missing or unsuccessful",
    )
    return dict(
        job_id=job,
        accounting=terminal,
        intent_sha256=object_hash(intent),
        completion_sha256=object_hash(done),
    )


def verify_composite_release_gate(plan_path, *, slurm=None):
    plan = ReplayPlan(plan_path)
    plan.verify()
    parent = plan.parent
    require(
        not any(
            path.exists()
            for path in (
                plan.directory / "STOP.json",
                parent.directory / "STOP.json",
                plan.root / "STOP",
            )
        ),
        "A preserved stop or technical failure blocks composite release",
    )
    required = {"src/sr_f2_replay/reporting.py", "scripts/sr_f2/release_replay.py"}
    manifest = read_json(within(plan.directory, plan.plan["extension_manifest"]))
    require(required <= set(manifest["files"]), "Replay manifest omits release executable source")
    require(
        Path(__file__).resolve() == plan.extension / "src/sr_f2_replay/reporting.py"
        and Path(parent_reporting.__file__).resolve()
        == parent.extension / "src/sr_f2_recovery/reporting.py"
        and Path(frozen.__file__).resolve() == plan.code / "src/sr_f2/reporting.py",
        "Loaded composite release modules differ from bound source roots",
    )
    matrix = global_task_matrix()
    require(
        len(matrix) == 53 and parent.matrix == matrix,
        "Composite release requires exactly the original 53 tasks",
    )
    inherited = {
        key for key, row in parent.allowed.items() if row["status"] == "INHERITED_COMPLETE"
    }
    blocked = {
        key
        for key, row in parent.allowed.items()
        if row["status"] == "BLOCKED_INCOMPLETE_GENERATION"
    }
    resumed = set(parent.tasks) - inherited - blocked
    require(
        len(inherited) == 35
        and len(resumed) == 13
        and len(blocked) == 5
        and set(plan.tasks) == blocked
        and all(parent.tasks[key]["kind"] == "endpoint" for key in blocked),
        "Composite coverage must be35 inherited,13 resumed and5 authorized endpoints",
    )
    parent_states = RecoveryQueue(parent).states()
    require(
        set(parent_states) == resumed
        and all(row["status"] == "COMPLETE" for row in parent_states.values()),
        "First recovery has missing, failed, active or unknown tasks",
    )
    from .orchestration import ReplayQueue, verify_controller_source

    verify_controller_source(plan)
    replay_states = ReplayQueue(plan).states()
    require(
        set(replay_states) == blocked
        and all(row["status"] == "COMPLETE" for row in replay_states.values()),
        "Replay continuation has missing, failed, active or unknown tasks",
    )
    user = read_json(plan.root / "orchestration/CONFIG.json")["user"]
    slurm = slurm or Slurm()
    live = slurm.queue(user)
    original = parent_reporting._original_allocations(parent, slurm, live, user)
    parent_pools = parent_reporting._recovery_pools(parent, slurm, live, user)
    replay_pools = _replay_pools(plan, slurm, live, user, parent_pools)
    dispatcher = _dispatcher(plan, slurm, live, user, replay_states)
    old_jobs = {row["job_id"] for row in original}
    jobs = old_jobs | {row["job_id"] for row in parent_pools.values()}
    jobs |= {row["job_id"] for row in replay_pools.values()}
    jobs.add(dispatcher["job_id"])
    require(
        len(jobs) == 9 + len(replay_pools),
        "Original, parent, replay or dispatcher allocations overlap or are missing",
    )
    _no_live_experiment_jobs(live, jobs, plan.root)
    receipts, inherited_rows, resumed_rows, replay_rows, accounting = [], [], [], [], []
    for task in matrix:
        key = task_id(task)
        if key in blocked:
            result = verify_endpoint_task(plan, task)
            replay_rows.append(_replayed_task(plan, task, replay_states[key], result, replay_pools))
            accounting.append(_replay_accounting(plan, task))
        else:
            result = verify_task(plan.root, task, parent.plan["registration_sha256"])
            if key in inherited:
                inherited_rows.append(
                    parent_reporting._inherited_task(parent, task, result, old_jobs)
                )
            else:
                resumed_rows.append(
                    parent_reporting._recovered_task(
                        parent, task, parent_states[key], result, parent_pools
                    )
                )
            accounting.append(parent_reporting.verify_accounting(parent, task))
        receipts.append(result)
    _workers_exact(parent.directory, parent_states, parent_pools)
    _workers_exact(plan.directory, replay_states, replay_pools)
    require(
        {p.name for p in (plan.directory / "tasks").iterdir() if p.is_dir()} == blocked,
        "Replay task evidence contains missing or unregistered attempts",
    )
    new_models = {row["model_id"] for row in frozen.scientific_matrix()}
    for kind in ("train", "endpoint"):
        require(
            len(new_models) == 12
            and {t["model_id"] for t in matrix if t["kind"] == kind} == new_models,
            "All twelve training and endpoint policies must finish",
        )
    replay_accounting = [row for row in accounting if row.get("authorized_replay") is True]
    require(
        len(replay_accounting) == 5
        and sum(row["physical_replay_records"] for row in replay_accounting) == 20
        and sum(row["canonical_added_records"] for row in replay_accounting) == 17
        and sum(row["overlap_verification_records"] for row in replay_accounting) == 3,
        "Authorized replay costs must reconcile exactly20 physical,17 added and3 repeated rows",
    )
    timing_rows = [row["replay_timing"] for row in replay_accounting]
    known_seconds = [
        row["generation_call_wall_seconds"]
        for row in timing_rows
        if row.get("generation_call_wall_seconds") is not None
    ]
    require(
        all(type(value) in (float, int) and value >= 0 for value in known_seconds),
        "Replay timing contains invalid observed durations",
    )
    plan.verify()
    _no_live_experiment_jobs(slurm.queue(user), jobs, plan.root)
    return dict(
        status="AUTHORIZED_AFTER_ALL_GPU_TASKS_VERIFIED",
        experiment_id=frozen.AMENDMENT_ID,
        registration_sha256=parent.plan["registration_sha256"],
        tasks=receipts,
        new_endpoint_records=183600,
        live_queue_and_accounting_verified=True,
        scientific_confirmation_separate_from_delivery=True,
        recovery=dict(
            parent_plan_sha256=parent.sha256,
            parent_audit_sha256=parent.plan["audit_sha256"],
            replay_plan_sha256=plan.sha256,
            replay_extension_manifest_sha256=plan.plan["extension_manifest_sha256"],
            max_concurrent_gpus=5,
            original_cancelled_allocations=original,
            parent_pools=parent_pools,
            replay_pools=replay_pools,
            dispatcher=dispatcher,
            inherited_tasks=inherited_rows,
            resumed_tasks=resumed_rows,
            replay_tasks=replay_rows,
            generation_accounting=accounting,
            physical_replay_records=20,
            canonical_added_records=17,
            overlap_verification_records=3,
            replay_time_cost=dict(
                known_generation_call_wall_seconds=sum(known_seconds),
                known_single_gpu_allocated_generation_seconds=sum(known_seconds),
                unknown_generation_time_tasks=5 - len(known_seconds),
                is_gpu_kernel_busy_time=False,
                per_task=timing_rows,
            ),
            cancelled_inflight_unpersisted_token_cost="UNKNOWN",
        ),
    )


def release_and_report(plan_path, *, slurm=None):
    """Original scientific functions and publication order, with an explicit recovery gate."""
    recovery = ReplayPlan(plan_path)
    root = recovery.root
    with shard_lease(root / "released"):
        cohort = frozen.cohort_verification(root)
        require(
            cohort["status"] == "PASS"
            and cohort["raw_evidence_status"] == "RAW_AND_GOLD_REVERIFIED",
            "Requested122 identity and complete raw/gold delivery are required",
        )
        authorization = verify_composite_release_gate(plan_path, slurm=slurm)
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
            "原始35项、首次恢复13项和授权重放后续5项均经独立解封验收。"
            "5组各重放一次: 共20条物理输出、17条补全和3条逐token一致性验证。原3条记录保留。"
            "重放成本单列。取消前未持久化的token成本仍为UNKNOWN。"
            "完整来源见AUTHORIZATION.json的recovery字段。\n"
        )
        recovery_cost = authorization["recovery"]
        timing = recovery_cost["replay_time_cost"]
        known_seconds = timing["known_generation_call_wall_seconds"]
        report += (
            f"\n重放生成调用的已知耗时合计为 {known_seconds:.3f} 秒。"
            f"生成耗时未知的任务为 {timing['unknown_generation_time_tasks']} 项。"
            "这里统计各次单卡分配内生成调用的墙钟时间之和。它不是GPU内核忙碌时间。\n\n"
            "| 任务 | 生成调用秒数 | 物理重放tokens | 重复验证tokens |\n"
            "| --- | ---: | ---: | ---: |\n"
        )
        for row in recovery_cost["generation_accounting"]:
            if row.get("authorized_replay") is not True:
                continue
            seconds = row["replay_timing"]["generation_call_wall_seconds"]
            duration = "UNKNOWN" if seconds is None else f"{seconds:.3f}"
            cost = row["replay_accounting"]
            report += (
                f"| {row['task']['model_id']} | {duration} | "
                f"{cost['physical_generated_tokens']} | {cost['duplicate_verification_tokens']} |\n"
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
