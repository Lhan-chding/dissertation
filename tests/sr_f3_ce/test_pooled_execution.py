"""Real frozen-controller fixtures for operational task pools; no model calls."""

from __future__ import annotations

import copy
import shutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_instrumented_execution import (
    complete_new,
    load_module,
    observation_receipt,
    seal_and_freeze,
)
from test_instrumented_execution import (
    setup as native_setup,
)

from sr_f3_ce import endpoint_runner, orchestration

CHECKOUT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pooled_env(tmp_path, monkeypatch):
    frozen = native_setup.__wrapped__(tmp_path, monkeypatch)
    seal_and_freeze(frozen)
    root, e = frozen.root, frozen.module
    operation = root / "operations" / ("d" * 40)
    operation.mkdir(parents=True)
    source = operation / "pooled_execution.py"
    shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce/pooled_execution.py", source)
    capacity_source = operation / "scheduler_capacity.py"
    shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce/scheduler_capacity.py", capacity_source)
    module = load_module(source, "fixed_task_pool_fixture")
    authorization = operation / "TASK_POOL_AUTHORIZATION.json"
    orchestration.save_json(
        authorization,
        dict(
            schema_version=1,
            scope=module.SCOPE,
            capacity_policy="LIVE_SLURM_QOS_AND_ASSOCIATION",
            scientific_settings_changed=False,
            preserve_history=True,
            no_duplicate_tasks=True,
            one_task_per_gpu=True,
            user_request="Use fixed tasks in multi-GPU jobs without duplicates.",
            run_identity_sha256=e._sha(root / "RUN_IDENTITY.json"),
            execution_freeze_sha256=e._sha(root / "EXECUTION_FREEZE.json"),
            config_sha256=e._sha(root / "orchestration/CONFIG.json"),
            extension_manifest_sha256=e._sha(frozen.manifest),
        ),
    )
    sources = operation / "OPERATION_SOURCES.json"
    orchestration.save_json(
        sources,
        dict(files={source.name: e._sha(source), capacity_source.name: e._sha(capacity_source)}),
    )
    identity_path = operation / "OPERATION_IDENTITY.json"
    value = dict(
        root=str(root),
        operation_commit="d" * 40,
        code_root=str(frozen.code),
        python=str(Path(sys.executable).resolve()),
        user="alice",
        task_pool=module.check_authorization(root, authorization),
        operation_sources_sha256=e._sha(sources),
        extension_manifest_path=str(frozen.manifest),
        extension_manifest_sha256=e._sha(frozen.manifest),
        config_sha256=e._sha(root / "orchestration/CONFIG.json"),
        run_identity_sha256=e._sha(root / "RUN_IDENTITY.json"),
    )
    orchestration.save_json(identity_path, value)
    value["identity_path"] = str(identity_path)
    cls = type(frozen.controller)
    baseline = {
        str(p.relative_to(root)): p.read_bytes()
        for base in (root / "code", root / "native")
        for p in base.rglob("*")
        if p.is_file()
    }
    baseline.update(
        {
            str(p.relative_to(root)): p.read_bytes()
            for p in (
                root / "orchestration/CONFIG.json",
                root / "EXECUTION_FREEZE.json",
            )
        }
    )
    monkeypatch.setattr(orchestration, "teacher_usage", orchestration.teacher_usage)
    pool = module.install(value, e)
    env = SimpleNamespace(
        frozen=frozen,
        root=root,
        module=module,
        value=value,
        pool=pool,
        controller=frozen.controller,
        slurm=frozen.slurm,
        extension=e,
        authorization=authorization,
        baseline=baseline,
        cls=cls,
        scheduler=dict(gpu_limit=8, submitted_limit=5, gpu_per_user_limit=None, per_job=None),
    )
    env.pool.capacity.execute = lambda command, **kwargs: scheduler_reply(env, command, **kwargs)
    return env


def scheduler_reply(env, command, **kwargs):
    assert kwargs == dict(capture_output=True, text=True, timeout=20, check=False)
    if command[:4] == ["sacctmgr", "-P", "show", "qos"]:
        fields = [
            "Name",
            "GrpTRES",
            "MaxTRESPU",
            "MaxTRES",
            "GrpJobs",
            "GrpSubmit",
            "MaxJobsPU",
            "MaxSubmitPU",
            "Flags",
        ]
        user_limit, per_job = env.scheduler["gpu_per_user_limit"], env.scheduler["per_job"]
        group = env.scheduler["gpu_limit"]
        values = [
            orchestration.TEACHER_QOS,
            "" if group is None else f"gres/gpu:pro6000={group}",
            "" if user_limit is None else f"gres/gpu:pro6000={user_limit}",
            "" if per_job is None else f"gres/gpu:pro6000={per_job}",
            "",
            "",
            "",
            str(env.scheduler["submitted_limit"]),
            "",
        ]
        stdout = "|".join(fields) + "\n" + "|".join(values) + "\n"
    elif command[:4] == ["sacctmgr", "-P", "show", "assoc"]:
        fields = [
            "Cluster",
            "Account",
            "User",
            "Partition",
            "GrpTRES",
            "MaxTRES",
            "MaxTRESPerNode",
            "GrpJobs",
            "GrpSubmit",
            "MaxJobs",
            "MaxSubmit",
        ]
        values = ["cluster", "rose", "alice", "", "", "", "", "", "", "", ""]
        stdout = "|".join(fields) + "\n" + "|".join(values) + "\n"
    elif command[0] == "sinfo":
        stdout = "node1|gpu:pro6000:8|760832|64|highmem|mix\n"
    else:
        raise AssertionError("Unexpected capacity command: " + repr(command))
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def intent_for(env, state):
    return orchestration.read_json(env.root / "orchestration/jobs" / state["key"] / "INTENT.json")


def task_outputs(env, intent, task, job, process_env, monkeypatch):
    with monkeypatch.context() as patch:
        for key, value in process_env.items():
            patch.setenv(key, value)
        artifact = env.root / (task + ".raw")
        artifact.write_text("synthetic retained task output; no model calls")
        evidence = env.frozen.old_tests.evidence(task)
        if orchestration.TASKS[task]["kind"] == "train":
            evidence["checkpoint_sha256"] = env.extension._sha(artifact)
        orchestration.write_task_receipt(
            env.root,
            task,
            artifacts=[artifact.name],
            evidence=evidence,
        )
        observation_receipt(env.frozen, intent, task, job)


def complete_pool(env, state, monkeypatch):
    intent = intent_for(env, state)
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", ",".join(str(3 + i) for i in range(state["gpu_count"]))
    )
    launches = []

    fixture = env

    def popen(command, *, env, start_new_session):
        assert start_new_session is True
        task = command[-1]
        assert command == intent["worker_entry_commands"][task]
        launches.append((command, copy.deepcopy(env)))
        task_outputs(fixture, intent, task, state["job_id"], env, monkeypatch)
        return SimpleNamespace(pid=71000 + len(launches), poll=lambda: 0)

    env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    fields = env.slurm.rows.pop(state["job_id"])
    env.slurm.accounting[state["job_id"]] = dict(
        job_id=state["job_id"],
        state="COMPLETED",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=fields["QOS"],
        name=fields["JobName"],
    )
    status = env.controller.status()
    return status, launches


def foreign_queue(env, *, jobs=0, gpus=0, user="alice"):
    for index in range(jobs):
        env.slurm.rows[str(9000 + index)] = dict(
            JobId=str(9000 + index),
            JobName="unrelated",
            QOS=orchestration.TEACHER_QOS,
            UserId=user + "(9)",
            ReqTRES="cpu=4,gres/gpu=" + str(gpus),
        )


@pytest.mark.parametrize("count", range(1, 9))
def test_dynamic_tasks_one_allocation_distinct_devices_and_real_commands(
    pooled_env, monkeypatch, count
):
    env = pooled_env
    if count < 8:
        foreign_queue(env, jobs=1, gpus=8 - count, user="bob")
    state = env.controller.submit("train")
    intent = intent_for(env, state)
    assert state["gpu_count"] == len(intent["task_ids"]) == count
    assert intent["execution_entry_schema"] == 2
    assert intent["cpus"] == 4 * count and intent["memory_gib"] == 90 * count
    assert "--gres=gpu:pro6000:" + str(count) in intent["sbatch"]
    assert intent["job_name"].endswith("-LH")
    assert intent["job_entry_command"][1].endswith("/pooled_execution.py")
    assert len([c for c in env.slurm.calls if c[0] == "sbatch"]) == 6
    status, launches = complete_pool(env, state, monkeypatch)
    assert set(intent["task_ids"]) <= set(status["completed"])
    assert [entry[1]["CUDA_VISIBLE_DEVICES"] for entry in launches] == [
        str(3 + i) for i in range(count)
    ]
    assert all(entry[1]["SRF3CE_JOB_KEY"] == state["key"] for entry in launches)
    assert {entry[1]["SRF3CE_CLAIM_ID"] for entry in launches} == {intent["claim_id"]}
    assert all("SRF3CE_RESUME_RECEIPT" not in entry[1] for entry in launches)


def test_class_and_frozen_originals_unchanged_new_instances_verify_pool(pooled_env, monkeypatch):
    env = pooled_env
    state = env.controller.submit("train")
    complete_pool(env, state, monkeypatch)
    assert type(env.controller) is env.cls is orchestration.Controller
    assert env.extension.install(env.root, env.frozen.manifest) is env.cls
    assert env.module.install(env.value, env.extension) is env.pool
    assert set(intent_for(env, state)["task_ids"]) <= endpoint_runner._completed(env.root)
    assert orchestration.read_json(env.root / "orchestration/CONFIG.json")["pool_size"] == 1
    for relative, original in env.baseline.items():
        assert (env.root / relative).read_bytes() == original


def test_mixed_legacy_and_pool_receipts_remain_independently_accepted(pooled_env, monkeypatch):
    env = pooled_env
    legacy = env.pool.original_submit(env.controller, "train")
    complete_new(env.frozen, legacy, monkeypatch)
    legacy_directory = env.root / "orchestration/jobs" / legacy["key"]
    originals = {p.name: p.read_bytes() for p in legacy_directory.iterdir() if p.is_file()}
    pooled = env.controller.submit("train")
    complete_pool(env, pooled, monkeypatch)
    done = endpoint_runner._completed(env.root)
    assert set(intent_for(env, legacy)["task_ids"] + intent_for(env, pooled)["task_ids"]) <= done
    assert {p.name: p.read_bytes() for p in legacy_directory.iterdir() if p.is_file()} == originals


def test_consecutive_submissions_do_not_repeat_claimed_tasks(pooled_env, monkeypatch):
    env = pooled_env
    first = env.controller.submit("train")
    assert env.controller.submit("train") is None
    complete_pool(env, first, monkeypatch)
    second = env.controller.submit("train")
    assert [s["gpu_count"] for s in (first, second)] == [8, 4]
    task_lists = [intent_for(env, s)["task_ids"] for s in (first, second)]
    assert len({task for tasks in task_lists for task in tasks}) == 12
    # Four ready tasks do not reserve the other four unused GPUs.
    assert env.controller.submit("train") is None


@pytest.mark.parametrize("jobs,gpus,user", [(5, 0, "alice"), (1, 8, "bob")])
def test_gpu_and_submission_caps_are_separate(pooled_env, jobs, gpus, user):
    env = pooled_env
    foreign_queue(env, jobs=jobs, gpus=gpus, user=user)
    calls = len(env.slurm.calls)
    rows = copy.deepcopy(env.slurm.rows)
    assert env.controller.submit("train") is None
    assert len(env.slurm.calls) == calls and env.slurm.rows == rows


def test_one_remaining_gpu_shrinks_pool_without_changing_science(pooled_env):
    env = pooled_env
    foreign_queue(env, jobs=1, gpus=7, user="bob")
    state = env.controller.submit("train")
    assert state["gpu_count"] == len(intent_for(env, state)["task_ids"]) == 1
    assert intent_for(env, state)["execution_entry_schema"] == 2


@pytest.mark.parametrize("ambiguous", ["ambiguous_submit", "ambiguous_release"])
def test_unknown_result_blocks_duplicate_submission(pooled_env, ambiguous):
    env = pooled_env
    setattr(env.slurm, ambiguous, True)
    state = env.controller.submit("train")
    assert state["status"] in {"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN"}
    calls = len(env.slurm.calls)
    with pytest.raises(RuntimeError, match="Unresolved"):
        env.controller.submit("train")
    assert len(env.slurm.calls) == calls


def test_phase_barrier_blocks_eval_and_release(pooled_env):
    env = pooled_env
    assert env.controller.submit("eval") is None
    assert env.controller.submit("release") is None
    assert not any("srf3ce-eval" in " ".join(c) for c in env.slurm.calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacity_policy", "FIXED_FIVE"),
        ("capacity_policy", True),
        ("scientific_settings_changed", True),
        ("preserve_history", False),
        ("no_duplicate_tasks", False),
        ("one_task_per_gpu", False),
        ("user_request", ""),
        ("scope", "unrelated"),
    ],
)
def test_authorization_scope_cannot_be_expanded(pooled_env, field, value):
    env = pooled_env
    auth = orchestration.read_json(env.authorization)
    auth[field] = value
    orchestration.save_json(env.authorization, auth)
    calls = len(env.slurm.calls)
    with pytest.raises(PermissionError):
        env.controller.submit("train")
    assert len(env.slurm.calls) == calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_count", 3),
        ("cpus", 4),
        ("memory_gib", 90),
        ("pool_operation_identity_sha256", "0" * 64),
        ("pool_authorization_sha256", "0" * 64),
        ("max_tasks_per_job", 3),
        ("job_name", "no-suffix"),
        ("task_ids", ["train--SRF3CE_RL_b0", "train--SRF3CE_RL_b0"]),
        ("task_ids", ["train--SRF3CE_RL_b0", "eval--SRF3CE_RL_b0"]),
    ],
)
def test_intent_tampering_rejected_even_if_state_hash_updated(pooled_env, field, value):
    env = pooled_env
    state = env.controller.submit("train")
    intent = intent_for(env, state)
    intent[field] = value
    directory = env.root / "orchestration/jobs" / state["key"]
    orchestration.save_json(directory / "INTENT.json", intent)
    state["intent_sha256"] = env.extension._sha(directory / "INTENT.json")
    orchestration.save_json(directory / "STATE.json", state)
    with pytest.raises(PermissionError):
        env.controller.attempts()


def test_stop_prevents_pool_submission(pooled_env):
    env = pooled_env
    (env.root / "STOP").write_text("retained stop")
    calls = len(env.slurm.calls)
    with pytest.raises(PermissionError, match="STOP"):
        env.controller.submit("train")
    assert len(env.slurm.calls) == calls


@pytest.mark.parametrize("devices", ["0", "0,0", "0,1,2", "bad,1"])
def test_runner_rejects_missing_duplicate_or_extra_devices(pooled_env, monkeypatch, devices):
    env = pooled_env
    state = env.controller.submit("train")
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    with pytest.raises(PermissionError):
        env.pool.run_job(state["key"], popen=lambda *a, **kw: pytest.fail("must not launch"))


@pytest.mark.parametrize("count", range(2, 9))
def test_child_failure_preserves_successful_siblings_and_reaps_out_of_order(
    pooled_env, monkeypatch, count
):
    env = pooled_env
    if count < 8:
        foreign_queue(env, jobs=1, gpus=8 - count, user="bob")
    state = env.controller.submit("train")
    intent = intent_for(env, state)
    directory = env.root / "orchestration/jobs" / state["key"]
    first, *others = intent["task_ids"]
    failed = {first} if count == 2 else {first, others[-1]}
    successes = [t for t in others if t not in failed]
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(count)))
    observations = []
    rounds = [0]
    fixture = env

    def popen(command, *, env, start_new_session):
        task = command[-1]
        assert start_new_session is True
        if task != first:
            if task not in failed:
                task_outputs(fixture, intent, task, state["job_id"], env, monkeypatch)
            return SimpleNamespace(
                pid=72002 + others.index(task), poll=lambda: 2 if task in failed else 0
            )

        def poll():
            rounds[0] += 1
            if rounds[0] >= 2:
                observations.append(all((directory / (t + ".EXIT.json")).is_file() for t in others))
                return 1
            return None

        return SimpleNamespace(pid=72001, poll=poll)

    with pytest.raises(RuntimeError, match="HANDOFF_REQUIRED"):
        env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    assert observations == [True]
    assert not (directory / "COMPLETE.json").exists()
    end = orchestration.read_json(directory / "POOL_END.json")
    expected_exits = {first: 1, **{t: 2 if t in failed else 0 for t in others}}
    assert end["verified_tasks"] == successes and end["exits"] == expected_exits
    assert end["automatic_retries"] == 0
    for task in successes:
        assert (env.root / "tasks" / task / "COMPLETE.json").exists()
        assert (env.root / "accounting/observability" / state["key"] / task / "FINAL.json").exists()


def test_parent_signal_targets_only_owned_process_groups(pooled_env, monkeypatch):
    env = pooled_env
    state = env.controller.submit("train")
    intent = intent_for(env, state)
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4,5,6,7,8")
    statuses, signals = {}, []

    def killpg(pid, signum):
        assert pid in statuses
        signals.append((pid, signum))
        statuses[pid] = -signum

    monkeypatch.setattr(env.module.os, "killpg", killpg)

    def popen(command, *, env, start_new_session):
        assert start_new_session
        pid = 73001 + len(statuses)
        statuses[pid] = None
        child = SimpleNamespace(pid=pid, poll=lambda: statuses[pid])
        if len(statuses) == len(intent["task_ids"]):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    with pytest.raises(RuntimeError, match="HANDOFF_REQUIRED"):
        env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    assert signals == [(73001 + i, signal.SIGTERM) for i in range(8)]
    end = orchestration.read_json(env.root / "orchestration/jobs" / state["key"] / "POOL_END.json")
    assert set(end["exits"]) == set(intent["task_ids"]) and end["signals"] == [signal.SIGTERM]


def test_pooled_timeout_never_replays_whole_allocation(pooled_env):
    env = pooled_env
    state = env.controller.submit("train")
    calls = len(env.slurm.calls)
    with pytest.raises(RuntimeError, match="HANDOFF_REQUIRED"):
        env.controller.resume(state["key"])
    assert len(env.slurm.calls) == calls


def test_successful_pool_cannot_be_launched_twice(pooled_env, monkeypatch):
    env = pooled_env
    state = env.controller.submit("train")
    complete_pool(env, state, monkeypatch)
    with pytest.raises(PermissionError):
        env.pool.run_job(state["key"], popen=lambda *a, **kw: pytest.fail("duplicate launch"))


@pytest.mark.parametrize("zero_gpu_capacity", [False, True])
def test_release_uses_pool_aware_cpu_worker_after_all_phases(
    pooled_env, monkeypatch, zero_gpu_capacity
):
    env = pooled_env
    # Complete actual frozen receipts/terminal checks through all stages with toy outputs.
    for kind, total in (("train", 12), ("eval", 15)):
        done = 0
        while done < total:
            state = env.controller.submit(kind)
            assert state is not None
            complete_pool(env, state, monkeypatch)
            done += len(intent_for(env, state)["task_ids"])
    if zero_gpu_capacity:
        env.scheduler["gpu_limit"] = 0
    state = env.controller.submit("release")
    intent = intent_for(env, state)
    assert intent["task_ids"] == ["release"] and state["gpu_count"] == 0
    assert intent["execution_entry_schema"] == 2
    assert intent["worker_entry_commands"]["release"][1].endswith("/pooled_execution.py")
    status, launches = complete_pool(env, state, monkeypatch)
    assert launches[0][1]["CUDA_VISIBLE_DEVICES"] == ""
    assert status["all_complete"] and not status["blocked"]
    assert set(orchestration.TASKS) == endpoint_runner._completed(env.root)


def test_dispatch_load_science_installs_pool_for_fresh_controller(pooled_env, monkeypatch):
    env = pooled_env
    from sr_f3_ce import bootstrap, external, protocol

    helper = load_module(CHECKOUT / "scripts/sr_f3_ce/dispatch_registered.py", "pool_load_fixture")
    # Fixture modules use actual frozen implementations but synthetic deployed paths;
    # do not replace their scheduling, receipt or scientific gate behavior.
    for module in (orchestration, protocol, bootstrap, external):
        name = module.__name__.rsplit(".", 1)[1]
        monkeypatch.setattr(
            module, "__file__", str(env.frozen.code / "src/sr_f3_ce" / (name + ".py"))
        )
    monkeypatch.setattr(helper, "load_extension", lambda *a, **kw: env.extension)
    monkeypatch.setattr(env.cls, "submit", env.pool.original_submit)
    monkeypatch.setattr(env.cls, "release", env.pool.original_release)
    monkeypatch.setattr(env.cls, "verify_job", env.pool.original_verify_job)
    monkeypatch.setattr(env.cls, "prepare_resume", env.pool.original_prepare_resume)
    monkeypatch.delattr(env.cls, env.module._OWNER)
    monkeypatch.setattr(env.extension, "_verify_intent", env.pool.original_intent)
    monkeypatch.setattr(env.extension, "teacher_usage", env.pool.original_teacher_usage)
    monkeypatch.setattr(orchestration, "teacher_usage", env.pool.original_teacher_usage)
    capacity = env.pool.capacity
    monkeypatch.setattr(
        helper,
        "load_capacity_source",
        lambda *a, **kw: SimpleNamespace(Capacity=lambda value: capacity),
    )
    api = helper.load_science(env.value)
    assert api.task_pool is not env.pool
    assert type(api.controller) is env.cls and api.controller.pool_size == 1
    env.pool, env.controller = api.task_pool, api.controller
    state = api.controller.submit("train")
    assert state["gpu_count"] == 8
    complete_pool(env, state, monkeypatch)
    assert set(intent_for(env, state)["task_ids"]) <= endpoint_runner._completed(env.root)


@pytest.mark.parametrize("count", range(3, 9))
@pytest.mark.parametrize(
    "status", ["SUBMISSION_UNKNOWN", "HELD", "RELEASE_UNKNOWN", "ACCOUNTING_UNKNOWN"]
)
def test_expanded_off_queue_reservations_count_fully_once_without_full_checks(
    pooled_env, monkeypatch, count, status
):
    env = pooled_env
    if count < 8:
        foreign_queue(env, jobs=1, gpus=8 - count, user="bob")
    state = env.controller.submit("train")
    assert state["gpu_count"] == count
    env.slurm.rows.pop(state["job_id"])
    reserved = dict(state, status=status)
    if status == "SUBMISSION_UNKNOWN":
        reserved.pop("job_id")
    queue = env.slurm.queue("alice")
    before = copy.deepcopy((queue, reserved))
    monkeypatch.setattr(
        env.extension, "verify_extension", lambda *a, **kw: pytest.fail("heavy quota check")
    )
    assert env.extension.teacher_usage(queue, [reserved], "alice") == dict(
        gpus=8, submitted=1, user_gpus=count
    )
    assert orchestration.teacher_usage(queue, [reserved], "alice") == dict(
        gpus=8, submitted=1, user_gpus=count
    )
    assert env.module.quota_usage(
        env.root,
        env.value,
        queue,
        [reserved],
        "alice",
        original=env.pool.original_teacher_usage,
        tasks=orchestration.TASKS,
    ) == dict(gpus=8, submitted=1, user_gpus=count)
    assert before == (queue, reserved)


@pytest.mark.parametrize("count", range(3, 9))
def test_live_expanded_allocations_use_real_queue_and_one_job_slot(pooled_env, count):
    env = pooled_env
    if count < 8:
        foreign_queue(env, jobs=1, gpus=8 - count, user="bob")
    state = env.controller.submit("train")
    queue = env.slurm.queue("alice")
    # Actual and generic typed TRES are alternative representations, not extra GPUs.
    queue[state["job_id"]]["AllocTRES"] = queue[state["job_id"]]["ReqTRES"]
    assert orchestration.teacher_usage(queue, [state], "alice") == dict(
        gpus=8, submitted=1, user_gpus=count
    )


@pytest.mark.parametrize("count", [True, "5", 5.0, None, -1, 9])
def test_legacy_invalid_unknown_counts_remain_rejected(pooled_env, count):
    with pytest.raises(PermissionError):
        orchestration.teacher_usage(
            {},
            [
                dict(
                    status="SUBMISSION_UNKNOWN",
                    gpu_count=count,
                    qos=orchestration.TEACHER_QOS,
                )
            ],
            "alice",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("intent_sha256", "0" * 64),
        ("run_identity_sha256", "0" * 64),
        ("qos", "other"),
        ("gpu_count", 4),
    ],
)
def test_expanded_unknown_cannot_escape_its_bound_intent(pooled_env, field, value):
    env = pooled_env
    state = env.controller.submit("train")
    env.slurm.rows.pop(state["job_id"])
    state.update(status="ACCOUNTING_UNKNOWN", **{field: value})
    with pytest.raises(PermissionError, match="reservation differs"):
        orchestration.teacher_usage({}, [state], "alice")


def test_unregistered_expanded_unknown_is_not_accepted_or_ignored(pooled_env):
    with pytest.raises(PermissionError):
        orchestration.teacher_usage(
            {},
            [
                dict(
                    key="not-registered",
                    status="SUBMISSION_UNKNOWN",
                    gpu_count=5,
                    qos=orchestration.TEACHER_QOS,
                )
            ],
            "alice",
        )


def test_same_expanded_unknown_claim_cannot_be_counted_twice(pooled_env):
    env = pooled_env
    state = env.controller.submit("train")
    env.slurm.rows.pop(state["job_id"])
    with pytest.raises(PermissionError, match="duplicated"):
        orchestration.teacher_usage({}, [state, state], "alice")


def test_quota_alias_replacement_is_detected_at_boundary(pooled_env, monkeypatch):
    env = pooled_env
    monkeypatch.setattr(env.extension, "teacher_usage", lambda *args: dict(gpus=0, submitted=0))
    with pytest.raises(PermissionError, match="accounting adapter changed"):
        env.pool.check()


def test_historical_capacity_remains_verifiable_after_live_limit_reduction(pooled_env, monkeypatch):
    env = pooled_env
    state = env.controller.submit("train")
    intent = intent_for(env, state)
    assert state["gpu_count"] == 8 and intent["capacity_limits"]["gpu_limit"] == 8
    env.scheduler["gpu_limit"] = 3
    assert env.pool.capacity.limits(fresh=True)["gpu_limit"] == 3
    monkeypatch.setattr(
        env.pool.capacity, "execute", lambda *a, **kw: pytest.fail("historical RPC")
    )
    env.pool.verify_intent(env.root, intent)
    queue = env.slurm.queue("alice")
    queue.pop(state["job_id"])
    assert env.pool.teacher_usage(queue, [state], "alice") == dict(gpus=8, submitted=1, user_gpus=8)
    complete_pool(env, state, monkeypatch)
    assert set(intent["task_ids"]) <= endpoint_runner._completed(env.root)


@pytest.mark.parametrize("field", ["gpu_limit", "per_job"])
def test_fresh_lowered_policy_keeps_accepted_allocation_held(pooled_env, monkeypatch, field):
    env = pooled_env
    with monkeypatch.context() as patch:
        patch.setattr(env.cls, "release", lambda *a: None)
        state = env.controller.submit("train")
    assert state["status"] == "HELD" and state["gpu_count"] == 8
    env.scheduler[field] = 7
    before = list(env.slurm.calls)
    with pytest.raises(PermissionError, match="capacity exceeded"):
        env.controller.release(state, intent_for(env, state))
    assert env.slurm.calls == before
    directory = env.root / "orchestration/jobs" / state["key"]
    assert not (directory / "RELEASE_INTENT.json").exists()
    assert orchestration.read_json(directory / "STATE.json")["status"] == "HELD"
    assert (
        orchestration.read_json(directory / "QUEUE_BEFORE_RELEASE.json")["capacity"][
            "max_tasks_per_job"
        ]
        == 7
    )


def test_fresh_policy_reduction_blocks_new_submit_without_claim(pooled_env):
    env = pooled_env
    env.pool.capacity.limits(fresh=True)
    foreign_queue(env, jobs=1, gpus=4, user="bob")
    env.scheduler["gpu_limit"] = 4
    calls = list(env.slurm.calls)
    directories = sorted((env.root / "orchestration/jobs").iterdir())
    assert env.controller.submit("train") is None
    assert env.slurm.calls == calls
    assert sorted((env.root / "orchestration/jobs").iterdir()) == directories


@pytest.mark.parametrize("group,per_job,expected", [(None, None, 8), (8, 3, 3), (4, None, 4)])
def test_actual_group_node_and_job_limits_define_pool_without_artificial_cap(
    pooled_env, group, per_job, expected
):
    env = pooled_env
    env.scheduler.update(gpu_limit=group, per_job=per_job)
    state = env.controller.submit("train")
    assert state["gpu_count"] == expected
    intent = intent_for(env, state)
    assert intent["max_tasks_per_job"] == expected
    assert intent["capacity_limits"]["gpu_limit"] is group


def test_per_user_limit_counts_own_usage_separately(pooled_env):
    env = pooled_env
    foreign_queue(env, jobs=1, gpus=2)
    env.scheduler["gpu_per_user_limit"] = 3
    state = env.controller.submit("train")
    assert state["gpu_count"] == 1
    assert env.pool.teacher_usage(env.slurm.queue("alice"), [state], "alice") == dict(
        gpus=3, submitted=2, user_gpus=3
    )


def test_legacy_single_gpu_resume_uses_seventh_to_eighth_gpu_without_config_change(
    pooled_env, monkeypatch
):
    env = pooled_env
    state = env.pool.original_submit(env.controller, "train")
    env.frozen.old_tests.expire_allocation(env.controller, state)
    foreign_queue(env, jobs=1, gpus=7, user="bob")
    config = (env.root / "orchestration/CONFIG.json").read_bytes()
    original_limits = (env.extension.GPU_LIMIT, env.extension.SUBMIT_LIMIT)
    resumed = env.controller.resume(state["key"], verifier=env.frozen.old_tests.resume_verifier)
    assert resumed["status"] == "SUBMITTED" and resumed["gpu_count"] == 1
    assert intent_for(env, resumed)["execution_entry_schema"] == 1
    assert original_limits == (env.extension.GPU_LIMIT, env.extension.SUBMIT_LIMIT)
    assert orchestration.GPU_LIMIT == 5 and orchestration.SUBMIT_LIMIT == 5
    assert (env.root / "orchestration/CONFIG.json").read_bytes() == config
    complete_new(env.frozen, resumed, monkeypatch)
    assert set(intent_for(env, resumed)["task_ids"]) <= endpoint_runner._completed(env.root)
    assert (env.root / "orchestration/CONFIG.json").read_bytes() == config


def test_native_limit_aliases_restore_on_base_exception(pooled_env, monkeypatch):
    env = pooled_env
    state = env.pool.original_submit(env.controller, "train")
    env.frozen.old_tests.expire_allocation(env.controller, state)
    foreign_queue(env, jobs=1, gpus=7, user="bob")
    before = (env.extension.GPU_LIMIT, env.extension.SUBMIT_LIMIT)
    config = (env.root / "orchestration/CONFIG.json").read_bytes()

    def interrupted(*args, **kwargs):
        assert env.extension.GPU_LIMIT == 8 and orchestration.GPU_LIMIT == 5
        raise KeyboardInterrupt("synthetic interrupted native resume")

    monkeypatch.setattr(env.pool, "original_submit", interrupted)
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        env.controller.resume(state["key"], verifier=env.frozen.old_tests.resume_verifier)
    assert before == (env.extension.GPU_LIMIT, env.extension.SUBMIT_LIMIT)
    assert (env.root / "orchestration/CONFIG.json").read_bytes() == config
    assert orchestration.GPU_LIMIT == 5
