"""CPU regressions for a rolling pool of frozen endpoint tasks; no model calls."""

from __future__ import annotations

import copy
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pooled_execution import (
    complete_pool,
    foreign_queue,
    intent_for,
    task_outputs,
)
from test_pooled_execution import pooled_env as pooled_setup

from sr_f3_ce import endpoint_runner, orchestration


@pytest.fixture
def rolling_env(tmp_path, monkeypatch):
    env = pooled_setup.__wrapped__(tmp_path, monkeypatch)
    authorization = orchestration.read_json(env.authorization)
    authorization["rolling_endpoints"] = True
    orchestration.save_json(env.authorization, authorization)
    identity = orchestration.read_json(env.value["identity_path"])
    identity["task_pool"] = env.module.check_authorization(env.root, env.authorization)
    orchestration.save_json(env.value["identity_path"], identity)
    env.value.update(identity)
    env.pool.value = copy.deepcopy(env.value)
    env.pool.identity_sha256 = env.extension._sha(Path(env.value["identity_path"]))
    env.pool.check()
    # Exercise the real phase gate and preserve historical schema-2 receipts.
    while True:
        state = env.controller.submit("train")
        if state is None:
            break
        assert intent_for(env, state)["execution_entry_schema"] == 2
        complete_pool(env, state, monkeypatch)
    assert (
        len([task for task in env.controller.status()["completed"] if task.startswith("train--")])
        == 12
    )
    return env


def endpoint_pool(env, monkeypatch, count=2):
    if count < 8:
        foreign_queue(env, jobs=1, gpus=8 - count, user="bob")
    state = env.controller.submit("eval")
    assert state is not None
    intent = intent_for(env, state)
    assert intent["execution_entry_schema"] == 3
    assert state["gpu_count"] == count
    assert len(intent["task_ids"]) == min(15, 3 * count)
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(3 + i) for i in range(count)))
    return state, intent, env.root / "orchestration/jobs" / state["key"]


def finish_accounting(env, state):
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


@pytest.mark.parametrize("count", [1, 2, 7, 8])
def test_claims_bounded_endpoints_once_but_reserves_only_live_gpu_capacity(
    rolling_env, monkeypatch, count
):
    env = rolling_env
    _state, intent, _ = endpoint_pool(env, monkeypatch, count)
    assert len(set(intent["task_ids"])) == min(15, 3 * count)
    assert intent["cpus"] == count * 4 and intent["memory_gib"] == count * 90
    assert "--gres=gpu:pro6000:" + str(count) in intent["sbatch"]
    assert env.controller.submit("eval") is None
    assert env.controller.submit("release") is None
    # Train history still verifies after rolling mode is authorized.
    status = env.controller.status()
    assert len([task for task in status["completed"] if task.startswith("train--")]) == 12
    for relative, original in env.baseline.items():
        assert (env.root / relative).read_bytes() == original


def test_fast_slot_refills_before_slow_sibling_without_overlap_or_duplicate(
    rolling_env, monkeypatch
):
    env = rolling_env
    state, intent, directory = endpoint_pool(env, monkeypatch)
    first, slow, third, *_ = intent["task_ids"]
    processes, launches, live_by_device = {}, [], {}
    verified = []
    original_verify = env.extension.verify_observation_receipt

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        verified.append(args[2])
        return result

    monkeypatch.setattr(env.extension, "verify_observation_receipt", verify)
    fixture = env

    def popen(command, *, env, start_new_session):
        task, device = command[-1], env["CUDA_VISIBLE_DEVICES"]
        assert start_new_session is True
        assert command == intent["worker_entry_commands"][task]
        assert task not in processes
        assert device not in live_by_device, "Two live endpoint workers share one GPU"
        if task == third:
            assert slow in live_by_device.values(), "Fast GPU waited for slow sibling"
            assert first in verified, "Slot refilled before frozen output verification"
            assert (directory / (first + ".EXIT.json")).exists()
        live_by_device[device] = task
        launches.append(task)
        task_outputs(fixture, intent, task, state["job_id"], env, monkeypatch)
        done = [False]
        pending_polls = [0]

        def poll():
            if done[0]:
                return 0
            if task == slow and third not in processes:
                pending_polls[0] += 1
                assert pending_polls[0] < 20, "Freed GPU was never refilled"
                return None
            done[0] = True
            del live_by_device[device]
            return 0

        child = SimpleNamespace(pid=81000 + len(processes), poll=poll)
        processes[task] = child
        return child

    env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    assert launches == intent["task_ids"]
    assert live_by_device == {}
    chain = {}
    for task in launches:
        launch = orchestration.read_json(directory / (task + ".LAUNCH.json"))
        exited = orchestration.read_json(directory / (task + ".EXIT.json"))
        slot = launch["slot_index"]
        assert slot in (0, 1)
        assert launch["device"] == str(3 + slot)
        assert launch["predecessor_task"] == chain.get(slot)
        if slot in chain:
            previous_exit = orchestration.read_json(directory / (chain[slot] + ".EXIT.json"))
            assert previous_exit["exited_at"] <= launch["launched_at"]
        assert launch["launched_at"] <= exited["exited_at"]
        assert exited["returncode"] == 0
        chain[slot] = task
    finish_accounting(env, state)
    assert set(intent["task_ids"]) <= set(env.controller.status()["completed"])
    with pytest.raises(PermissionError):
        env.pool.run_job(state["key"], popen=lambda *a, **kw: pytest.fail("duplicate worker"))


@pytest.mark.parametrize("failure", ["worker", "receipt"])
def test_failure_stops_refill_preserves_successful_sibling_and_never_retries(
    rolling_env, monkeypatch, failure
):
    env = rolling_env
    state, intent, directory = endpoint_pool(env, monkeypatch)
    first, second, *_ = intent["task_ids"]
    launched = []
    fixture = env

    def popen(command, *, env, start_new_session):
        task = command[-1]
        launched.append(task)
        assert task in (first, second), "Started queued endpoint after failure"
        if task == second:
            task_outputs(fixture, intent, task, state["job_id"], env, monkeypatch)
        # A zero exit without frozen scientific receipts is also a failure.
        code = 4 if task == first and failure == "worker" else 0
        return SimpleNamespace(pid=82000 + len(launched), poll=lambda: code)

    with pytest.raises(RuntimeError, match="HANDOFF_REQUIRED"):
        env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    assert launched == [first, second]
    assert (env.root / "tasks" / second / "COMPLETE.json").exists()
    assert not (directory / "COMPLETE.json").exists()
    end = orchestration.read_json(directory / "POOL_END.json")
    assert end["automatic_retries"] == 0
    assert second in end["verified_tasks"]
    assert first in end["failures"]
    assert not any(
        (directory / (task + ".LAUNCH.json")).exists() for task in intent["task_ids"][2:]
    )


def test_signal_does_not_refill_and_targets_only_live_owned_children(rolling_env, monkeypatch):
    env = rolling_env
    state, intent, directory = endpoint_pool(env, monkeypatch)
    statuses, signalled, launched = {}, [], []

    def killpg(pid, signum):
        assert pid in statuses
        signalled.append((pid, signum))
        statuses[pid] = -signum

    monkeypatch.setattr(env.module.os, "killpg", killpg)

    def popen(command, *, env, start_new_session):
        launched.append(command[-1])
        pid = 83000 + len(launched)
        statuses[pid] = None
        if len(launched) == 2:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return SimpleNamespace(pid=pid, poll=lambda: statuses[pid])

    with pytest.raises(RuntimeError, match="HANDOFF_REQUIRED"):
        env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    assert launched == intent["task_ids"][:2]
    assert set(signalled) == {(83001, signal.SIGTERM), (83002, signal.SIGTERM)}
    assert not (directory / "COMPLETE.json").exists()
    end = orchestration.read_json(directory / "POOL_END.json")
    assert end["signals"] == [signal.SIGTERM]
    assert end["automatic_retries"] == 0


@pytest.mark.parametrize("field", ["slot_index", "predecessor_task", "launched_at", "device"])
def test_completed_pool_rejects_tampered_slot_chain(rolling_env, monkeypatch, field):
    env = rolling_env
    state, intent, directory = endpoint_pool(env, monkeypatch)
    fixture = env
    counter = [0]

    def popen(command, *, env, start_new_session):
        counter[0] += 1
        task_outputs(fixture, intent, command[-1], state["job_id"], env, monkeypatch)
        return SimpleNamespace(pid=84000 + counter[0], poll=lambda: 0)

    env.pool.run_job(state["key"], popen=popen, sleep=lambda _: None)
    finish_accounting(env, state)
    assert set(intent["task_ids"]) <= set(env.controller.status()["completed"])
    # Re-verification cannot trust the former COMPLETE state after receipt corruption.
    task = intent["task_ids"][2]
    path = directory / (task + ".LAUNCH.json")
    launch = orchestration.read_json(path)
    if field == "slot_index":
        launch[field] = 99
    elif field == "predecessor_task":
        launch[field] = task
    elif field == "launched_at":
        launch[field] = "1900-01-01T00:00:00+00:00" if isinstance(launch[field], str) else 0
    else:
        launch[field] = "unallocated-device"
    orchestration.save_json(path, launch)
    with pytest.raises((PermissionError, ValueError)):
        env.pool.verify_job(env.controller, state)


def test_new_capacity_allows_disjoint_remaining_endpoint_claim(rolling_env, monkeypatch):
    env = rolling_env
    first, first_intent, _ = endpoint_pool(env, monkeypatch, count=2)
    assert len(first_intent["task_ids"]) == 6
    # Simulate the unrelated user's allocation ending, without changing our first pool.
    env.slurm.rows.pop("9000")
    second = env.controller.submit("eval")
    second_intent = intent_for(env, second)
    assert second["gpu_count"] == 6
    assert len(second_intent["task_ids"]) == 9
    assert set(first_intent["task_ids"]).isdisjoint(second_intent["task_ids"])
    assert len(set(first_intent["task_ids"] + second_intent["task_ids"])) == 15
    assert env.controller.submit("eval") is None
    assert env.slurm.rows[first["job_id"]]["JobId"] == first["job_id"]


@pytest.mark.parametrize("corruption", [None, "identity", "source"])
def test_actual_previous_operation_schema2_stays_verifiable(tmp_path, monkeypatch, corruption):
    import shutil
    import subprocess

    from test_instrumented_execution import load_module
    from test_pooled_execution import CHECKOUT, scheduler_reply

    previous = subprocess.run(
        [
            "git",
            "show",
            "4a8eb2ef2c9da1c431e107895f8d43c75c0cc783:scripts/sr_f3_ce/pooled_execution.py",
        ],
        cwd=CHECKOUT,
        capture_output=True,
        check=False,
    )
    if previous.returncode:
        pytest.skip("The historical git object is not present in this deployment")
    env = pooled_setup.__wrapped__(tmp_path, monkeypatch)

    def restore_native():
        pool = env.pool
        monkeypatch.setattr(env.cls, "submit", pool.original_submit)
        monkeypatch.setattr(env.cls, "release", pool.original_release)
        monkeypatch.setattr(env.cls, "verify_job", pool.original_verify_job)
        monkeypatch.setattr(env.cls, "prepare_resume", pool.original_prepare_resume)
        monkeypatch.delattr(env.cls, env.module._OWNER)
        monkeypatch.setattr(env.extension, "_verify_intent", pool.original_intent)
        monkeypatch.setattr(env.extension, "teacher_usage", pool.original_teacher_usage)
        monkeypatch.setattr(orchestration, "teacher_usage", pool.original_teacher_usage)

    # Deploy actual historical source before it owns any pool. Never rewrite it again.
    restore_native()
    old_operation = Path(env.value["identity_path"]).parent
    old_source = old_operation / "pooled_execution.py"
    old_source.write_bytes(previous.stdout)
    sources = orchestration.read_json(old_operation / "OPERATION_SOURCES.json")
    sources["files"]["pooled_execution.py"] = env.extension._sha(old_source)
    orchestration.save_json(old_operation / "OPERATION_SOURCES.json", sources)
    identity = orchestration.read_json(env.value["identity_path"])
    identity["operation_sources_sha256"] = env.extension._sha(
        old_operation / "OPERATION_SOURCES.json"
    )
    orchestration.save_json(env.value["identity_path"], identity)
    env.value.update(identity)
    env.module = load_module(old_source, "rolling_previous_pool_fixture")
    env.pool = env.module.install(env.value, env.extension)
    env.pool.capacity.execute = lambda command, **kwargs: scheduler_reply(env, command, **kwargs)
    foreign_queue(env, jobs=1, gpus=5, user="bob")
    previous_state = env.controller.submit("train")
    assert previous_state["gpu_count"] == 3
    assert intent_for(env, previous_state)["execution_entry_schema"] == 2
    complete_pool(env, previous_state, monkeypatch)
    env.slurm.rows.pop("9000")
    preserved = {
        path: path.read_bytes()
        for base in (old_operation, env.root / "orchestration/jobs" / previous_state["key"])
        for path in base.rglob("*")
        if path.is_file()
    }

    # Independent deployment owns subsequent work and delegates old receipt validation.
    restore_native()
    new_operation = env.root / "operations" / ("e" * 40)
    new_operation.mkdir()
    for name in ("pooled_execution.py", "scheduler_capacity.py"):
        shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce" / name, new_operation / name)
    env.module = load_module(new_operation / "pooled_execution.py", "rolling_next_pool_fixture")
    authorization = orchestration.read_json(env.authorization)
    authorization["rolling_endpoints"] = True
    env.authorization = new_operation / "TASK_POOL_AUTHORIZATION.json"
    orchestration.save_json(env.authorization, authorization)
    sources = dict(
        files={
            name: env.extension._sha(new_operation / name)
            for name in (
                "pooled_execution.py",
                "scheduler_capacity.py",
            )
        }
    )
    orchestration.save_json(new_operation / "OPERATION_SOURCES.json", sources)
    identity.update(
        operation_commit="e" * 40,
        task_pool=env.module.check_authorization(env.root, env.authorization),
        operation_sources_sha256=env.extension._sha(new_operation / "OPERATION_SOURCES.json"),
    )
    identity_path = new_operation / "OPERATION_IDENTITY.json"
    orchestration.save_json(identity_path, identity)
    env.value = dict(identity, identity_path=str(identity_path))
    env.pool = env.module.install(env.value, env.extension)
    env.pool.capacity.execute = lambda command, **kwargs: scheduler_reply(env, command, **kwargs)
    reserved = dict(previous_state, status="SUBMITTED")
    assert env.pool.teacher_usage({}, [reserved], "alice") == dict(gpus=3, submitted=1, user_gpus=3)
    if corruption:
        if corruption == "source":
            old_source.write_bytes(previous.stdout + b"\n# altered historical source\n")
        else:
            old_identity = orchestration.read_json(old_operation / "OPERATION_IDENTITY.json")
            old_identity["run_identity_sha256"] = "0" * 64
            orchestration.save_json(old_operation / "OPERATION_IDENTITY.json", old_identity)
        with pytest.raises(PermissionError):
            env.pool.teacher_usage({}, [reserved], "alice")
        return
    old_intent = intent_for(env, previous_state)
    env.pool.verify_intent(env.root, old_intent)
    assert env.pool.verify_job(env.controller, previous_state) == old_intent["task_ids"]
    while current := env.controller.submit("train"):
        complete_pool(env, current, monkeypatch)
    eval_state, eval_intent, _ = endpoint_pool(env, monkeypatch, count=7)
    assert len(eval_intent["task_ids"]) == 15 and eval_state["gpu_count"] == 7
    assert all(path.read_bytes() == data for path, data in preserved.items())


def test_all_fifteen_rolling_endpoints_unlock_frozen_release(rolling_env, monkeypatch):
    env = rolling_env
    state, intent, _ = endpoint_pool(env, monkeypatch, count=7)
    assert env.controller.submit("release") is None
    status, launches = complete_pool(env, state, monkeypatch)
    assert len(launches) == len(intent["task_ids"]) == 15
    assert set(intent["task_ids"]) <= endpoint_runner._completed(env.root)
    assert set(intent["task_ids"]) <= set(status["completed"])
    release = env.controller.submit("release")
    assert release is not None and release["gpu_count"] == 0
    assert intent_for(env, release)["execution_entry_schema"] == 2
    status, _ = complete_pool(env, release, monkeypatch)
    assert status["all_complete"] and not status["blocked"]
