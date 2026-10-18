"""Supersede one proven never-started pool without replaying any task/history."""

from __future__ import annotations

import copy
import shutil

import pytest
from test_pooled_execution import foreign_queue, intent_for
from test_rolling_endpoints import endpoint_pool
from test_rolling_endpoints import rolling_env as rolling_setup

from sr_f3_ce import orchestration


@pytest.fixture
def split_env(tmp_path, monkeypatch):
    env = rolling_setup.__wrapped__(tmp_path, monkeypatch)
    old_state, old_intent, old_directory = endpoint_pool(env, monkeypatch, count=7)
    env.slurm.rows.pop(old_state["job_id"])
    old_pool = env.pool
    operation = env.root / "operations" / ("e" * 40)
    operation.mkdir()
    for name in ("pooled_execution.py", "scheduler_capacity.py", "OPERATION_SOURCES.json"):
        shutil.copyfile(old_pool.operation / name, operation / name)
    terminal = dict(
        job_id=old_state["job_id"],
        state="CANCELLED",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=old_intent["qos"],
        name=old_intent["job_name"],
        start="Unknown",
        elapsed_raw=0,
        alloc_tres="",
    )
    terminal_path = operation / "CANCELLED_TERMINAL.json"
    orchestration.save_json(terminal_path, terminal)
    receipt = dict(
        schema_version=1,
        scope="SRF3CE_PENDING_POOL_SUPERSESSION",
        root=str(env.root),
        run_identity_sha256=env.value["run_identity_sha256"],
        predecessor_key=old_state["key"],
        predecessor_job_id=old_state["job_id"],
        predecessor_intent_sha256=env.extension._sha(old_directory / "INTENT.json"),
        predecessor_state_sha256=env.extension._sha(old_directory / "STATE.json"),
        terminal_path=str(terminal_path),
        terminal_sha256=env.extension._sha(terminal_path),
        terminal=terminal,
        queue_absent=True,
        no_execution=True,
        no_outputs=True,
        preserve_history=True,
        pre_cancel_pending_verified=True,
        cancel_pending_only=True,
        task_ids=old_intent["task_ids"],
    )
    receipt_path = operation / "SUPERSESSION.json"
    orchestration.save_json(receipt_path, receipt)
    auth = orchestration.read_json(env.authorization)
    auth["pending_split"] = dict(
        receipt_path=str(receipt_path),
        receipt_sha256=env.extension._sha(receipt_path),
        groups=[
            dict(gpu_count=4, task_ids=old_intent["task_ids"][:8]),
            dict(gpu_count=3, task_ids=old_intent["task_ids"][8:]),
        ],
    )
    auth_path = operation / "TASK_POOL_AUTHORIZATION.json"
    orchestration.save_json(auth_path, auth)
    value = copy.deepcopy(env.value)
    value.update(
        operation_commit="e" * 40, identity_path=str(operation / "OPERATION_IDENTITY.json")
    )
    value["task_pool"] = env.module.check_authorization(env.root, auth_path)
    orchestration.save_json(value["identity_path"], value)
    # An actual deployment uses a fresh Python process; restore captured native
    # methods before installing the new operation in this isolated CPU fixture.
    for name in ("submit", "release", "verify_job", "prepare_resume", "attempts"):
        monkeypatch.setattr(env.cls, name, getattr(old_pool, "original_" + name))
    monkeypatch.delattr(env.cls, env.module._OWNER)
    monkeypatch.setattr(env.extension, "_verify_intent", old_pool.original_intent)
    monkeypatch.setattr(env.extension, "teacher_usage", old_pool.original_teacher_usage)
    monkeypatch.setattr(orchestration, "teacher_usage", old_pool.original_teacher_usage)
    module = __import__("test_instrumented_execution").load_module(
        operation / "pooled_execution.py", "pending_split_fixture"
    )
    pool = module.install(value, env.extension)
    pool.capacity.execute = old_pool.capacity.execute
    env.pool, env.module, env.value, env.authorization = pool, module, value, auth_path
    env.old_state, env.old_intent, env.old_directory = old_state, old_intent, old_directory
    env.receipt_path, env.terminal_path = receipt_path, terminal_path
    env.old_files = {p.name: p.read_bytes() for p in old_directory.iterdir() if p.is_file()}
    return env


def test_split_claims_exact_groups_once_preserving_history(split_env):
    env = split_env
    assert env.old_state["key"] not in {s["key"] for s in env.controller.attempts()}
    rows = [(env.old_intent, env.old_state)]
    assert env.pool.filter_rows(rows) == []
    first = env.controller.submit("eval")
    second = env.controller.submit("eval")
    assert [first["gpu_count"], second["gpu_count"]] == [4, 3]
    assert intent_for(env, first)["task_ids"] == env.old_intent["task_ids"][:8]
    assert intent_for(env, second)["task_ids"] == env.old_intent["task_ids"][8:]
    assert env.controller.submit("eval") is None
    assert len(set(intent_for(env, first)["task_ids"] + intent_for(env, second)["task_ids"])) == 15
    assert env.old_files == {
        p.name: p.read_bytes() for p in env.old_directory.iterdir() if p.is_file()
    }
    assert len([t for t in env.controller.completed() if t.startswith("train--")]) == 12
    for relative, original in env.baseline.items():
        assert (env.root / relative).read_bytes() == original


def test_six_free_gpus_submit_four_then_wait_for_three(split_env):
    env = split_env
    foreign_queue(env, jobs=1, gpus=2, user="bob")
    first = env.controller.submit("eval")
    assert first["gpu_count"] == 4
    assert env.controller.submit("eval") is None
    assert env.controller.ready_tasks() == env.old_intent["task_ids"][8:]


@pytest.mark.parametrize(
    "marker",
    ["STARTED.json", "POOL_END.json", "COMPLETE.json", "launch", "exit", "observer", "old_receipt"],
)
def test_old_execution_evidence_fails_closed(split_env, marker):
    env = split_env
    task = env.old_intent["task_ids"][0]
    if marker in {"launch", "exit"}:
        path = env.old_directory / (task + (".LAUNCH.json" if marker == "launch" else ".EXIT.json"))
    elif marker == "observer":
        path = env.root / "accounting/observability" / env.old_state["key"] / "unexpected.json"
    elif marker == "old_receipt":
        path = env.root / "tasks" / task / "COMPLETE.json"
    else:
        path = env.old_directory / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    orchestration.save_json(path, dict(job_id=env.old_state["job_id"]))
    with pytest.raises(PermissionError):
        env.controller.attempts()


@pytest.mark.parametrize("which", ["state", "intent", "terminal", "receipt"])
def test_retained_bound_evidence_mutation_fails_closed(split_env, which):
    env = split_env
    path = {
        "state": env.old_directory / "STATE.json",
        "intent": env.old_directory / "INTENT.json",
        "terminal": env.terminal_path,
        "receipt": env.receipt_path,
    }[which]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PermissionError):
        env.controller.attempts()


def test_new_task_completion_is_not_mistaken_for_old_execution(split_env):
    env = split_env
    task = env.old_intent["task_ids"][0]
    path = env.root / "tasks" / task / "COMPLETE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    orchestration.save_json(path, dict(job_id="replacement"))
    assert env.old_state["key"] not in {s["key"] for s in env.controller.attempts()}


def test_partial_claim_and_undersized_capacity_do_not_repack(split_env):
    env = split_env
    tasks = env.old_intent["task_ids"]
    with pytest.raises(PermissionError, match="partially claimed"):
        env.pool.select_split_group(tasks[1:], {"gpus": 7}, {"max_tasks_per_job": 8})
    assert env.pool.select_split_group(tasks, {"gpus": 3}, {"max_tasks_per_job": 8}) == (0, [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "COMPLETED"),
        ("start", "2026-10-18T00:00:00"),
        ("elapsed_raw", 1),
        ("alloc_tres", "gres/gpu=7"),
    ],
)
def test_allocated_or_non_cancelled_terminal_cannot_be_authorized(split_env, field, value):
    env = split_env
    terminal = orchestration.read_json(env.terminal_path)
    terminal[field] = value
    orchestration.save_json(env.terminal_path, terminal)
    receipt = orchestration.read_json(env.receipt_path)
    receipt.update(terminal=terminal, terminal_sha256=env.extension._sha(env.terminal_path))
    orchestration.save_json(env.receipt_path, receipt)
    split = copy.deepcopy(env.value["task_pool"]["pending_split"])
    split["receipt_sha256"] = env.extension._sha(env.receipt_path)
    with pytest.raises(PermissionError, match="terminal cancellation"):
        env.module.check_pending_split(env.root, split)


@pytest.mark.parametrize("change", ["overlap", "missing", "allocation"])
def test_split_authorization_rejects_task_loss_duplication_and_changed_allocation(
    split_env, change
):
    env = split_env
    split = copy.deepcopy(env.value["task_pool"]["pending_split"])
    if change == "overlap":
        split["groups"][1]["task_ids"][0] = split["groups"][0]["task_ids"][0]
    elif change == "missing":
        split["groups"][1]["task_ids"].pop()
    else:
        split["groups"][0]["gpu_count"] = 3
    with pytest.raises(PermissionError, match="exact disjoint"):
        env.module.check_pending_split(env.root, split)


def test_new_intent_must_match_one_complete_authorized_group(split_env):
    env = split_env
    state = env.controller.submit("eval")
    intent = intent_for(env, state)
    intent["task_ids"] = intent["task_ids"][:-1]
    with pytest.raises(PermissionError, match="authorized split group"):
        env.pool.verify_intent(env.root, intent)


def test_filter_keeps_unrelated_failed_and_unknown_attempts(split_env):
    env = split_env
    unrelated = [
        dict(key="job-" + "a" * 16, status="FAILED"),
        dict(key="job-" + "b" * 16, status="SUBMISSION_UNKNOWN"),
    ]
    assert env.pool.filter_states([env.old_state, *unrelated]) == unrelated
