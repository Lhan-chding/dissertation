"""Composite release fixtures: no model, scheduler or scientific TEST scoring."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f2_replay import orchestration, reporting
from sr_f12.orchestration import file_hash
from sr_f12.protocol import TEACHER_QOS, object_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def change(path, **updates):
    value = json.loads(path.read_text())
    value.update(updates)
    write(path, value)


class Slurm:
    def __init__(self):
        self.live, self.accounting = {}, {}

    def queue(self, user):
        assert user == "alice"
        return self.live

    def terminal(self, job):
        return self.accounting.get(job)


@pytest.fixture
def composite(tmp_path, monkeypatch):
    root = tmp_path / "original"
    directory = root / "replay/authorized"
    matrix = reporting.global_task_matrix()
    keys = [reporting.task_id(task) for task in matrix]
    blocked = set(keys[-5:])
    parent = SimpleNamespace(
        root=root,
        directory=root / "recovery/parent",
        sha256="parent",
        extension=Path(reporting.parent_reporting.__file__).resolve().parents[2],
        tasks=dict(zip(keys, matrix, strict=True)),
        matrix=matrix,
        allowed={
            key: dict(
                task=task,
                status=(
                    "INHERITED_COMPLETE"
                    if i < 35
                    else "BLOCKED_INCOMPLETE_GENERATION"
                    if key in blocked
                    else "RESUME_ALLOWED"
                ),
            )
            for i, (key, task) in enumerate(zip(keys, matrix, strict=True))
        },
        plan=dict(python=sys.executable, registration_sha256="registered", audit_sha256="audit"),
    )
    plan = SimpleNamespace(
        root=root,
        directory=directory,
        path=directory / "PLAN.json",
        parent=parent,
        extension=Path(reporting.__file__).resolve().parents[2],
        code=Path(reporting.frozen.__file__).resolve().parents[2],
        sha256="replay",
        tasks={
            key: dict(task=parent.tasks[key], preserved_count=3 if key == keys[-1] else 0)
            for key in blocked
        },
        verify=lambda: True,
        plan=dict(extension_manifest="MANIFEST.json", extension_manifest_sha256="manifest"),
    )
    write(
        directory / "MANIFEST.json",
        dict(
            files={
                "src/sr_f2_replay/reporting.py": "bound",
                "scripts/sr_f2/release_replay.py": "bound",
            }
        ),
    )
    write(root / "orchestration/CONFIG.json", dict(user="alice"))
    states = {
        key: dict(status="COMPLETE", pool_id="RECOVERY_POOL_1", task=parent.tasks[key])
        for key in keys[35:-5]
    }
    replay_states = {
        key: dict(status="COMPLETE", pool_id="REPLAY_POOL_1", task=parent.tasks[key])
        for key in blocked
    }
    parent_pools = {f"RECOVERY_POOL_{i}": dict(job_id=str(200 + i)) for i in range(1, 4)}
    replay_pools = {"REPLAY_POOL_1": dict(job_id="301")}
    monkeypatch.setattr(reporting, "ReplayPlan", lambda path: plan)
    monkeypatch.setattr(
        reporting, "RecoveryQueue", lambda p: SimpleNamespace(states=lambda: states)
    )
    monkeypatch.setattr(
        orchestration, "ReplayQueue", lambda p: SimpleNamespace(states=lambda: replay_states)
    )
    monkeypatch.setattr(orchestration, "verify_controller_source", lambda p: None)
    monkeypatch.setattr(
        reporting.parent_reporting,
        "_original_allocations",
        lambda *args: [dict(job_id=str(i)) for i in range(100, 105)],
    )
    monkeypatch.setattr(reporting.parent_reporting, "_recovery_pools", lambda *args: parent_pools)
    monkeypatch.setattr(reporting, "_replay_pools", lambda *args: replay_pools)
    monkeypatch.setattr(reporting, "_dispatcher", lambda *args: dict(job_id="400"))
    monkeypatch.setattr(
        reporting.parent_reporting, "_inherited_task", lambda p, t, r, j: dict(task=t, receipt=r)
    )
    monkeypatch.setattr(
        reporting.parent_reporting, "_recovered_task", lambda p, t, s, r, j: dict(task=t, receipt=r)
    )
    monkeypatch.setattr(reporting, "_replayed_task", lambda p, t, s, r, j: dict(task=t, receipt=r))
    monkeypatch.setattr(reporting, "verify_task", lambda p, t, s: dict(t, receipt_sha256="valid"))
    monkeypatch.setattr(
        reporting, "verify_endpoint_task", lambda p, t: dict(t, receipt_sha256="valid")
    )
    monkeypatch.setattr(reporting.parent_reporting, "verify_accounting", lambda p, t: dict(task=t))
    monkeypatch.setattr(
        reporting,
        "_replay_accounting",
        lambda p, t: dict(
            task=t,
            authorized_replay=True,
            physical_replay_records=4,
            canonical_added_records=4 - p.tasks[reporting.task_id(t)]["preserved_count"],
            overlap_verification_records=p.tasks[reporting.task_id(t)]["preserved_count"],
            replay_timing=dict(status="UNKNOWN", generation_call_wall_seconds=None),
        ),
    )
    for base, values in ((parent.directory, states), (directory, replay_states)):
        for key, state in values.items():
            (base / "pools" / state["pool_id"] / "workers" / key).mkdir(parents=True)
    for key in blocked:
        (directory / "tasks" / key).mkdir(parents=True)
    return SimpleNamespace(
        plan=plan,
        parent_states=states,
        replay_states=replay_states,
        parent_pools=parent_pools,
        replay_pools=replay_pools,
        slurm=Slurm(),
    )


def test_composite_one_replacement_covers_all_five(composite):
    result = reporting.verify_composite_release_gate(composite.plan.path, slurm=composite.slurm)
    assert len(result["tasks"]) == 53
    recovery = result["recovery"]
    assert [len(recovery[k]) for k in ("inherited_tasks", "resumed_tasks", "replay_tasks")] == [
        35,
        13,
        5,
    ]
    assert [
        recovery[k]
        for k in (
            "physical_replay_records",
            "canonical_added_records",
            "overlap_verification_records",
        )
    ] == [20, 17, 3]
    assert recovery["cancelled_inflight_unpersisted_token_cost"] == "UNKNOWN"


@pytest.mark.parametrize(
    "stage,status",
    [("parent", "FAILED"), ("parent", "RUNNING"), ("replay", "CLAIMED"), ("replay", "UNKNOWN")],
)
def test_incomplete_or_unknown_task_blocks(composite, stage, status):
    states = composite.parent_states if stage == "parent" else composite.replay_states
    next(iter(states.values()))["status"] = status
    with pytest.raises(PermissionError, match="missing, failed, active or unknown"):
        reporting.verify_composite_release_gate(composite.plan.path, slurm=composite.slurm)


def test_missing_task_blocks(composite):
    composite.replay_states.pop(next(iter(composite.replay_states)))
    with pytest.raises(PermissionError, match="missing"):
        reporting.verify_composite_release_gate(composite.plan.path, slurm=composite.slurm)


@pytest.mark.parametrize("job,command", [("301", "unrelated"), ("999", "experiment")])
def test_live_or_unregistered_job_blocks(composite, job, command):
    composite.slurm.live[job] = dict(
        Command=str(composite.plan.root / "new/run.sbatch") if command == "experiment" else command
    )
    with pytest.raises(PermissionError, match=r"squeue|Unregistered"):
        reporting.verify_composite_release_gate(composite.plan.path, slurm=composite.slurm)


def test_unclaimed_worker_blocks(composite):
    (composite.plan.directory / "pools/REPLAY_POOL_1/workers/unclaimed").mkdir()
    with pytest.raises(PermissionError, match="Unclaimed"):
        reporting.verify_composite_release_gate(composite.plan.path, slurm=composite.slurm)


def test_extra_or_missing_replay_cost_blocks(composite):
    for task in composite.plan.tasks.values():
        task["preserved_count"] = 0
    with pytest.raises(PermissionError, match="20 physical"):
        reporting.verify_composite_release_gate(composite.plan.path, slurm=composite.slurm)


def test_reporting_never_scores_after_failed_gate(composite, monkeypatch):
    called = []
    next(iter(composite.replay_states.values()))["status"] = "FAILED"
    monkeypatch.setattr(reporting, "shard_lease", lambda p: nullcontext())
    monkeypatch.setattr(
        reporting.frozen,
        "cohort_verification",
        lambda p: dict(
            status="PASS", raw_evidence_status="RAW_AND_GOLD_REVERIFIED", receipt_sha256="cohort"
        ),
    )
    monkeypatch.setattr(reporting.frozen, "score_new_endpoints", lambda *a: called.append(a))
    with pytest.raises(PermissionError):
        reporting.release_and_report(composite.plan.path, slurm=composite.slurm)
    assert not called
    assert not (composite.plan.root / "released/AUTHORIZATION.json").exists()


@pytest.fixture
def accounting(tmp_path, monkeypatch):
    root, directory = tmp_path / "root", tmp_path / "replay"
    task = dict(kind="endpoint", model_id="model")
    ledger = root / "accounting/endpoint-model.jsonl"
    ledger.parent.mkdir(parents=True)
    rows = [
        dict(group_seed=77, group_row_index=i, group_size=4, tokens=[10] * (i + 1))
        for i in range(4)
    ]
    write(root / "sealed/evaluation/model/groups/one.json", dict(records=rows))
    write(directory / "staged.json", dict(records=rows))
    spec = dict(
        original_ledger_path="accounting/endpoint-model.jsonl", group_seed=77, preserved_count=3
    )
    plan = SimpleNamespace(root=root, directory=directory, tasks={reporting.task_id(task): spec})
    events = [
        dict(
            experiment_id=reporting.EXPERIMENT_ID,
            task_id="endpoint-model",
            kind="completion_attempts",
            count=4,
            metadata=dict(group_seed=77),
        )
    ]
    events += [
        dict(
            experiment_id=reporting.EXPERIMENT_ID,
            task_id="endpoint-model",
            kind="generated_tokens",
            count=i + 1,
            metadata=dict(group_seed=77, row_index=i),
        )
        for i in range(3)
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in events))
    proof = dict(
        staged_path="staged.json",
        timing=dict(status="UNKNOWN", generation_call_wall_seconds=None),
        accounting=dict(
            physical_attempt_groups=1,
            physical_attempt_rows=4,
            canonical_new_rows=1,
            duplicate_verification_rows=3,
        ),
    )
    monkeypatch.setattr(reporting, "verify_replay_receipt", lambda p, t: proof)
    return SimpleNamespace(plan=plan, task=task, ledger=ledger, events=events, proof=proof)


def test_replay_tokens_complete_only_the_missing_suffix(accounting):
    result = reporting._replay_accounting(accounting.plan, accounting.task)
    assert result["records"] == 4 and result["canonical_added_records"] == 1
    assert result["overlap_verification_records"] == 3
    assert len(accounting.ledger.read_text().splitlines()) == 4  # No synthetic token event added.


@pytest.mark.parametrize(
    "mutation", ["missing_original", "duplicate_attempt", "synthetic_suffix", "wrong_tokens"]
)
def test_replay_ledger_gap_or_duplicate_blocks(accounting, mutation):
    rows = accounting.events
    if mutation == "missing_original":
        rows.pop()
    elif mutation == "duplicate_attempt":
        rows.append(rows[0])
    elif mutation == "synthetic_suffix":
        rows.append(dict(rows[-1], count=4, metadata=dict(group_seed=77, row_index=3)))
    else:
        rows[-1]["count"] = 999
    accounting.ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PermissionError, match="gap or repeated"):
        reporting._replay_accounting(accounting.plan, accounting.task)


def test_replay_token_mismatch_provenance_blocks(accounting, monkeypatch):
    def mismatch(*args):
        raise PermissionError("Existing tokens differ")

    monkeypatch.setattr(reporting, "verify_replay_receipt", mismatch)
    with pytest.raises(PermissionError, match="tokens differ"):
        reporting._replay_accounting(accounting.plan, accounting.task)


@pytest.fixture
def pools(tmp_path, monkeypatch):
    directory = tmp_path / "replay"
    plan = SimpleNamespace(
        directory=directory,
        sha256="replay",
        tasks={str(i): {} for i in range(5)},
        parent=SimpleNamespace(sha256="parent"),
    )
    states = {key: dict(pool_id="REPLAY_POOL_1", claim_sha256="claim" + key) for key in plan.tasks}
    claims = object_hash({key: row["claim_sha256"] for key, row in states.items()})
    monkeypatch.setattr(
        orchestration, "ReplayQueue", lambda p: SimpleNamespace(states=lambda: states)
    )
    proof = dict(parent_job_id="201", parent_pool_id="RECOVERY_POOL_1")
    monkeypatch.setattr(
        orchestration, "Controller", lambda *a, **k: SimpleNamespace(_parent=lambda *a: ({}, proof))
    )
    slurm = Slurm()
    parent_pools = {"RECOVERY_POOL_1": dict(job_id="201")}
    for pool in ("REPLAY_POOL_2", "REPLAY_POOL_3"):
        write(
            directory / "pools" / pool / "NOT_NEEDED.json",
            dict(
                pool_id=pool,
                plan_sha256="replay",
                task_ids=list(plan.tasks),
                task_claims_sha256=claims,
                reason="ALL_ENDPOINTS_ALREADY_CLAIMED",
            ),
        )
    base = directory / "pools/REPLAY_POOL_1"
    base.mkdir(parents=True)
    script = base / "run.sbatch"
    script.write_text("#!/bin/sh\nexit 0\n")
    intent = dict(
        pool_id="REPLAY_POOL_1",
        gpu_count=2,
        plan_sha256="replay",
        user="alice",
        script=str(script),
        script_sha256=file_hash(script),
        job_name="srf2replay-one",
        comment="bound",
        parent_pool_id="RECOVERY_POOL_1",
        parent_job_id="201",
        parent_terminal_sha256=object_hash(proof),
    )
    for name, value in {
        "INTENT": intent,
        "STATE": dict(
            status="DRAINED",
            gpu_count=2,
            plan_sha256="replay",
            pool_id="REPLAY_POOL_1",
            job_id="301",
        ),
        "STARTED": dict(plan_sha256="replay", job_id="301", devices=["0", "1"]),
        "COMPLETE": dict(
            status="DRAINED",
            plan_sha256="replay",
            pool_id="REPLAY_POOL_1",
            job_id="301",
            parent_terminal_sha256=object_hash(proof),
        ),
        "PARENT_TERMINAL": proof,
        "ALLOCATION_VERIFIED": dict(
            fields=dict(
                JobId="301",
                Account="rose",
                Partition="cluster02",
                QOS=TEACHER_QOS,
                JobName=intent["job_name"],
                Comment="bound",
                Command=str(script),
                UserId="alice(1000)",
                ReqTRES="cpu=8,mem=160G,gres/gpu=2,gres/gpu:pro6000=2",
                Features="highmem",
                JobState="PENDING",
                Reason="JobHeldUser",
            )
        ),
        "RELEASE_INTENT": dict(job_id="301"),
        "RELEASE": dict(returncode=0),
    }.items():
        write(base / (name + ".json"), value)
    slurm.accounting["301"] = dict(
        job_id="301",
        state="COMPLETED",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=TEACHER_QOS,
        name=intent["job_name"],
    )
    for name in ("CAPACITY_BEFORE_SUBMIT", "CAPACITY_BEFORE_RELEASE", "CAPACITY_AT_START"):
        write(
            base / (name + ".json"),
            dict(
                plan_sha256="replay",
                proposed_gpus=2 if name.endswith("SUBMIT") else 0,
                capacity=dict(teacher_gpus=3, run_gpus=3, user_submissions=3),
                parent_terminal_sha256={"REPLAY_POOL_1": object_hash(proof)},
            ),
        )
    return SimpleNamespace(
        plan=plan, slurm=slurm, parent_pools=parent_pools, base=base, states=states
    )


def test_pool_gate_accepts_proven_unused_pools(pools):
    actual = reporting._replay_pools(pools.plan, pools.slurm, {}, "alice", pools.parent_pools)
    assert set(actual) == {"REPLAY_POOL_1"}


@pytest.mark.parametrize("state", [None, "FAILED", "RUNNING"])
def test_pool_accounting_missing_failed_or_nonterminal_blocks(pools, state):
    if state is None:
        pools.slurm.accounting.clear()
    else:
        pools.slurm.accounting["301"]["state"] = state
    with pytest.raises(PermissionError, match="sacct"):
        reporting._replay_pools(pools.plan, pools.slurm, {}, "alice", pools.parent_pools)


def test_unused_pool_with_submission_intent_blocks(pools):
    write(pools.base.parent / "REPLAY_POOL_2/INTENT.json", {})
    with pytest.raises(PermissionError, match="Unused replacement"):
        reporting._replay_pools(pools.plan, pools.slurm, {}, "alice", pools.parent_pools)


def test_wrong_parent_terminal_proof_blocks(pools):
    change(pools.base / "PARENT_TERMINAL.json", parent_job_id="wrong")
    with pytest.raises(PermissionError, match="drained parent"):
        reporting._replay_pools(pools.plan, pools.slurm, {}, "alice", pools.parent_pools)


def test_capacity_unknown_or_over_five_blocks(pools):
    change(
        pools.base / "CAPACITY_BEFORE_SUBMIT.json",
        capacity=dict(teacher_gpus=4, run_gpus=4, user_submissions=3),
    )
    with pytest.raises(PermissionError, match="capacity snapshot"):
        reporting._replay_pools(pools.plan, pools.slurm, {}, "alice", pools.parent_pools)


@pytest.fixture
def dispatcher(tmp_path):
    plan = SimpleNamespace(directory=tmp_path / "replay", sha256="replay")
    base = plan.directory / "dispatcher"
    base.mkdir(parents=True)
    script = base / "run.sbatch"
    script.write_text("#!/bin/sh\nexit 0\n")
    states = {str(i): dict(claim_sha256="claim" + str(i)) for i in range(5)}
    intent = dict(
        plan_sha256="replay",
        user="alice",
        qos="override-limits-but-killable",
        job_name="dispatch",
        comment="bound",
        script=str(script),
        script_sha256=file_hash(script),
    )
    for name, value in {
        "INTENT": intent,
        "STATE": dict(job_id="400", status="COMPLETE"),
        "STARTED": dict(job_id="400", plan_sha256="replay", intent_sha256=object_hash(intent)),
        "COMPLETE": dict(
            status="COMPLETE",
            reason="ALL_ENDPOINTS_CLAIMED",
            job_id="400",
            plan_sha256="replay",
            task_claims_sha256=object_hash(
                {key: row["claim_sha256"] for key, row in states.items()}
            ),
        ),
        "RELEASE_INTENT": dict(job_id="400"),
        "RELEASE": dict(returncode=0),
        "SUBMISSION": dict(returncode=0, stdout="400;cluster\n"),
        "ALLOCATION_VERIFIED": dict(
            fields=dict(
                JobId="400",
                Account="rose",
                QOS="override-limits-but-killable",
                Partition="cluster02",
                JobName="dispatch",
                Comment="bound",
                Command=str(script),
                JobState="PENDING",
                Reason="JobHeldUser",
                Requeue="0",
                UserId="alice(1000)",
                ReqTRES="cpu=4,mem=12G",
                ReqNodeList="cpu-2",
                NumCPUs="4",
                NumNodes="1",
                TimeLimit="3-00:00:00",
            )
        ),
        "ALLOCATION_AT_START": dict(
            fields=dict(
                JobId="400",
                Account="rose",
                QOS="override-limits-but-killable",
                JobName="dispatch",
                Comment="bound",
                Command=str(script),
                JobState="RUNNING",
                Requeue="0",
                UserId="alice(1000)",
                ReqTRES="cpu=1,mem=2G",
            )
        ),
    }.items():
        write(base / (name + ".json"), value)
    slurm = Slurm()
    slurm.accounting["400"] = dict(
        job_id="400",
        state="COMPLETED",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos="override-limits-but-killable",
        name="dispatch",
    )
    return SimpleNamespace(plan=plan, base=base, slurm=slurm, states=states)


def test_dispatcher_requires_real_terminal_and_zero_gpu(dispatcher):
    result = reporting._dispatcher(
        dispatcher.plan, dispatcher.slurm, {}, "alice", dispatcher.states
    )
    assert result["accounting"]["qos"] == "override-limits-but-killable"


@pytest.mark.parametrize(
    "mutation", ["missing_accounting", "failed_exit", "gpu", "missing_release", "unknown_state"]
)
def test_dispatcher_fail_closed(dispatcher, mutation):
    if mutation == "missing_accounting":
        dispatcher.slurm.accounting.clear()
    elif mutation == "failed_exit":
        dispatcher.slurm.accounting["400"]["exit_code"] = "1:0"
    elif mutation == "gpu":
        path = dispatcher.base / "ALLOCATION_AT_START.json"
        value = json.loads(path.read_text())
        value["fields"]["ReqTRES"] += ",gres/gpu=1"
        write(path, value)
    elif mutation == "missing_release":
        (dispatcher.base / "RELEASE.json").unlink()
    else:
        change(dispatcher.base / "STATE.json", status="RELEASE_UNKNOWN")
    with pytest.raises((PermissionError, FileNotFoundError)):
        reporting._dispatcher(dispatcher.plan, dispatcher.slurm, {}, "alice", dispatcher.states)


def test_replayed_task_requires_original_launch_exit_receipt(tmp_path):
    task = dict(kind="endpoint", model_id="model")
    key = reporting.task_id(task)
    spec = dict(task=task)
    plan = SimpleNamespace(
        directory=tmp_path,
        path=tmp_path / "PLAN.json",
        extension=tmp_path / "code",
        tasks={key: spec},
        sha256="plan",
        parent=SimpleNamespace(sha256="parent", plan=dict(python=sys.executable)),
    )
    result = dict(task, receipt_sha256="receipt")
    state = dict(
        task=task,
        status="COMPLETE",
        result=result,
        parent_plan_sha256="parent",
        pool_id="pool",
        job_id="301",
        device="0",
        claim_id="claim",
        claim_sha256="claim-sha",
    )
    pools = dict(pool=dict(job_id="301", devices=["0"]))
    base = tmp_path / "pools/pool/workers" / key
    launch = dict(
        task=task,
        device="0",
        claim_id="claim",
        plan_sha256="plan",
        task_definition_sha256=object_hash(spec),
        command=[
            sys.executable,
            str(plan.extension / "scripts/sr_f2/replay_endpoint.py"),
            "--plan",
            str(plan.path),
            "--task-id",
            key,
        ],
    )
    write(base / "LAUNCH_INTENT.json", launch)
    write(base / "PROCESS_EXIT.json", dict(returncode=0))
    assert reporting._replayed_task(plan, task, state, result, pools)["receipt"] == result
    change(base / "PROCESS_EXIT.json", returncode=1)
    with pytest.raises(PermissionError, match="successful process exit"):
        reporting._replayed_task(plan, task, state, result, pools)
