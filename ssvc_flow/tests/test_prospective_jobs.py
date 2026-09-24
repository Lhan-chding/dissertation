"""Prospective ordering, deduplication, and durable scheduler intent checks."""

import json
from pathlib import Path

import pytest

from src.prospective_selection.jobs import BASELINES, LEVELS, TaskRegistry


@pytest.fixture
def registry(tmp_path):
    protocol = json.loads(
        (Path(__file__).parents[1] / "docs/prospective_selection/design/protocol.json").read_text()
    )
    return TaskRegistry(tmp_path / "registry", protocol)


def frozen(r):
    p = r.protocol
    return r.write_freeze(
        dict(
            source_version="tested-commit",
            model=p["model"],
            recipes=p["actions"],
            feature_schemas=dict.fromkeys(LEVELS, "v1"),
            scaling=dict.fromkeys(LEVELS, 1),
            alphas=dict.fromkeys(LEVELS, 0.1),
            best_static="R4",
            T_panel={"panel_id": "T"},
            test_n=12,
            test_draws=16,
            primary_comparison=[LEVELS[3], LEVELS[2]],
            horizon=32,
            secondary_analyses=list(BASELINES),
            selection_rule={"practical_tie": 0.005},
            frozen_selectors={level: {"level": level, "coefficients": [1, 2]} for level in LEVELS},
        )
    )


def decision(r, **kw):
    f = r.load_freeze()
    return dict(
        lineage_id=63001,
        origin_id="63001_t32",
        origin_step=32,
        source_recipe="R0",
        freeze_id=f["freeze_id"],
        selector_ids=f["selector_ids"],
        choices=dict.fromkeys(LEVELS, "R3"),
        made_before_run=True,
        **kw,
    )


def source(seed=61001):
    return dict(lineage_id=seed, source_recipe="R0" if seed % 2 else "R1")


def branch(seed=61001, **kw):
    return {
        **source(seed),
        "origin_id": f"{seed}_t32",
        "origin_step": 32,
        "recipe_id": "R0",
        "repeat": 1,
        "schedule_id": "continuation-1-0123456789abcdef",
        "branch_schedule_seed": 73011,
        **kw,
    }


SCHEDULES = {
    str(i): {"sampler_seed": 73010 + i, "schedule_id": f"continuation-{i}-0123456789abcdef"}
    for i in (1, 2)
}


def dependencies(r, seed=61001):
    a = r.register_task("source", source(seed))
    obs = {
        **source(seed),
        "origin_id": f"{seed}_t32",
        "origin_step": 32,
        "policy_id": f"{seed}_t32",
        "panel_id": "P",
        "snapshot_t": 32,
        "draw_role": "predecision",
    }
    b = r.register_task("prestate", obs, [a["task_id"]])
    return [a["task_id"], b["task_id"]]


def regbranch(r, body, deps=None):
    return r.register_task(
        "branch", body, dependencies(r, int(body["lineage_id"])) if deps is None else deps
    )


def make_test_branches(r):
    return r.register_test_branches("63001_t32", dependencies(r, 63001), branch_schedules=SCHEDULES)


def test_duplicate_selector_does_not_create_task(registry):
    a = regbranch(registry, branch(selector_id="Z0"))
    assert a == regbranch(registry, branch(selector_id="Z3"))
    assert len(registry.tasks()) == 3
    with pytest.raises(ValueError, match="immutable"):
        regbranch(registry, branch(extra=5))


def test_dependencies(registry):
    a = registry.register_task("source", source())
    b = regbranch(registry, branch(), dependencies(registry))
    with pytest.raises(ValueError, match="dependencies"):
        registry.assert_can_execute(b["task_id"])
    registry.mark_complete(a["task_id"], {"status": "COMPLETE"})
    registry.mark_complete(b["dependencies"][1], {"status": "COMPLETE"})
    registry.assert_can_execute(b["task_id"])
    with pytest.raises(ValueError, match="already complete"):
        registry.assert_can_execute(a["task_id"])


def test_checkpoint_milestone(registry):
    a = registry.register_task("source", source())
    marker = registry.root / "sources/61001/H32.json"
    dep = {"task_id": a["task_id"], "completed_artifact": str(marker)}
    obs = {
        **source(),
        "origin_id": "61001_t32",
        "origin_step": 32,
        "policy_id": "p",
        "panel_id": "P",
        "snapshot_t": 32,
        "draw_role": "predecision",
    }
    state = registry.register_task("prestate", obs, [dep])
    b = regbranch(registry, branch(), [dep, state["task_id"]])
    with pytest.raises(ValueError, match="dependencies"):
        registry.assert_can_execute(b["task_id"])
    marker.parent.mkdir(parents=True)
    checkpoint = marker.parent / "H32.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    marker.write_text(
        json.dumps(
            {
                "step": 32,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": "fixture",
                    "identity": {},
                    "state_hash": "fixture",
                },
            }
        )
    )
    registry.mark_complete(state["task_id"], {"status": "COMPLETE"})
    registry.assert_can_execute(b["task_id"])


def test_test_source_and_branch_guard(registry):
    with pytest.raises(ValueError, match="frozen"):
        registry.register_task("source", source(63001))
    frozen(registry)
    registry.register_task("source", source(63001))
    with pytest.raises(ValueError, match="pre-existing decision"):
        regbranch(registry, branch(63001))
    assert not any(t["kind"] == "branch" for t in registry.tasks())


@pytest.mark.parametrize(
    "filename",
    ["checkpoints/H00.pt", "MANIFEST.json", "segments/block_8/COMMIT.json", "checkpoint_H32.pt"],
)
def test_preexisting_future_artifacts(registry, filename):
    frozen(registry)
    path = registry.branch_output("63001_t32", "R0", 1) / filename
    path.parent.mkdir(parents=True)
    path.write_bytes(b"started")
    with pytest.raises(ValueError, match="artifacts already exist"):
        registry.write_decision(decision(registry))


def test_exact_union_and_binding(registry):
    frozen(registry)
    registry.write_decision(decision(registry))
    tasks = make_test_branches(registry)
    recipes = {"R0", "R3", "R4", *BASELINES}
    assert {(t["payload"]["recipe_id"], t["payload"]["repeat"]) for t in tasks} == {
        (r, n) for r in recipes for n in (1, 2)
    }
    assert make_test_branches(registry) == tasks
    with pytest.raises(ValueError, match="union"):
        regbranch(registry, branch(63001, recipe_id="R7"))
    with pytest.raises(ValueError, match="repeat"):
        regbranch(registry, branch(63001, repeat=3))
    digest = registry.load_freeze()["selector_ids"][LEVELS[0]]
    (registry.root / "frozen_selectors" / f"{digest}.json").write_text("{}")
    with pytest.raises(ValueError, match="content changed"):
        registry.assert_can_execute(tasks[0]["task_id"])


def test_unregistered_test_identity(registry):
    frozen(registry)
    with pytest.raises(ValueError, match="first-N"):
        registry.register_task("source", source(63013))
    d = decision(registry)
    d["origin_step"] = 96
    with pytest.raises(ValueError, match="origin step"):
        registry.write_decision(d)
    d["origin_step"] = 32
    d["freeze_id"] = "wrong"
    with pytest.raises(ValueError, match="frozen models"):
        registry.write_decision(d)


def test_changed_decision_blocked_at_execution(registry):
    frozen(registry)
    registry.write_decision(decision(registry))
    task = make_test_branches(registry)[0]
    path = registry.root / "decisions/63001_t32.json"
    value = json.loads(path.read_text())
    value["choices"][LEVELS[0]] = "R0"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="decision changed"):
        registry.assert_can_execute(task["task_id"])


def test_submission_intent_before_callback_unknown_never_retried(registry):
    registry.register_task("source", source())
    calls = []

    def ambiguous(task, path):
        assert (registry.root / "intents" / f"{task['task_id']}.json").is_file()
        assert path.is_file()
        calls.append(task)
        raise TimeoutError("sbatch may have succeeded")

    with pytest.raises(RuntimeError, match="outcome unknown"):
        registry.submit_ready(lambda: [], ambiguous)
    assert registry.submit_ready(lambda: [], ambiguous) == []
    assert len(calls) == 1


@pytest.mark.parametrize("available,active,expected", [(5, 0, 5), (4, 0, 4), (5, 4, 1), (4, 4, 0)])
def test_pending_running_cap(registry, available, active, expected):
    for seed in range(61001, 61009):
        registry.register_task("source", source(seed))
    calls = []

    def submit(task, path):
        calls.append(task)
        return str(1000 + len(calls))

    rows = [{"job_id": str(i), "gpus": 1} for i in range(active)]
    assert len(registry.submit_ready(lambda: rows, submit, available_gpus=available)) == expected
    assert registry.submit_ready(lambda: rows, submit, available_gpus=available) == []
    assert len(calls) == expected


def test_reject_multi_gpu_and_over_cap(registry):
    with pytest.raises(ValueError, match="exactly one"):
        registry.submit_ready(lambda: [{"job_id": "1", "gpus": 2}], lambda *a: "2")
    with pytest.raises(ValueError, match=r"1\.\.5"):
        registry.submit_ready(lambda: [], lambda *a: "2", max_gpu_jobs=6)


def test_frozen_evaluation_draws(registry):
    frozen(registry)
    registry.write_decision(decision(registry))
    body = {
        **branch(63001),
        "policy_id": "R0policy",
        "panel_id": "T",
        "horizon": 32,
        "n_draws": 16,
        "draw_role": "final_evaluation",
    }
    b = regbranch(registry, branch(63001))
    registry.register_task("evaluation", body, [b["task_id"]])
    with pytest.raises(ValueError, match="panel/horizon/draws"):
        registry.register_task("evaluation", {**body, "n_draws": 64}, [b["task_id"]])


def test_missing_or_wrong_dependency_kind_rejected(registry):
    with pytest.raises(ValueError, match="explicit dependencies"):
        registry.register_task("branch", branch())
    smoke = registry.register_task("smoke", {"smoke_id": "initial"})
    with pytest.raises(ValueError, match="same lineage"):
        registry.register_task("branch", branch(), [smoke["task_id"]])


def test_schedule_is_common_across_origins(registry):
    regbranch(registry, branch())
    with pytest.raises(ValueError, match="immutable"):
        regbranch(registry, branch(61003, schedule_id="continuation-1-fedcba9876543210"))


def test_completion_requires_real_explicit_receipt(registry):
    task = registry.register_task("source", source())
    with pytest.raises(ValueError, match="COMPLETE"):
        registry.mark_complete(task["task_id"], {"status": "STARTED"})


def test_numeric_string_lineage_same_identity(registry):
    assert registry.register_task("source", source()) == registry.register_task(
        "source", dict(source(), lineage_id="61001")
    )


def test_origin_alias_cannot_duplicate_physical_run(registry):
    with pytest.raises(ValueError, match="canonical"):
        regbranch(registry, branch(origin_id="alias_of_61001_t32"))


def terminal_receipt(job_id="100", **updates):
    from datetime import datetime, timezone

    return {
        "source": "sacct",
        "job_id": job_id,
        "state": "FAILED",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "code_version": "reviewed-commit",
        "failure_class": "IMPLEMENTATION",
        **updates,
    }


def submitted_task(registry):
    task = registry.register_task("source", source())
    registry.submit_ready(lambda: [], lambda *args: "100")
    return task


def test_explicit_terminal_recovery_preserves_original_and_publishes_intent(registry):
    task = submitted_task(registry)
    tid = task["task_id"]
    original_intent = (registry.root / "intents" / f"{tid}.json").read_bytes()
    original_submission = (registry.root / "submissions" / f"{tid}.json").read_bytes()
    calls = []

    def submit(record, path):
        intent = registry.root / "recoveries" / tid / "attempt_1/intent.json"
        evidence = json.loads(intent.read_text())
        assert evidence["terminal_receipt"]["job_id"] == "100"
        assert evidence["reason"] == "Reviewed loader correction; resume committed state"
        assert path.is_file()
        calls.append(record)
        return "101;cluster02"

    receipt = registry.resubmit_terminal(
        tid,
        terminal_receipt(),
        submit,
        reason="Reviewed loader correction; resume committed state",
        query_active=lambda: [],
    )
    assert receipt["job_id"] == "101"
    assert len(calls) == 1
    assert (registry.root / "intents" / f"{tid}.json").read_bytes() == original_intent
    assert (registry.root / "submissions" / f"{tid}.json").read_bytes() == original_submission
    with pytest.raises(ValueError, match="one recovery"):
        registry.resubmit_terminal(
            tid, terminal_receipt("101"), submit, reason="second attempt", query_active=lambda: []
        )
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "UNKNOWN", "COMPLETED"])
def test_recovery_never_uses_unknown_live_or_success_state(registry, state):
    task = submitted_task(registry)
    with pytest.raises(ValueError, match="terminal failure"):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt(state=state),
            lambda *a: "101",
            reason="review",
            query_active=lambda: [],
        )
    assert not (registry.root / "recoveries").exists()


def test_recovery_requires_matching_recorded_job_and_no_live_conflict(registry):
    task = submitted_task(registry)
    with pytest.raises(ValueError, match="recorded submission job ID"):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt("999"),
            lambda *a: "101",
            reason="review",
            query_active=lambda: [],
        )
    with pytest.raises(ValueError, match="still active"):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt(),
            lambda *a: "101",
            reason="review",
            query_active=lambda: [{"job_id": "100", "gpus": 1}],
        )


def test_numerical_failure_requires_reviewed_correction(registry):
    task = submitted_task(registry)
    evidence = terminal_receipt(failure_class="NUMERICAL")
    with pytest.raises(ValueError, match="reviewed correction"):
        registry.resubmit_terminal(
            task["task_id"], evidence, lambda *a: "101", reason="review", query_active=lambda: []
        )
    evidence["reviewed_correction"] = "Reviewed nonfinite mask bug, targeted regression passed"
    assert (
        registry.resubmit_terminal(
            task["task_id"], evidence, lambda *a: "101", reason="review", query_active=lambda: []
        )["job_id"]
        == "101"
    )


def test_unknown_initial_submission_cannot_be_recovered_by_guessing_job_id(registry):
    task = registry.register_task("source", source())

    def timeout(*args):
        raise TimeoutError("unknown")

    with pytest.raises(RuntimeError, match="outcome unknown"):
        registry.submit_ready(lambda: [], timeout)
    with pytest.raises(ValueError, match="original submission outcome unknown"):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt(),
            lambda *a: "101",
            reason="review",
            query_active=lambda: [],
        )


def test_recovery_capacity_check_precedes_intent(registry):
    task = submitted_task(registry)
    active = [{"job_id": str(n), "gpus": 1} for n in range(201, 206)]
    with pytest.raises(ValueError, match="no free"):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt(),
            lambda *a: "101",
            reason="review",
            query_active=lambda: active,
        )
    assert not (registry.root / "recoveries").exists()


def test_latest_recovery_job_counted_once_not_with_absent_original(registry):
    task = submitted_task(registry)
    registry.resubmit_terminal(
        task["task_id"],
        terminal_receipt(),
        lambda *a: "101",
        reason="review",
        query_active=lambda: [],
    )
    for seed in range(61002, 61008):
        registry.register_task("source", source(seed))
    count = []

    def submit(*args):
        count.append(1)
        return str(200 + len(count))

    active = [{"job_id": "101", "gpus": 1}]
    assert len(registry.submit_ready(lambda: active, submit)) == 4
    assert registry.submit_ready(lambda: active, submit) == []


def test_unknown_recovery_is_durable_and_never_retried(registry):
    task = submitted_task(registry)
    count = []

    def timeout(*args):
        count.append(1)
        raise TimeoutError("unknown recovery")

    with pytest.raises(RuntimeError, match="recovery submission outcome unknown"):
        registry.resubmit_terminal(
            task["task_id"], terminal_receipt(), timeout, reason="review", query_active=lambda: []
        )
    with pytest.raises(ValueError, match="one recovery"):
        registry.resubmit_terminal(
            task["task_id"], terminal_receipt(), timeout, reason="review", query_active=lambda: []
        )
    assert registry.submit_ready(lambda: [], timeout) == []
    assert count == [1]


def test_test_recovery_cannot_change_frozen_source_version(registry):
    frozen(registry)
    task = registry.register_task("source", source(63001))
    registry.submit_ready(lambda: [], lambda *a: "100")
    with pytest.raises(ValueError, match="frozen source version"):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt(code_version="modified"),
            lambda *a: "101",
            reason="review",
            query_active=lambda: [],
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"source": "squeue"},
        {"failure_class": "UNKNOWN"},
        {"code_version": ""},
        {"observed_at": "2026-01-01T00:00:00"},
        {"observed_at": "invalid"},
    ],
)
def test_recovery_requires_reviewed_evidence(registry, updates):
    task = submitted_task(registry)
    with pytest.raises(ValueError):
        registry.resubmit_terminal(
            task["task_id"],
            terminal_receipt(**updates),
            lambda *a: "101",
            reason="review",
            query_active=lambda: [],
        )


def test_stale_terminal_observation_rejected(registry):
    task = submitted_task(registry)
    evidence = terminal_receipt(observed_at="2000-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="predates"):
        registry.resubmit_terminal(
            task["task_id"], evidence, lambda *a: "101", reason="review", query_active=lambda: []
        )


def test_numerical_correction_requires_text_not_boolean(registry):
    task = submitted_task(registry)
    evidence = terminal_receipt(failure_class="NUMERICAL", reviewed_correction=False)
    with pytest.raises(ValueError, match="reviewed correction"):
        registry.resubmit_terminal(
            task["task_id"], evidence, lambda *a: "101", reason="review", query_active=lambda: []
        )
