"""CPU-only identity and scheduling regressions for the separate efficiency helper."""

from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_submission_reconciliation import CHECKOUT, FROZEN, load_module, repair_env

from sr_f3_ce import endpoint_runner, protocol


@pytest.fixture
def efficient_env(tmp_path, monkeypatch):
    env = repair_env.__wrapped__(tmp_path, monkeypatch)
    env.api.protocol = protocol
    env.api.external = SimpleNamespace()
    operation = Path(env.value["identity_path"]).parent
    source = operation / "dispatch_efficiency.py"
    shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce/dispatch_efficiency.py", source)
    sources_path = operation / "OPERATION_SOURCES.json"
    sources = env.helper.read(sources_path)
    sources["files"][source.name] = env.helper.sha(source)
    env.helper.save(sources_path, sources)
    efficiency_authorization = env.root / "operations/DISPATCH_EFFICIENCY_AUTHORIZATION.json"
    env.helper.save(
        efficiency_authorization,
        dict(user_request="Avoid redundant identity scans; preserve frozen science and history."),
    )
    env.efficiency_authorization = efficiency_authorization
    env.value.update(
        operation_sources_sha256=env.helper.sha(sources_path),
        dispatch_efficiency=dict(
            schema_version=1,
            authorization_path=str(efficiency_authorization),
            authorization_sha256=env.helper.sha(efficiency_authorization),
            user_request="Remove redundant dispatcher validation while preserving frozen settings.",
            scientific_settings_changed=False,
        ),
    )
    env.helper.save(env.value["identity_path"], env.value)
    module = load_module(source, "registered_efficiency_fixture")
    counts = Counter()
    controls = SimpleNamespace(fail_gate=None)
    original_identity = env.api.controller.identity
    original_protocol = protocol.verify_run
    original_extension = env.api.extension.verify_extension

    def identity():
        counts["original_identity"] += 1
        if controls.fail_gate == "identity":
            raise PermissionError("startup original identity refused")
        return original_identity()

    def verify_run(*args, **kwargs):
        science = kwargs.get("science", False)
        counts["protocol_science" if science else "protocol_basic"] += 1
        if science and controls.fail_gate == "protocol":
            raise PermissionError("startup scientific freeze refused")
        return original_protocol(*args, **kwargs)

    def verify_extension(*args, **kwargs):
        science = kwargs.get("science", False)
        counts["extension_science" if science else "extension_basic"] += 1
        if science and controls.fail_gate == "extension":
            raise PermissionError("startup extension refused")
        return original_extension(*args, **kwargs)

    monkeypatch.setattr(env.api.controller, "identity", identity)
    monkeypatch.setattr(protocol, "verify_run", verify_run)
    monkeypatch.setattr(env.api.extension, "verify_extension", verify_extension)
    io = SimpleNamespace(sha=env.helper.sha, read=env.helper.read, save=env.helper.save)
    env.efficiency_module = module
    env.make_efficiency = lambda: module.Efficiency(env.value, env.api, io)
    env.efficiency = env.make_efficiency()
    env.api.efficiency = env.efficiency
    env.counts = counts
    env.controls = controls
    env.original_identity = identity
    env.identity_class_method = type(env.api.controller).identity
    return env


def test_check_cannot_run_before_full_startup_authentication(efficient_env):
    env = efficient_env
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.counts == Counter()
    assert env.api.controller.identity is env.original_identity
    assert env.transport.submitted == []


def test_full_startup_once_then_only_instance_small_identity_checks(efficient_env):
    env = efficient_env
    receipt = env.efficiency.start()
    expected = env.helper.sha(env.root / "RUN_IDENTITY.json")
    assert isinstance(receipt, dict)
    assert env.counts["original_identity"] == 1
    assert env.counts["protocol_science"] == 1
    assert env.counts["extension_science"] == 1
    baseline = copy.deepcopy(env.counts)
    for _ in range(30):
        assert env.efficiency.check() == expected
        assert env.api.controller.identity() == expected
    assert env.counts == baseline
    assert env.efficiency.start() == receipt
    assert env.counts == baseline
    receipt["caller_mutation"] = True
    assert "caller_mutation" not in env.efficiency.start()
    assert type(env.api.controller).identity is env.identity_class_method
    assert env.transport.submitted == []


@pytest.mark.parametrize("gate", ["identity", "protocol", "extension"])
def test_startup_validation_failure_does_not_install_adapter(efficient_env, gate):
    env = efficient_env
    env.controls.fail_gate = gate
    with pytest.raises(PermissionError, match="startup"):
        env.efficiency.start()
    assert env.api.controller.identity is env.original_identity
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


@pytest.mark.parametrize(
    "relative",
    [
        "RUN_IDENTITY.json",
        "orchestration/CONFIG.json",
        "EXECUTION_FREEZE.json",
        "deployment/SOURCE_MANIFEST.json",
        "plan/config/SR_F3_CE.json",
        "extension_manifest",
        "extension_amendment",
        "operation_identity",
        "operation_sources",
        "helper_source",
    ],
)
def test_bound_identity_bytes_changing_after_startup_prevents_submission(efficient_env, relative):
    env = efficient_env
    env.efficiency.start()
    target = {
        "extension_manifest": env.frozen.manifest,
        "extension_amendment": env.frozen.manifest.with_name("AMENDMENT.json"),
        "operation_identity": Path(env.value["identity_path"]),
        "operation_sources": Path(env.value["identity_path"]).with_name("OPERATION_SOURCES.json"),
        "helper_source": Path(env.value["identity_path"]).with_name("dispatch_efficiency.py"),
    }.get(relative, env.root / relative)
    target.write_bytes(target.read_bytes() + b" \n")
    baseline = copy.deepcopy(env.counts)
    with pytest.raises(PermissionError):
        env.efficiency.check()
    with pytest.raises(PermissionError):
        env.api.controller.identity()
    assert env.counts == baseline and env.transport.submitted == []


def test_stop_added_after_startup_prevents_submission(efficient_env):
    env = efficient_env
    env.efficiency.start()
    (env.root / "STOP").write_text("retain this operational/scientific stop")
    with pytest.raises(PermissionError):
        env.api.controller.identity()
    assert (env.root / "STOP").read_text() == "retain this operational/scientific stop"
    assert env.transport.submitted == []


@pytest.mark.parametrize("binding", ["controller_root", "controller_code", "code_symlink"])
def test_changed_runtime_root_or_code_binding_fails_closed(efficient_env, binding, tmp_path):
    env = efficient_env
    env.efficiency.start()
    other = tmp_path / "other"
    other.mkdir()
    if binding == "controller_root":
        env.api.controller.root = other
    elif binding == "controller_code":
        env.api.controller.code = other
    else:
        original = env.root / "code"
        retained = env.root / "retained-code"
        original.rename(retained)
        original.symlink_to(retained, target_is_directory=True)
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


def test_other_controller_instance_and_worker_completion_keep_full_identity(efficient_env):
    env = efficient_env
    env.efficiency.start()
    before = copy.deepcopy(env.counts)
    config = env.helper.read(env.root / "orchestration/CONFIG.json")
    other = type(env.api.controller)(
        env.root,
        env.frozen.code,
        python=config["python"],
        user=config["user"],
        pool_size=config["pool_size"],
        cpu_partition=config["cpu_policy"]["partition"],
        cpu_qos=config["cpu_policy"]["qos"],
        cpu_node=config["cpu_policy"]["cpu_node"],
        slurm=env.raw,
    )
    other.identity()
    assert env.counts["protocol_basic"] > before["protocol_basic"]
    count = env.counts["protocol_basic"]
    assert set(endpoint_runner._completed(env.root)) == env.helper.ENGINES
    assert env.counts["protocol_basic"] > count
    assert type(env.api.controller).identity is env.identity_class_method
    assert "identity" not in other.__dict__


def test_second_adapter_cannot_silently_replace_authenticated_instance(efficient_env):
    env = efficient_env
    env.efficiency.start()
    installed = env.api.controller.identity
    baseline = copy.deepcopy(env.counts)
    with pytest.raises(PermissionError):
        env.make_efficiency().start()
    assert env.api.controller.identity == installed
    assert env.counts == baseline and env.transport.submitted == []


def test_completed_artifact_validation_is_not_cached_by_identity_adapter(
    efficient_env, monkeypatch
):
    env = efficient_env
    env.efficiency.start()
    submitted = env.api.controller.submit("train")
    FROZEN.complete_new(env.frozen, submitted, monkeypatch)
    directory = env.root / "orchestration/jobs" / submitted["key"]
    intent = env.helper.read(directory / "INTENT.json")
    task = intent["task_ids"][0]
    assert task in env.api.controller.completed()
    (env.root / (task + ".raw")).write_text("corrupted after the successful completion check")
    with pytest.raises(PermissionError):
        env.api.controller.completed()
    with pytest.raises(PermissionError):
        endpoint_runner._completed(env.root)
    assert len(env.transport.submitted) == 1


@pytest.mark.parametrize(
    "relative",
    [
        "plan/PACKAGE_SHA256.json",
        "plan/manifests/MODEL_INPUTS.jsonl",
        "PROTOCOL_SELECTION.json",
        "authorization/REQUEST.md",
    ],
)
def test_absent_legacy_input_becoming_present_invalidates_startup(efficient_env, relative):
    env = efficient_env
    target = env.root / relative
    assert not target.exists()
    env.efficiency.start()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("new input was not present during startup")
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


@pytest.mark.parametrize("relative", ["RUN_IDENTITY.json", "EXECUTION_FREEZE.json"])
def test_bound_file_disappearing_fails_closed(efficient_env, relative):
    env = efficient_env
    target = env.root / relative
    env.efficiency.start()
    target.rename(target.with_suffix(".retained.json"))
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_selected_revision_cannot_start_without_required_selection_inputs(efficient_env, revision):
    env = efficient_env
    configuration = env.root / "plan/config/SR_F3_CE.json"
    config = env.helper.read(configuration)
    config["protocol_revision"] = revision
    env.helper.save(configuration, config)
    with pytest.raises(PermissionError):
        env.efficiency.start()
    assert env.counts == Counter() and env.api.controller.identity is env.original_identity


@pytest.mark.parametrize("authorization", ["authorization", "efficiency_authorization"])
def test_authorization_original_remains_a_live_binding(efficient_env, authorization):
    env = efficient_env
    env.efficiency.start()
    target = getattr(env, authorization)
    target.write_bytes(target.read_bytes() + b" \n")
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


@pytest.mark.parametrize("relative", ["PUBLICATION.json", "TYPE_AUDIT_RECEIPT.json"])
def test_typed_publication_originals_remain_live_bindings(efficient_env, relative):
    env = efficient_env
    env.efficiency.start()
    target = env.root / "external/revisions/typed_v1" / relative
    target.write_bytes(target.read_bytes() + b" \n")
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


def test_binding_change_during_full_startup_does_not_install_adapter(efficient_env, monkeypatch):
    env = efficient_env
    original = protocol.verify_run

    def concurrent_change(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("science") is True:
            target = env.root / "EXECUTION_FREEZE.json"
            target.write_bytes(target.read_bytes() + b" \n")
        return result

    monkeypatch.setattr(protocol, "verify_run", concurrent_change)
    with pytest.raises(PermissionError):
        env.efficiency.start()
    assert env.api.controller.identity is env.original_identity
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


def test_replaced_controller_identity_entry_is_rejected(efficient_env, monkeypatch):
    env = efficient_env
    env.efficiency.start()
    monkeypatch.setattr(env.api.controller, "identity", lambda: env.value["run_identity_sha256"])
    with pytest.raises(PermissionError):
        env.efficiency.check()
    assert env.transport.submitted == []


def test_efficiency_requires_explicit_nonempty_authorization(efficient_env):
    env = efficient_env
    env.value["dispatch_efficiency"] = {}
    with pytest.raises(PermissionError):
        env.make_efficiency().start()
    assert env.api.controller.identity is env.original_identity and env.counts == Counter()


def fast_driver(env):
    return env.helper.Driver(
        env.value,
        env.api,
        env.root / "operations/dispatcher/attempts/0002",
        typed=lambda *args: env.frozen.typed_verified,
    )


def add_cpu_dispatcher(env):
    env.raw.rows["dispatcher"] = dict(
        JobId="dispatcher",
        JobName="SRF3CE-DISPATCH-N0002-LH",
        Comment="cpu-dispatcher",
        QOS=env.helper.QOS,
        UserId="alice(1)",
        ReqTRES="cpu=4,mem=12G,node=1",
        AllocTRES="cpu=4,mem=12G,node=1",
        JobState="RUNNING",
    )


def forbid_controller_walks(env, monkeypatch, *, forbid_submit=False):
    def forbidden(*args, **kwargs):
        raise AssertionError("The capacity-first path must not repeat a full controller walk")

    for name in ("status", "tick") + (("submit",) if forbid_submit else ()):
        monkeypatch.setattr(env.api.controller, name, forbidden)


def test_full_capacity_poll_does_not_walk_complete_history_or_submit(efficient_env, monkeypatch):
    from test_submission_reconciliation import unrelated_queue

    env = efficient_env
    env.efficiency.start()
    unrelated_queue(env, 4)
    add_cpu_dispatcher(env)
    driver = fast_driver(env)
    baseline = copy.deepcopy(env.counts)
    forbid_controller_walks(env, monkeypatch, forbid_submit=True)
    assert driver.poll() == "WAIT_CAPACITY"
    for _ in range(3):
        assert driver.poll() in {"WAIT_CAPACITY", "UNCHANGED"}
    assert env.counts == baseline and env.transport.submitted == []


def test_available_slots_fill_frozen_task_order_without_status_tick_or_duplicates(
    efficient_env, monkeypatch
):
    from test_submission_reconciliation import unrelated_queue

    env = efficient_env
    env.efficiency.start()
    unrelated_queue(env, 2)
    add_cpu_dispatcher(env)
    driver = fast_driver(env)
    forbid_controller_walks(env, monkeypatch)
    driver.poll()
    assert len(env.transport.submitted) == 2
    intents = [
        env.helper.read(Path(command[-1]).with_name("INTENT.json"))
        for command in env.transport.submitted
    ]
    expected = [
        key for key, task in env.api.orchestration.TASKS.items() if task["kind"] == "train"
    ][:2]
    assert [intent["task_ids"] for intent in intents] == [[key] for key in expected]
    assert len({intent["key"] for intent in intents}) == 2
    assert all(
        intent["gpu_count"] == 1 and intent["job_name"].endswith("-LH") for intent in intents
    )
    assert env.api.orchestration.teacher_usage(
        env.raw.queue("alice"), env.repair.effective_states(), "alice"
    ) == dict(gpus=4, submitted=5)
    for _ in range(3):
        driver.poll()
    assert len(env.transport.submitted) == 2
    assert env.counts["original_identity"] == 1


def test_slot_freed_inside_original_submit_retries_even_if_poll_signature_is_unchanged(
    efficient_env, monkeypatch
):
    from test_submission_reconciliation import unrelated_queue

    env = efficient_env
    env.efficiency.start()
    unrelated_queue(env, 3)
    add_cpu_dispatcher(env)
    driver = fast_driver(env)
    forbid_controller_walks(env, monkeypatch)
    original_submit = env.api.controller.submit
    original_ready = env.api.controller.ready_tasks
    calls = []

    def raced_submit(kind, **kwargs):
        calls.append(kind)
        if len(calls) != 1:
            return original_submit(kind, **kwargs)
        env.raw.rows["late"] = dict(env.raw.rows["9000"], JobId="late", JobName="other-late")

        def slot_frees_after_queue_snapshot():
            result = original_ready()
            env.raw.rows.pop("late", None)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(env.api.controller, "ready_tasks", slot_frees_after_queue_snapshot)
            result = original_submit(kind, **kwargs)
        assert result is None
        return result

    monkeypatch.setattr(env.api.controller, "submit", raced_submit)
    driver.poll()
    assert calls == ["train"] and env.transport.submitted == []
    previous_signature = driver.signature
    assert env.helper.digest(driver.snapshot()) == previous_signature
    driver.poll()
    assert calls == ["train", "train"] and len(env.transport.submitted) == 1
    driver.poll()
    assert len(env.transport.submitted) == 1


def test_capacity_consumed_during_intent_verification_defers_without_failing_driver(
    efficient_env, monkeypatch
):
    from test_submission_reconciliation import unrelated_queue

    env = efficient_env
    env.efficiency.start()
    unrelated_queue(env, 3)
    add_cpu_dispatcher(env)
    driver = fast_driver(env)
    forbid_controller_walks(env, monkeypatch)
    original_verify = env.frozen.module._verify_intent

    def consumes_capacity(root, intent):
        result = original_verify(root, intent)
        env.raw.rows["late"] = dict(env.raw.rows["9000"], JobId="late", JobName="other-late")
        return result

    monkeypatch.setattr(env.frozen.module, "_verify_intent", consumes_capacity)
    assert driver.poll() == "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
    assert env.transport.submitted == []
    results = list(env.repair.base.glob("*/attempts/*/RESULT.json"))
    assert len(results) == 1
    result = env.helper.read(results[0])
    assert result["status"] == "LOCAL_CAPACITY_DEFERRED" and result["sbatch_invoked"] is False
    assert not (driver.directory / "END.json").exists()


@pytest.mark.parametrize("all_complete", [False, True])
def test_complete_snapshot_hint_requires_full_status_before_delivery(
    efficient_env, monkeypatch, all_complete
):
    env = efficient_env
    env.efficiency.start()
    driver = fast_driver(env)
    snapshot = driver.snapshot()
    snapshot["jobs"] = [
        dict(
            key=key,
            status="COMPLETE",
            kind=task["kind"],
            task_ids=[key],
            terminal=None,
            unknown=None,
            job_id=None,
        )
        for key, task in env.api.orchestration.TASKS.items()
    ]
    monkeypatch.setattr(driver, "snapshot", lambda: snapshot)
    observed = []
    delivered = []

    def verified_status():
        observed.append("full-status")
        return dict(
            all_complete=all_complete,
            blocked=False,
            completed=list(env.helper.ENGINES),
            ready=[],
            attempts=[],
        )

    def deliver():
        assert observed == ["full-status"]
        delivered.append(True)
        return "DONE"

    monkeypatch.setattr(env.api.controller, "status", verified_status)
    monkeypatch.setattr(driver, "deliver", deliver)
    if all_complete:
        assert driver.poll() == "DONE"
    else:
        with pytest.raises(RuntimeError, match="verification is incomplete"):
            driver.poll()
    assert observed == ["full-status"]
    assert delivered == ([True] if all_complete else [])
    assert env.transport.submitted == []


def test_no_candidate_while_tasks_run_waits_without_full_status(efficient_env, monkeypatch):
    env = efficient_env
    env.efficiency.start()
    driver = fast_driver(env)
    monkeypatch.setattr(driver, "candidate_kind", lambda snapshot: None)
    forbid_controller_walks(env, monkeypatch, forbid_submit=True)
    baseline = copy.deepcopy(env.counts)
    assert driver.poll() == "WAIT_FIXED_TASKS"
    assert env.counts == baseline and env.transport.submitted == []


def test_running_cpu_wait_loop_uses_small_checks_and_preserves_lease_handoff(
    efficient_env, monkeypatch
):
    from test_submission_reconciliation import unrelated_queue

    env = efficient_env
    # Establish the ordinary previous repair allocation before exercising the
    # efficiency runtime. The script and lease still bind the exact current bytes.
    prior_registration = dict(env.value)
    prior_registration.pop("dispatch_efficiency")
    state = env.helper.submit_driver(prior_registration, env.api, 2, previous=1)
    cpu = env.raw.rows[state["job_id"]]
    cpu.update(AllocTRES=cpu["ReqTRES"], NodeList="cpu-2", RunTime="00:00:00")
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    unrelated_queue(env, 4)
    env.efficiency.start()
    baseline = copy.deepcopy(env.counts)
    submitted = len(env.transport.submitted)
    forbid_controller_walks(env, monkeypatch, forbid_submit=True)

    def forbidden_identity(*args, **kwargs):
        raise AssertionError("The wait loop must not replay the outer full operation identity")

    monkeypatch.setattr(env.helper, "identity", forbidden_identity)
    waits = []

    def next_poll(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            cpu["RunTime"] = "2-23:50:00"
        assert len(waits) <= 2

    monkeypatch.setattr(env.helper.time, "sleep", next_poll)
    assert env.helper.run(env.value, env.api, 2) == "HANDOFF_REQUIRED"
    assert waits == [env.value["poll_seconds"], env.value["poll_seconds"]]
    assert env.counts == baseline and len(env.transport.submitted) == submitted
    end = env.helper.read(env.helper.attempts_root(env.value) / "0002/END.json")
    assert end["scientific_time_budget_changed"] is False
    assert not (env.root / "STOP").exists()


@pytest.fixture
def retired_driver(tmp_path, monkeypatch):
    env = repair_env.__wrapped__(tmp_path, monkeypatch)
    state = env.helper.submit_driver(env.value, env.api, 2, previous=1)
    prior = env.helper.attempts_root(env.value) / "0002"
    intent = env.helper.read(prior / "INTENT.json")
    env.helper.save(prior / "STARTED.json", dict(job_id=state["job_id"], at=1))
    env.raw.rows.pop(state["job_id"])
    terminal = dict(
        job_id=state["job_id"],
        state="CANCELLED",
        exit_code="0:15",
        user="alice",
        account="rose",
        qos=env.helper.QOS,
        name=intent["job_name"],
    )
    env.raw.accounting[state["job_id"]] = terminal
    transition = env.root / "operations/efficiency_handoff/TRANSITION.json"
    env.helper.save(
        transition,
        dict(
            status="OPERATOR_HANDOFF_CANCELLED",
            job_id=state["job_id"],
            orchestration_lease_held=True,
            queue_absent=True,
            cancel_returncode=0,
            terminal=terminal,
        ),
    )
    authorization = env.root / "operations/efficiency_handoff/AUTHORIZATION.json"
    env.helper.save(
        authorization,
        dict(
            schema_version=1,
            scope="SRF3CE_DISPATCH_EFFICIENCY",
            scientific_settings_changed=False,
            new_scientific_tasks=0,
            user_request="Replace the delayed CPU dispatcher; preserve science.",
            run_identity_sha256=env.helper.sha(env.root / "RUN_IDENTITY.json"),
            execution_freeze_sha256=env.helper.sha(env.root / "EXECUTION_FREEZE.json"),
            predecessor=dict(
                sequence=2,
                job_id=state["job_id"],
                identity_path=str(Path(env.value["identity_path"]).relative_to(env.root)),
                identity_sha256=env.helper.sha(env.value["identity_path"]),
                intent_sha256=env.helper.sha(prior / "INTENT.json"),
                state_sha256=env.helper.sha(prior / "STATE.json"),
                end_sha256=None,
            ),
            transition_path=str(transition.relative_to(env.root)),
            transition_sha256=env.helper.sha(transition),
            terminal=terminal,
        ),
    )
    env.value["dispatch_efficiency"] = env.helper.check_efficiency_authorization(
        env.root, authorization
    )
    env.retired = SimpleNamespace(
        state=state,
        intent=intent,
        directory=prior,
        terminal=terminal,
        transition=transition,
        authorization=authorization,
    )
    return env


def test_only_exact_fenced_cancelled_cpu_predecessor_can_continue(retired_driver):
    env = retired_driver
    prior = env.retired
    before = {p: p.read_bytes() for p in prior.directory.iterdir() if p.is_file()}
    submitted = len(env.transport.submitted)
    assert (
        env.helper.verify_efficiency_predecessor(
            env.value, env.api, prior.directory, prior.intent, prior.state, env.raw.queue("alice")
        )
        == prior.terminal
    )
    assert len(env.transport.submitted) == submitted
    assert all(p.read_bytes() == body for p, body in before.items())


@pytest.mark.parametrize(
    "mutation",
    [
        "active",
        "accounting_unknown",
        "accounting_changed",
        "wrong_sequence",
        "wrong_job",
        "wrong_intent",
        "authorization",
        "prior_state",
        "new_end",
        "transition",
    ],
)
def test_cpu_efficiency_transition_rejects_active_unknown_or_changed_history(
    retired_driver, mutation
):
    env = retired_driver
    prior = env.retired
    directory, intent, state = prior.directory, dict(prior.intent), dict(prior.state)
    if mutation == "active":
        env.raw.rows[state["job_id"]] = {"JobId": state["job_id"]}
    elif mutation == "accounting_unknown":
        del env.raw.accounting[state["job_id"]]
    elif mutation == "accounting_changed":
        env.raw.accounting[state["job_id"]] = dict(prior.terminal, state="RUNNING")
    elif mutation == "wrong_sequence":
        directory = directory.with_name("0001")
    elif mutation == "wrong_job":
        state["job_id"] = "888888"
    elif mutation == "wrong_intent":
        intent["operation_identity_sha256"] = "0" * 64
    elif mutation == "new_end":
        env.helper.save(directory / "END.json", dict(status="FAILED"))
    else:
        target = {
            "authorization": prior.authorization,
            "prior_state": directory / "STATE.json",
            "transition": prior.transition,
        }[mutation]
        target.write_bytes(target.read_bytes() + b" \n")
    submitted = len(env.transport.submitted)
    with pytest.raises(PermissionError):
        env.helper.verify_efficiency_predecessor(
            env.value, env.api, directory, intent, state, env.raw.queue("alice")
        )
    assert len(env.transport.submitted) == submitted
