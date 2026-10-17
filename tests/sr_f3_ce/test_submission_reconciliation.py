"""CPU-only regression of audited submission repair against the frozen controller.

Fake Slurm owns every allocation in these fixtures. The existing instrumented
worker fixture writes synthetic artifacts; no model or network call is made.
"""

from __future__ import annotations

import copy
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import endpoint_runner, orchestration

CHECKOUT = Path(__file__).resolve().parents[2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FROZEN = load_module(
    Path(__file__).with_name("test_instrumented_execution.py"), "repair_frozen_fixture"
)


@pytest.fixture
def repair_env(tmp_path, monkeypatch):
    frozen = FROZEN.setup.__wrapped__(tmp_path, monkeypatch)
    FROZEN.seal_and_freeze(frozen)
    helper = load_module(
        CHECKOUT / "scripts/sr_f3_ce/dispatch_registered.py", "repair_dispatch_fixture"
    )
    root = frozen.root
    original_preparer = root / "operations" / helper.TYPED_COMMIT / "finalize_external_types.py"
    original_preparer.parent.mkdir(parents=True)
    shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce/finalize_external_types.py", original_preparer)

    def register(commit, authorization=None):
        operation = root / "operations" / commit
        operation.mkdir()
        for name in helper.HELPERS:
            shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce" / name, operation / name)
        monkeypatch.setattr(helper, "__file__", str(operation / "dispatch_registered.py"))
        identity_path = helper.register(
            root, sys.executable, "alice", commit, frozen.manifest, authorization
        )
        return dict(helper.identity(identity_path), identity_path=str(identity_path))

    old = register("c" * 40)
    api = SimpleNamespace(
        orchestration=orchestration,
        controller=frozen.controller,
        slurm=frozen.slurm,
        extension=frozen.module,
    )
    old_state = helper.submit_driver(old, api, 1)
    prior = helper.attempts_root(old) / "0001"
    helper.save(prior / "STARTED.json", dict(job_id=old_state["job_id"], at=1))
    helper.save(prior / "END.json", dict(status="FAILED", error="retained original failure", at=2))
    fields = frozen.slurm.rows.pop(old_state["job_id"])
    frozen.slurm.accounting[old_state["job_id"]] = dict(
        job_id=old_state["job_id"],
        state="FAILED",
        exit_code="1:0",
        user="alice",
        account="rose",
        qos=helper.QOS,
        name=fields["JobName"],
    )
    original_records = {p.name: p.read_bytes() for p in prior.iterdir() if p.is_file()}
    module = load_module(
        CHECKOUT / "scripts/sr_f3_ce/reconcile_submissions.py", "repair_test_module"
    )
    authorization = root / "operations/SUBMISSION_REPAIR_AUTHORIZATION.json"
    helper.save(
        authorization,
        dict(
            schema_version=module.SCHEMA,
            scope="SRF3CE_OPERATIONAL_SUBMISSION_REPAIR",
            authorized_actions=module.ACTIONS,
            scientific_settings_changed=False,
            run_identity_sha256=helper.sha(root / "RUN_IDENTITY.json"),
            execution_freeze_sha256=helper.sha(root / "EXECUTION_FREEZE.json"),
            user_request="Repair the dispatcher and ledger; preserve settings and history.",
            predecessor=dict(
                sequence=1,
                job_id=old_state["job_id"],
                identity_path=old["identity_path"],
                identity_sha256=helper.sha(old["identity_path"]),
                records={name: module._hash(body) for name, body in original_records.items()},
            ),
        ),
    )
    value = register("d" * 40, authorization)
    io = SimpleNamespace(
        read=helper.read, save=helper.save, sha=helper.sha, state_rows=helper.state_rows
    )
    repair = module.Repair(value, api, io)
    api.repair = repair
    raw = frozen.slurm
    original_run = raw.run
    transport = SimpleNamespace(responses=[], submitted=[], accounting_response=None, events=[])

    def run(command):
        if command[0] == "sacct":
            transport.events.append("accounting")
            response = transport.accounting_response
            if callable(response):
                return response(command)
            return response or SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[0] == "sbatch":
            transport.submitted.append(copy.deepcopy(command))
            transport.events.append("sbatch")
            if transport.responses:
                response = transport.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                if callable(response):
                    return response(command)
                return response
        return original_run(command)

    monkeypatch.setattr(raw, "run", run)
    return SimpleNamespace(
        frozen=frozen,
        root=root,
        helper=helper,
        module=module,
        repair=repair,
        value=value,
        api=api,
        raw=raw,
        transport=transport,
        authorization=authorization,
        prior=prior,
        original_records=original_records,
        old_state=old_state,
    )


def explicit_response(env):
    return SimpleNamespace(returncode=1, stdout="", stderr=env.module.QOS_REJECTION + "\n")


def legacy_rejection(env, response=None):
    """Create the real historical unknown state without the new operational guard."""
    env.transport.responses.append(response or explicit_response(env))
    env.api.controller.slurm = env.raw
    try:
        state = env.api.controller.submit("train")
    finally:
        env.api.controller.slurm = env.repair.slurm
    assert state["status"] == "SUBMISSION_UNKNOWN" and not state.get("job_id")
    directory = env.root / "orchestration/jobs" / state["key"]
    return state, env.helper.read(directory / "INTENT.json"), directory


def unrelated_queue(env, count, *, user="alice", gpus=1):
    for index in range(count):
        job = str(9000 + index)
        env.raw.rows[job] = dict(
            JobId=job,
            JobName="unrelated-" + job,
            Comment="other-" + job,
            QOS=env.helper.QOS,
            UserId=user + "(99)",
            ReqTRES=f"cpu=4,gres/gpu={gpus}",
            AllocTRES="(null)",
            JobState="PENDING",
        )


def journal(env, key, sequence=1):
    return env.repair.base / key / "attempts" / f"{sequence:04d}"


def test_rejected_claim_retry_completes_with_original_endpoint_verifier(repair_env, monkeypatch):
    env = repair_env
    state, intent, directory = legacy_rejection(env)
    originals = {
        name: (directory / name).read_bytes()
        for name in ("INTENT.json", "STATE.json", "SUBMISSION.json", "run.sbatch")
    }
    freeze_before = (env.root / "EXECUTION_FREEZE.json").read_bytes()
    assert env.repair.prepare()["status"] == "PREPARED_NO_NEW_SUBMISSIONS"
    assert len(env.transport.submitted) == 1
    assert env.repair.retry_pending() is None
    retried = env.helper.read(directory / "STATE.json")
    assert retried["key"] == state["key"] and retried["status"] == "SUBMITTED"
    assert env.helper.read(directory / "INTENT.json")["claim_id"] == intent["claim_id"]
    assert (journal(env, state["key"]) / "BEFORE_STATE.raw.json").read_bytes() == originals[
        "STATE.json"
    ]
    for name in ("INTENT.json", "SUBMISSION.json", "run.sbatch"):
        assert (directory / name).read_bytes() == originals[name]
    receipt = env.helper.read(journal(env, state["key"]) / "RESULT.json")
    assert receipt["status"] == "ACCEPTED" and receipt["job_id"] == retried["job_id"]
    FROZEN.complete_new(env.frozen, retried, monkeypatch)
    assert intent["task_ids"][0] in endpoint_runner._completed(env.root)
    assert env.repair.retry_pending() is None and len(env.transport.submitted) == 2
    assert (env.root / "EXECUTION_FREEZE.json").read_bytes() == freeze_before


@pytest.mark.parametrize("user,gpus", [("alice", 1), ("alice", 0), ("bob", 1)])
def test_fresh_queue_after_slow_verification_defers_before_sbatch(
    repair_env, monkeypatch, user, gpus
):
    env = repair_env
    original_verify = env.frozen.module._verify_intent

    def slow_verify(root, intent):
        result = original_verify(root, intent)
        env.transport.events.append("verified")
        unrelated_queue(env, 5, user=user, gpus=gpus)
        return result

    monkeypatch.setattr(env.frozen.module, "_verify_intent", slow_verify)
    state = env.api.controller.submit("train")
    assert state["status"] == "SUBMISSION_UNKNOWN" and env.transport.submitted == []
    directory = journal(env, state["key"])
    result = env.helper.read(directory / "RESULT.json")
    assert result["status"] == "LOCAL_CAPACITY_DEFERRED" and result["sbatch_invoked"] is False
    assert not (directory / "MAY_HAVE_SUBMITTED.json").exists()
    capacity = env.helper.read(directory / "QUEUE_IMMEDIATELY_BEFORE_SUBMIT.json")
    assert max(capacity["teacher_usage"].values()) == 6
    assert capacity["includes_candidate"] is True
    candidate = next(
        row for row in env.helper.state_rows(env.root) if row[1]["key"] == state["key"]
    )
    assert env.repair.rejection(*candidate) is not None


def test_retry_rechecks_capacity_after_accounting_crosscheck(repair_env):
    env = repair_env
    state, _, _ = legacy_rejection(env)

    def slow_accounting(_):
        unrelated_queue(env, 5)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    env.transport.accounting_response = slow_accounting
    assert env.repair.retry_pending() == "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
    assert len(env.transport.submitted) == 1
    assert env.helper.read(journal(env, state["key"]) / "RESULT.json")["sbatch_invoked"] is False


@pytest.mark.parametrize("retry", [False, True])
def test_projected_exactly_five_counts_candidate_once(repair_env, retry):
    env = repair_env
    if retry:
        state, _, directory = legacy_rejection(env)
    unrelated_queue(env, 4)
    if retry:
        assert env.repair.retry_pending() is None
        state = env.helper.read(directory / "STATE.json")
    else:
        state = env.api.controller.submit("train")
    assert state["status"] == "SUBMITTED"
    capacity = env.helper.read(journal(env, state["key"]) / "QUEUE_IMMEDIATELY_BEFORE_SUBMIT.json")
    assert capacity["teacher_usage"] == dict(gpus=5, submitted=5)
    assert len(env.transport.submitted) == 1 + int(retry)


def test_local_deferral_and_explicit_qos_rejection_can_wait_then_retry(repair_env):
    env = repair_env
    state, _, directory = legacy_rejection(env)
    original = (directory / "SUBMISSION.json").read_bytes()
    unrelated_queue(env, 5)
    assert env.repair.retry_pending() == "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
    assert len(env.transport.submitted) == 1
    env.raw.rows.clear()
    env.transport.responses.append(explicit_response(env))
    assert env.repair.retry_pending() == "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
    assert len(env.transport.submitted) == 2
    assert env.repair.retry_pending() is None
    assert len(env.transport.submitted) == 3
    results = [
        env.helper.read(journal(env, state["key"], number) / "RESULT.json")["status"]
        for number in (1, 2, 3)
    ]
    assert results == ["LOCAL_CAPACITY_DEFERRED", "EXPLICIT_QOS_REJECTION", "ACCEPTED"]
    assert (directory / "SUBMISSION.json").read_bytes() == original
    assert env.helper.read(directory / "STATE.json")["status"] == "SUBMITTED"


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(returncode=1, stdout="", stderr="connection timed out"),
        SimpleNamespace(returncode=0, stdout="208999\n", stderr=""),
        SimpleNamespace(returncode=1, stdout="208999\n", stderr="transport failed"),
    ],
)
def test_ambiguous_or_accepted_original_response_is_never_retried(repair_env, response):
    env = repair_env
    if response.returncode == 0:
        # Crash after an accepted reply but before assigning the job id to STATE.
        state, _, directory = legacy_rejection(env)
        env.helper.save(directory / "SUBMISSION.json", vars(response))
    else:
        state, _, directory = legacy_rejection(env, response)
    before = (directory / "STATE.json").read_bytes()
    with pytest.raises(RuntimeError, match="unknown"):
        env.repair.retry_pending()
    assert (directory / "STATE.json").read_bytes() == before
    assert len(env.transport.submitted) == 1 and not journal(env, state["key"]).exists()


@pytest.mark.parametrize(
    "outcome",
    [
        TimeoutError("SSH response lost"),
        SimpleNamespace(returncode=1, stdout="", stderr="transport closed"),
    ],
)
def test_retry_transport_uncertainty_is_journaled_and_not_repeated(repair_env, outcome):
    env = repair_env
    state, _, _ = legacy_rejection(env)
    env.transport.responses.append(outcome)
    with pytest.raises((TimeoutError, RuntimeError)):
        env.repair.retry_pending()
    retained = {p.name: p.read_bytes() for p in journal(env, state["key"]).iterdir() if p.is_file()}
    with pytest.raises(RuntimeError, match="unknown"):
        env.repair.retry_pending()
    assert len(env.transport.submitted) == 2
    assert all(
        (journal(env, state["key"]) / name).read_bytes() == body for name, body in retained.items()
    )
    assert not journal(env, state["key"], 2).exists()


def test_accepted_retry_crash_does_not_submit_again(repair_env, monkeypatch):
    env = repair_env
    state, _, directory = legacy_rejection(env)
    original_save = env.repair.io.save

    def crash_before_state(path, value, **kwargs):
        if Path(path) == directory / "STATE.json" and value.get("status") == "HELD":
            raise OSError("simulated crash before state assignment")
        return original_save(path, value, **kwargs)

    monkeypatch.setattr(env.repair.io, "save", crash_before_state)
    with pytest.raises(OSError, match="simulated crash"):
        env.repair.retry_pending()
    with pytest.raises(RuntimeError, match="unknown"):
        env.repair.retry_pending()
    assert len(env.transport.submitted) == 2
    assert env.helper.read(journal(env, state["key"]) / "ACCEPTED.json")["job_id"] in env.raw.rows
    assert not journal(env, state["key"], 2).exists()


@pytest.mark.parametrize(
    "record",
    [
        "ALLOCATION_VERIFIED.json",
        "RELEASE_INTENT.json",
        "RELEASE.json",
        "STARTED.json",
        "COMPLETE.json",
        "TERMINAL.json",
        "SUBMISSION_EXCEPTION.json",
        "task.LAUNCH.json",
        "task.EXIT.json",
        "task",
        "observer",
        "training",
    ],
)
def test_any_execution_or_output_evidence_blocks_same_claim_retry(repair_env, record):
    env = repair_env
    state, intent, directory = legacy_rejection(env)
    if record == "task":
        target = env.root / "tasks" / intent["task_ids"][0]
    elif record == "observer":
        target = env.root / "accounting/observability" / state["key"]
    elif record == "training":
        model = orchestration.TASKS[intent["task_ids"][0]]["model_id"]
        target = env.root / "runs" / model
    else:
        target = directory / record
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"retained execution evidence")
    with pytest.raises(PermissionError):
        env.repair.retry_pending()
    assert (
        target.read_bytes() == b"retained execution evidence" and len(env.transport.submitted) == 1
    )


@pytest.mark.parametrize(
    "evidence",
    [
        "queue_name",
        "queue_comment",
        "accounting_name",
        "accounting_comment",
        "accounting_unavailable",
        "accounting_malformed",
    ],
)
def test_matching_or_unavailable_scheduler_evidence_blocks_retry(repair_env, evidence):
    env = repair_env
    _, intent, _ = legacy_rejection(env)
    if evidence.startswith("queue"):
        unrelated_queue(env, 1)
        field = "JobName" if evidence.endswith("name") else "Comment"
        env.raw.rows["9000"][field] = intent["job_name" if field == "JobName" else "comment"]
    elif evidence == "accounting_unavailable":
        env.transport.accounting_response = SimpleNamespace(
            returncode=1, stdout="", stderr="unavailable"
        )
    else:
        text = (
            "malformed"
            if evidence == "accounting_malformed"
            else "9000|{}|{}\n".format(
                intent["job_name"] if evidence.endswith("name") else "other",
                intent["comment"] if evidence.endswith("comment") else "other",
            )
        )
        env.transport.accounting_response = SimpleNamespace(returncode=0, stdout=text, stderr="")
    with pytest.raises((RuntimeError, PermissionError)):
        env.repair.retry_pending()
    assert len(env.transport.submitted) == 1


def test_completed_job_reconciliation_only_verifies_and_preserves_before_state(
    repair_env, monkeypatch
):
    env = repair_env
    state = env.api.controller.submit("train")
    directory = env.root / "orchestration/jobs" / state["key"]
    before = (directory / "STATE.json").read_bytes()
    FROZEN.complete_new(env.frozen, state, monkeypatch)
    # Model the durable ledger not having been refreshed when the CPU driver died.
    (directory / "STATE.json").write_bytes(before)
    submitted = len(env.transport.submitted)
    assert env.repair.reconcile_completed() == [state["key"]]
    saved = env.repair.base / state["key"] / "completion"
    assert (saved / "BEFORE_STATE.raw.json").read_bytes() == before
    assert env.helper.read(saved / "RECONCILIATION.json")["new_model_calls"] == 0
    assert env.helper.read(directory / "STATE.json")["status"] == "COMPLETE"
    assert env.repair.reconcile_completed() == []
    assert len(env.transport.submitted) == submitted
    assert "train--SRF3CE_RL_b0" in endpoint_runner._completed(env.root)


def test_bad_completed_output_cannot_be_reconciled(repair_env, monkeypatch):
    env = repair_env
    state = env.api.controller.submit("train")
    directory = env.root / "orchestration/jobs" / state["key"]
    before = (directory / "STATE.json").read_bytes()
    FROZEN.complete_new(env.frozen, state, monkeypatch)
    (directory / "STATE.json").write_bytes(before)
    intent = env.helper.read(directory / "INTENT.json")
    (env.root / (intent["task_ids"][0] + ".raw")).write_bytes(b"corrupted retained artifact")
    with pytest.raises(PermissionError):
        env.repair.reconcile_completed()
    assert (directory / "STATE.json").read_bytes() == before
    assert not (env.repair.base / state["key"] / "completion/RECONCILIATION.json").exists()


def test_cpu_repair_has_new_sequence_and_lease_and_preserves_failed_originals(repair_env):
    env = repair_env
    assert env.repair.prepare()["new_model_calls"] == 0
    state = env.helper.submit_driver(env.value, env.api, 2, previous=1)
    directory = env.helper.attempts_root(env.value) / "0002"
    intent = env.helper.read(directory / "INTENT.json")
    assert state["status"] == "SUBMITTED" and state["job_id"] != env.old_state["job_id"]
    assert intent["previous"] == 1 and intent["gpu_count"] == 0
    assert intent["job_name"] == "SRF3CE-DISPATCH-N0002-LH" and "--time=4320" in intent["sbatch"]
    assert (
        str(directory) in intent["script"]
        and env.value["identity_path"] in (directory / "run.sbatch").read_text()
    )
    assert not (directory / "STARTED.json").exists()
    assert all(
        (env.prior / name).read_bytes() == body for name, body in env.original_records.items()
    )
    assert (
        env.helper.read(env.repair.base / "DRIVER_PREDECESSOR_VERIFIED.json")["terminal"]["state"]
        == "FAILED"
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        env.helper.submit_driver(env.value, env.api, 2, previous=1)
    assert len(env.transport.submitted) == 1


@pytest.mark.parametrize(
    "mutation", ["active", "unknown", "completed", "wrong_exit", "wrong_user", "wrong_name"]
)
def test_cpu_repair_requires_bound_live_failed_predecessor(repair_env, mutation):
    env = repair_env
    job = env.old_state["job_id"]
    terminal = env.raw.accounting[job]
    if mutation == "active":
        env.raw.rows[job] = {"JobId": job}
    elif mutation == "unknown":
        del env.raw.accounting[job]
    else:
        field, value = {
            "completed": ("state", "COMPLETED"),
            "wrong_exit": ("exit_code", "0:0"),
            "wrong_user": ("user", "bob"),
            "wrong_name": ("name", "unrelated"),
        }[mutation]
        terminal[field] = value
    with pytest.raises(PermissionError):
        env.helper.submit_driver(env.value, env.api, 2, previous=1)
    assert (
        env.transport.submitted == []
        and not (env.helper.attempts_root(env.value) / "0002").exists()
    )


@pytest.mark.parametrize(
    "record", ["authorization", "predecessor", "operation_source", "freeze", "scientific_source"]
)
def test_source_authorization_and_frozen_identity_changes_fail_closed(repair_env, record):
    env = repair_env
    target = {
        "authorization": env.authorization,
        "predecessor": env.prior / "END.json",
        "operation_source": Path(env.value["identity_path"]).with_name("reconcile_submissions.py"),
        "freeze": env.root / "EXECUTION_FREEZE.json",
        "scientific_source": env.frozen.manifest.with_name("observe_runtime.py"),
    }[record]
    target.write_bytes(target.read_bytes() + b" \n")
    with pytest.raises(PermissionError):
        env.helper.identity(env.value["identity_path"])
    assert env.transport.submitted == []


@pytest.mark.parametrize(
    "record",
    [
        "BEFORE_STATE.raw.json",
        "INTENT.json",
        "RESULT.json",
        "MAY_HAVE_SUBMITTED.json",
        "original_submission",
    ],
)
def test_rejection_journal_tampering_fails_closed(repair_env, record):
    env = repair_env
    state, _, directory = legacy_rejection(env)
    env.transport.responses.append(explicit_response(env))
    assert env.repair.retry_pending() == "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
    target = (
        directory / "SUBMISSION.json"
        if record == "original_submission"
        else journal(env, state["key"]) / record
    )
    value = env.helper.read(target)
    if record in {"INTENT.json", "MAY_HAVE_SUBMITTED.json"}:
        value["command"] = ["sbatch", "unregistered"]
    elif record == "RESULT.json":
        value["submission_kind"] = "unregistered"
    elif record == "original_submission":
        value["stderr"] = "changed original rejection"
    else:
        value["status"] = "HELD"
    env.helper.save(target, value)
    with pytest.raises(PermissionError):
        env.repair.retry_pending()
    assert len(env.transport.submitted) == 2


def test_original_timeout_recovery_uses_new_key_and_bound_existing_checkpoint(repair_env):
    env = repair_env
    first = env.api.controller.submit("train")
    directory = env.root / "orchestration/jobs" / first["key"]
    intent = env.helper.read(directory / "INTENT.json")
    row = env.raw.rows.pop(first["job_id"])
    env.raw.accounting[first["job_id"]] = dict(
        job_id=first["job_id"],
        state="TIMEOUT",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=row["QOS"],
        name=row["JobName"],
    )
    env.api.controller.status()
    model = orchestration.TASKS[intent["task_ids"][0]]["model_id"]
    checkpoint = env.root / "runs" / model / "checkpoint-proof.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("retained checkpoint identity proof; no model call")
    original_records = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    verified = []

    def verifier(root, task, previous):
        verified.append(previous)
        assert task == intent["task_ids"][0] and previous == first["key"]
        return dict(
            status="PASS",
            task_id=task,
            previous_job_key=previous,
            previous_job_id=first["job_id"],
            run_identity_sha256=env.helper.sha(root / "RUN_IDENTITY.json"),
            resume_kind="checkpoint",
            resume_state_sha256="c" * 64,
            execution_freeze_sha256=env.helper.sha(root / "EXECUTION_FREEZE.json"),
            artifacts=[orchestration.artifact_record(root, str(checkpoint.relative_to(root)))],
        )

    second = env.api.controller.resume(first["key"], verifier=verifier)
    successor = env.helper.read(env.root / "orchestration/jobs" / second["key"] / "INTENT.json")
    assert second["status"] == "SUBMITTED" and second["key"] != first["key"]
    assert (
        successor["resume_of"] == first["key"]
        and successor["worker_entry_commands"] == intent["worker_entry_commands"]
    )
    assert verified == [first["key"]]
    assert all((directory / name).read_bytes() == body for name, body in original_records.items())
    assert env.api.controller.resume(first["key"], verifier=verifier) == second
    assert len(env.transport.submitted) == 2
    with pytest.raises(PermissionError, match="unassigned"):
        env.repair.slurm.submit(intent["sbatch"])


@pytest.mark.parametrize(
    "blocked_status",
    ["SUBMISSION_UNKNOWN", "ACCOUNTING_UNKNOWN", "HELD", "RELEASE_UNKNOWN", "FAILED"],
)
def test_later_global_blocker_prevents_retry_of_earlier_known_rejection(
    repair_env, monkeypatch, blocked_status
):
    env = repair_env
    first = legacy_rejection(env)
    # Assemble two independently retained submissions, then restore the actual
    # global blocker before exercising repair. No production gate is replaced.
    with monkeypatch.context() as patch:
        patch.setattr(env.api.controller, "blocking_attempts", lambda attempts: [])
        second = legacy_rejection(env)
    known, blocked = sorted((first, second), key=lambda row: row[0]["key"])
    known_state, known_intent, known_directory = known
    blocked_state, _, blocked_directory = blocked
    assert known_state["key"] < blocked_state["key"]
    assert env.repair.rejection(known_intent, known_state) is not None
    blocked_state = dict(blocked_state, status=blocked_status)
    if blocked_status == "SUBMISSION_UNKNOWN":
        env.helper.save(
            blocked_directory / "SUBMISSION.json",
            dict(returncode=1, stdout="", stderr="Submission response lost; outcome unknown"),
        )
    else:
        blocked_state["job_id"] = "888888"
    env.helper.save(blocked_directory / "STATE.json", blocked_state)
    before = {
        record: record.read_bytes()
        for directory in (known_directory, blocked_directory)
        for record in directory.iterdir()
        if record.is_file()
    }
    submitted = len(env.transport.submitted)

    with pytest.raises(RuntimeError, match="unknown or blocked; not retried"):
        env.repair.retry_pending()

    assert len(env.transport.submitted) == submitted
    assert not journal(env, known_state["key"]).exists()
    assert not journal(env, blocked_state["key"]).exists()
    assert all(record.read_bytes() == body for record, body in before.items())
