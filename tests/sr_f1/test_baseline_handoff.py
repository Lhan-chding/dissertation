"""CPU-only takeover, ENGINE gate, and crash-safe source promotion."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/sr_f1/baseline_handoff.py"
SPEC = importlib.util.spec_from_file_location("baseline_handoff", MODULE_PATH)
HANDOFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANDOFF)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def deployment(folder, label):
    path = folder / "src/sr_f1/example.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(label)
    files = {"src/sr_f1/example.py": HANDOFF.sha(path)}
    receipt = dict(
        source_commit=label * 40,
        source_dirty_files=[],
        source_file_hashes=files,
        source_tree_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    write(folder / "SOURCE_DEPLOYMENT.json", receipt)
    return receipt


@pytest.fixture
def authorized(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    fields = dict(
        JobId="196501",
        JobName="srf11-controller-multigpu",
        UserId="varun024(123)",
        Account="rose",
        QOS="override-limits-but-killable",
        JobState="RUNNING",
        Command=str(root / "code/scripts/sr_f1/controller.sbatch"),
        StdOut=str(root / "technical_incidents/multigpu_20261010/controller_196501.log"),
        ReqTRES="cpu=1,mem=3G,node=1",
        AllocTRES="cpu=1,mem=3G,node=1",
    )
    identity = {key: fields[key] for key in ("JobName", "Account", "QOS", "Command", "StdOut")}
    identity.update(UserId="varun024", Comment="")
    auth = dict(old_controller_job_id="196501", old_controller_identity=identity)
    write(root / HANDOFF.INCIDENT / "HANDOFF_AUTHORIZATION.json", auth)
    return root, auth, fields


def cpu_snapshot(terminal=False):
    return dict(
        job_id="196501",
        queue=[] if terminal else [["196501", "RUNNING"]],
        accounting=[
            dict(
                JobIDRaw="196501",
                JobName="srf11-controller-multigpu",
                User="varun024",
                Account="rose",
                QOS="override-limits-but-killable",
                State="CANCELLED" if terminal else "RUNNING",
                AllocTRES="cpu=1,mem=3G,node=1",
                ExitCode="0:15",
                Comment="",
            )
        ],
    )


def test_absent_comment_is_bound_as_empty(authorized):
    _, auth, fields = authorized
    HANDOFF.verify_cpu(fields, auth, "196508")
    HANDOFF.verify_cpu({**fields, "Comment": "(null)"}, auth, "196508")


@pytest.mark.parametrize(
    "change",
    [
        {"JobId": "196508"},
        {"JobName": "unrelated"},
        {"Account": "another"},
        {"UserId": "other(123)"},
        {"Command": "/other/controller.sbatch"},
        {"StdOut": "/other.log"},
        {"Comment": "other"},
        {"QOS": "other"},
        {"ReqTRES": "cpu=1,gres/gpu=1"},
        {"AllocTRES": "cpu=1,gres/gpu:pro6000=1"},
        {"AllocTRES": ""},
    ],
)
def test_identity_or_gpu_change_forbids_cancel(authorized, change):
    _, auth, fields = authorized
    with pytest.raises(RuntimeError):
        HANDOFF.verify_cpu({**fields, **change}, auth, "196508")


def test_engine_id_can_never_be_cancel_target(authorized):
    _, auth, fields = authorized
    with pytest.raises(RuntimeError, match="REFUSE_ENGINE"):
        HANDOFF.verify_cpu(fields, auth, "196501")


def test_cancel_only_cpu_once_even_after_ambiguous_response(authorized):
    root, auth, fields = authorized
    slurm = Mock()
    slurm.snapshot.side_effect = [cpu_snapshot(), cpu_snapshot(True)]
    slurm.detail.return_value = (fields, {"returncode": 0})
    slurm.run.side_effect = OSError("response lost")
    HANDOFF.stop_cpu(root, auth, "196508", slurm=slurm, sleep=Mock())
    slurm.run.assert_called_once_with(["scancel", "196501"])
    assert HANDOFF.read(root / HANDOFF.INCIDENT / "CPU_CANCEL_RESULT.json")["returncode"] is None
    resumed = Mock()
    resumed.snapshot.side_effect = [cpu_snapshot(), cpu_snapshot(True)]
    HANDOFF.stop_cpu(root, auth, "196508", slurm=resumed, sleep=Mock())
    resumed.run.assert_not_called()
    resumed.detail.assert_not_called()


def test_cancel_not_sent_when_target_has_gpu(authorized):
    root, auth, fields = authorized
    slurm = Mock()
    slurm.snapshot.return_value = cpu_snapshot()
    slurm.detail.return_value = ({**fields, "ReqTRES": "cpu=4,gres/gpu=1"}, {})
    with pytest.raises(RuntimeError, match="REFUSE_GPU"):
        HANDOFF.stop_cpu(root, auth, "196508", slurm=slurm, sleep=Mock())
    slurm.run.assert_not_called()
    assert not (root / HANDOFF.INCIDENT / "CPU_CANCEL_INTENT.json").exists()


def test_unknown_cpu_visibility_does_not_cancel_or_infer_terminal(authorized):
    root, auth, _ = authorized
    slurm = Mock()
    invisible = dict(job_id="196501", queue=[], accounting=[])
    slurm.snapshot.side_effect = [
        HANDOFF.ObservationUnknown("network"),
        invisible,
        cpu_snapshot(True),
    ]
    sleeper = Mock()
    HANDOFF.stop_cpu(root, auth, "196508", slurm=slurm, sleep=sleeper)
    slurm.run.assert_not_called()
    assert sleeper.call_count == 2


def state(complete=True):
    return dict(
        test_sealed=True,
        tasks={
            "COMMON_START": {"status": "COMPLETE", "attempts": [{"job_id": "1"}]},
            "ENGINE": {
                "status": "COMPLETE" if complete else "ACTIVE",
                "attempts": [
                    {
                        "job_id": "196508",
                        "accounting": {"terminal_state": "COMPLETED", "exit_code": "0:0"},
                    }
                ],
            },
            "BASELINE": {"status": "WAITING", "attempts": []},
            "SRF1_A_s71001": {"status": "WAITING", "attempts": []},
        },
    )


def engine_snapshot():
    return dict(queue=[], accounting=[dict(JobIDRaw="196508", State="COMPLETED", ExitCode="0:0")])


def scheduler_for(value):
    scheduler = SimpleNamespace(
        state=value, _load=Mock(), _observe=Mock(), _marker_valid=Mock(return_value=True)
    )
    scheduler.tick = Mock(side_effect=AssertionError("watcher must not tick"))
    scheduler._submit = Mock(side_effect=AssertionError("watcher must not submit"))
    return scheduler


def test_complete_engine_needs_slurm_marker_and_real_receipt(tmp_path):
    scheduler, verify, slurm = scheduler_for(state()), Mock(), Mock()
    slurm.snapshot.return_value = engine_snapshot()
    assert HANDOFF.observe_engine(tmp_path, scheduler, verify, slurm) == engine_snapshot()
    verify.assert_called_once_with(tmp_path)
    scheduler._marker_valid.assert_called_once_with(
        "ENGINE", scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
    )
    scheduler.tick.assert_not_called()
    scheduler._submit.assert_not_called()


@pytest.mark.parametrize("status", ["ACTIVE", "UNKNOWN", "REGISTERED", "SUBMITTING"])
def test_unfinished_engine_is_observed_only(tmp_path, status):
    value = state(False)
    value["tasks"]["ENGINE"]["status"] = status
    scheduler, verify, slurm = scheduler_for(value), Mock(), Mock()
    assert HANDOFF.observe_engine(tmp_path, scheduler, verify, slurm) is None
    scheduler._observe.assert_called_once_with("ENGINE")
    verify.assert_not_called()
    slurm.snapshot.assert_not_called()
    scheduler.tick.assert_not_called()
    scheduler._submit.assert_not_called()


@pytest.mark.parametrize(
    "snapshot",
    [
        {"queue": [["196508", "COMPLETING"]], "accounting": []},
        {"queue": [], "accounting": []},
    ],
)
def test_completion_marker_does_not_override_missing_terminal(tmp_path, snapshot):
    scheduler, verify, slurm = scheduler_for(state()), Mock(), Mock()
    slurm.snapshot.return_value = snapshot
    assert HANDOFF.observe_engine(tmp_path, scheduler, verify, slurm) is None
    verify.assert_not_called()


@pytest.mark.parametrize("field,value", [("State", "FAILED"), ("ExitCode", "1:0")])
def test_terminal_failure_never_activates(tmp_path, field, value):
    scheduler, verify, slurm = scheduler_for(state()), Mock(), Mock()
    snapshot = engine_snapshot()
    snapshot["accounting"][0][field] = value
    slurm.snapshot.return_value = snapshot
    with pytest.raises(RuntimeError, match="TERMINAL_NOT_SUCCESSFUL"):
        HANDOFF.observe_engine(tmp_path, scheduler, verify, slurm)
    verify.assert_not_called()


def test_engine_receipt_failure_propagates_before_handoff(tmp_path):
    scheduler, slurm = scheduler_for(state()), Mock()
    slurm.snapshot.return_value = engine_snapshot()
    with pytest.raises(ValueError, match="not equal"):
        HANDOFF.observe_engine(
            tmp_path, scheduler, Mock(side_effect=ValueError("not equal")), slurm
        )
    assert not (tmp_path / HANDOFF.INCIDENT).exists()


@pytest.mark.parametrize("change", ["started", "raw", "unsealed", "stop"])
def test_downstream_activity_forbids_takeover(tmp_path, change):
    value = state()
    if change == "started":
        value["tasks"]["BASELINE"]["attempts"] = [{"job_id": "2"}]
    elif change == "raw":
        write(tmp_path / "raw/evaluation/a.jsonl", {})
    elif change == "unsealed":
        value["test_sealed"] = False
    else:
        (tmp_path / "STOP").touch()
    with pytest.raises(RuntimeError):
        HANDOFF.prebaseline(tmp_path, value)


@pytest.fixture
def activation(tmp_path):
    root = tmp_path / "run"
    before = deployment(root / "code", "a")
    after = deployment(root / HANDOFF.CANDIDATE, "b")
    write(root / "EXECUTION_FREEZE.json", {"frozen": True})
    write(root / HANDOFF.INCIDENT / "HANDOFF_AUTHORIZATION.json", {"authorized": True})
    evidence = root / HANDOFF.INCIDENT / "activation/evidence/ENGINE.json"
    write(evidence, {"status": "PASS"})
    intent = dict(
        run_root=str(root),
        before=before,
        after=after,
        original_execution_freeze_sha256=HANDOFF.sha(root / "EXECUTION_FREEZE.json"),
        authorization_sha256=HANDOFF.sha(root / HANDOFF.INCIDENT / "HANDOFF_AUTHORIZATION.json"),
        historical_artifact_hashes={str(evidence.relative_to(root)): HANDOFF.sha(evidence)},
    )
    write(root / HANDOFF.INCIDENT / "activation/ACTIVATION_INTENT.json", intent)
    return root, intent


@pytest.mark.parametrize("interrupt_at", [1, 2])
def test_restart_after_each_atomic_rename_recovers_without_overwrite(activation, interrupt_at):
    root, intent = activation
    calls = []

    def interrupt(source, target):
        calls.append((source, target))
        source.rename(target)
        if len(calls) == interrupt_at:
            raise OSError("interrupted after durable rename")

    with pytest.raises(OSError):
        HANDOFF.recover_moves(root, intent, rename=interrupt)
    HANDOFF.recover_moves(root, intent)
    HANDOFF.recover_moves(root, intent)
    assert HANDOFF.source_inventory(root / HANDOFF.PRESERVED) == intent["before"]
    assert HANDOFF.source_inventory(root / "code") == intent["after"]
    assert not (root / HANDOFF.CANDIDATE).exists()


@pytest.mark.parametrize("changed", ["code", "candidate", "evidence", "freeze"])
def test_changed_identity_blocks_directory_moves(activation, changed):
    root, intent = activation
    path = {
        "code": root / "code/src/sr_f1/example.py",
        "candidate": root / HANDOFF.CANDIDATE / "src/sr_f1/example.py",
        "evidence": root / next(iter(intent["historical_artifact_hashes"])),
        "freeze": root / "EXECUTION_FREEZE.json",
    }[changed]
    path.write_text("changed")
    with pytest.raises(RuntimeError):
        HANDOFF.recover_moves(root, intent)
    assert (root / "code").exists()
    assert (root / HANDOFF.CANDIDATE).exists()
    assert not (root / HANDOFF.PRESERVED).exists()


def test_undeclared_source_file_rejected_before_takeover(activation):
    root, _ = activation
    (root / HANDOFF.CANDIDATE / "src/sr_f1/extra.py").write_text("extra")
    with pytest.raises(RuntimeError, match="SOURCE_INVENTORY_CHANGED"):
        HANDOFF.source_inventory(root / HANDOFF.CANDIDATE)


def test_activation_recovers_between_preflight_and_activation_receipt(activation):
    root, intent = activation
    checked = {"status": "PASS", "repair": {"source_commit": "b" * 40}, "test_sealed": True}
    check_receipt = dict(returncode=0, stdout=json.dumps(checked), stderr="", time="first")
    builder_receipt = dict(returncode=0, stdout=json.dumps({"repair": "built"}), stderr="")
    runner = Mock(side_effect=[builder_receipt, check_receipt])
    original_save = HANDOFF.save

    def interrupted(path, value):
        if path.name == "SOURCE_ACTIVATED.json":
            raise OSError("interrupted before receipt")
        original_save(path, value)

    with patch.object(HANDOFF, "save", side_effect=interrupted), pytest.raises(OSError):
        HANDOFF.activate(root, root / "plan", Path("/python"), {}, runner=runner)
    preflight = root / HANDOFF.INCIDENT / "activation/PREFLIGHT.json"
    original_hash = HANDOFF.sha(preflight)
    resumed_runner = Mock(return_value={**check_receipt, "time": "second"})
    HANDOFF.activate(root, root / "plan", Path("/python"), {}, runner=resumed_runner)
    assert HANDOFF.sha(preflight) == original_hash
    resumed_runner.assert_called_once()
    assert (root / HANDOFF.INCIDENT / "activation/SOURCE_ACTIVATED.json").exists()
    assert HANDOFF.source_inventory(root / HANDOFF.PRESERVED) == intent["before"]


def test_handoff_lock_is_compatible_with_scheduler_and_retains_inode(tmp_path):
    from mm_dev.orchestration import process_lease

    path = tmp_path / "lock"
    with HANDOFF.lease(path):
        inode = path.stat().st_ino
        with pytest.raises(RuntimeError, match="LEASE_BUSY"), process_lease(path):
            pytest.fail("two controllers acquired the lock")
    with process_lease(path):
        assert path.stat().st_ino == inode


def test_slurm_full_owner_queue_ignores_unrelated_jobs_without_mutation():
    results = iter(
        [
            dict(returncode=0, stdout="999|RUNNING\n", stderr=""),
            dict(
                returncode=0,
                stdout="196508|engine|COMPLETED|varun024|rose|teacher|gres/gpu=4|0:0|comment\n",
                stderr="",
            ),
        ]
    )
    runner = Mock(side_effect=lambda args: next(results))
    snapshot = HANDOFF.Slurm(runner, owner="varun024").snapshot("196508")
    assert snapshot["queue"] == []
    assert snapshot["accounting"][0]["State"] == "COMPLETED"
    assert {call.args[0][0] for call in runner.call_args_list} == {"squeue", "sacct"}
    assert "--user=varun024" in runner.call_args_list[0].args[0]


def test_slurm_unknown_is_distinct_from_terminal():
    runner = Mock(return_value=dict(returncode=1, stdout="", stderr="unavailable"))
    with pytest.raises(HANDOFF.ObservationUnknown):
        HANDOFF.Slurm(runner, owner="varun024").snapshot("196508")


def test_controller_finishing_during_takeover_is_not_cancelled_twice(authorized):
    root, auth, fields = authorized
    slurm = Mock()
    slurm.snapshot.side_effect = [cpu_snapshot(), cpu_snapshot(True)]
    slurm.detail.return_value = ({**fields, "JobState": "FAILED"}, {})
    HANDOFF.stop_cpu(root, auth, "196508", slurm=slurm, sleep=Mock())
    slurm.run.assert_not_called()
    assert not (root / HANDOFF.INCIDENT / "CPU_CANCEL_INTENT.json").exists()


def test_false_completion_marker_prevents_engine_acceptance(tmp_path):
    scheduler, slurm, verify = scheduler_for(state()), Mock(), Mock()
    scheduler._marker_valid.return_value = False
    slurm.snapshot.return_value = engine_snapshot()
    with pytest.raises(RuntimeError, match="COMPLETION_MARKER_INVALID"):
        HANDOFF.observe_engine(tmp_path, scheduler, verify, slurm)
    verify.assert_not_called()
