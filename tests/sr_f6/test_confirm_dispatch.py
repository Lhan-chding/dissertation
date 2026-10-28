"""Confirmation submission identity, durable intent, live capacity and failure closure."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import confirm_dispatch as dispatch


@pytest.fixture
def environment(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    root = project / "sr_f6_confirm_v3_20261028"
    root.mkdir()
    monkeypatch.setattr(dispatch, "PROJECT", project)
    monkeypatch.setattr(dispatch, "ROOT", root)
    monkeypatch.setattr(dispatch.getpass, "getuser", lambda: "varun024")
    calls, checks = [], []

    def verify(base, *, parent, require_preflight):
        checks.append((base, parent, require_preflight))
        return dict(
            root=root,
            registration_root=root,
            registration_sha256="f" * 64,
            deployment=Path(dispatch.__file__).resolve().parents[2],
        )

    def submit(argv, **kwargs):
        calls.append(argv)
        parent = next(item.rsplit("-", 1)[-1] for item in argv if item.startswith("--job-name="))
        attempt = root / f"SUBMISSIONS/{parent}.attempt.json"
        assert attempt.is_file(), "Intent must exist before scheduler invocation"
        return SimpleNamespace(returncode=0, stdout=str(1000 + len(calls)) + "\n", stderr="")

    monkeypatch.setitem(
        sys.modules, "sr_f6.confirm_registration", SimpleNamespace(verify_registered=verify)
    )
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": True})
    monkeypatch.setattr(
        dispatch, "_job_states", lambda jobs: {job: ("RUNNING", None) for job in jobs}
    )
    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    return SimpleNamespace(root=root, calls=calls, checks=checks)


def test_exact_parents_require_sealed_preflight_and_never_resubmit(environment):
    e = environment
    result = dispatch.submit(e.root)
    assert [row["parent"] for row in result] == ["b0", "b1", "b2"]
    assert e.checks == [(e.root, parent, True) for parent in ("b0", "b1", "b2")]
    for argv in e.calls:
        assert "--no-requeue" in argv
        assert any(arg.startswith("--job-name=LH-SRF6-Confirm-") for arg in argv)
        assert argv[-1].endswith("scripts/sr_f6/confirm.sbatch")
    again = dispatch.submit(e.root)
    assert len(e.calls) == 3
    assert all(row["status"] == "ALREADY_SUBMITTED" for row in again)


def test_live_capacity_rechecked_for_every_parent(environment, monkeypatch):
    e = environment
    snapshots = iter([True, False])
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": next(snapshots)})
    result = dispatch.submit(e.root)
    assert len(e.calls) == 1 and result[-1]["status"] == "NO_LEGAL_CAPACITY"
    assert not (e.root / "SUBMISSIONS/b1.attempt.json").exists()


def test_capacity_absence_never_defaults_unknown_to_zero(environment, monkeypatch):
    monkeypatch.setattr(dispatch, "capacity", lambda: {})
    with pytest.raises(KeyError):
        dispatch.submit(environment.root)
    assert not environment.calls


def test_preserved_failure_stops_before_any_submit(environment):
    path = environment.root / "CONFIRM/failures/original.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"TECHNICAL_FAILURE_STOP"}')
    with pytest.raises(PermissionError, match="technical failure"):
        dispatch.submit(environment.root)
    assert not environment.calls


def test_failure_appearing_during_dispatch_blocks_next_parent(environment, monkeypatch):
    e = environment
    old = dispatch.subprocess.run

    def submit(argv, **kwargs):
        result = old(argv, **kwargs)
        path = e.root / "CONFIRM/failures/new.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        return result

    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    with pytest.raises(PermissionError, match="technical failure"):
        dispatch.submit(e.root)
    assert len(e.calls) == 1


def test_ambiguous_scheduler_output_is_retained_without_retry(environment, monkeypatch):
    e = environment
    monkeypatch.setattr(
        dispatch.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="unknown", stderr=""),
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        dispatch.submit(e.root)
    receipt = json.loads((e.root / "SUBMISSIONS/b0.result.json").read_text())
    assert receipt["status"] == "FAILED_OR_UNKNOWN" and receipt["job_id"] is None
    with pytest.raises(PermissionError, match="no automatic retry"):
        dispatch.submit(e.root)


def test_submission_exception_leaves_unknown_intent_not_retried(environment, monkeypatch):
    e = environment

    def failed(*args, **kwargs):
        raise OSError("scheduler connection lost")

    monkeypatch.setattr(dispatch.subprocess, "run", failed)
    with pytest.raises(OSError):
        dispatch.submit(e.root)
    assert (e.root / "SUBMISSIONS/b0.attempt.json").exists()
    with pytest.raises(PermissionError, match="unknown"):
        dispatch.submit(e.root)


def test_orphan_receipt_blocks(environment):
    path = environment.root / "SUBMISSIONS/b0.result.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(PermissionError, match="Orphan"):
        dispatch.submit(environment.root)
    assert not environment.calls


def test_changed_registration_pin_cannot_skip_a_previous_job(environment):
    e = environment
    dispatch.submit(e.root)
    path = e.root / "SUBMISSIONS/b0.attempt.json"
    old = json.loads(path.read_text())
    path.write_text(json.dumps(dict(old, registration_sha256="0" * 64)))
    with pytest.raises(PermissionError, match="changed"):
        dispatch.submit(e.root)
    assert len(e.calls) == 3


def test_symlinked_submission_directory_rejected(environment, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (environment.root / "SUBMISSIONS").symlink_to(other)
    with pytest.raises(PermissionError, match=r"Canonical|symlink"):
        dispatch.submit(environment.root)
    assert not environment.calls


def test_wrong_project_root_and_account_rejected(environment, monkeypatch):
    other = environment.root.parent / "another-project"
    other.mkdir()
    with pytest.raises(PermissionError, match="base root"):
        dispatch.submit(other)
    monkeypatch.setattr(dispatch.getpass, "getuser", lambda: "another-account")
    with pytest.raises(PermissionError, match="shared account"):
        dispatch.submit(environment.root)
    assert not environment.calls


def test_consumed_slot_without_matching_execution_blocks_submission(environment):
    e = environment
    path = e.root / "CONFIRM/execution_slots/b0-0.json"
    path.parent.mkdir(parents=True)
    slot = dict(parent="b0", shard=0, task_id="LH-SRF6-Confirm-1000-b0-0")
    path.write_text(json.dumps(slot))
    with pytest.raises(PermissionError, match="no matching execution"):
        dispatch.submit(e.root)
    execution = e.root / "CONFIRM/executions/LH-SRF6-Confirm-1000-b0-0.json"
    execution.parent.mkdir(parents=True)
    execution.write_text(json.dumps(dict(slot, status="changed")))
    with pytest.raises(PermissionError, match="no matching execution"):
        dispatch.submit(e.root)
    assert not e.calls


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "UNKNOWN"])
def test_terminal_or_unknown_job_without_failure_marker_blocks_new_parent(
    environment, monkeypatch, state
):
    e = environment
    snapshots = iter([True, False])
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": next(snapshots)})
    dispatch.submit(e.root)
    monkeypatch.setattr(dispatch, "_job_states", lambda jobs: {job: (state, "1:0") for job in jobs})
    with pytest.raises(PermissionError, match="failed or is unknown"):
        dispatch.submit(e.root)
    assert len(e.calls) == 1 and not (e.root / "CONFIRM/failures").exists()


def test_completed_slurm_job_requires_both_complete_workers(environment, monkeypatch):
    e = environment
    snapshots = iter([True, False])
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": next(snapshots)})
    dispatch.submit(e.root)
    monkeypatch.setattr(
        dispatch, "_job_states", lambda jobs: {job: ("COMPLETED", "0:0") for job in jobs}
    )
    with pytest.raises(PermissionError, match="complete original worker"):
        dispatch.submit(e.root)
    assert len(e.calls) == 1
    for shard in (0, 1):
        task = f"LH-SRF6-Confirm-1001-b0-{shard}"
        slot = dict(parent="b0", shard=shard, task_id=task)
        for name, value in (
            (f"execution_slots/b0-{shard}.json", slot),
            (f"executions/{task}.json", slot),
            (f"workers/{task}.json", dict(slot, status="COMPLETE")),
        ):
            path = e.root / "CONFIRM" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
    # b0 terminal is complete; b1 and b2 are new active jobs.
    monkeypatch.setattr(
        dispatch,
        "_job_states",
        lambda jobs: {
            job: ("COMPLETED", "0:0") if job == "1001" else ("RUNNING", None) for job in jobs
        },
    )
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": True})
    result = dispatch.submit(e.root)
    assert result[0]["status"] == "ALREADY_SUBMITTED" and len(e.calls) == 3


def test_exact_recorded_job_state_queries_and_terminal_failure(monkeypatch):
    calls = []

    def query(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "squeue":
            assert argv[argv.index("--jobs") + 1] == "100,101"
            return SimpleNamespace(returncode=0, stdout="100|RUNNING\n", stderr="")
        assert argv[0] == "sacct" and argv[argv.index("--jobs") + 1] == "101"
        return SimpleNamespace(returncode=0, stdout="101|CANCELLED by 123|0:15\n", stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", query)
    assert dispatch._job_states({"100", "101"}) == {
        "100": ("RUNNING", None),
        "101": ("CANCELLED", "0:15"),
    }
    assert len(calls) == 2


@pytest.mark.parametrize("response", ["", "999|COMPLETED|0:0\n", "101||\n"])
def test_missing_unrelated_or_unknown_job_response_fails_closed(monkeypatch, response):
    def query(argv, **kwargs):
        return SimpleNamespace(
            returncode=0, stdout="" if argv[0] == "squeue" else response, stderr=""
        )

    monkeypatch.setattr(dispatch.subprocess, "run", query)
    with pytest.raises(PermissionError):
        dispatch._job_states({"101"})


def test_scheduler_query_failure_never_allows_submission(monkeypatch):
    monkeypatch.setattr(
        dispatch.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="unavailable"),
    )
    with pytest.raises(PermissionError, match="Cannot determine"):
        dispatch._job_states({"101"})
