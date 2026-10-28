"""Submission evidence, stage permissions and resource gates without real jobs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import interface_v3_dispatch as dispatch
from sr_f6 import interface_v3_registration as registration
from sr_f6 import interface_v3_screen as screen


@pytest.fixture
def environment(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    root = project / "sr_f6_interface_v3_20261028"
    for stage in (1, 2):
        (root / f"STAGE{stage}").mkdir(parents=True)
    monkeypatch.setattr(dispatch, "PROJECT", project)
    monkeypatch.setattr(dispatch, "ROOT", root)
    monkeypatch.setattr(dispatch.getpass, "getuser", lambda: "varun024")
    calls = []
    checks = []

    def verify(base, *, stage, parent, require_preflight):
        checks.append((base, stage, parent, require_preflight))
        return dict(
            root=root / f"STAGE{stage}",
            registration_root=root,
            stage=stage,
            registration_sha256="f" * 64,
            deployment=Path(dispatch.__file__).resolve().parents[2],
        )

    def submit(argv, **kwargs):
        calls.append(argv)
        parent = next(item.rsplit("-", 1)[-1] for item in argv if item.startswith("--job-name="))
        stage = 2 if "-S2-" in " ".join(argv) else 1
        attempt = root / f"STAGE{stage}/SUBMISSIONS/{parent}.attempt.json"
        assert attempt.is_file(), "Intent must be durable before scheduler submission"
        return SimpleNamespace(returncode=0, stdout=str(1000 + len(calls)) + "\n", stderr="")

    monkeypatch.setattr(registration, "verify_registered", verify)
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": True})
    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    return SimpleNamespace(root=root, calls=calls, checks=checks)


def test_stage1_submits_three_fixed_parents_and_never_duplicates(environment):
    e = environment
    result = dispatch.submit(e.root, stage=1)
    assert [row["parent"] for row in result] == ["b0", "b1", "b2"]
    assert len(e.calls) == 3
    for argv in e.calls:
        assert "--no-requeue" in argv
        assert any(arg.startswith("--job-name=LH-SRF6-V3-S1-") for arg in argv)
        assert argv[-1].endswith("scripts/sr_f6/interface_v3.sbatch")
        assert any("SRF6_STAGE=1" in arg for arg in argv)
    again = dispatch.submit(e.root, stage=1)
    assert len(e.calls) == 3
    assert all(row["status"] == "ALREADY_SUBMITTED" for row in again)


def test_capacity_blocks_without_leaving_submission_intent(environment, monkeypatch):
    e = environment
    monkeypatch.setattr(dispatch, "capacity", lambda: {"can_submit_two_gpu_job": False})
    result = dispatch.submit(e.root, stage=1)
    assert result[0]["status"] == "NO_LEGAL_CAPACITY"
    assert not e.calls
    assert not list((e.root / "STAGE1/SUBMISSIONS").glob("*.attempt.json"))


def test_stage2_only_submits_authorized_gray_parent(environment, monkeypatch):
    e = environment
    calls = []
    monkeypatch.setattr(screen, "stage2_authorized", lambda root: calls.append(root) or ["b1"])
    result = dispatch.submit(e.root, stage=2)
    assert calls == [e.root]
    assert [row["parent"] for row in result] == ["b1"]
    assert e.checks == [(e.root, 2, "b1", True)]
    assert len(e.calls) == 1
    assert "--job-name=LH-SRF6-V3-S2-b1" in e.calls[0]


@pytest.mark.parametrize("parents", [[], ["b0", "b0"], ["b2", "b0"], ["b3"], ("b0",), None])
def test_missing_or_malformed_stage2_permission_cannot_submit(environment, monkeypatch, parents):
    monkeypatch.setattr(screen, "stage2_authorized", lambda root: parents)
    with pytest.raises(PermissionError, match="gray parents"):
        dispatch.submit(environment.root, stage=2)
    assert not environment.calls


@pytest.mark.parametrize("failed_stage", [1, 2])
def test_any_preserved_failure_blocks_both_stages(environment, monkeypatch, failed_stage):
    e = environment
    path = e.root / f"STAGE{failed_stage}/SCREEN/failures/failure.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"TECHNICAL_FAILURE_STOP"}')
    monkeypatch.setattr(screen, "stage2_authorized", lambda root: ["b0"])
    for stage in (1, 2):
        with pytest.raises(PermissionError, match="technical failure"):
            dispatch.submit(e.root, stage=stage)
    assert not e.calls


def test_uncertain_scheduler_outcome_is_durable_and_not_retried(environment, monkeypatch):
    e = environment
    monkeypatch.setattr(
        dispatch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="unknown", stderr=""),
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        dispatch.submit(e.root)
    receipt = json.loads((e.root / "STAGE1/SUBMISSIONS/b0.result.json").read_text())
    assert receipt["status"] == "FAILED_OR_UNKNOWN" and receipt["job_id"] is None
    with pytest.raises(PermissionError, match="no automatic retry"):
        dispatch.submit(e.root)


def test_orphan_receipt_cannot_be_treated_as_unsent(environment):
    path = environment.root / "STAGE1/SUBMISSIONS/b0.result.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(PermissionError, match="Orphan"):
        dispatch.submit(environment.root)
    assert not environment.calls


def test_cross_stage_or_changed_pin_submission_evidence_is_rejected(environment):
    e = environment
    dispatch.submit(e.root)
    path = e.root / "STAGE1/SUBMISSIONS/b0.attempt.json"
    old = json.loads(path.read_text())
    path.write_text(json.dumps(dict(old, stage=2)))
    with pytest.raises(PermissionError, match="changed"):
        dispatch.submit(e.root)
    assert len(e.calls) == 3


@pytest.mark.parametrize("stage", [0, 3, True, "1", None])
def test_unregistered_stages_fail_before_submission(environment, stage):
    with pytest.raises(ValueError, match="stage"):
        dispatch.submit(environment.root, stage=stage)
    assert not environment.calls


def test_stage_output_symlink_is_rejected(environment, tmp_path):
    root = environment.root
    (root / "STAGE2").rmdir()
    (root / "STAGE2").symlink_to(root / "STAGE1", target_is_directory=True)
    with pytest.raises(PermissionError, match=r"Canonical|symlink"):
        dispatch.submit(root)
    assert not environment.calls


def test_wrong_base_root_rejected_even_inside_project(environment):
    other = environment.root.parent / "another-project"
    other.mkdir()
    with pytest.raises(PermissionError, match="base root"):
        dispatch.submit(other)
    assert not environment.calls
