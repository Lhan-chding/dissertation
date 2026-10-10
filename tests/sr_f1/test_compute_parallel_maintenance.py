"""Maintenance captures only authenticated project evidence and never submits jobs."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compute_maintenance",
    Path(__file__).resolve().parents[2] / "scripts/sr_f1/compute_parallel_maintenance.py",
)
OPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPS)


@pytest.fixture
def case(tmp_path):
    root = tmp_path
    OPS.save(
        root / "orchestration/STATE.json",
        {
            "test_sealed": True,
            "tasks": {
                "COMMON_START": {"attempts": []},
                "ENGINE": {
                    "attempts": [
                        {
                            "job_id": "196508",
                            "attempt_id": "ENGINE_attempt0004",
                            "job_name": "engine-name",
                            "comment": "engine-comment",
                        }
                    ]
                },
                "BASELINE": {"attempts": []},
            },
        },
    )
    OPS.save(root / "manifests/ALLOCATION_PERMISSION.json", {"owner": "owner", "account": "rose"})
    OPS.save(
        root / "technical_incidents/baseline_parallel_20261010/HANDOFF_SUBMISSION.json",
        {"returncode": 0, "stdout": "196654\n"},
    )
    gpu = dict(
        JobId="196508",
        UserId="owner(1)",
        Account="rose",
        QOS=OPS.TEACHER_QOS,
        JobName="engine-name",
        Comment="engine-comment",
        Command=str(root / "orchestration/attempts/ENGINE_attempt0004.sh"),
        ReqTRES="gres/gpu=4",
        AllocTRES="gres/gpu=4",
    )
    cpu = dict(
        JobId="196654",
        UserId="owner(1)",
        Account="rose",
        QOS="cpu-qos",
        JobName="srf11-baseline-handoff",
        Comment="srf11-baseline-parallel-20261010",
        Command=str(root / "technical_incidents/baseline_parallel_20261010/handoff.sbatch"),
        ReqTRES="cpu=1",
        AllocTRES="cpu=1",
    )
    observed = []
    edits = {}

    def run(args):
        observed.append(args)
        assert args[0] in {"scontrol", "squeue", "sacct"}
        if args[0] == "scontrol":
            values = dict(cpu if args[3] == "196654" else gpu)
            values.update(edits)
            return {"returncode": 0, "stdout": " ".join(f"{k}={v}" for k, v in values.items())}
        if args[0] == "squeue":
            return {"returncode": 0, "stdout": "999999\n"}
        return {
            "returncode": 0,
            "stdout": "196654|CANCELLED by 1|0:0|owner|rose|srf11-baseline-handoff|cpu-qos|\n"
            "196508|CANCELLED|0:0|owner|rose|engine-name|" + OPS.TEACHER_QOS + "|\n",
        }

    return root, run, edits, observed


def test_identity_and_terminal_only_read_targeted_jobs(case):
    root, run, _, observed = case
    identity = OPS.identities(root, run=run)
    terminal = OPS.terminal_jobs(root, {"identities": identity}, run=run)
    assert terminal["gpu_job_id"] == "196508" and terminal["controller_job_id"] == "196654"
    assert terminal["unrelated_jobs_modified"] == 0
    assert all(args[0] not in {"scancel", "sbatch", "srun"} for args in observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("UserId", "someone(2)"),
        ("Account", "other"),
        ("JobName", "unrelated"),
        ("Command", "/unrelated.sh"),
        ("Comment", "other"),
        ("ReqTRES", "gres/gpu=1"),
    ],
)
def test_identity_rejects_unrelated_or_gpu_controller(case, field, value):
    root, run, edits, _ = case
    edits[field] = value
    with pytest.raises(PermissionError):
        OPS.identities(root, run=run)


@pytest.mark.parametrize("queued", ["196508\n", "196654\n"])
def test_preserve_refuses_completing_or_live_jobs(case, queued):
    root, run, _, _ = case
    identity = OPS.identities(root, run=run)

    def active(args):
        return {"returncode": 0, "stdout": queued} if args[0] == "squeue" else run(args)

    with pytest.raises(PermissionError, match="STILL_QUEUED"):
        OPS.terminal_jobs(root, {"identities": identity}, run=active)


def test_unknown_accounting_is_not_terminal(case):
    root, run, _, _ = case
    identity = OPS.identities(root, run=run)

    def unknown(args):
        return {"returncode": 0, "stdout": ""} if args[0] == "sacct" else run(args)

    with pytest.raises(PermissionError, match="MISSING"):
        OPS.terminal_jobs(root, {"identities": identity}, run=unknown)


def test_receipts_idempotent_but_never_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    OPS.save(path, {"x": 1})
    before = path.read_bytes()
    OPS.save(path, {"x": 1})
    assert path.read_bytes() == before
    with pytest.raises(PermissionError, match="IMMUTABLE"):
        OPS.save(path, {"x": 2})


def test_stable_copy_resumes_but_rejects_source_changes(tmp_path):
    root, target = tmp_path / "source", tmp_path / "saved"
    OPS.save(root / "raw/data.json", {"raw": 1})
    first = OPS.copy_files(root, target, ["raw"], stable=True)
    assert OPS.copy_files(root, target, ["raw"], stable=True) == first
    (root / "raw/data.json").write_text("changed")
    with pytest.raises(PermissionError, match="PRESERVED_BYTES_CHANGED"):
        OPS.copy_files(root, target, ["raw"], stable=True)


def test_dynamic_partial_snapshot_keeps_prior_bytes(tmp_path):
    root, target = tmp_path / "source", tmp_path / "saved"
    OPS.save(root / "raw/data.json", {"raw": 1})
    first = OPS.copy_files(root, target, ["raw"], stable=False)
    (root / "raw/data.json").write_text("changed")
    assert OPS.copy_files(root, target, ["raw"], stable=False) == first


def test_copy_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "unrelated")
    with pytest.raises(PermissionError, match="SYMLINK"):
        OPS.copy_files(source, tmp_path / "saved", ["link"], stable=True)


def test_rejects_accepted_engine_before_snapshot(case):
    root, run, _, _ = case
    OPS.save(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json", {"status": "PASS"})
    with pytest.raises(PermissionError, match="ALREADY_ACCEPTED"):
        OPS.identities(root, run=run)


def test_rejects_downstream_work(case):
    root, run, _, _ = case
    path = root / "orchestration/STATE.json"
    state = OPS.read(path)
    state["tasks"]["BASELINE"]["attempts"] = [{"job_id": "5"}]
    path.write_text(__import__("json").dumps(state))
    with pytest.raises(PermissionError, match="DOWNSTREAM"):
        OPS.identities(root, run=run)


def test_preserve_only_reuses_receipts_without_second_reconciliation(case, monkeypatch):
    import json

    import sr_f1.orchestration

    root, run, _, observed = case
    for relative in OPS.COPY_PATHS:
        path = root / relative
        if not path.exists():
            OPS.save(path, {})
    code = root / "code"
    OPS.save(code / "source.json", {"code": True})
    deployment = {
        "source_commit": "a" * 40,
        "source_file_hashes": {"source.json": OPS.sha(code / "source.json")},
    }
    OPS.save(code / "SOURCE_DEPLOYMENT.json", deployment)
    (root / "ENGINE_MULTIGPU_REPAIR.json").write_text(json.dumps({"source_commit": "a" * 40}))
    candidate = root / "code_baseline_parallel_candidate_20261010"
    OPS.save(candidate / "source.json", {"candidate": True})
    OPS.save(
        candidate / "SOURCE_DEPLOYMENT.json",
        {
            "source_commit": "b" * 40,
            "source_file_hashes": {"source.json": OPS.sha(candidate / "source.json")},
        },
    )
    auth = root / "technical_incidents/baseline_parallel_20261010/HANDOFF_AUTHORIZATION.json"
    auth.write_text(
        json.dumps(
            {"candidate_source_deployment_sha256": OPS.sha(candidate / "SOURCE_DEPLOYMENT.json")}
        )
    )
    calls = []

    class Scheduler:
        def __init__(self, *args, **kwargs):
            self.state = OPS.read(root / "orchestration/STATE.json")

        def _load(self):
            pass

        def _observe(self, task):
            calls.append(task)
            self.state["tasks"][task]["attempts"][-1]["accounting"] = {
                "terminal_state": "CANCELLED"
            }

        def _phase(self):
            pass

        def _save(self, event):
            (root / "orchestration/STATE.json").write_text(json.dumps(self.state))
            (root / "orchestration/journal.jsonl").write_text(
                json.dumps({"event": event, "state": self.state}) + "\n"
            )

    monkeypatch.setattr(sr_f1.orchestration, "Scheduler", Scheduler)
    OPS.snapshot(root, run=run)

    def review(_root):
        return {"status": "PASS_PRESERVED_FULL_STATE", "committed_logical_step": 2}

    first = OPS.preserve(root, run=run, review=review)
    second = OPS.preserve(root, run=run, review=review)
    assert first == second
    assert calls == ["ENGINE"]
    assert not (root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists()
    assert not (root / "BASELINE_PARALLEL_REPAIR.json").exists()
    assert OPS.inventory(root / "code_before_engine_compute_parallel_20261010") == deployment
    assert all(args[0] not in {"scancel", "sbatch"} for args in observed)
