import copy
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts/sr_f5"
sys.path.insert(0, str(SCRIPTS))
MODULE = runpy.run_path(str(SCRIPTS / "advance_development.py"))
sys.path.remove(str(SCRIPTS))


class Scheduler:
    def __init__(self):
        self.active = set()
        self.state = "COMPLETED"

    def user_job_ids(self, user):
        return self.active

    def terminal(self, job_id):
        return dict(
            job_id=job_id,
            state=self.state,
            exit_code="0:0",
            user="me",
            account=MODULE["ACCOUNT"],
            qos=MODULE["QOS"],
        )


def receipt(root):
    return dict(
        status="TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED",
        jobs=[
            dict(
                pair=i,
                job_id=str(i + 1),
                verified=dict(
                    JobId=str(i + 1),
                    JobName=f"LH-SRF5-G2-P{i}",
                    StdOut=str(root / "slurm" / f"G2-P{i}-{i + 1}.log"),
                ),
            )
            for i in range(2)
        ],
    )


def test_running_or_failed_census_cannot_advance(tmp_path):
    scheduler = Scheduler()
    scheduler.active = {"1"}
    assert (
        MODULE["census_terminal"](scheduler, receipt(tmp_path), tmp_path, "me")[0] == "WAIT_FOR_G2"
    )
    scheduler.active = set()
    scheduler.state = "FAILED"
    assert MODULE["census_terminal"](scheduler, receipt(tmp_path), tmp_path, "me")[0] == (
        "G2_TECHNICAL_FAILURE_REQUIRES_DIAGNOSIS"
    )
    scheduler.state = "COMPLETED"
    assert (
        MODULE["census_terminal"](scheduler, receipt(tmp_path), tmp_path, "me")[0]
        == "G2_JOBS_COMPLETE"
    )


def test_cannot_bind_other_project_job_or_unknown_receipt(tmp_path):
    original = receipt(tmp_path)
    for field, value in (("StdOut", "/other/project/log"), ("JobName", "someone-elses-job")):
        altered = copy.deepcopy(original)
        altered["jobs"][0]["verified"][field] = value
        with pytest.raises(PermissionError):
            MODULE["census_terminal"](Scheduler(), altered, tmp_path, "me")
    original["status"] = "SUBMITTING"
    with pytest.raises(PermissionError):
        MODULE["census_terminal"](Scheduler(), original, tmp_path, "me")


@pytest.mark.parametrize("gate", ["FAIL", "PASS"])
def test_next_stage_requires_actual_gate_pass(tmp_path, monkeypatch, gate):
    advance = MODULE["advance"]
    namespace = advance.__globals__
    root, code = tmp_path / "run", tmp_path / "code"
    root.mkdir()
    code.mkdir()
    revision = "a" * 40
    (root / "prior.json").write_text(json.dumps(receipt(root)))
    checks = root / "CONTINUATION" / revision
    checks.mkdir(parents=True)
    (checks / "CPU_TESTS.json").write_text(json.dumps(dict(status="PASS", code_revision=revision)))
    monkeypatch.setitem(namespace, "PROJECT", tmp_path)
    monkeypatch.setitem(namespace, "__file__", str(code / "scripts/sr_f5/advance_development.py"))
    monkeypatch.setitem(namespace, "Slurm", Scheduler)
    monkeypatch.setattr(namespace["getpass"], "getuser", lambda: "me")
    monkeypatch.setitem(namespace, "aggregate", lambda _: dict(gate=dict(status=gate)))
    calls = []
    monkeypatch.setitem(namespace, "freeze", lambda *_: calls.append("freeze"))

    def submit(_):
        calls.append("submit")
        return dict(status="TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED")

    monkeypatch.setitem(namespace, "submit", submit)
    result = advance(SimpleNamespace(root=root, code=code, revision=revision, receipt="prior.json"))
    if gate == "FAIL":
        assert not calls
        assert result["status"] == "G2_FEASIBILITY_FAILURE_REQUIRES_DOCUMENTED_DECISION"
    else:
        assert calls == ["freeze", "submit"]
        assert result["status"] == "G3_SUBMITTED_AFTER_REAL_G2_PASS"
