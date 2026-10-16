"""Local fake-Slurm validation; never invokes a real model or scheduler."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import orchestration

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def fields(intent, job, *, held=False):
    tres = f"cpu=4,mem={intent['memory_gib']}G,node=1"
    if intent["gpu_count"]:
        tres += ",gres/gpu=1,gres/gpu:pro6000=1"
    return dict(
        JobId=job,
        JobName=intent["job_name"],
        Account="rose",
        QOS=intent["qos"],
        Partition="cluster02",
        Command=intent["script"],
        Comment=intent["comment"],
        UserId=intent["user"] + "(123)",
        NumCPUs="4",
        NumNodes="1",
        ReqTRES=tres,
        AllocTRES="(null)" if held else tres,
        Features="highmem" if intent["gpu_count"] else "(null)",
        ReqNodeList="cpu-2",
        NodeList="cpu-2",
        Requeue="0",
        TimeLimit="3-00:00:00",
        JobState="PENDING" if held else "RUNNING",
        Reason="JobHeldUser" if held else "None",
    )


class Slurm:
    def __init__(self, h):
        self.h, self.jobs, self.scripts, self.terminals = h, {}, {}, {}
        self.calls = []
        self.submit_unknown = False
        self.release_unknown = False
        self.mismatch = None
        self.post_hold_extra = False

    def queue(self, user):
        return self.jobs.copy()

    def show(self, job):
        result = self.jobs[job].copy()
        if self.mismatch:
            result[self.mismatch[0]] = self.mismatch[1]
        return result

    def batch_script(self, job):
        return self.scripts[job]

    def terminal(self, job):
        return self.terminals.get(job)

    def run(self, command):
        self.calls.append(command)
        if command[0] == "sbatch":
            if self.submit_unknown:
                raise TimeoutError("submission unknown")
            path = Path(command[-1])
            intent = self.h.read(path.parent / "INTENT.json")
            self.jobs["100"] = fields(intent, "100", held=True)
            self.scripts["100"] = path.read_text()
            if self.post_hold_extra:
                for number in range(200, 205):
                    self.jobs[str(number)] = fields(intent, str(number))
            return subprocess.CompletedProcess(command, 0, "100\n", "")
        if command[:2] == ["scontrol", "release"]:
            if self.release_unknown:
                raise TimeoutError("release unknown")
            self.jobs["100"].update(
                JobState="RUNNING", Reason="None", AllocTRES=self.jobs["100"]["ReqTRES"]
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    module = load(HERE / "submit_selected_init.py", "selected_init_fixture")
    h = load(REPO / "scripts/sr_f3_ce/dispatch_protocol_ablation.py", "audited_dispatch_fixture")
    root = tmp_path / "experiment/ablation"
    root.mkdir(parents=True)
    h.save(root / "RUN_IDENTITY.json", {"identity": "frozen"})
    h.save(root / "deployment/SOURCE_MANIFEST.json", {"source": "652"})
    h.save(h.operation_directory(root) / "OPERATION_IDENTITY.json", {"operation": "755"})
    h.save(
        root / "PROTOCOL_SELECTION.json",
        dict(status="PROTOCOL_SELECTED", selected_revision="v2", ablation_root=str(root)),
    )
    old_registration = dict(root=str(root), python=sys.executable, user="fixture-user")
    payload = root.parent / "deployment/initialize_selected.sh"
    payload.parent.mkdir(parents=True)
    payload.write_text("#!/bin/bash\nset -euo pipefail\ntrue\n")
    payload.chmod(0o444)
    rows = []
    monkeypatch.setattr(h, "_verified_rows", lambda *args: rows)
    api = SimpleNamespace(
        ablation=SimpleNamespace(
            verify_selection=lambda path: h.read(path / "PROTOCOL_SELECTION.json")
        ),
        orchestration=orchestration,
        slurm=Slurm(h),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(module.socket, "gethostname", lambda: "cpu-2.cluster")
    return SimpleNamespace(
        module=module,
        h=h,
        root=root,
        api=api,
        old=old_registration,
        payload=payload,
        rows=rows,
        monkeypatch=monkeypatch,
    )


def register(s):
    return s.module.register(
        s.root, s.h, s.api, s.old, s.payload, s.h.sha(s.payload), s.old["python"]
    )


def submit(s):
    return s.module.submit(s.root, s.h, s.api, s.old)


@pytest.mark.parametrize(
    "status", ["STACK_REPRODUCTION_FAILURE", "STOP_PROTOCOL_FAILURE", "PENDING"]
)
def test_only_successful_complete_selection_can_register(fixture, status):
    s = fixture
    s.h.save(
        s.root / "PROTOCOL_SELECTION.json",
        dict(status=status, selected_revision="v2", ablation_root=str(s.root)),
    )
    with pytest.raises(PermissionError, match="successful frozen"):
        register(s)
    assert not s.module.directory(s.root).exists()
    assert not (s.root.parent / "selected_v2").exists()


def test_register_is_immutable_and_never_creates_selected_root(fixture):
    s = fixture
    first, second = register(s), register(s)
    assert first == second
    assert not (s.root.parent / "selected_v2").exists()
    assert not s.api.slurm.calls


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_existing_selected_root_blocks_fresh_registration(fixture, revision):
    s = fixture
    s.h.save(
        s.root / "PROTOCOL_SELECTION.json",
        dict(status="PROTOCOL_SELECTED", selected_revision=revision, ablation_root=str(s.root)),
    )
    (s.root.parent / ("selected_" + revision)).mkdir()
    with pytest.raises(PermissionError, match="already exists"):
        register(s)


def test_payload_must_have_explicit_matching_hash_and_read_only_mode(fixture):
    s = fixture
    with pytest.raises(PermissionError, match="read-only"):
        s.module.register(s.root, s.h, s.api, s.old, s.payload, "0" * 64, s.old["python"])
    s.payload.chmod(0o644)
    with pytest.raises(PermissionError, match="read-only"):
        register(s)
    assert not s.api.slurm.calls


def test_real_preparation_payload_env_bash_header_is_accepted(fixture):
    s = fixture
    s.payload.chmod(0o644)
    s.payload.write_text("#!/usr/bin/env bash\nset -euo pipefail\ntrue\n")
    s.payload.chmod(0o444)
    registration = register(s)
    intent = s.module.intent_for(s.root, s.h, registration)
    assert intent["payload_command"] == ["/bin/bash", str(s.payload)]


def test_normal_submission_has_exact_cpu_policy_and_one_sequence(fixture):
    s = fixture
    register(s)
    state = submit(s)
    assert state["status"] == "SUBMITTED"
    intent = s.h.read(s.module.directory(s.root) / "INTENT.json")
    assert intent["job_name"] == "SRF3CE-SELECTED-INIT-LH"
    assert intent["gpu_count"] == 0
    assert intent["memory_gib"] == 12 and intent["cpus"] == 4
    assert "--nodelist=cpu-2" in intent["sbatch"]
    assert "--no-requeue" in intent["sbatch"]
    assert "--time=4320" in intent["sbatch"]
    assert not any("--gres" in arg for arg in intent["sbatch"])
    with pytest.raises(PermissionError, match="never resubmit"):
        submit(s)
    assert len(s.api.slurm.calls) == 2
    assert not (s.root.parent / "selected_v2").exists()


def test_five_submitted_jobs_prevent_cpu_submission(fixture):
    s = fixture
    registration = register(s)
    intent = s.module.intent_for(s.root, s.h, registration)
    for number in range(1, 6):
        s.api.slurm.jobs[str(number)] = fields(intent, str(number))
    result = submit(s)
    assert result == dict(status="WAITING_FOR_CAPACITY", teacher_usage=dict(gpus=0, submitted=5))
    assert not s.api.slurm.calls
    assert not (s.module.directory(s.root) / "INTENT.json").exists()


def test_cpu_is_counted_once_in_post_hold_quota(fixture):
    s = fixture
    registration = register(s)
    intent = s.module.intent_for(s.root, s.h, registration)
    for number in range(1, 5):
        s.api.slurm.jobs[str(number)] = fields(intent, str(number))
    submit(s)
    receipt = s.h.read(s.module.directory(s.root) / "QUEUE_BEFORE_RELEASE.json")
    assert receipt["teacher_usage"] == dict(gpus=0, submitted=5)


def test_other_users_teacher_gpu_usage_is_not_ignored(fixture):
    s = fixture
    registration = register(s)
    intent = dict(s.module.intent_for(s.root, s.h, registration), user="other-user", gpu_count=1)
    for number in range(1, 7):
        s.api.slurm.jobs[str(number)] = fields(intent, str(number))
    result = submit(s)
    assert result["teacher_usage"] == dict(gpus=6, submitted=0)
    assert not s.api.slurm.calls


def test_racing_cpu_capacity_leaves_new_job_held(fixture):
    s = fixture
    register(s)
    s.api.slurm.post_hold_extra = True
    with pytest.raises(PermissionError, match="remains held"):
        submit(s)
    assert len(s.api.slurm.calls) == 1
    assert s.h.read(s.module.directory(s.root) / "STATE.json")["status"] == "HELD"


@pytest.mark.parametrize("stage", ["submit", "release"])
def test_unknown_outcomes_are_durable_and_never_retried(fixture, stage):
    s = fixture
    register(s)
    setattr(s.api.slurm, stage + "_unknown", True)
    with pytest.raises(TimeoutError):
        submit(s)
    state = s.h.read(s.module.directory(s.root) / "STATE.json")
    assert state["status"] == ("SUBMISSION_UNKNOWN" if stage == "submit" else "RELEASE_UNKNOWN")
    count = len(s.api.slurm.calls)
    with pytest.raises(PermissionError, match="never resubmit"):
        submit(s)
    assert len(s.api.slurm.calls) == count


@pytest.mark.parametrize(
    "field,value",
    [
        ("QOS", "rose"),
        ("Account", "other"),
        ("JobName", "missing-LH-suffix"),
        ("Requeue", "1"),
        ("ReqNodeList", "cpu-1"),
    ],
)
def test_actual_held_fields_must_match(fixture, field, value):
    s = fixture
    register(s)
    s.api.slurm.mismatch = (field, value)
    with pytest.raises(PermissionError):
        submit(s)
    assert len(s.api.slurm.calls) == 1


def test_shared_ablation_lock_blocks_any_submission(fixture):
    s = fixture
    register(s)
    with orchestration.lease(s.h.submissions(s.root)), pytest.raises(RuntimeError):
        submit(s)
    assert not s.api.slurm.calls


def test_prior_unknown_accounting_does_not_disappear_from_capacity(fixture):
    s = fixture
    registration = register(s)
    intent = s.module.intent_for(s.root, s.h, registration)
    state = dict(key="old-driver", job_id="20", gpu_count=0, qos=s.module.QOS, status="SUBMITTED")
    s.rows.append((Path("/fixture-only"), intent, state))
    with pytest.raises(PermissionError, match="accounting is unknown"):
        submit(s)
    assert state["status"] == "SUBMITTED"
    assert not s.api.slurm.calls


def test_execute_checks_actual_cpu_and_runs_payload_once(fixture):
    s = fixture
    register(s)
    submit(s)
    s.monkeypatch.setenv("SLURM_JOB_ID", "100")
    calls = []

    def run_payload(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return subprocess.CompletedProcess(command, 0)

    result = s.module.execute(s.root, s.h, s.api, s.old, run_payload=run_payload)
    assert result["status"] == "PAYLOAD_PROCESS_COMPLETED"
    assert calls == [["/bin/bash", str(s.payload)]]
    with pytest.raises(PermissionError, match="already started"):
        s.module.execute(s.root, s.h, s.api, s.old, run_payload=run_payload)
    assert len(calls) == 1


def test_failed_payload_is_preserved_and_never_rerun(fixture):
    s = fixture
    register(s)
    submit(s)
    s.monkeypatch.setenv("SLURM_JOB_ID", "100")
    calls = []

    def run_payload(command, **kwargs):
        calls.append(command)
        raise KeyboardInterrupt("interrupted initialization")

    with pytest.raises(KeyboardInterrupt):
        s.module.execute(s.root, s.h, s.api, s.old, run_payload=run_payload)
    failure = s.h.read(s.module.directory(s.root) / "EXECUTION_FAILURE.json")
    assert "KeyboardInterrupt" in failure["error"]
    with pytest.raises(PermissionError):
        s.module.execute(s.root, s.h, s.api, s.old, run_payload=run_payload)
    assert len(calls) == 1


def test_payload_changed_after_submission_is_rejected_before_execution(fixture):
    s = fixture
    register(s)
    submit(s)
    s.monkeypatch.setenv("SLURM_JOB_ID", "100")
    s.payload.chmod(0o644)
    s.payload.write_text("#!/bin/bash\nfalse\n")
    s.payload.chmod(0o444)
    with pytest.raises(PermissionError, match="read-only"):
        s.module.execute(
            s.root,
            s.h,
            s.api,
            s.old,
            run_payload=lambda *args, **kwargs: pytest.fail("must not execute"),
        )


def test_terminal_mode_only_records_real_accounting(fixture):
    s = fixture
    register(s)
    submit(s)
    count = len(s.api.slurm.calls)
    assert s.module.terminal(s.root, s.h, s.api, s.old)["status"] == "ACCOUNTING_UNKNOWN_OR_ACTIVE"
    s.api.slurm.terminals["100"] = dict(
        job_id="100",
        state="COMPLETED",
        exit_code="0:0",
        user=s.old["user"],
        account="rose",
        qos=s.module.QOS,
        name=s.module.JOB_NAME,
    )
    assert s.module.terminal(s.root, s.h, s.api, s.old)["status"] == "ACCOUNTING_UNKNOWN_OR_ACTIVE"
    assert not (s.module.directory(s.root) / "TERMINAL.json").exists()
    s.api.slurm.jobs.pop("100")
    result = s.module.terminal(s.root, s.h, s.api, s.old)
    assert result["scientific_acceptance_claimed"] is False
    assert s.h.read(s.module.directory(s.root) / "TERMINAL.json") == result["terminal"]
    assert len(s.api.slurm.calls) == count


def test_login_host_cannot_execute_even_with_a_copied_job_environment(fixture):
    s = fixture
    register(s)
    submit(s)
    s.monkeypatch.setenv("SLURM_JOB_ID", "100")
    s.monkeypatch.setattr(s.module.socket, "gethostname", lambda: "login-node")
    with pytest.raises(PermissionError, match="Actual CPU"):
        s.module.execute(
            s.root,
            s.h,
            s.api,
            s.old,
            run_payload=lambda *args, **kwargs: pytest.fail("must not execute on login"),
        )
    assert not (s.module.directory(s.root) / "STARTED.json").exists()


def test_audited_helper_pin_precedes_any_scientific_import(tmp_path):
    module = load(HERE / "submit_selected_init.py", "selected_init_support_fixture")
    path = (
        tmp_path
        / "operations/ablation_driver"
        / module.DISPATCH_COMMIT
        / "dispatch_protocol_ablation.py"
    )
    path.parent.mkdir(parents=True)
    path.write_text("raise AssertionError('unverified module executed')\n")
    with pytest.raises(PermissionError, match="dispatcher bytes differ"):
        module.support(tmp_path)
    assert (
        hashlib.sha256(
            (REPO / "scripts/sr_f3_ce/dispatch_protocol_ablation.py").read_bytes()
        ).hexdigest()
        == module.DISPATCH_SHA256
    )
