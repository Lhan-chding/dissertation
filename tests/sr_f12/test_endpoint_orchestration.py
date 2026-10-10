"""Independent endpoint controller must never loosen frozen training gates."""

import json
import sys
from types import SimpleNamespace

import pytest

from sr_f12.endpoint_orchestration import MODEL_IDS, EndpointController
from sr_f12.orchestration import TEACHER_QOS, file_hash, save_json
from sr_f12.protocol import AMENDMENT_ID, object_hash, scientific_matrix


class FakeSlurm:
    def __init__(self):
        self.jobs, self.accounting, self.calls = {}, {}, []

    def queue(self, user):
        return dict(self.jobs)

    def terminal(self, job):
        return self.accounting.get(job)

    def show(self, job):
        return self.jobs[job]

    def run(self, args):
        self.calls.append(args)
        if args[0] == "sbatch":
            options = dict(
                arg[2:].split("=", 1) for arg in args if "=" in arg and arg.startswith("--")
            )
            job = str(100 + len(self.jobs))
            self.jobs[job] = dict(
                JobId=job,
                UserId="alice(1)",
                Account="rose",
                Partition="cluster02",
                QOS=TEACHER_QOS,
                JobName=options["job-name"],
                Comment=options["comment"],
                Command=args[-1],
                ReqTRES="cpu=4,mem=90G,gres/gpu=1,gres/gpu:pro6000=1",
                Features="highmem",
                JobState="PENDING",
                Reason="JobHeldUser",
            )
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if args[:2] == ["scontrol", "release"]:
            self.jobs[args[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    root, code, model = [tmp_path / name for name in ("run", "endpoint_code", "model")]
    for item in (root, code, model):
        item.mkdir()
    manifest = tmp_path / "SOURCE.json"
    manifest.write_text("{}")
    instance = EndpointController(
        root,
        code,
        model,
        manifest,
        "e" * 40,
        python=sys.executable,
        user="alice",
        slurm=FakeSlurm(),
    )
    monkeypatch.setattr(instance, "identity", lambda: "c" * 64)
    return instance


def prepare_main(instance):
    plan = {"science": "unchanged"}
    save_json(instance.root / "config/SR_F1_2.json", plan)
    save_json(
        instance.root / "AMENDMENT.json",
        dict(amendment_id=AMENDMENT_ID, source_commit="a" * 40, config_sha256=object_hash(plan)),
    )
    save_json(instance.root / "SCIENCE_FREEZE.json", dict(source_commit="a" * 40))
    states = {}
    for i, run in enumerate(scientific_matrix()):
        key, job = run["model_id"], str(1000 + i)
        directory = instance.root / "orchestration/tasks" / key
        directory.mkdir(parents=True)
        script = directory / "run.sbatch"
        script.write_text("#!/bin/bash\nexit 0\n")
        intent = dict(job_name="srf12-" + key, script=str(script), script_sha256=file_hash(script))
        save_json(directory / "INTENT.json", intent)
        state = dict(task_id=key, job_id=job, status="COMPLETE")
        save_json(directory / "STATE.json", state)
        terminal = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user="alice",
            account="rose",
            qos=TEACHER_QOS,
            name=intent["job_name"],
        )
        save_json(directory / "TERMINAL.json", terminal)
        save_json(directory / "VERIFIED_COMPLETE.json", dict(task_id=key, checked=True))
        instance.slurm.accounting[job] = terminal
        states[key] = state
    save_json(instance.root / "orchestration/STATE.json", dict(tasks=states))


def test_waiting_does_not_submit_read_test_or_touch_training(controller, monkeypatch):
    import sr_f12.endpoint as endpoint

    monkeypatch.setattr(
        endpoint, "verify_main_completion", lambda root: pytest.fail("premature gate")
    )
    result = controller.tick()
    assert result["stage"] == "WAITING_FOR_ALL_MAIN_TERMINAL"
    assert controller.slurm.calls == []
    assert not (controller.root / "orchestration").exists()
    assert not (controller.root / "sealed").exists()


def test_registry_files_do_not_replace_live_terminal(controller, monkeypatch):
    import sr_f12.endpoint as endpoint

    prepare_main(controller)
    monkeypatch.setattr(
        endpoint, "verify_main_completion", lambda root: pytest.fail("premature gate")
    )
    controller.slurm.accounting.pop("1000")
    assert controller.main_gate({}) is None
    controller.slurm.accounting["1000"] = dict(job_id="1000", state="FAILED", exit_code="1:0")
    with pytest.raises(PermissionError, match="terminal success"):
        controller.main_gate({})


def test_accounted_job_still_in_queue_is_not_terminal(controller, monkeypatch):
    import sr_f12.endpoint as endpoint

    prepare_main(controller)
    monkeypatch.setattr(
        endpoint, "verify_main_completion", lambda root: pytest.fail("premature gate")
    )
    assert controller.main_gate({"1000": {}}) is None


def test_terminal_plus_deep_state_gate_is_immutable(controller, monkeypatch):
    import sr_f12.endpoint as endpoint

    prepare_main(controller)
    checked = []
    monkeypatch.setattr(
        endpoint, "verify_main_completion", lambda root: checked.append(root) or dict(status="PASS")
    )
    result = controller.main_gate({})
    assert len(checked) == 1
    assert len(result["terminals"]) == 12
    assert (controller.directory / "MAIN_ACCEPTANCE.json").exists()
    assert controller.main_gate({}) == result
    assert controller.slurm.calls == []


def test_endpoint_submission_uses_four_remaining_cards(controller, monkeypatch):
    monkeypatch.setattr(controller, "main_gate", lambda queue: {"status": "PASS"})
    controller.slurm.jobs["999"] = dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=1")
    result = controller.tick()
    assert result["teacher_gpu_usage"] == 5
    assert len(result["tasks"]) == 4
    assert len([c for c in controller.slurm.calls if c[0] == "sbatch"]) == 4
    for command in controller.slurm.calls:
        if command[0] == "sbatch":
            assert "--qos=" + TEACHER_QOS in command
            assert "--gres=gpu:pro6000:1" in command
            assert "--hold" in command
    assert not result["experiment_delivered"]
    assert not (controller.root / "orchestration/CONFIG.json").exists()


def test_unknown_main_submission_reserves_capacity_and_blocks(controller, monkeypatch):
    monkeypatch.setattr(controller, "main_gate", lambda queue: {"status": "PASS"})
    path = controller.root / "orchestration/tasks/orphan/INTENT.json"
    save_json(path, {"task_id": "orphan"})
    result = controller.tick()
    assert result["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert result["teacher_gpu_usage"] == 1
    assert controller.slurm.calls == []


def test_release_requires_every_endpoint_terminal(controller):
    with pytest.raises(PermissionError, match="terminal and verified"):
        controller.release_all()


def test_complete_is_report_pending_not_delivery(controller, monkeypatch):
    import sr_f12.endpoint as endpoint

    monkeypatch.setattr(controller, "main_gate", lambda queue: {"status": "PASS"})
    monkeypatch.setattr(controller, "refresh", lambda queue: None)
    monkeypatch.setattr(controller, "verify_task", lambda key: {"status": "PASS"})
    for model in MODEL_IDS:
        save_json(
            controller.task_dir("EVAL_" + model) / "STATE.json",
            dict(task_id="EVAL_" + model, status="COMPLETE"),
        )
    calls = []

    def release(root, model):
        calls.append(model)
        return dict(
            status="RELEASED_AFTER_ALL_MAIN_COMPLETE",
            model_id=model,
            count=16452 if model == MODEL_IDS[0] else 17604,
            records=[],
            records_sha256=object_hash([]),
        )

    monkeypatch.setattr(endpoint, "release_scores", release)
    result = controller.tick()
    assert len(calls) == 13
    assert result["stage"] == "EVALUATIONS_COMPLETE_REPORT_PENDING"
    done = json.loads((controller.directory / "COMPLETE.json").read_text())
    assert done["endpoint_records"] == 227700
    assert done["report_pending"] is True
    assert done["experiment_delivered"] is False


def test_future_endpoint_source_appends_to_original_amendment(controller, monkeypatch):
    import sr_f12.evaluation as evaluation

    monkeypatch.undo()
    # Restore object-specific path values without invoking source constructor again.
    amendment = dict(amendment_id=AMENDMENT_ID, source_commit="a" * 40)
    save_json(controller.root / "AMENDMENT.json", amendment)
    save_json(
        controller.root / "MODEL_ENVIRONMENT_IDENTITY.json", dict(model_path=str(controller.model))
    )
    save_json(controller.source_manifest, dict(git_commit="e" * 40))
    monkeypatch.setattr(evaluation, "verify_source_manifest", lambda *a: "c" * 64)
    registration = dict(
        status="REGISTERED_ENDPOINT_IMPLEMENTATION",
        amendment_id=AMENDMENT_ID,
        amendment_sha256=object_hash(amendment),
        training_source_commit="a" * 40,
        source_commit="e" * 40,
        source_sha256="c" * 64,
        scientific_settings_unchanged=True,
        model_ids=list(MODEL_IDS),
    )
    save_json(controller.root / "ENDPOINT_IMPLEMENTATION.json", registration)
    assert controller.identity() == "c" * 64
    assert json.loads((controller.root / "AMENDMENT.json").read_text()) == amendment
    registration["training_source_commit"] = "b" * 40
    save_json(controller.root / "ENDPOINT_IMPLEMENTATION.json", registration)
    with pytest.raises(PermissionError):
        controller.identity()
