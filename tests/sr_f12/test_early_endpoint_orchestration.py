import sys
from types import SimpleNamespace

import pytest
from test_endpoint_orchestration import FakeSlurm, prepare_main

from sr_f12.early_endpoint_orchestration import EarlyEndpointController, submission_usage
from sr_f12.orchestration import controller_lease, file_hash, read_json, save_json
from sr_f12.protocol import TEACHER_QOS, scientific_matrix


class PlacementSlurm(FakeSlurm):
    def run(self, args):
        if args[:3] == ["scontrol", "show", "hostnames"]:
            return SimpleNamespace(returncode=0, stdout=args[3].replace(",", "\n"), stderr="")
        result = super().run(args)
        if args[0] == "sbatch":
            job = result.stdout.strip()
            self.jobs[job]["ExcNodeList"] = next(
                (a.split("=", 1)[1] for a in args if a.startswith("--exclude=")), "(null)"
            )
        return result


@pytest.fixture
def early(tmp_path, monkeypatch):
    root, code, model = [tmp_path / n for n in ("run", "code", "model")]
    for p in (root, code, model):
        p.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    c = EarlyEndpointController(
        root,
        code,
        model,
        manifest,
        "e" * 40,
        python=sys.executable,
        user="alice",
        slurm=PlacementSlurm(),
    )
    monkeypatch.setattr(c, "identity", lambda: "c" * 64)
    prepare_main(c)
    for run in scientific_matrix():
        key = run["model_id"]
        p = root / "orchestration/tasks" / key / "INTENT.json"
        v = read_json(p)
        v.update(task_id=key, gpu_count=1, comment="training-" + key)
        save_json(p, v)
    # Every training path has an immutable submission; two are still active.
    snap = read_json(root / "orchestration/STATE.json")
    for run in scientific_matrix()[-2:]:
        key = run["model_id"]
        d = root / "orchestration/tasks" / key
        state = read_json(d / "STATE.json")
        state["status"] = "RUNNING"
        save_json(d / "STATE.json", state)
        snap["tasks"][key] = state
        intent = read_json(d / "INTENT.json")
        c.slurm.jobs[state["job_id"]] = dict(
            JobId=state["job_id"],
            UserId="alice(1)",
            Account="rose",
            Partition="cluster02",
            QOS=TEACHER_QOS,
            JobName=intent["job_name"],
            Comment=intent["comment"],
            Command=intent["script"],
            ReqTRES="cpu=4,mem=90G,gres/gpu=1,gres/gpu:pro6000=1",
            Features="highmem",
            JobState="RUNNING",
            Reason="None",
            NodeList="gpu-train-" + state["job_id"],
        )
    save_json(root / "orchestration/STATE.json", snap)
    monkeypatch.setattr(c, "main_gate", lambda q: None)
    monkeypatch.setattr(
        c,
        "model_gate",
        lambda model: (
            {"sealed": True}
            if c.task_main_state(model) and c.task_main_state(model)["status"] == "COMPLETE"
            else None
        ),
    )
    return c


def test_early_fill_spare_capacity_excludes_training_nodes_and_never_scores(early, monkeypatch):
    monkeypatch.setattr(early, "release_all", lambda: pytest.fail("premature score release"))
    result = early.tick()
    assert len(result["tasks"]) == 3
    assert result["teacher_gpu_usage"] == 5
    assert result["scores_sealed"] is True
    assert not (early.directory / "MAIN_ACCEPTANCE.json").exists()
    for args in early.slurm.calls:
        if args[0] == "sbatch":
            assert "--exclude=gpu-train-1010,gpu-train-1011" in args
            task = next(a.split("srf12-", 1)[1] for a in args if a.startswith("--job-name="))
            assert (
                "run_early_evaluation.py"
                in read_json(early.task_dir(task) / "INTENT.json")["command"][1]
            )
    count = len(early.slurm.calls)
    early.tick()
    assert len(early.slurm.calls) == count


@pytest.mark.parametrize(
    "status", ["PENDING", "HELD", "SUBMITTED", "SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN", "FAILED"]
)
def test_training_needs_precede_evaluation(early, status):
    key = scientific_matrix()[-1]["model_id"]
    p = early.root / "orchestration/tasks" / key / "STATE.json"
    v = read_json(p)
    v["status"] = status
    save_json(p, v)
    early.tick()
    assert early.slurm.calls == []


def test_missing_last_training_prevents_any_evaluation(early):
    key = scientific_matrix()[-1]["model_id"]
    d = early.root / "orchestration/tasks" / key
    (d / "STATE.json").unlink()
    (d / "INTENT.json").unlink()
    early.tick()
    assert early.slurm.calls == []


def test_complete_still_in_queue_not_accepted(early):
    early.slurm.jobs["1000"] = {
        "JobState": "COMPLETING",
        "QOS": TEACHER_QOS,
        "ReqTRES": "gres/gpu=1",
    }
    early.tick()
    assert early.slurm.calls == []


def test_teacher_cpu_uses_submission_slot(early):
    early.slurm.jobs["9000"] = {"QOS": TEACHER_QOS, "ReqTRES": "cpu=2"}
    result = early.tick()
    assert len(result["tasks"]) == 2 and result["teacher_gpu_usage"] == 4
    assert submission_usage(early.slurm.jobs, [*early.attempts(), *early.main_attempts()]) == 5


def test_queue_unknown_fails_closed(early, monkeypatch):
    monkeypatch.setattr(
        early.slurm, "queue", lambda user: (_ for _ in ()).throw(RuntimeError("SSH unknown"))
    )
    with pytest.raises(RuntimeError, match="SSH unknown"):
        early.tick()
    assert early.slurm.calls == []


def test_priority_change_before_release_keeps_endpoint_held(early, monkeypatch):
    original = early.slurm.run

    def run(args):
        result = original(args)
        if args[0] == "sbatch":
            early.slurm.jobs["1011"]["JobState"] = "PENDING"
        return result

    monkeypatch.setattr(early.slurm, "run", run)
    with pytest.raises(PermissionError, match="stays held"):
        early.tick()
    assert not any(args[:2] == ["scontrol", "release"] for args in early.slurm.calls)
    assert len(early.attempts()) == 1


def test_node_exclusion_change_before_release_keeps_held(early, monkeypatch):
    original = early.slurm.run

    def run(args):
        result = original(args)
        if args[0] == "sbatch":
            early.slurm.jobs["1011"]["NodeList"] = "gpu-other"
        return result

    monkeypatch.setattr(early.slurm, "run", run)
    with pytest.raises(PermissionError, match="exclude every"):
        early.tick()
    assert not any(args[:2] == ["scontrol", "release"] for args in early.slurm.calls)


def test_old_endpoint_lease_blocks_new_controller(early):
    from sr_f12.endpoint_orchestration import run_endpoint_controller

    with controller_lease(early.directory), pytest.raises(RuntimeError, match="owns the lease"):
        run_endpoint_controller(early, once=True)
    assert early.slurm.calls == []


def test_identity_uses_parsed_manifest_hash_not_formatted_bytes(early, monkeypatch):
    import sr_f12.early_authorization as authorization
    from sr_f12.protocol import object_hash

    data = {"git_commit": "e" * 40, "files": {"x": "abc"}}
    save_json(early.source_manifest, data)
    expected = object_hash(data)
    assert file_hash(early.source_manifest) != expected
    save_json(early.root / "MODEL_ENVIRONMENT_IDENTITY.json", {"model_path": str(early.model)})
    monkeypatch.setattr(
        authorization,
        "verify_early_implementation",
        lambda *a: {"source_manifest_sha256": expected},
    )
    assert EarlyEndpointController.identity(early) == expected


def test_state_changes_during_deep_release_gate_are_rechecked(early, monkeypatch):
    original = early.model_gate
    calls = []

    def gate(model):
        calls.append(model)
        if len(calls) == 2:
            early.slurm.jobs["1011"]["JobState"] = "PENDING"
        return original(model)

    monkeypatch.setattr(early, "model_gate", gate)
    with pytest.raises(PermissionError, match="stays held"):
        early.tick()
    assert not any(args[:2] == ["scontrol", "release"] for args in early.slurm.calls)
