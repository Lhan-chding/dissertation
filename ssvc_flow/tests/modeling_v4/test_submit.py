"""Tiny scheduler substitutes: no Slurm, SSH, torch, or experiment execution."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def submitter():
    path = Path(__file__).parents[2] / "scripts/submit_modeling_v4.py"
    spec = importlib.util.spec_from_file_location("v4_submit_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def options(tmp_path, submitter):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"operations": {"max_concurrent_gpus": 3, "default_workers": 2}}))
    tasks = tmp_path / "tasks.json"
    table = {
        "schema": "ssvc-v4-task-list-1",
        "config": json.loads(config.read_text()),
        "root": str(tmp_path / "campaign"),
        "workers": 2,
        "campaign_id": "fixture",
        "tasks": [{"id": "fixture"}],
    }
    table["task_list_hash"] = submitter.digest(table)
    tasks.write_text(json.dumps(table))
    reuse = tmp_path / "reuse.json"
    reuse.write_text("{}")
    return submitter.parser().parse_args(
        [
            "--config",
            str(config),
            "--tasks",
            str(tasks),
            "--reuse",
            str(reuse),
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--python",
            "/server/python",
        ]
    )


class Scheduler:
    def __init__(self):
        self.calls = []
        self.states = {}
        self.next_id = 100
        self.bad_reply = False

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[0] == "sbatch":
            self.next_id += 1
            job_id = str(self.next_id)
            self.states[job_id] = "PENDING"
            stdout = "unrecognized response" if self.bad_reply else job_id + "\n"
        elif command[0] == "squeue":
            ids = command[command.index("--jobs") + 1].split(",")
            stdout = "".join(
                f"{i}|{self.states[i]}\n"
                for i in ids
                if self.states.get(i) in {"PENDING", "RUNNING", "COMPLETING"}
            )
        elif command[0] == "sacct":
            job_id = command[command.index("--jobs") + 1]
            stdout = f"{job_id}|{self.states[job_id]}|\n" if job_id in self.states else ""
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    @property
    def submissions(self):
        return [c for c, _ in self.calls if c[0] == "sbatch"]


def test_default_dry_run_is_pure_and_explicit(submitter, options):
    scheduler = Scheduler()
    result = submitter.execute(options, run=scheduler)
    assert result["status"] == "DRY_RUN"
    assert not scheduler.calls
    assert not Path(options.campaign_root).exists()
    assert len(result["commands"]) == 3
    for command in result["commands"]:
        assert "--no-requeue" in command["argv"]
        assert "--time=3-00:00:00" in command["argv"]
        assert "--dependency" not in command["shell"]
    assert all("--gres=gpu:pro6000:1" in x["argv"] for x in result["commands"][1:])


def test_three_workers_allowed_four_rejected(submitter, options):
    options.workers = 3
    table = json.loads(Path(options.tasks).read_text())
    table["workers"] = 3
    table["task_list_hash"] = submitter.digest(
        {k: v for k, v in table.items() if k != "task_list_hash"}
    )
    Path(options.tasks).write_text(json.dumps(table))
    assert len(submitter.build_plan(options)["commands"]) == 4
    options.workers = 4
    with pytest.raises(ValueError, match=r"three|limit|workers"):
        submitter.build_plan(options)


def test_real_gpu_submit_needs_campaign_switch(submitter, options):
    options.submit = True
    scheduler = Scheduler()
    with pytest.raises(ValueError, match="execute-gpu"):
        submitter.execute(options, run=scheduler)
    assert not scheduler.submissions


def test_cpu_and_gpu_submit_parallel_then_resume_reuses_active(submitter, options, monkeypatch):
    options.submit = options.execute_gpu = True
    monkeypatch.setenv("SBATCH_GRES", "gpu:pro6000:8")
    monkeypatch.setenv("PYTHONPATH", "/unexpected")
    scheduler = Scheduler()
    first = submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 3
    assert first["status"] == "SUBMITTED"
    for _, kwargs in scheduler.calls:
        assert "SBATCH_GRES" not in kwargs["env"]
        assert "PYTHONPATH" not in kwargs["env"]
    options.execute_gpu = False
    with pytest.raises(ValueError, match="resume"):
        submitter.execute(options, run=scheduler)
    options.resume = True
    second = submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 3
    assert {x["action"] for x in second["jobs"]} == {"REUSED_ACTIVE"}


def test_resume_retries_only_failed_jobs_and_preserves_attempts(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    scheduler.states = {"101": "COMPLETED", "102": "FAILED", "103": "RUNNING"}
    options.resume = True
    result = submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 4
    assert "--resume" in scheduler.submissions[-1]
    assert [x["action"] for x in result["jobs"]] == [
        "REUSED_COMPLETED",
        "SUBMITTED",
        "REUSED_ACTIVE",
    ]
    record = json.loads((Path(options.campaign_root) / "SLURM_SUBMISSIONS.json").read_text())
    assert len(record["jobs"]["gpu_worker_0"]["attempts"]) == 2


def test_phase_label_cannot_duplicate_active_workers(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    options.resume, options.stage, options.phase = True, "gpu", "firstmap"
    result = submitter.execute(options, run=scheduler)
    assert all(x["action"] == "REUSED_ACTIVE" for x in result["jobs"])
    assert len(scheduler.submissions) == 3


def test_uncertain_submission_never_blindly_retried(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    scheduler.bad_reply = True
    with pytest.raises(RuntimeError, match=r"uncertain|Uncertain"):
        submitter.execute(options, run=scheduler)
    options.resume = True
    with pytest.raises(RuntimeError, match=r"uncertain|Uncertain"):
        submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 1


def test_changed_identity_or_tasks_cannot_resume(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    options.resume = True
    table = json.loads(Path(options.tasks).read_text())
    table["tasks"] = [{"id": "changed"}]
    table["task_list_hash"] = submitter.digest(
        {k: v for k, v in table.items() if k != "task_list_hash"}
    )
    Path(options.tasks).write_text(json.dumps(table))
    with pytest.raises(ValueError, match=r"changed|identity"):
        submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 3


def test_unknown_scheduler_state_blocks_duplicate(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    scheduler.states = {}
    options.resume = True
    with pytest.raises(RuntimeError, match=r"unknown|Unknown"):
        submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 3


def test_cpu_only_and_print_only_take_precedence(submitter, options):
    options.stage, options.submit, options.print_only = "cpu", True, True
    options.origin_subset = "seed101_X_BASE_init7001_a8"
    scheduler = Scheduler()
    plan = submitter.execute(options, run=scheduler)
    assert len(plan["commands"]) == 1
    assert "--origin-subset" in plan["commands"][0]["argv"]
    assert not scheduler.calls
    options.print_only = False
    submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 1


def test_sbatch_scripts_parse_without_executing():
    scripts = Path(__file__).parents[2] / "scripts"
    for name in ("modeling_v4_cpu.sbatch", "modeling_v4_gpu.sbatch"):
        subprocess.run(["bash", "-n", str(scripts / name)], check=True)


def test_completed_worker_waiting_for_bridge_can_continue(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    scheduler.states = {"101": "COMPLETED", "102": "COMPLETED", "103": "COMPLETED"}
    table = json.loads(Path(options.tasks).read_text())
    for worker, status in enumerate(("WAITING_FOR_BRIDGE", "COMPLETED_REQUESTED_STAGES")):
        path = Path(options.campaign_root) / "workers" / f"worker_{worker}" / "LATEST.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "worker_id": worker,
                    "workers": 2,
                    "status": status,
                    "campaign_id": table["campaign_id"],
                    "task_list_hash": table["task_list_hash"],
                }
            )
        )
    options.resume = True
    result = submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 4
    assert [x["action"] for x in result["jobs"]] == [
        "REUSED_COMPLETED",
        "SUBMITTED",
        "REUSED_COMPLETED",
    ]


def test_task_root_cannot_split_campaign_budget(submitter, options):
    options.campaign_root += "_other"
    with pytest.raises(ValueError, match="root"):
        submitter.build_plan(options)


def test_unicode_task_hash_matches_actual_campaign_canonical_hash(submitter, options):
    from src.modeling_v3.io import canonical_hash

    table = json.loads(Path(options.tasks).read_text())
    table["inputs"] = {"label": "真实响应图：校准与独立测试"}
    table["task_list_hash"] = canonical_hash(
        {k: v for k, v in table.items() if k != "task_list_hash"}
    )
    Path(options.tasks).write_text(json.dumps(table, ensure_ascii=False))
    assert len(submitter.build_plan(options)["commands"]) == 3


def child_table(submitter, options):
    previous = json.loads(Path(options.tasks).read_text())
    child = json.loads(json.dumps(previous))
    child["parent_task_list_hash"] = previous["task_list_hash"]
    child["tasks"].append({"id": "D_map_new", "kind": "map", "stage": "D", "worker": 0})
    extension = {
        "schema": "ssvc-v4-phase-extension-1",
        "phase": "D",
        "campaign_id": child["campaign_id"],
        "config_hash": submitter.digest(child["config"]),
        "workers": 2,
        "tasks": [child["tasks"][-1]],
    }
    extension["extension_hash"] = submitter.digest(extension)
    child["phase_extensions"] = [extension]
    child["stages"] = ["D"]
    child["task_list_hash"] = submitter.digest(
        {k: v for k, v in child.items() if k != "task_list_hash"}
    )
    path = Path(options.tasks).with_name("tasks_D.json")
    path.write_text(json.dumps(child))
    options.tasks = str(path)
    return path, child


def test_explicit_phase_extension_waits_for_active_workers_then_reuses_authorization(
    submitter, options
):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    child_table(submitter, options)
    options.stage, options.resume, options.extend_phase, options.execute_gpu = (
        "gpu",
        True,
        True,
        False,
    )
    with pytest.raises(ValueError, match="active"):
        submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 3
    scheduler.states["102"] = scheduler.states["103"] = "COMPLETED"
    result = submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 5
    assert all(row["action"] == "SUBMITTED" for row in result["jobs"])
    record = json.loads((Path(options.campaign_root) / "SLURM_SUBMISSIONS.json").read_text())
    assert record["gpu_execution_enabled"]
    assert len([row for row in record["jobs"].values() if row["kind"] == "gpu"]) == 4


def test_extension_cannot_rewrite_old_tasks_or_overwrite_old_manifest(submitter, options):
    options.submit = options.execute_gpu = True
    scheduler = Scheduler()
    submitter.execute(options, run=scheduler)
    path, child = child_table(submitter, options)
    options.stage, options.resume, options.extend_phase = "gpu", True, True
    scheduler.states["102"] = scheduler.states["103"] = "COMPLETED"
    child["tasks"][0]["id"] = "rewritten_old_task"
    child["task_list_hash"] = submitter.digest(
        {k: v for k, v in child.items() if k != "task_list_hash"}
    )
    path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match=r"append|old|existing"):
        submitter.execute(options, run=scheduler)
    assert len(scheduler.submissions) == 3
