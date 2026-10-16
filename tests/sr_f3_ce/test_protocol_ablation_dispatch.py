"""CPU fixtures exercise the frozen Slurm guards without any model calls."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import ablation, orchestration

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts/sr_f3_ce/dispatch_protocol_ablation.py"
# Exact non-model submission evidence copied from the real 207205 ledger.
# Embedded here so the reviewed auxiliary test is independently portable.
MANUAL_SAMPLE = {
    "INTENT.json": {
        "ablation_manifest_sha256": (
            "eade7e9470ea7016e1109a347153ace9f930ac698779b654f93006228fb44db1"
        ),
        "code_commit": "652e52d530520a276b6420ff0f4ee479dcde6db1",
        "comment": "srf3ce-fix-ablation--P1--b0",
        "cpu_node": None,
        "cpus": 4,
        "created": "2026-10-16T07:39:34.069010+00:00",
        "gpu_count": 1,
        "job_name": "SRF3CE-ABLATION-P1-b0-LH",
        "key": "ablation--P1--b0",
        "memory_gib": 90,
        "partition": "cluster02",
        "qos": "soujanya-poria-startfund-2026-03",
        "run_identity_sha256": "7ede019e5e2429903076dbb9bb794eb069917be0f4aa33f315b40b1531c3312f",
        "sbatch": [
            "sbatch",
            "--hold",
            "--parsable",
            "--no-requeue",
            "--open-mode=append",
            "--account=rose",
            "--partition=cluster02",
            "--qos=soujanya-poria-startfund-2026-03",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--mem=90G",
            "--time=4320",
            "--gres=gpu:pro6000:1",
            "--constraint=highmem",
            "--job-name=SRF3CE-ABLATION-P1-b0-LH",
            "--comment=srf3ce-fix-ablation--P1--b0",
            "--output=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation/ablation/submissions/ablation--P1--b0/slurm-%j.log",
            "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation/ablation/submissions/ablation--P1--b0/run.sbatch",
        ],
        "script": (
            "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/"
            "ablation/ablation/submissions/ablation--P1--b0/run.sbatch"
        ),
        "script_sha256": "547b02367ce0fe2fbddf115567e92245d8cb3ff499a6ccc0c2c620e6b015a2ae",
        "task": {
            "block": 0,
            "job_name": "SRF3CE-ABLATION-P1-b0-LH",
            "outputs": 256,
            "parent_id": "SRF2_J6_s71001",
            "protocol_revision": "v1",
            "variant": "P1",
        },
        "user": "varun024",
    },
    "STATE.json": {
        "gpu_count": 1,
        "intent_sha256": "39fa1fa04ba29c7d2088b17474195a1090b9c2a9558d0be6501967a2de16baa0",
        "job_id": "207205",
        "key": "ablation--P1--b0",
        "qos": "soujanya-poria-startfund-2026-03",
        "status": "SUBMITTED",
    },
    "ALLOCATION_VERIFIED.json": {
        "fields": {
            "Account": "rose",
            "AccrueTime": "Unknown",
            "AllocTRES": "(null)",
            "BatchFlag": "1",
            "Command": (
                "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/"
                "ablation/ablation/submissions/ablation--P1--b0/run.sbatch"
            ),
            "Comment": "srf3ce-fix-ablation--P1--b0",
            "Contiguous": "0",
            "CoreSpec": "*",
            "Deadline": "N/A",
            "DelayBoot": "00:00:00",
            "Dependency": "(null)",
            "EligibleTime": "Unknown",
            "EndTime": "Unknown",
            "ExcNodeList": "(null)",
            "ExitCode": "0:0",
            "Features": "highmem",
            "GroupId": "varun024(295800338)",
            "JobId": "207205",
            "JobName": "SRF3CE-ABLATION-P1-b0-LH",
            "JobState": "PENDING",
            "LastSchedEval": "2026-10-16T07:39:34",
            "Licenses": "(null)",
            "MCS_label": "N/A",
            "MinCPUsNode": "4",
            "MinMemoryNode": "90G",
            "MinTmpDiskNode": "0",
            "Network": "(null)",
            "Nice": "0",
            "NodeList": "",
            "NumCPUs": "4",
            "NumNodes": "1-1",
            "NumTasks": "1",
            "OverSubscribe": "OK",
            "Partition": "cluster02",
            "Power": "",
            "Priority": "0",
            "QOS": "soujanya-poria-startfund-2026-03",
            "Reason": "JobHeldUser",
            "Reboot": "0",
            "ReqNodeList": "(null)",
            "ReqTRES": "cpu=4,mem=90G,node=1,billing=8,gres/gpu=1,gres/gpu:pro6000=1",
            "Requeue": "0",
            "Restarts": "0",
            "RunTime": "00:00:00",
            "Scheduler": "Main",
            "SecsPreSuspend": "0",
            "StartTime": "Unknown",
            "StdErr": (
                "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/"
                "ablation/ablation/submissions/ablation--P1--b0/slurm-207205.log"
            ),
            "StdIn": "/dev/null",
            "StdOut": (
                "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/"
                "ablation/ablation/submissions/ablation--P1--b0/slurm-207205.log"
            ),
            "SubmitTime": "2026-10-16T07:39:34",
            "SuspendTime": "None",
            "TimeLimit": "3-00:00:00",
            "TimeMin": "N/A",
            "TresPerNode": "gres/gpu:pro6000:1",
            "TresPerTask": "cpu:4",
            "UserId": "varun024(295800338)",
            "WorkDir": "/home/varun024",
        },
        "job_id": "207205",
        "script_sha256": "547b02367ce0fe2fbddf115567e92245d8cb3ff499a6ccc0c2c620e6b015a2ae",
    },
    "RELEASE.json": {
        "at": "2026-10-16T07:39:34.142250+00:00",
        "returncode": 0,
        "stderr": "",
        "stdout": "",
    },
    "run.sbatch": "#!/bin/bash\n"
    "set -euo pipefail\n"
    "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
    "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
    "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
    "export "
    "PYTHONPATH=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation/code/src\n"
    "cd "
    "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation/code\n"
    "test ! -e "
    "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation/STOP\n"
    "exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 "
    "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/"
    "ablation/code/scripts/sr_f3_ce/run_protocol_ablation.py "
    "--mode run --root "
    "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation "
    "--block 0 --variant P1\n",
}


def module():
    spec = importlib.util.spec_from_file_location("ablation_dispatch_fixture", HELPER)
    result = importlib.util.module_from_spec(spec)
    exec(compile(HELPER.read_bytes(), str(HELPER), "exec"), result.__dict__)
    return result


def fields(intent, job, *, running=False):
    gpu = intent["gpu_count"]
    tres = f"cpu=4,mem={intent['memory_gib']}G,node=1"
    if gpu:
        tres += ",gres/gpu=1,gres/gpu:pro6000=1"
    return dict(
        JobId=job,
        JobName=intent["job_name"],
        Account="rose",
        Partition="cluster02",
        QOS=intent["qos"],
        Command=intent["script"],
        Comment=intent["comment"],
        NumCPUs="4",
        NumNodes="1",
        ReqTRES=tres,
        AllocTRES=tres if running else "(null)",
        UserId=intent["user"] + "(42)",
        Features="highmem" if gpu else "(null)",
        ReqNodeList="(null)" if gpu else "cpu-2",
        NodeList="gpu-1" if gpu else "cpu-2",
        Requeue="0",
        TimeLimit="3-00:00:00",
        JobState="RUNNING" if running else "PENDING",
        Reason="None" if running else "JobHeldUser",
    )


class Slurm:
    def __init__(self, helper):
        self.helper = helper
        self.jobs = {}
        self.scripts = {}
        self.terminals = {}
        self.calls = []
        self.terminal_calls = []
        self.next_job = 100
        self.unknown_submit = False
        self.unknown_release = False
        self.script_changed = False
        self.extra_after_hold = False

    def queue(self, user):
        return copy.deepcopy(self.jobs)

    def show(self, job):
        return copy.deepcopy(self.jobs[job])

    def batch_script(self, job):
        return self.scripts[job] + ("# altered\n" if self.script_changed else "")

    def terminal(self, job):
        self.terminal_calls.append(job)
        return self.terminals.get(job)

    def run(self, command):
        self.calls.append(command)
        if command[0] == "sbatch":
            if self.unknown_submit:
                raise TimeoutError("network outcome unknown")
            path = Path(command[-1])
            intent = self.helper.read(path.parent / "INTENT.json")
            job = str(self.next_job)
            self.next_job += 1
            self.jobs[job] = fields(intent, job)
            self.scripts[job] = path.read_text()
            if self.extra_after_hold:
                other = dict(intent, user="another-user")
                for number in range(200, 205):
                    self.jobs[str(number)] = fields(other, str(number), running=True)
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if command[:2] == ["scontrol", "release"]:
            if self.unknown_release:
                raise TimeoutError("release outcome unknown")
            job = command[2]
            self.jobs[job].update(
                JobState="RUNNING", Reason="None", AllocTRES=self.jobs[job]["ReqTRES"]
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    helper = module()
    root = tmp_path / "ablation"
    root.mkdir()
    (root / "deployment").mkdir()
    (root / "code").mkdir()
    helper.save(root / "RUN_IDENTITY.json", {"fixture": "run"})
    helper.save(root / "ABLATION_MANIFEST.json", {"fixture": "manifest"})
    helper.save(root / "deployment/SOURCE_MANIFEST.json", {"fixture": "source"})
    registration = dict(
        schema_version=1,
        status="REGISTERED_ABLATION_DISPATCH",
        root=str(root),
        source_commit=helper.SOURCE_COMMIT,
        driver_commit="a" * 40,
        python=sys.executable,
        user="fixture-user",
        poll_seconds=60,
        run_identity_sha256=helper.sha(root / "RUN_IDENTITY.json"),
        ablation_manifest_sha256=helper.sha(root / "ABLATION_MANIFEST.json"),
        scientific_source_manifest_sha256=helper.sha(root / "deployment/SOURCE_MANIFEST.json"),
        helper_source_manifest_sha256="b" * 64,
        tasks=ablation.task_matrix(),
        max_gpus=5,
        max_submitted=5,
        scientific_retry_authorized=False,
        scientific_execution_authorized=False,
    )
    helper.save(helper.operation_directory(root) / "OPERATION_IDENTITY.json", registration)
    selections = []

    def verify_selection(selected_root):
        selections.append(selected_root)
        return helper.read(selected_root / "PROTOCOL_SELECTION.json")

    api = SimpleNamespace(
        ablation=SimpleNamespace(
            task_matrix=ablation.task_matrix,
            verify_manifest=lambda root: None,
            verify_selection=verify_selection,
        ),
        orchestration=orchestration,
        slurm=Slurm(helper),
    )
    monkeypatch.setattr(helper, "verify_operation", lambda *args: registration)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return SimpleNamespace(
        h=helper,
        root=root,
        reg=registration,
        api=api,
        selections=selections,
        monkeypatch=monkeypatch,
    )


def add_attempt(s, key, job, *, driver=False, terminal=None, complete=False):
    h = s.h
    directory = (h.drivers(s.root) if driver else h.submissions(s.root)) / key
    directory.mkdir(parents=True)
    intent = h._intent(s.reg, key, driver=driver)
    body = h._body(
        s.reg, task=None if driver else s.reg["tasks"][key], driver_key=key if driver else None
    )
    (directory / "run.sbatch").write_text(body)
    intent["script_sha256"] = h.sha(directory / "run.sbatch")
    intent["sbatch"] = h._sbatch(intent, directory)
    h.save(directory / "INTENT.json", intent)
    state = dict(
        key=key,
        status="SUBMITTED",
        gpu_count=0 if driver else 1,
        qos=h.QOS,
        job_id=job,
        intent_sha256=h.sha(directory / "INTENT.json"),
    )
    h.save(directory / "STATE.json", state)
    h.save(directory / "SUBMISSION.json", dict(returncode=0, stdout=job + "\n", stderr=""))
    h.save(
        directory / "ALLOCATION_VERIFIED.json",
        dict(job_id=job, script_sha256=intent["script_sha256"], fields=fields(intent, job)),
    )
    h.save(directory / "RELEASE_INTENT.json", dict(job_id=job))
    # The genuine manual release receipt has no job_id.
    h.save(directory / "RELEASE.json", dict(returncode=0, stdout="", stderr=""))
    s.api.slurm.jobs[job] = fields(intent, job, running=True)
    s.api.slurm.scripts[job] = body
    if driver:
        s.monkeypatch.setenv("SLURM_JOB_ID", job)
    if terminal:
        s.api.slurm.jobs.pop(job)
        s.api.slurm.terminals[job] = dict(
            job_id=job,
            state=terminal,
            exit_code="0:0" if terminal == "COMPLETED" else "1:0",
            user=s.reg["user"],
            account="rose",
            qos=h.QOS,
            name=intent["job_name"],
        )
    if complete:
        raw = s.root / "ablation/raw" / key
        h.save(raw / "ALLOCATION.json", dict(job_id=job))
        h.save(
            raw / "COMPLETE.json",
            dict(
                status="COMPLETE",
                task_id=key,
                job_id=job,
                job_name=intent["job_name"],
                user=s.reg["user"],
                records=256,
                ablation_manifest_sha256=s.reg["ablation_manifest_sha256"],
                optimizer_steps=0,
                teacher_forward_calls=0,
                monitor_calls=0,
                allocation_sha256=h.sha(raw / "ALLOCATION.json"),
                raw_groups=[{"fixture": i} for i in range(64)],
                raw_artifacts=[{"fixture": i} for i in range(128)],
            ),
        )
    return directory


def cpu(s):
    return add_attempt(s, "driver-0001", "99", driver=True)


def test_actual_manual_schema_and_script_are_accepted():
    h = module()
    intent = MANUAL_SAMPLE["INTENT.json"]
    root = Path(intent["script"]).parents[3]
    body = MANUAL_SAMPLE["run.sbatch"]
    python = body.splitlines()[-1].split()[1]
    reg = dict(root=str(root), python=python)
    assert h._body(reg, task=intent["task"]) == body
    assert h._sbatch(intent, Path(intent["script"]).parent) == intent["sbatch"]
    assert h.hashlib.sha256(body.encode()).hexdigest() == intent["script_sha256"]
    actual = MANUAL_SAMPLE["ALLOCATION_VERIFIED.json"]
    orchestration.verify_allocation(
        actual["fields"], dict(intent, job_id="207205"), intent["user"], held=True
    )
    assert MANUAL_SAMPLE["RELEASE.json"]["returncode"] == 0


def test_cpu_submission_waits_for_fifth_existing_gpu(setup):
    s = setup
    for index, key in enumerate(list(s.reg["tasks"])[:5]):
        add_attempt(s, key, str(index + 1))
    result = s.h.submit_driver(s.reg, s.api)
    assert result["status"] == "WAITING_FOR_CPU_SLOT"
    assert result["teacher_usage"] == dict(gpus=5, submitted=5)
    assert not s.api.slurm.calls
    assert not s.h.drivers(s.root).exists()


def test_cpu_held_verified_released_once_and_counts_submit_slot(setup):
    s = setup
    for index, key in enumerate(list(s.reg["tasks"])[:4]):
        add_attempt(s, key, str(index + 1))
    result = s.h.submit_driver(s.reg, s.api)
    assert result["status"] == "SUBMITTED"
    directory = s.h.drivers(s.root) / "driver-0001"
    intent = s.h.read(directory / "INTENT.json")
    assert intent["job_name"] == "SRF3CE-ABLDRIVER-N0001-LH"
    assert "--nodelist=cpu-2" in intent["sbatch"]
    assert all("gres=" not in argument for argument in intent["sbatch"])
    assert s.h.read(directory / "QUEUE_BEFORE_RELEASE.json")["teacher_usage"] == dict(
        gpus=4, submitted=5
    )
    with pytest.raises(PermissionError, match="already attempted"):
        s.h.submit_driver(s.reg, s.api)
    assert len(s.api.slurm.calls) == 2


def test_tick_cpu_plus_four_gpus_does_not_submit(setup):
    s = setup
    cpu(s)
    for index, key in enumerate(list(s.reg["tasks"])[:4]):
        add_attempt(s, key, str(index + 1))
    result = s.h.tick(s.reg, s.api)
    assert result["teacher_usage"] == dict(gpus=4, submitted=5)
    assert result["status"] == "WAITING"
    assert not s.api.slurm.calls


def test_tick_submits_next_missing_registered_task_only_once(setup):
    s = setup
    cpu(s)
    keys = list(s.reg["tasks"])
    add_attempt(s, keys[0], "1", terminal="COMPLETED", complete=True)
    first = s.h.tick(s.reg, s.api)
    second = s.h.tick(s.reg, s.api)
    assert first["task_id"] == keys[1]
    assert second["task_id"] == keys[2]
    sbatch = [call for call in s.api.slurm.calls if call[0] == "sbatch"]
    assert len(sbatch) == 2
    assert keys[0] not in " ".join(" ".join(call) for call in sbatch)
    for call in sbatch:
        assert "--hold" in call and "--no-requeue" in call
        assert any(
            argument.endswith("-LH") for argument in call if argument.startswith("--job-name=")
        )


def test_accounting_unknown_reserves_capacity_and_only_polls_terminal(setup):
    s = setup
    cpu(s)
    key = next(iter(s.reg["tasks"]))
    directory = add_attempt(s, key, "1")
    s.api.slurm.jobs.pop("1")
    first = s.h.tick(s.reg, s.api)
    second = s.h.tick(s.reg, s.api)
    assert first["status"] == second["status"] == "WAITING_FOR_ACCOUNTING"
    assert first["teacher_usage"] == dict(gpus=1, submitted=2)
    assert s.h.read(directory / "STATE.json")["status"] == "ACCOUNTING_UNKNOWN"
    assert s.api.slurm.terminal_calls == ["1", "1"]
    assert not s.api.slurm.calls


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"])
def test_model_terminal_failure_never_retries_even_with_complete(setup, status):
    s = setup
    cpu(s)
    key = next(iter(s.reg["tasks"]))
    directory = add_attempt(s, key, "1", terminal=status, complete=True)
    with pytest.raises(PermissionError, match="requires review"):
        s.h.tick(s.reg, s.api)
    assert s.h.read(directory / "STATE.json")["status"] == "FAILED"
    assert not s.api.slurm.calls
    assert not (s.root / "PROTOCOL_SELECTION.json").exists()


def test_completed_slurm_without_complete_receipt_stops(setup):
    s = setup
    cpu(s)
    directory = add_attempt(s, next(iter(s.reg["tasks"])), "1", terminal="COMPLETED")
    with pytest.raises(PermissionError, match="requires review"):
        s.h.tick(s.reg, s.api)
    assert (
        s.h.read(directory / "RECEIPT_FAILURE.json")["receipt_only_reconciliation_required"]
        is False
    )
    assert not s.api.slurm.calls


def test_unknown_sbatch_is_durable_and_never_repeated(setup):
    s = setup
    cpu(s)
    s.api.slurm.unknown_submit = True
    with pytest.raises(TimeoutError):
        s.h.tick(s.reg, s.api)
    directory = s.h.submissions(s.root) / next(iter(s.reg["tasks"]))
    assert s.h.read(directory / "STATE.json")["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(PermissionError, match="requires review"):
        s.h.tick(s.reg, s.api)
    assert len(s.api.slurm.calls) == 1


def test_unknown_release_stays_reserved_and_never_repeated(setup):
    s = setup
    cpu(s)
    s.api.slurm.unknown_release = True
    with pytest.raises(TimeoutError):
        s.h.tick(s.reg, s.api)
    directory = s.h.submissions(s.root) / next(iter(s.reg["tasks"]))
    assert s.h.read(directory / "STATE.json")["status"] == "RELEASE_UNKNOWN"
    with pytest.raises(PermissionError, match="requires review"):
        s.h.tick(s.reg, s.api)
    assert len(s.api.slurm.calls) == 2


def test_actual_batch_script_change_keeps_job_held(setup):
    s = setup
    cpu(s)
    original = s.api.slurm.batch_script
    s.monkeypatch.setattr(
        s.api.slurm,
        "batch_script",
        lambda job: original(job) + ("# tamper\n" if job != "99" else ""),
    )
    with pytest.raises(PermissionError, match="remains held"):
        s.h.tick(s.reg, s.api)
    assert len(s.api.slurm.calls) == 1
    directory = s.h.submissions(s.root) / next(iter(s.reg["tasks"]))
    assert s.h.read(directory / "STATE.json")["status"] == "HELD"


def test_capacity_race_after_hold_never_releases(setup):
    s = setup
    cpu(s)
    s.api.slurm.extra_after_hold = True
    with pytest.raises(PermissionError, match="Capacity changed"):
        s.h.tick(s.reg, s.api)
    assert len(s.api.slurm.calls) == 1


def test_foreign_teacher_gpus_count_but_rose_jobs_do_not(setup):
    s = setup
    cpu(s)
    intent = s.h._intent(s.reg, next(iter(s.reg["tasks"])))
    for number in range(1, 6):
        s.api.slurm.jobs[str(number)] = fields(
            dict(intent, user="other-user"), str(number), running=True
        )
    s.api.slurm.jobs["88"] = fields(dict(intent, qos="rose"), "88", running=True)
    result = s.h.tick(s.reg, s.api)
    assert result["status"] == "WAITING"
    assert result["teacher_usage"] == dict(gpus=5, submitted=1)
    assert not s.api.slurm.calls


@pytest.mark.parametrize("index", [0, 8])
def test_unsubmitted_partial_is_not_resampled(setup, index):
    s = setup
    cpu(s)
    key = list(s.reg["tasks"])[index]
    s.h.save(s.root / "ablation/raw" / key / "groups/group.pending.jsonl", {"partial": True})
    with pytest.raises(PermissionError, match="never regenerate"):
        s.h.tick(s.reg, s.api)
    assert not s.api.slurm.calls


def test_unregistered_raw_task_blocks_all_submissions(setup):
    s = setup
    cpu(s)
    s.h.save(s.root / "ablation/raw/not-a-task/partial.json", {"partial": True})
    with pytest.raises(PermissionError, match="outside the registered"):
        s.h.tick(s.reg, s.api)
    assert not s.api.slurm.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("JobName", "missing-suffix"),
        ("QOS", "rose"),
        ("Account", "other"),
        ("Requeue", "1"),
        ("NodeList", "cpu-1"),
        ("JobState", "PENDING"),
        ("AllocTRES", "cpu=4,mem=12G,gres/gpu=1"),
    ],
)
def test_actual_cpu_allocation_must_match(setup, field, value):
    s = setup
    cpu(s)
    s.api.slurm.jobs["99"][field] = value
    with pytest.raises(PermissionError):
        s.h.tick(s.reg, s.api)
    assert not s.api.slurm.calls


def test_cpu_environment_and_singleton_are_required(setup):
    s = setup
    cpu(s)
    s.monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(PermissionError, match="CPU Slurm"):
        s.h.tick(s.reg, s.api)
    path = s.h.operation_directory(s.root) / "DRIVER.lock"
    with s.h.lock(path), pytest.raises(RuntimeError, match="Another ablation"), s.h.lock(path):
        raise AssertionError("must not enter")


def test_manual_shared_submit_lock_blocks_mutation(setup):
    s = setup
    cpu(s)
    with orchestration.lease(s.h.submissions(s.root)), pytest.raises(RuntimeError):
        s.h.tick(s.reg, s.api)
    assert not s.api.slurm.calls


def test_eight_successes_and_ninth_running_cannot_select(setup):
    s = setup
    cpu(s)
    for number, key in enumerate(s.reg["tasks"], 1):
        add_attempt(
            s, key, str(number), terminal="COMPLETED" if number < 9 else None, complete=True
        )
    result = s.h.tick(s.reg, s.api)
    assert result["status"] == "WAITING"
    assert not s.selections
    assert not (s.h.operation_directory(s.root) / "SELECT_INTENT.json").exists()


def completed_matrix(s):
    cpu(s)
    for number, key in enumerate(s.reg["tasks"], 1):
        add_attempt(s, key, str(number), terminal="COMPLETED", complete=True)


def test_all_nine_real_successes_invoke_frozen_selection_only_once(setup):
    s = setup
    completed_matrix(s)
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        s.h.save(
            s.root / "PROTOCOL_SELECTION.json",
            dict(ablation_root=str(s.root), status="PROTOCOL_SELECTED", selected_revision="v2"),
        )
        return subprocess.CompletedProcess(command, 0, stdout="selected\n", stderr="")

    first = s.h.tick(s.reg, s.api, execute=execute)
    second = s.h.tick(s.reg, s.api, execute=execute)
    assert first == second
    assert first["status"] == "SELECTION_COMPLETE"
    assert first["scientific_execution_authorized"] is False
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        s.reg["python"],
        str(s.root / "code/scripts/sr_f3_ce/run_protocol_ablation.py"),
        "--mode",
        "select",
        "--root",
        str(s.root),
    ]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert not s.api.slurm.calls


def test_selection_invocation_unknown_is_not_repeated(setup):
    s = setup
    completed_matrix(s)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        raise TimeoutError("unknown selection")

    with pytest.raises(TimeoutError):
        s.h.tick(s.reg, s.api, execute=execute)
    with pytest.raises(PermissionError, match="outcome is unknown"):
        s.h.tick(s.reg, s.api, execute=execute)
    assert len(calls) == 1


def test_stable_queue_snapshot_does_not_append_runtime_noise(setup):
    s = setup
    cpu(s)
    for number, key in enumerate(list(s.reg["tasks"])[:4], 1):
        add_attempt(s, key, str(number))
    s.h.tick(s.reg, s.api)
    path = s.h.operation_directory(s.root) / "EVENTS.jsonl"
    before = path.read_bytes()
    s.api.slurm.jobs["1"]["RunTime"] = "00:03:42"
    s.h.tick(s.reg, s.api)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "path,field,value",
    [
        ("RELEASE.json", "returncode", 1),
        ("SUBMISSION.json", "stdout", "222\n"),
        ("ALLOCATION_VERIFIED.json", "script_sha256", "0" * 64),
    ],
)
def test_original_held_release_proof_is_required(setup, path, field, value):
    s = setup
    cpu(s)
    directory = add_attempt(s, next(iter(s.reg["tasks"])), "1")
    evidence = s.h.read(directory / path)
    evidence[field] = value
    s.h.save(directory / path, evidence)
    with pytest.raises(PermissionError, match="proof"):
        s.h.tick(s.reg, s.api)
    assert not s.api.slurm.calls


def test_resealed_task_metadata_cannot_change_frozen_matrix(setup):
    s = setup
    cpu(s)
    directory = add_attempt(s, next(iter(s.reg["tasks"])), "1")
    intent = s.h.read(directory / "INTENT.json")
    intent["task"]["outputs"] = 512
    s.h.save(directory / "INTENT.json", intent)
    state = s.h.read(directory / "STATE.json")
    state["intent_sha256"] = s.h.sha(directory / "INTENT.json")
    s.h.save(directory / "STATE.json", state)
    with pytest.raises(PermissionError, match="identity"):
        s.h.tick(s.reg, s.api)
    assert not s.api.slurm.calls


def source_fixture(tmp_path):
    h = module()
    root = tmp_path / "source-root"
    root.mkdir()
    names = (
        "src/sr_f3_ce/ablation.py",
        "src/sr_f3_ce/protocol.py",
        "src/sr_f3_ce/orchestration.py",
        "scripts/sr_f3_ce/run_protocol_ablation.py",
    )
    rows = []
    for name in names:
        path = root / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture = True\n")
        rows.append(dict(path=name, sha256=h.sha(path), bytes=path.stat().st_size))
    h.save(root / "deployment/SOURCE_MANIFEST.json", dict(code_commit=h.SOURCE_COMMIT, files=rows))
    h.save(
        root / "RUN_IDENTITY.json",
        dict(
            code_commit=h.SOURCE_COMMIT,
            run_stage="PROTOCOL_ABLATION",
            root=str(root),
            bound_files={
                "deployment/SOURCE_MANIFEST.json": h.sha(root / "deployment/SOURCE_MANIFEST.json")
            },
        ),
    )
    return h, root


@pytest.mark.parametrize(
    "mutation",
    ["wrong_commit", "pyc", "extra_source", "native_extension", "changed_source", "stop"],
)
def test_source_boundary_rejects_unbound_code(tmp_path, mutation):
    h, root = source_fixture(tmp_path)
    h.verify_sources(root, h.SOURCE_COMMIT)
    commit = h.SOURCE_COMMIT
    if mutation == "wrong_commit":
        commit = "f" * 40
    elif mutation == "pyc":
        (root / "code/cache.pyc").write_bytes(b"unchecked bytecode")
    elif mutation == "extra_source":
        (root / "code/new.py").write_text("new = True\n")
    elif mutation == "native_extension":
        (root / "code/src/sr_f3_ce/ablation.cpython-312-x86_64-linux-gnu.so").write_bytes(
            b"unregistered native extension"
        )
    elif mutation == "changed_source":
        (root / "code/src/sr_f3_ce/ablation.py").write_text("changed = True\n")
    else:
        (root / "STOP").write_text("stop\n")
    with pytest.raises(PermissionError):
        h.verify_sources(root, commit)


def test_registered_helper_hash_and_explicit_deployment_required(tmp_path, monkeypatch):
    h, root = source_fixture(tmp_path)
    commit = "a" * 40
    directory = h.operation_directory(root) / commit
    directory.mkdir(parents=True)
    source = directory / h.HELPER
    source.write_bytes(HELPER.read_bytes())
    test = directory / h.HELPER_TEST
    test.write_bytes(Path(__file__).read_bytes())
    h.save(
        directory / "SOURCE_MANIFEST.json",
        dict(
            driver_commit=commit,
            scientific_source_commit=h.SOURCE_COMMIT,
            files=[
                dict(path=path.name, sha256=h.sha(path), bytes=path.stat().st_size)
                for path in (source, test)
            ],
        ),
    )
    monkeypatch.setattr(h, "__file__", str(source))
    assert h.helper_sources(root, h.SOURCE_COMMIT, commit) == directory
    source.write_text("# changed\n")
    with pytest.raises(PermissionError, match="bytes changed"):
        h.helper_sources(root, h.SOURCE_COMMIT, commit)
