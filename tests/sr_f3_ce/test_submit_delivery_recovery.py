import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/submit_delivery_recovery.py"
spec = importlib.util.spec_from_file_location("submit_recovery_test", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def case(tmp_path, monkeypatch):
    source = tmp_path / "operations/test/submit_delivery_recovery.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(SOURCE.read_bytes())
    monkeypatch.setattr(m, "__file__", str(source))
    script = tmp_path / "operations/delivery_recovery/0001/run.sbatch"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\ntrue\n")
    auth = dict(
        root=str(tmp_path),
        extra_sources={str(source.relative_to(tmp_path)): m.sha(source)},
        recovery_directory="operations/delivery_recovery/0001",
        script=str(script.relative_to(tmp_path)),
        script_sha256=m.sha(script),
        user="testuser",
        job_name="delivery-LH",
        comment="exact-recovery",
        node="cpu-1",
        qos="rose",
        account="rose",
        cpus=22,
        memory_mib=67584,
        time_limit="3-00:00:00",
    )
    authorization = source.parent / "AUTH.json"
    authorization.write_text(json.dumps(auth))
    calls = []
    faults = {}

    def runner(argv, **kwargs):
        calls.append(argv)
        if faults.get("timeout") == argv[0]:
            raise subprocess.TimeoutExpired(argv, 30)
        if argv[0] == "sacctmgr" and "qos" in argv:
            text = "Name|GrpSubmit|MaxSubmitPU|GrpJobs|MaxJobsPU|Flags|\nrose||5||2||\n"
        elif argv[0] == "sacctmgr":
            text = (
                "Cluster|Account|User|Partition|QOS|GrpSubmit|MaxSubmit|GrpJobs|MaxJobs|\n"
                "x|rose|testuser||rose,soujanya-poria-startfund-2026-03|||||\n"
            )
        elif argv[0] == "squeue":
            text = (
                "99|testuser|rose|rose|cluster02|PENDING\n"
                if any(x[0] == "sbatch" for x in calls)
                else ""
            )
            text += faults.get("queue_extra", "")
        elif argv[:4] == ["scontrol", "show", "node", "cpu-1"]:
            text = (
                "NodeName=cpu-1 Partitions=cluster02 State=MIXED ActiveFeatures=cpu_ok "
                "CPUEfctv=96 CPUAlloc=42 RealMemory=339968 AllocMem=129024 MemSpecLimit=5120"
            )
        elif argv[0] == "sbatch":
            text = faults.get("sbatch_output", "99\n")
        elif argv[:3] == ["scontrol", "show", "job"]:
            text = (
                "JobId=99 JobName=delivery-LH Account=rose QOS=rose Partition=cluster02 "
                f"Command={script} Comment=exact-recovery NumCPUs=22 NumTasks=1 CPUs/Task=22 "
                "ReqNodeList=cpu-1 TimeLimit=3-00:00:00 Requeue=0 JobState=PENDING "
                "Reason=JobHeldUser RunTime=00:00:00 UserId=testuser(123) NumNodes=1 "
                "NodeList=(null) ReqTRES=cpu=22,mem=66G,node=1"
            )
            if faults.get("plugin_12g"):
                text = text.replace("mem=66G", "mem=12G")
            if faults.get("started"):
                text = text.replace("JobState=PENDING", "JobState=RUNNING")
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            text = script.read_text() + ("changed" if faults.get("script_changed") else "")
        elif argv[0] == "sacct":
            text = "99|PENDING|Unknown|0|\n"
        elif argv[:2] == ["scontrol", "release"]:
            text = ""
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    operation = m.Submission(tmp_path, authorization, runner=runner, validator=lambda *a: None)
    return operation, calls, faults, authorization


def test_one_submission_released_and_idempotent(case):
    op, calls, _, _ = case
    result = op.submit()
    assert result["job_id"] == "99" and result["status"] == "RELEASED"
    n = len(calls)
    assert op.submit() == result
    assert len(calls) == n
    assert sum(x[0] == "sbatch" for x in calls) == 1
    assert not any(x[0] in ("scancel", "srun") or "requeue" in x for x in calls)
    assert (op.work / "MAY_HAVE_SUBMITTED.json").exists()


@pytest.mark.parametrize(
    "fault,value",
    [
        ("timeout", "sbatch"),
        ("sbatch_output", "99\n100\n"),
        ("started", True),
        ("script_changed", True),
        ("timeout", "scontrol"),
    ],
)
def test_unknown_or_race_never_resubmits(case, fault, value):
    op, calls, faults, _ = case
    # scontrol timeout occurs in preflight before submission and is safe.
    faults[fault] = value
    with pytest.raises((PermissionError, subprocess.TimeoutExpired)):
        op.submit()
    previous = sum(x[0] == "sbatch" for x in calls)
    if (op.work / "MAY_HAVE_SUBMITTED.json").exists():
        faults.clear()
        with pytest.raises(PermissionError, match="Prior submission"):
            op.submit()
        assert sum(x[0] == "sbatch" for x in calls) == previous
    assert ["scontrol", "release", "99"] not in calls


def test_capacity_prevents_submission(case):
    op, calls, faults, _ = case
    faults["queue_extra"] = "".join(f"{i}|testuser|rose|rose|cluster02|PENDING\n" for i in range(5))
    with pytest.raises(PermissionError, match="capacity"):
        op.submit()
    assert not any(x[0] == "sbatch" for x in calls)


def test_release_timeout_keeps_unknown_and_no_retry(case):
    op, calls, _, _ = case
    original = op.runner

    def run(argv, **kwargs):
        if argv[:2] == ["scontrol", "release"]:
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, 30)
        return original(argv, **kwargs)

    op.runner = run
    with pytest.raises(subprocess.TimeoutExpired):
        op.submit()
    assert (op.work / "RELEASE_INTENT.json").exists()
    assert not (op.work / "RELEASE.json").exists()
    with pytest.raises(PermissionError, match="Prior submission"):
        op.submit()
    assert sum(x[0] == "sbatch" for x in calls) == 1


def test_dedicated_lock_race(case):
    op, calls, _, _ = case
    op.work.mkdir(exist_ok=True)
    with (op.work / "SUBMIT.lock").open("a") as lock:
        m.fcntl.flock(lock, m.fcntl.LOCK_EX | m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            op.submit()
    assert not calls


def test_changed_receipt_rejects_idempotence(case):
    op, _, _, _ = case
    op.submit()
    (op.work / "SUBMITTED.json").write_text("{}")
    with pytest.raises(PermissionError, match="chain changed"):
        op.submit()


@pytest.mark.parametrize(
    "key,value",
    [
        ("memory_mib", 12288),
        ("cpus", 8),
        ("node", "cpu-2"),
        ("qos", "override-limits-but-killable"),
    ],
)
def test_resource_changes_rejected(case, key, value):
    op, calls, _, _ = case
    op.auth[key] = value
    with pytest.raises(PermissionError, match="resources"):
        op.submit()
    assert not calls


def test_mutated_authorization_rejected(case):
    op, calls, _, auth = case
    auth.write_text(auth.read_text() + "\n")
    with pytest.raises(PermissionError, match="Authorization changed"):
        op.submit()
    assert not calls


def test_tampered_command_receipt_rejected(case):
    op, _, _, _ = case
    result = op.submit()
    path = op.work / next(iter(result["command_files"]))
    path.write_text("{}")
    with pytest.raises(PermissionError, match="command evidence changed"):
        op.submit()


@pytest.mark.parametrize(
    "old,new",
    [
        ("rose||5||2||", "rose|10|5||2||"),
        ("rose||5||2||", "rose||6||2||"),
        ("CPUAlloc=42", "CPUAlloc=94"),
        ("AllocMem=129024", "AllocMem=330000"),
        ("State=MIXED", "State=MIXED+DRAIN"),
    ],
)
def test_unproven_capacity_prevents_sbatch(case, old, new):
    op, calls, _, _ = case
    original = op.runner

    def run(argv, **kwargs):
        result = original(argv, **kwargs)
        result.stdout = result.stdout.replace(old, new)
        return result

    op.runner = run
    with pytest.raises(PermissionError):
        op.submit()
    assert not any(x[0] == "sbatch" for x in calls)


def test_running_limit_prevents_sbatch(case):
    op, calls, faults, _ = case
    faults["queue_extra"] = "".join(f"{i}|testuser|rose|rose|cluster02|RUNNING\n" for i in range(2))
    with pytest.raises(PermissionError, match="capacity"):
        op.submit()
    assert not any(x[0] == "sbatch" for x in calls)


def test_release_counts_original_job_once(case):
    op, calls, faults, _ = case
    faults["queue_extra"] = "".join(f"{i}|testuser|rose|rose|cluster02|PENDING\n" for i in range(4))
    assert op.submit()["status"] == "RELEASED"
    assert sum(x[0] == "sbatch" for x in calls) == 1


@pytest.mark.parametrize("change", ["wrong_release", "intent_mismatch", "missing_pair", "started"])
def test_rehashed_command_semantic_tampering_rejected(case, change):
    op, _, _, _ = case
    result = op.submit()
    name = next(n for n in result["command_files"] if n.endswith("016_release.result.json"))
    if change == "intent_mismatch":
        name = name.replace(".result.json", ".intent.json")
    elif change == "started":
        name = next(n for n in result["command_files"] if n.endswith("006_held_job.result.json"))
    if change == "missing_pair":
        del result["command_files"][name]
    else:
        path = op.work / name
        value = m.read(path)
        if change == "started":
            value["stdout"] = value["stdout"].replace("JobState=PENDING", "JobState=RUNNING")
        else:
            value["argv"] = ["scontrol", "release", "123"]
        path.write_text(json.dumps(value))
        result["command_files"][name] = m.sha(path)
    (op.work / "RELEASE.json").write_text(json.dumps(result))
    with pytest.raises(PermissionError):
        op.submit()


def test_immutable_atomic_receipt_publication(tmp_path, monkeypatch):
    target = tmp_path / "RELEASE.json"
    original = m.os.link
    seen = []

    def link(source, destination):
        assert not Path(destination).exists()
        seen.append(m.read(source))
        original(source, destination)

    monkeypatch.setattr(m.os, "link", link)
    m.save(target, {"status": "RELEASED"})
    assert seen == [{"status": "RELEASED"}]
    monkeypatch.setattr(m.os, "link", original)
    with pytest.raises(FileExistsError):
        m.save(target, {"status": "CHANGED"})
    assert m.read(target) == {"status": "RELEASED"}
    assert list(tmp_path.iterdir()) == [target]


def test_plugin_rewrites_memory_to_12g_blocks_release_and_retry(case):
    op, calls, faults, _ = case
    faults["plugin_12g"] = True
    with pytest.raises(PermissionError, match="Resource request differs"):
        op.submit()
    assert (op.work / "SUBMITTED.json").exists()
    assert not (op.work / "ALLOCATION_VERIFIED.json").exists()
    assert ["scontrol", "release", "99"] not in calls
    faults.clear()
    with pytest.raises(PermissionError, match="Prior submission"):
        op.submit()
    assert sum(argv[0] == "sbatch" for argv in calls) == 1


def test_cpu_policy_capacity_requires_all_22_cores(case):
    op, calls, _, _ = case
    original = op.runner

    def runner(argv, **kwargs):
        result = original(argv, **kwargs)
        if argv[:3] == ["scontrol", "show", "node"]:
            result.stdout = result.stdout.replace("CPUAlloc=42", "CPUAlloc=75")
        return result

    op.runner = runner
    with pytest.raises(PermissionError, match="Insufficient CPUs"):
        op.submit()
    assert not any(argv[0] == "sbatch" for argv in calls)
