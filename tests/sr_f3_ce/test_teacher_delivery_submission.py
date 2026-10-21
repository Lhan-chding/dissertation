"""Teacher replacement preserves the retired attempt and cannot duplicate jobs."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce"
SOURCE = BASE / "teacher_delivery/submit_delivery_recovery.py"
spec = importlib.util.spec_from_file_location("teacher_submit_test", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
HISTORY_SOURCE = BASE / "teacher_delivery/teacher_history.py"
spec = importlib.util.spec_from_file_location("teacher_history_test", HISTORY_SOURCE)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    deploy = tmp_path / "operations/new"
    deploy.mkdir(parents=True)
    for name in ("submit_delivery_recovery.py", "teacher_history.py"):
        (deploy / name).write_bytes((BASE / "teacher_delivery" / name).read_bytes())
    monkeypatch.setattr(m, "__file__", str(deploy / SOURCE.name))
    work = tmp_path / "operations/delivery_recovery/20261021_delivery_teacher"
    work.mkdir(parents=True)
    script = work / "run.sbatch"
    script.write_text("#!/bin/bash\n# teacher exact original builder\n")
    auth = dict(
        root=str(tmp_path),
        extra_sources={str(p.relative_to(tmp_path)): m.sha(p) for p in deploy.glob("*.py")},
        recovery_directory=str(work.relative_to(tmp_path)),
        script=str(script.relative_to(tmp_path)),
        script_sha256=m.sha(script),
        user="testuser",
        job_name="teacher-LH",
        comment="teacher",
        node="cpu-1",
        qos=m.TEACHER_QOS,
        account="rose",
        cpus=22,
        memory_mib=67584,
        time_limit="3-00:00:00",
        retirement_directory=h.RETIREMENT,
        retired_directories=sorted(h.DIRECTORIES),
    )
    old_work = tmp_path / h.OLD_WORK
    old_work.mkdir(parents=True)
    old_script = old_work / "run.sbatch"
    old_script.write_text("#!/bin/bash\n# preserved rose\n")
    old_deploy = tmp_path / "operations" / h.COMMITS[0]
    old_deploy.mkdir(parents=True)
    for name in (
        "recover_delivery.py",
        "submit_delivery_recovery.py",
        "verify_recovered_migration.py",
    ):
        (old_deploy / name).write_bytes((BASE / name).read_bytes())
    old_auth = {
        **auth,
        "extra_sources": {
            str((old_deploy / "submit_delivery_recovery.py").relative_to(tmp_path)): m.sha(
                old_deploy / "submit_delivery_recovery.py"
            )
        },
        "qos": "rose",
        "script": str(old_script.relative_to(tmp_path)),
        "script_sha256": m.sha(old_script),
        "recovery_directory": h.OLD_WORK,
        "job_name": "rose-LH",
        "comment": "old",
    }
    save(old_deploy / "RECOVERY_AUTHORIZATION.json", old_auth)
    for commit, names in zip(
        h.COMMITS[1:],
        (
            ("HELD_RESUME_AUTHORIZATION.json", "resume_held_delivery.py"),
            ("HELD_CAPACITY_AUTHORIZATION.json", "continue_held_delivery.py"),
        ),
        strict=True,
    ):
        for name in names:
            p = tmp_path / "operations" / commit / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}\n")
    for name in h.DIRECTORIES:
        p = tmp_path / name / "preserved.json"
        save(p, {"old": True})
    auth["retired_files"] = {
        str(p.relative_to(tmp_path)): m.sha(p)
        for p in tmp_path.rglob("*")
        if p.is_file() and not p.is_relative_to(deploy) and not p.is_relative_to(work)
    }
    authorization = deploy / "AUTH.json"
    save(authorization, auth)
    calls, faults = [], {}
    state = {"submitted": False, "cancelled": False}

    def runner(argv, **kwargs):
        calls.append(argv)
        if faults.get("timeout") == argv[0] or (
            faults.get("release_timeout") and argv[:2] == ["scontrol", "release"]
        ):
            raise subprocess.TimeoutExpired(argv, 30)
        if argv[0] == "sacctmgr" and "qos" in argv:
            text = f"Name|GrpSubmit|MaxSubmitPU|GrpJobs|MaxJobsPU|Flags|\n{m.TEACHER_QOS}||5||||\n"
        elif argv[0] == "sacctmgr":
            text = (
                "Cluster|Account|User|Partition|QOS|GrpSubmit|MaxSubmit|GrpJobs|MaxJobs|\n"
                f"x|rose|testuser||rose,{m.TEACHER_QOS}|||||\n"
            )
        elif argv[0] == "squeue":
            if "-j" in argv:
                text = "" if state["cancelled"] else "216142|PENDING\n"
            else:
                text = (
                    f"99|testuser|rose|{m.TEACHER_QOS}|cluster02|PENDING\n"
                    if state["submitted"]
                    else ""
                )
                text += faults.get("extra_queue", "")
        elif argv[:3] == ["scontrol", "show", "node"]:
            text = (
                f"NodeName=cpu-1 Partitions=cluster02 State={faults.get('node', 'MIXED')} "
                f"ActiveFeatures=cpu_ok CPUEfctv=96 CPUAlloc={faults.get('cpualloc', 42)} "
                f"RealMemory=339968 AllocMem={faults.get('memalloc', 129024)} "
                "MemSpecLimit=5120"
            )
        elif argv[0] == "sbatch":
            state["submitted"] = True
            text = faults.get("sbatch_output", "99\n")
        elif argv[:3] == ["scontrol", "show", "job"]:
            old = argv[3] == "216142"
            job, name, qos, path, comment = (
                ("216142", "rose-LH", "rose", old_script, "old")
                if old
                else ("99", "teacher-LH", m.TEACHER_QOS, script, "teacher")
            )
            text = (
                f"JobId={job} JobName={name} Account=rose QOS={qos} Partition=cluster02 "
                f"Command={path} Comment={comment} NumCPUs=22 NumTasks=1 CPUs/Task=22 "
                "ReqNodeList=cpu-1 TimeLimit=3-00:00:00 Requeue=0 JobState=PENDING "
                "Reason=JobHeldUser RunTime=00:00:00 UserId=testuser(123) NumNodes=1 "
                "NodeList=(null) ReqTRES=cpu=22,mem=66G,node=1"
            )
            if faults.get("started"):
                text = text.replace("JobState=PENDING", "JobState=RUNNING")
            if faults.get("wrong_qos") and not old:
                text = text.replace(f"QOS={m.TEACHER_QOS}", "QOS=rose")
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            text = (old_script if argv[3] == "216142" else script).read_text()
        elif argv[0] == "scancel":
            state["cancelled"] = True
            text = ""
        elif argv[0] == "sacct":
            old = argv[argv.index("-j") + 1] == "216142"
            if old and "ExitCode" in argv[-1]:
                text = "" if faults.get("terminal_lag") else "216142|CANCELLED by 123|None|0||0:0\n"
            elif old:
                text = "216142|PENDING|Unknown|0|\n"
            else:
                text = "" if faults.get("accounting_lag") else "99|PENDING|Unknown|0|\n"
        elif argv[:2] == ["scontrol", "release"]:
            text = ""
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    op = m.Submission(tmp_path, authorization, runner=runner, validator=lambda *_: None)
    return op, calls, faults, state


def test_retire_submit_continue_full_chain(case):
    op, calls, _, _ = case
    assert op.retire()["status"] == "CANCELLED_UNSTARTED_RECOVERY"
    assert op.submit()["status"] == "SUBMITTED"
    assert op.continue_job()["status"] == "RELEASED"
    assert op.prior_release()["job_id"] == "99"
    assert sum(a[0] == "scancel" for a in calls) == 1
    assert sum(a[0] == "sbatch" for a in calls) == 1
    assert sum(a[:2] == ["scontrol", "release"] for a in calls) == 1
    before = len(calls)
    op.retire()
    op.submit()
    op.continue_job()
    assert len(calls) == before


def test_retirement_accounting_lag_only_reads_repeat(case):
    op, calls, faults, _ = case
    faults["terminal_lag"] = True
    assert op.retire()["status"] == "WAIT_TERMINAL"
    assert op.retire()["status"] == "WAIT_TERMINAL"
    faults.clear()
    assert op.retire()["status"] == "CANCELLED_UNSTARTED_RECOVERY"
    assert sum(a[0] == "scancel" for a in calls) == 1


@pytest.mark.parametrize(
    "state", ["IDLE+COMPLETING+PLANNED", "MIXED+COMPLETING", "ALLOCATED", "IDLE+PLANNED"]
)
def test_pending_node_flags_do_not_prevent_held_submission_or_release(case, state):
    op, calls, faults, _ = case
    op.retire()
    faults["node"] = state
    assert op.submit()["status"] == "SUBMITTED"
    assert op.continue_job()["status"] == "WAIT_CAPACITY"
    assert op.continue_job()["status"] == "WAIT_CAPACITY"
    assert not any(a[:2] == ["scontrol", "release"] for a in calls)
    faults.clear()
    op.continue_job()
    op.prior_release()


def test_empty_sacct_wait_does_not_consume_release(case):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults["accounting_lag"] = True
    assert op.continue_job()["status"] == "WAIT_ACCOUNTING"
    assert op.continue_job()["status"] == "WAIT_ACCOUNTING"
    faults.clear()
    assert op.continue_job()["status"] == "RELEASED"
    op.prior_release()
    assert sum(a[0] == "sbatch" for a in calls) == 1


@pytest.mark.parametrize("fault", ["started", "wrong_qos"])
def test_started_or_rose_worker_rejected(case, fault):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults[fault] = True
    with pytest.raises(PermissionError):
        op.continue_job()
    faults.clear()
    with pytest.raises(PermissionError, match="Incomplete poll"):
        op.continue_job()
    assert not any(a[:2] == ["scontrol", "release"] for a in calls)


def test_unknown_sbatch_never_repeated(case):
    op, calls, faults, _ = case
    op.retire()
    faults["sbatch_output"] = "99\n100\n"
    with pytest.raises(PermissionError):
        op.submit()
    faults.clear()
    with pytest.raises(PermissionError, match="Prior submission"):
        op.submit()
    assert sum(a[0] == "sbatch" for a in calls) == 1


def test_unknown_release_never_repeated(case):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults["release_timeout"] = True
    with pytest.raises(subprocess.TimeoutExpired):
        op.continue_job()
    faults.clear()
    with pytest.raises(PermissionError, match="Release already"):
        op.continue_job()
    assert sum(a[:2] == ["scontrol", "release"] for a in calls) == 1


def test_started_old_job_never_cancelled(case):
    op, calls, faults, _ = case
    faults["started"] = True
    with pytest.raises(PermissionError):
        op.retire()
    assert not any(a[0] == "scancel" for a in calls)


@pytest.mark.parametrize("kind", ["cpualloc", "memalloc"])
def test_capacity_wait_conserves_resources(case, kind):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults[kind] = 96 if kind == "cpualloc" else 300000
    assert op.continue_job()["status"] == "WAIT_CAPACITY"
    assert not any(a[:2] == ["scontrol", "release"] for a in calls)


def test_history_cannot_discard_incomplete_old_poll(case):
    op, _, _, _ = case
    op.retire()
    old = op.root / sorted(h.DIRECTORIES)[0] / "preserved.json"
    old.unlink()
    with pytest.raises((PermissionError, FileNotFoundError)):
        op.submit()


def test_new_historical_file_blocks(case):
    op, _, _, _ = case
    op.retire()
    save(op.root / sorted(h.DIRECTORIES)[0] / "new.json", {})
    with pytest.raises(PermissionError, match="inventory changed"):
        op.submit()


def test_retirement_proof_rejects_changed_held_identity(case):
    op, _, _, _ = case
    op.retire()
    journal = op.root / h.RETIREMENT
    path = journal / "001_held_job.result.json"
    data = m.read(path)
    data["stdout"] = data["stdout"].replace("QOS=rose", "QOS=other")
    save(path, data)
    proof = m.read(journal / "RETIREMENT.json")
    proof["command_files"][path.name] = m.sha(path)
    save(journal / "RETIREMENT.json", proof)
    with pytest.raises(PermissionError, match="Held job request differs"):
        op.history(require_retired=True)


def test_pending_accounting_after_queue_disappears_is_retryable(case):
    op, calls, faults, _ = case
    runner = op.runner

    def lag(argv, **kwargs):
        value = runner(argv, **kwargs)
        if argv[0] == "sacct" and "ExitCode" in argv[-1] and faults.get("pending_lag"):
            value.stdout = "216142|PENDING|None|0||0:0\n"
        return value

    op.runner = lag
    faults["pending_lag"] = True
    assert op.retire()["status"] == "WAIT_TERMINAL"
    faults.clear()
    assert op.retire()["status"] == "CANCELLED_UNSTARTED_RECOVERY"
    assert sum(a[0] == "scancel" for a in calls) == 1


def test_teacher_final_gate_archives_real_retirement_and_poll_chain(case):
    import zipfile

    op, _, faults, _ = case
    # Original scientific acceptance is exercised by test_teacher_delivery_worker;
    # this fixture isolates the additional teacher gate using the real scheduler chain.
    source = op.authorization.parent / "verify_recovered_migration.py"
    source.write_text("""import json
from pathlib import Path
def verify(root, authorization, *, runner):
    root = Path(root)
    for name in ("FINAL_RECOVERY_VERIFIED", "FINAL_MIGRATION_VERIFIED"):
        (root / "delivery" / (name + ".json")).write_text(json.dumps({"status": name}))
    return {"archive": {"sha256": "fixture-archive-sha"}}
""")
    auth = m.read(op.authorization)
    auth["extra_sources"][str(source.relative_to(op.root))] = m.sha(source)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    op.retire()
    op.submit()
    faults["node"] = "IDLE+COMPLETING+PLANNED"
    op.continue_job()
    faults.clear()
    release = op.continue_job()
    delivery = op.root / "delivery"
    delivery.mkdir()
    archive = delivery / "SR_F3_CE_20261016.zip"
    paths = op.history(require_retired=True)[1] + [
        op.work / n for n in release["teacher_audit_files"]
    ]
    with zipfile.ZipFile(archive, "w") as package:
        for path in sorted(set(paths)):
            package.write(path, "evidence/" + str(path.relative_to(op.root)))
    result = op.verify_final()
    assert result["status"] == "FINAL_TEACHER_RECOVERY_VERIFIED"
    assert result["retired_job_id"] == "216142"
    assert result["job_id"] == "99"
    assert result["supersedes"] == ["FINAL_HELD_RECOVERY_VERIFIED", "FINAL_HELD_CAPACITY_VERIFIED"]
    assert op.verify_final() == result
    # Removing unique old interrupted-poll audit must fail, despite passing inner gates.
    with zipfile.ZipFile(archive, "w") as package:
        for path in sorted(set(paths)):
            if path.name != "preserved.json":
                package.write(path, "evidence/" + str(path.relative_to(op.root)))
    with pytest.raises(PermissionError, match="history missing"):
        op.verify_final()


def test_poll_bytes_cannot_be_changed_after_release(case):
    op, _, _, _ = case
    op.retire()
    op.submit()
    result = op.continue_job()
    path = op.work / next(n for n in result["teacher_audit_files"] if n.endswith("RESULT.json"))
    save(path, {"status": "PASS"})
    with pytest.raises(PermissionError, match="polling inventory changed"):
        op.prior_release()


def test_unknown_cancel_never_repeated(case):
    op, calls, faults, _ = case
    faults["timeout"] = "scancel"
    with pytest.raises(subprocess.TimeoutExpired):
        op.retire()
    faults.clear()
    with pytest.raises(PermissionError, match="Cancellation outcome unknown"):
        op.retire()
    assert sum(a[0] == "scancel" for a in calls) == 1
