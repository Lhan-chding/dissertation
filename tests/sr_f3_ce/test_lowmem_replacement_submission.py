"""Low-memory replacement preserves the retired attempt and cannot duplicate jobs."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce"
SOURCE = BASE / "lowmem_replacement/submit_delivery_recovery.py"
spec = importlib.util.spec_from_file_location("lowmem_submit_test", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
HISTORY_SOURCE = BASE / "lowmem_replacement/lowmem_history.py"
spec = importlib.util.spec_from_file_location("lowmem_history_test", HISTORY_SOURCE)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    deploy = tmp_path / "operations/new"
    deploy.mkdir(parents=True)
    for name in ("submit_delivery_recovery.py", "lowmem_history.py"):
        (deploy / name).write_bytes((BASE / "lowmem_replacement" / name).read_bytes())
    monkeypatch.setattr(m, "__file__", str(deploy / SOURCE.name))
    work = tmp_path / "operations/delivery_recovery/20261021_delivery_lowmem_replacement"
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
        allowed_nodes=["cpu-1", "cpu-2"],
        qos=m.TEACHER_QOS,
        account="rose",
        cpus=4,
        memory_mib=12288,
        time_limit="3-00:00:00",
        retirement_directory=h.RETIREMENT,
        retired_directories=sorted(h.DIRECTORIES),
    )
    (tmp_path / "operations/delivery_recovery/20261021_delivery_teacher").mkdir(parents=True)
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
        (old_deploy / name).write_bytes((BASE / "lowmem_delivery" / name).read_bytes())
    old_auth = {
        **auth,
        "extra_sources": {
            str((old_deploy / "submit_delivery_recovery.py").relative_to(tmp_path)): m.sha(
                old_deploy / "submit_delivery_recovery.py"
            )
        },
        "qos": m.TEACHER_QOS,
        "node": "cpu-1",
        "cpus": 4,
        "memory_mib": 12288,
        "script": str(old_script.relative_to(tmp_path)),
        "script_sha256": m.sha(old_script),
        "recovery_directory": h.OLD_WORK,
        "job_name": "rose-LH",
        "comment": "old",
    }
    save(old_deploy / "RECOVERY_AUTHORIZATION.json", old_auth)
    # Prior 216142 history is an isolated boundary fixture: its production helper
    # is independently exercised by test_teacher_delivery_submission.
    (old_deploy / "lowmem_history.py").write_text(
        "def verify_history(root, auth, *, require_retired=False):\n"
        "    assert require_retired\n"
        "    return []\n"
    )
    old_invocation = "poll_submit_1"
    old_commands = old_work / "commands" / old_invocation
    old_spec = importlib.util.spec_from_file_location(
        "bound_old_submit", old_deploy / "submit_delivery_recovery.py"
    )
    old_module = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old_module)
    old_argv = dict(
        old_module.Submission.expected_commands(
            SimpleNamespace(user=old_auth["user"], script=old_script), "216983"
        )
    )
    for index, label in enumerate(("qos", "assoc", "queue", "node", "sbatch"), 1):
        argv = old_argv[label]
        save(old_commands / f"{index:03d}_{label}.intent.json", dict(argv=argv, at=1))
        save(
            old_commands / f"{index:03d}_{label}.result.json",
            dict(argv=argv, at=2, returncode=0, stdout="216983\n" if label == "sbatch" else ""),
        )
    save(
        old_commands / "RESULT.json",
        dict(
            status="READY",
            files={
                p.name: m.sha(p)
                for p in old_commands.glob("*.json")
                if not p.name.startswith("005_")
            },
        ),
    )
    save(
        old_work / "SUBMITTED.json",
        dict(
            job_id="216983",
            invocation=old_invocation,
            authorization_sha256=m.sha(old_deploy / "RECOVERY_AUTHORIZATION.json"),
            script_sha256=m.sha(old_script),
        ),
    )
    partial = old_work / "commands/poll_continue_2"
    held_fields = (
        f"JobId=216983 JobName=rose-LH Account=rose QOS={m.TEACHER_QOS} Partition=cluster02 "
        f"Command={old_script} Comment=old NumCPUs=4 NumTasks=1 CPUs/Task=4 "
        "ReqNodeList=(null) Features=cpu_ok&cpu_ok TimeLimit=3-00:00:00 Requeue=0 "
        "JobState=PENDING Reason=JobHeldUser RunTime=00:00:00 UserId=testuser(123) "
        "NumNodes=1 NodeList=(null) ReqTRES=cpu=4,mem=12G,node=1"
    )
    held_argv = old_argv["held_job"]
    save(partial / "001_held_job.intent.json", dict(argv=held_argv, at=10))
    save(
        partial / "001_held_job.result.json",
        dict(argv=held_argv, at=11, returncode=0, stdout=held_fields),
    )
    norm_op = tmp_path / "operations" / h.NORMALIZATION_COMMIT
    norm_op.mkdir()
    norm_source = norm_op / "normalize_lowmem_constraint.py"
    norm_source.write_bytes((BASE / "normalize_lowmem_constraint.py").read_bytes())
    norm_auth = dict(
        root=str(tmp_path),
        job_id="216983",
        source=str(norm_source.relative_to(tmp_path)),
        source_sha256=m.sha(norm_source),
        original_authorization=dict(
            path=str((old_deploy / "RECOVERY_AUTHORIZATION.json").relative_to(tmp_path)),
            sha256=m.sha(old_deploy / "RECOVERY_AUTHORIZATION.json"),
        ),
        partial_poll=dict(
            path=str(partial.relative_to(tmp_path)),
            files={p.name: m.sha(p) for p in partial.iterdir()},
        ),
    )
    save(norm_op / "NORMALIZATION_AUTHORIZATION.json", norm_auth)
    audit = old_work / h.NORMALIZATION_AUDIT
    save(audit / "AUTHORIZATION.json", norm_auth)
    save(
        audit / "SOURCE.json",
        dict(
            path=norm_auth["source"],
            sha256=norm_auth["source_sha256"],
            utf8=norm_source.read_text(),
        ),
    )
    for i, label in enumerate(("held_job", "stored_script", "held_accounting", "normalize"), 1):
        argv = (
            old_argv[label]
            if label != "normalize"
            else ["scontrol", "update", "JobId=216983", "Features=cpu_ok"]
        )
        stdout = dict(
            held_job=held_fields,
            stored_script=old_script.read_text(),
            held_accounting="216983|PENDING|Unknown|0|\n",
            normalize="",
        )[label]
        stderr = (
            "Changing features of a submitted job is not allowed; cancel and resubmit"
            if label == "normalize"
            else ""
        )
        save(audit / f"{i:03d}_{label}.intent.json", dict(argv=argv, at=20 + i * 2))
        save(
            audit / f"{i:03d}_{label}.result.json",
            dict(
                argv=argv,
                at=21 + i * 2,
                returncode=1 if label == "normalize" else 0,
                stdout=stdout,
                stderr=stderr,
            ),
        )
    save(
        audit / "STOP.json",
        dict(
            status="NORMALIZATION_STOPPED",
            error="PermissionError('Command failed; inspect preserved receipt: normalize')",
            at=30,
        ),
    )

    save(old_work / "preserved.json", {"old": True})
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
                text = "" if state["cancelled"] else "216983|PENDING\n"
            else:
                text = (
                    f"99|testuser|rose|{m.TEACHER_QOS}|cluster02|PENDING\n"
                    if state["submitted"]
                    else ""
                )
                text += faults.get("extra_queue", "")
        elif argv[:3] == ["scontrol", "show", "node"]:
            text = "\n".join(
                f"NodeName={node} Partitions=cluster02 Arch={faults.get('arch', 'x86_64')} "
                f"State={faults.get('node2' if node == 'cpu-2' else 'node', 'MIXED')} "
                f"ActiveFeatures=cpu_ok CPUEfctv=96 CPUAlloc={faults.get('cpualloc', 42)} "
                f"RealMemory=339968 AllocMem={faults.get('memalloc', 129024)} MemSpecLimit=5120"
                for node in ("cpu-1", "cpu-2")
            )
        elif argv[0] == "sbatch":
            state["submitted"] = True
            text = faults.get("sbatch_output", "99\n")
        elif argv[:3] == ["scontrol", "show", "job"]:
            old = argv[3] == "216983"
            job, name, qos, path, comment = (
                ("216983", "rose-LH", m.TEACHER_QOS, old_script, "old")
                if old
                else ("99", "teacher-LH", m.TEACHER_QOS, script, "teacher")
            )
            text = (
                f"JobId={job} JobName={name} Account=rose QOS={qos} Partition=cluster02 "
                f"Command={path} Comment={comment} NumCPUs={4} "
                f"NumTasks=1 CPUs/Task={4} "
                f"ReqNodeList={'(null)'} "
                f"Features={'cpu_ok&cpu_ok' if old else faults.get('features', 'cpu_ok')} "
                "TimeLimit=3-00:00:00 Requeue=0 JobState=PENDING "
                "Reason=JobHeldUser RunTime=00:00:00 UserId=testuser(123) NumNodes=1 "
                f"NodeList=(null) ReqTRES=cpu={4},mem={12}G,node=1"
            )
            if faults.get("started"):
                text = text.replace("JobState=PENDING", "JobState=RUNNING")
            if faults.get("wrong_qos") and not old:
                text = text.replace(f"QOS={m.TEACHER_QOS}", "QOS=rose")
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            text = (old_script if argv[3] == "216983" else script).read_text()
        elif argv[0] == "scancel":
            state["cancelled"] = True
            text = ""
        elif argv[0] == "sacct":
            old = argv[argv.index("-j") + 1] == "216983"
            if old and "ExitCode" in argv[-1]:
                text = "" if faults.get("terminal_lag") else "216983|CANCELLED by 123|None|0||0:0\n"
            elif old:
                text = "216983|PENDING|Unknown|0|\n"
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
    faults["node"] = faults["node2"] = state
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
    faults[kind] = 96 if kind == "cpualloc" else 330000
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
    data["stdout"] = data["stdout"].replace(f"QOS={m.TEACHER_QOS}", "QOS=other")
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
            value.stdout = "216983|PENDING|None|0||0:0\n"
        return value

    op.runner = lag
    faults["pending_lag"] = True
    assert op.retire()["status"] == "WAIT_TERMINAL"
    faults.clear()
    assert op.retire()["status"] == "CANCELLED_UNSTARTED_RECOVERY"
    assert sum(a[0] == "scancel" for a in calls) == 1


@pytest.mark.parametrize("features", ["cpu_ok", "cpu_ok&cpu_ok"])
def test_teacher_final_gate_archives_real_retirement_and_poll_chain(case, features):
    import zipfile

    op, _, faults, _ = case
    # Original scientific acceptance is exercised by test_teacher_delivery_worker;
    # this fixture isolates the additional teacher gate using the real scheduler chain.
    source = op.authorization.parent / "verify_recovered_migration.py"
    source.write_text("""import json, hashlib
from pathlib import Path
def verify(root, authorization, *, runner):
    root = Path(root)
    auth = json.loads(Path(authorization).read_text())
    profile = root / auth["profile_receipt"]
    archive = {"sha256": "fixture-archive-sha"}
    for name in ("FINAL_RECOVERY_VERIFIED", "FINAL_MIGRATION_VERIFIED"):
        receipt = {"status": name, "archive": archive,
                   "memory_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                   "delivery_revision": {"scope": "isolated outer gate fixture"}}
        (root / "delivery" / (name + ".json")).write_text(json.dumps(receipt))
    return {"archive": archive}
""")
    auth = m.read(op.authorization)
    profile = op.work / "MEMORY_PROFILE.json"
    save(profile, {"scope": "fixture only"})
    auth["profile_receipt"] = str(profile.relative_to(op.root))
    auth["extra_sources"][str(source.relative_to(op.root))] = m.sha(source)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    op.retire()
    op.submit()
    faults["node"] = faults["node2"] = "IDLE+COMPLETING+PLANNED"
    op.continue_job()
    faults.clear()
    faults["features"] = features
    release = op.continue_job()
    assert m.read(op.work / "ALLOCATION_VERIFIED.json")["fields"]["Features"] == features
    for name in release["command_files"]:
        if name.endswith("held_job.result.json"):
            assert "Features=" + features + " " in m.read(op.work / name)["stdout"]
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
    assert result["status"] == "FINAL_LOWMEM_REPLACEMENT_VERIFIED"
    assert result["retired_job_id"] == "216983"
    assert result["job_id"] == "99"
    assert result["supersedes"] == [
        "FINAL_HELD_RECOVERY_VERIFIED",
        "FINAL_HELD_CAPACITY_VERIFIED",
        "FINAL_TEACHER_RECOVERY_VERIFIED",
        "FINAL_LOWMEM_RECOVERY_VERIFIED",
        "FINAL_CONSTRAINT_NORMALIZATION_VERIFIED",
    ]
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


@pytest.mark.parametrize(
    "state", ["DOWN", "IDLE+DRAIN", "DRAINING", "IDLE+NOT_RESPONDING", "MAINT"]
)
def test_unavailable_first_cpu_candidate_does_not_block_second(case, state):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults["node"] = state
    faults["node2"] = "MIXED"
    assert op.continue_job()["status"] == "RELEASED"
    capacity = m.read(op.work / "CAPACITY_BEFORE_RELEASE.json")
    assert capacity["eligible_nodes"] == ["cpu-2"]
    assert capacity["nodes"]["cpu-1"]["State"] == state
    assert sum(a[:2] == ["scontrol", "release"] for a in calls) == 1


@pytest.mark.parametrize("states", [("DOWN", "IDLE+DRAIN"), ("MAINT", "NOT_RESPONDING")])
def test_all_unavailable_candidates_leave_repeatable_complete_wait(case, states):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults["node"], faults["node2"] = states
    for _ in range(2):
        result = op.continue_job()
        assert result["status"] == "WAIT_CAPACITY"
        assert all(state in result["reason"] for state in states)
        assert len(result["files"]) == 14
    assert not any(a[:2] == ["scontrol", "release"] for a in calls)
    assert not (op.work / "RELEASE_INTENT.json").exists()


def test_wrong_candidate_architecture_cannot_release(case):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults["arch"] = "aarch64"
    with pytest.raises(PermissionError, match="architecture"):
        op.continue_job()
    assert not any(a[:2] == ["scontrol", "release"] for a in calls)


@pytest.mark.parametrize("marker", ["RELEASE_INTENT.json", "RELEASE.json", "STARTED.json"])
def test_retirement_refuses_any_old_release_or_start_marker(case, marker):
    op, calls, _, _ = case
    save(op.root / h.OLD_WORK / marker, {"job_id": "216983"})
    with pytest.raises(PermissionError):
        op.retire()
    assert not any(a[0] == "scancel" for a in calls)


def test_retirement_refuses_missing_old_query_result_even_if_rebound(case):
    op, calls, _, _ = case
    path = op.root / h.OLD_WORK / "commands/poll_submit_1/RESULT.json"
    path.unlink()
    auth = m.read(op.authorization)
    del auth["retired_files"][str(path.relative_to(op.root))]
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    with pytest.raises(PermissionError, match="Initial submit inventory differs"):
        op.retire()
    assert not any(a[0] == "scancel" for a in calls)


def test_retirement_refuses_unknown_old_command_even_if_rebound(case):
    op, calls, _, _ = case
    directory = op.root / h.OLD_WORK / "commands/poll_submit_1"
    path = directory / "001_qos.result.json"
    value = m.read(path)
    value.pop("returncode")
    value["status"] = "UNKNOWN"
    save(path, value)
    marker = directory / "RESULT.json"
    value = m.read(marker)
    value["files"][path.name] = m.sha(path)
    save(marker, value)
    auth = m.read(op.authorization)
    for changed in (path, marker):
        auth["retired_files"][str(changed.relative_to(op.root))] = m.sha(changed)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    with pytest.raises(PermissionError, match=r"command unknown|command unknown or changed"):
        op.retire()
    assert not any(a[0] == "scancel" for a in calls)


def test_lowmem_request_cannot_silently_return_to_22_cores(case):
    op, calls, _, _ = case
    auth = m.read(op.authorization)
    auth.update(cpus=22, memory_mib=67584)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    with pytest.raises(PermissionError, match="resources differ"):
        op.submit()
    assert calls == []


@pytest.mark.parametrize(
    "argv",
    [["scontrol", "update", "JobId=216983", "QOS=rose"], ["scontrol", "show", "job", "999", "-o"]],
)
def test_retirement_refuses_changed_old_query_even_if_rebound(case, argv):
    op, calls, _, _ = case
    directory = op.root / h.OLD_WORK / "commands/poll_submit_1"
    changed = []
    for kind in ("intent", "result"):
        path = directory / f"001_qos.{kind}.json"
        value = m.read(path)
        value["argv"] = argv
        save(path, value)
        changed.append(path)
    marker = directory / "RESULT.json"
    value = m.read(marker)
    for path in changed:
        value["files"][path.name] = m.sha(path)
    save(marker, value)
    changed.append(marker)
    auth = m.read(op.authorization)
    for path in changed:
        auth["retired_files"][str(path.relative_to(op.root))] = m.sha(path)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    with pytest.raises(PermissionError, match="Historical command unknown or changed"):
        op.retire()
    assert not any(a[0] == "scancel" for a in calls)


@pytest.mark.parametrize(
    "features",
    [
        "",
        "gpu_ok",
        "cpu_ok|gpu_ok",
        "cpu_ok&gpu_ok",
        "cpu_ok&cpu_ok&cpu_ok",
        "(cpu_ok)",
        "cpu_ok,cpu_ok",
    ],
)
def test_held_constraint_rejects_every_non_authorized_expression(case, features):
    op, calls, faults, _ = case
    op.retire()
    op.submit()
    faults["features"] = features
    with pytest.raises(PermissionError, match="constraint differs"):
        op.continue_job()
    assert not any(c[:2] == ["scontrol", "release"] for c in calls)


@pytest.mark.parametrize(
    "change", ["unknown", "wrong_rc", "wrong_reason", "wrong_stop", "result_added", "extra_query"]
)
def test_only_exact_known_normalization_refusal_can_retire(case, change):
    op, calls, _, _ = case
    audit = op.root / h.OLD_WORK / h.NORMALIZATION_AUDIT
    path = audit / "004_normalize.result.json"
    value = m.read(path)
    if change == "unknown":
        value = dict(status="UNKNOWN", error="TimeoutExpired", at=value["at"])
    elif change == "wrong_rc":
        value["returncode"] = 0
    elif change == "wrong_reason":
        value["stderr"] = "permission denied"
    elif change == "wrong_stop":
        path = audit / "STOP.json"
        value = dict(status="NORMALIZATION_STOPPED", error="PermissionError('other')", at=30)
    elif change == "result_added":
        path = audit / "RESULT.json"
        value = dict(status="NORMALIZED")
    else:
        path = audit / "005_release.intent.json"
        value = dict(argv=["scontrol", "release", "216983"], at=31)
    save(path, value)
    auth = m.read(op.authorization)
    auth["retired_files"][str(path.relative_to(op.root))] = m.sha(path)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    with pytest.raises(PermissionError):
        op.retire()
    assert not any(call[0] == "scancel" for call in calls)


def test_failed_normalization_source_text_must_be_preserved(case):
    op, calls, _, _ = case
    path = op.root / h.OLD_WORK / h.NORMALIZATION_AUDIT / "SOURCE.json"
    value = m.read(path)
    value["utf8"] = "source removed"
    save(path, value)
    auth = m.read(op.authorization)
    auth["retired_files"][str(path.relative_to(op.root))] = m.sha(path)
    save(op.authorization, auth)
    op = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    with pytest.raises(PermissionError, match="complete source/authority"):
        op.retire()
    assert calls == []
