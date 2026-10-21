"""The queued recovery releases once to Slurm, not only after a node becomes idle."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce"


def module(name):
    path = BASE / "queued_delivery" / name
    spec = importlib.util.spec_from_file_location("queued_" + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module("submit_delivery_recovery.py")
h = module("lowmem_history.py")
w = module("recover_delivery.py")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    deploy = tmp_path / "operations/new"
    deploy.mkdir(parents=True)
    for name in ("submit_delivery_recovery.py", "lowmem_history.py"):
        (deploy / name).write_bytes((BASE / "queued_delivery" / name).read_bytes())
    monkeypatch.setattr(m, "__file__", str(deploy / "submit_delivery_recovery.py"))
    work = tmp_path / "operations/delivery_recovery/20261021_delivery_queued"
    work.mkdir(parents=True)
    script = work / "run.sbatch"
    script.write_text("#!/bin/bash\n# isolated frozen packaging fixture\n")
    auth = dict(
        root=str(tmp_path),
        extra_sources={str(p.relative_to(tmp_path)): m.sha(p) for p in deploy.glob("*.py")},
        recovery_directory=str(work.relative_to(tmp_path)),
        script=str(script.relative_to(tmp_path)),
        script_sha256=m.sha(script),
        user="testuser",
        job_name="queued-LH",
        comment="queued",
        allowed_nodes=["cpu-1", "cpu-2"],
        qos=m.TEACHER_QOS,
        account="rose",
        cpus=4,
        memory_mib=12288,
        time_limit="12:00:00",
        queue_policy="SLURM_MANAGED_CPU_QUEUE",
    )
    (tmp_path / "operations/delivery_recovery/20261021_delivery_teacher").mkdir(parents=True)
    authorization = deploy / "AUTH.json"
    save(authorization, auth)
    calls, faults, state = [], {}, {"submitted": False}

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
            text = (
                f"99|testuser|rose|{m.TEACHER_QOS}|cluster02|PENDING\n"
                if state["submitted"]
                else ""
            ) + faults.get("extra_queue", "")
        elif argv[:3] == ["scontrol", "show", "node"]:
            text = "\n".join(
                f"NodeName={node} Partitions=cluster02 Arch={faults.get('arch', 'x86_64')} "
                f"State={faults.get('node2' if node == 'cpu-2' else 'node', 'ALLOCATED')} "
                f"ActiveFeatures={faults.get('feature', 'cpu_ok')} "
                f"CPUEfctv=96 CPUAlloc={faults.get('cpualloc', 96)} "
                f"RealMemory=339968 AllocMem={faults.get('memalloc', 327680)} MemSpecLimit=5120"
                for node in ("cpu-1", "cpu-2")
            )
            if faults.get("inventory"):
                text = text.splitlines()[0]
        elif argv[0] == "sbatch":
            state["submitted"] = True
            text = faults.get("sbatch_output", "99\n")
        elif argv[:3] == ["scontrol", "show", "job"]:
            text = (
                f"JobId=99 JobName=queued-LH Account=rose QOS={m.TEACHER_QOS} Partition=cluster02 "
                f"Command={script} Comment=queued NumCPUs=4 NumTasks=1 CPUs/Task=4 "
                f"ReqNodeList=(null) Features={faults.get('features', 'cpu_ok&cpu_ok')} "
                "TimeLimit=12:00:00 Requeue=0 JobState=PENDING Reason=JobHeldUser "
                "RunTime=00:00:00 UserId=testuser(123) NumNodes=1 NodeList=(null) "
                "ReqTRES=cpu=4,mem=12G,node=1"
            )
            if faults.get("started"):
                text = text.replace("JobState=PENDING", "JobState=RUNNING")
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            text = script.read_text()
        elif argv[0] == "sacct":
            text = "" if faults.get("accounting_lag") else "99|PENDING|Unknown|0|\n"
        elif argv[:2] == ["scontrol", "release"]:
            text = ""
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    op = m.Submission(tmp_path, authorization, runner=runner, validator=lambda *_: None)
    # Scheduling tests isolate already independently verified history below.
    op.history = lambda **kwargs: (None, [])
    return op, calls, faults, state


@pytest.mark.parametrize("state", ["ALLOCATED", "IDLE+COMPLETING+PLANNED", "DOWN+DRAIN"])
def test_busy_nodes_release_once_to_scheduler(case, state):
    op, calls, faults, _ = case
    faults["node"] = faults["node2"] = state
    op.submit()
    result = op.continue_job()
    assert result["status"] == "RELEASED"
    capacity = m.read(op.work / "CAPACITY_BEFORE_RELEASE.json")
    assert capacity["eligible_nodes"] == []
    assert capacity["configured_nodes"] == ["cpu-1", "cpu-2"]
    assert capacity["queue_policy"] == "SLURM_MANAGED_CPU_QUEUE"
    assert op.continue_job() == result
    assert len([c for c in calls if c[0] == "sbatch"]) == 1
    assert len([c for c in calls if c[:2] == ["scontrol", "release"]]) == 1
    assert not any(c[0] in ("scancel", "srun") or c[:2] == ["scontrol", "update"] for c in calls)
    assert not (op.work / "STARTED.json").exists()


@pytest.mark.parametrize(
    "fault,value",
    [("arch", "aarch64"), ("feature", "gpu_ok"), ("inventory", True), ("node", "???")],
)
def test_malformed_or_incompatible_nodes_do_not_submit(case, fault, value):
    op, calls, faults, _ = case
    faults[fault] = value
    with pytest.raises((PermissionError, ValueError)):
        op.submit()
    assert not any(c[0] == "sbatch" for c in calls)


def test_submission_quota_still_blocks(case):
    op, calls, faults, _ = case
    faults["extra_queue"] = "".join(
        f"{i}|testuser|rose|{m.TEACHER_QOS}|cluster02|PENDING\n" for i in range(5)
    )
    assert op.submit()["status"] == "WAIT_CAPACITY"
    assert not any(c[0] == "sbatch" for c in calls)


def test_queue_policy_explicitly_authorized(case):
    op, _, _, _ = case
    op.auth.pop("queue_policy")
    with pytest.raises(PermissionError, match="resources"):
        op.validate()


def test_accounting_lag_safe_repeat_without_duplicate(case):
    op, calls, faults, _ = case
    op.submit()
    faults["accounting_lag"] = True
    assert op.continue_job()["status"] == "WAIT_ACCOUNTING"
    faults.clear()
    assert op.continue_job()["status"] == "RELEASED"
    assert len([c for c in calls if c[0] == "sbatch"]) == 1


@pytest.mark.parametrize("fault", ["started", "release_timeout"])
def test_unsafe_outcomes_cannot_repeat_release(case, fault):
    op, calls, faults, _ = case
    op.submit()
    faults[fault] = True
    with pytest.raises((PermissionError, subprocess.TimeoutExpired)):
        op.continue_job()
    faults.clear()
    with pytest.raises(PermissionError):
        op.continue_job()
    assert len([c for c in calls if c[:2] == ["scontrol", "release"]]) <= 1


def test_unknown_submission_never_duplicates(case):
    op, calls, faults, _ = case
    faults["sbatch_output"] = "unexpected acceptance"
    with pytest.raises(PermissionError):
        op.submit()
    faults.clear()
    with pytest.raises(PermissionError):
        op.submit()
    assert len([c for c in calls if c[0] == "sbatch"]) == 1


def test_frozen_builder_and_final_worker_checks_unchanged():
    original = (BASE / "lowmem_replacement/recover_delivery.py").read_text()
    expected = original.replace("3-00:00:00", "12:00:00").replace(
        'and a.get("time_limit") == "12:00:00",',
        'and a.get("time_limit") == "12:00:00"\n'
        '            and a.get("queue_policy") == "SLURM_MANAGED_CPU_QUEUE",',
    )
    assert (BASE / "queued_delivery/recover_delivery.py").read_text() == expected
    assert (BASE / "queued_delivery/verify_recovered_migration.py").read_bytes() == (
        BASE / "lowmem_replacement/verify_recovered_migration.py"
    ).read_bytes()
    assert len(w.CHECKS) == 7 and w.COVERAGE["records"] == 233340


def test_final_gate_requires_all_archived_audit_and_same_memory_binding(case):
    op, _, _, _ = case
    source = op.authorization.parent / "verify_recovered_migration.py"
    source.write_text("""import json, hashlib
from pathlib import Path
def verify(root, authorization, *, runner):
    root = Path(root)
    auth = json.loads(Path(authorization).read_text())
    profile = root / auth["profile_receipt"]
    archive = {"sha256": "fixture-archive-sha"}
    for name in ("FINAL_RECOVERY_VERIFIED", "FINAL_MIGRATION_VERIFIED"):
        (root / "delivery" / (name + ".json")).write_text(json.dumps({
            "status": name, "archive": archive,
            "memory_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "delivery_revision": {"fixture_only": True}}))
    return {"archive": archive}
""")
    auth = m.read(op.authorization)
    profile = op.work / "MEMORY_PROFILE.json"
    save(profile, {"fixture_only": True})
    history = op.root / "operations/retired.json"
    save(history, {"job_id": "217029", "immutable": True})
    auth["profile_receipt"] = str(profile.relative_to(op.root))
    auth["extra_sources"][str(source.relative_to(op.root))] = m.sha(source)
    save(op.authorization, auth)
    new = m.Submission(op.root, op.authorization, runner=op.runner, validator=lambda *_: None)
    new.history = lambda **kwargs: (None, [history])
    new.submit()
    release = new.continue_job()
    delivery = new.root / "delivery"
    delivery.mkdir()
    archive = delivery / "SR_F3_CE_20261016.zip"
    paths = [history] + [new.work / n for n in release["teacher_audit_files"]]
    with zipfile.ZipFile(archive, "w") as package:
        for path in paths:
            package.write(path, "evidence/" + str(path.relative_to(new.root)))
    receipt = new.verify_final()
    assert receipt["status"] == "FINAL_QUEUED_RECOVERY_VERIFIED"
    assert receipt["retired_job_id"] == "217029"
    assert "FINAL_LOWMEM_REPLACEMENT_VERIFIED" in receipt["supersedes"]
    assert new.verify_final() == receipt
    with zipfile.ZipFile(archive, "w") as package:
        for path in paths[1:]:
            package.write(path, "evidence/" + str(path.relative_to(new.root)))
    with pytest.raises(PermissionError, match="history missing"):
        new.verify_final()


@pytest.fixture
def retired(tmp_path, monkeypatch):
    operation = tmp_path / "operations" / h.OLD_OPERATION
    operation.mkdir(parents=True)
    work = tmp_path / h.OLD_WORK
    work.mkdir(parents=True)
    script = work / "run.sbatch"
    script.write_text("#!/bin/bash\n# never started original\n")
    for name in ("submit_delivery_recovery.py", "recover_delivery.py"):
        (operation / name).write_bytes((BASE / "lowmem_replacement" / name).read_bytes())
    # Previous nested history is independently tested in its existing suite.
    (operation / "lowmem_history.py").write_text(
        "def verify_history(root, auth, *, require_retired=False):\n"
        "    assert require_retired\n    return []\n"
    )
    auth = dict(
        root=str(tmp_path),
        extra_sources={str(p.relative_to(tmp_path)): h.sha(p) for p in operation.glob("*.py")},
        recovery_directory=h.OLD_WORK,
        script=str(script.relative_to(tmp_path)),
        script_sha256=h.sha(script),
        source_sha256=h.sha(operation / "recover_delivery.py"),
        user="testuser",
        job_name="old-LH",
        comment="old",
        allowed_nodes=["cpu-1", "cpu-2"],
        qos=m.TEACHER_QOS,
        account="rose",
        cpus=4,
        memory_mib=12288,
        time_limit="3-00:00:00",
        retired_files={},
        files={},
        retired_directories=[],
    )
    authority = operation / "RECOVERY_AUTHORIZATION.json"
    save(authority, auth)
    old_sha = h.sha(authority)
    monkeypatch.setattr(h, "OLD_AUTH_SHA256", old_sha)
    old = h.load(operation / "submit_delivery_recovery.py", "original_queued_test")
    original = old.Submission(tmp_path, authority, validator=lambda *_: None)
    original.validate()
    held = (
        f"JobId=217029 JobName=old-LH Account=rose QOS={m.TEACHER_QOS} Partition=cluster02 "
        f"Command={script} Comment=old NumCPUs=4 NumTasks=1 CPUs/Task=4 "
        "ReqNodeList=(null) Features=cpu_ok&cpu_ok TimeLimit=3-00:00:00 Requeue=0 "
        "JobState=PENDING Reason=JobHeldUser RunTime=00:00:00 UserId=testuser(123) "
        "NumNodes=1 NodeList=(null) ReqTRES=cpu=4,mem=12G,node=1"
    )
    argv = dict(original.expected_commands("217029"))
    initial = work / "commands/poll_submit_1"
    for index, label in enumerate(("qos", "assoc", "queue", "node", "sbatch"), 1):
        save(initial / f"{index:03d}_{label}.intent.json", {"argv": argv[label], "at": 1})
        save(
            initial / f"{index:03d}_{label}.result.json",
            {
                "argv": argv[label],
                "at": 2,
                "returncode": 0,
                "stdout": "217029\n" if label == "sbatch" else "",
            },
        )
    save(
        initial / "RESULT.json",
        {
            "status": "READY",
            "files": {
                p.name: h.sha(p) for p in initial.glob("*.json") if not p.name.startswith("005_")
            },
        },
    )
    save(
        work / "SUBMITTED.json",
        {
            "job_id": "217029",
            "invocation": "poll_submit_1",
            "authorization_sha256": old_sha,
            "script_sha256": auth["script_sha256"],
        },
    )
    poll = work / "commands/poll_continue_2"
    for index, label in enumerate(
        ("held_job", "stored_script", "held_accounting", "qos", "assoc", "queue", "node"), 1
    ):
        save(poll / f"{index:03d}_{label}.intent.json", {"argv": argv[label], "at": 3})
        save(
            poll / f"{index:03d}_{label}.result.json",
            {
                "argv": argv[label],
                "at": 4,
                "returncode": 0,
                "stdout": {
                    "held_job": held,
                    "stored_script": script.read_text(),
                    "held_accounting": "217029|PENDING|None|0|\n",
                }.get(label, ""),
            },
        )
    save(
        poll / "RESULT.json",
        {"status": "WAIT_CAPACITY", "files": {p.name: h.sha(p) for p in poll.glob("*.json")}},
    )
    audit = tmp_path / h.RETIREMENT
    save(
        audit / "AUTHORIZATION.json",
        {
            "user_instruction": (
                "Cancel original 217029 and complete full package by another legitimate method; "
                "no experimental rerun"
            ),
            "job_id": "217029",
            "prior_authorization_sha256": old_sha,
            "at": 10,
        },
    )
    fmt = "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES,ExitCode"
    commands = [
        ("001_held", argv["held_job"], held),
        ("002_script", argv["stored_script"], script.read_text()),
        (
            "003_accounting",
            ["sacct", "-n", "-P", "-j", "217029", fmt],
            "217029|PENDING|None|0||0:0\n",
        ),
        ("004_cancel", ["scancel", "--state=PENDING", "217029"], ""),
        ("005_queue", ["squeue", "-h", "-j", "217029", "-o", "%i|%T|%R"], ""),
        (
            "006_terminal",
            ["sacct", "-n", "-P", "-j", "217029", fmt],
            "217029|CANCELLED by 123|None|0||0:0\n",
        ),
    ]
    for index, (label, command, stdout) in enumerate(commands):
        save(audit / (label + ".intent.json"), {"argv": command, "at": 11 + index * 2})
        save(
            audit / (label + ".result.json"),
            {"argv": command, "at": 12 + index * 2, "stdout": stdout, "returncode": 0},
        )
    save(
        audit / "PRECHECK.json",
        {
            "status": "UNSTARTED_ZERO_ALLOCATION",
            "at": 16.5,
            "original_files": h.inventory(tmp_path, h.OLD_WORK),
        },
    )
    save(
        audit / "RETIREMENT.json",
        {
            "status": "CANCELLED_UNSTARTED_RECOVERY",
            "job_id": "217029",
            "prior_authorization_sha256": old_sha,
            "at": 23,
            "terminal": ["217029", "CANCELLED by 123", "None", "0", "", "0:0"],
        },
    )
    new_auth = dict(
        retirement_directory=h.RETIREMENT,
        retired_directories=[h.OLD_WORK, h.RETIREMENT],
        retired_files={},
    )

    def rebind():
        new_auth["retired_files"] = {
            str(p.relative_to(tmp_path)): h.sha(p) for p in tmp_path.rglob("*") if p.is_file()
        }

    rebind()
    return tmp_path, new_auth, audit, rebind


def test_exact_retired_history_passes_and_pins_all_bytes(retired):
    root, auth, _, _ = retired
    paths = h.verify_history(root, auth, require_retired=True)
    assert {str(p.relative_to(root)) for p in paths} == set(auth["retired_files"])
    (root / h.OLD_WORK / "run.sbatch").write_text("changed")
    with pytest.raises(PermissionError, match="bytes changed"):
        h.verify_history(root, auth, require_retired=True)


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("006_terminal.result.json", "stdout", "217029|CANCELLED|None|1|cpu=4|0:0\n"),
        ("005_queue.result.json", "stdout", "217029|PENDING\n"),
        ("004_cancel.result.json", "returncode", 1),
        ("004_cancel.result.json", "argv", ["scancel", "217029"]),
        ("003_accounting.result.json", "stdout", "217029|RUNNING|None|0||0:0\n"),
        ("001_held.result.json", "at", 1000),
        ("RETIREMENT.json", "job_id", "216983"),
    ],
)
def test_retirement_semantics_reject_even_rehashed_changes(retired, name, field, value):
    root, auth, audit, rebind = retired
    path = audit / name
    body = h.read(path)
    body[field] = value
    save(path, body)
    rebind()
    with pytest.raises(PermissionError):
        h.verify_history(root, auth, require_retired=True)


@pytest.mark.parametrize("marker", ["STARTED", "RELEASE", "ALLOCATION_VERIFIED"])
def test_retired_job_cannot_advance(retired, marker):
    root, auth, _, rebind = retired
    save(root / h.OLD_WORK / (marker + ".json"), {})
    rebind()
    with pytest.raises(PermissionError, match="advanced"):
        h.verify_history(root, auth, require_retired=True)


def test_retired_missing_poll_result_not_repaired(retired):
    root, auth, _, rebind = retired
    (root / h.OLD_WORK / "commands/poll_continue_2/RESULT.json").unlink()
    rebind()
    with pytest.raises(FileNotFoundError):
        h.verify_history(root, auth, require_retired=True)
