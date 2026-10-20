"""Real submission-chain fixtures stop at empty accounting, then resume only release."""

import importlib.util
import json
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/resume_held_delivery.py"
SUBMIT = SOURCE.with_name("submit_delivery_recovery.py")
spec = importlib.util.spec_from_file_location("held_resume_test", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "TARGET_ROOT", str(tmp_path))
    old = tmp_path / "operations" / m.OLD_COMMIT
    old.mkdir(parents=True)
    (old / "submit_delivery_recovery.py").write_bytes(SUBMIT.read_bytes())
    (old / "recover_delivery.py").write_text("""import json
from pathlib import Path
class Recovery:
    def __init__(self, root, authorization, **kwargs): self.root = Path(root)
    def validate(self): pass
    def verify_final(self):
        return json.loads((self.root / "delivery/FINAL_RECOVERY_VERIFIED.json").read_text())
""")
    (old / "verify_recovered_migration.py").write_text("""import json
from pathlib import Path
def verify(root, authorization, **kwargs):
    return json.loads((Path(root) / "delivery/FINAL_MIGRATION_VERIFIED.json").read_text())
""")
    work = tmp_path / m.WORK
    work.mkdir(parents=True)
    script = work / "run.sbatch"
    script.write_text("#!/bin/bash\ntrue\n")
    old_auth = old / "RECOVERY_AUTHORIZATION.json"
    old_value = dict(
        root=str(tmp_path),
        recovery_directory=m.WORK,
        source_sha256=m.sha(old / "recover_delivery.py"),
        extra_sources={str((old / n).relative_to(tmp_path)): m.sha(old / n) for n in m.SOURCES[1:]},
        script=str(script.relative_to(tmp_path)),
        script_sha256=m.sha(script),
        node="cpu-1",
        qos="rose",
        account="rose",
        cpus=22,
        memory_mib=67584,
        time_limit="3-00:00:00",
        user="testuser",
        job_name="delivery-LH",
        comment="exact",
    )
    old_auth.write_text(json.dumps(old_value))
    monkeypatch.setattr(m, "OLD_AUTH_SHA", m.sha(old_auth))
    calls, faults = [], {"empty": True}

    def runner(argv, **kwargs):
        calls.append(argv)
        if faults.get("timeout") == argv[0]:
            raise subprocess.TimeoutExpired(argv, 30)
        if argv[0] == "sacctmgr" and "qos" in argv:
            text = "Name|GrpSubmit|MaxSubmitPU|GrpJobs|MaxJobsPU|Flags|\nrose||5||2||\n"
        elif argv[0] == "sacctmgr":
            text = (
                "Cluster|Account|User|Partition|QOS|GrpSubmit|MaxSubmit|GrpJobs|MaxJobs|\n"
                "x|rose|testuser||rose|||||\n"
            )
        elif argv[0] == "squeue":
            text = (
                "216142|testuser|rose|rose|cluster02|PENDING\n"
                if any(c[0] == "sbatch" for c in calls) or faults.get("submitted")
                else ""
            )
        elif argv[:3] == ["scontrol", "show", "node"]:
            text = (
                "NodeName=cpu-1 Partitions=cluster02 State=MIXED ActiveFeatures=cpu_ok "
                "CPUEfctv=96 CPUAlloc=42 RealMemory=339968 AllocMem=129024 MemSpecLimit=5120"
            )
        elif argv[0] == "sbatch":
            text = "216142\n"
        elif argv[:3] == ["scontrol", "show", "job"]:
            text = (
                "JobId=216142 JobName=delivery-LH Account=rose QOS=rose Partition=cluster02 "
                f"Command={script} Comment=exact NumCPUs=22 NumTasks=1 CPUs/Task=22 "
                "ReqNodeList=cpu-1 TimeLimit=3-00:00:00 Requeue=0 JobState=PENDING "
                "Reason=JobHeldUser RunTime=00:00:00 UserId=testuser(123) NumNodes=1 "
                "NodeList=(null) ReqTRES=cpu=22,mem=66G,node=1"
            )
            if faults.get("started"):
                text = text.replace("JobState=PENDING", "JobState=RUNNING")
            if faults.get("wrongjob"):
                text = text.replace("JobId=216142", "JobId=216143")
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            text = script.read_text() + ("changed" if faults.get("script") else "")
        elif argv[0] == "sacct":
            text = "" if faults.get("empty") else "216142|PENDING|Unknown|0|\n"
        elif argv[:2] == ["scontrol", "release"]:
            if faults.get("release_error"):
                return SimpleNamespace(returncode=1, stdout="", stderr="denied")
            text = ""
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    submit_m = m.load_bound(
        old / "submit_delivery_recovery.py",
        m.sha(old / "submit_delivery_recovery.py"),
        "held_resume_fixture_submission",
    )
    op = submit_m.Submission(tmp_path, old_auth, runner=runner)
    with pytest.raises(PermissionError, match="Accounting"):
        op.submit()
    command_dirs = list((work / "commands").iterdir())
    command_dirs[0].rename(tmp_path / m.OLD_COMMANDS)
    faults["empty"] = False
    faults["submitted"] = True
    new = tmp_path / "operations" / ("a" * 40)
    new.mkdir()
    source = new / SOURCE.name
    source.write_bytes(SOURCE.read_bytes())
    monkeypatch.setattr(m, "__file__", str(source))
    files = [
        old_auth,
        *(old / n for n in m.SOURCES),
        *(work / n for n in m.ORIGINALS),
        *(tmp_path / m.OLD_COMMANDS).iterdir(),
    ]
    auth = dict(
        schema_version=1,
        status="AUTHORIZED_HELD_DELIVERY_RESUME",
        scope="EXACT_216142_ACCOUNTING_LAG_ONLY",
        root=str(tmp_path),
        job_id=m.JOB,
        operation_commit=m.OLD_COMMIT,
        recovery_directory=m.WORK,
        original_command_directory=m.OLD_COMMANDS,
        audit_directory=m.AUDIT,
        resumed_invocation=m.INVOCATION,
        resume_operation_commit="a" * 40,
        source_sha256=m.sha(source),
        original_authorization={
            "path": str(old_auth.relative_to(tmp_path)),
            "sha256": m.sha(old_auth),
        },
        files={str(p.relative_to(tmp_path)): m.sha(p) for p in files},
    )
    authorization = new / "HELD_RESUME_AUTHORIZATION.json"
    authorization.write_text(json.dumps(auth))
    calls.clear()
    return SimpleNamespace(
        root=tmp_path,
        work=work,
        auth=authorization,
        value=auth,
        faults=faults,
        calls=calls,
        source=source,
        runner=runner,
        op=lambda: m.Resume(tmp_path, authorization, runner=runner),
    )


def test_exact_empty_lag_resume_keeps_original_and_single_release(case):
    original = {p: p.read_bytes() for p in (case.root / m.OLD_COMMANDS).iterdir()}
    result = case.op().resume()
    assert result["status"] == "RESUMED"
    assert all(p.read_bytes() == b for p, b in original.items())
    assert sum(c[:2] == ["scontrol", "release"] for c in case.calls) == 1
    assert not any(c[0] in ("sbatch", "scancel", "srun") or "requeue" in c for c in case.calls)
    count = len(case.calls)
    assert case.op().resume() == result
    assert len(case.calls) == count
    release = m.read(case.work / "RELEASE.json")
    assert len(release["command_files"]) == 32
    assert m.sha(case.work / "RELEASE.json") == result["release_sha256"]


@pytest.mark.parametrize("fault", ["empty", "started", "wrongjob", "script", "release_error"])
def test_current_unsafe_or_unknown_never_retry(case, fault):
    case.faults[fault] = True
    with pytest.raises(PermissionError):
        case.op().resume()
    assert not (case.work / "RELEASE.json").exists()
    count = len(case.calls)
    with pytest.raises(PermissionError):
        case.op().resume()
    assert len(case.calls) == count
    assert not any(c[0] in ("sbatch", "scancel") for c in case.calls)


@pytest.mark.parametrize("name", ["SUBMITTED.json", "MAY_HAVE_SUBMITTED.json", "run.sbatch"])
def test_changed_original_rejected(case, name):
    with (case.work / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(PermissionError, match="Pinned evidence"):
        case.op().resume()
    assert case.calls == []


def test_changed_original_command_even_rehashed_rejected(case):
    path = case.root / m.OLD_COMMANDS / "005_sbatch.result.json"
    record = m.read(path)
    record["stdout"] = "216143\n"
    path.write_text(json.dumps(record))
    case.value["files"][str(path.relative_to(case.root))] = m.sha(path)
    case.auth.write_text(json.dumps(case.value))
    with pytest.raises(PermissionError, match="sbatch job"):
        case.op().resume()


def test_source_binding(case):
    case.source.write_text(case.source.read_text() + "\n# changed\n")
    with pytest.raises(PermissionError, match="source changed"):
        case.op().resume()


@pytest.mark.parametrize(
    "field,value",
    [("job_id", "216143"), ("operation_commit", "b" * 40), ("resumed_invocation", "../elsewhere")],
)
def test_scope_binding(case, field, value):
    case.value[field] = value
    case.auth.write_text(json.dumps(case.value))
    with pytest.raises(PermissionError):
        case.op().resume()
    assert not case.calls


def prepare_final(case, *, missing=None, corrupt=None):
    case.op().resume()
    delivery = case.root / "delivery"
    delivery.mkdir()
    archive = delivery / "SR_F3_CE_20261016.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in sorted(case.root.rglob("*")):
            if path.is_file() and "delivery" not in path.relative_to(case.root).parts:
                name = "evidence/" + str(path.relative_to(case.root))
                if missing and name.endswith(missing):
                    continue
                z.writestr(
                    name, b"changed" if corrupt and name.endswith(corrupt) else path.read_bytes()
                )
    record = dict(path=str(archive), sha256=m.sha(archive), bytes=archive.stat().st_size)
    (delivery / "FINAL_RECOVERY_VERIFIED.json").write_text(
        json.dumps(dict(job_id=m.JOB, status="FINAL_RECOVERY_VERIFIED", archive=record))
    )
    (delivery / "FINAL_MIGRATION_VERIFIED.json").write_text(
        json.dumps(dict(recovery_job_id=m.JOB, status="FINAL_MIGRATION_VERIFIED", archive=record))
    )
    Path(str(archive) + ".verification.json").write_text('{"status":"PASS"}')


def test_final_gate_archives_resume_and_retains_original_acceptance(case):
    prepare_final(case)
    final = case.op().verify_final()
    assert final["status"] == "FINAL_HELD_RECOVERY_VERIFIED"
    assert len(final["acceptance_receipts"]) == 3
    assert any(n.endswith("008_held_accounting.result.json") for n in final["archived_files"])
    assert case.op().verify_final() == final


@pytest.mark.parametrize(
    "missing,corrupt",
    [
        ("RESUMED.json", None),
        ("008_held_accounting.result.json", None),
        (None, "COPY_LINEAGE.json"),
        (None, "resume_held_delivery.py"),
    ],
)
def test_archive_missing_or_changed_refuses_final(case, missing, corrupt):
    prepare_final(case, missing=missing, corrupt=corrupt)
    with pytest.raises(PermissionError, match=r"archive|Archived"):
        case.op().verify_final()
    assert not (case.root / "delivery/FINAL_HELD_RECOVERY_VERIFIED.json").exists()


def test_copied_chain_changed_refuses_final(case):
    prepare_final(case)
    path = case.work / "commands" / m.INVOCATION / "005_sbatch.result.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(PermissionError, match="changed"):
        case.op().verify_final()


def test_real_iso_started_time_enforces_audit_before_worker(case):
    from datetime import datetime, timezone

    case.op().resume()
    (case.work / "STARTED.json").write_text(
        json.dumps({"at": datetime.fromtimestamp(time.time() + 1, timezone.utc).isoformat()})
    )
    case.op().verify_audit()
    (case.work / "STARTED.json").write_text(json.dumps({"at": "2000-01-01T00:00:00+00:00"}))
    with pytest.raises(PermissionError, match="after worker"):
        case.op().verify_audit()


def test_audit_is_final_before_release_gate_publication(case, monkeypatch):
    original = m.save
    observed = []

    def checked(path, value):
        if Path(path) == case.work / "RELEASE.json":
            assert (case.root / m.AUDIT / "RESUMED.json").exists()
            proof = m.read(case.root / m.AUDIT / "RESUMED.json")
            assert proof["release_sha256"] == m.hashlib.sha256(m.encoded(value)).hexdigest()
            observed.append(True)
        return original(path, value)

    monkeypatch.setattr(m, "save", checked)
    case.op().resume()
    assert observed == [True]


def test_nonempty_original_accounting_not_authorized_even_with_rehash(case):
    path = case.root / m.OLD_COMMANDS / "008_held_accounting.result.json"
    value = m.read(path)
    value["stdout"] = "216142|PENDING|Unknown|0|\n"
    path.write_text(json.dumps(value))
    case.value["files"][str(path.relative_to(case.root))] = m.sha(path)
    case.auth.write_text(json.dumps(case.value))
    with pytest.raises(PermissionError, match="empty accounting"):
        case.op().resume()


def test_existing_final_tamper_refuses_idempotent_success(case):
    prepare_final(case)
    case.op().verify_final()
    path = case.root / "delivery/FINAL_HELD_RECOVERY_VERIFIED.json"
    value = m.read(path)
    value["job_id"] = "other"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="Existing held-resume"):
        case.op().verify_final()


def test_final_requires_prior_gate_success_and_same_job(case):
    prepare_final(case)
    path = case.root / "delivery/FINAL_RECOVERY_VERIFIED.json"
    value = m.read(path)
    value["job_id"] = "216143"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="Original recovery"):
        case.op().verify_final()


def test_user_request_hash_if_supplied_must_match(case):
    case.value.update(user_request="Authorized exact job resume", user_request_sha256="0" * 64)
    case.auth.write_text(json.dumps(case.value))
    with pytest.raises(PermissionError, match="User request"):
        case.op().resume()
