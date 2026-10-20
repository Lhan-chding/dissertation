"""Exercise the exact ab7 capacity interruption with unchanged original submission code."""

import importlib.util
import json
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "scripts/sr_f3_ce/continue_held_delivery.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(SOURCE, "held_capacity_test")
prior_tests = load(HERE / "test_resume_held_delivery.py", "held_capacity_prior_fixture")


@pytest.fixture
def case(tmp_path, monkeypatch):
    case = prior_tests.case.__wrapped__(tmp_path, monkeypatch)
    root, faults, base_runner = case.root, case.faults, case.runner
    old_op = root / "operations" / m.HELD_COMMIT
    old_op.mkdir()
    prior_source = old_op / "resume_held_delivery.py"
    body = prior_tests.SOURCE.read_text().replace(prior_tests.m.TARGET_ROOT, str(root))
    # The source on disk contains production literals; fixture pins its own exact root/auth.
    body = body.replace(
        "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2", str(root)
    )
    body = body.replace(
        "9bd7d76cb4b81dbd4df3dab1de5d70e3d2b0a04886f15f5fece142b102cf5e57",
        prior_tests.m.OLD_AUTH_SHA,
    )
    prior_source.write_text(body)
    old_auth = old_op / "HELD_RESUME_AUTHORIZATION.json"
    prior_value = dict(
        case.value, resume_operation_commit=m.HELD_COMMIT, source_sha256=m.sha(prior_source)
    )
    old_auth.write_text(json.dumps(prior_value))
    faults["capacity_full"] = True

    def runner(argv, **kwargs):
        if argv[:2] == ["scontrol", "release"] and faults.get("release_timeout"):
            case.calls.append(argv)
            raise subprocess.TimeoutExpired(argv, 30)
        result = base_runner(argv, **kwargs)
        if argv[:3] == ["scontrol", "show", "node"]:
            if faults.get("capacity_full"):
                result.stdout = result.stdout.replace("CPUAlloc=42", "CPUAlloc=96")
            if faults.get("completing"):
                result.stdout = result.stdout.replace("State=MIXED", "State=MIXED+COMPLETING")
        if argv[0] == "sacctmgr" and "qos" in argv and faults.get("badpolicy"):
            result.stdout = result.stdout.replace("rose||5||2||", "rose||6||2||")
        if argv[:3] == ["scontrol", "show", "node"] and faults.get("badfeatures"):
            result.stdout = result.stdout.replace("ActiveFeatures=cpu_ok", "ActiveFeatures=unknown")
        if argv[:2] == ["scontrol", "release"] and faults.get("assert_audit"):
            assert not (case.work / "RELEASE.json").exists()
        return result

    prior = load(prior_source, "held_capacity_prior_live")
    with pytest.raises(PermissionError, match="Insufficient CPUs"):
        prior.Resume(root, old_auth, runner=runner).resume()
    assert (case.work / "ALLOCATION_VERIFIED.json").exists()
    files = set(prior_value["files"])
    files.update(
        str(p.relative_to(root))
        for p in (old_auth, prior_source, case.work / "ALLOCATION_VERIFIED.json")
    )
    files.update(
        str(p.relative_to(root)) for p in (case.work / "commands" / m.INVOCATION).iterdir()
    )
    files.update(str(p.relative_to(root)) for p in (root / m.AUDIT).rglob("*") if p.is_file())
    new_op = root / "operations" / ("b" * 40)
    new_op.mkdir()
    source = new_op / SOURCE.name
    source.write_bytes(SOURCE.read_bytes())
    monkeypatch.setattr(m, "__file__", str(source))
    monkeypatch.setattr(m, "TARGET_ROOT", str(root))
    value = dict(
        schema_version=1,
        status="AUTHORIZED_HELD_CAPACITY_CONTINUATION",
        scope="EXACT_216142_AB7_CAPACITY_STOP_ONLY",
        root=str(root),
        job_id=m.JOB,
        original_operation_commit=m.OLD_COMMIT,
        held_operation_commit=m.HELD_COMMIT,
        operation_commit="b" * 40,
        recovery_directory=m.WORK,
        journal_directory=m.JOURNAL,
        resumed_invocation=m.NEW_INVOCATION,
        source_sha256=m.sha(source),
        prior_authorization=dict(path=str(old_auth.relative_to(root)), sha256=m.sha(old_auth)),
        files={n: m.sha(root / n) for n in files},
    )
    auth = new_op / "HELD_CAPACITY_AUTHORIZATION.json"
    auth.write_text(json.dumps(value))
    faults["capacity_full"] = False
    case.calls.clear()
    case.op = lambda: m.Continuation(root, auth, runner=runner)
    case.auth, case.value, case.source = auth, value, source
    return case


def test_wait_twice_then_available_single_same_job_release(case):
    original = {n: (case.root / n).read_bytes() for n in case.value["files"]}
    case.faults["capacity_full"] = True
    for _ in range(2):
        assert case.op().run()["status"] == "WAIT_CAPACITY"
        assert not (case.root / m.JOURNAL / "CONTINUE_INTENT.json").exists()
        assert not (case.work / "CAPACITY_BEFORE_RELEASE.json").exists()
        assert not (case.work / "RELEASE_INTENT.json").exists()
    case.faults["capacity_full"] = False
    assert case.op().run()["status"] == "CONTINUED"
    assert all((case.root / n).read_bytes() == body for n, body in original.items())
    assert sum(c[:2] == ["scontrol", "release"] for c in case.calls) == 1
    assert not any(c[0] in ("sbatch", "scancel", "srun") or "requeue" in c for c in case.calls)
    release = m.read(case.work / "RELEASE.json")
    assert len(release["command_files"]) == 32
    count = len(case.calls)
    assert case.op().run()["status"] == "CONTINUED"
    assert len(case.calls) == count


def test_completing_state_waits_without_relaxing_old_capacity(case):
    case.faults["completing"] = True
    result = case.op().run()
    assert result["status"] == "WAIT_CAPACITY" and result["reason"] == "Node unavailable"
    case.faults["completing"] = False
    assert case.op().run()["status"] == "CONTINUED"


@pytest.mark.parametrize("fault", ["started", "wrongjob", "script", "empty", "release_timeout"])
def test_unknown_or_unsafe_attempt_cannot_retry(case, fault):
    case.faults[fault] = True
    with pytest.raises((PermissionError, subprocess.TimeoutExpired)):
        case.op().run()
    count = len(case.calls)
    case.faults[fault] = False
    with pytest.raises((PermissionError, FileNotFoundError)):
        case.op().run()
    assert len(case.calls) == count
    assert not (case.work / "RELEASE.json").exists()
    assert sum(c[:2] == ["scontrol", "release"] for c in case.calls) <= 1


@pytest.mark.parametrize(
    "suffix",
    ["ALLOCATION_VERIFIED.json", "012_node.result.json", "run.sbatch", "PRECHECK_VERIFIED.json"],
)
def test_pinned_partial_mutation_refused(case, suffix):
    name = next(n for n in case.value["files"] if n.endswith(suffix))
    path = case.root / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PermissionError, match="Pinned partial"):
        case.op().run()
    assert case.calls == []


def test_wrong_job_authorization_refused(case):
    case.value["job_id"] = "215520"
    case.auth.write_text(json.dumps(case.value))
    with pytest.raises(PermissionError, match="scope"):
        case.op().run()


def test_old_capacity_cannot_be_reclassified_ready(case):
    name = next(n for n in case.value["files"] if n.endswith("012_node.result.json"))
    path = case.root / name
    value = m.read(path)
    value["stdout"] = value["stdout"].replace("CPUAlloc=96", "CPUAlloc=42")
    path.write_text(json.dumps(value))
    case.value["files"][name] = m.sha(path)
    case.auth.write_text(json.dumps(case.value))
    with pytest.raises(PermissionError, match="authorized capacity stop"):
        case.op().run()


def test_publication_waits_for_complete_audit(case, monkeypatch):
    original = m.save
    observed = []

    def checked(path, value):
        if Path(path) == case.work / "RELEASE.json":
            proof = m.read(case.root / m.JOURNAL / "CONTINUED.json")
            assert proof["release_sha256"] == m.hashlib.sha256(m.encoded(value)).hexdigest()
            observed.append(True)
        return original(path, value)

    monkeypatch.setattr(m, "save", checked)
    case.op().run()
    assert observed == [True]


def prepare_final(case, missing=None, corrupt=None):
    case.op().run()
    (case.work / "STARTED.json").write_text(
        json.dumps({"at": datetime.fromtimestamp(time.time() + 1, timezone.utc).isoformat()})
    )
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
    info = dict(path=str(archive), sha256=m.sha(archive), bytes=archive.stat().st_size)
    (delivery / "FINAL_RECOVERY_VERIFIED.json").write_text(
        json.dumps(dict(job_id=m.JOB, status="FINAL_RECOVERY_VERIFIED", archive=info))
    )
    (delivery / "FINAL_MIGRATION_VERIFIED.json").write_text(
        json.dumps(dict(recovery_job_id=m.JOB, status="FINAL_MIGRATION_VERIFIED", archive=info))
    )
    Path(str(archive) + ".verification.json").write_text('{"status":"PASS"}')


def test_final_gate_all_prior_partial_and_new_complete_archive(case):
    prepare_final(case)
    result = case.op().verify_final()
    assert result["status"] == "FINAL_HELD_CAPACITY_VERIFIED"
    assert len(result["acceptance_receipts"]) == 3
    assert any(m.INVOCATION + "/012_node.result.json" in n for n in result["archived_files"])
    assert case.op().verify_final() == result
    assert not (case.root / "delivery/FINAL_HELD_RECOVERY_VERIFIED.json").exists()


@pytest.mark.parametrize(
    "missing,corrupt",
    [
        ("CONTINUED.json", None),
        ("012_node.result.json", None),
        (None, "COPY_LINEAGE.json"),
        (None, "continue_held_delivery.py"),
        ("HELD_RESUME_AUTHORIZATION.json", None),
    ],
)
def test_final_archive_missing_or_changed_fails(case, missing, corrupt):
    prepare_final(case, missing, corrupt)
    with pytest.raises(PermissionError, match=r"archive|Archived"):
        case.op().verify_final()


def test_tampered_wait_poll_cannot_be_reused(case):
    case.faults["capacity_full"] = True
    case.op().run()
    result = next((case.root / m.JOURNAL).rglob("RESULT.json"))
    value = m.read(result)
    value["reason"] = "arbitrary failure"
    result.write_text(json.dumps(value))
    case.faults["capacity_full"] = False
    count = len(case.calls)
    with pytest.raises(PermissionError, match="Unrecognized"):
        case.op().run()
    assert len(case.calls) == count


def test_unknown_query_error_not_capacity_wait(case):
    case.faults["timeout"] = "sacctmgr"
    with pytest.raises(subprocess.TimeoutExpired):
        case.op().run()
    case.faults.pop("timeout")
    with pytest.raises(FileNotFoundError):
        case.op().run()
    assert not (case.work / "RELEASE_INTENT.json").exists()


@pytest.mark.parametrize("fault", ["badpolicy", "badfeatures"])
def test_unexpected_policy_is_not_retryable_capacity_wait(case, fault):
    case.faults[fault] = True
    with pytest.raises(PermissionError):
        case.op().run()
    case.faults[fault] = False
    count = len(case.calls)
    with pytest.raises(FileNotFoundError):
        case.op().run()
    assert len(case.calls) == count
    assert not (case.work / "RELEASE_INTENT.json").exists()


def test_capacity_stale_after_ready_never_releases_or_retries(case, monkeypatch):
    real_time = m.time.time

    def jumped_time():
        return real_time() + (120 if (case.work / "CAPACITY_BEFORE_RELEASE.json").exists() else 0)

    monkeypatch.setattr(m.time, "time", jumped_time)
    with pytest.raises(PermissionError, match="stale"):
        case.op().run()
    assert not any(c[:2] == ["scontrol", "release"] for c in case.calls)
    count = len(case.calls)
    with pytest.raises(PermissionError, match="advanced"):
        case.op().run()
    assert len(case.calls) == count


def test_final_requires_original_recovery_exact_job(case):
    prepare_final(case)
    path = case.root / "delivery/FINAL_RECOVERY_VERIFIED.json"
    value = m.read(path)
    value["job_id"] = "216128"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="Original acceptance"):
        case.op().verify_final()
