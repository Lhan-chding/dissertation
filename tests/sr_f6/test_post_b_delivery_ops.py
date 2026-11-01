"""Real filesystem and rejected scheduler transitions; no SSH or live Slurm."""

import base64
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "post_b_ops_test", repo / "scripts/sr_f6_post_b_delivery/operations.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    project = tmp_path.resolve()
    ops = project / "operations/PILOT_POST_B_DELIVERY_20261101"
    ops.parent.mkdir()
    monkeypatch.setattr(m, "PROJECT", project)
    monkeypatch.setattr(m, "OPS", ops)
    monkeypatch.setattr(m, "CODE", project / "b7")
    revision = "a" * 40
    payload = m.make_package(revision, {name: (repo / name).read_bytes() for name in m.MEMBERS})
    binding = m.digest(m.encode(payload["manifest"]))
    target = Path(payload["manifest"]["deployment"])
    calls, queued = [], []
    accounting = "237289|COMPLETED|0:0|\n"

    def query(argv):
        if argv[0] == "squeue":
            return "\n".join("|".join(row) for row in queued)
        if argv[0] == "sacct":
            return accounting
        if argv[0] == "scontrol":
            intent = m.read(ops / "SUBMIT_INTENT.json")
            return (
                f"JobId=900 JobName={m.JOB_NAME} Account=rose Partition=cluster02 "
                f"QOS={m.QOS} Requeue=0 TimeLimit=02:00:00 Comment={intent['comment']} "
                "UserId=varun024(1000) ReqTRES=cpu=4,mem=12G,node=1"
            )
        raise AssertionError(argv)

    def sbatch(argv, **kwargs):
        assert (ops / "SUBMIT_INTENT.json").exists()
        assert argv[0] == "sbatch"
        assert set(kwargs["env"]) == {"PATH", "LANG", "TZ"}
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="900\n", stderr="")

    monkeypatch.setattr(m, "query", query)
    monkeypatch.setattr(m, "check_evidence", lambda actual: {"full_delivery_verified": False})
    monkeypatch.setattr(m.getpass, "getuser", lambda: m.USER)
    monkeypatch.setattr(m.subprocess, "run", sbatch)

    def deployed():
        m.deploy(payload, revision, binding)
        monkeypatch.setattr(m, "__file__", str(target / m.SCRIPT))
        return m.identity(revision, binding)

    def frozen():
        actual = deployed()
        result = m.freeze(actual)
        return actual, result["contract_ref"]["sha256"]

    return SimpleNamespace(**locals())


def test_deploy_fixed_four_files_and_manifest_only(fixture):
    f = fixture
    actual = f.deployed()
    assert len(f.m.MEMBERS) == 4
    assert f.m.PREFIX + "recovery_readonly.py" in f.m.MEMBERS
    assert json.loads((f.m.OPS / "DEPLOYED.json").read_text())["files"] == 5
    assert f.m.identity(f.revision, f.binding) == actual
    assert {
        p.relative_to(f.target).as_posix() for p in f.target.rglob("*") if p.is_file()
    } == f.m.MEMBERS | {f.m.MANIFEST}
    with pytest.raises(PermissionError, match="Prior deployment"):
        f.m.deploy(f.payload, f.revision, f.binding)


@pytest.mark.parametrize("change", ["revision", "member", "payload", "hash"])
def test_untrusted_package_rejected_before_any_publication(fixture, change):
    f = fixture
    package = json.loads(json.dumps(f.payload))
    revision, binding = f.revision, f.binding
    if change == "revision":
        revision = "main"
    elif change == "member":
        package["contents"]["../outside"] = base64.b64encode(b"x").decode()
    elif change == "payload":
        package["contents"][f.m.DRIVER] = base64.b64encode(b'print("tampered")').decode()
    else:
        binding = "0" * 64
    with pytest.raises(PermissionError):
        f.m.deploy(package, revision, binding)
    assert not f.ops.exists() and not f.target.exists()


def test_source_symlink_and_tampering_refused(fixture, monkeypatch):
    f = fixture
    f.deployed()
    driver = f.target / f.m.DRIVER
    driver.chmod(0o600)
    driver.write_text("changed")
    with pytest.raises(PermissionError, match="source changed"):
        f.m.identity(f.revision, f.binding)
    driver.unlink()
    driver.symlink_to(f.repo / f.m.DRIVER)
    with pytest.raises(PermissionError, match="Symlink"):
        f.m.identity(f.revision, f.binding)
    monkeypatch.setattr(f.m, "__file__", str(f.repo / f.m.SCRIPT))
    with pytest.raises(PermissionError, match="actual immutable"):
        f.m.identity(f.revision, f.binding)


def test_freeze_six_field_contract_and_rejects_refreeze(fixture):
    f = fixture
    actual, pin = f.frozen()
    value = f.m.read(f.ops / "CONTRACT.json")
    assert set(value) == {
        "schema",
        "output_directory",
        "driver_sha256",
        "source_deployment",
        "b_job_id",
        "round_b_complete_sha256",
    }
    assert f.m.metadata(actual, pin)["source"] == actual
    with pytest.raises(PermissionError, match="Prior/partial freeze"):
        f.m.freeze(actual)


def test_partial_freeze_cannot_continue_implicitly(fixture, monkeypatch):
    f = fixture
    actual = f.deployed()
    real_write = f.m.write

    def interrupted(path, value):
        if path.name == "SOURCE_BINDING.json":
            raise OSError("interrupted after contract")
        return real_write(path, value)

    monkeypatch.setattr(f.m, "write", interrupted)
    with pytest.raises(OSError):
        f.m.freeze(actual)
    assert (f.ops / "CONTRACT.json").exists()
    with pytest.raises(PermissionError, match="Prior/partial freeze"):
        f.m.freeze(actual)


@pytest.mark.parametrize(
    "accounting,queued",
    [
        ("237289|RUNNING|0:0|\n", []),
        ("237289|FAILED|1:0|\n", []),
        ("237289|COMPLETED|0:0|\n", [["237289", "B", "COMPLETING", "(null)"]]),
        ("", []),
    ],
)
def test_b_terminal_and_queue_gone_are_required(fixture, monkeypatch, accounting, queued):
    f = fixture
    actual = f.deployed()
    monkeypatch.setattr(
        f.m,
        "query",
        lambda args: (
            accounting if args[0] == "sacct" else "\n".join("|".join(row) for row in queued)
        ),
    )
    with pytest.raises(PermissionError):
        f.m.freeze(actual)
    assert not (f.ops / "CONTRACT.json").exists()


def test_real_five_slot_gate_does_not_leave_intent(fixture):
    f = fixture
    actual, pin = f.frozen()
    f.queued.extend([[str(n), "unrelated", "RUNNING", "(null)"] for n in range(101, 106)])
    result = f.m.submit(actual, pin)
    assert result["status"] == "NO_LEGAL_CAPACITY" and result["requested_gpus"] == 0
    assert not f.calls and not (f.ops / "SUBMIT_INTENT.json").exists()


def test_four_slots_allows_exact_cpu_job_once(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()
    monkeypatch.setenv("SBATCH_GRES", "gpu:pro6000:8")
    f.queued.extend([[str(n), "unrelated", "RUNNING", "(null)"] for n in range(101, 105)])
    result = f.m.submit(actual, pin)
    assert result["status"] == "SUBMITTED" and result["job_id"] == "900"
    assert result["request_check"]["status"] == "REQUEST_VERIFIED"
    argv = f.calls[0]
    assert "--cpus-per-task=4" in argv and "--mem=12G" in argv and "--time=02:00:00" in argv
    assert "--no-requeue" in argv and not any("gpu" in arg or "--gres=" in arg for arg in argv)
    with pytest.raises(PermissionError, match="Prior/partial/unknown"):
        f.m.submit(actual, pin)
    assert len(f.calls) == 1


@pytest.mark.parametrize(
    "artifact",
    [
        "SUBMIT_INTENT.json",
        "SUBMISSION.json",
        "RESULT",
        "cpu_cache_900",
        "audit-900.log",
        ".pending-submission",
    ],
)
def test_any_previous_or_partial_execution_forbids_submission(fixture, artifact):
    f = fixture
    actual, pin = f.frozen()
    (f.ops / artifact).touch()
    with pytest.raises(PermissionError, match="Prior/partial/unknown"):
        f.m.submit(actual, pin)
    assert not f.calls


def test_live_same_named_job_requires_reconciliation(fixture):
    f = fixture
    actual, pin = f.frozen()
    f.queued.append(["900", f.m.JOB_NAME, "PENDING", "(null)"])
    with pytest.raises(PermissionError, match="reconciliation"):
        f.m.submit(actual, pin)
    assert not f.calls


@pytest.mark.parametrize("mode", ["timeout", "ambiguous", "failed"])
def test_unknown_submission_never_retries(fixture, monkeypatch, mode):
    f = fixture
    actual, pin = f.frozen()

    def unknown(argv, **kwargs):
        f.calls.append(argv)
        assert (f.ops / "SUBMIT_INTENT.json").exists()
        if mode == "timeout":
            raise subprocess.TimeoutExpired("sbatch", 20)
        return SimpleNamespace(
            returncode=1 if mode == "failed" else 0, stdout="not-a-job-id", stderr="ambiguous"
        )

    monkeypatch.setattr(f.m.subprocess, "run", unknown)
    assert f.m.submit(actual, pin)["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(PermissionError, match="Prior/partial/unknown"):
        f.m.submit(actual, pin)
    assert len(f.calls) == 1


def test_resource_mismatch_keeps_submitted_job_and_forbids_retry(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()
    original_query = f.m.query
    monkeypatch.setattr(
        f.m, "query", lambda argv: original_query(argv).replace("mem=12G", "mem=66G")
    )
    result = f.m.submit(actual, pin)
    assert result["status"] == "SUBMITTED"
    assert result["request_check"]["status"] == "SUBMITTED_REQUEST_UNVERIFIED_NO_RETRY"
    with pytest.raises(PermissionError):
        f.m.submit(actual, pin)
    assert len(f.calls) == 1


@pytest.mark.parametrize("count", [0, 1, 2])
def test_reconciliation_only_matches_saved_unique_comment(fixture, monkeypatch, count):
    f = fixture
    actual, pin = f.frozen()

    def timeout(argv, **kwargs):
        f.calls.append(argv)
        raise subprocess.TimeoutExpired("sbatch", 20)

    monkeypatch.setattr(f.m.subprocess, "run", timeout)
    f.m.submit(actual, pin)
    original = (f.ops / "SUBMISSION.json").read_bytes()
    comment = f.m.read(f.ops / "SUBMIT_INTENT.json")["comment"]
    rows = [f"{900 + n}|{f.m.JOB_NAME}|RUNNING|0:0|{comment}|" for n in range(count)]
    rows.append(f"800|{f.m.JOB_NAME}|COMPLETED|0:0|another-comment|")
    monkeypatch.setattr(f.m, "query", lambda argv: "\n".join(rows) if argv[0] == "sacct" else "")
    result = f.m.reconcile(actual, pin)
    assert result["job_id"] == ("900" if count == 1 else None)
    assert result["status"] == (
        "RECONCILED_SUBMITTED_NO_RETRY" if count == 1 else "SUBMISSION_STILL_UNKNOWN_NO_RETRY"
    )
    assert (f.ops / "SUBMISSION.json").read_bytes() == original
    assert len(f.calls) == 1
    with pytest.raises(PermissionError):
        f.m.submit(actual, pin)


def test_read_is_nonmutating_and_never_claims_delivery(fixture):
    f = fixture
    actual, _ = f.frozen()
    before = {str(p): p.read_bytes() for p in f.ops.rglob("*") if p.is_file()}
    result = f.m.status(actual)
    after = {str(p): p.read_bytes() for p in f.ops.rglob("*") if p.is_file()}
    assert before == after
    assert result["full_delivery_verified"] is result["evaluation_authorized"] is False
    assert not f.calls


def test_canonical_and_write_scope_rejections(fixture):
    f = fixture
    with pytest.raises(PermissionError, match="Absolute canonical"):
        f.m.child(f.project, "../outside")
    alias = f.project / "alias"
    alias.symlink_to(f.ops.parent, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        f.m.canonical(alias / "x")
    with pytest.raises(PermissionError, match="Writes limited"):
        f.m.write(f.project / "other-project.json", {})


def test_capacity_rechecked_after_evidence_before_intent(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()

    def slow_evidence(_):
        f.queued.extend([[str(n), "other", "PENDING", "(null)"] for n in range(101, 106)])
        return {"verified": True}

    monkeypatch.setattr(f.m, "check_evidence", slow_evidence)
    assert f.m.submit(actual, pin)["status"] == "NO_LEGAL_CAPACITY"
    assert not f.calls and not (f.ops / "SUBMIT_INTENT.json").exists()


def test_unknown_queue_format_cannot_underestimate_capacity(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()
    original_query = f.m.query
    monkeypatch.setattr(
        f.m,
        "query",
        lambda args: (
            "100_[1-99]|other|PENDING|(null)" if args[0] == "squeue" else original_query(args)
        ),
    )
    with pytest.raises(PermissionError, match="Unknown queue shape"):
        f.m.submit(actual, pin)
    assert not f.calls


def test_read_rejects_modified_contract(fixture):
    f = fixture
    actual, _ = f.frozen()
    (f.ops / "CONTRACT.json").write_text("{}")
    with pytest.raises(PermissionError, match="Contract differs"):
        f.m.status(actual)


def test_read_rejects_modified_completed_result(fixture):
    f = fixture
    actual, _ = f.frozen()
    output = f.ops / "RESULT"
    output.mkdir()
    f.m.write(output / "RESULT.json", {"status": "PRIMARY_FACTS_RECOVERY_RECHECK_PENDING"})
    f.m.write(output / "COMPLETE.json", {"result_ref": f.m.reference(output / "RESULT.json")})
    (output / "RESULT.json").write_text("{}")
    with pytest.raises(PermissionError, match="result hash differs"):
        f.m.status(actual)


def test_driver_import_cannot_mutate_deployment_with_bytecode(fixture, monkeypatch):
    f = fixture
    actual = f.deployed()
    monkeypatch.setattr(f.m.sys, "dont_write_bytecode", False)
    with pytest.raises(PermissionError, match="Wrong audit boundaries"):
        f.m.driver(actual)
    assert not list(f.target.rglob("__pycache__"))
