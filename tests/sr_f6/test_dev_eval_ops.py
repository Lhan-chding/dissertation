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
        "post_b_ops_test", repo / "scripts/sr_f6_dev_eval/operations.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    project = tmp_path.resolve()
    ops = project / "operations/PILOT_DEV_EVAL_20261101"
    ops.parent.mkdir()
    monkeypatch.setattr(m, "PROJECT", project)
    monkeypatch.setattr(m, "OPS", ops)
    monkeypatch.setattr(m, "CODE", project / "b7")
    monkeypatch.setattr(m, "ROOT", project / "root")
    revision = "a" * 40
    payload = m.make_package(revision, {name: (repo / name).read_bytes() for name in m.MEMBERS})
    binding = m.digest(m.encode(payload["manifest"]))
    target = Path(payload["manifest"]["deployment"])
    calls, queued = [], []
    accounting = "237289|COMPLETED|0:0|\n238798|CANCELLED by 1000|0:15|\n"

    def query(argv):
        if argv[0] == "squeue":
            return "\n".join("|".join(row) for row in queued)
        if argv[0] == "sacct":
            return accounting
        if argv[0] == "scontrol":
            intent = m.read(ops / "SUBMIT_INTENT.json")
            return (
                f"JobId=900 JobName={m.JOB_NAME} Account=rose Partition=cluster02 "
                f"QOS={m.QOS} Requeue=0 TimeLimit=23:00:00 Comment={intent['comment']} "
                "UserId=varun024(1000) ReqTRES=cpu=8,mem=66G,node=1,gres/gpu=2,gres/gpu:pro6000=2"
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
    capacity = {"gpu_limit": 8, "submit_limit": 5, "reserved_gpus": 0, "submitted_by_user": 0}
    monkeypatch.setattr(m, "capacity_provider", lambda: lambda: dict(capacity))
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
    assert f.m.PREFIX + "runtime_adapter.py" in f.m.MEMBERS
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


def test_freeze_twelve_field_contract_and_rejects_refreeze(fixture):
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
        "execution_source",
        "user_authorization",
        "delivery_audit_waived_by_user",
        "full_delivery_verified",
        "formal_authorized",
        "test_authorized",
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
    assert result["status"] == "NO_LEGAL_CAPACITY" and result["requested_gpus"] == 2
    assert not f.calls and not (f.ops / "SUBMIT_INTENT.json").exists()


def test_four_slots_allows_exact_two_gpu_job_once(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()
    monkeypatch.setenv("SBATCH_GRES", "gpu:pro6000:8")
    f.queued.extend([[str(n), "unrelated", "RUNNING", "(null)"] for n in range(101, 105)])
    result = f.m.submit(actual, pin)
    assert result["status"] == "SUBMITTED" and result["job_id"] == "900"
    assert result["request_check"]["status"] == "REQUEST_VERIFIED"
    argv = f.calls[0]
    assert "--cpus-per-task=8" in argv and "--mem=66G" in argv and "--time=23:00:00" in argv
    assert "--no-requeue" in argv and "--gres=gpu:pro6000:2" in argv
    with pytest.raises(PermissionError, match="Prior/partial/unknown"):
        f.m.submit(actual, pin)
    assert len(f.calls) == 1


@pytest.mark.parametrize(
    "artifact",
    [
        "SUBMIT_INTENT.json",
        "SUBMISSION.json",
        "RESULT",
        "dev_cache_900",
        "dev-900.log",
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
        f.m, "query", lambda argv: original_query(argv).replace("mem=66G", "mem=12G")
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
    assert result["full_delivery_verified"] is False
    assert result["evaluation_authorized"] is True
    assert result["formal_authorized"] is result["test_authorized"] is False
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
    f.m.write(output / "SUMMARY.json", {"status": "COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED"})
    f.m.write(output / "COMPLETE.json", {"summary_ref": f.m.reference(output / "SUMMARY.json")})
    (output / "SUMMARY.json").write_text("{}")
    with pytest.raises(PermissionError, match="result hash differs"):
        f.m.status(actual)


def test_driver_import_cannot_mutate_deployment_with_bytecode(fixture, monkeypatch):
    f = fixture
    actual = f.deployed()
    monkeypatch.setattr(f.m.sys, "dont_write_bytecode", False)
    with pytest.raises(PermissionError, match="Wrong DEV boundaries"):
        f.m.driver(actual)
    assert not list(f.target.rglob("__pycache__"))


@pytest.mark.parametrize(
    "reserved,user_jobs,allowed", [(7, 0, False), (8, 0, False), (6, 4, True), (0, 5, False)]
)
def test_teacher_gpu_and_user_capacity_are_independent(fixture, reserved, user_jobs, allowed):
    f = fixture
    actual, pin = f.frozen()
    f.capacity.update(reserved_gpus=reserved, submitted_by_user=user_jobs)
    result = f.m.submit(actual, pin)
    assert result["status"] == ("SUBMITTED" if allowed else "NO_LEGAL_CAPACITY")
    assert len(f.calls) == int(allowed)
    assert (f.ops / "SUBMIT_INTENT.json").exists() == allowed


def test_capacity_rechecks_global_gpu_reservations_after_evidence(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()

    def evidence(_):
        f.capacity["reserved_gpus"] = 7
        return {"evaluation_pristine": True}

    monkeypatch.setattr(f.m, "check_evidence", evidence)
    assert f.m.submit(actual, pin)["status"] == "NO_LEGAL_CAPACITY"
    assert not f.calls and not (f.ops / "SUBMIT_INTENT.json").exists()


def test_unknown_global_reservation_blocks_before_intent(fixture, monkeypatch):
    f = fixture
    actual, pin = f.frozen()

    def unknown():
        raise PermissionError("Unknown GPU reservation; do not assume it is zero")

    monkeypatch.setattr(f.m, "capacity_provider", lambda: unknown)
    with pytest.raises(PermissionError, match="Unknown GPU"):
        f.m.submit(actual, pin)
    assert not f.calls and not (f.ops / "SUBMIT_INTENT.json").exists()


@pytest.mark.parametrize(
    "state,queued",
    [
        ("RUNNING", ""),
        ("FAILED", ""),
        ("COMPLETED", ""),
        ("CANCELLED", "238798|oldaudit|COMPLETING|(null)"),
        ("CANCELLED", "237289|B|COMPLETING|(null)"),
    ],
)
def test_cancelled_audit_and_b_must_leave_queue(fixture, monkeypatch, state, queued):
    f = fixture
    actual = f.deployed()
    monkeypatch.setattr(
        f.m,
        "query",
        lambda args: (
            f"237289|COMPLETED|0:0|\n238798|{state}|0:15|\n" if args[0] == "sacct" else queued
        ),
    )
    with pytest.raises(PermissionError):
        f.m.freeze(actual)
    assert not (f.ops / "CONTRACT.json").exists()


def test_user_waiver_is_fixed_dev_only_and_does_not_claim_audit(fixture):
    f = fixture
    actual, _ = f.frozen()
    value = f.m.read(f.ops / "CONTRACT.json")
    assert value["execution_source"] == actual
    auth = value["user_authorization"]
    assert auth["status"] == "USER_AUTHORIZED_REGISTERED_DEV_EVAL"
    assert auth["delivery_audit_waived_by_user"] is value["delivery_audit_waived_by_user"] is True
    assert auth["full_delivery_verified"] is value["full_delivery_verified"] is False
    assert auth["evaluation_authorized"] is True
    assert (
        auth["training_authorized"] is auth["formal_authorized"] is auth["test_authorized"] is False
    )
    assert (auth["endpoints"], auth["responses"]) == (12, 2304)
    assert all(
        set(auth[key]) == {"path", "sha256"}
        for key in ("plan_ref", "spec_ref", "phase_ref", "admission_ref")
    )
    value["user_authorization"]["formal_authorized"] = True
    (f.ops / "CONTRACT.json").write_bytes(f.m.encode(value))
    with pytest.raises(PermissionError, match="Contract differs"):
        f.m.metadata(actual, f.m.digest(f.m.data(f.ops / "CONTRACT.json")))


@pytest.mark.parametrize("tamper", [None, "manifest", "operator", "identity", "limits", "unknown"])
def test_capacity_provider_authenticates_frozen_chain(tmp_path, monkeypatch, tamper):
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6_dev_eval/operations.py"
    spec = importlib.util.spec_from_file_location("dev_capacity_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    v8 = tmp_path / "v8"
    name = "scripts/sr_f6_memory_resume_ops/operations.py"
    source_path = v8 / name
    source_path.parent.mkdir(parents=True)
    source = """from types import SimpleNamespace
calls=[]
def identity(revision, deployment, binding):
    assert len(revision)==40 and deployment.endswith('/v8') and len(binding)==64
    calls.append('identity')
def frozen_v4():
    assert calls == ['identity']
    calls.append('v4')
    return SimpleNamespace(original=original), None
def original():
    calls.append('original')
    return SimpleNamespace(scheduler=scheduler, metadata=lambda: {'fixture':True},
                           query=lambda argv: 'fixture')
def scheduler(metadata):
    assert metadata == {'fixture':True} and calls == ['identity','v4','original']
    calls.append('scheduler')
    return SimpleNamespace(capacity=capacity), None
def capacity(query):
    assert query([])=='fixture'
    return {'gpu_limit':8,'submit_limit':5,'reserved_gpus':6,'submitted_by_user':4}
"""
    if tamper == "identity":
        source = source.replace("assert len(revision)==40", "assert False and len(revision)==40")
    elif tamper == "limits":
        source = source.replace("'gpu_limit':8", "'gpu_limit':10")
    elif tamper == "unknown":
        source = source.replace("'reserved_gpus':6", "'reserved_gpus':None")
    source_path.write_text(source)
    manifest = {
        "files": {name: {"bytes": len(source.encode()), "sha256": m.digest(source.encode())}}
    }
    manifest_path = v8 / "RESUME_SOURCE.json"
    manifest_path.write_bytes(m.encode(manifest))
    monkeypatch.setattr(m, "V8", v8)
    monkeypatch.setattr(m, "V8_BINDING", m.digest(m.data(manifest_path)))
    if tamper == "manifest":
        manifest_path.write_text("{}")
    elif tamper == "operator":
        source_path.write_text(source + "\nraise AssertionError('untrusted source executed')\n")
    before = {str(p): p.read_bytes() for p in v8.rglob("*") if p.is_file()}
    if tamper:
        with pytest.raises((PermissionError, AssertionError)):
            m.capacity_provider()()
    else:
        snapshot = m.capacity_provider()
        assert (
            snapshot()
            == snapshot()
            == {"gpu_limit": 8, "submit_limit": 5, "reserved_gpus": 6, "submitted_by_user": 4}
        )
    assert before == {str(p): p.read_bytes() for p in v8.rglob("*") if p.is_file()}


def test_tracking_reads_started_endpoints_without_requiring_pristine(fixture, monkeypatch):
    f = fixture
    actual, _ = f.frozen()
    endpoint = f.m.ROOT / "PHASE_RUNS/PILOT_V3/EVALUATION/b0/U10/step12"
    endpoint.mkdir(parents=True)
    (endpoint / "STARTED.json").write_text('{"status":"STARTED"}')
    lane = f.ops / "RESULT/LANES/0"
    lane.mkdir(parents=True)
    f.m.write(lane / "STARTED.json", {"at": "fixture", "status": "STARTED"})
    monkeypatch.setattr(
        f.m, "check_evidence", lambda _: pytest.fail("Read must not require pristine")
    )
    before = {str(p): p.read_bytes() for p in f.project.rglob("*") if p.is_file()}
    result = f.m.status(actual)
    assert result["endpoints"]["b0/U10/step12"]["STARTED.json"]["value"]["status"] == "STARTED"
    assert result["execution"]["LANES/0/STARTED.json"]["summary"]["status"] == "STARTED"
    assert len(result["endpoints"]) == 12
    assert before == {str(p): p.read_bytes() for p in f.project.rglob("*") if p.is_file()}
