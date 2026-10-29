"""Synthetic isolated operation checks, no scheduler or model invocation."""

import ast
import builtins
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6_memory_repair_ops/operations.py"
    spec = importlib.util.spec_from_file_location("memory_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actual_validator_schema():
    """Read the actual module contract without importing its Torch runtime."""
    path = Path(__file__).resolve().parents[2] / "src/sr_f6_memory_repair/validation.py"
    tree = ast.parse(path.read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SCHEMA" for target in node.targets)
    ]
    assert len(values) == 1
    return values[0]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path / "root")
    monkeypatch.setattr(m, "OPS", tmp_path / "ops")
    monkeypatch.setattr(m, "CODE", tmp_path / "code")
    m.ROOT.mkdir()
    m.CODE.mkdir()
    actual = dict(revision="a" * 40, deployment=str(tmp_path / "source"), binding_sha256="b" * 64)
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    monkeypatch.setattr(m.getpass, "getuser", lambda: m.USER)

    def write_once(path, value):
        with path.open("x") as f:
            json.dump(value, f)

    snapshot = dict(
        gpu_limit=8,
        submit_limit=5,
        reserved_gpus=7,
        submitted_by_user=4,
        can_submit_two_gpu_job=False,
    )
    old = SimpleNamespace(
        clean_submission_environment=lambda: None,
        metadata=lambda: {},
        scheduler=lambda facts: (SimpleNamespace(capacity=lambda **kw: snapshot), None),
        query=lambda argv: "",
        write_once=write_once,
        now=lambda: "SYNTHETIC",
    )
    monkeypatch.setattr(m, "original", lambda: old)
    monkeypatch.setattr(m, "submission_metadata", lambda *args: dict(status="METADATA_ONLY"))
    monkeypatch.setattr(m, "prior_allocation_terminal", lambda *args: {})
    monkeypatch.setattr(m, "quiescent", lambda query: dict(target_locks="OBSERVED_FREE"))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="90001\n", stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    return m, actual, contract, snapshot, calls


def test_one_independent_lane_uses_legal_last_gpu(setup):
    m, a, c, _, calls = setup
    receipt = m.submit(a, "validate", c, m.sha(c))
    assert receipt["status"] == "SUBMITTED" and len(calls) == 1
    assert "--gres=gpu:pro6000:1" in calls[0] and "--mem=33G" in calls[0]
    assert "--no-requeue" in calls[0]
    assert not any("b0" in x for x in calls[0])


@pytest.mark.parametrize("field,value", [("reserved_gpus", 8), ("submitted_by_user", 5)])
def test_full_capacity_no_intent(setup, field, value):
    m, a, c, s, calls = setup
    s[field] = value
    assert m.submit(a, "validate", c, m.sha(c))["status"] == "NO_LEGAL_CAPACITY"
    assert not calls and not (m.OPS / "validate_ATTEMPT.json").exists()


def test_duplicate_intent_blocks_even_missing_receipt(setup):
    m, a, c, _, calls = setup
    m.OPS.mkdir()
    (m.OPS / "validate_ATTEMPT.json").write_text("{}")
    with pytest.raises(PermissionError, match="Prior or partial"):
        m.submit(a, "validate", c, m.sha(c))
    assert not calls


def test_unknown_submit_preserved_not_retried(setup, monkeypatch):
    m, a, c, _, calls = setup

    def timeout(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 25)

    monkeypatch.setattr(m.subprocess, "run", timeout)
    assert m.submit(a, "validate", c, m.sha(c))["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(PermissionError):
        m.submit(a, "validate", c, m.sha(c))
    assert len(calls) == 1


def test_resume_requires_terminal_validation_metadata(setup, monkeypatch):
    m, a, c, _, calls = setup
    seen = []

    def fail(*args):
        seen.append(args)
        raise PermissionError("Incomplete terminal validation")

    monkeypatch.setattr(m, "validation_terminal", fail)
    with pytest.raises(PermissionError, match="Incomplete"):
        m.submit(a, "resume", c, m.sha(c), dict(path=str(c), sha256=m.sha(c)))
    assert seen and not calls and not m.OPS.exists()


def test_validate_forbids_validation_ref(setup):
    m, a, c, _, calls = setup
    with pytest.raises(PermissionError, match="Wrong validation"):
        m.submit(a, "validate", c, m.sha(c), {})
    assert not calls


def test_resume_missing_validation_blocks(setup):
    m, a, c, _, calls = setup
    with pytest.raises(PermissionError, match="Actual immutable"):
        m.submit(a, "resume", c, m.sha(c))
    assert not calls


def test_unknown_capacity_never_zero(setup, monkeypatch):
    m, a, c, _, calls = setup
    old = m.original()

    def fail(**kw):
        raise PermissionError("Unknown GPU reservation")

    old.scheduler = lambda facts: (SimpleNamespace(capacity=fail), None)
    with pytest.raises(PermissionError, match="Unknown GPU"):
        m.submit(a, "validate", c, m.sha(c))
    assert not calls and not (m.OPS / "validate_ATTEMPT.json").exists()


def test_science_and_launch_resources_fixed(setup):
    m, a, c, _, _ = setup
    argv = m.argv_for(a, "resume", c, m.sha(c), dict(path=str(c), sha256=m.sha(c)))
    assert "--cpus-per-task=4" in argv and "--time=23:00:00" in argv
    assert m.TASKS == ("PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11")
    assert "--qos=soujanya-poria-startfund-2026-03" in argv
    with pytest.raises(PermissionError):
        m.argv_for(a, "round-b", c, m.sha(c))


def test_export_injection_rejected(setup):
    m, a, c, _, _ = setup
    with pytest.raises(PermissionError, match="Unsafe"):
        m.argv_for(a, "validate", str(c) + ",EVIL=x", m.sha(c))


def test_symlink_ops_rejected_before_write(setup):
    m, a, c, _, calls = setup
    other = m.PROJECT / "other"
    other.mkdir()
    m.OPS.symlink_to(other, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        m.submit(a, "validate", c, m.sha(c))
    assert not calls and not list(other.iterdir())


def test_target_active_blocks_submit(setup, monkeypatch):
    m, a, c, _, calls = setup

    def fail(query):
        raise PermissionError("Original b1 still live")

    monkeypatch.setattr(m, "quiescent", fail)
    with pytest.raises(PermissionError, match="still live"):
        m.submit(a, "validate", c, m.sha(c))
    assert not calls


def test_allocation_checks_both_requested_and_allocated(setup):
    m, _, _, _, _ = setup
    fields = (
        "JobId=90 JobName=LH-SRF6-Memory-Validate QOS="
        + m.QOS
        + " NumCPUs=4 Partition=cluster02 Account=rose UserId=varun024(10) "
    )
    tres = "cpu=4,mem=33G,gres/gpu=1,gres/gpu:pro6000=1"
    old = SimpleNamespace(query=lambda argv: fields + "ReqTRES=" + tres + " AllocTRES=" + tres)
    assert m.allocation(old, "validate", "90")["JobId"] == "90"
    old.query = lambda argv: (
        fields + "ReqTRES=" + tres + " AllocTRES=cpu=4,mem=128G,gres/gpu:pro6000:1"
    )
    with pytest.raises(PermissionError):
        m.allocation(old, "validate", "90")


def test_real_quiescence_queries_no_purged_job_squeue(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path / "root")
    jobs = m.ROOT / "PHASE_RUNS/PILOT_V3/JOBS"
    failure = jobs / "WORKERS/A/1/ATTEMPTS/A-233996/FAILURE.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(
        json.dumps(dict(status="FAILED_PRESERVED", completed=[], sampling_retry_authorized=False))
    )
    for p in (
        jobs / "WORKERS/A/1/.worker.lock",
        jobs / "TASKS/PRIMARY-b1-U10/.task.lock",
        m.ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b1/r0/U10/.run.lock",
    ):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    calls = []

    def query(argv):
        calls.append(argv)
        return "233996|RUNNING|0:0\n233996.1|FAILED|1:0" if argv[0] == "sacct" else "233996.0"

    assert m.quiescent(query)["original_b0"] == "UNTOUCHED"
    assert calls[1] == ["squeue", "-h", "--steps", "-u", m.USER, "-o", "%i"]

    def active(argv):
        return query(argv) if argv[0] == "sacct" else "233996.1"

    with pytest.raises(PermissionError, match="still live"):
        m.quiescent(active)


def prepare_launch(setup, monkeypatch, action):
    m, a, c, _, calls = setup
    ref = dict(path=str(c), sha256=m.sha(c)) if action == "resume" else None
    monkeypatch.setattr(m, "validation_terminal", lambda *a: {})
    m.submit(a, action, c, m.sha(c), ref)
    calls.clear()
    monkeypatch.setenv("SLURM_JOB_ID", "90001")
    monkeypatch.setenv("SLURM_JOB_NAME", m.NAMES[action])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(m, "allocation", lambda *args: {})
    return m, a, c, ref, calls


def test_failed_child_preserves_attempt_and_stops_queue(setup, monkeypatch):
    m, a, c, ref, calls = prepare_launch(setup, monkeypatch, "resume")

    def fail(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(m.subprocess, "run", fail)
    with pytest.raises(PermissionError, match="child failed"):
        m.launch(a, "resume", c, m.sha(c), ref)
    assert len(calls) == 1 and "PRIMARY-b1-U10" in calls[0]
    assert m.read(m.OPS / "resume_FAILURE.json")["retry_authorized"] is False
    assert not (m.OPS / "resume_COMPLETE.json").exists()
    with pytest.raises(PermissionError, match="Prior launch"):
        m.launch(a, "resume", c, m.sha(c), ref)
    assert len(calls) == 1


def test_resume_fresh_process_queue_and_actual_complete_refs(setup, monkeypatch):
    m, a, c, ref, calls = prepare_launch(setup, monkeypatch, "resume")

    def run(argv, **kw):
        calls.append(argv)
        task = argv[argv.index("--task-id") + 1]
        for path in (
            m.ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task / "COMPLETE.json",
            c.parent / "EXECUTIONS" / task / "COMPLETE.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        assert "PYTHONPATH" not in kw["env"] and "HOME" not in kw["env"]
        assert kw["env"]["HF_HOME"].startswith(str(m.OPS))
        assert argv[1:4] == ["-I", "-B", "-u"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", run)
    m.launch(a, "resume", c, m.sha(c), ref)
    assert [argv[argv.index("--task-id") + 1] for argv in calls] == list(m.TASKS)
    value = m.read(m.OPS / "resume_COMPLETE.json")
    assert len(value["completed"]) == 6
    assert value["status"] == "OPERATION_COMPLETE_NOT_SCIENTIFIC_PASS"
    assert value["original_worker_failure_preserved"] is True


def test_child_exit_zero_without_evidence_is_not_complete(setup, monkeypatch):
    m, a, c, ref, calls = prepare_launch(setup, monkeypatch, "validate")
    with pytest.raises(FileNotFoundError):
        m.launch(a, "validate", c, m.sha(c), ref)
    assert len(calls) == 1 and "sr_f6_memory_repair.validation" in calls[0][5]
    assert (m.OPS / "validate_FAILURE.json").exists()
    assert not (m.OPS / "validate_COMPLETE.json").exists()


def test_submit_never_imports_repair_runtime_or_torch(setup, monkeypatch):
    m, a, c, _, _ = setup
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("torch", "sr_f6_memory_repair", "sr_f6.pilot"))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert m.submit(a, "validate", c, m.sha(c))["status"] == "SUBMITTED"
    intent = m.read(m.OPS / "validate_ATTEMPT.json")
    assert intent["verification"]["status"] == "METADATA_ONLY"


def test_lightweight_metadata_binds_exact_source_and_only_small_refs(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path / "root")
    monkeypatch.setattr(m, "CODE", tmp_path / "code")
    monkeypatch.setattr(m, "FAILED_OPS", tmp_path / "old_ops")
    m.FAILED_OPS.mkdir()
    m.ROOT.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    a = dict(revision="a" * 40, deployment=str(source), binding_sha256="b" * 64)
    binding = dict(files={name: dict(sha256="c" * 64) for name in m.MEMBERS})
    for name in m.MEMBERS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# bound by identity before metadata")
    (source / "MEMORY_REPAIR_SOURCE.json").write_text(json.dumps(binding))
    refpath = m.ROOT / "ref.json"
    refpath.write_text("{}")
    ref = dict(path=str(refpath), sha256=m.sha(refpath))
    value = dict(
        schema="srf6-memory-repair-contract-v2",
        max_host_bytes=8 * 2**30,
        max_disk_bytes=96 * 2**30,
        experiment_root=str(m.ROOT),
        original_job_id="233996",
        original_step_id="233996.1",
        original_execution_id="LH-SRF6-A-1-A-233996-PRIMARY-b1-U10",
        producer=dict(revision=m.PRODUCER, deployment=str(m.CODE), source_manifest_ref=ref),
        repair_source=dict(
            revision=a["revision"],
            deployment=str(source),
            files=[dict(path=name, sha256="c" * 64) for name in m.MEMBERS],
        ),
        plan_ref=ref,
    )
    value.update(
        {
            key: ref
            for key in (
                "original_ready_ref",
                "step0_commit_ref",
                "original_failure_ref",
                "original_task_failure_ref",
                "original_worker_failure_ref",
            )
        }
    )
    failure = m.FAILED_OPS / "validate_FAILURE.json"
    failure.write_text(
        json.dumps(
            dict(
                job_id="234358",
                status="FAILED_PRESERVED",
                error_type="PermissionError",
                error="Wrong actual GPU/CPU/host-memory allocation",
                retry_authorized=False,
            )
        )
    )
    value["old_allocation_failure_ref"] = dict(path=str(failure), sha256=m.sha(failure))
    c = tmp_path / "contract.json"
    c.write_text(json.dumps(value))
    assert m.submission_metadata(a, c, m.sha(c), dict(plan={}, manifest={})) == dict(
        status="SUBMISSION_METADATA_ONLY_NOT_ADMISSION", full_verification="IN_GPU_CHILD"
    )
    value["repair_source"]["files"].pop()
    c.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="exact operation"):
        m.submission_metadata(a, c, m.sha(c), dict(plan={}, manifest={}))


@pytest.mark.parametrize(
    "state,live",
    [("RUNNING|0:0", ""), ("FAILED|1:0", ""), ("COMPLETED|0:0", "90001"), ("COMPLETED|0:0", "")],
)
def test_validation_terminal_matches_actual_validator_and_requires_success(setup, state, live):
    m, a, c, _, _ = setup
    out = m.OPS / "VALIDATION"
    out.mkdir(parents=True)
    verifier = Path(a["deployment"]) / "src/sr_f6_memory_repair/validation.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("# actual bound code")
    refs = {}
    for name in ("evidence_ref", "tensors_ref"):
        path = out / name
        path.write_bytes(b"unread tensor bytes")
        refs[name] = dict(path=str(path), sha256=m.sha(path))
    value = dict(
        schema=actual_validator_schema(),
        contract_sha256=m.sha(c),
        verifier_source_ref=dict(path=str(verifier), sha256=m.sha(verifier)),
        **refs,
    )
    result = out / "RESULT.json"
    result.write_text(json.dumps(value))
    ref = dict(path=str(result), sha256=m.sha(result))
    (m.OPS / "validate_ATTEMPT.json").write_text(
        json.dumps(dict(source=a, contract=dict(path=str(c), sha256=m.sha(c))))
    )
    (m.OPS / "validate_SUBMISSION.json").write_text(
        json.dumps(
            dict(
                status="SUBMITTED",
                job_id="90001",
                source=a,
                attempt_sha256=m.sha(m.OPS / "validate_ATTEMPT.json"),
            )
        )
    )
    (m.OPS / "validate_COMPLETE.json").write_text(
        json.dumps(dict(job_id="90001", source=a, completed=[ref]))
    )
    old = SimpleNamespace(query=lambda argv: "90001|" + state if argv[0] == "sacct" else live)
    if state == "COMPLETED|0:0" and not live:
        assert m.validation_terminal(old, a, c, m.sha(c), ref)["job_id"] == "90001"
        value["schema"] = "srf6-memory-storage-validation-v1"
        result.write_text(json.dumps(value))
        ref["sha256"] = m.sha(result)
        with pytest.raises(PermissionError, match="unrelated validation"):
            m.validation_terminal(old, a, c, m.sha(c), ref)
    else:
        with pytest.raises(PermissionError, match=r"terminal|queued"):
            m.validation_terminal(old, a, c, m.sha(c), ref)


@pytest.mark.parametrize(
    "artifact", [None, "validate_STARTED.json", "VALIDATION", "resume_ATTEMPT.json"]
)
def test_v2_requires_preserved_zero_model_allocation_failure(tmp_path, monkeypatch, artifact):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    monkeypatch.setattr(m, "FAILED_OPS", tmp_path / "old")
    m.FAILED_OPS.mkdir()
    source = dict(
        revision=m.FAILED_REVISION, deployment=str(tmp_path / "sr_f6_memory_repair_code_e24d557")
    )
    intent = dict(source=source, action="validate", validation=None, argv=["--mem=128G"])
    intent_path = m.FAILED_OPS / "validate_ATTEMPT.json"
    intent_path.write_text(json.dumps(intent))
    (m.FAILED_OPS / "validate_SUBMISSION.json").write_text(
        json.dumps(
            dict(
                source=source,
                status="SUBMITTED",
                job_id="234358",
                attempt_sha256=m.sha(intent_path),
            )
        )
    )
    old = SimpleNamespace(
        query=lambda argv: "234358|FAILED|1:0" if argv[0] == "sacct" else "233996"
    )
    if artifact:
        path = m.FAILED_OPS / artifact
        path.mkdir() if artifact == "VALIDATION" else path.write_text("{}")
        with pytest.raises(PermissionError, match="model or recovery"):
            m.prior_allocation_terminal(old)
    else:
        assert m.prior_allocation_terminal(old)["status"] == "FAILED_BEFORE_MODEL_PRESERVED"
        old.query = lambda argv: "234358|FAILED|1:0" if argv[0] == "sacct" else "234358"
        with pytest.raises(PermissionError, match="still queued"):
            m.prior_allocation_terminal(old)


def test_v2_source_uses_new_ops_and_scheduler_real_33g():
    m = load()
    assert m.OPS.name == "PILOT_MEMORY_REPAIR_V2_20261029"
    assert m.FAILED_OPS.name == "PILOT_MEMORY_REPAIR_V1_20261029"
    argv = m.argv_for(
        dict(revision="a" * 40, deployment=str(m.PROJECT / "new"), binding_sha256="b" * 64),
        "validate",
        m.OPS / "CONTRACT.json",
        "c" * 64,
    )
    assert "--mem=33G" in argv and "--mem=128G" not in argv
