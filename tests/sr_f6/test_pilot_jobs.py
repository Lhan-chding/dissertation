"""CPU operations fixtures only; no model loading, server access or scientific PASS."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from sr_f6 import pilot_jobs as jobs
from sr_f6.planning import formal_paths


@pytest.fixture
def case(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    selected = [
        p for p in formal_paths(config) if p["repeat"] == 0 and p["arm"] in ("U10", "U11", "C_SHUF")
    ]
    paths = [
        dict(
            parent=p["parent"],
            arm=p["arm"],
            order=i + 1,
            audit_root=f"PHASE_RUNS/PILOT_V1/PILOT_FROZEN/runs/{p['parent']}/r0/{p['arm']}",
        )
        for i, p in enumerate(selected)
    ]
    endpoints = [dict(endpoint_id=f"{p}/parent") for p in ("b0", "b1", "b2")]
    endpoints += [dict(endpoint_id=f"{p['parent']}/{p['arm']}/step12") for p in paths]
    pins = dict(
        phase_identity_sha256="a" * 64,
        pilot_spec_sha256="b" * 64,
        admission_ref=dict(path="PHASES/PILOT_V1/ADMISSION.json", sha256="c" * 64),
    )
    admitted = dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope="PILOT_V1_REGISTERED_ONLY",
        phase_context={"fixture": "SYNTHETIC_ONLY"},
        pilot_spec_ref={"sha256": "b" * 64},
        pilot_spec=dict(
            phase="PILOT_V1",
            paths=paths,
            evaluation=dict(endpoints=endpoints),
            phase_identity=dict(sha256="a" * 64),
            actual_source_manifest=dict(sha256="d" * 64),
            budgets=dict(
                primary_responses=13824, recovery_new_responses=1024, dev_eval_responses=2304
            ),
        ),
    )
    monkeypatch.setattr(
        jobs.pilot_contract, "require_admission", lambda root, **kwargs: copy.deepcopy(admitted)
    )
    monkeypatch.setattr(jobs, "_worker_environment", lambda root: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    ref = jobs.freeze_plan(tmp_path, **pins)
    plan, _ = jobs.verify_plan(tmp_path, ref)
    return dict(root=tmp_path, pins=pins, admitted=admitted, ref=ref, plan=plan)


def test_fixed_two_rounds_and_twelve_endpoint_partition(case):
    plan = case["plan"]
    assert plan["queues"]["A"] == [
        ["PRIMARY-b0-U11", "PRIMARY-b0-C_SHUF", "PRIMARY-b0-U10"],
        ["PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11"],
    ]
    assert plan["queues"]["B"] == [
        ["PRIMARY-b2-U10", "PRIMARY-b2-U11", "PRIMARY-b2-C_SHUF"],
        ["RECOVERY-b0-U11"],
    ]
    assert [len(q) for q in plan["queues"]["EVAL"]] == [6, 6]
    assert len({t["task_id"] for t in plan["tasks"]}) == 22
    assert jobs.freeze_plan(case["root"], **case["pins"]) == case["ref"]
    assert not plan.get("formal_authorized", False) and plan["gate_status"] == "NOT_SIGNED"


def test_plan_cannot_be_retrofitted_or_tampered(case, monkeypatch, tmp_path):
    other = tmp_path / "after-start"
    (other / case["admitted"]["pilot_spec"]["paths"][0]["audit_root"] / "checkpoints").mkdir(
        parents=True
    )
    with pytest.raises(PermissionError, match="before first pilot"):
        jobs.freeze_plan(other, **case["pins"])
    file = case["root"] / jobs.PLAN
    obj = json.loads(file.read_bytes())
    obj["queues"]["A"][0].reverse()
    file.write_text(json.dumps(obj))
    with pytest.raises(PermissionError):
        jobs.verify_plan(case["root"], jobs._ref(case["root"], file))
    monkeypatch.setattr(
        jobs.pilot_contract,
        "require_admission",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("actual evidence missing")),
    )
    with pytest.raises(PermissionError, match="actual evidence"):
        jobs.build_plan(case["root"], **case["pins"])


def controlled_worker(case, monkeypatch):
    calls, verified = [], []

    def facts(root, ref, admitted, task, *, receipt=None):
        path = jobs._result_path(root, task)
        value = receipt if receipt is not None else (jobs._read(path) if path.exists() else None)
        if value is not None:
            assert value["task"] == task and value["plan_ref"] == ref
            producer = jobs.recovery._bound(root, value["producer"])
            if producer.get("valid") is not True:
                raise PermissionError("SYNTHETIC invalid producer facts")
            verified.append(task["task_id"])
        return value

    def execute(root, ref, plan, admitted, task, execution_id, attempt):
        calls.append((task["task_id"], execution_id))
        producer = attempt / "CONTROL_PRODUCER.json"
        jobs._write(producer, dict(valid=True, scope="SYNTHETIC_ONLY"))
        return jobs._ref(root, producer)

    monkeypatch.setattr(jobs, "_verify_task", facts)
    monkeypatch.setattr(jobs, "_execute", execute)
    monkeypatch.setattr(
        jobs,
        "_registered",
        lambda root, admitted, task: (root / "SYNTHETIC_PRODUCERS" / task["task_id"], {}),
    )
    return calls, verified, execute


def run(case, **kwargs):
    return jobs.run_worker(case["root"], plan_ref=case["ref"], **kwargs)


def test_fixed_worker_order_unique_attempts_and_complete_reuse(case, monkeypatch):
    calls, verified, _ = controlled_worker(case, monkeypatch)
    report = run(case, round_id="A", worker=0, attempt_id="first")
    assert [c[0] for c in calls] == case["plan"]["queues"]["A"][0]
    assert len({c[1] for c in calls}) == 3 and all(c[1].startswith("LH-SRF6-") for c in calls)
    assert report["status"] == "QUEUE_COMPLETE_NOT_GATE_SIGNED" and len(verified) == 3
    run(case, round_id="A", worker=0, attempt_id="new-process")
    assert len(calls) == 3
    with pytest.raises(PermissionError, match="immutable"):
        run(case, round_id="A", worker=0, attempt_id="first")
    with pytest.raises(PermissionError, match="Prior fixed round"):
        run(case, round_id="B", worker=0, attempt_id="too-early")
    assert len(calls) == 3


def test_failure_preserved_no_auto_retry_and_peer_finishes(case, monkeypatch):
    calls, _, success = controlled_worker(case, monkeypatch)
    failed_task = case["plan"]["queues"]["A"][0][0]

    def fail(*args):
        calls.append((args[4]["task_id"], args[5]))
        raise RuntimeError("SYNTHETIC process failure")

    monkeypatch.setattr(jobs, "_execute", fail)
    with pytest.raises(RuntimeError, match="process failure"):
        run(case, round_id="A", worker=0, attempt_id="failed")
    failure = case["root"] / jobs.DIRECTORY / "WORKERS/A/0/ATTEMPTS/failed/FAILURE.json"
    assert jobs._read(failure)["peer_cancellation_requested"] is False
    with pytest.raises(PermissionError, match="explicit resume"):
        run(case, round_id="A", worker=0, attempt_id="not-a-retry-permit")
    assert len(calls) == 1
    monkeypatch.setattr(jobs, "_execute", success)
    run(case, round_id="A", worker=1, attempt_id="peer")
    assert failure.exists()
    run(case, round_id="A", worker=0, attempt_id="explicit", resume_tasks=[failed_task])
    assert len(calls) == 7 and failure.exists()


def test_false_producer_never_publishes_task_complete(case, monkeypatch):
    controlled_worker(case, monkeypatch)

    def invalid(root, ref, plan, admitted, task, execution_id, attempt):
        jobs._write(attempt / "bad.json", dict(valid=False, status="COMPLETE"))
        return jobs._ref(root, attempt / "bad.json")

    monkeypatch.setattr(jobs, "_execute", invalid)
    with pytest.raises(PermissionError, match="invalid producer"):
        run(case, round_id="A", worker=0, attempt_id="invalid")
    task = case["plan"]["tasks"][0]
    assert not jobs._result_path(case["root"], task).exists()


def test_actual_cli_arguments_and_fresh_child_per_model_task(case, monkeypatch, tmp_path):
    calls = []
    directory = tmp_path / "producer"
    monkeypatch.setattr(jobs, "_registered", lambda *args: (directory, {}))

    def invoke(command, log):
        calls.append(command)
        execution = command[command.index("--task-id") + 1]
        if "sr_f6.pilot" in command:
            path = directory / "TRAINER_ATTEMPTS" / execution / "COMPLETE.json"
        else:
            endpoint = command[command.index("--endpoint-id") + 1]
            path = case["root"] / jobs.evaluation.DIRECTORY / endpoint / "COMPLETE.json"
        jobs._write(path, dict(scope="SYNTHETIC_ONLY"))

    monkeypatch.setattr(jobs, "_invoke", invoke)
    for task in (case["plan"]["tasks"][0], case["plan"]["tasks"][10]):
        attempt = tmp_path / task["task_id"]
        attempt.mkdir()
        jobs._execute(
            case["root"],
            case["ref"],
            case["plan"],
            case["admitted"],
            task,
            task["task_id"],
            attempt,
        )
    assert len(calls) == 2 and calls[0][:4] == [sys.executable, "-u", "-m", "sr_f6.pilot"]
    assert "--admission" in calls[0] and "--admission-sha256" in calls[1]
    assert "--max-steps" not in str(calls) and "--seed" not in str(calls)


def test_completed_producer_rebuilds_only_operations_receipt(case, monkeypatch):
    calls, _, _ = controlled_worker(case, monkeypatch)
    task = case["plan"]["tasks"][0]
    attempt = jobs._result_path(case["root"], task).parent / "ATTEMPTS/interrupted"
    jobs._write(
        attempt / "STARTED.json", dict(task=task, plan_ref=case["ref"], execution_id="already-ran")
    )
    jobs._write(attempt / "FAILURE.json", dict(status="FAILED_PRESERVED"))
    directory, _ = jobs._registered(case["root"], case["admitted"], task)
    producer = directory / "TRAINER_ATTEMPTS/already-ran/COMPLETE.json"
    jobs._write(producer, dict(valid=True, scope="SYNTHETIC_ONLY"))
    run(case, round_id="A", worker=0, attempt_id="resume-derived-only")
    assert [c[0] for c in calls] == case["plan"]["queues"]["A"][0][1:]
    assert jobs._read(jobs._result_path(case["root"], task))["producer"] == jobs._ref(
        case["root"], producer
    )
    assert (attempt / "FAILURE.json").exists()


def test_eval_worker_verifies_exact_recovery_once_before_six_fixed_endpoints(case, monkeypatch):
    calls, _, _ = controlled_worker(case, monkeypatch)
    comparisons = []
    monkeypatch.setattr(jobs, "_dependency_check", lambda *a: None)
    monkeypatch.setattr(jobs, "_evaluation_barrier", lambda *a, **k: None)
    monkeypatch.setattr(
        jobs, "_comparison", lambda *a: comparisons.append("actual-verifier-control") or ()
    )
    run(case, round_id="EVAL", worker=0, attempt_id="eval-control")
    assert comparisons == ["actual-verifier-control"]
    assert [c[0] for c in calls] == case["plan"]["queues"]["EVAL"][0]


def test_recovery_calls_only_existing_registered_prepare_run_compare(case, monkeypatch):
    root, calls = case["root"], []
    task = next(t for t in case["plan"]["tasks"] if t["kind"] == "RECOVERY")
    primary_dir = root / "SYNTHETIC_PRIMARY"
    identity = {"fixture": True}
    monkeypatch.setattr(jobs, "_registered", lambda *a: (primary_dir, identity))
    primary_execution = {"path": "SYNTHETIC_PRIMARY/EXECUTIONS/first/END.json", "sha256": "e" * 64}
    jobs._write(root / "primary.json", dict(execution=primary_execution))
    monkeypatch.setattr(
        jobs, "_verify_task", lambda *a: dict(producer=jobs._ref(root, root / "primary.json"))
    )

    class HeldControl:
        def __init__(self, directory, identity):
            self.directory, self.identity, self.held = directory, identity, False

        def __enter__(self):
            self.held = True
            return self

        def __exit__(self, *args):
            self.held = False

        def published(self):
            return [dict(step=i, commit_sha256=str(i) * 64) for i in range(13)]

    monkeypatch.setattr(jobs, "RunTransactions", HeldControl)

    def prepare(tx, **kwargs):
        assert tx.held and kwargs["source_commit_sha256"] == "4" * 64
        assert kwargs["primary_execution_ref"] == primary_execution
        calls.append("prepare")
        jobs._write(root / jobs.recovery.BINDING, dict(primary_execution_ref=primary_execution))
        return dict(binding_ref=jobs._ref(root, root / jobs.recovery.BINDING))

    def invoke(command, log):
        assert "--recovery-binding" in command and command[command.index("--parent") + 1] == "b0"
        assert command[command.index("--arm") + 1] == "U11"
        calls.append("new-process-run")
        jobs._write(
            root / jobs.recovery.DIRECTORY / "TRAINER_ATTEMPTS/recovery-control/COMPLETE.json",
            dict(execution={"path": "recovered/END.json", "sha256": "f" * 64}),
        )

    def compare(primary, restored, **kwargs):
        assert (
            primary.held and restored.held and kwargs["primary_execution_ref"] == primary_execution
        )
        calls.append("compare")
        report = dict(
            status="PASS", scope="SERVER_ACTUAL", exact=True, independent_process_verified=True
        )
        output = root / "SYNTHETIC_COMPARE.json"
        jobs._write(output, report)
        return dict(report, receipt_reference=jobs._ref(root, output))

    monkeypatch.setattr(jobs.recovery, "prepare_recovery", prepare)
    monkeypatch.setattr(jobs.recovery, "compare_recovery", compare)
    monkeypatch.setattr(
        jobs.recovery,
        "verify_recovery",
        lambda *a, **k: dict(
            directory=str(root / jobs.recovery.DIRECTORY), identity=identity, scope="SERVER_ACTUAL"
        ),
    )
    monkeypatch.setattr(jobs, "_training_done", lambda *a, **k: a[3])
    monkeypatch.setattr(jobs, "_invoke", invoke)
    admitted = dict(case["admitted"], phase_context={"fixture": True})
    attempt = root / "recovery-control"
    attempt.mkdir()
    ref = jobs._execute(
        root, case["ref"], case["plan"], admitted, task, "recovery-control", attempt
    )
    assert calls == ["prepare", "new-process-run", "compare"]
    assert jobs.recovery._bound(root, ref)["binding_ref"]["path"] == jobs.recovery.BINDING


@pytest.mark.parametrize("window", ["before_compare", "after_compare", "existing_fail"])
def test_recovery_completed_training_never_restarts_model_to_repair_operations(
    case, monkeypatch, window
):
    root = case["root"]
    task = next(t for t in case["plan"]["tasks"] if t["kind"] == "RECOVERY")
    primary_dir, identity = root / "SYNTHETIC_PRIMARY", {"fixture": True}
    recovered = root / jobs.recovery.DIRECTORY
    monkeypatch.setattr(jobs, "_registered", lambda *a: (primary_dir, identity))
    monkeypatch.setattr(jobs, "_training_done", lambda *a, **k: a[3])
    monkeypatch.setattr(jobs, "_dependency_check", lambda *a: None)
    monkeypatch.setattr(
        jobs,
        "_execute",
        lambda *a: (_ for _ in ()).throw(AssertionError("No new model producer allowed")),
    )

    class HeldControl:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(jobs, "RunTransactions", HeldControl)
    monkeypatch.setattr(
        jobs.recovery,
        "verify_recovery",
        lambda *a, **k: dict(directory=str(recovered), identity=identity, scope="SERVER_ACTUAL"),
    )
    primary_execution = {"path": "original/END.json", "sha256": "e" * 64}
    recovery_execution = {"path": "recovery/END.json", "sha256": "f" * 64}
    jobs._write(root / jobs.recovery.BINDING, dict(primary_execution_ref=primary_execution))
    binding_ref = jobs._ref(root, root / jobs.recovery.BINDING)
    old = jobs._result_path(root, task).parent / "ATTEMPTS/interrupted-ops"
    jobs._write(
        old / "STARTED.json",
        dict(task=task, plan_ref=case["ref"], execution_id="original-recovery"),
    )
    trained = recovered / "TRAINER_ATTEMPTS/original-recovery/COMPLETE.json"
    jobs._write(trained, dict(execution=recovery_execution))
    report = dict(
        status="FAIL" if window == "existing_fail" else "PASS",
        scope="SERVER_ACTUAL",
        exact=window != "existing_fail",
        independent_process_verified=True,
        binding_ref=binding_ref,
        primary_execution_ref=primary_execution,
        recovery_execution_ref=recovery_execution,
    )
    comparison = recovered / "COMPARISONS/original.json"
    calls = []

    def compare(*args, **kwargs):
        calls.append("new-comparison")
        jobs._write(comparison, report)
        return dict(report, receipt_reference=jobs._ref(root, comparison))

    def verify(*args, **kwargs):
        calls.append("verify-original-comparison")
        assert kwargs["comparison_ref"] == jobs._ref(root, comparison)
        return dict(
            report, receipt_reference=kwargs["comparison_ref"], verified_from_saved_facts=True
        )

    monkeypatch.setattr(jobs.recovery, "compare_recovery", compare)
    monkeypatch.setattr(jobs.recovery, "verify_comparison", verify)
    if window != "before_compare":
        jobs._write(comparison, report)
    if window == "existing_fail":
        with pytest.raises(PermissionError, match="Existing recovery comparison failed"):
            run(case, round_id="B", worker=1, attempt_id="repair-only")
        assert not jobs._result_path(root, task).exists() and calls == []
    else:
        result = run(case, round_id="B", worker=1, attempt_id="repair-only")
        assert result["status"] == "QUEUE_COMPLETE_NOT_GATE_SIGNED"
        assert calls == (
            ["new-comparison"] if window == "before_compare" else ["verify-original-comparison"]
        )
        combined = jobs._read(old / "RECOVERY_RESULT.json")
        assert combined["training_ref"] == jobs._ref(root, trained)
        assert combined["recovery_execution_ref"] == recovery_execution
        assert len(list((recovered / "TRAINER_ATTEMPTS").iterdir())) == 1


def test_model_environment_requires_lh_prefix_and_one_gpu(tmp_path, monkeypatch):
    jobs._write(tmp_path / "CONFIG.json", dict(deployment=dict(python=sys.executable)))
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-control")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_NAME", "LH-SRF6wrong")
    with pytest.raises(PermissionError, match="LH-SRF6-"):
        jobs._worker_environment(tmp_path)
    monkeypatch.setenv("SLURM_JOB_NAME", "LH-SRF6-Pilot-A")
    jobs._worker_environment(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(PermissionError, match="exactly one"):
        jobs._worker_environment(tmp_path)


def _environment():
    return dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))


def _wait_file(path, *, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return path.read_text()
        time.sleep(0.05)
    raise AssertionError("CPU child did not report readiness")


def test_sigterm_cleans_only_owned_cpu_child_group(tmp_path):
    log = tmp_path / "owned.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        "import os,time;print(os.getpid(),flush=True);time.sleep(60)",
    ]
    script = (
        "from pathlib import Path;from sr_f6.pilot_jobs import _invoke;"
        f"_invoke({command!r},Path({str(log)!r}))"
    )
    driver = subprocess.Popen(
        [sys.executable, "-c", script],
        env=_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    peer = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    owned = None
    try:
        owned = int(_wait_file(log).strip())
        driver.terminate()
        assert driver.wait(timeout=10) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(owned, 0)
        assert peer.poll() is None
    finally:
        for process in (driver, peer):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        if owned is not None:
            with suppress(ProcessLookupError):
                os.killpg(owned, signal.SIGKILL)


def test_two_real_cpu_workers_serialize_prerequisites_before_eval_barrier(tmp_path):
    root = str(tmp_path)
    script = """
import json,sys,time
from pathlib import Path
from sr_f6.pilot_jobs import _lock,_evaluation_barrier,DIRECTORY
root=Path(sys.argv[1]);worker=int(sys.argv[2])
with _lock(root/"prerequisite.lock",wait=True):
    marker=root/"exclusive"
    assert not marker.exists()
    marker.write_text(str(worker))
    time.sleep(.15)
    marker.unlink()
    (root/f"verified{worker}").write_text("true")
_evaluation_barrier(root,{"path":"fixture","sha256":"a"*64},worker,"paired",timeout=10)
assert (root/"verified0").exists() and (root/"verified1").exists()
(root/f"started{worker}").write_text("true")
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", script, root, str(w)],
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for w in (0, 1)
    ]
    for child in children:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
    assert (tmp_path / "started0").exists() and (tmp_path / "started1").exists()


def test_peer_failure_stops_barrier_without_starting_endpoints(tmp_path):
    failure = tmp_path / jobs.DIRECTORY / "WORKERS/EVAL/1/ATTEMPTS/same/FAILURE.json"
    jobs._write(failure, dict(status="FAILED_PRESERVED"))
    with pytest.raises(PermissionError, match="Peer prerequisite"):
        jobs._evaluation_barrier(
            tmp_path, dict(path="fixture", sha256="a" * 64), 0, "same", timeout=1
        )
