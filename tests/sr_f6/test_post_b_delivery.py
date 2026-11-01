"""Boundary tests for the independent, CPU-only post-B delivery driver."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6_post_b_delivery/audit.py"
    spec = importlib.util.spec_from_file_location("post_b_delivery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def contract(driver, tmp_path, monkeypatch):
    base = tmp_path.resolve()
    ops = base / "ops/PILOT_POST_B_DELIVERY_20261101"
    ops.mkdir(parents=True)
    script = save(base / "audit.py", {"fixture": True})
    monkeypatch.setattr(driver, "BASE", base)
    monkeypatch.setattr(driver, "NEW_OPS", ops)
    monkeypatch.setattr(driver, "__file__", str(script))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    value = dict(
        schema=driver.SCHEMA,
        output_directory=str(ops / "RESULT"),
        driver_sha256=driver.sha(script),
        source_deployment=str(driver.DEPLOYMENT),
        b_job_id="237289",
        round_b_complete_sha256=driver.B_COMPLETE_SHA,
    )
    path = save(ops / "CONTRACT.json", value)
    return path, value


def test_bound_rejects_tampering_and_escape(driver, tmp_path):
    base = tmp_path.resolve()
    path = save(base / "file.json", {"x": 1})
    reference = driver.ref(path)
    assert driver.bound(base, reference) == {"x": 1}
    path.write_text("{}")
    with pytest.raises(PermissionError, match="hash differs"):
        driver.bound(base, reference)
    with pytest.raises(PermissionError, match=r"Absolute plain|escapes"):
        driver.under(base, "../outside")


def test_symlink_ancestor_rejected(driver, tmp_path):
    base = tmp_path.resolve()
    (base / "real").mkdir()
    (base / "alias").symlink_to(base / "real", target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        driver.plain(base / "alias/no-file-needed")


@pytest.mark.parametrize("cpu, cuda", [("2", ""), ("1", "0"), ("1", None)])
def test_environment_rejected_before_output(driver, contract, monkeypatch, cpu, cuda):
    path, _ = contract
    monkeypatch.setenv("OMP_NUM_THREADS", cpu)
    if cuda is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
    with pytest.raises(PermissionError, match="OMP1"):
        driver.run(path, driver.sha(path))
    assert not (path.parent / "RESULT").exists()


def test_driver_hash_rejected_before_intent(driver, contract):
    path, value = contract
    value["driver_sha256"] = "0" * 64
    save(path, value)
    with pytest.raises(PermissionError, match="Driver differs"):
        driver.run(path, driver.sha(path))
    assert not (path.parent / "RESULT").exists()


def test_exception_preserved_without_scientific_fail_or_retry(driver, contract, monkeypatch):
    path, _ = contract

    def failed():
        raise TimeoutError("scheduler unavailable")

    monkeypatch.setattr(driver, "scheduler", failed)
    with pytest.raises(TimeoutError):
        driver.run(path, driver.sha(path))
    output = path.parent / "RESULT"
    failure_bytes = (output / "FAILURE.json").read_bytes()
    failure = json.loads(failure_bytes)
    assert failure["status"] == "DELIVERY_VERIFICATION_INCOMPLETE"
    assert failure["scientific_gate_status"] == "NOT_EVALUATED"
    assert not (output / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        driver.run(path, driver.sha(path))
    assert (output / "FAILURE.json").read_bytes() == failure_bytes


def test_measurement_false_does_not_claim_scientific_fail(driver, contract, monkeypatch):
    path, _ = contract
    monkeypatch.setattr(driver, "scheduler", lambda: {"job_id": "900"})
    monkeypatch.setattr(driver, "source_and_pins", lambda: ({}, {}, {}, {}))
    monkeypatch.setattr(driver, "verify_b", lambda plan: ({"verified": True}, {}))
    monkeypatch.setattr(
        driver,
        "verify_training",
        lambda *args: {"exposure": {"registered_exposure_met": False}, "paths": 9},
    )
    result = driver.run(path, driver.sha(path))
    assert result["status"] == "DELIVERY_FACTS_NOT_SCIENTIFIC_GATE"
    assert result["full_delivery_verified"] is True
    assert result["recovery_recheck_status"] == "EXACT_SAVED_FACTS_AND_HISTORY_VERIFIED"
    assert result["gate_status"] == "NOT_SIGNED"
    assert result["exposure"]["registered_exposure_met"] is False
    assert result["evaluation_authorized"] is result["formal_authorized"] is False
    complete = json.loads((path.parent / "RESULT/COMPLETE.json").read_text())
    assert driver.bound(path.parent / "RESULT", complete["result_ref"]) == result


def allocation(**changes):
    fields = dict(
        JobId="900",
        JobState="RUNNING",
        JobName="LH-SRF6-PostB-Delivery",
        NumCPUs="4",
        Requeue="0",
        TimeLimit="02:00:00",
        ReqTRES="cpu=4,mem=12G,node=1,billing=4",
        AllocTRES="cpu=4,mem=12G,node=1,billing=4",
        Account="rose",
        Partition="cluster02",
        QOS="soujanya-poria-startfund-2026-03",
    )
    fields.update(changes)
    return " ".join(f"{k}={v}" for k, v in fields.items())


@pytest.mark.parametrize(
    "change",
    [
        dict(Requeue="1"),
        dict(NumCPUs="8"),
        dict(AllocTRES="cpu=4,mem=12G,gres/gpu=1"),
        dict(ReqTRES="cpu=4,mem=8G"),
        dict(JobState="PENDING"),
        dict(TimeLimit="23:00:00"),
        dict(Account="wrong"),
        dict(Partition="wrong"),
        dict(QOS="wrong"),
    ],
)
def test_scheduler_rejects_wrong_actual_allocation(driver, monkeypatch, change):
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    monkeypatch.setattr(driver, "command", lambda args: allocation(**change))
    with pytest.raises(PermissionError):
        driver.scheduler()


@pytest.mark.parametrize(
    "accounting,queue",
    [
        ("237289|RUNNING|0:0|\n", "900\n"),
        ("237289|FAILED|1:0|\n", "900\n"),
        ("237289|COMPLETED|0:0|\n", "237289\n900\n"),
        ("", "900\n"),
    ],
)
def test_b_must_be_terminal_and_absent(driver, monkeypatch, accounting, queue):
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    monkeypatch.setattr(
        driver,
        "command",
        lambda args: {"scontrol": allocation(), "sacct": accounting, "squeue": queue}[args[0]],
    )
    with pytest.raises(PermissionError):
        driver.scheduler()


def test_live_audit_job_does_not_conflict_with_absent_b(driver, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    monkeypatch.setattr(
        driver,
        "command",
        lambda args: {
            "scontrol": allocation(),
            "sacct": "237289|COMPLETED|0:0|\n",
            "squeue": "900\n901\n",
        }[args[0]],
    )
    assert driver.scheduler()["job_id"] == "900"


def test_no_existing_run_is_created(driver, tmp_path):
    path = tmp_path.resolve() / "missing"
    called = []
    tx = SimpleNamespace(RunTransactions=lambda *args: called.append(args))
    with pytest.raises(PermissionError, match="Existing run"):
        driver.existing_transaction(tx, path, {})
    assert not path.exists() and not called


def test_existing_lock_and_identity_required(driver, tmp_path):
    path = tmp_path.resolve()
    (path / "checkpoints").mkdir()
    save(path / "RUN_IDENTITY.json", {"id": "original"})
    called = []
    tx = SimpleNamespace(RunTransactions=lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        driver.existing_transaction(tx, path, {"id": "original"})
    (path / ".run.lock").touch()
    with pytest.raises(PermissionError, match="identity differs"):
        driver.existing_transaction(tx, path, {"id": "changed"})
    assert not called
    driver.existing_transaction(tx, path, {"id": "original"})
    assert len(called) == 1


@pytest.mark.parametrize("mutation", [False, True])
def test_calls_original_verifiers_under_all_locks(driver, tmp_path, monkeypatch, mutation):
    base = tmp_path.resolve()
    output = base / "ops/RESULT"
    output.mkdir(parents=True)
    monkeypatch.setattr(driver, "ROOT", base / "experiment")
    monkeypatch.setattr(driver, "NEW_OPS", base / "ops")
    monkeypatch.setattr(driver, "DEPLOYMENT", base / "source")
    monkeypatch.setattr(driver, "sys", SimpleNamespace(modules={}, path=[], platform="linux"))
    paths = [
        dict(
            parent=p,
            arm=a,
            audit_root=f"runs/{p}/{a}",
            transaction_identity={"path": f"{p}-{a}", "sha256": "fixture"},
        )
        for p, a in sorted(driver.PAIRS)
    ]
    live, calls = set(), []

    class Transaction:
        def __init__(self, directory, identity):
            self.directory, self.identity = directory, identity

        def __enter__(self):
            live.add(self.directory)
            return self

        def __exit__(self, *_):
            live.remove(self.directory)

    monkeypatch.setattr(driver, "existing_transaction", lambda _, d, i: Transaction(d, i))
    monkeypatch.setattr(
        driver,
        "bound",
        lambda _, reference: {"name": reference["path"]},
    )

    def original_run(tx, root, *, attempt_pins):
        assert len(live) == 10
        assert root == driver.ROOT and attempt_pins["context"] == {"frozen": True}
        calls.append(("primary", tx.directory))
        return {"scope": "SERVER_ACTUAL", "directory": str(tx.directory)}

    def snapshot(path):
        return ("changed" if mutation and calls else "original", str(path))

    def forbidden(*args, **kwargs):
        pytest.fail("Historical/recovery verifier must not be called")

    def module(name, **functions):
        return SimpleNamespace(
            __name__=name,
            __file__=str(driver.DEPLOYMENT / "src" / (name.replace(".", "/") + ".py")),
            **functions,
        )

    modules = {
        "sr_f6.pilot_audit": module(
            "sr_f6.pilot_audit",
            _verify_run=original_run,
            _snapshot=snapshot,
            _exposure=lambda runs: {"count": len(runs), "registered_exposure_met": False},
            _comparison_inventory=forbidden,
        ),
        "sr_f6.recovery": module(
            "sr_f6.recovery",
            verify_recovery=forbidden,
            verify_comparison=forbidden,
        ),
        "sr_f6.transactions": module("sr_f6.transactions"),
        "torch": SimpleNamespace(
            get_num_threads=lambda: 1, cuda=SimpleNamespace(is_initialized=lambda: False)
        ),
    }
    monkeypatch.setattr(driver, "importlib", SimpleNamespace(import_module=modules.__getitem__))

    def read_only_recovery(primary, recovered, **kwargs):
        assert len(live) == 10 and primary.identity == recovered.identity
        assert kwargs["phase_context"] == {"identity": {"frozen": True}}
        return recovery_facts()

    monkeypatch.setattr(
        driver,
        "load_recovery_consumer",
        lambda: SimpleNamespace(verify_saved_recovery=read_only_recovery),
    )
    producer = {
        "producer": {key: {} for key in ("binding_ref", "comparison_ref", "recovery_execution_ref")}
    }
    if mutation:
        with pytest.raises(PermissionError, match="Run changed"):
            driver.verify_training(
                output, {"paths": paths}, {"identity": {"frozen": True}}, {}, producer
            )
        assert len(calls) == 1
        assert all(p.name.endswith("_STARTED.json") for p in output.glob("PRIMARY-*.json"))
    else:
        facts = driver.verify_training(
            output, {"paths": paths}, {"identity": {"frozen": True}}, {}, producer
        )
        assert len(calls) == 9 and facts["paths"] == 9
        assert all(call[0] == "primary" for call in calls)
        assert driver.recovery_passed(facts["recovery"])
        assert facts["exposure"] == {"count": 9, "registered_exposure_met": False}
        assert len(facts["measurements"]) == 10
        assert all(m["elapsed_seconds"] >= 0 for m in facts["measurements"])
        assert facts["measurements"][0]["usage_after"]["maxrss_unit"] == "KiB"
        assert len(list(output.glob("PRIMARY-*.json"))) == 18
    assert live == set()


@pytest.fixture
def b_closure(driver, tmp_path, monkeypatch):
    base = tmp_path.resolve()
    root, ops = base / "root", base / "b-ops"
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "B_OPS", ops)
    contract = save(ops / "CONTRACT.json", {"bound": True})
    monkeypatch.setattr(driver, "B_CONTRACT_SHA", driver.sha(contract))
    queues = [["PRIMARY-b2-U10", "PRIMARY-b2-U11", "PRIMARY-b2-C_SHUF"], ["RECOVERY-b0-U11"]]
    plan, rows = {"tasks": []}, []
    for worker, queue in enumerate(queues):
        for task_id in queue:
            task = {"task_id": task_id, "worker": worker}
            plan["tasks"].append(task)
            task_dir = root / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task_id
            execution_dir = ops / "EXECUTIONS" / task_id
            producer = {"original": True}
            if task_id == "RECOVERY-b0-U11":
                for key in (
                    "training_ref",
                    "binding_ref",
                    "comparison_ref",
                    "recovery_execution_ref",
                ):
                    producer[key] = driver.ref(
                        save(
                            task_dir / (key + ".json"),
                            dict(
                                status="PASS",
                                scope="SERVER_ACTUAL",
                                exact=True,
                                independent_process_verified=True,
                                new_generated_responses=1024,
                            ),
                        )
                    )
            producer_ref = driver.ref(save(task_dir / "producer.json", producer))
            original = dict(
                status="TASK_COMPLETE_NOT_GATE_SIGNED",
                task=task,
                plan_ref={"path": driver.PLAN, "sha256": driver.PLAN_SHA},
                producer=producer_ref,
            )
            original_ref = driver.ref(save(task_dir / "COMPLETE.json", original))
            offload = dict(
                closed=True,
                storage_failures=[],
                cleanup_errors=[],
                live_device_bytes=0,
                live_host_bytes=0,
                live_disk_bytes=0,
            )
            storage = dict(
                status="ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS",
                contract_ref=driver.ref(contract),
                offload=offload,
            )
            storage_ref = driver.ref(save(execution_dir / "STORAGE_FACTS.json", storage))
            execution_ref = driver.ref(
                save(
                    execution_dir / "COMPLETE.json",
                    {**original, "storage_facts_ref": storage_ref, "offload": offload},
                )
            )
            rows.append(
                dict(
                    task_id=task_id,
                    task_complete_ref=original_ref,
                    execution_complete_ref=execution_ref,
                    storage_facts_ref=storage_ref,
                )
            )
        folder = ops / "LANES" / str(worker)
        save(folder / "STARTED.json", dict(job_id="237289", worker=worker, queue=queue))
        save(
            folder / "COMPLETE.json",
            dict(
                job_id="237289",
                worker=worker,
                status="QUEUE_COMPLETE_NOT_GATE_SIGNED",
                completed=[r for r in rows if r["task_id"] in queue],
            ),
        )
    outer = dict(
        job_id="237289",
        status="ROUND_B_COMPLETE_NOT_GATE_SIGNED",
        contract_ref=driver.ref(contract),
        completed=rows,
        queues=queues,
    )
    path = save(ops / "round_b_COMPLETE.json", outer)
    monkeypatch.setattr(driver, "B_COMPLETE_SHA", driver.sha(path))
    return plan, outer, path


def test_all_four_b_closures_verified(driver, b_closure):
    plan, _, _ = b_closure
    facts, producer = driver.verify_b(plan)
    assert len(facts["receipts"]) == 4
    assert producer["scope"] == "HASH_BOUND_SAVED_RECEIPTS_ONLY_NOT_RECOMPUTED"
    assert producer["receipts"]["comparison_ref"]["exact"] is True


@pytest.mark.parametrize(
    "failure",
    ["round_b_FAILURE.json", "LANES/0/FAILURE.json", "EXECUTIONS/PRIMARY-b2-U11/FAILURE.json"],
)
def test_new_b_failures_block_delivery(driver, b_closure, failure):
    plan, _, path = b_closure
    save(path.parent / failure, {"failure": True})
    with pytest.raises(PermissionError, match="failure exists"):
        driver.verify_b(plan)


def test_missing_lane_closure_blocks_delivery(driver, b_closure):
    plan, _, path = b_closure
    (path.parent / "LANES/1/COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        driver.verify_b(plan)


def test_original_task_tamper_blocks_delivery(driver, b_closure):
    plan, outer, _ = b_closure
    Path(outer["completed"][0]["task_complete_ref"]["path"]).write_text("{}")
    with pytest.raises(PermissionError, match="hash differs"):
        driver.verify_b(plan)


def recovery_facts():
    return dict(
        status="RECOVERY_FACTS_RECHECKED_NOT_GATE_SIGNED",
        comparison=dict(
            status="PASS",
            exact=True,
            differences=[],
            independent_process_verified=True,
            new_generated_responses=1024,
        ),
        comparison_history=dict(status="ALL_SAVED_FACTS_VERIFIED"),
        new_responses=0,
        new_updates=0,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAIL"),
        ("exact", False),
        ("differences", ["state"]),
        ("independent_process_verified", False),
        ("new_generated_responses", 0),
    ],
)
def test_recovery_complete_label_cannot_replace_inner_gate(driver, field, value):
    facts = recovery_facts()
    assert driver.recovery_passed(facts)
    facts["comparison"][field] = value
    assert not driver.recovery_passed(facts)


def test_unknown_comparison_history_blocks_full_delivery(driver):
    facts = recovery_facts()
    facts["comparison_history"]["status"] = "HISTORICAL_TECHNICAL_UNKNOWN"
    assert not driver.recovery_passed(facts)


def test_consumer_tamper_rejected_before_import(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "__file__", str(tmp_path / "audit.py"))
    (tmp_path / "recovery_readonly.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(PermissionError, match="consumer differs"):
        driver.load_recovery_consumer()


def test_consumer_source_pin_matches_runtime_member(driver):
    assert (
        driver.sha(Path(driver.__file__).with_name("recovery_readonly.py"))
        == driver.RECOVERY_CONSUMER_SHA
    )
