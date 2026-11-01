"""Original primary audit equivalence and narrow old-OOM proof regression."""

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import pilot_audit
from sr_f6.transactions import RunTransactions

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


consumer = load(
    "post_b_primary_consumer", ROOT / "scripts/sr_f6_post_b_delivery_v2/primary_readonly.py"
)
fixture = load("post_b_primary_audit_fixture", Path(__file__).with_name("test_pilot_audit.py"))
prototype, complete = fixture.prototype, fixture.complete


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("name", ["_verify_run", "verify_execution_inventory"])
def test_static_original_assertions_and_all_successful_event_logic_are_unchanged(name):
    original = ast.parse(inspect.getsource(getattr(pilot_audit, name))).body[0]
    new_name = "_verify_run_readonly" if name == "_verify_run" else name
    actual = ast.parse(inspect.getsource(getattr(consumer, new_name))).body[0]

    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = "same"
            kept = [
                (arg, default)
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
                if arg.arg != "historical_oom_proof_ref"
            ]
            node.args.kwonlyargs = [arg for arg, _ in kept]
            node.args.kw_defaults = [value for _, value in kept]
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "frozen":
                return ast.Name(id=node.attr, ctx=ast.Load())
            return self.generic_visit(node)

        def visit_Call(self, node):
            node.keywords = [v for v in node.keywords if v.arg != "historical_oom_proof_ref"]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if any(
                isinstance(t, ast.Name) and t.id in {"failed_numeric_events", "partial_diagnostics"}
                for t in node.targets
            ):
                return None
            if name == "verify_execution_inventory" and any(
                isinstance(t, ast.Name) and t.id == "result" for t in node.targets
            ):
                return ast.Return(value=self.visit(node.value))
            return self.generic_visit(node)

        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == "failed_numeric_events":
                return None
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_reconcile_registered_oom"
                for n in ast.walk(node)
            ):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if any(
                isinstance(n, ast.Name) and n.id == "partial_diagnostics" for n in ast.walk(node)
            ):
                return None
            return self.generic_visit(node)

        def visit_Return(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "result":
                return None
            return self.generic_visit(node)

    assert ast.dump(Normalize().visit(actual), include_attributes=False) == ast.dump(
        Normalize().visit(original), include_attributes=False
    )


def test_actual_tiny_torch_primary_core_matches_original_and_writes_nothing(complete):
    with RunTransactions(complete["run"], complete["identity"]) as tx:
        expected = pilot_audit._verify_run(tx, complete["experiment"], synthetic=True)
        before = snapshot(complete["experiment"])
        actual = consumer._verify_run_readonly(tx, complete["experiment"], synthetic=True)
        assert actual == expected
        assert snapshot(complete["experiment"]) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_call",
        "missing_token",
        "unknown_kind",
        "orphan_task",
        "missing_end",
        "raw_tamper",
        "audit_failure",
    ],
)
def test_original_evidence_mutations_are_still_rejected(complete, mutation):
    run = complete["run"]
    account = run / "accounting/original.jsonl"
    if mutation in ("extra_call", "missing_token", "unknown_kind"):
        rows = account.read_text().splitlines()
        if mutation == "extra_call":
            rows.append(rows[0])
        elif mutation == "missing_token":
            rows.pop(1)
        else:
            row = json.loads(rows[0])
            row["kind"] = "unknown"
            rows.append(json.dumps(row))
        account.write_text("\n".join(rows) + "\n")
    elif mutation == "orphan_task":
        (run / "accounting/orphan.jsonl").write_text("")
    elif mutation == "missing_end":
        (run / "EXECUTIONS/original/END.json").unlink()
    elif mutation == "raw_tamper":
        raw = run / "rollouts/step-001/groups/00/raw.jsonl"
        raw.write_text(raw.read_text().replace('"full_text":', '"fake":', 1))
    else:
        with (run / "NUMERICAL_TOKEN_AUDIT.jsonl").open("a") as stream:
            stream.write('{"status":"UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"}\n')
    with RunTransactions(run, complete["identity"]) as tx:
        with pytest.raises((PermissionError, KeyError, ValueError)):
            pilot_audit._verify_run(tx, complete["experiment"], synthetic=True)
        with pytest.raises((PermissionError, KeyError, ValueError)):
            consumer._verify_run_readonly(tx, complete["experiment"], synthetic=True)


@pytest.fixture
def oom(tmp_path, monkeypatch):
    """Synthetic proof tree exercises assertions, never a GPU/public admission."""
    project = tmp_path / "project"
    project.mkdir()
    root = project / "experiment"
    root.mkdir()
    run = root / consumer.RUN
    run.mkdir(parents=True)
    monkeypatch.setattr(consumer, "PROJECT", project)
    monkeypatch.setattr(consumer, "EXPERIMENT", root)
    monkeypatch.setattr(consumer.phases, "PROJECT", project)
    old, successor = consumer.OLD_TASK, "separate-restored-process"

    def write(path, value, raw=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = value if raw else json.dumps(value, sort_keys=True).encode()
        path.write_bytes(payload)
        return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())

    def relative(ref):
        return dict(ref, path=str(Path(ref["path"]).relative_to(root)))

    identity = dict(
        phase="PILOT_V3", parent="b1", arm="U10", lane="PRIMARY", repeat=0, run_id="fixture-run"
    )
    marker0 = dict(step=0, policy_sha256="a" * 64, commit_sha256="b" * 64)
    marker1 = dict(
        step=1,
        policy_before_sha256="a" * 64,
        commit_sha256="c" * 64,
        references={"rollout": {"path": "rollouts/step-001/rollout.json", "sha256": "d" * 64}},
    )
    step0 = write(run / "checkpoints/step-000/COMMIT.json", marker0)
    step1 = write(run / "checkpoints/step-001/COMMIT.json", marker1)
    ready = write(
        run / "rollouts/step-001/READY.json",
        dict(
            status="ROLLOUT_READY",
            references=marker1["references"],
            sampler_state={"rollout_cursor": {"step": 1, "groups": 16, "rows": 128}},
        ),
    )
    rows = [
        dict(
            row_id=f"row-{i}",
            qid=f"q-{i // 8}",
            tokens=[i + 1],
            sampler_logprobs=[-0.1],
            token_path_sha256="e" * 64,
            policy_hash="a" * 64,
            prompt_tensor_hash="f" * 64,
            source_sha256="0" * 64,
            task_id=old,
        )
        for i in range(128)
    ]
    traces = [
        dict(
            row_id=row["row_id"],
            qid=row["qid"],
            tokens=row["tokens"],
            sampler_logprobs=[-0.1],
            train_logprobs=[-0.1],
            reference_logprobs=[-0.1],
            token_path_sha256=row["token_path_sha256"],
        )
        for row in rows[:27]
    ]
    event = dict(
        status="UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP",
        task_id=old,
        committed_source_step=0,
        optimizer_started=False,
        parameter_state_changed=False,
        requires_checkpoint_restore=False,
        error_type="OutOfMemoryError",
        error="CUDA out of memory fixture",
        run_id=identity["run_id"],
        arm="U10",
        audit_root=str(run),
        policy_before_sha256=marker0["policy_sha256"],
        original_trajectories=[{k: v for k, v in row.items() if k != "task_id"} for row in rows],
        probability_vectors=traces,
        diagnostics=pilot_audit.training._numerical_audit(traces),
    )
    line = json.dumps(event, sort_keys=True).encode() + b"\n"
    monkeypatch.setattr(consumer, "AUDIT_BYTES", len(line))
    monkeypatch.setattr(consumer, "AUDIT_SHA256", hashlib.sha256(line).hexdigest())
    audit_ref = write(project / "original-audit.jsonl", line, True)
    write(run / "NUMERICAL_TOKEN_AUDIT.jsonl", line, True)
    original_files = [step0, ready] + [
        write(run / f"checkpoints/step-000/saved-{i}", b"original dependency", True)
        for i in range(86)
    ]
    trainer = write(
        run / "TRAINER_ATTEMPTS" / old / "FAILURE.json",
        dict(
            status="FAILED_PRESERVED",
            task_id=old,
            error_type="OutOfMemoryError",
            error=event["error"],
            sampling_retry_authorized=False,
            requires_fresh_process_restore=True,
        ),
    )
    task_fail = write(
        root / "PHASE_RUNS/PILOT_V3/JOBS/TASKS/PRIMARY-b1-U10/ATTEMPTS" / old / "FAILURE.json",
        dict(status="FAILED_PRESERVED"),
    )
    worker_fail = write(
        root / "PHASE_RUNS/PILOT_V3/JOBS/WORKERS/A/1/ATTEMPTS/A-233996/FAILURE.json",
        dict(status="FAILED_PRESERVED"),
    )
    begin = dict(
        start_step=0,
        start_commit_sha256=marker0["commit_sha256"],
        process_identity={"kind": "LINUX_PROC", "pid": 1},
        restored=False,
    )
    restored = dict(begin, process_identity={"kind": "LINUX_PROC", "pid": 2}, restored=True)
    bref = write(run / "EXECUTIONS" / old / "BEGIN.json", begin)
    cref = write(run / "EXECUTIONS" / old / "COMMITS.jsonl", b"", True)
    fref = write(
        run / "EXECUTIONS" / old / "FAILURE.json",
        dict(
            status="INCOMPLETE_SEGMENT",
            scope="SERVER_ACTUAL",
            task_id=old,
            error_type="OutOfMemoryError",
            error=event["error"],
            begin_reference=relative(bref),
        ),
    )
    ledger = [
        dict(
            kind="extra_forward_sequences",
            count=1,
            metadata=dict(purpose=purpose, completion_tokens=1),
        )
        for _ in range(27)
        for purpose in ("training_reference", "training_policy")
    ]
    aref = write(
        run / "accounting" / f"{old}.jsonl",
        b"".join(json.dumps(x).encode() + b"\n" for x in ledger),
        True,
    )
    rref = write(run / "runtime_identity" / f"{old}.json", dict(fixture_only=True))
    write(run / "EXECUTIONS" / successor / "BEGIN.json", restored)
    write(
        run / "EXECUTIONS" / successor / "COMMITS.jsonl",
        json.dumps(
            dict(
                step=1,
                new_model_calls=0,
                reused_complete_groups=16,
                commit_sha256=marker1["commit_sha256"],
            )
        ).encode()
        + b"\n",
        True,
    )
    (run / "PREUPDATE_EVIDENCE/step-001").mkdir(parents=True)
    trace = write(
        root / "PHASE_RUNS/PILOT_V3/JOBS/TASKS/PRIMARY-b1-U10/ATTEMPTS" / old / "process.log",
        b'File "training.py", line 537\n(result["loss"] / 128).backward()\nOutOfMemoryError',
        True,
    )
    monkeypatch.setattr(consumer, "TRACE_SHA256", trace["sha256"])
    contract = dict(
        schema="srf6-memory-full-contract-v7",
        experiment_root=str(root),
        original_execution_id=old,
        original_audit_bytes=len(line),
        original_audit_path=str(run / "NUMERICAL_TOKEN_AUDIT.jsonl"),
        original_audit_ref=audit_ref,
        original_files=original_files,
        original_ready_ref=ready,
        step0_commit_ref=step0,
        original_failure_ref=trainer,
        original_task_failure_ref=task_fail,
        original_worker_failure_ref=worker_fail,
    )
    v7 = write(project / "v7/CONTRACT.json", contract)
    monkeypatch.setattr(consumer, "V7", v7)
    source = write(project / "v7/source.py", b"# preserved verifier", True)
    evidence = write(project / "v7/EVIDENCE.json", dict(fixture_only=True))
    result = write(
        project / "v7/RESULT.json",
        dict(
            status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
            contract_sha256=v7["sha256"],
            evidence_ref=evidence,
            verifier_source_ref=source,
            tensors_ref={"path": str(project / "must-not-read-TENSORS.pt"), "sha256": "1" * 64},
        ),
    )
    cold = write(
        project / "v7/COLD.json",
        dict(
            status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
            contract_sha256=v7["sha256"],
            validation_ref=result,
            zero_new_samples=True,
            zero_optimizer_updates=True,
            generated_responses=0,
            optimizer_updates=0,
            original_ready_ref=ready,
            verifier_source_ref=source,
        ),
    )
    complete = write(
        project / "v7/COMPLETE.json",
        dict(
            status="COMPLETED_FULL_VALIDATION_NO_TRAINING",
            job_id="235864",
            contract_ref=v7,
            result_ref=result,
            cold_verification_ref=cold,
        ),
    )
    v8 = write(
        project / "v8/CONTRACT.json",
        dict(
            schema="srf6-memory-resume-contract-v8",
            full_contract_ref=v7,
            full_result_ref=result,
            full_complete_ref=complete,
            full_cold_ref=cold,
        ),
    )
    monkeypatch.setattr(consumer, "V8", v8)
    proof = write(
        project / "proof.json",
        dict(
            schema="srf6-post-b-registered-oom-proof-v1",
            memory_contract_ref=v7,
            resume_contract_ref=v8,
            old_execution_refs=dict(
                begin_ref=bref,
                commits_ref=cref,
                failure_ref=fref,
                accounting_ref=aref,
                runtime_ref=rref,
            ),
            traceback_ref=trace,
        ),
    )
    tx = SimpleNamespace(directory=run, identity=identity, checkpoints=run / "checkpoints")
    kwargs = dict(
        root=root,
        event=event,
        line=line,
        offset=0,
        markers=[marker0, marker1],
        begins={old: begin, successor: restored},
        fact_by_task={
            old: dict(
                journal_steps=[],
                counts=dict(
                    completion_attempts=128,
                    generated_tokens=128,
                    extra_forward_sequences=54,
                    diagnostic_gradient_calls=48,
                ),
            )
        },
        rows_by_step={1: rows},
        evidence_tasks={1: successor},
        bound_slices=set(),
        proof_ref=proof,
        synthetic=False,
    )
    return SimpleNamespace(
        tx=tx,
        kwargs=kwargs,
        project=project,
        proof=proof,
        contract=contract,
        write=write,
        successor=successor,
        step1=step1,
    )


def test_fixed_oom_proof_retains_partial_diagnostics_without_rerunning_validation(oom):
    before = snapshot(oom.project)
    result, diagnostics = consumer._reconcile_registered_oom(oom.tx, **oom.kwargs)
    assert result["optimizer_updates"] == result["regenerated_responses"] == 0
    assert result["complete_numerical_pass"] is False
    assert result["partial_forward_count"] == 27 and result["partial_diagnostic_calls"] == 48
    assert len(diagnostics) == 16 and sum(diagnostics.values()) == 16
    assert result["v7_tensors_reverified"] is False
    assert result["old_optimizer_tensor_state_observed"] is False
    assert snapshot(oom.project) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "owner",
        "step",
        "optimizer",
        "parameter",
        "restore_required",
        "different_run",
        "wrong_offset",
        "wrong_bytes",
        "synthetic",
        "old_commit",
        "old_wrapper",
        "not_restored",
        "same_process",
        "new_calls",
        "different_raw",
        "partial_count",
        "partial_stats",
        "partial_forward_accounting",
        "missing_dependency",
        "source_hash",
        "missing_proof",
    ],
)
def test_registered_oom_exception_rejects_any_scope_or_proof_mutation(oom, mutation):
    k = oom.kwargs
    if mutation == "owner":
        k["event"]["task_id"] = "other"
    elif mutation == "step":
        k["event"]["committed_source_step"] = 1
    elif mutation == "optimizer":
        k["event"]["optimizer_started"] = True
    elif mutation == "parameter":
        k["event"]["parameter_state_changed"] = True
    elif mutation == "restore_required":
        k["event"]["requires_checkpoint_restore"] = True
    elif mutation == "different_run":
        oom.tx.identity["parent"] = "b0"
    elif mutation == "wrong_offset":
        k["offset"] = 1
    elif mutation == "wrong_bytes":
        k["line"] += b" "
    elif mutation == "synthetic":
        k["synthetic"] = True
    elif mutation == "old_commit":
        k["fact_by_task"][consumer.OLD_TASK]["journal_steps"] = [1]
    elif mutation == "old_wrapper":
        (oom.tx.directory / "PREUPDATE_EVIDENCE/step-001" / f"{consumer.OLD_TASK}.json").write_text(
            "{}"
        )
    elif mutation == "not_restored":
        k["begins"][oom.successor]["restored"] = False
    elif mutation == "same_process":
        k["begins"][oom.successor]["process_identity"] = k["begins"][consumer.OLD_TASK][
            "process_identity"
        ]
    elif mutation == "new_calls":
        path = oom.tx.directory / "EXECUTIONS" / oom.successor / "COMMITS.jsonl"
        v = json.loads(path.read_text())
        v["new_model_calls"] = 128
        path.write_text(json.dumps(v) + "\n")
    elif mutation == "different_raw":
        k["rows_by_step"][1][0]["tokens"] = [999]
    elif mutation == "partial_count":
        k["event"]["probability_vectors"] = k["event"]["probability_vectors"][:26]
    elif mutation == "partial_stats":
        k["event"]["diagnostics"]["status"] = "NOT_FINITE"
    elif mutation == "partial_forward_accounting":
        k["fact_by_task"][consumer.OLD_TASK]["counts"]["extra_forward_sequences"] = 55
    elif mutation == "missing_dependency":
        Path(oom.contract["original_files"][-1]["path"]).unlink()
    elif mutation == "source_hash":
        (oom.project / "v7/source.py").write_text("# changed")
    else:
        k["proof_ref"] = None
    with pytest.raises((PermissionError, KeyError, FileNotFoundError)):
        consumer._reconcile_registered_oom(oom.tx, **k)


@pytest.mark.parametrize(
    "failures", [[], [{"task_id": "other"}], [{"task_id": consumer.OLD_TASK}] * 2]
)
def test_public_target_requires_preserved_oom_exactly_once(monkeypatch, failures):
    tx = SimpleNamespace(identity=dict(phase="PILOT_V3", parent="b1", arm="U10"))
    monkeypatch.setattr(
        consumer,
        "_verify_run_readonly",
        lambda *a, **k: {"inventory": {"failed_numeric_events": failures}},
    )
    with pytest.raises(PermissionError, match="must remain present exactly once"):
        consumer.verify_primary_run(tx, consumer.EXPERIMENT)
