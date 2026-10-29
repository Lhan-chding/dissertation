"""Complete fixed V3 CPU lifecycle; none of these fixtures grants actual admission."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from mm_core.training import state_hash, trainable_state
from sr_f3_ce.training import fresh_optimizer
from sr_f6 import (
    evaluation,
    historical_confirmation,
    pilot,
    pilot_audit,
    pilot_contract,
    pilot_jobs,
    pilot_v3_inputs,
    pilot_v3_registration,
    recovery,
)
from sr_f6.transactions import RunTransactions
from srf6_ref.measurement import full_truth_response
from tests.sr_f6.test_evaluation import SyntheticRuntime
from tests.sr_f6.test_pilot import CPUTrainerRuntime
from tests.sr_f6.test_pilot_v3_inputs import v3_factory  # noqa: F401
from tests.sr_f6.test_registration import factory, prototype, write  # noqa: F401

V3 = pilot_contract.for_phase("PILOT_V3")
PIN_KEYS = (
    "phase_identity_sha256",
    "origin_identity_sha256",
    "pilot_spec_sha256",
    "input_overlay",
    "input_overlay_sha256",
    "prompt_version",
)


class V3CPUTrainer(CPUTrainerRuntime):
    def generate_group(self, safe, root, seeds, *, protocol, on_completion):
        def capture(row):
            row.update({key: copy.deepcopy(self.identity[key]) for key in PIN_KEYS})
            on_completion(row)

        return super().generate_group(safe, root, seeds, protocol=protocol, on_completion=capture)


class V3CPUEvaluation(SyntheticRuntime):
    def generate_group(self, model_input, run_root, seeds, *, protocol, on_completion):
        def capture(row):
            row["phase"] = "PILOT_V3"
            for key in (
                "pilot_spec_sha256",
                "input_overlay",
                "input_overlay_sha256",
                "prompt_version",
            ):
                row[key] = copy.deepcopy(self.identity[key])
            on_completion(row)

        return super().generate_group(
            model_input, run_root, seeds, protocol=protocol, on_completion=capture
        )


@pytest.fixture
def prepared(v3_factory, monkeypatch, prototype):  # noqa: F811
    project = v3_factory()
    torch.manual_seed(807)
    model = torch.nn.Linear(2, 1).float()
    parent_hash = state_hash(trainable_state(model))
    for ref in project.refs.values():
        path = project.root / ref["path"]
        report = json.loads(path.read_bytes())
        report["reference_hash"] = report["policy_hash"] = parent_hash
        write(path, report)
        ref["sha256"] = recovery._file_hash(path)
    # This fixture replaces only the external historical transport. Its explicit
    # synthetic scope never goes through production admission/access/model load.
    binding = dict(path="SYNTHETIC_CONFIRMATION.json", sha256="7" * 64)
    monkeypatch.setattr(
        historical_confirmation,
        "require_confirmation_binding",
        lambda *args, **kwargs: dict(binding_ref=binding, scope="SYNTHETIC_ONLY"),
    )
    overlay = pilot_v3_inputs.freeze_overlay(project.root, phase_context=project.context)
    frozen = pilot_v3_registration.freeze_pilot_v3_registration(
        project.root,
        phase_context=project.context,
        parent_references=project.refs,
        overlay_ref=overlay["manifest_ref"],
        confirmation_binding_ref=binding,
    )
    truth = full_truth_response(prototype["world"], prototype["query"])
    return project, frozen, model, truth


def _fresh(project, frozen, model, truth, directory, identity, task):
    runtime = V3CPUTrainer(
        copy.deepcopy(model),
        dict(
            phase="PILOT_V3",
            execution_mode="TRAINING",
            execution_origin="SYNTHETIC_TEST",
            run_root=str(project.root),
            audit_root=str(directory),
            run_id=identity["run_id"],
            parent_id=identity["parent"],
            reference_parent=identity["parent"],
            step=0,
            source_sha256=identity["source_sha256"],
            data_hash=identity["data_sha256"],
            reference_hash=identity["reference_sha256"],
            config_sha256=identity["config_sha256"],
            task_id=task,
            pilot_spec_sha256=frozen["spec_sha256"],
            **{key: identity[key] for key in PIN_KEYS if key != "pilot_spec_sha256"},
        ),
        truth,
    )
    write(directory / "runtime_identity" / (task + ".json"), runtime.identity)
    return runtime, *fresh_optimizer(runtime.model)


def test_complete_fixed_v3_cpu_lifecycle_and_original_byte_preservation(prepared):
    project, frozen, model, truth = prepared
    root, spec = project.root, frozen["spec"]
    originals = {p: p.read_bytes() for p in (root / "DATA").rglob("*") if p.is_file()}
    completed, audited = {}, []
    for item in spec["paths"]:
        checked, _, directory, identity = pilot._registered_path(
            root, project.context, frozen["spec_sha256"], item["parent"], item["arm"]
        )
        assert checked["spec"] == spec and identity["phase"] == "PILOT_V3"
        runtime, optimizer, scheduler = _fresh(
            project, frozen, model, truth, directory, identity, "primary"
        )
        with RunTransactions(directory, identity) as tx:
            result = pilot.run_synthetic(
                tx,
                runtime,
                optimizer,
                scheduler,
                experiment_root=root,
                registration_ref=identity["rollout_registration"],
            )
            audited.append(pilot_audit.verify_synthetic_run(tx, root=root))
        completed[(item["parent"], item["arm"])] = (directory, identity, result)
        assert result["status"] == "SYNTHETIC_ONLY" and runtime.calls == 1536
        assert result["phase"] == "PILOT_V3"
        assert len(runtime.received) == 12 * 16
    assert sum(value[2]["segment_updates"] for value in completed.values()) == 108
    assert sum(value[2]["segment_new_model_calls"] for value in completed.values()) == 13824
    assert sum(value["responses"] for value in audited) == 13824
    directory, identity, primary = completed[("b0", "U11")]
    common = dict(phase_context=project.context, pilot_spec_sha256=frozen["spec_sha256"])
    with RunTransactions(directory, identity) as tx:
        bound = recovery.prepare_recovery(
            tx,
            experiment_root=root,
            primary_execution_ref=primary["execution"],
            source_commit_sha256=tx.published()[4]["commit_sha256"],
            **common,
        )
    lane = Path(bound["directory"])
    assert lane == root / V3.recovery_directory
    assert bound["source_step"] == 4 and not (lane / "rollouts/step-005").exists()
    runtime, optimizer, scheduler = _fresh(
        project, frozen, model, truth, lane, identity, "recovery"
    )
    with RunTransactions(lane, identity) as tx:
        result = pilot._drive(
            tx,
            runtime,
            optimizer,
            scheduler,
            experiment_root=root,
            registration_ref=identity["rollout_registration"],
            access_factory=nullcontext,
            journal_factory=recovery.ExecutionJournal.synthetic,
            scope="SYNTHETIC_ONLY",
            storage_lane="RECOVERY",
            recovery_binding_ref=bound["binding_ref"],
        )
    assert result["segment_updates"] == 8 and runtime.calls == 1024
    with (
        RunTransactions(directory, identity) as primary_tx,
        RunTransactions(lane, identity) as rec_tx,
    ):
        compared = recovery.compare_recovery(
            primary_tx,
            rec_tx,
            root=root,
            binding_ref=bound["binding_ref"],
            primary_execution_ref=primary["execution"],
            recovery_execution_ref=result["execution"],
            **common,
        )
    assert compared["scope"] == "SYNTHETIC_ONLY" and compared["exact"] is True
    assert compared["status"] == "SYNTHETIC_ONLY"
    plan = evaluation._plan(spec)
    public, tasks = evaluation._inputs(root, plan, spec=spec, phase_context=project.context)
    admitted = dict(pilot_spec=spec, phase_context=project.context)
    reports = []
    for endpoint in plan["endpoints"]:
        with evaluation._source(root, spec, endpoint) as source:
            ident = evaluation._identity(
                root, admitted, endpoint, source, frozen["spec_sha256"], {}
            )
            ident["execution_origin"] = "SYNTHETIC_TEST"
            runtime = V3CPUEvaluation(truth)
            runtime.identity = ident
            with evaluation.EvaluationLedger(root, ident, plan, public, tasks) as ledger:
                report = evaluation._execute(ledger, runtime)
                assert runtime.calls == 96 and report["status"] == "SYNTHETIC_ONLY"
                reports.append(report)
    summary = evaluation.summarize_reports(reports)
    assert summary["checkpoints"] == 12 and summary["responses"] == 2304
    assert summary["status"] == "SYNTHETIC_ONLY" and summary["training_authorized"] is False
    assert originals == {path: path.read_bytes() for path in originals}
    assert not (root / "PHASE_RUNS/PILOT_V1").exists()


def test_old_admission_cannot_start_v3_or_unknown_phase(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        pilot_contract,
        "require_admission",
        lambda *args, **kwargs: (
            called.append(kwargs)
            or dict(
                status="VERIFIED_PILOT_ADMISSION",
                scope="PILOT_V1_REGISTERED_ONLY",
                generation_authorized=True,
                pilot_spec=dict(phase="PILOT_V1"),
            )
        ),
    )
    pins = dict(
        phase="PILOT_V3",
        phase_identity_sha256="a" * 64,
        pilot_spec_sha256="b" * 64,
        admission_ref={},
    )
    with pytest.raises(PermissionError, match="Actual PILOT"):
        pilot_jobs.build_plan(tmp_path, **pins)
    with pytest.raises(PermissionError, match="Actual PILOT"):
        evaluation._admitted(tmp_path, "a" * 64, "b" * 64, {}, phase="PILOT_V3")
    with pytest.raises(PermissionError, match="explicit registered"):
        pilot_jobs.build_plan(tmp_path, **{**pins, "phase": "FORMAL_V1"})
    assert len(called) == 2 and all(value["phase"] == "PILOT_V3" for value in called)


@pytest.mark.parametrize(
    "key", ["pilot_spec_sha256", "input_overlay", "input_overlay_sha256", "prompt_version"]
)
def test_v3_raw_and_loader_pins_are_checked(key):
    identity = dict(
        phase="PILOT_V3",
        pilot_spec_sha256="a" * 64,
        input_overlay={"path": "registered", "sha256": "b" * 64},
        input_overlay_sha256="b" * 64,
        prompt_version="srf6-interface-v3",
    )
    changed = {**identity, key: "changed"}
    with pytest.raises(PermissionError):
        evaluation._overlay_identity(changed, identity)
    with pytest.raises(PermissionError):
        recovery._overlay_identity(
            changed, identity, pilot_spec_sha256=identity["pilot_spec_sha256"]
        )


def test_driver_and_evaluation_loaders_receive_v3_spec_pin(prepared, monkeypatch):
    from sr_f6 import runtime as native

    project, frozen, _, _ = prepared
    captured = []

    def stop_before_model(*args, **kwargs):
        captured.append(kwargs)
        raise RuntimeError("SYNTHETIC stop before native model loading")

    monkeypatch.setattr(native, "load", stop_before_model)
    monkeypatch.setattr(pilot, "_context", lambda *args, **kwargs: project.context)
    admitted = dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope=V3.admission_scope,
        generation_authorized=True,
        pilot_spec=frozen["spec"],
        phase_context=project.context,
    )
    monkeypatch.setattr(pilot_contract, "require_admission", lambda *args, **kwargs: admitted)
    pins = dict(
        phase="PILOT_V3",
        phase_identity_sha256=project.context["identity_sha256"],
        pilot_spec_sha256=frozen["spec_sha256"],
        admission_ref=dict(path=V3.admission_path, sha256="f" * 64),
    )
    with pytest.raises(RuntimeError, match="before native"):
        pilot.run(project.root, "b0", "U11", "synthetic-no-load", **pins)
    with pytest.raises(RuntimeError, match="before native"):
        evaluation.run_endpoint(
            project.root, endpoint_id="b0/parent", task_id="synthetic-no-load", **pins
        )
    assert [item["mode"] for item in captured] == ["TRAINING", "EVALUATION"]
    assert all(
        item["phase"] == "PILOT_V3" and item["pilot_spec_sha256"] == frozen["spec_sha256"]
        for item in captured
    )
    assert all("PHASE_RUNS/PILOT_V3/" in str(item["audit_root"]) for item in captured)
    assert not (project.root / "PHASE_RUNS/PILOT_V1").exists()


def test_v3_jobs_namespace_budget_and_child_cli_pins(prepared, monkeypatch):
    project, frozen, _, _ = prepared
    admitted = dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope=V3.admission_scope,
        pilot_spec=frozen["spec"],
        phase_context=project.context,
        pilot_spec_ref={"sha256": frozen["spec_sha256"]},
    )
    monkeypatch.setattr(pilot_contract, "require_admission", lambda *args, **kwargs: admitted)
    pins = dict(
        phase="PILOT_V3",
        phase_identity_sha256=project.context["identity_sha256"],
        pilot_spec_sha256=frozen["spec_sha256"],
        admission_ref=dict(path=V3.admission_path, sha256="f" * 64),
    )
    ref = pilot_jobs.freeze_plan(project.root, **pins)
    plan, checked = pilot_jobs.verify_plan(project.root, ref)
    assert ref["path"] == V3.jobs_directory + "/PLAN.json"
    assert plan["phase"] == "PILOT_V3" and plan["pins"] == pins
    assert len(plan["tasks"]) == 22
    assert [[len(queue) for queue in plan["queues"][r]] for r in ("A", "B", "EVAL")] == [
        [3, 3],
        [3, 1],
        [6, 6],
    ]
    assert plan["budgets"] == frozen["spec"]["budgets"]
    captured = []

    def stop(command, log):
        captured.append(command)
        raise RuntimeError("SYNTHETIC no subprocess")

    monkeypatch.setattr(pilot_jobs, "_invoke", stop)
    for kind in ("PRIMARY", "EVALUATION"):
        task = next(task for task in plan["tasks"] if task["kind"] == kind)
        with pytest.raises(RuntimeError, match="no subprocess"):
            pilot_jobs._execute(project.root, ref, plan, checked, task, "fixture", project.root)
        assert pilot_jobs._result_path(project.root, task, phase="PILOT_V3").is_relative_to(
            project.root / V3.jobs_directory
        )
    assert all(command[command.index("--phase") + 1] == "PILOT_V3" for command in captured)
    assert all(
        command[command.index("--pilot-spec-sha256") + 1] == frozen["spec_sha256"]
        for command in captured
    )
    wrong = {**ref, "path": pilot_jobs.PLAN}
    with pytest.raises((FileNotFoundError, PermissionError)):
        pilot_jobs.verify_plan(project.root, wrong)
    assert not (project.root / "PHASE_RUNS/PILOT_V1").exists()


def test_v3_workload_summary_waits_for_complete_original_fact_audit(prepared, monkeypatch):
    project, frozen, _, _ = prepared
    admitted = dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope=V3.admission_scope,
        pilot_spec=frozen["spec"],
        phase_context=project.context,
        pilot_spec_ref={"sha256": frozen["spec_sha256"]},
    )
    monkeypatch.setattr(pilot_contract, "require_admission", lambda *args, **kwargs: admitted)
    pins = dict(
        phase="PILOT_V3",
        phase_identity_sha256=project.context["identity_sha256"],
        pilot_spec_sha256=frozen["spec_sha256"],
        admission_ref=dict(path=V3.admission_path, sha256="f" * 64),
    )
    ref = pilot_jobs.freeze_plan(project.root, **pins)
    plan, _ = pilot_jobs.verify_plan(project.root, ref)
    producer = dict(binding_ref={"fixture": "binding"}, comparison_ref={"fixture": "comparison"})
    producer_path = project.root / "SYNTHETIC_PRODUCER.json"
    write(producer_path, producer)
    producer_ref = pilot_jobs._ref(project.root, producer_path)
    for task in plan["tasks"]:
        write(
            pilot_jobs._result_path(project.root, task, phase="PILOT_V3"),
            dict(producer=producer_ref, scope="SYNTHETIC_ONLY"),
        )
    monkeypatch.setattr(
        pilot_jobs, "_verify_task", lambda *args, **kwargs: dict(producer=producer_ref)
    )
    monkeypatch.setattr(pilot_jobs, "_comparison", lambda *args: ())
    monkeypatch.setattr(
        evaluation,
        "aggregate",
        lambda root, **kwargs: write(
            root / V3.evaluation_directory / "SUMMARY.json", dict(scope="SYNTHETIC_ONLY")
        ),
    )
    seen = []

    def audit(root, **kwargs):
        seen.append(kwargs)
        return dict(status="INSUFFICIENT_REGISTERED_EXPOSURE", scope="SYNTHETIC_ONLY")

    monkeypatch.setattr(pilot_audit, "audit", audit)
    with pytest.raises(PermissionError, match="Complete original pilot fact audit"):
        pilot_jobs.aggregate(project.root, plan_ref=ref)
    assert len(seen) == 1 and seen[0]["phase"] == "PILOT_V3"
    assert seen[0]["pilot_spec_sha256"] == frozen["spec_sha256"]
    assert seen[0]["evaluation_summary_ref"]["path"] == V3.evaluation_directory + "/SUMMARY.json"
    assert not (project.root / V3.jobs_directory / "SUMMARY.json").exists()
