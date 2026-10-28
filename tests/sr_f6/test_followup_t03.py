"""Synthetic-only T03 fixtures: no test authorizes a model call or scientific PASS."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash
from sr_f6.common import digest, file_hash
from sr_f6.replay import _RUNTIME_ACCESS
from sr_f6_followup import source, t03
from sr_f6_followup import t03_access as access
from sr_f6_followup import t03_dispatch as dispatch
from sr_f6_followup import t03_runtime as runtime

_spec = importlib.util.spec_from_file_location(
    "t03_cpu_fixture", Path(__file__).with_name("test_numerical_replay.py")
)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "followup"
    root.mkdir()
    producer = tmp_path / "producer"
    producer.mkdir()
    code = Path(t03.kernel.__file__).resolve().parents[2]
    consumer = dict(
        deployment=str(code),
        revision="synthetic",
        source_manifest_sha256="c" * 64,
        registration_sha256="d" * 64,
    )
    qid = "DEV_SCHEMA_CONFIRM_SYNTHETIC_0000_P0_v0"
    records, raws, policies = [], {}, {}
    tiny = {}
    for parent in t03.PARENTS:
        tiny[parent], rows, raw, reference = fixture.prepare(tmp_path / parent)
        policy = state_hash(reference)
        policies[parent] = policy
        for a, b in zip(rows, raw, strict=True):
            shared = dict(
                parent=parent,
                qid=qid,
                row_id=f"{parent}/{qid}/{a['draw']}",
                phase="SCHEMA_CONFIRM_V3",
                policy_tensor_hash=policy,
                source_sha256="a" * 64,
                registration_sha256="b" * 64,
                protocol_revision="v2",
                sr_f6_protocol_version="structured_v4-srf6-1.0",
            )
            a.update(shared)
            b.update(shared)
            records.append(a)
            raws[b["row_id"]] = b
    selection = dict(
        qid=qid,
        parents=list(t03.PARENTS),
        draws=list(range(8)),
        sequences=24,
        new_generated_responses=0,
        optimizer_updates=0,
    )
    adapter = tmp_path / "adapter"
    write(adapter / "adapter_config.json", {})
    config = dict(
        master_seed=1,
        training=dict(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=1536),
        numerics={k: "float32" for k in ("model_dtype", "input_float_dtype", "probability_dtype")},
        model=dict(
            model_path=str(tmp_path / "model"), parents={p: str(adapter) for p in t03.PARENTS}
        ),
    )
    producer_id = dict(
        deployment=str(code), revision="frozen-synthetic", registration_sha256="b" * 64
    )
    context = dict(
        root=root,
        consumer=consumer,
        registration=dict(producer=producer_id),
        producer_context=dict(
            root=producer,
            config=config,
            registration_sha256="b" * 64,
            registration=dict(
                source_manifest_sha256="a" * 64,
                data_sha256="e" * 64,
                parent_content_hashes={p: "f" * 64 for p in t03.PARENTS},
            ),
            public=[dict(qid=qid)],
            spec=dict(followup_requirements=dict(numerical_replay_selection=selection)),
        ),
        records=records,
        raw_by_row=raws,
        evidence=dict(parent_tensor_hashes=policies),
        support_binding_ref=dict(path=str(root / "SUPPORT_BINDING.json"), sha256="1" * 64),
        review=dict(
            package_ref=dict(path=str(root / "REVIEW/PACKAGE.json"), sha256="2" * 64),
            audit_ref=dict(path=str(root / "REVIEW/AUDIT.json"), sha256="3" * 64),
        ),
    )
    calls = []
    monkeypatch.setattr(
        source,
        "require_support",
        lambda root, purpose: calls.append(purpose) or copy.deepcopy(context),
    )
    t03.freeze_spec(root)
    spec, ref = t03.verified_spec(root, context)
    shared = dict(consumer_registration_sha256=consumer["registration_sha256"], spec_ref=ref)
    write(
        root / "T03/SUBMISSIONS/t03.attempt.json",
        dict(
            status="SUBMISSION_INTENT_NOT_COMPLETION",
            argv=dispatch.argv_for(root, context),
            **shared,
        ),
    )
    write(
        root / "T03/SUBMISSIONS/t03.result.json",
        dict(status="SUBMITTED", job_id="123", returncode=0, stdout="123\n", **shared),
    )
    return SimpleNamespace(
        root=root,
        context=context,
        spec=spec,
        ref=ref,
        tiny=tiny,
        calls=calls,
        monkeypatch=monkeypatch,
    )


def loaded_fixture(setup, parent="b0"):
    r = setup.tiny[parent]
    task = f"LH-SRF6-T03-123-{parent}"
    r.identity.update(
        phase=runtime.PHASE,
        execution_mode="REPLAY",
        audit_root=str(setup.root / "T03/RUNS" / task),
        task_id=task,
        parent_id=parent,
        policy_hash=setup.context["evidence"]["parent_tensor_hashes"][parent],
        reference_hash=setup.context["evidence"]["parent_tensor_hashes"][parent],
        run_root=str(setup.root),
    )
    r._reference = r.reference
    r._adapter_names = tuple(r.reference)
    rows, raw, _ = t03.selected_parent(setup.context, parent)
    runtime._LOADED[r] = dict(
        model=r.model,
        identity=copy.deepcopy(r.identity),
        spec_sha256=digest(setup.spec),
        spec_ref=copy.deepcopy(setup.ref),
        records_sha256=digest(rows),
        raw_sha256=digest(raw),
        consumer=copy.deepcopy(setup.context["consumer"]),
        producer=copy.deepcopy(setup.context["registration"]["producer"]),
    )
    return r


def test_exact_original_selection_and_distinct_identities(setup):
    assert set(setup.spec["groups"]) == set(t03.PARENTS)
    assert sum(len(g["row_ids"]) for g in setup.spec["groups"].values()) == 24
    assert setup.spec["context"]["consumer"] != setup.spec["context"]["producer"]
    bad = copy.deepcopy(setup.context)
    bad["producer_context"]["spec"]["followup_requirements"]["numerical_replay_selection"][
        "qid"
    ] = "changed"
    with pytest.raises(PermissionError):
        t03.build_spec(bad)
    bad = copy.deepcopy(setup.context)
    bad["raw_by_row"][bad["records"][0]["row_id"]]["tokens"][0] += 1
    with pytest.raises(PermissionError):
        t03.build_spec(bad)


@pytest.mark.parametrize("target", ["spec", "context"])
def test_session_cannot_replace_loaded_original_authority(setup, target):
    r = loaded_fixture(setup)
    context, spec = copy.deepcopy(setup.context), copy.deepcopy(setup.spec)
    if target == "spec":
        spec["rules"]["atol"] = 1
    else:
        first = context["records"][0]
        first["tokens"][0] += 1
        context["raw_by_row"][first["row_id"]]["tokens"][0] += 1
    with pytest.raises(PermissionError), access.open_replay(r, context, spec, setup.ref):
        pass
    assert not (Path(r.identity["audit_root"]) / "T03").exists()


def test_authenticated_session_exact_budget_and_no_generation(setup):
    r = loaded_fixture(setup)
    rows, _, _ = t03.selected_parent(setup.context, "b0")
    with access.open_replay(r, setup.context, setup.spec, setup.ref) as session:
        session.bind_prepared(r.prepared, access=_RUNTIME_ACCESS)
        for row in rows:
            for purpose, grad in (("t03_reference", False), ("t03_native", True)):
                with (
                    session.forward(
                        [dict(row, prepared=r.prepared)],
                        purpose=purpose,
                        grad=grad,
                        access=_RUNTIME_ACCESS,
                    ),
                    session.native_call(
                        r.prepared["inputs"],
                        completion_start=3,
                        lengths=[len(row["tokens"])],
                        purpose=purpose,
                        grad=grad,
                        access=_RUNTIME_ACCESS,
                    ),
                ):
                    session.account(
                        "extra_forward_sequences",
                        1,
                        dict(purpose=purpose, completion_tokens=len(row["tokens"])),
                    )
        with (
            pytest.raises(PermissionError),
            session.forward([rows[0]], purpose="t03_reference", grad=False, access=_RUNTIME_ACCESS),
        ):
            pass
        with pytest.raises(PermissionError):
            runtime.ReplayRuntime.generate_group(r)
    with pytest.raises(PermissionError):
        access.active_session(r)


def test_failed_forward_consumes_permit_and_parameter_mutation_blocks(setup):
    r = loaded_fixture(setup)
    rows, _, _ = t03.selected_parent(setup.context, "b0")
    with access.open_replay(r, setup.context, setup.spec, setup.ref) as session:
        with (
            pytest.raises(RuntimeError),
            session.forward([rows[0]], purpose="t03_reference", grad=False, access=_RUNTIME_ACCESS),
        ):
            raise RuntimeError("synthetic failure")
        with (
            pytest.raises(PermissionError),
            session.forward([rows[0]], purpose="t03_reference", grad=False, access=_RUNTIME_ACCESS),
        ):
            pass
        with torch.no_grad():
            r.model.frozen.add_(1)
        with pytest.raises(PermissionError):
            session.assert_active()


def test_support_failure_precedes_model_construction(setup):
    calls = []

    def deny(*args, **kwargs):
        raise PermissionError("Review not complete")

    setup.monkeypatch.setattr(source, "require_support", deny)
    setup.monkeypatch.setattr(
        runtime.ReplayRuntime, "__init__", lambda *a, **k: calls.append("model")
    )
    with pytest.raises(PermissionError):
        runtime.load(setup.root, "b0", "LH-SRF6-T03-123-b0", setup.root / "audit", spec=setup.spec)
    assert calls == []


def test_real_load_function_binds_inputs_before_live_session(setup):
    import mm_dev.runtime as backend

    calls = []
    tiny = setup.tiny["b0"]

    def initialize(self, *args, **kwargs):
        self.model = tiny.model
        self.identity = {"generation_config_expanded": dict(eos_token_id=[99], pad_token_id=0)}
        self.processor = tiny.processor
        self.eos_ids = [99]
        self.torch = torch
        self.account = kwargs["account"]
        calls.append("model")

    setup.monkeypatch.setattr(runtime.ReplayRuntime, "__init__", initialize)
    setup.monkeypatch.setattr(runtime, "configure_parent_training", lambda *a, **k: None)
    setup.monkeypatch.setattr(runtime, "capture_rng", lambda: "synthetic")
    setup.monkeypatch.setattr(runtime, "restore_rng", lambda value: calls.append(value))
    setup.monkeypatch.setattr(runtime, "seed_all", lambda seed: None)
    setup.monkeypatch.setattr(backend, "configure_audited_backend", lambda: None)
    setup.monkeypatch.setattr(backend, "actual_cuda_identity", lambda: dict(scope="SYNTHETIC_ONLY"))
    setup.monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    setup.monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    task = "LH-SRF6-T03-123-b0"
    audit = setup.root / "T03/RUNS" / task
    result, context = runtime.load(setup.root, "b0", task, audit, spec=setup.spec)
    assert calls == ["model", "synthetic"]
    assert runtime.verify_loaded(result)["spec_ref"] == setup.ref
    assert result._input_root == setup.context["producer_context"]["root"]
    receipt = json.loads((audit / "runtime_identity" / f"{task}.json").read_text())
    assert receipt["consumer"] == setup.context["consumer"] and receipt["optimizer"] is None
    with access.open_replay(result, context, setup.spec, setup.ref) as session:
        session.assert_active()


def install_success(setup):
    """Materialize mocked SERVER_ACTUAL-shaped evidence from explicit CPU fixtures."""
    for parent in t03.PARENTS:
        r = setup.tiny[parent]
        rows, raw, policy = t03.selected_parent(setup.context, parent)
        task = f"LH-SRF6-T03-123-{parent}"
        audit = setup.root / "T03/RUNS" / task
        # Original kernel runs with its explicit SYNTHETIC_ONLY scope.
        report = t03.kernel.run_synthetic(r, rows, raw, r.reference, output_directory=audit / "T03")
        assert report["status"] == "SYNTHETIC_ONLY"
        attempt = dict(
            phase="CONFIRM_T03_V1",
            parent=parent,
            task_id=task,
            spec_ref=setup.ref,
            consumer=setup.spec["context"]["consumer"],
            producer=setup.spec["context"]["producer"],
            generated_responses=0,
            optimizer_updates=0,
        )
        write(audit / "ATTEMPT.json", attempt)
        write(setup.root / f"T03/SLOTS/{parent}.json", attempt)
        hardware = dict(
            cuda_visible_device_count=1, cuda_uuid="SYNTHETIC", cuda_name="RTX PRO 6000 SYNTHETIC"
        )
        identity = dict(
            phase=runtime.PHASE,
            execution_mode="REPLAY",
            run_root=str(setup.root),
            audit_root=str(audit),
            input_root=str(setup.context["producer_context"]["root"]),
            parent_id=parent,
            reference_parent=parent,
            task_id=task,
            run_id=task,
            step=0,
            policy_hash=policy,
            reference_hash=policy,
            fresh_optimizer=False,
            optimizer_updates=0,
            generated_responses=0,
            dtype="float32",
            probability_dtype="float32",
            consumer=setup.spec["context"]["consumer"],
            producer=setup.spec["context"]["producer"],
            optimizer=None,
            scheduler=None,
            hardware=hardware,
            trainable_parameters=report["parameter_inventory"],
            generation_config_expanded=dict(eos_token_id=[99], pad_token_id=0),
        )
        write(audit / f"runtime_identity/{task}.json", identity)
        events = []
        for i, event in enumerate(r.fixture_accounting):
            event = copy.deepcopy(event)
            event.update(task_id=task, parent=parent)
            event["metadata"].update(
                qid=rows[i // 2]["qid"],
                row_id=rows[i // 2]["row_id"],
                consumer_registration_sha256=setup.context["consumer"]["registration_sha256"],
                producer_registration_sha256="b" * 64,
            )
            events.append(event)
        ledger = audit / f"accounting/{task}.jsonl"
        ledger.parent.mkdir()
        ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
        report.update(
            scope="SERVER_ACTUAL",
            status="PASS",
            parent=parent,
            task_id=task,
            context=setup.spec["context"],
            spec_ref=setup.ref,
            inputs=setup.spec["inputs"],
            hardware=hardware,
            execution_attempt_ref=t03._ref(setup.root, f"T03/RUNS/{task}/ATTEMPT.json"),
        )
        report["accounting_expected_relative_path"] = str(ledger.relative_to(setup.root))
        report["accounting"].update(
            scope="SERVER_ACTUAL",
            reference=dict(path=str(ledger.relative_to(setup.root)), sha256=file_hash(ledger)),
            events=events,
        )
        # Kernel recomputation receives the changed genuine fixture event identities.
        report["assessment"] = t03.kernel.assess_facts(report)
        assert report["assessment"]["status"] == "PASS", report["assessment"]
        write(audit / "T03/RESULT.json", report)
        write(
            audit / "T03/STARTED.json",
            {
                k: report[k]
                for k in (
                    "scope",
                    "parent",
                    "task_id",
                    "context",
                    "spec_ref",
                    "inputs",
                    "hardware",
                    "execution_attempt_ref",
                )
            },
        )


def test_complete_summary_recomputes_actual_kernel_but_fixture_is_not_real(setup):
    install_success(setup)
    summary = t03.verify_completed(setup.root)
    assert summary["status"] == "PASS", summary
    assert set(summary["parents"]) == set(t03.PARENTS)
    assert summary["generated_responses"] == summary["optimizer_updates"] == 0
    assert summary["training_authorized"] is False


@pytest.mark.parametrize(
    "extra",
    [
        "FAILURES/orphan.txt",
        "FAILURES/nested/unknown.json",
        "RUNS/LH-SRF6-T03-123-b0/T03/orphan.json",
        "RUNS/extra/ATTEMPT.json",
        "SLOTS/unknown.json",
    ],
)
def test_unknown_descendant_blocks_complete_summary_as_technical(setup, extra):
    install_success(setup)
    write(setup.root / "T03" / extra, {"synthetic": "unexpected"})
    result = t03.verify_completed(setup.root)
    assert result["status"] == "TECHNICAL_INVALID" and result["errors"]


def test_malformed_ledger_is_technical_not_scientific_failure(setup):
    install_success(setup)
    path = setup.root / "T03/RUNS/LH-SRF6-T03-123-b0/accounting/LH-SRF6-T03-123-b0.jsonl"
    path.write_text("{}\n")
    assert t03.verify_completed(setup.root)["status"] == "TECHNICAL_INVALID"


def test_missing_parent_is_incomplete_and_bad_boundary_is_technical(setup):
    assert t03.verify_completed(setup.root)["status"] == "INCOMPLETE"
    install_success(setup)
    path = setup.root / "T03/RUNS/LH-SRF6-T03-123-b0/T03/RESULT.json"
    report = json.loads(path.read_text())
    report["boundaries"][0]["finish_reason"] = "eos"
    write(path, report)
    assert t03.verify_completed(setup.root)["status"] == "TECHNICAL_INVALID"


def test_guarded_runtime_original_batch_construction_and_kernel_lifecycle(setup):
    """Only native model math is mocked; all new permissions and old kernel execute."""
    tiny = setup.tiny["b0"]
    r = runtime.ReplayRuntime.__new__(runtime.ReplayRuntime)
    bound = loaded_fixture(setup)
    r.model = bound.model
    r.identity = copy.deepcopy(bound.identity)
    r._reference = bound._reference
    r._adapter_names = bound._adapter_names
    r.processor = tiny.processor
    r.eos_ids = tiny.eos_ids
    r.torch = torch
    r.protocol_revision = "v2"
    r._input_root = setup.context["producer_context"]["root"]
    tiny.prepared["routing"]["protocol_revision"] = "v2"
    runtime._LOADED[r] = dict(
        runtime._LOADED[bound], model=r.model, identity=copy.deepcopy(r.identity)
    )
    events = []

    def record(kind, count, metadata):
        events.append(
            dict(
                task_id=r.identity["task_id"],
                parent="b0",
                kind=kind,
                count=count,
                metadata=metadata,
            )
        )

    r.account = record
    setup.monkeypatch.setattr(
        runtime.FP32Runtime, "prepare_input", lambda *args, **kwargs: tiny.prepared
    )

    def native(self, inputs, *, completion_start, lengths, purpose, grad):
        assert completion_start == 3 and lengths == [inputs["input_ids"].shape[-1] - 3]
        row = access.active_session(self)._forward["row"]
        self.reserve(
            "extra_forward_sequences", 1, purpose=purpose, completion_tokens=len(row["tokens"])
        )
        adapter = self.model.adapter if grad else self._reference["adapter"]
        value = tiny.value(row, adapter)
        return [value]

    setup.monkeypatch.setattr(runtime.FP32Runtime, "_native_forward", native)
    # Terminal ledger is a test callback, while actual call-budget events are captured above.
    setup.monkeypatch.setattr(
        t03.kernel,
        "_terminal_accounting",
        lambda session: dict(
            scope="SERVER_ACTUAL",
            events=copy.deepcopy(events),
            snapshot="AFTER_NUMERICAL_AND_RESTORATION",
            reference=dict(
                path=f"T03/RUNS/{r.identity['task_id']}/accounting/{r.identity['task_id']}.jsonl",
                sha256="a" * 64,
            ),
        ),
    )
    before = fixture.snapshot(tiny)
    with access.open_replay(r, setup.context, setup.spec, setup.ref) as session:
        report = t03.kernel._execute(session, hardware={"scope": "SYNTHETIC_ONLY"})
        assert session._calls == 16
    assert report["assessment"]["status"] == "PASS", report
    assert len(events) == 16 and len(report["comparisons"]) == 16
    assert fixture.snapshot(tiny) == before
    assert report["generated_responses"] == report["optimizer_updates"] == 0


def test_full_numerical_failure_is_distinct_from_technical_unknown(setup):
    install_success(setup)
    task = "LH-SRF6-T03-123-b0"
    path = setup.root / f"T03/RUNS/{task}/T03/RESULT.json"
    report = json.loads(path.read_text())
    report["comparisons"][0]["loss"]["f6"] += 1
    report["assessment"] = t03.kernel.assess_facts(report)
    assert report["assessment"]["status"] == "FAIL"
    report["status"] = "FAIL"
    write(path, report)
    attempt = json.loads((setup.root / f"T03/RUNS/{task}/ATTEMPT.json").read_text())
    write(
        setup.root / f"T03/FAILURES/{task}.json",
        dict(
            attempt,
            status="FAIL",
            result_ref=t03._ref(setup.root, str(path.relative_to(setup.root))),
        ),
    )
    assert t03.verify_completed(setup.root)["status"] == "FAIL"


def test_evidence_membership_change_during_read_is_technical(setup):
    install_success(setup)
    original = t03._validate_result

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        write(setup.root / "T03/FAILURES/.late", {})
        return result

    setup.monkeypatch.setattr(t03, "_validate_result", changing)
    assert t03.verify_completed(setup.root)["status"] == "TECHNICAL_INVALID"
