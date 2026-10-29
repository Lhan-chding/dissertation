"""Synthetic CPU schema regressions only; no actual scientific admission."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_core.training import state_hash
from sr_f6.common import digest, file_hash
from sr_f6_closure import projection
from sr_f6_followup import source, t03
from sr_f6_followup import t03_dispatch as dispatch

_spec = importlib.util.spec_from_file_location(
    "closure_old_synthetic_fixture", Path(__file__).with_name("test_followup_t03.py")
)
historical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(historical)
fixture = historical.fixture
write = historical.write


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
            b.update({k: v for k, v in shared.items() if k != "row_id"})
            b.pop("row_id", None)
            records.append(a)
            raws[a["row_id"]] = b
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
    assert all("row_id" not in row for row in raws.values())
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


def evidence_snapshot(setup):
    return dict(
        context=digest(t03._jsonable(setup.context)),
        files={
            str(path.relative_to(setup.root)): file_hash(path)
            for path in sorted(setup.root.rglob("*"))
            if path.is_file()
        },
    )


def result_path(setup, parent="b0"):
    return setup.root / f"T03/RUNS/LH-SRF6-T03-123-{parent}/T03/RESULT.json"


def test_real_wire_schema_old_reader_errors_and_repair_is_read_only(setup):
    assert all("row_id" not in row for row in setup.context["raw_by_row"].values())
    for parent in t03.PARENTS:
        _, raw, _ = t03.selected_parent(setup.context, parent)
        assert setup.spec["groups"][parent]["raw_sha256"] == digest(raw)
    historical.install_success(setup)
    before = evidence_snapshot(setup)
    old = t03.verify_completed(setup.root)
    assert old["status"] == "TECHNICAL_INVALID"
    assert len(old["errors"]) == 3
    assert all(error["error"] == "KeyError: 'row_id'" for error in old["errors"])
    proof = projection.verify_projection()
    fixed = projection.verify_completed(setup.root)
    assert fixed["status"] == "PASS", fixed  # Mocked CPU evidence only.
    assert fixed["generated_responses"] == fixed["optimizer_updates"] == 0
    assert fixed["training_authorized"] is False
    assert evidence_snapshot(setup) == before
    assert projection.verify_projection() == proof


@pytest.mark.parametrize("mutation", ["identity", "swap", "qid", "tokens", "probabilities"])
def test_original_vector_binding_cannot_be_weakened(setup, mutation):
    historical.install_success(setup)
    path = result_path(setup)
    report = json.loads(path.read_text())
    vectors = report["probability_vectors"]
    if mutation == "identity":
        vectors[0]["row_id"] = "substituted"
    elif mutation == "swap":
        vectors[0], vectors[1] = vectors[1], vectors[0]
    elif mutation == "qid":
        vectors[0]["qid"] = "substituted"
    elif mutation == "tokens":
        vectors[0]["tokens"][0] += 1
    else:
        vectors[0]["sampler_logprobs"][0] += 0.01
    write(path, report)
    result = projection.verify_completed(setup.root)
    assert result["status"] == "TECHNICAL_INVALID"
    assert any(
        "Original tokens/probabilities changed" in error["error"] for error in result["errors"]
    )


def test_raw_association_changes_are_rejected_before_reader(setup):
    historical.install_success(setup)
    keys = list(setup.context["raw_by_row"])
    raw = setup.context["raw_by_row"]
    raw[keys[0]], raw[keys[1]] = raw[keys[1]], raw[keys[0]]
    with pytest.raises(PermissionError):
        projection.verify_completed(setup.root)


@pytest.mark.parametrize("state", ["FAIL", "INCONCLUSIVE"])
def test_unfavorable_numerical_result_remains_unfavorable(setup, state):
    historical.install_success(setup)
    path = result_path(setup)
    report = json.loads(path.read_text())
    if state == "FAIL":
        report["comparisons"][0]["loss"]["f6"] += 1
    else:
        for comparison in report["comparisons"]:
            for metric in comparison["parameters"].values():
                for key in (
                    "f6_nonzero_elements",
                    "native_nonzero_elements",
                    "violations",
                ):
                    metric[key] = 0
                for key in (
                    "max_absolute_difference",
                    "max_tolerance_ratio",
                    "f6_l2",
                    "native_l2",
                    "difference_l2",
                ):
                    metric[key] = 0.0
    report["assessment"] = t03.kernel.assess_facts(report)
    assert report["assessment"]["status"] == state
    report["status"] = state
    write(path, report)
    task = "LH-SRF6-T03-123-b0"
    attempt = json.loads((setup.root / f"T03/RUNS/{task}/ATTEMPT.json").read_text())
    write(
        setup.root / f"T03/FAILURES/{task}.json",
        dict(
            attempt,
            status=state,
            result_ref=t03._ref(setup.root, str(path.relative_to(setup.root))),
        ),
    )
    assert projection.verify_completed(setup.root)["status"] == state


@pytest.mark.parametrize(
    "case", ["boundary", "runtime", "inventory", "ledger", "start", "restoration", "errors"]
)
def test_original_identity_and_completeness_guards_remain(setup, case):
    historical.install_success(setup)
    path = result_path(setup)
    report = json.loads(path.read_text())
    audit = path.parent.parent
    if case == "boundary":
        report["boundaries"][0]["finish_reason"] = "eos"
    elif case == "runtime":
        runtime_path = audit / "runtime_identity/LH-SRF6-T03-123-b0.json"
        identity = json.loads(runtime_path.read_text())
        identity["policy_hash"] = "0" * 64
        write(runtime_path, identity)
    elif case == "inventory":
        report["parameter_inventory"] = []
    elif case == "ledger":
        (audit / "accounting/LH-SRF6-T03-123-b0.jsonl").write_text("{}\n")
    elif case == "start":
        start_path = audit / "T03/STARTED.json"
        start = json.loads(start_path.read_text())
        start["task_id"] = "other"
        write(start_path, start)
    elif case == "restoration":
        report["restoration"]["policy_before_sha256"] = "0" * 64
    else:
        report["errors"] = ["synthetic incomplete execution"]
    write(path, report)
    assert projection.verify_completed(setup.root)["status"] == "TECHNICAL_INVALID"


@pytest.mark.parametrize(
    "extra",
    ["FAILURES/orphan.txt", "FAILURES/nested/a", "RUNS/extra/ATTEMPT.json", "SLOTS/orphan.json"],
)
def test_unknown_members_are_not_ignored(setup, extra):
    historical.install_success(setup)
    write(setup.root / "T03" / extra, {})
    assert projection.verify_completed(setup.root)["status"] == "TECHNICAL_INVALID"


def test_missing_parent_remains_incomplete(setup):
    assert projection.verify_completed(setup.root)["status"] == "INCOMPLETE"


def test_change_during_read_remains_technical(setup):
    historical.install_success(setup)
    original = projection._validate_result

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        write(setup.root / "T03/FAILURES/.late", {})
        return result

    setup.monkeypatch.setattr(projection, "_validate_result", changing)
    result = projection.verify_completed(setup.root)
    assert result["status"] == "TECHNICAL_INVALID"
    assert any(e["error"] == "T03 evidence changed during verification" for e in result["errors"])


@pytest.mark.parametrize("marker", ["missing", "wrong_result", "wrong_status"])
def test_numerical_failure_marker_must_match_original_result(setup, marker):
    historical.install_success(setup)
    path = result_path(setup)
    report = json.loads(path.read_text())
    report["comparisons"][0]["loss"]["f6"] += 1
    report["assessment"] = t03.kernel.assess_facts(report)
    report["status"] = "FAIL"
    write(path, report)
    task = "LH-SRF6-T03-123-b0"
    attempt = json.loads((setup.root / f"T03/RUNS/{task}/ATTEMPT.json").read_text())
    if marker != "missing":
        ref = t03._ref(setup.root, str(path.relative_to(setup.root)))
        if marker == "wrong_result":
            ref["sha256"] = "0" * 64
        write(
            setup.root / f"T03/FAILURES/{task}.json",
            dict(attempt, status="PASS" if marker == "wrong_status" else "FAIL", result_ref=ref),
        )
    result = projection.verify_completed(setup.root)
    assert result["parents"]["b0"]["status"] == "FAIL"
    assert result["status"] == "TECHNICAL_INVALID"


def test_recorded_execution_failure_remains_technical(setup):
    historical.install_success(setup)
    task = "LH-SRF6-T03-123-b0"
    audit = setup.root / f"T03/RUNS/{task}"
    attempt = json.loads((audit / "ATTEMPT.json").read_text())
    failure = dict(attempt, status="TECHNICAL_FAILURE_STOP", error="Original CPU fixture failure")
    write(audit / "FAILURE.json", failure)
    write(setup.root / f"T03/FAILURES/{task}.json", failure)
    result = projection.verify_completed(setup.root)
    assert result["status"] == result["parents"]["b0"]["status"] == "TECHNICAL_INVALID"
    assert result["errors"] == []


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("len(events) == 16", "len(events) == 15"),
        lambda text: text.replace('trace["row_id"] == record["row_id"]', "True"),
        lambda text: text.replace("if before != _tree_snapshot(root):", "if False:"),
        lambda text: text.replace(
            "from sr_f6_followup.source import require_support",
            "from other.source import require_support",
        ),
        lambda text: text + "\n\ndef verify_completed(root):\n    return {}\n",
        lambda text: text.replace(
            "def verify_completed(root):", "@decorator\ndef verify_completed(root):"
        ),
        lambda text: text.replace(
            'trace["qid"] == record["qid"] == original["qid"]', 'trace["qid"] == original["qid"]'
        ),
    ],
)
def test_only_exact_frozen_ast_transformation_is_accepted(change):
    old = Path(t03.__file__).read_text()
    new = Path(projection.__file__).read_text()
    projection.prove_projection_sources(old, new)
    with pytest.raises(PermissionError):
        projection.prove_projection_sources(old, change(new))


def test_changed_original_or_loaded_function_or_helper_is_rejected(monkeypatch):
    old, new = Path(t03.__file__).read_text(), Path(projection.__file__).read_text()
    assert hashlib.sha256(old.encode()).hexdigest() == projection.ORIGINAL_SOURCE_SHA256
    with pytest.raises(PermissionError):
        projection.prove_projection_sources(old + "\n", new)
    monkeypatch.setattr(projection, "_validate_result", lambda *a: {})
    with pytest.raises(PermissionError, match="Loaded projection"):
        projection.verify_projection()
    monkeypatch.undo()
    monkeypatch.setattr(projection, "selected_parent", lambda *a: {})
    with pytest.raises(PermissionError, match="Frozen helper replaced"):
        projection.verify_projection()
