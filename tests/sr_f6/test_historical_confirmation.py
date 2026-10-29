"""Synthetic transport fixtures; no result is actual confirmation evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from sr_f6 import historical_confirmation as bridge
from sr_f6 import phases
from sr_f6.common import digest, file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bridge._payload(value))
    return dict(path=str(path), sha256=file_hash(path))


def manifest(path, deployment):
    return write(
        path,
        dict(
            files=[dict(path=str(p), sha256=file_hash(p)) for p in sorted(deployment.rglob("*.py"))]
        ),
    )


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    root = project / phases.ROOT.name
    root.mkdir()
    monkeypatch.setattr(phases, "PROJECT", project)
    for name in (
        "FOLLOWUP_ROOT",
        "PRODUCER_ROOT",
        "FOLLOWUP_DEPLOYMENT",
        "PRODUCER_DEPLOYMENT",
        "HISTORY_ROOT",
        "HISTORY_DEPLOYMENT",
    ):
        directory = project / getattr(bridge, name).name
        directory.mkdir()
        monkeypatch.setattr(bridge, name, directory)
    producer, consumer = bridge.PRODUCER_DEPLOYMENT, bridge.FOLLOWUP_DEPLOYMENT
    (producer / "src/sr_f6").mkdir(parents=True)
    (producer / "src/sr_f6/__init__.py").write_text("")
    (producer / "src/sr_f6/confirm_registration.py").write_text("# SYNTHETIC fixture only\n")
    package = consumer / "src/sr_f6_followup"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "source.py").write_text(
        """
import hashlib, json, pathlib
ROOT = pathlib.Path("""
        + repr(str(bridge.FOLLOWUP_ROOT))
        + """)
def _path(path): return pathlib.Path(path)
def _reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest())
def require_support(root, purpose): return json.loads((ROOT / "FIXTURE.json").read_text())
def verify_registered(root, require_support=False):
    return json.loads((ROOT / "FIXTURE.json").read_text())
"""
    )
    (package / "admission.py").write_text("""
from . import source
def require_confirmation(root):
    value = source.require_support(root, "admission")["receipt"]
    return dict(value, receipt_ref=source._reference(source.ROOT / "ADMISSION/RECEIPT.json"))
""")
    (package / "t03.py").write_text("""
from . import source
def verify_completed(root): return source.require_support(root, "t03")["numerical"]
""")
    old_manifest = manifest(bridge.PRODUCER_ROOT / "SOURCE_MANIFEST.json", producer)
    old_registration = write(
        bridge.PRODUCER_ROOT / "REGISTRATION.json",
        dict(source_manifest_sha256=old_manifest["sha256"]),
    )
    monkeypatch.setattr(bridge, "PRODUCER_REGISTRATION_SHA256", old_registration["sha256"])
    producer_identity = dict(
        root=str(bridge.PRODUCER_ROOT),
        deployment=str(producer),
        revision=bridge.PRODUCER_REVISION,
        registration_sha256=old_registration["sha256"],
        source_manifest_sha256=old_manifest["sha256"],
    )
    new_manifest = manifest(bridge.FOLLOWUP_ROOT / "SOURCE_MANIFEST.json", consumer)
    registered = dict(
        schema="srf6-followup-registration-v2",
        root=str(bridge.FOLLOWUP_ROOT),
        revision=bridge.FOLLOWUP_REVISION,
        deployment=str(consumer),
        source_manifest=new_manifest,
        producer=producer_identity,
    )
    registration_ref = write(bridge.FOLLOWUP_ROOT / "REGISTRATION.json", registered)
    consumer_identity = dict(
        deployment=str(consumer),
        revision=bridge.FOLLOWUP_REVISION,
        source_manifest_sha256=new_manifest["sha256"],
        registration_sha256=registration_ref["sha256"],
    )
    support = dict(dependencies=[], producer=producer_identity, source_rows=9216)
    support_ref = write(bridge.FOLLOWUP_ROOT / "SUPPORT_BINDING.json", support)
    assessment_ref = write(
        bridge.FOLLOWUP_ROOT / "ADMISSION/ASSESSMENT.json", dict(status="SYNTHETIC_TEST")
    )
    receipt = dict(
        status="CONFIRMATION_GATES_PASS",
        gate_status="CONFIRMATION_FACTS_SEALED",
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        assessment=assessment_ref,
    )
    receipt_ref = write(bridge.FOLLOWUP_ROOT / "ADMISSION/RECEIPT.json", receipt)
    numerical = dict(
        status="PASS",
        scope="SERVER_ACTUAL",
        parents={p: dict(status="PASS") for p in ("b0", "b1", "b2")},
        inputs=[],
        generated_responses=0,
        optimizer_updates=0,
    )
    review = dict(
        status="REVIEW_COMPLETE_NOT_GATE_SIGNED",
        completed=True,
        synthetic=False,
        reviewed_original_answers=360,
        independently_reviewed_original_answers=360,
        inputs=[],
        execution_origin="SYNTHETIC_TEST",
    )
    context = dict(
        receipt=receipt,
        consumer=consumer_identity,
        support_binding=support,
        support_binding_ref=support_ref,
        numerical=numerical,
        review=review,
        spec=dict(
            repair_history=dict(
                status="HISTORICAL_NEGATIVE_EVIDENCE_NOT_AUTHORITY", completed=False
            )
        ),
    )
    write(bridge.FOLLOWUP_ROOT / "FIXTURE.json", context)
    request = dict(
        nonce="a" * 32,
        followup_root=str(bridge.FOLLOWUP_ROOT),
        followup_deployment=str(consumer),
        producer_deployment=str(producer),
        registration_ref=registration_ref,
        receipt_ref=receipt_ref,
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", bridge.BOOTSTRAP],
        input=bridge._payload(request),
        capture_output=True,
        check=True,
        cwd=consumer,
        env=dict(os.environ, PYTHONPATH="/wrong/current/source"),
    )
    envelope = json.loads(result.stdout)
    return dict(
        root=root,
        registered=registered,
        registration_ref=registration_ref,
        receipt_ref=receipt_ref,
        request=request,
        envelope=envelope,
    )


def test_real_isolated_dual_namespace_transport_rejects_synthetic(case):
    envelope = case["envelope"]
    assert envelope["isolated"] == 1
    assert envelope["child_pid"] != os.getpid()
    assert envelope["result"]["producer_file"].startswith(str(bridge.PRODUCER_DEPLOYMENT))
    assert envelope["result"]["consumer_file"].startswith(str(bridge.FOLLOWUP_DEPLOYMENT))
    assert (
        envelope["request_sha256"] == hashlib.sha256(bridge._payload(case["request"])).hexdigest()
    )
    with pytest.raises(PermissionError, match="Actual unchanged"):
        bridge.verify_followup_confirmation(
            case["root"],
            followup_registration_ref=case["registration_ref"],
            confirmation_receipt_ref=case["receipt_ref"],
        )
    attempts = list((case["root"] / bridge.DIRECTORY).iterdir())
    assert len(attempts) == 1
    failure = json.loads((attempts[0] / "FAILURE.json").read_bytes())
    assert failure["status"] == "UNKNOWN"
    assert failure["execution"]["returncode"] == 0
    assert (attempts[0] / "stdout.json").is_file()
    assert (attempts[0] / "stderr.log").is_file()
    assert not (attempts[0] / "RESULT.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "nonce",
        "request",
        "seal",
        "359",
        "361",
        "parent",
        "FAIL",
        "UNKNOWN",
        "old_review",
        "old_t03",
        "synthetic",
    ],
)
def test_contract_rejects_incomplete_or_misattributed_synthetic_shapes(case, change):
    # Strip only the transport fixture marker to reach each independent shape
    # check. These local dicts are never frozen, admitted, or called actual facts.
    value = copy.deepcopy(case["envelope"])
    facts = value["result"]
    facts["review"]["execution_origin"] = "ACTUAL_REVIEW"
    if change == "nonce":
        value["nonce"] = "b" * 32
    elif change == "request":
        value["request_sha256"] = "0" * 64
    elif change == "seal":
        facts["receipt"]["gate_status"] = "NOT_SIGNED"
    elif change in ("359", "361"):
        facts["review"]["independently_reviewed_original_answers"] = int(change)
    elif change == "parent":
        del facts["numerical"]["parents"]["b2"]
    elif change in ("FAIL", "UNKNOWN"):
        facts["numerical"]["parents"]["b1"]["status"] = change
    elif change == "old_review":
        facts["consumer"]["revision"] = "02fbf89" + "0" * 33
    elif change == "old_t03":
        facts["producer_file"] = str(
            bridge.FOLLOWUP_DEPLOYMENT / "src/sr_f6/confirm_registration.py"
        )
    else:
        facts["numerical"]["scope"] = "SYNTHETIC_TEST"
    value["result_sha256"] = digest(facts)
    with pytest.raises(PermissionError):
        bridge._facts(value, case["request"], case["registered"])


def test_source_tampering_and_shadow_namespace_fail_before_launch(case):
    path = bridge.FOLLOWUP_DEPLOYMENT / "src/sr_f6_followup/admission.py"
    path.write_text(path.read_text() + "\n# mutated\n")
    with pytest.raises(PermissionError, match="source content"):
        bridge._request(case["registration_ref"], case["receipt_ref"])
    shadow = bridge.FOLLOWUP_DEPLOYMENT / "src/sr_f6"
    shadow.mkdir()
    with pytest.raises(PermissionError, match="shadow"):
        bridge._request(case["registration_ref"], case["receipt_ref"])


def test_snapshot_detects_every_attempt_and_model_state_without_hashing(case, monkeypatch):
    model = bridge.PRODUCER_ROOT.parent / "bound-model.bin"
    model.write_bytes(b"synthetic model payload, never a real model")
    write(
        bridge.FOLLOWUP_ROOT / "SUPPORT_BINDING.json",
        dict(dependencies=[dict(path=str(model), sha256=file_hash(model))]),
    )
    before = bridge._snapshot()
    monkeypatch.setattr(
        bridge, "file_hash", lambda *_: pytest.fail("Warm snapshot must not hash weights")
    )
    assert bridge._snapshot() == before
    extra = bridge.FOLLOWUP_ROOT / "T03/RUNS/orphan/ATTEMPT.json"
    extra.parent.mkdir(parents=True)
    extra.write_text('{"status":"UNKNOWN"}')
    assert bridge._snapshot() != before
    extra.unlink()
    # Existing source/leaf mutation is independently detected by metadata.
    model.write_bytes(b"changed")
    assert bridge._snapshot()["states"][str(model)] != before["states"][str(model)]


def test_wrong_actual_current_phase_is_rejected(case):
    with pytest.raises((PermissionError, KeyError)):
        bridge.require_confirmation_binding(
            case["root"],
            phase_context={"identity_sha256": "0" * 64},
            confirmation_binding_ref=dict(path=bridge.BINDING_PATH, sha256="0" * 64),
        )


def test_warm_binding_reuses_process_proof_and_rejects_new_attempt(case, monkeypatch):
    """Inject a synthetic *internal* cache only to exercise the warm guard.

    The public verifier rejects this fixture above; no actual authority cache
    is populated by fixture data through any public production API.
    """
    context = dict(identity_sha256="c" * 64, scope="SYNTHETIC_TEST")
    monkeypatch.setattr(bridge, "_phase", lambda root, supplied: context)
    root = case["root"]
    folder = root / bridge.DIRECTORY / ("a" * 32)
    folder.mkdir(parents=True)
    transcript = write(folder / "REQUEST.json", dict(scope="SYNTHETIC_TEST"))
    transcript["path"] = str((folder / "REQUEST.json").relative_to(root))
    facts = copy.deepcopy(case["envelope"]["result"])
    report = dict(
        facts=facts,
        facts_sha256=digest(facts),
        transcript=[transcript],
        status="SYNTHETIC_TEST",
        training_authorized=False,
    )
    report_ref = write(folder / "RESULT.json", report)
    report_ref["path"] = str((folder / "RESULT.json").relative_to(root))
    verification = dict(report, report_ref=report_ref)
    binding = bridge._binding(context, case["registration_ref"], case["receipt_ref"], verification)
    binding_ref = write(root / bridge.BINDING_PATH, binding)
    binding_ref["path"] = bridge.BINDING_PATH
    key = (os.getpid(), str(root), digest(case["registration_ref"]), digest(case["receipt_ref"]))
    monkeypatch.setitem(bridge._CACHE, key, (verification, bridge._snapshot(facts["inputs"])))
    monkeypatch.setattr(
        bridge,
        "verify_followup_confirmation",
        lambda *a, **k: pytest.fail("Unchanged process proof must not launch a child"),
    )
    first = bridge.require_confirmation_binding(
        root, phase_context=context, confirmation_binding_ref=binding_ref
    )
    assert first["binding"]["training_authorized"] is False
    first["verification"]["facts"]["review"]["status"] = "tampered caller copy"
    second = bridge.require_confirmation_binding(
        root, phase_context=context, confirmation_binding_ref=binding_ref
    )
    assert second["verification"]["facts"]["review"]["status"] != "tampered caller copy"
    extra = bridge.FOLLOWUP_ROOT / "T03/RUNS/extra/ATTEMPT.json"
    write(extra, dict(status="UNKNOWN"))
    with pytest.raises(PermissionError, match="membership changed"):
        bridge.require_confirmation_binding(
            root, phase_context=context, confirmation_binding_ref=binding_ref
        )
