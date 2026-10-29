"""Synthetic isolated transport fixtures, never scientific confirmation evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from sr_f6 import historical_closure as bridge
from sr_f6 import phases
from sr_f6.common import digest, file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bridge._payload(value))
    return dict(path=str(path), sha256=file_hash(path))


def manifest(path, code_root, **extra):
    return write(
        path,
        dict(
            files=[
                dict(path=str(p), sha256=file_hash(p))
                for p in sorted(code_root.rglob("*"))
                if p.suffix in (".py", ".sbatch")
            ],
            **extra,
        ),
    )


def child(request, cwd, tmp_path):
    scratch = tmp_path / "private-cache"
    scratch.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", bridge.BOOTSTRAP],
        input=bridge._payload(request),
        capture_output=True,
        cwd=cwd,
        env=dict(
            os.environ,
            PYTHONPATH="/wrong/current/producer",
            TMPDIR=str(scratch),
            XDG_CACHE_HOME=str(scratch),
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
        "CLOSURE_ROOT",
        "CLOSURE_DEPLOYMENT",
    ):
        directory = project / getattr(bridge, name).name
        directory.mkdir()
        monkeypatch.setattr(bridge, name, directory)
    producer, consumer, repair = (
        bridge.PRODUCER_DEPLOYMENT,
        bridge.FOLLOWUP_DEPLOYMENT,
        bridge.CLOSURE_DEPLOYMENT,
    )
    old_package = producer / "src/sr_f6"
    old_package.mkdir(parents=True)
    (old_package / "__init__.py").write_text("")
    (old_package / "confirm_registration.py").write_text("# SYNTHETIC fixture only\n")
    package = consumer / "src/sr_f6_followup"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "source.py").write_text(
        "import hashlib, json, pathlib\n"
        f"ROOT = pathlib.Path({str(bridge.FOLLOWUP_ROOT)!r})\n"
        f"PRODUCER_DEPLOYMENT = pathlib.Path({str(producer)!r})\n"
        "def _path(path): return pathlib.Path(path)\n"
        "def _hash(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()\n"
        "def _read(path): return json.loads(pathlib.Path(path).read_bytes())\n"
        "def _state(path): return dict(path=str(path))\n"
        "def _reference(path): return dict(path=str(path), sha256=_hash(path))\n"
        "def _bound(ref):\n"
        "    assert _hash(ref['path']) == ref['sha256'], 'Changed fixture reference'\n"
        "    return _read(ref['path'])\n"
        "def require_support(root, purpose): return _read(ROOT / 'FIXTURE.json')\n"
        "def verify_registered(root, require_support=False):\n"
        "    return _read(ROOT / 'FIXTURE.json')\n"
    )
    repair_package = repair / "src/sr_f6_closure"
    repair_package.mkdir(parents=True)
    (repair_package / "__init__.py").write_text("")
    (repair_package / "source.py").write_text(
        "from pathlib import Path\n"
        "from sr_f6_followup import source as original\n"
        f"ROOT = Path({str(bridge.CLOSURE_ROOT)!r})\n"
        f"ORIGINAL_DEPLOYMENT = Path({str(consumer)!r})\n"
        f"CODE = Path({str(repair)!r})\n"
        "def deployment(revision): return CODE\n"
        "def source_paths(code):\n"
        "    return sorted(p for p in code.rglob('*') if p.suffix in ('.py', '.sbatch'))\n"
        "def original_context(): return original.verify_registered(original.ROOT)\n"
        "def contract(): return original._read(ROOT / 'CONTRACT.json')\n"
        "def history(): return original._read(original.ROOT / 'FIXTURE.json')['history']\n"
        "def verify_registered(): raise AssertionError('Do not spoof original bootstrap')\n"
    )
    # The child is genuinely isolated, but these fixture namespaces are only
    # stand-ins. The marker below prevents public promotion of this transport.
    (repair_package / "runner.py").write_text(
        "def guard_write(event, args, allowed):\n"
        "    if event == 'open' and isinstance(args[0], str):\n"
        "        if args[1] and any(c in args[1] for c in 'wax+'):\n"
        "            raise PermissionError('Synthetic read-only guard')\n"
    )
    (repair_package / "projection.py").write_text("# SYNTHETIC projection stand-in\n")
    (repair_package / "admission_projection.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from . import source\n"
        "CALLS = 0\n"
        "def evaluate(root):\n"
        "    global CALLS\n"
        "    CALLS += 1\n"
        "    assert CALLS == 1, 'No second numerical reconstruction'\n"
        "    context = source.original._read(source.original.ROOT / 'FIXTURE.json')\n"
        "    mode = context.get('fixture_behavior')\n"
        "    if mode == 'module_origin':\n"
        "        __spec__.origin = '/wrong/current/source.py'\n"
        "    if mode == 'lazy_source':\n"
        "        source.__spec__.origin = '/wrong/lazy/source.py'\n"
        "    if mode == 'hostile_cache':\n"
        "        attempt = Path(os.environ['TMPDIR']).parent\n"
        "        for key in context['fixture_cache_variables']:\n"
        "            assert Path(os.environ[key]).is_relative_to(attempt), key\n"
        "            assert 'hostile-parent-cache' not in os.environ[key], key\n"
        "        assert os.environ['TMP'] == os.environ['TEMP'] == os.environ['TMPDIR']\n"
        "    return context['bundle']\n"
    )
    scripts = repair / "scripts/sr_f6_closure"
    scripts.mkdir(parents=True)
    (scripts / "bootstrap.py").write_text("# SYNTHETIC inventoried original launcher\n")
    (scripts / "verify.sbatch").write_text("# SYNTHETIC inventoried scheduler launcher\n")
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
    execution_manifest = manifest(bridge.FOLLOWUP_ROOT / "SOURCE_MANIFEST.json", consumer)
    execution_registration_ref = write(
        bridge.FOLLOWUP_ROOT / "REGISTRATION.json",
        dict(
            schema="srf6-followup-registration-v2",
            root=str(bridge.FOLLOWUP_ROOT),
            revision=bridge.FOLLOWUP_REVISION,
            deployment=str(consumer),
            source_manifest=execution_manifest,
            producer=producer_identity,
        ),
    )
    monkeypatch.setattr(
        bridge, "FOLLOWUP_REGISTRATION_SHA256", execution_registration_ref["sha256"]
    )
    consumer_identity = dict(
        deployment=str(consumer),
        revision=bridge.FOLLOWUP_REVISION,
        registration_sha256=execution_registration_ref["sha256"],
        source_manifest_sha256=execution_manifest["sha256"],
    )
    support = dict(dependencies=[], producer=producer_identity, source_rows=9216)
    support_ref = write(bridge.FOLLOWUP_ROOT / "SUPPORT_BINDING.json", support)
    history = [
        write(bridge.FOLLOWUP_ROOT / "ADMISSION" / name, dict(status="UNKNOWN", name=name))
        for name in ("ATTEMPT.json", "ASSESSMENT.json", "RECEIPT.json")
    ]
    history.append(write(project / "diagnostic.json", dict(status="TECHNICAL_INVALID")))
    repair_manifest = manifest(
        bridge.CLOSURE_ROOT / "SOURCE_MANIFEST.json",
        repair,
        schema="srf6-technical-closure-source-v1",
        revision=bridge.CLOSURE_REVISION,
        deployment=str(repair),
    )
    contract = write(bridge.CLOSURE_ROOT / "CONTRACT.json", dict(history=history))
    permissions = dict(training_allowed=False, generation_allowed=False, evaluation_allowed=False)
    registration_ref = write(
        bridge.CLOSURE_ROOT / "REGISTRATION.json",
        dict(
            schema="srf6-technical-closure-source-v1",
            revision=bridge.CLOSURE_REVISION,
            deployment=str(repair),
            status="BOUND_NOT_AUTHORIZED",
            original_execution_consumer=consumer_identity,
            source_manifest=repair_manifest,
            contract=contract,
            **permissions,
        ),
    )
    identity = dict(
        original_execution_consumer=consumer_identity,
        repair_verifier=dict(
            revision=bridge.CLOSURE_REVISION,
            deployment=str(repair),
            registration=registration_ref,
            source_manifest=repair_manifest,
            contract=contract,
        ),
    )
    numerical = dict(
        status="PASS",
        scope="SERVER_ACTUAL",
        parents={p: dict(status="PASS") for p in ("b0", "b1", "b2")},
        inputs=[],
        generated_responses=0,
        optimizer_updates=0,
    )
    numerical_ref = write(bridge.CLOSURE_ROOT / "NUMERICAL.json", numerical)
    inner = dict(status="CONFIRMATION_GATES_PASS", numerical_sha256=digest(numerical))
    inner_ref = write(bridge.CLOSURE_ROOT / "ORIGINAL_RULE_ASSESSMENT.json", inner)
    assessment_ref = write(
        bridge.CLOSURE_ROOT / "ASSESSMENT.json",
        dict(
            schema="srf6-independent-technical-closure-v1",
            status="CONFIRMATION_GATES_PASS",
            scope="EXISTING_CONFIRMATION_EVIDENCE_ONLY_SEPARATE_PILOT_REQUIRED",
            identity=identity,
            preserved_history=history,
            original_rule_assessment=inner_ref,
            numerical=numerical_ref,
            old_admission_status="UNKNOWN_NOT_MODIFIED",
            new_model_calls=0,
            optimizer_updates=0,
            **permissions,
        ),
    )
    receipt_ref = write(
        bridge.CLOSURE_ROOT / "RECEIPT.json",
        dict(
            schema="srf6-independent-technical-closure-v1",
            status="CONFIRMATION_GATES_PASS",
            assessment=assessment_ref,
            identity=identity,
            gate_status="CORRECTED_CONFIRMATION_FACTS_SEALED",
            **permissions,
        ),
    )
    write(bridge.CLOSURE_ROOT / "ATTEMPT.json", dict(identity=identity, new_model_calls=0))
    pins = {p.name: file_hash(p) for p in bridge.CLOSURE_ROOT.iterdir()}
    monkeypatch.setattr(bridge, "CLOSURE_PINS", pins)
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
        consumer=consumer_identity,
        support_binding=support,
        support_binding_ref=support_ref,
        review=review,
        bundle=dict(assessment=inner, numerical=numerical),
        history=history,
        spec=dict(
            repair_history=dict(
                status="HISTORICAL_NEGATIVE_EVIDENCE_NOT_AUTHORITY", completed=False
            )
        ),
    )
    fixture = bridge.FOLLOWUP_ROOT / "FIXTURE.json"
    write(fixture, context)
    request = dict(
        nonce="a" * 32,
        followup_root=str(bridge.FOLLOWUP_ROOT),
        followup_deployment=str(consumer),
        producer_deployment=str(producer),
        closure_root=str(bridge.CLOSURE_ROOT),
        closure_deployment=str(repair),
        closure_pins=pins,
        registration_ref=registration_ref,
        receipt_ref=receipt_ref,
    )
    result = child(request, consumer, project)
    assert result.returncode == 0, result.stderr.decode()
    registered = bridge._request(registration_ref, receipt_ref)
    return dict(
        root=root,
        request=request,
        envelope=json.loads(result.stdout),
        registered=registered,
        registration_ref=registration_ref,
        receipt_ref=receipt_ref,
        fixture=fixture,
    )


def test_actual_isolated_transport_keeps_three_identities_and_rejects_synthetic(case):
    envelope = case["envelope"]
    assert envelope["isolated"] == 1 and envelope["child_pid"] != os.getpid()
    assert (
        envelope["request_sha256"] == hashlib.sha256(bridge._payload(case["request"])).hexdigest()
    )
    facts = envelope["result"]
    assert facts["consumer"]["revision"] == bridge.FOLLOWUP_REVISION
    assert facts["repair_verifier"]["revision"] == bridge.CLOSURE_REVISION
    assert facts["producer"]["revision"] == bridge.PRODUCER_REVISION
    assert facts["old_admission_status"] == "UNKNOWN_NOT_MODIFIED"
    with pytest.raises(PermissionError, match="Actual unchanged"):
        bridge.verify_corrected_confirmation(
            case["root"],
            closure_registration_ref=case["registration_ref"],
            corrected_confirmation_receipt_ref=case["receipt_ref"],
        )
    attempts = list((case["root"] / bridge.DIRECTORY).iterdir())
    assert len(attempts) == 1
    failure = json.loads((attempts[0] / "FAILURE.json").read_bytes())
    assert failure["status"] == "UNKNOWN" and failure["execution"]["returncode"] == 0
    assert (attempts[0] / "stdout.json").is_file() and (attempts[0] / "stderr.log").is_file()
    assert not (attempts[0] / "RESULT.json").exists()
    assert not (case["root"] / bridge.BINDING_PATH).exists()


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
        "repair_revision",
        "repair_origin",
        "old_unknown",
        "history",
        "permission",
        "producer",
        "inputs",
        "synthetic",
    ],
)
def test_shape_and_identity_guards_never_promote_fixture(case, change):
    # Remove only fixture marker to reach distinct malformed-envelope checks.
    # These in-memory dicts are never admitted, bound, or considered real facts.
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
        facts["consumer"] = facts["repair_verifier"]
    elif change == "repair_revision":
        facts["repair_verifier"]["revision"] = "0" * 40
    elif change == "repair_origin":
        facts["repair_file"] = str(
            bridge.FOLLOWUP_DEPLOYMENT / "src/sr_f6_closure/admission_projection.py"
        )
    elif change == "old_unknown":
        facts["old_admission_status"] = "CONFIRMATION_GATES_PASS"
    elif change == "history":
        facts["corrected_history"] = []
    elif change == "permission":
        facts["receipt"]["training_allowed"] = True
    elif change == "producer":
        facts["producer_file"] = str(
            bridge.CLOSURE_DEPLOYMENT / "src/sr_f6/confirm_registration.py"
        )
    elif change == "inputs":
        facts["inputs"] = []
    else:
        facts["numerical"]["scope"] = "SYNTHETIC_TEST"
    value["result_sha256"] = digest(facts)
    with pytest.raises(PermissionError):
        bridge._facts(value, case["request"], case["registered"])


@pytest.mark.parametrize(
    "member", ["source.py", "runner.py", "projection.py", "admission_projection.py"]
)
def test_changed_repair_source_rejected_before_child(case, member):
    path = bridge.CLOSURE_DEPLOYMENT / "src/sr_f6_closure" / member
    path.write_text(path.read_text() + "\n# altered\n")
    with pytest.raises(PermissionError, match="source content"):
        bridge._request(case["registration_ref"], case["receipt_ref"])


def test_original_unknown_is_not_a_corrected_receipt(case):
    reference = dict(
        path=str(bridge.FOLLOWUP_ROOT / "ADMISSION/RECEIPT.json"),
        sha256=file_hash(bridge.FOLLOWUP_ROOT / "ADMISSION/RECEIPT.json"),
    )
    with pytest.raises(PermissionError, match="actual separate"):
        bridge._request(case["registration_ref"], reference)


@pytest.mark.parametrize("change", ["extra", "missing", "changed", "shadow"])
def test_complete_new_membership_and_pins_are_required(case, change):
    if change == "extra":
        write(bridge.CLOSURE_ROOT / "EXTRA.json", {})
    elif change == "missing":
        (bridge.CLOSURE_ROOT / "NUMERICAL.json").unlink()
    elif change == "changed":
        write(bridge.CLOSURE_ROOT / "NUMERICAL.json", dict(status="FAIL"))
    else:
        (bridge.CLOSURE_DEPLOYMENT / "src/sr_f6").mkdir()
    with pytest.raises(PermissionError):
        bridge._request(case["registration_ref"], case["receipt_ref"])


@pytest.mark.parametrize("change", ["negative", "new_producer", "module_origin", "lazy_source"])
def test_isolated_child_reconstructs_facts_and_rechecks_import_origins(case, change, tmp_path):
    request = copy.deepcopy(case["request"])
    if change == "negative":
        context = json.loads(case["fixture"].read_bytes())
        context["bundle"]["numerical"]["parents"]["b0"]["status"] = "FAIL"
        write(case["fixture"], context)
    elif change == "new_producer":
        new = tmp_path / "new_producer"
        (new / "src/sr_f6").mkdir(parents=True)
        (new / "src/sr_f6/__init__.py").write_text("")
        (new / "src/sr_f6/confirm_registration.py").write_text("# new producer forbidden\n")
        request["producer_deployment"] = str(new)
    else:
        # Keep inventoried bytes unchanged and let the fixture evaluate mutate
        # loaded origins after the first source check, exercising the final one.
        context = json.loads(case["fixture"].read_bytes())
        context["fixture_behavior"] = change
        write(case["fixture"], context)
    result = child(request, bridge.FOLLOWUP_DEPLOYMENT, tmp_path)
    assert result.returncode != 0
    assert not result.stdout
    expected = (
        "Corrected actual facts"
        if change == "negative"
        else "Wrong frozen producer"
        if change == "new_producer"
        else "Loaded repair origin/spec differs"
    )
    assert expected in result.stderr.decode()


def test_hostile_inherited_library_caches_are_isolated_in_actual_subprocess(case, monkeypatch):
    context = json.loads(case["fixture"].read_bytes())
    context["fixture_behavior"] = "hostile_cache"
    context["fixture_cache_variables"] = [*bridge.CACHE_LOCATIONS, "TMPDIR", "TMP", "TEMP"]
    write(case["fixture"], context)
    hostile = case["root"].parent / "hostile-parent-cache"
    for key in context["fixture_cache_variables"]:
        monkeypatch.setenv(key, str(hostile / key))
    with pytest.raises(PermissionError, match="Actual unchanged"):
        bridge.verify_corrected_confirmation(
            case["root"],
            closure_registration_ref=case["registration_ref"],
            corrected_confirmation_receipt_ref=case["receipt_ref"],
        )
    attempts = list((case["root"] / bridge.DIRECTORY).iterdir())
    assert len(attempts) == 1
    failure = json.loads((attempts[0] / "FAILURE.json").read_bytes())
    # Fixture cache assertions happened inside the true isolated process; only
    # synthetic scientific facts were rejected by the parent, as required.
    assert failure["execution"]["returncode"] == 0
    assert (attempts[0] / "stderr.log").read_bytes() == b""
    assert not hostile.exists()


def test_snapshot_checks_new_and_old_closures_and_model_metadata_without_weight_hash(
    case, monkeypatch
):
    model = bridge.PRODUCER_ROOT.parent / "bound-model.bin"
    model.write_bytes(b"fixture-only never model data")
    write(
        bridge.FOLLOWUP_ROOT / "SUPPORT_BINDING.json",
        dict(dependencies=[dict(path=str(model), sha256=file_hash(model))]),
    )
    before = bridge._snapshot()
    monkeypatch.setattr(
        bridge, "file_hash", lambda *_: pytest.fail("Warm snapshot must not hash weights")
    )
    assert bridge._snapshot() == before
    for base in (
        bridge.CLOSURE_ROOT,
        bridge.CLOSURE_DEPLOYMENT,
        bridge.HISTORY_ROOT,
        bridge.FOLLOWUP_ROOT,
    ):
        extra = base / "NEW.json"
        extra.write_text("{}")
        assert bridge._snapshot() != before
        extra.unlink()
    model.write_bytes(b"changed")
    assert bridge._snapshot()["states"][str(model)] != before["states"][str(model)]


def test_warm_binding_keeps_original_consumer_and_invalidates_changed_repair(case, monkeypatch):
    # Only inject private synthetic cache to exercise warm checks. The public
    # cold verifier above rejects the fixture; no actual gate is signed here.
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
        "verify_corrected_confirmation",
        lambda *a, **k: pytest.fail("Warm proof must not launch child"),
    )
    first = bridge.require_confirmation_binding(
        root, phase_context=context, confirmation_binding_ref=binding_ref
    )
    assert first["binding"]["consumer"]["revision"] == bridge.FOLLOWUP_REVISION
    assert first["binding"]["repair_verifier"]["revision"] == bridge.CLOSURE_REVISION
    first["verification"]["facts"]["review"]["status"] = "changed caller copy"
    second = bridge.require_confirmation_binding(
        root, phase_context=context, confirmation_binding_ref=binding_ref
    )
    assert second["verification"]["facts"]["review"]["status"] != "changed caller copy"
    path = bridge.CLOSURE_DEPLOYMENT / "src/sr_f6_closure/source.py"
    path.write_text(path.read_text() + "\n# changed after successful proof\n")
    with pytest.raises(PermissionError, match="membership changed"):
        bridge.require_confirmation_binding(
            root, phase_context=context, confirmation_binding_ref=binding_ref
        )


def test_wrong_actual_current_phase_is_rejected(case):
    with pytest.raises((PermissionError, KeyError)):
        bridge.require_confirmation_binding(
            case["root"],
            phase_context=dict(identity_sha256="0" * 64),
            confirmation_binding_ref=dict(path=bridge.BINDING_PATH, sha256="0" * 64),
        )


def test_bootstrap_has_no_old_seal_or_spoofed_launcher():
    assert "repair_source.verify_registered(" not in bridge.BOOTSTRAP
    assert "admission.require_confirmation(" not in bridge.BOOTSTRAP
    assert 'sys.modules["__main__"].__file__ =' not in bridge.BOOTSTRAP
    assert bridge.BOOTSTRAP.count("admission_projection.evaluate(") == 1
    assert 'values["CONTRACT.json"] == repair_source.contract()' in bridge.BOOTSTRAP
