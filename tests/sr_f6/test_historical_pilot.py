"""CPU transport/source controls only, never actual scientific PILOT evidence.

Real isolated Python processes execute a deliberately synthetic historical
verifier. Real phases verifies each fixture's immutable source/data graph. Only
the parent's loaded-module guard is patched because pytest runs outside that
fixture deployment; its rejection boundary has a separate test. The old verifier
stub is not a substitute for the actual evidence replay covered by pilot_audit.
"""

from __future__ import annotations

import copy
import inspect
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import historical_pilot as bridge
from sr_f6 import phases
from sr_f6.common import digest, file_hash
from sr_f6.runtime import _verify_module_origins

ORIGINAL_LOADED = bridge._loaded

HISTORICAL_STUB = """
import json, os, pathlib, sys
def verify_runtime_pilot(root, **arguments):
    root = pathlib.Path(root)
    mode = os.environ.get("BRIDGE_CONTROL_MODE", "normal")
    with (root / "CONTROL_CALLS.jsonl").open("a") as out:
        out.write(json.dumps(dict(file=__file__, cwd=os.getcwd(), argv=sys.argv,
            isolated=sys.flags.isolated, no_bytecode=sys.dont_write_bytecode,
            visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
            pythonpath=os.environ.get("PYTHONPATH"), args=arguments,
            path=sys.path)) + "\\n")
    if mode == "exit":
        print("controlled verifier failure", file=sys.stderr)
        raise SystemExit(7)
    if mode == "bad_json":
        print("not a JSON response")
        raise SystemExit(0)
    if mode == "source_drift":
        with pathlib.Path(__file__).open("a") as out:
            out.write("\\n# changed during verification\\n")
    if mode == "evidence_drift":
        (root / "CENSUS/new-unresolved-attempt.json").write_text("{}")
    value = json.loads((root / "CONTROL_FACTS.json").read_text())
    value["phase_identity_sha256"] = arguments["phase_identity_sha256"]
    value["pilot_spec_sha256"] = arguments["pilot_spec_sha256"]
    return value
"""


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bridge._payload(value))
    return dict(path=relative, sha256=file_hash(path))


def _manifest(directory):
    return dict(
        files=[
            dict(
                path=p.relative_to(directory).as_posix(),
                bytes=p.stat().st_size,
                sha256=file_hash(p),
            )
            for p in sorted(directory.rglob("*.py"))
        ]
    )


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = tmp_path.resolve() / "project"
    root = project / phases.ROOT.name
    root.mkdir(parents=True)
    monkeypatch.setattr(phases, "PROJECT", project)
    old = project / "sr_f6_code_aaaaaaa"
    pilot = project / "sr_f6_code_bbbbbbb"
    current = project / "sr_f6_code_ccccccc"
    (old / "src/sr_f6").mkdir(parents=True)
    (old / "src/sr_f6/runtime.py").write_text("# synthetic S0 source\n")
    (pilot / "src/sr_f6").mkdir(parents=True)
    local = Path(bridge.__file__).parent
    for name in ("phases.py", "common.py", "contract.py"):
        shutil.copyfile(local / name, pilot / "src/sr_f6" / name)
    with (pilot / "src/sr_f6/contract.py").open("a") as file:
        file.write(f"\nPROJECT = Path({str(project)!r})\n")
    (pilot / "src/sr_f6/__init__.py").write_text("")
    (pilot / "src/sr_f6/runtime.py").write_text(
        "from pathlib import Path\n" + inspect.getsource(_verify_module_origins)
    )
    (pilot / "src/sr_f6/admission.py").write_text("# synthetic historical admission module\n")
    (pilot / "src/sr_f6/pilot_audit.py").write_text(HISTORICAL_STUB)
    shutil.copytree(pilot, current)
    (current / "src/sr_f6/pilot_audit.py").write_text(
        "raise RuntimeError('S2 verifier must never be imported by historical child')\n"
    )
    shutil.copyfile(local / "historical_pilot.py", current / "src/sr_f6/historical_pilot.py")
    monkeypatch.setattr(bridge, "_actual_deployment", lambda: current)
    monkeypatch.setattr(bridge, "_loaded", lambda *args: None)
    config = Path(__file__).parents[2] / "configs/sr_f6.json"
    (root / "CONFIG.json").write_bytes(config.read_bytes())
    write(root, "IMPLEMENTATION_REGISTRATION.json", phases.AMENDMENTS)
    write(root, "SOURCE_BINDING.json", {"fixture": "SYNTHETIC_CONTROL"})
    image = root / "DATA/images/control.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"SYNTHETIC_IMAGE_CONTROL")
    data = write(
        root,
        "DATA/MANIFEST.json",
        dict(
            config_sha256=digest(json.loads(config.read_bytes())),
            files=[
                dict(
                    path="DATA/images/control.png",
                    sha256=file_hash(image),
                    bytes=image.stat().st_size,
                )
            ],
        ),
    )
    source = _manifest(old)
    source_ref = write(root, "SOURCE_MANIFEST.json", source)
    origin = dict(
        config_sha256=file_hash(root / "CONFIG.json"),
        implementation_digest=digest(phases.AMENDMENTS),
        revision="a" * 40,
        source_manifest=source_ref,
        data_manifest=data,
    )
    origin_ref = write(root, "IDENTITY.json", origin)
    first = write(root, "CENSUS/CONTROL.json", dict(context=origin, status="NOT_RUN"))
    upstream = dict(
        producer_phase=phases.ORIGIN, identity_sha256=origin_ref["sha256"], evidence=[first]
    )

    def bind(phase, deployment, revision, evidence):
        manifest = phases.build_manifest(deployment, revision=revision)
        before = {i["path"]: i for i in source["files"]}
        after = {i["path"]: i for i in manifest["files"]}
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        delta = phases.source_delta(
            source, manifest, reasons={p: "Synthetic transport fixture only" for p in changed}
        )
        return phases.register_phase(
            root,
            phase=phase,
            deployment=deployment,
            revision=revision,
            origin_deployment=old,
            upstream=evidence,
            expected_delta=delta,
        )

    producer = bind("PILOT_V1", pilot, "b" * 40, [upstream])
    spec = write(root, bridge.SPEC_PATH, dict(phase="PILOT_V1", context=producer["identity"]))
    admission = write(
        root,
        "PHASES/PILOT_V1/ADMISSION.json",
        dict(context=producer["identity"], inputs=[spec], external_inputs=[], status="PASS"),
    )
    second = dict(
        producer_phase="PILOT_V1", identity_sha256=producer["identity_sha256"], evidence=[admission]
    )
    consumer = bind("FORMAL_V1", current, "c" * 40, [upstream, second])
    binding = write(
        root, bridge.RECOVERY + "/RECOVERY_BINDING.json", {"fixture": "SYNTHETIC_CONTROL"}
    )
    compared = write(
        root,
        bridge.RECOVERY + "/COMPARISONS/" + "d" * 32 + ".json",
        {"fixture": "SYNTHETIC_CONTROL"},
    )
    evaluation = write(
        root, "PHASE_RUNS/PILOT_V1/EVALUATION/SUMMARY.json", {"fixture": "SYNTHETIC_CONTROL"}
    )
    request = dict(
        formal_phase_identity_sha256=consumer["identity_sha256"],
        pilot_phase_identity_sha256=producer["identity_sha256"],
        pilot_spec_sha256=spec["sha256"],
        admission_ref=admission,
        recovery_binding_ref=binding,
        recovery_comparison_ref=compared,
        evaluation_summary_ref=evaluation,
    )
    # Controlled transport payload only. It does not represent real CUDA facts;
    # the deployed production function must independently reconstruct those.
    facts = dict(
        format="srf6-runtime-pilot-audit-v1",
        scope="SERVER_ACTUAL",
        status="COMPLETE_FACTS_NOT_GATE_SIGNED",
        gate_status="NOT_SIGNED",
        training_authorized=False,
        formal_authorized=False,
        paths=9,
        logical_updates=108,
        primary_responses=13824,
        primary_actual_generated_responses=13824,
        recovery_responses=1024,
        evaluation_responses=2304,
        runs=[dict(parent=p, arm=a) for p in ("b0", "b1", "b2") for a in ("U10", "U11", "C_SHUF")],
        exposure={"registered_exposure_met": True},
        recovery={},
        evaluation={},
        inputs=[write(root, "PHASE_RUNS/PILOT_V1/CONTROL_VERIFIED_LEAF.json", {"original": True})],
    )
    write(root, "CONTROL_FACTS.json", facts)
    return SimpleNamespace(
        root=root,
        project=project,
        old=old,
        pilot=pilot,
        current=current,
        producer=producer,
        consumer=consumer,
        request=request,
        facts=facts,
    )


def run(case):
    return bridge.verify_historical_pilot(case.root, **case.request)


def attempts(case):
    folder = case.root / bridge.DIRECTORY
    return sorted(p for p in folder.iterdir() if p.is_dir()) if folder.exists() else []


def test_actual_child_isolates_s2_and_pythonpath_and_keeps_producers_distinct(case, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(case.current / "src"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    sys.modules["sr_f6.transport_parent_poison"] = SimpleNamespace(
        __file__=str(case.current / "src/sr_f6/pilot_audit.py")
    )
    try:
        report = run(case)
    finally:
        del sys.modules["sr_f6.transport_parent_poison"]
    calls = [
        json.loads(line) for line in (case.root / "CONTROL_CALLS.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 1
    call = calls[0]
    assert call["file"] == str(case.pilot / "src/sr_f6/pilot_audit.py")
    assert call["isolated"] == 1 and call["no_bytecode"] is True
    assert call["visible"] == "" and call["pythonpath"] is None
    assert (
        call["path"][0] == str(case.pilot / "src") and str(case.current / "src") not in call["path"]
    )
    assert report["context"] == case.consumer["identity"]
    assert report["producer_context"] == case.producer["identity"]
    assert report["context"] != report["producer_context"]
    assert report["execution"]["command"] == [sys.executable, "-I", "-B", "-c", bridge.BOOTSTRAP]
    assert datetime.fromisoformat(report["execution"]["started_at"]) <= datetime.fromisoformat(
        report["execution"]["completed_at"]
    )
    assert report["execution"]["child_pid"] > 0
    assert report["status"] == "FACTS_RECOMPUTED_NOT_AUTHORIZED"
    assert not any(
        report[k]
        for k in (
            "training_authorized",
            "generation_authorized",
            "evaluation_authorized",
            "formal_authorized",
        )
    )
    assert "scope" not in report  # The CPU bridge is not itself a CUDA producer.
    assert not list(case.pilot.rglob("__pycache__"))
    assert not (case.root / "PHASE_RUNS/PILOT_V1/RUNTIME_PILOT_AUDIT/FACTS.json").exists()
    assert (case.root / report["report_ref"]["path"]).is_file()


@pytest.mark.parametrize(
    "status",
    [
        "INSUFFICIENT_REGISTERED_EXPOSURE",
        "RECOVERY_MISMATCH",
        "RECOVERY_COMPARISON_HISTORY_UNKNOWN",
    ],
)
def test_historical_negative_or_unknown_is_preserved_without_permission(case, status):
    facts = {**case.facts, "status": status, "original_details": {"unfavorable": [0, -1, None]}}
    write(case.root, "CONTROL_FACTS.json", facts)
    result = run(case)
    expected = {
        **facts,
        "phase_identity_sha256": case.request["pilot_phase_identity_sha256"],
        "pilot_spec_sha256": case.request["pilot_spec_sha256"],
    }
    assert result["historical_result"] == expected
    assert result["historical_result_sha256"] == digest(expected)
    assert result["formal_authorized"] is False and result["gate_status"] == "NOT_SIGNED"


@pytest.mark.parametrize(
    "mutation", ["synthetic", "nested_synthetic", "fake_pass", "permission", "wrong_budget"]
)
def test_synthetic_fake_pass_or_unauthorized_child_payload_cannot_pass(case, mutation):
    value = copy.deepcopy(case.facts)
    if mutation == "synthetic":
        value["scope"] = "SYNTHETIC_ONLY"
    elif mutation == "nested_synthetic":
        value["runs"][0]["scope"] = "SYNTHETIC_ONLY"
    elif mutation == "fake_pass":
        value["status"] = "PASS"
    elif mutation == "permission":
        value["formal_authorized"] = True
    else:
        value["primary_actual_generated_responses"] += 1
    write(case.root, "CONTROL_FACTS.json", value)
    with pytest.raises(bridge.HistoricalPilotVerificationError):
        run(case)
    attempt = attempts(case)[0]
    assert (attempt / "FAILURE.json").is_file() and not (attempt / "RESULT.json").exists()


@pytest.mark.parametrize("mutation", ["path", "ref_schema", "sha", "phase", "comparison"])
def test_wrong_request_rejected_before_launch(case, mutation):
    if mutation == "path":
        case.request["admission_ref"]["path"] = "../ADMISSION.json"
    elif mutation == "ref_schema":
        case.request["admission_ref"]["allow"] = True
    elif mutation == "sha":
        case.request["pilot_spec_sha256"] = "f" * 64
    elif mutation == "phase":
        case.request["pilot_phase_identity_sha256"] = "f" * 64
    else:
        case.request["recovery_comparison_ref"]["path"] = (
            bridge.RECOVERY + "/COMPARISONS/different.json"
        )
    with pytest.raises(PermissionError):
        run(case)
    assert attempts(case) == [] and not (case.root / "CONTROL_CALLS.jsonl").exists()


@pytest.mark.parametrize("mode", ["exit", "bad_json", "source_drift", "evidence_drift"])
def test_actual_process_failure_bad_output_or_drift_is_preserved_without_retry(
    case, monkeypatch, mode
):
    monkeypatch.setenv("BRIDGE_CONTROL_MODE", mode)
    with pytest.raises(bridge.HistoricalPilotVerificationError):
        run(case)
    attempt = attempts(case)[0]
    assert (
        (attempt / "REQUEST.json").is_file()
        and (attempt / "stdout.json").is_file()
        and (attempt / "stderr.log").is_file()
    )
    failure = json.loads((attempt / "FAILURE.json").read_bytes())
    assert not failure["formal_authorized"] and not (attempt / "RESULT.json").exists()
    assert len((case.root / "CONTROL_CALLS.jsonl").read_text().splitlines()) == 1
    if mode == "exit":
        assert failure["returncode"] == 7
        assert "controlled verifier failure" in (attempt / "stderr.log").read_text()


def test_saved_fake_facts_are_never_an_input_and_each_call_recomputes(case):
    write(
        case.root,
        "PHASE_RUNS/PILOT_V1/RUNTIME_PILOT_AUDIT/FACTS.json",
        dict(status="PASS", formal_authorized=True),
    )
    first, second = run(case), run(case)
    assert len((case.root / "CONTROL_CALLS.jsonl").read_text().splitlines()) == 2
    assert first["historical_result"] == second["historical_result"]
    assert first["report_ref"] != second["report_ref"] and len(attempts(case)) == 2


def test_original_input_closure_is_reachable_by_actual_phase_dependency_checker(case):
    report = run(case)
    leaf = case.facts["inputs"][0]
    assert leaf in report["inputs"]
    assert len({item["path"] for item in report["inputs"]}) == len(report["inputs"])
    with phases._ContentCache(case.root) as cache:
        phases._dependencies(case.root, report, report["report_ref"]["path"], cache, set(), set())
    write(case.root, leaf["path"], {"original": False})
    with (
        phases._ContentCache(case.root) as cache,
        pytest.raises(PermissionError, match="dependency changed"),
    ):
        phases._dependencies(case.root, report, report["report_ref"]["path"], cache, set(), set())


def test_consumer_loaded_source_guard_refuses_pytest_outside_fixture_deployment(case):
    # The fixture patches this guard only for transport tests; real entry rejects
    # the current pytest module origin instead of relabeling it as synthetic S2.
    manifest = json.loads(
        (case.root / case.consumer["identity"]["source_manifest"]["path"]).read_bytes()
    )
    with pytest.raises(PermissionError, match="outside its FORMAL"):
        ORIGINAL_LOADED(case.current, manifest)
